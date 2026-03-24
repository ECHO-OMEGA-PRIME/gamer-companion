"""Probability Engine — Bayesian prediction + MCTS planning.

Predicts enemy positions, economy, buy decisions, and next plays
using time-decayed sightings, audio cues, historical patterns,
utility usage, and timing analysis. MCTS planner simulates possible
futures to recommend optimal strategies.
"""

from __future__ import annotations
import math
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class ZoneProbability:
    """Probability of enemy presence in a map zone."""
    zone: str
    probability: float  # 0.0 - 1.0
    confidence: float
    reasoning: str
    last_updated: float = 0
    contributing_factors: List[str] = field(default_factory=list)


class ProbabilityEngine:
    """Bayesian prediction engine for enemy positions and actions.

    Uses:
    - Last known positions + time decay
    - Historical round patterns (they rush B 40% of the time)
    - Economy state -> likely buy/strat
    - Sound cues (footsteps, abilities)
    - Timing analysis (round time remaining)
    - Utility usage (smoked A -> probably going B)
    - Kill feed analysis (entry fragger dead -> less likely to rush)
    """

    def __init__(self):
        self._zone_priors: Dict[str, float] = {}
        self._decay_rate = 0.15
        self._audio_weight = 0.3
        self._pattern_weight = 0.25
        self._utility_weight = 0.2

    def predict_enemy_positions(
        self,
        match_graph,
        frame_history=None,
        audio_events: List[dict] = None,
        round_time_remaining: float = None,
    ) -> List[ZoneProbability]:
        """Generate probability distribution of enemy locations."""
        predictions = []

        for zone_name, zone in match_graph.zones.items():
            base_prob = self._zone_priors.get(zone_name, 0.1)
            factors = []

            # Factor 1: Recent sightings (time-decayed)
            sighting_boost = 0
            for player in match_graph.players.values():
                if player.team == "enemy" and player.last_position == zone_name:
                    age = time.time() - player.last_seen_ts
                    boost = math.exp(-self._decay_rate * age)
                    sighting_boost += boost
                    if boost > 0.1:
                        factors.append(f"spotted {age:.0f}s ago")

            # Factor 2: Audio cues
            audio_boost = 0
            if audio_events:
                for event in audio_events:
                    if event.get("zone") == zone_name:
                        age = time.time() - event["timestamp"]
                        ab = self._audio_weight * math.exp(-0.2 * age)
                        audio_boost += ab
                        if ab > 0.05:
                            factors.append(f"audio: {event.get('type', 'sound')}")

            # Factor 3: Time pressure
            time_factor = 1.0
            if round_time_remaining is not None:
                if round_time_remaining < 30:
                    if "site" in zone_name.lower():
                        time_factor = 1.5
                        factors.append("late round -> site rush likely")
                elif round_time_remaining > 90:
                    if "spawn" in zone_name.lower() or "main" in zone_name.lower():
                        time_factor = 1.3
                        factors.append("early round -> default positions")

            # Factor 4: Historical patterns
            tendencies = match_graph.get_enemy_tendencies()
            pattern_boost = 0
            pref_sites = tendencies.get("preferred_site", {})
            total_hits = sum(pref_sites.values()) or 1
            if zone_name in pref_sites:
                pattern_boost = self._pattern_weight * (
                    pref_sites[zone_name] / total_hits
                )
                if pattern_boost > 0.05:
                    factors.append(
                        f"historical: {pref_sites[zone_name]}/{total_hits} rounds"
                    )

            # Factor 5: Utility usage
            utility_boost = 0
            recent_utility = [
                u for u in match_graph.utility_log
                if u["team"] == "enemy" and time.time() - u["timestamp"] < 30
            ]
            for util in recent_utility:
                if util["zone"] != zone_name:
                    utility_boost += 0.05
                    factors.append(f"utility at {util['zone']} -> possible fake")
                else:
                    utility_boost += 0.15
                    factors.append("utility used here -> preparing entry")

            # Combine
            combined = (
                (base_prob + sighting_boost + audio_boost
                 + pattern_boost + utility_boost) * time_factor
            )
            combined = min(combined, 0.95)

            confidence = min(
                0.9,
                0.3 + sighting_boost * 0.5
                + audio_boost * 0.3 + pattern_boost * 0.2,
            )

            predictions.append(ZoneProbability(
                zone=zone_name,
                probability=round(combined, 3),
                confidence=round(confidence, 3),
                reasoning="; ".join(factors[:4]) or "base prior",
                last_updated=time.time(),
                contributing_factors=factors,
            ))

        # Normalize
        total = sum(p.probability for p in predictions) or 1
        for p in predictions:
            p.probability = round(p.probability / total, 3)

        return sorted(predictions, key=lambda x: x.probability, reverse=True)

    def predict_enemy_buy(self, match_graph) -> dict:
        """Predict enemy team's buy based on economy estimation."""
        est_money = match_graph._estimate_enemy_economy()
        round_num = match_graph.round_number
        score_diff = match_graph.score_enemy - match_graph.score_ally

        if est_money >= 4500:
            prediction = "full_buy"
            confidence = 0.85
            expected = "rifles + full utility"
            advice = "Expect full buy. Play standard, use utility."
        elif est_money >= 3500:
            if score_diff < -3:
                prediction = "force_buy"
                confidence = 0.65
                expected = "desperate force — SMGs or galil/famas + some utility"
                advice = "They're desperate. Expect aggression. Hold angles."
            else:
                prediction = "full_buy"
                confidence = 0.6
                expected = "budget rifles, limited utility"
                advice = "Borderline buy. They might have gaps in utility."
        elif est_money >= 2000:
            prediction = "force_buy"
            confidence = 0.6
            expected = "SMGs or pistol armor, limited utility"
            advice = "Force buy likely. Play mid-range, punish poor utility."
        elif est_money >= 1000:
            prediction = "half_buy"
            confidence = 0.55
            expected = "upgraded pistols, maybe one SMG"
            advice = "Half buy. Play close angles, don't give away weapons."
        else:
            prediction = "eco"
            confidence = 0.75
            expected = "default pistols only"
            advice = "Eco round. Anti-eco positions. Don't give away weapons."

        return {
            "prediction": prediction, "confidence": confidence,
            "expected": expected, "advice": advice,
            "estimated_money": est_money, "round": round_num,
        }

    def predict_next_play(self, match_graph, frame_history=None) -> dict:
        """Predict what the enemy team will do this round."""
        tendencies = match_graph.get_enemy_tendencies()
        buy = self.predict_enemy_buy(match_graph)
        positions = self.predict_enemy_positions(match_graph, frame_history)

        top_zone = positions[0] if positions else None
        pref_site = max(
            tendencies.get("preferred_site", {}).items(),
            key=lambda x: x[1], default=("unknown", 0),
        )

        rush_likely = tendencies.get("rush_frequency", 0) > 0.3
        site_total = sum(tendencies.get("preferred_site", {}).values()) or 1

        return {
            "predicted_buy": buy["prediction"],
            "predicted_site": pref_site[0],
            "site_confidence": round(pref_site[1] / site_total, 2),
            "hottest_zone": top_zone.zone if top_zone else "unknown",
            "zone_probability": top_zone.probability if top_zone else 0,
            "zone_reasoning": top_zone.reasoning if top_zone else "",
            "rush_likely": rush_likely,
            "most_dangerous_player": tendencies.get("most_dangerous_player", {}),
            "recommended_setup": self._recommend_setup(
                buy["prediction"], pref_site[0],
                match_graph.side, rush_likely,
            ),
        }

    def _recommend_setup(
        self, enemy_buy: str, likely_site: str,
        our_side: str, rush_likely: bool,
    ) -> str:
        """Generate tactical recommendation based on predictions."""
        if our_side == "defense":
            if enemy_buy == "eco":
                return (
                    f"Anti-eco setup. Stack {likely_site} with close angles. "
                    f"Don't overextend."
                )
            elif enemy_buy == "full_buy" and rush_likely:
                return (
                    f"Rush expected at {likely_site}. Stack 3 players. "
                    f"Molly + flash ready."
                )
            elif enemy_buy == "full_buy":
                return (
                    f"Standard setup. {likely_site} is their preference — "
                    f"keep retake positions ready."
                )
            else:
                return "Force buy expected. Play aggressive for early picks."
        else:
            if enemy_buy == "eco":
                return (
                    "Enemy on eco. Default execute, save utility for "
                    "retake defense."
                )
            elif enemy_buy == "full_buy":
                return (
                    "Full buy defense. Use all utility. Consider fakes "
                    "to split their setup."
                )
            else:
                return (
                    "Weak buy defense. Fast execute should overwhelm. "
                    "Save utility."
                )


class MCTSPlanner:
    """Monte Carlo Tree Search for multi-step action planning.

    Simulates possible futures to find optimal strategy:
    1. What if we push A? (simulate)
    2. What if we fake B then go A? (simulate)
    3. What if we play default? (simulate)

    Each simulation uses the probability engine to model enemy responses,
    then evaluates the outcome. After N simulations, pick the plan with
    the highest win rate.
    """

    @dataclass
    class Node:
        action: str
        visits: int = 0
        wins: float = 0
        children: List = field(default_factory=list)
        parent: Optional[object] = None

        @property
        def ucb1(self) -> float:
            if self.visits == 0:
                return float("inf")
            parent_visits = self.parent.visits if self.parent else self.visits
            exploitation = self.wins / self.visits
            exploration = math.sqrt(
                2 * math.log(parent_visits) / self.visits
            )
            return exploitation + exploration

    def __init__(self, simulation_budget: int = 200, max_depth: int = 5):
        self.budget = simulation_budget
        self.max_depth = max_depth

    def plan(
        self,
        current_state: dict,
        available_actions: List[str],
        probability_engine: ProbabilityEngine = None,
    ) -> dict:
        """Run MCTS to find the best action plan."""
        root = self.Node(action="root")

        for action in available_actions:
            root.children.append(self.Node(action=action, parent=root))

        for _ in range(self.budget):
            # SELECT
            node = self._select(root)
            # EXPAND
            if node.visits > 0 and not node.children:
                self._expand(node, available_actions)
                if node.children:
                    node = random.choice(node.children)
            # SIMULATE
            reward = self._simulate(node, current_state, probability_engine)
            # BACKPROPAGATE
            self._backprop(node, reward)

        if not root.children:
            return {
                "action": "wait", "confidence": 0.0,
                "reasoning": "no actions available",
            }

        best = max(root.children, key=lambda n: n.visits)
        return {
            "action": best.action,
            "confidence": round(best.wins / max(best.visits, 1), 2),
            "reasoning": (
                f"MCTS: {best.wins:.0f}/{best.visits} wins "
                f"({best.wins / max(best.visits, 1) * 100:.0f}%)"
            ),
            "simulations": self.budget,
            "all_actions": [
                {
                    "action": c.action,
                    "win_rate": round(c.wins / max(c.visits, 1), 2),
                    "visits": c.visits,
                }
                for c in sorted(
                    root.children, key=lambda n: n.visits, reverse=True
                )
            ],
        }

    def _select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1)
        return node

    def _expand(self, node, actions):
        for action in actions:
            node.children.append(self.Node(action=action, parent=node))

    def _simulate(self, node, state, prob_engine) -> float:
        """Simulate a game outcome from this state."""
        score = 0.5
        if "rush" in node.action:
            score += 0.1 if state.get("enemy_buy") in ("eco", "half_buy") else -0.1
        if "default" in node.action:
            score += 0.05
        if "fake" in node.action:
            score += 0.15 if state.get("enemy_stack_detected") else -0.05
        score += random.gauss(0, 0.1)
        return max(0, min(1, score))

    def _backprop(self, node, reward):
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent
