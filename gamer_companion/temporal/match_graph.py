"""Match Graph — Live knowledge graph of the current match state.

Tracks players (allies/enemies), map zones, economy, utility usage,
and round history. Enables queries like "who controls mid?",
"what's the enemy's likely buy?", "which player tends to flank?"
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger


@dataclass
class PlayerNode:
    """A tracked player in the match."""
    player_id: str
    team: str  # "ally" | "enemy"
    role: Optional[str] = None  # "entry", "support", "anchor", "lurk", "igl"
    agent_character: Optional[str] = None  # "Jett", "Sage", etc.
    weapon: Optional[str] = None
    hp_estimate: Optional[int] = None
    last_position: Optional[str] = None
    last_seen_ts: float = 0
    kills: int = 0
    deaths: int = 0
    tendencies: List[str] = field(default_factory=list)
    economy_estimate: Optional[int] = None
    threat_score: float = 0.5
    consistency_score: float = 0.5
    detected_via: str = "vision"  # "vision", "audio", "minimap", "inference"


@dataclass
class MapZone:
    """A zone/area on the game map."""
    name: str  # "A_site", "B_main", "mid", "spawn"
    control: str = "neutral"  # "ally" | "enemy" | "neutral" | "contested"
    last_activity_ts: float = 0
    enemy_sightings: int = 0
    ally_present: bool = False
    danger_score: float = 0.0
    utility_used: List[str] = field(default_factory=list)


class MatchGraph:
    """Live knowledge graph representing current match state.

    Enables queries like:
    - "Who controls mid?" -> zone.control
    - "Who is their best player?" -> max(threat_score)
    - "When was the last time an enemy was seen at B?" -> last_activity_ts
    - "What's the enemy's likely buy this round?" -> full economy sim
    - "Which player tends to flank?" -> tendencies filter
    """

    def __init__(self, map_name: str = "unknown", game_profile=None):
        self.map_name = map_name
        self.profile = game_profile
        self.round_number: int = 0
        self.score_ally: int = 0
        self.score_enemy: int = 0
        self.side: str = "unknown"  # "attack" | "defense"
        self.players: Dict[str, PlayerNode] = {}
        self.zones: Dict[str, MapZone] = {}
        self.round_events: List[dict] = []
        self.round_history: List[dict] = []
        self.utility_log: List[dict] = []

    def update_player(self, player_id: str, **kwargs):
        """Update or create a player node."""
        if player_id not in self.players:
            self.players[player_id] = PlayerNode(
                player_id=player_id,
                team=kwargs.get("team", "unknown"),
            )
        p = self.players[player_id]
        for k, v in kwargs.items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, v)
        if "last_position" in kwargs:
            p.last_seen_ts = time.time()
        # Update threat score based on kills
        if p.team == "enemy":
            kd = p.kills / max(p.deaths, 1)
            p.threat_score = min(1.0, kd * 0.3 + 0.2)

    def update_zone(self, zone_name: str, **kwargs):
        """Update or create a map zone."""
        if zone_name not in self.zones:
            self.zones[zone_name] = MapZone(name=zone_name)
        z = self.zones[zone_name]
        for k, v in kwargs.items():
            if hasattr(z, k) and v is not None:
                setattr(z, k, v)
        z.last_activity_ts = time.time()

    def record_event(self, event_type: str, **data):
        """Record a round event (kill, plant, defuse, etc.)."""
        self.round_events.append({
            "type": event_type, "timestamp": time.time(),
            "round": self.round_number, **data,
        })

    def record_utility(self, util_type: str, zone: str, team: str):
        """Record utility usage (smoke, flash, molly)."""
        self.utility_log.append({
            "type": util_type, "zone": zone, "team": team,
            "round": self.round_number, "timestamp": time.time(),
        })
        if zone in self.zones:
            self.zones[zone].utility_used.append(util_type)

    def end_round(self, winner: str):
        """Close out a round and archive its data."""
        summary = {
            "round": self.round_number,
            "winner": winner,
            "events": self.round_events.copy(),
            "enemy_economy_est": self._estimate_enemy_economy(),
            "utility_used": [
                u for u in self.utility_log
                if u["round"] == self.round_number
            ],
        }
        self.round_history.append(summary)
        self.round_events.clear()
        self.round_number += 1
        if winner == "ally":
            self.score_ally += 1
        else:
            self.score_enemy += 1
        # Reset zone utility for new round
        for z in self.zones.values():
            z.utility_used.clear()

    def _estimate_enemy_economy(self) -> int:
        """Full economy simulation based on round outcomes + observed weapons."""
        losses_in_row = 0
        for r in reversed(self.round_history):
            if r["winner"] == "enemy":
                break
            losses_in_row += 1

        loss_bonus = min(1400 + 500 * losses_in_row, 3400)

        # Check if we've observed expensive weapons
        if self.profile and hasattr(self.profile, "weapons") and self.profile.weapons:
            observed_weapons = [
                p.weapon for p in self.players.values()
                if p.team == "enemy" and p.weapon
            ]
            for w in observed_weapons:
                wp = self.profile.weapons.get(w)
                if wp and hasattr(wp, "cost") and wp.cost > 3000:
                    return max(loss_bonus, 4500)

        return loss_bonus

    def get_enemy_tendencies(self) -> dict:
        """Analyze enemy patterns across rounds."""
        if len(self.round_history) < 3:
            return {"insufficient_data": True}

        patterns = {
            "rush_frequency": 0,
            "preferred_site": {},
            "eco_aggression": 0,
            "default_formation": "unknown",
            "utility_usage_rate": 0,
            "most_dangerous_player": None,
        }
        total_rounds = len(self.round_history)
        for r in self.round_history:
            for e in r["events"]:
                if e["type"] == "site_hit" and e.get("site"):
                    site = e["site"]
                    patterns["preferred_site"][site] = (
                        patterns["preferred_site"].get(site, 0) + 1
                    )
                if e["type"] == "rush" and e.get("time_elapsed", 30) < 20:
                    patterns["rush_frequency"] += 1

        patterns["rush_frequency"] = round(
            patterns["rush_frequency"] / max(total_rounds, 1), 2
        )
        patterns["utility_usage_rate"] = round(
            len([u for u in self.utility_log if u["team"] == "enemy"])
            / max(total_rounds, 1), 1
        )

        # Find most dangerous enemy
        enemies = [p for p in self.players.values() if p.team == "enemy"]
        if enemies:
            most_dangerous = max(enemies, key=lambda p: p.threat_score)
            patterns["most_dangerous_player"] = {
                "id": most_dangerous.player_id,
                "threat_score": most_dangerous.threat_score,
                "tendencies": most_dangerous.tendencies,
                "kills": most_dangerous.kills,
            }

        return patterns

    def context_for_llm(self) -> str:
        """Generate compact context string for injection into LLM prompt."""
        alive_allies = sum(
            1 for p in self.players.values()
            if p.team == "ally" and (p.hp_estimate or 0) > 0
        )
        alive_enemies = sum(
            1 for p in self.players.values()
            if p.team == "enemy" and (p.hp_estimate or 0) > 0
        )
        enemy_positions = [
            f"{p.player_id} at {p.last_position} "
            f"({time.time() - p.last_seen_ts:.0f}s ago, threat:{p.threat_score:.1f})"
            for p in self.players.values()
            if p.team == "enemy" and p.last_position
            and time.time() - p.last_seen_ts < 30
        ]
        zone_control = [
            f"{z.name}: {z.control} (danger:{z.danger_score:.1f})"
            for z in self.zones.values()
            if z.control != "neutral"
        ]
        tendencies = self.get_enemy_tendencies()
        pref_site = max(
            tendencies.get("preferred_site", {}).items(),
            key=lambda x: x[1], default=("unknown", 0),
        )
        return (
            f"Map: {self.map_name} | Round {self.round_number} | "
            f"Score: {self.score_ally}-{self.score_enemy} | Side: {self.side}\n"
            f"Alive: {alive_allies}v{alive_enemies}\n"
            f"Enemy positions: {'; '.join(enemy_positions) or 'unknown'}\n"
            f"Zone control: {'; '.join(zone_control) or 'all neutral'}\n"
            f"Enemy economy est: ${self._estimate_enemy_economy()}\n"
            f"Rush frequency: {tendencies.get('rush_frequency', 0):.0%}\n"
            f"Preferred site: {pref_site[0]}\n"
            f"Most dangerous: "
            f"{tendencies.get('most_dangerous_player', {}).get('id', 'unknown')}"
        )
