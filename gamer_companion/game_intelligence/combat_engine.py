"""Combat Engine — Combat decision making based on game state.

Evaluates threats, decides fight/flight/reposition, manages engagement
timing, and coordinates with the aim system for target selection.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
from loguru import logger


class CombatAction(Enum):
    ENGAGE = "engage"           # Push and fight
    HOLD = "hold"               # Hold position, wait for enemy
    PEEK = "peek"               # Quick peek for info
    RETREAT = "retreat"         # Fall back to safety
    REPOSITION = "reposition"   # Move to better angle
    FLASH_PEEK = "flash_peek"   # Utility + peek
    TRADE = "trade"             # Move to trade a teammate
    BAIT = "bait"               # Draw enemy fire for team
    ROTATE = "rotate"           # Change site/position entirely
    SAVE = "save"               # Save weapon, don't engage


@dataclass
class Threat:
    """A detected enemy threat."""
    entity_id: str
    position: Optional[Tuple[float, float]] = None
    distance: float = 0
    health_estimate: int = 100
    weapon: str = "unknown"
    is_visible: bool = False
    last_seen: float = 0
    confidence: float = 0.5
    threat_score: float = 0.5


@dataclass
class EngagementContext:
    """Context for a combat engagement decision."""
    my_health: int = 100
    my_armor: int = 0
    my_weapon: str = "rifle"
    my_ammo: int = 30
    my_position: Tuple[float, float] = (0, 0)
    threats: List[Threat] = field(default_factory=list)
    teammates_alive: int = 4
    enemies_alive: int = 5
    round_time_left: float = 115.0
    has_bomb: bool = False
    bomb_planted: bool = False
    economy: int = 4000
    utility_available: List[str] = field(default_factory=list)


class CombatEngine:
    """Make combat decisions based on game state.

    Decision hierarchy:
    1. Survival check (am I about to die?)
    2. Numbers advantage (1v3 = retreat, 3v1 = push)
    3. Positional advantage (high ground, cover, angles)
    4. Utility available (flash = peek, smoke = reposition)
    5. Economy (save if expensive weapon + low round chance)
    6. Objective (bomb timer, site control)
    """

    def __init__(
        self,
        aggression: float = 0.5,
        risk_tolerance: float = 0.5,
    ):
        self.aggression = aggression  # 0 = passive, 1 = hyper-aggro
        self.risk_tolerance = risk_tolerance
        self._engagement_history: List[dict] = []

    def decide(self, ctx: EngagementContext) -> Tuple[CombatAction, float, str]:
        """Decide combat action given context.

        Returns: (action, confidence, reasoning)
        """
        scores: Dict[CombatAction, float] = {}
        reasons: Dict[CombatAction, str] = {}

        # Factor 1: Health check
        health_ratio = ctx.my_health / 100
        if health_ratio < 0.2 and ctx.my_armor == 0:
            scores[CombatAction.RETREAT] = 0.9
            reasons[CombatAction.RETREAT] = "critical_health"
            scores[CombatAction.SAVE] = 0.7
            reasons[CombatAction.SAVE] = "low_hp_save"

        # Factor 2: Numbers advantage
        if ctx.enemies_alive > 0:
            num_advantage = (ctx.teammates_alive + 1) / ctx.enemies_alive
        else:
            num_advantage = 5.0

        if num_advantage >= 2.0:
            scores[CombatAction.ENGAGE] = 0.7 + self.aggression * 0.2
            reasons[CombatAction.ENGAGE] = f"numbers_{num_advantage:.1f}v1"
        elif num_advantage <= 0.5:
            scores[CombatAction.RETREAT] = 0.6
            reasons[CombatAction.RETREAT] = f"outnumbered_{num_advantage:.1f}"
            scores[CombatAction.SAVE] = 0.5
            reasons[CombatAction.SAVE] = "outnumbered_save"

        # Factor 3: Visible threats
        visible = [t for t in ctx.threats if t.is_visible]
        if visible:
            closest = min(visible, key=lambda t: t.distance)
            if closest.distance < 300:
                scores[CombatAction.ENGAGE] = max(
                    scores.get(CombatAction.ENGAGE, 0),
                    0.6 + self.aggression * 0.3,
                )
                reasons[CombatAction.ENGAGE] = "close_enemy"
            elif closest.health_estimate < 50:
                scores[CombatAction.ENGAGE] = max(
                    scores.get(CombatAction.ENGAGE, 0), 0.8,
                )
                reasons[CombatAction.ENGAGE] = "enemy_low_hp"

        # Factor 4: Utility available
        if "flashbang" in ctx.utility_available:
            scores[CombatAction.FLASH_PEEK] = 0.6
            reasons[CombatAction.FLASH_PEEK] = "has_flash"
        if "smoke" in ctx.utility_available:
            scores[CombatAction.REPOSITION] = max(
                scores.get(CombatAction.REPOSITION, 0), 0.5,
            )
            reasons[CombatAction.REPOSITION] = "has_smoke"

        # Factor 5: Bomb context
        if ctx.bomb_planted:
            if ctx.round_time_left < 10:
                scores[CombatAction.ENGAGE] = max(
                    scores.get(CombatAction.ENGAGE, 0), 0.9,
                )
                reasons[CombatAction.ENGAGE] = "bomb_timer_critical"
        elif ctx.has_bomb and ctx.round_time_left < 30:
            scores[CombatAction.ENGAGE] = max(
                scores.get(CombatAction.ENGAGE, 0), 0.7,
            )
            reasons[CombatAction.ENGAGE] = "must_plant"

        # Factor 6: No threats + no info → peek or hold
        if not visible and not scores:
            if self.aggression > 0.6:
                scores[CombatAction.PEEK] = 0.5
                reasons[CombatAction.PEEK] = "aggro_peek"
            else:
                scores[CombatAction.HOLD] = 0.5
                reasons[CombatAction.HOLD] = "default_hold"

        # Default: hold if nothing else
        if not scores:
            scores[CombatAction.HOLD] = 0.4
            reasons[CombatAction.HOLD] = "default"

        # Apply aggression bias
        for action in [CombatAction.ENGAGE, CombatAction.PEEK, CombatAction.FLASH_PEEK]:
            if action in scores:
                scores[action] *= (0.7 + self.aggression * 0.6)
        for action in [CombatAction.RETREAT, CombatAction.SAVE]:
            if action in scores:
                scores[action] *= (1.3 - self.aggression * 0.6)

        # Select best action
        best = max(scores, key=scores.get)
        confidence = min(1.0, scores[best])
        reason = reasons.get(best, "calculated")

        self._engagement_history.append({
            "action": best.value,
            "confidence": round(confidence, 3),
            "reason": reason,
            "timestamp": time.time(),
        })

        return (best, confidence, reason)

    def get_stats(self) -> dict:
        if not self._engagement_history:
            return {"decisions": 0}
        action_counts = {}
        for e in self._engagement_history:
            a = e["action"]
            action_counts[a] = action_counts.get(a, 0) + 1
        return {
            "decisions": len(self._engagement_history),
            "action_distribution": action_counts,
            "aggression": self.aggression,
            "risk_tolerance": self.risk_tolerance,
        }
