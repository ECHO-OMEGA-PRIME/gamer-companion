"""Aggression Controller — Dynamic aggression scaling based on game context.

Instead of static aggression levels, this adapts in real-time based on:
- Score (winning → can afford aggro, losing → play safe)
- Economy (rich → push, poor → save)
- Round context (early → info gather, late → commit)
- Tilt level (tilted → simplify, calm → full repertoire)
- Team composition (more alive → push, fewer → careful)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class AggressionState:
    """Current dynamic aggression state."""
    base_aggression: float = 0.5
    dynamic_modifier: float = 0.0
    effective_aggression: float = 0.5
    reason: str = "default"


class AggressionController:
    """Dynamically adjust aggression based on game context.

    Output: effective_aggression (0.0 to 1.0) that feeds into
    CombatEngine, AimEngine, and MovementEngine.
    """

    def __init__(self, base_aggression: float = 0.5):
        self._base = base_aggression
        self._state = AggressionState(base_aggression=base_aggression)
        self._history: list = []

    def update(
        self,
        score_diff: int = 0,        # +3 = winning by 3, -3 = losing by 3
        economy: int = 4000,         # Current money
        round_time_left: float = 115,  # Seconds
        tilt_level: float = 0.0,     # 0-1
        teammates_alive: int = 5,
        enemies_alive: int = 5,
        bomb_planted: bool = False,
    ) -> AggressionState:
        """Compute dynamic aggression based on game state."""
        modifier = 0.0
        reasons = []

        # Score context
        if score_diff >= 5:
            modifier += 0.15
            reasons.append("big_lead")
        elif score_diff >= 2:
            modifier += 0.08
            reasons.append("leading")
        elif score_diff <= -5:
            modifier -= 0.1
            reasons.append("big_deficit")
        elif score_diff <= -2:
            modifier -= 0.05
            reasons.append("trailing")

        # Economy context
        if economy >= 6000:
            modifier += 0.1
            reasons.append("rich")
        elif economy <= 1500:
            modifier -= 0.15
            reasons.append("eco")

        # Round time
        if round_time_left < 20:
            modifier += 0.2
            reasons.append("time_pressure")
        elif round_time_left < 40:
            modifier += 0.1
            reasons.append("mid_round")

        # Tilt adjustment (high tilt → simplify = less aggro)
        if tilt_level > 0.6:
            modifier -= 0.15
            reasons.append("tilted")
        elif tilt_level > 0.4:
            modifier -= 0.05
            reasons.append("frustrated")

        # Numbers
        if teammates_alive > enemies_alive:
            advantage = (teammates_alive - enemies_alive) * 0.08
            modifier += advantage
            reasons.append(f"numbers_{teammates_alive}v{enemies_alive}")
        elif enemies_alive > teammates_alive:
            modifier -= (enemies_alive - teammates_alive) * 0.1
            reasons.append(f"outnumbered_{teammates_alive}v{enemies_alive}")

        # Bomb planted
        if bomb_planted:
            modifier += 0.15
            reasons.append("bomb_down")

        # Compute effective
        effective = max(0.0, min(1.0, self._base + modifier))

        self._state = AggressionState(
            base_aggression=self._base,
            dynamic_modifier=round(modifier, 3),
            effective_aggression=round(effective, 3),
            reason="+".join(reasons) if reasons else "neutral",
        )

        self._history.append({
            "aggression": self._state.effective_aggression,
            "reason": self._state.reason,
            "timestamp": time.time(),
        })

        return self._state

    @property
    def aggression(self) -> float:
        return self._state.effective_aggression

    @property
    def state(self) -> AggressionState:
        return self._state

    def set_base(self, aggression: float):
        self._base = max(0.0, min(1.0, aggression))

    def get_stats(self) -> dict:
        return {
            "base": self._base,
            "effective": self._state.effective_aggression,
            "modifier": self._state.dynamic_modifier,
            "reason": self._state.reason,
            "history_length": len(self._history),
        }
