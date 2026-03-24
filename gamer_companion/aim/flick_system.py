"""Flick System — Fast snap aim with overshoot and correction.

Handles the rapid aim movement when a new target appears.
Models the human flick pattern:
1. Fast ballistic move toward target (covers ~70-90%)
2. Small overshoot past target
3. Correction movement back to target
4. Final micro-adjustments
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class FlickPhase:
    """A phase of a flick movement."""
    name: str                 # "ballistic", "overshoot", "correction", "settle"
    dx: float
    dy: float
    duration_ms: float
    completed: bool = False


@dataclass
class FlickResult:
    """Result of computing a flick."""
    phases: List[FlickPhase]
    total_dx: float
    total_dy: float
    total_duration_ms: float
    overshoot_px: float
    target_id: str = ""


class FlickSystem:
    """Compute human-like flick aim movements.

    A flick consists of multiple phases:
    1. BALLISTIC: Fast move covering 70-95% of distance (skill-dependent)
    2. OVERSHOOT: Slight over-correction (2-15% past target)
    3. CORRECTION: Move back toward target
    4. SETTLE: Tiny micro-adjustments

    Higher skill = more accurate ballistic phase, less overshoot.
    """

    def __init__(
        self,
        skill_level: float = 0.85,
        speed_multiplier: float = 1.0,
    ):
        self._skill = max(0.1, min(1.0, skill_level))
        self._speed_mult = speed_multiplier
        self._flick_history: List[FlickResult] = []
        self._max_history = 50

    def compute_flick(
        self,
        dx: float,
        dy: float,
        target_id: str = "",
        target_width: float = 30,
    ) -> FlickResult:
        """Compute a multi-phase flick from current crosshair to target delta.

        Args:
            dx: Horizontal distance to target (pixels)
            dy: Vertical distance to target (pixels)
            target_id: Identifier for the target
            target_width: Target hitbox width (affects overshoot scale)
        """
        distance = math.hypot(dx, dy)
        if distance < 1:
            return FlickResult(
                phases=[], total_dx=0, total_dy=0,
                total_duration_ms=0, overshoot_px=0, target_id=target_id,
            )

        phases: List[FlickPhase] = []
        angle = math.atan2(dy, dx)

        # Phase 1: BALLISTIC — fast move toward target
        coverage = 0.70 + self._skill * 0.25  # 70-95%
        coverage += random.gauss(0, 0.03)
        coverage = max(0.65, min(0.98, coverage))

        ballistic_dist = distance * coverage
        b_dx = math.cos(angle) * ballistic_dist
        b_dy = math.sin(angle) * ballistic_dist

        # Ballistic time: faster for higher skill
        base_time = 80 + (1 - self._skill) * 120  # 80-200ms
        ballistic_time = base_time / self._speed_mult
        ballistic_time *= random.uniform(0.85, 1.15)

        phases.append(FlickPhase(
            name="ballistic",
            dx=round(b_dx, 2),
            dy=round(b_dy, 2),
            duration_ms=round(ballistic_time, 1),
        ))

        # Phase 2: OVERSHOOT — slight over-correction
        overshoot_pct = (1 - self._skill) * 0.15 + 0.02  # 2-17%
        overshoot_pct *= random.uniform(0.5, 1.5)
        overshoot_dist = distance * overshoot_pct

        # Overshoot direction: mostly forward but with lateral scatter
        scatter_angle = angle + random.gauss(0, 0.15)
        o_dx = math.cos(scatter_angle) * overshoot_dist
        o_dy = math.sin(scatter_angle) * overshoot_dist

        overshoot_time = 15 + random.uniform(5, 25)

        phases.append(FlickPhase(
            name="overshoot",
            dx=round(o_dx, 2),
            dy=round(o_dy, 2),
            duration_ms=round(overshoot_time, 1),
        ))

        # Phase 3: CORRECTION — move back toward target
        # Need to correct: we're at ballistic + overshoot, target is at dx, dy
        current_x = b_dx + o_dx
        current_y = b_dy + o_dy
        correction_dx = dx - current_x
        correction_dy = dy - current_y

        correction_time = 30 + random.uniform(10, 40)

        phases.append(FlickPhase(
            name="correction",
            dx=round(correction_dx, 2),
            dy=round(correction_dy, 2),
            duration_ms=round(correction_time, 1),
        ))

        # Phase 4: SETTLE — micro-adjustments
        settle_dx = random.gauss(0, 1.5)
        settle_dy = random.gauss(0, 1.5)
        settle_time = 10 + random.uniform(5, 20)

        phases.append(FlickPhase(
            name="settle",
            dx=round(settle_dx, 2),
            dy=round(settle_dy, 2),
            duration_ms=round(settle_time, 1),
        ))

        total_dx = sum(p.dx for p in phases)
        total_dy = sum(p.dy for p in phases)
        total_time = sum(p.duration_ms for p in phases)

        result = FlickResult(
            phases=phases,
            total_dx=round(total_dx, 2),
            total_dy=round(total_dy, 2),
            total_duration_ms=round(total_time, 1),
            overshoot_px=round(overshoot_dist, 1),
            target_id=target_id,
        )

        self._flick_history.append(result)
        if len(self._flick_history) > self._max_history:
            self._flick_history.pop(0)

        return result

    def set_skill(self, level: float):
        self._skill = max(0.1, min(1.0, level))

    def get_avg_overshoot(self, last_n: int = 20) -> float:
        """Average overshoot in pixels."""
        recent = self._flick_history[-last_n:]
        if not recent:
            return 0.0
        return round(sum(f.overshoot_px for f in recent) / len(recent), 1)

    def get_avg_flick_time(self, last_n: int = 20) -> float:
        """Average total flick time in ms."""
        recent = self._flick_history[-last_n:]
        if not recent:
            return 0.0
        return round(sum(f.total_duration_ms for f in recent) / len(recent), 1)

    def get_stats(self) -> dict:
        return {
            "skill_level": self._skill,
            "speed_multiplier": self._speed_mult,
            "total_flicks": len(self._flick_history),
            "avg_overshoot_px": self.get_avg_overshoot(),
            "avg_flick_time_ms": self.get_avg_flick_time(),
        }
