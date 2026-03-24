"""Aim Engine — Core aiming system with tracking, flick, and prefire modes.

Computes aim adjustments given target positions. Does NOT move the mouse
directly — outputs dx/dy deltas that the input system applies with
humanization.
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum
from loguru import logger


class AimMode(Enum):
    TRACKING = "tracking"   # Smooth follow (moving targets)
    FLICK = "flick"         # Fast snap (new target appears)
    PREFIRE = "prefire"     # Pre-aim common angle
    SPRAY = "spray"         # Recoil compensation during spray
    IDLE = "idle"           # No active target


@dataclass
class AimTarget:
    """A target to aim at."""
    entity_id: str
    screen_x: float
    screen_y: float
    world_distance: float = 0
    hitbox_width: float = 30   # pixels
    hitbox_height: float = 50  # pixels
    velocity_x: float = 0     # pixels/sec
    velocity_y: float = 0
    health: int = 100
    is_moving: bool = False
    threat_score: float = 0.5


@dataclass
class AimResult:
    """Output of the aim engine."""
    dx: float  # Horizontal mouse delta (pixels)
    dy: float  # Vertical mouse delta (pixels)
    mode: AimMode
    confidence: float
    target_id: Optional[str] = None
    time_to_target_ms: float = 0


class AimEngine:
    """Core aiming system.

    Modes:
    - TRACKING: Smooth proportional aim at moving targets
    - FLICK: Fast aim at newly-appeared targets (Bezier-like)
    - PREFIRE: Slow aim toward expected positions
    - SPRAY: Compensate for weapon recoil pattern

    Output is (dx, dy) mouse deltas — the input system handles
    actual movement with humanization.
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        smoothing: float = 0.15,
        flick_speed: float = 2.0,
        skill_level: float = 0.85,
    ):
        self.sensitivity = sensitivity
        self.smoothing = smoothing  # 0 = instant, 1 = very slow
        self.flick_speed = flick_speed
        self.skill_level = skill_level  # 0-1, affects accuracy
        self._current_mode = AimMode.IDLE
        self._current_target: Optional[AimTarget] = None
        self._spray_index = 0
        self._last_aim_time = 0
        self._screen_center = (960, 540)  # Default 1080p

    def set_screen_center(self, x: int, y: int):
        self._screen_center = (x, y)

    def compute(
        self, target: AimTarget,
        spray_pattern: List[Tuple[float, float]] = None,
    ) -> AimResult:
        """Compute aim adjustment for a target.

        Args:
            target: Target to aim at
            spray_pattern: Per-shot recoil offsets (dx_deg, dy_deg)
        """
        cx, cy = self._screen_center
        raw_dx = target.screen_x - cx
        raw_dy = target.screen_y - cy
        dist = math.hypot(raw_dx, raw_dy)

        # Determine aim mode
        now = time.time()
        dt = now - self._last_aim_time if self._last_aim_time > 0 else 0.033

        if self._current_target and self._current_target.entity_id == target.entity_id:
            if dist > 200:
                mode = AimMode.FLICK
            else:
                mode = AimMode.TRACKING
        else:
            mode = AimMode.FLICK if dist > 100 else AimMode.TRACKING

        self._current_target = target
        self._current_mode = mode
        self._last_aim_time = now

        # Compute aim delta
        if mode == AimMode.FLICK:
            dx, dy = self._flick_aim(raw_dx, raw_dy, dist, target)
        else:
            dx, dy = self._tracking_aim(raw_dx, raw_dy, dist, target, dt)

        # Apply spray compensation
        if spray_pattern and self._spray_index < len(spray_pattern):
            sx, sy = spray_pattern[self._spray_index]
            dx -= sx * self.sensitivity
            dy -= sy * self.sensitivity
            self._spray_index += 1

        # Apply skill-based inaccuracy
        inaccuracy = (1.0 - self.skill_level) * 5.0
        dx += random.gauss(0, inaccuracy)
        dy += random.gauss(0, inaccuracy)

        # Confidence based on distance and mode
        confidence = max(0.1, min(0.99, 1.0 - dist / 1000))
        if mode == AimMode.FLICK:
            confidence *= 0.8

        ttt = dist / max(1, self.flick_speed * 500) * 1000

        return AimResult(
            dx=round(dx, 2),
            dy=round(dy, 2),
            mode=mode,
            confidence=round(confidence, 3),
            target_id=target.entity_id,
            time_to_target_ms=round(ttt, 1),
        )

    def reset_spray(self):
        """Reset spray compensation index (new burst)."""
        self._spray_index = 0

    def _tracking_aim(
        self, raw_dx: float, raw_dy: float,
        dist: float, target: AimTarget, dt: float,
    ) -> Tuple[float, float]:
        """Smooth proportional tracking."""
        # Lead the target if moving
        lead_frames = 3
        lead_x = target.velocity_x * dt * lead_frames
        lead_y = target.velocity_y * dt * lead_frames

        dx = (raw_dx + lead_x) * (1 - self.smoothing) * self.sensitivity
        dy = (raw_dy + lead_y) * (1 - self.smoothing) * self.sensitivity

        return (dx, dy)

    def _flick_aim(
        self, raw_dx: float, raw_dy: float,
        dist: float, target: AimTarget,
    ) -> Tuple[float, float]:
        """Fast snap aim toward target."""
        # Flick covers most of the distance, leave rest for tracking
        coverage = min(0.95, 0.7 + self.skill_level * 0.25)

        dx = raw_dx * coverage * self.sensitivity * self.flick_speed
        dy = raw_dy * coverage * self.sensitivity * self.flick_speed

        # Overshoot probability
        if random.random() < 0.1:
            overshoot = random.uniform(1.02, 1.15)
            dx *= overshoot
            dy *= overshoot

        return (dx, dy)

    @property
    def mode(self) -> AimMode:
        return self._current_mode

    def get_stats(self) -> dict:
        return {
            "mode": self._current_mode.value,
            "sensitivity": self.sensitivity,
            "skill_level": self.skill_level,
            "spray_index": self._spray_index,
            "target": self._current_target.entity_id if self._current_target else None,
        }
