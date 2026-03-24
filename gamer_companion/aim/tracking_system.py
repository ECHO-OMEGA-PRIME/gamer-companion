"""Tracking System — Smooth aim tracking for moving targets.

Maintains aim on a moving target with:
- Velocity prediction (lead the target)
- Smoothing (avoid robotic snapping)
- Adaptive speed (faster when far, slower when close)
- Target locking with hysteresis
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from loguru import logger


@dataclass
class TrackedTarget:
    """A target being tracked over time."""
    entity_id: str
    x: float
    y: float
    vx: float = 0.0          # Velocity x (pixels/sec)
    vy: float = 0.0          # Velocity y (pixels/sec)
    last_seen: float = field(default_factory=time.time)
    frames_tracked: int = 0
    locked: bool = False
    distance_px: float = 0   # Distance from crosshair


@dataclass
class TrackingOutput:
    """Output from the tracking system."""
    dx: float                 # Aim delta X (pixels)
    dy: float                 # Aim delta Y (pixels)
    target_id: str
    predicted_x: float       # Where we think target will be
    predicted_y: float
    tracking_error: float    # Current error distance
    is_locked: bool
    confidence: float


class TrackingSystem:
    """Smooth aim tracking for moving targets.

    Uses exponential smoothing with velocity prediction.
    Tracks target position history to estimate velocity,
    then leads the aim ahead of the target's motion.

    Lock-on: Target is "locked" when tracking error < threshold
    for N consecutive frames. Locked targets get tighter smoothing.
    """

    def __init__(
        self,
        smoothing: float = 0.15,
        prediction_frames: int = 3,
        lock_threshold_px: float = 20.0,
        lock_frames: int = 5,
    ):
        self._smoothing = smoothing       # 0=instant, 1=very slow
        self._prediction_frames = prediction_frames
        self._lock_threshold = lock_threshold_px
        self._lock_frames = lock_frames
        self._targets: dict[str, TrackedTarget] = {}
        self._active_target: Optional[str] = None
        self._crosshair = (960.0, 540.0)
        self._history: List[TrackingOutput] = []
        self._max_history = 60
        self._consecutive_on_target = 0

    def set_crosshair(self, x: float, y: float):
        """Set current crosshair/screen center position."""
        self._crosshair = (x, y)

    def update_target(self, entity_id: str, x: float, y: float):
        """Update a target's position (call each frame)."""
        now = time.time()

        if entity_id in self._targets:
            target = self._targets[entity_id]
            dt = now - target.last_seen
            if dt > 0 and dt < 1.0:
                # Exponential moving average for velocity
                raw_vx = (x - target.x) / dt
                raw_vy = (y - target.y) / dt
                alpha = 0.3  # Velocity smoothing
                target.vx = target.vx * (1 - alpha) + raw_vx * alpha
                target.vy = target.vy * (1 - alpha) + raw_vy * alpha

            target.x = x
            target.y = y
            target.last_seen = now
            target.frames_tracked += 1
        else:
            self._targets[entity_id] = TrackedTarget(
                entity_id=entity_id, x=x, y=y, last_seen=now,
            )

        # Update distance
        cx, cy = self._crosshair
        self._targets[entity_id].distance_px = math.hypot(x - cx, y - cy)

    def remove_target(self, entity_id: str):
        if entity_id in self._targets:
            del self._targets[entity_id]
            if self._active_target == entity_id:
                self._active_target = None

    def set_active(self, entity_id: str) -> bool:
        """Set which target to actively track."""
        if entity_id not in self._targets:
            return False
        self._active_target = entity_id
        self._consecutive_on_target = 0
        return True

    def compute(self) -> Optional[TrackingOutput]:
        """Compute aim adjustment for the active target.

        Returns None if no active target.
        """
        if not self._active_target:
            return None

        target = self._targets.get(self._active_target)
        if not target:
            self._active_target = None
            return None

        cx, cy = self._crosshair

        # Predict future position
        dt = 1.0 / 60  # Assume ~60fps tick rate
        pred_x = target.x + target.vx * dt * self._prediction_frames
        pred_y = target.y + target.vy * dt * self._prediction_frames

        # Raw delta
        raw_dx = pred_x - cx
        raw_dy = pred_y - cy
        error = math.hypot(raw_dx, raw_dy)

        # Adaptive smoothing: less smoothing when far, more when close
        dist_factor = min(1.0, error / 200)  # Normalize to [0,1]
        effective_smooth = self._smoothing * (1 - dist_factor * 0.5)

        if target.locked:
            # Tighter tracking when locked
            effective_smooth *= 0.5

        # Apply smoothing
        dx = raw_dx * (1 - effective_smooth)
        dy = raw_dy * (1 - effective_smooth)

        # Lock-on detection
        if error < self._lock_threshold:
            self._consecutive_on_target += 1
            if self._consecutive_on_target >= self._lock_frames:
                target.locked = True
        else:
            self._consecutive_on_target = max(0, self._consecutive_on_target - 2)
            if self._consecutive_on_target == 0:
                target.locked = False

        confidence = max(0.1, min(0.99, 1.0 - error / 500))
        if target.locked:
            confidence = min(0.99, confidence + 0.2)

        output = TrackingOutput(
            dx=round(dx, 2),
            dy=round(dy, 2),
            target_id=target.entity_id,
            predicted_x=round(pred_x, 1),
            predicted_y=round(pred_y, 1),
            tracking_error=round(error, 1),
            is_locked=target.locked,
            confidence=round(confidence, 3),
        )

        self._history.append(output)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return output

    def get_closest_target(self) -> Optional[str]:
        """Get the entity_id of the closest target to crosshair."""
        if not self._targets:
            return None
        closest = min(
            self._targets.values(),
            key=lambda t: t.distance_px,
        )
        return closest.entity_id

    def prune_stale(self, max_age_s: float = 2.0):
        """Remove targets not seen recently."""
        now = time.time()
        stale = [
            eid for eid, t in self._targets.items()
            if now - t.last_seen > max_age_s
        ]
        for eid in stale:
            self.remove_target(eid)

    def get_avg_error(self, last_n: int = 30) -> float:
        """Get average tracking error over recent frames."""
        recent = self._history[-last_n:]
        if not recent:
            return 0.0
        return round(sum(o.tracking_error for o in recent) / len(recent), 1)

    def get_stats(self) -> dict:
        active = self._targets.get(self._active_target)
        return {
            "tracked_targets": len(self._targets),
            "active_target": self._active_target,
            "is_locked": active.locked if active else False,
            "avg_error": self.get_avg_error(),
            "consecutive_on": self._consecutive_on_target,
            "history_size": len(self._history),
        }
