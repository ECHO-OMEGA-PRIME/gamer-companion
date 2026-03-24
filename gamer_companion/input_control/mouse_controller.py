"""Mouse Controller — Bezier curve mouse movement with humanization.

All mouse movement uses cubic Bezier curves with randomized control points
to produce smooth, human-like trajectories. No straight-line teleportation.
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional
from loguru import logger

try:
    import ctypes
    import ctypes.wintypes

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_ABSOLUTE = 0x8000

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False


@dataclass
class MouseState:
    """Current mouse position and button state."""
    x: int = 0
    y: int = 0
    left_down: bool = False
    right_down: bool = False
    last_move_time: float = 0


def cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    """Evaluate cubic Bezier at parameter t in [0, 1]."""
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def generate_bezier_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    steps: int = 20,
    jitter: float = 0.3,
) -> List[Tuple[int, int]]:
    """Generate a Bezier curve path from start to end.

    Control points are randomized to create natural-looking curves.
    jitter controls how much the path deviates from a straight line.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)

    # Control points offset perpendicular to the line
    spread = dist * jitter
    angle = math.atan2(dy, dx)
    perp = angle + math.pi / 2

    cp1 = (
        start[0] + dx * 0.25 + random.gauss(0, spread * 0.5) * math.cos(perp),
        start[1] + dy * 0.25 + random.gauss(0, spread * 0.5) * math.sin(perp),
    )
    cp2 = (
        start[0] + dx * 0.75 + random.gauss(0, spread * 0.3) * math.cos(perp),
        start[1] + dy * 0.75 + random.gauss(0, spread * 0.3) * math.sin(perp),
    )

    path = []
    for i in range(steps + 1):
        t = i / steps
        px, py = cubic_bezier(
            (float(start[0]), float(start[1])),
            cp1, cp2,
            (float(end[0]), float(end[1])),
            t,
        )
        path.append((round(px), round(py)))

    return path


def fitts_law_time(distance: float, target_size: float, a: float = 0.1, b: float = 0.15) -> float:
    """Fitts' Law: time = a + b * log2(1 + distance/target_size).

    Returns estimated movement time in seconds.
    """
    if target_size <= 0:
        target_size = 1
    return a + b * math.log2(1 + distance / target_size)


class MouseController:
    """Humanized mouse controller with Bezier curves and Fitts' Law.

    Features:
    - Cubic Bezier curve trajectories (no straight lines)
    - Fitts' Law speed scaling (far = fast, near = slow)
    - Gaussian micro-jitter on final position
    - Overshoot + correction on fast flicks
    - Variable step timing (accelerate-decelerate)
    """

    def __init__(
        self,
        overshoot_chance: float = 0.15,
        overshoot_distance: float = 8.0,
        final_jitter_px: float = 2.0,
    ):
        self._state = MouseState()
        self._overshoot_chance = overshoot_chance
        self._overshoot_dist = overshoot_distance
        self._final_jitter = final_jitter_px
        self._refresh_position()

    def move_to(
        self, x: int, y: int,
        target_size: float = 20.0,
        steps: int = 0,
    ) -> List[Tuple[int, int]]:
        """Move mouse to (x, y) along a Bezier curve.

        Returns the path taken (list of points). Does NOT actually
        move the system cursor — call execute_path() for that.

        Args:
            x, y: Target position
            target_size: Size of target in pixels (for Fitts' Law)
            steps: Number of interpolation steps (0 = auto from distance)
        """
        self._refresh_position()
        start = (self._state.x, self._state.y)
        end = (x, y)

        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        if dist < 2:
            return [end]

        # Auto steps from distance
        if steps == 0:
            steps = max(8, min(60, int(dist / 8)))

        path = generate_bezier_path(start, end, steps=steps)

        # Overshoot simulation
        if random.random() < self._overshoot_chance and dist > 50:
            overshoot_x = end[0] + random.gauss(0, self._overshoot_dist)
            overshoot_y = end[1] + random.gauss(0, self._overshoot_dist)
            path.append((round(overshoot_x), round(overshoot_y)))
            # Correction back
            correction = generate_bezier_path(
                (round(overshoot_x), round(overshoot_y)),
                end, steps=5, jitter=0.1,
            )
            path.extend(correction)

        # Final jitter
        if self._final_jitter > 0:
            last = path[-1]
            jx = round(last[0] + random.gauss(0, self._final_jitter))
            jy = round(last[1] + random.gauss(0, self._final_jitter))
            path[-1] = (jx, jy)

        return path

    def execute_path(self, path: List[Tuple[int, int]], total_time_ms: float = 0):
        """Execute a mouse path by moving the system cursor.

        Args:
            path: List of (x, y) points
            total_time_ms: Total time in ms (0 = Fitts' Law auto)
        """
        if not HAS_WIN32 or not path:
            return

        if total_time_ms <= 0 and len(path) >= 2:
            dist = math.hypot(
                path[-1][0] - path[0][0],
                path[-1][1] - path[0][1],
            )
            total_time_ms = fitts_law_time(dist, 20.0) * 1000

        step_delay = total_time_ms / max(len(path), 1) / 1000

        for px, py in path:
            user32.SetCursorPos(px, py)
            self._state.x = px
            self._state.y = py
            if step_delay > 0:
                time.sleep(step_delay)

        self._state.last_move_time = time.time()

    def click(self, button: str = "left"):
        """Click at current position."""
        if not HAS_WIN32:
            return
        if button == "left":
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(random.uniform(0.04, 0.08))
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._state.left_down = False
        elif button == "right":
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            time.sleep(random.uniform(0.04, 0.08))
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            self._state.right_down = False

    def get_position(self) -> Tuple[int, int]:
        self._refresh_position()
        return (self._state.x, self._state.y)

    def _refresh_position(self):
        if not HAS_WIN32:
            return
        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        self._state.x = pt.x
        self._state.y = pt.y
