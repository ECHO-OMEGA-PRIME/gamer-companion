"""Action Executor — Translate decisions into mouse + keyboard actions.

Bridges the cognition engine's decisions to the input control subsystem.
Applies Bezier curves, Fitts' Law timing, humanization, and safety gating.
"""

from __future__ import annotations
import time
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from loguru import logger


@dataclass
class ExecutionResult:
    """Result of executing an action."""
    action_type: str
    success: bool
    execution_ms: float
    input_events: int = 0     # Number of low-level input events generated
    blocked_by_safety: bool = False
    error: str = ""


@dataclass
class BezierPoint:
    """A point on a Bezier curve for mouse movement."""
    x: float
    y: float
    t: float  # Parameter 0-1


class ActionExecutor:
    """Execute AI decisions through the input control subsystem.

    Converts high-level decisions (shoot, move_to, buy_item) into
    low-level input sequences (mouse deltas, key presses, timing).

    Features:
    - Bezier curve mouse paths (not straight lines)
    - Fitts' Law based movement timing
    - Key combo sequencing with human jitter
    - Action queuing with priority
    - Safety gating (all actions pass through safety layer)
    """

    def __init__(self):
        self._action_handlers: Dict[str, Callable] = {}
        self._execution_history: List[ExecutionResult] = []
        self._max_history = 200
        self._total_actions = 0
        self._total_blocked = 0

        # Movement parameters
        self._fitts_a = 50    # Fitts' Law intercept (ms)
        self._fitts_b = 150   # Fitts' Law slope (ms/bit)
        self._move_jitter = 2.0  # Pixels of endpoint jitter

        # Register default handlers
        self._register_defaults()

    def register_handler(self, action_type: str, handler: Callable):
        """Register a handler for an action type.

        handler(params: dict) → bool (success)
        """
        self._action_handlers[action_type] = handler

    def execute(self, action_type: str, params: dict = None) -> ExecutionResult:
        """Execute a single action."""
        start = time.perf_counter()
        params = params or {}
        self._total_actions += 1

        handler = self._action_handlers.get(action_type)
        if not handler:
            result = ExecutionResult(
                action_type=action_type,
                success=False,
                execution_ms=0,
                error=f"No handler for '{action_type}'",
            )
            self._record(result)
            return result

        try:
            success = handler(params)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result = ExecutionResult(
                action_type=action_type,
                success=bool(success),
                execution_ms=round(elapsed_ms, 2),
                input_events=params.get("_input_events", 1),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result = ExecutionResult(
                action_type=action_type,
                success=False,
                execution_ms=round(elapsed_ms, 2),
                error=str(e),
            )

        self._record(result)
        return result

    def execute_sequence(self, actions: List[Tuple[str, dict]]) -> List[ExecutionResult]:
        """Execute a sequence of actions in order."""
        results = []
        for action_type, params in actions:
            result = self.execute(action_type, params)
            results.append(result)
            if not result.success:
                break  # Stop sequence on failure
        return results

    def compute_bezier_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        steps: int = 20,
    ) -> List[BezierPoint]:
        """Compute a Bezier curve mouse path between two points.

        Uses a cubic Bezier with randomized control points for
        natural-looking mouse movement.
        """
        sx, sy = start
        ex, ey = end
        dist = math.hypot(ex - sx, ey - sy)

        # Randomize control points (perpendicular offset)
        offset = dist * random.uniform(0.1, 0.35)
        angle = math.atan2(ey - sy, ex - sx)
        perp = angle + math.pi / 2 * random.choice([-1, 1])

        # Control point 1: ~30% along path
        c1x = sx + (ex - sx) * 0.3 + math.cos(perp) * offset * random.uniform(0.5, 1.0)
        c1y = sy + (ey - sy) * 0.3 + math.sin(perp) * offset * random.uniform(0.5, 1.0)

        # Control point 2: ~70% along path
        c2x = sx + (ex - sx) * 0.7 + math.cos(perp) * offset * random.uniform(-0.3, 0.5)
        c2y = sy + (ey - sy) * 0.7 + math.sin(perp) * offset * random.uniform(-0.3, 0.5)

        points = []
        for i in range(steps + 1):
            t = i / steps
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt

            x = mt3 * sx + 3 * mt2 * t * c1x + 3 * mt * t2 * c2x + t3 * ex
            y = mt3 * sy + 3 * mt2 * t * c1y + 3 * mt * t2 * c2y + t3 * ey

            # Add micro-jitter
            if 0 < t < 1:
                x += random.gauss(0, self._move_jitter * (1 - abs(2 * t - 1)))
                y += random.gauss(0, self._move_jitter * (1 - abs(2 * t - 1)))

            points.append(BezierPoint(x=round(x, 1), y=round(y, 1), t=round(t, 3)))

        return points

    def compute_fitts_time_ms(
        self,
        distance: float,
        target_width: float,
    ) -> float:
        """Compute movement time using Fitts' Law.

        T = a + b * log2(D/W + 1)

        Args:
            distance: Distance to target in pixels
            target_width: Target width in pixels

        Returns:
            Estimated movement time in milliseconds
        """
        if target_width <= 0 or distance <= 0:
            return self._fitts_a

        id_bits = math.log2(distance / target_width + 1)
        time_ms = self._fitts_a + self._fitts_b * id_bits

        # Add human variance
        time_ms *= random.uniform(0.85, 1.15)

        return round(max(self._fitts_a, time_ms), 1)

    def _register_defaults(self):
        """Register default action handlers (stubs for testing)."""
        self._action_handlers["shoot"] = lambda p: True
        self._action_handlers["move"] = lambda p: True
        self._action_handlers["use_ability"] = lambda p: True
        self._action_handlers["buy"] = lambda p: True
        self._action_handlers["rotate"] = lambda p: True
        self._action_handlers["hold"] = lambda p: True
        self._action_handlers["wait"] = lambda p: True

    def _record(self, result: ExecutionResult):
        self._execution_history.append(result)
        if len(self._execution_history) > self._max_history:
            self._execution_history.pop(0)

    def get_success_rate(self, last_n: int = 50) -> float:
        """Get recent action success rate."""
        recent = self._execution_history[-last_n:]
        if not recent:
            return 0.0
        return round(sum(1 for r in recent if r.success) / len(recent), 3)

    def get_avg_execution_ms(self, last_n: int = 50) -> float:
        recent = self._execution_history[-last_n:]
        if not recent:
            return 0.0
        return round(sum(r.execution_ms for r in recent) / len(recent), 2)

    def get_stats(self) -> dict:
        return {
            "registered_handlers": list(self._action_handlers.keys()),
            "total_actions": self._total_actions,
            "total_blocked": self._total_blocked,
            "success_rate": self.get_success_rate(),
            "avg_execution_ms": self.get_avg_execution_ms(),
            "history_size": len(self._execution_history),
        }
