"""Autonomous Controller — Master loop for perception → cognition → action.

The top-level orchestrator for autonomous play. Runs the See → Think → Act
loop at target framerate, coordinating all subsystems through a single
coherent pipeline.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from loguru import logger

from gamer_companion.autonomous.safety_layer import SafetyLayer
from gamer_companion.autonomous.mode_manager import (
    ModeManager, PlayMode,
)


@dataclass
class LoopMetrics:
    """Performance metrics for the autonomous loop."""
    tick_count: int = 0
    total_perception_ms: float = 0
    total_cognition_ms: float = 0
    total_action_ms: float = 0
    last_tick_ms: float = 0
    max_tick_ms: float = 0
    actions_executed: int = 0
    actions_skipped: int = 0


@dataclass
class GameState:
    """Current game state as understood by the AI."""
    raw_frame: Any = None
    detections: List[dict] = field(default_factory=list)
    ocr_text: Dict[str, str] = field(default_factory=dict)
    audio_events: List[dict] = field(default_factory=list)
    game_phase: str = "unknown"
    health: Optional[int] = None
    ammo: Optional[int] = None
    score: Optional[dict] = None
    minimap_data: Optional[dict] = None
    timestamp: float = 0


@dataclass
class ActionPlan:
    """A planned action from the cognition engine."""
    action_type: str  # "mouse_move", "click", "key_press", "key_combo", "wait"
    params: dict = field(default_factory=dict)
    priority: float = 0.5
    confidence: float = 0.5
    reasoning: str = ""
    delay_ms: float = 0


class AutonomousController:
    """Master controller for the See → Think → Act loop.

    Architecture:
        1. PERCEIVE: Capture frame + audio → detect objects + read UI
        2. THINK: Analyze state → plan actions (MCTS/LLM/heuristic)
        3. ACT: Execute planned actions through safety-gated input
        4. LEARN: Record experience → update strategy weights

    The controller does NOT directly call Win32 input APIs. All actions
    route through SafetyLayer → InputControl subsystem.
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        safety: Optional[SafetyLayer] = None,
        mode_manager: Optional[ModeManager] = None,
    ):
        self.target_fps = target_fps
        self._tick_interval = 1.0 / target_fps
        self.safety = safety or SafetyLayer()
        self.mode = mode_manager or ModeManager()
        self._running = False
        self._game_state = GameState()
        self._action_queue: List[ActionPlan] = []
        self._metrics = LoopMetrics()

        # Subsystem hooks — set by the orchestrator
        self._perceive_fn = None
        self._think_fn = None
        self._act_fn = None
        self._learn_fn = None

        # Wire kill switch
        self.safety.on_kill(self._on_kill)

    def set_perceive(self, fn):
        """Set perception function: () -> GameState."""
        self._perceive_fn = fn

    def set_think(self, fn):
        """Set cognition function: (GameState) -> List[ActionPlan]."""
        self._think_fn = fn

    def set_act(self, fn):
        """Set action function: (ActionPlan) -> bool."""
        self._act_fn = fn

    def set_learn(self, fn):
        """Set learning function: (GameState, List[ActionPlan], results) -> None."""
        self._learn_fn = fn

    async def run(self):
        """Main autonomous loop."""
        self._running = True
        logger.info(
            f"Autonomous controller started. "
            f"Mode={self.mode.mode.value} FPS={self.target_fps}"
        )

        try:
            while self._running and not self.safety.is_killed:
                tick_start = time.perf_counter()

                # 1. PERCEIVE
                t0 = time.perf_counter()
                if self._perceive_fn:
                    try:
                        self._game_state = await self._call_async(
                            self._perceive_fn
                        )
                    except Exception as e:
                        logger.error(f"Perception error: {e}")
                perception_ms = (time.perf_counter() - t0) * 1000

                # 2. THINK
                t1 = time.perf_counter()
                actions = []
                if self._think_fn and self.mode.allows_input:
                    try:
                        actions = await self._call_async(
                            self._think_fn, self._game_state
                        )
                    except Exception as e:
                        logger.error(f"Cognition error: {e}")
                cognition_ms = (time.perf_counter() - t1) * 1000

                # 3. ACT (through safety gate)
                t2 = time.perf_counter()
                results = []
                if actions and self._act_fn and self.safety.is_active:
                    for action in sorted(
                        actions, key=lambda a: a.priority, reverse=True
                    ):
                        if self.safety.check_action(
                            action.action_type, action.params
                        ):
                            try:
                                ok = await self._call_async(
                                    self._act_fn, action
                                )
                                results.append((action, ok))
                                self._metrics.actions_executed += 1
                            except Exception as e:
                                logger.error(f"Action error: {e}")
                                results.append((action, False))
                        else:
                            self._metrics.actions_skipped += 1
                action_ms = (time.perf_counter() - t2) * 1000

                # 4. LEARN
                if self._learn_fn and results:
                    try:
                        await self._call_async(
                            self._learn_fn,
                            self._game_state, actions, results,
                        )
                    except Exception:
                        pass

                # Metrics
                tick_ms = (time.perf_counter() - tick_start) * 1000
                self._metrics.tick_count += 1
                self._metrics.total_perception_ms += perception_ms
                self._metrics.total_cognition_ms += cognition_ms
                self._metrics.total_action_ms += action_ms
                self._metrics.last_tick_ms = tick_ms
                self._metrics.max_tick_ms = max(
                    self._metrics.max_tick_ms, tick_ms
                )

                # Frame pacing
                elapsed = time.perf_counter() - tick_start
                sleep_time = self._tick_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Autonomous controller cancelled")
        except Exception as e:
            logger.error(f"Autonomous controller error: {e}")
        finally:
            self._running = False
            logger.info("Autonomous controller stopped")

    def stop(self):
        """Stop the autonomous loop gracefully."""
        self._running = False

    def _on_kill(self, reason: str):
        self._running = False
        logger.critical(f"Controller killed: {reason}")

    async def _call_async(self, fn, *args):
        """Call a function, awaiting if it's async."""
        result = fn(*args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @property
    def is_running(self) -> bool:
        return self._running

    def get_metrics(self) -> dict:
        n = max(self._metrics.tick_count, 1)
        return {
            "ticks": self._metrics.tick_count,
            "avg_perception_ms": round(
                self._metrics.total_perception_ms / n, 2
            ),
            "avg_cognition_ms": round(
                self._metrics.total_cognition_ms / n, 2
            ),
            "avg_action_ms": round(
                self._metrics.total_action_ms / n, 2
            ),
            "avg_tick_ms": round(
                self._metrics.last_tick_ms, 2
            ),
            "max_tick_ms": round(self._metrics.max_tick_ms, 2),
            "actions_executed": self._metrics.actions_executed,
            "actions_skipped": self._metrics.actions_skipped,
            "target_fps": self.target_fps,
            "mode": self.mode.mode.value,
            "safety": self.safety.get_stats(),
        }
