"""Mode Manager — Switch between 8 play modes with appropriate safety levels.

Modes:
  OBSERVE   — Watch and analyze only (no input)
  ASSIST    — Suggestions via overlay/voice (no input)
  COACH     — Real-time voice coaching (no input)
  COPILOT   — AI handles movement, human aims (partial input)
  AUTONOMOUS— Full AI control (full input)
  TRAINING  — AI runs drills/aim trainers (full input, practice only)
  MIMIC     — Play in a pro player's style (full input, constrained)
  SWARM     — Multi-AI squad coordination (full input, multi-agent)
"""

from __future__ import annotations
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict
from loguru import logger


class PlayMode(Enum):
    OBSERVE = "observe"
    ASSIST = "assist"
    COACH = "coach"
    COPILOT = "copilot"
    AUTONOMOUS = "autonomous"
    TRAINING = "training"
    MIMIC = "mimic"
    SWARM = "swarm"


@dataclass
class ModeConfig:
    """Configuration for a play mode."""
    mode: PlayMode
    allows_mouse: bool = False
    allows_keyboard: bool = False
    allows_voice_output: bool = False
    allows_overlay: bool = True
    ai_control_pct: float = 0.0  # 0.0 = no AI input, 1.0 = full
    requires_safety: bool = False
    description: str = ""


# Default configurations for each mode
MODE_CONFIGS: Dict[PlayMode, ModeConfig] = {
    PlayMode.OBSERVE: ModeConfig(
        mode=PlayMode.OBSERVE,
        allows_voice_output=False, allows_overlay=True,
        ai_control_pct=0.0, description="Watch and analyze only",
    ),
    PlayMode.ASSIST: ModeConfig(
        mode=PlayMode.ASSIST,
        allows_voice_output=True, allows_overlay=True,
        ai_control_pct=0.0, description="Suggestions via overlay and voice",
    ),
    PlayMode.COACH: ModeConfig(
        mode=PlayMode.COACH,
        allows_voice_output=True, allows_overlay=True,
        ai_control_pct=0.0, description="Real-time voice coaching",
    ),
    PlayMode.COPILOT: ModeConfig(
        mode=PlayMode.COPILOT,
        allows_mouse=True, allows_keyboard=True,
        allows_voice_output=True, allows_overlay=True,
        ai_control_pct=0.5, requires_safety=True,
        description="AI handles movement, human aims/shoots",
    ),
    PlayMode.AUTONOMOUS: ModeConfig(
        mode=PlayMode.AUTONOMOUS,
        allows_mouse=True, allows_keyboard=True,
        allows_voice_output=True, allows_overlay=True,
        ai_control_pct=1.0, requires_safety=True,
        description="Full AI control",
    ),
    PlayMode.TRAINING: ModeConfig(
        mode=PlayMode.TRAINING,
        allows_mouse=True, allows_keyboard=True,
        allows_overlay=True,
        ai_control_pct=1.0, requires_safety=True,
        description="AI runs drills and aim trainers",
    ),
    PlayMode.MIMIC: ModeConfig(
        mode=PlayMode.MIMIC,
        allows_mouse=True, allows_keyboard=True,
        allows_voice_output=True, allows_overlay=True,
        ai_control_pct=1.0, requires_safety=True,
        description="Play in a pro player's style",
    ),
    PlayMode.SWARM: ModeConfig(
        mode=PlayMode.SWARM,
        allows_mouse=True, allows_keyboard=True,
        allows_voice_output=True, allows_overlay=True,
        ai_control_pct=1.0, requires_safety=True,
        description="Multi-AI squad coordination",
    ),
}


class ModeManager:
    """Manage play modes with safety-aware transitions.

    Tracks mode history, enforces valid transitions, and notifies
    subscribers on mode changes.
    """

    # Modes that require explicit confirmation to enter
    DANGEROUS_MODES = {
        PlayMode.AUTONOMOUS, PlayMode.SWARM, PlayMode.MIMIC,
    }

    def __init__(self, initial_mode: PlayMode = PlayMode.OBSERVE):
        self._current_mode = initial_mode
        self._config = MODE_CONFIGS[initial_mode]
        self._mode_start = time.time()
        self._history: List[dict] = []
        self._callbacks: List[Callable] = []
        self._locked = False

    def switch(
        self, new_mode: PlayMode, confirmed: bool = False,
    ) -> bool:
        """Switch to a new play mode.

        Args:
            new_mode: Target mode
            confirmed: Required True for dangerous modes
        """
        if self._locked:
            logger.warning("Mode switch blocked: mode is locked")
            return False

        if new_mode == self._current_mode:
            return True

        if new_mode in self.DANGEROUS_MODES and not confirmed:
            logger.warning(
                f"Mode {new_mode.value} requires explicit confirmation. "
                "Pass confirmed=True."
            )
            return False

        old_mode = self._current_mode
        duration = time.time() - self._mode_start

        self._history.append({
            "from": old_mode.value,
            "to": new_mode.value,
            "duration": round(duration, 2),
            "timestamp": time.time(),
        })

        self._current_mode = new_mode
        self._config = MODE_CONFIGS[new_mode]
        self._mode_start = time.time()

        logger.info(
            f"Mode: {old_mode.value} → {new_mode.value} "
            f"(control={self._config.ai_control_pct*100:.0f}%)"
        )

        for cb in self._callbacks:
            try:
                cb(old_mode, new_mode)
            except Exception as e:
                logger.error(f"Mode callback error: {e}")

        return True

    def lock(self):
        """Lock current mode — cannot be changed until unlocked."""
        self._locked = True

    def unlock(self):
        self._locked = False

    def on_change(self, callback: Callable[[PlayMode, PlayMode], None]):
        self._callbacks.append(callback)

    @property
    def mode(self) -> PlayMode:
        return self._current_mode

    @property
    def config(self) -> ModeConfig:
        return self._config

    @property
    def allows_input(self) -> bool:
        return self._config.allows_mouse or self._config.allows_keyboard

    @property
    def ai_control(self) -> float:
        return self._config.ai_control_pct

    def get_stats(self) -> dict:
        return {
            "current_mode": self._current_mode.value,
            "ai_control_pct": self._config.ai_control_pct,
            "allows_mouse": self._config.allows_mouse,
            "allows_keyboard": self._config.allows_keyboard,
            "allows_voice": self._config.allows_voice_output,
            "requires_safety": self._config.requires_safety,
            "locked": self._locked,
            "mode_duration": round(time.time() - self._mode_start, 1),
            "transitions": len(self._history),
        }
