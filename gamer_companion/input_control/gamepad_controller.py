"""Gamepad Controller — XInput controller support for console-style games.

Provides virtual gamepad input for games that support controllers:
- Analog stick movement with dead zones
- Trigger pressure with curves
- Button press/release with timing
- Vibration feedback integration
- Stick drift simulation for realism
"""

from __future__ import annotations
import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from loguru import logger


@dataclass
class StickState:
    """Analog stick state (-1.0 to 1.0 per axis)."""
    x: float = 0.0
    y: float = 0.0

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y, self.x))

    def apply_deadzone(self, deadzone: float = 0.15) -> 'StickState':
        """Apply radial dead zone."""
        mag = self.magnitude()
        if mag < deadzone:
            return StickState(0.0, 0.0)
        # Rescale outside dead zone to [0, 1]
        scale = (mag - deadzone) / (1.0 - deadzone) / mag
        return StickState(
            x=round(max(-1.0, min(1.0, self.x * scale)), 4),
            y=round(max(-1.0, min(1.0, self.y * scale)), 4),
        )


@dataclass
class GamepadState:
    """Full gamepad state snapshot."""
    left_stick: StickState = field(default_factory=StickState)
    right_stick: StickState = field(default_factory=StickState)
    left_trigger: float = 0.0    # 0.0 to 1.0
    right_trigger: float = 0.0   # 0.0 to 1.0
    buttons: Dict[str, bool] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def is_pressed(self, button: str) -> bool:
        return self.buttons.get(button, False)


# Standard button map (XInput layout)
XINPUT_BUTTONS = [
    "A", "B", "X", "Y",
    "LB", "RB",
    "L3", "R3",
    "START", "SELECT",
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
]


class GamepadController:
    """Virtual gamepad controller with human-like input.

    Features:
    - Analog stick movement with smooth interpolation
    - Dead zone handling (radial, axial)
    - Trigger curves (linear, exponential, S-curve)
    - Button press/release with timing jitter
    - Stick drift for realism
    - Haptic feedback state tracking
    """

    def __init__(self, deadzone: float = 0.15, drift_amount: float = 0.02):
        self._state = GamepadState()
        self._deadzone = deadzone
        self._drift_amount = drift_amount
        self._history: List[GamepadState] = []
        self._max_history = 100
        self._vibration = (0.0, 0.0)  # (left_motor, right_motor) 0-1

    def set_left_stick(self, x: float, y: float, smooth: bool = True) -> StickState:
        """Set left stick position with optional smoothing."""
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))

        if smooth:
            # Interpolate 70% toward target (simulates physical stick movement)
            cx, cy = self._state.left_stick.x, self._state.left_stick.y
            x = cx + (x - cx) * 0.7
            y = cy + (y - cy) * 0.7

        # Add micro-drift
        if self._drift_amount > 0:
            x += random.gauss(0, self._drift_amount)
            y += random.gauss(0, self._drift_amount)

        stick = StickState(x=round(x, 4), y=round(y, 4))
        self._state.left_stick = stick.apply_deadzone(self._deadzone)
        self._record()
        return self._state.left_stick

    def set_right_stick(self, x: float, y: float, smooth: bool = True) -> StickState:
        """Set right stick position (aim stick)."""
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))

        if smooth:
            cx, cy = self._state.right_stick.x, self._state.right_stick.y
            x = cx + (x - cx) * 0.7
            y = cy + (y - cy) * 0.7

        if self._drift_amount > 0:
            x += random.gauss(0, self._drift_amount * 0.5)
            y += random.gauss(0, self._drift_amount * 0.5)

        stick = StickState(x=round(x, 4), y=round(y, 4))
        self._state.right_stick = stick.apply_deadzone(self._deadzone)
        self._record()
        return self._state.right_stick

    def set_trigger(self, trigger: str, value: float, curve: str = "linear") -> float:
        """Set trigger value with response curve.

        Args:
            trigger: "left" or "right"
            value: 0.0 to 1.0
            curve: "linear", "exponential", "s_curve"
        """
        value = max(0.0, min(1.0, value))

        # Apply response curve
        if curve == "exponential":
            value = value ** 2
        elif curve == "s_curve":
            # Sigmoid-like
            value = 1.0 / (1.0 + math.exp(-10 * (value - 0.5)))

        value = round(value, 4)

        if trigger == "left":
            self._state.left_trigger = value
        elif trigger == "right":
            self._state.right_trigger = value

        self._record()
        return value

    def press_button(self, button: str) -> bool:
        """Press a button."""
        if button not in XINPUT_BUTTONS:
            logger.warning(f"Unknown button: {button}")
            return False

        self._state.buttons[button] = True
        self._record()
        return True

    def release_button(self, button: str) -> bool:
        """Release a button."""
        self._state.buttons[button] = False
        self._record()
        return True

    def release_all(self):
        """Release all inputs (panic/reset)."""
        self._state = GamepadState()
        self._record()

    def is_pressed(self, button: str) -> bool:
        return self._state.is_pressed(button)

    def set_vibration(self, left_motor: float, right_motor: float):
        """Set vibration state (for tracking, actual vibration needs driver)."""
        self._vibration = (
            max(0.0, min(1.0, left_motor)),
            max(0.0, min(1.0, right_motor)),
        )

    def aim_direction(self, angle_deg: float, magnitude: float = 1.0) -> StickState:
        """Set right stick by angle and magnitude (aim assist helper)."""
        rad = math.radians(angle_deg)
        x = math.cos(rad) * min(1.0, magnitude)
        y = math.sin(rad) * min(1.0, magnitude)
        return self.set_right_stick(x, y, smooth=True)

    def move_direction(self, angle_deg: float, magnitude: float = 1.0) -> StickState:
        """Set left stick by angle and magnitude."""
        rad = math.radians(angle_deg)
        x = math.cos(rad) * min(1.0, magnitude)
        y = math.sin(rad) * min(1.0, magnitude)
        return self.set_left_stick(x, y, smooth=True)

    @property
    def state(self) -> GamepadState:
        return self._state

    def _record(self):
        self._state.timestamp = time.time()
        if len(self._history) >= self._max_history:
            self._history.pop(0)
        self._history.append(GamepadState(
            left_stick=StickState(self._state.left_stick.x, self._state.left_stick.y),
            right_stick=StickState(self._state.right_stick.x, self._state.right_stick.y),
            left_trigger=self._state.left_trigger,
            right_trigger=self._state.right_trigger,
            buttons=dict(self._state.buttons),
            timestamp=self._state.timestamp,
        ))

    def get_stats(self) -> dict:
        pressed = [b for b, v in self._state.buttons.items() if v]
        return {
            "left_stick": (self._state.left_stick.x, self._state.left_stick.y),
            "right_stick": (self._state.right_stick.x, self._state.right_stick.y),
            "left_trigger": self._state.left_trigger,
            "right_trigger": self._state.right_trigger,
            "buttons_pressed": pressed,
            "vibration": self._vibration,
            "history_size": len(self._history),
        }
