"""Keyboard Controller — Key press/release/combos with humanized timing.

Provides humanized keyboard input with:
- Variable press durations (gaussian around typical human press ~80-120ms)
- Inter-key delays for combos/sequences
- Key hold support with variable timing
- Full virtual key code mapping
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from loguru import logger

try:
    import ctypes

    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

# Virtual key codes for common game keys
VK_MAP: Dict[str, int] = {
    # Letters
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    # Numbers
    **{str(i): 0x30 + i for i in range(10)},
    # Function keys
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
    # Common
    "space": 0x20, "enter": 0x0D, "escape": 0x1B, "tab": 0x09,
    "shift": 0xA0, "lshift": 0xA0, "rshift": 0xA1,
    "ctrl": 0xA2, "lctrl": 0xA2, "rctrl": 0xA3,
    "alt": 0xA4, "lalt": 0xA4, "ralt": 0xA5,
    "backspace": 0x08, "delete": 0x2E, "insert": 0x2D,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
    # Mouse buttons (for completeness)
    "mouse1": 0x01, "mouse2": 0x02, "mouse3": 0x04,
}


@dataclass
class KeyState:
    """Track which keys are currently held."""
    held_keys: Dict[str, float] = field(default_factory=dict)


class KeyboardController:
    """Humanized keyboard input controller.

    Features:
    - Gaussian press duration (~80-120ms, std=15ms)
    - Inter-key delay for combos
    - Key hold with variable timing
    - Combo execution (e.g., Ctrl+Shift+S)
    """

    def __init__(
        self,
        press_duration_ms: float = 100.0,
        press_duration_std: float = 15.0,
        inter_key_delay_ms: float = 30.0,
    ):
        self._press_duration = press_duration_ms
        self._press_std = press_duration_std
        self._inter_key_delay = inter_key_delay_ms
        self._state = KeyState()

    def press(self, key: str, duration_ms: float = 0):
        """Press and release a key with humanized timing.

        Args:
            key: Key name (e.g., "W", "space", "ctrl")
            duration_ms: Hold duration (0 = random human-like)
        """
        vk = self._resolve_vk(key)
        if vk is None:
            logger.warning(f"Unknown key: {key}")
            return

        if duration_ms <= 0:
            duration_ms = max(
                30, random.gauss(self._press_duration, self._press_std)
            )

        self._key_down(vk)
        time.sleep(duration_ms / 1000)
        self._key_up(vk)

    def hold(self, key: str):
        """Hold a key down (must call release() later)."""
        vk = self._resolve_vk(key)
        if vk is None:
            return
        self._key_down(vk)
        self._state.held_keys[key.lower()] = time.time()

    def release(self, key: str):
        """Release a held key."""
        vk = self._resolve_vk(key)
        if vk is None:
            return
        self._key_up(vk)
        self._state.held_keys.pop(key.lower(), None)

    def release_all(self):
        """Release all held keys."""
        for key in list(self._state.held_keys.keys()):
            self.release(key)

    def combo(self, keys: List[str], final_press_ms: float = 0):
        """Execute a key combo (e.g., ["ctrl", "shift", "s"]).

        Holds all modifier keys, presses the final key, then releases.
        """
        if not keys:
            return

        modifiers = keys[:-1]
        final = keys[-1]

        # Hold modifiers
        for mod in modifiers:
            vk = self._resolve_vk(mod)
            if vk:
                self._key_down(vk)
                time.sleep(self._inter_key_delay / 1000)

        # Press final key
        self.press(final, duration_ms=final_press_ms)

        # Release modifiers in reverse
        for mod in reversed(modifiers):
            vk = self._resolve_vk(mod)
            if vk:
                time.sleep(self._inter_key_delay / 1000)
                self._key_up(vk)

    def type_sequence(self, keys: List[str], delay_ms: float = 0):
        """Press a sequence of keys one at a time with delays."""
        if delay_ms <= 0:
            delay_ms = self._inter_key_delay

        for key in keys:
            self.press(key)
            time.sleep(delay_ms / 1000)

    @property
    def held(self) -> List[str]:
        return list(self._state.held_keys.keys())

    def _resolve_vk(self, key: str) -> Optional[int]:
        key_lower = key.lower()
        if key_lower in VK_MAP:
            return VK_MAP[key_lower]
        if len(key) == 1 and key.isalpha():
            return ord(key.upper())
        return None

    def _key_down(self, vk: int):
        if HAS_WIN32:
            user32.keybd_event(vk, 0, 0, 0)

    def _key_up(self, vk: int):
        if HAS_WIN32:
            user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
