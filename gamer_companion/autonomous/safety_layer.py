"""Safety Layer — Hard limits, kill switch, rate limiting, session boundaries.

Absolute boundaries that cannot be overridden by any other module:
- Kill switch: F12 = permanent disable, any key during autonomous = pause
- Max 15 APS (human maximum ~12)
- Min 90ms reaction time (human minimum ~150ms)
- Random micro-pauses every 30-60s
- Session limit: auto-stop at configurable hours
- Full JSON audit log of every input
"""

from __future__ import annotations
import time
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from pathlib import Path
from loguru import logger


@dataclass
class InputAuditEntry:
    """A single audited input action."""
    timestamp: float
    action_type: str  # "mouse_move", "mouse_click", "key_press", "key_release"
    details: dict = field(default_factory=dict)
    latency_ms: float = 0
    was_throttled: bool = False


class SafetyLayer:
    """Absolute safety boundaries for autonomous play.

    This layer wraps ALL input. No mouse/keyboard action reaches the OS
    without passing through here first. It enforces:

    1. APS (Actions Per Second) hard cap
    2. Minimum reaction time floor
    3. Session duration limit
    4. Kill switch (immediate halt)
    5. Micro-pause injection (human-like hesitation)
    6. Full audit trail
    """

    def __init__(
        self,
        max_aps: float = 15.0,
        min_reaction_ms: float = 90.0,
        session_limit_hours: float = 4.0,
        micro_pause_interval: tuple = (30.0, 60.0),
        audit_log_path: Optional[str] = None,
    ):
        self.max_aps = max_aps
        self.min_reaction_ms = min_reaction_ms
        self.session_limit_hours = session_limit_hours
        self._micro_pause_range = micro_pause_interval

        self._enabled = True
        self._paused = False
        self._killed = False
        self._session_start = time.time()
        self._action_times: deque = deque(maxlen=200)
        self._last_action_time = 0.0
        self._total_actions = 0
        self._throttled_actions = 0
        self._next_micro_pause = self._calc_next_pause()

        self._audit_log: List[InputAuditEntry] = []
        self._audit_path = Path(audit_log_path) if audit_log_path else None
        if self._audit_path:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)

        self._kill_callbacks: List[Callable] = []
        self._pause_callbacks: List[Callable] = []
        self._lock = threading.Lock()

    def check_action(self, action_type: str, details: dict = None) -> bool:
        """Check if an action is allowed. Returns True if permitted.

        This is the GATE. Every input must call this first.
        """
        with self._lock:
            if self._killed:
                return False

            if self._paused:
                return False

            now = time.time()

            # Session limit
            elapsed_hours = (now - self._session_start) / 3600
            if elapsed_hours >= self.session_limit_hours:
                logger.warning(
                    f"Session limit reached ({self.session_limit_hours}h). "
                    "Auto-stopping."
                )
                self.kill("session_limit")
                return False

            # APS rate limiting
            window_start = now - 1.0
            recent = sum(
                1 for t in self._action_times if t > window_start
            )
            if recent >= self.max_aps:
                self._throttled_actions += 1
                self._audit(action_type, details or {}, 0, throttled=True)
                return False

            # Minimum reaction time
            since_last = (now - self._last_action_time) * 1000
            if self._last_action_time > 0 and since_last < self.min_reaction_ms:
                self._throttled_actions += 1
                self._audit(action_type, details or {}, since_last, throttled=True)
                return False

            # Micro-pause injection
            if now >= self._next_micro_pause:
                self._next_micro_pause = self._calc_next_pause()
                # Skip this action — simulate a brief human hesitation
                return False

            # Action permitted
            self._action_times.append(now)
            self._last_action_time = now
            self._total_actions += 1
            self._audit(action_type, details or {}, since_last)
            return True

    def kill(self, reason: str = "manual"):
        """Permanently disable all input. Cannot be undone without restart."""
        self._killed = True
        self._enabled = False
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        for cb in self._kill_callbacks:
            try:
                cb(reason)
            except Exception as e:
                logger.error(f"Kill callback error: {e}")
        self._flush_audit()

    def pause(self):
        """Pause all input. Resumable."""
        self._paused = True
        logger.info("Safety: PAUSED")
        for cb in self._pause_callbacks:
            try:
                cb(True)
            except Exception:
                pass

    def resume(self):
        """Resume from pause."""
        if not self._killed:
            self._paused = False
            logger.info("Safety: RESUMED")
            for cb in self._pause_callbacks:
                try:
                    cb(False)
                except Exception:
                    pass

    def on_kill(self, callback: Callable[[str], None]):
        self._kill_callbacks.append(callback)

    def on_pause(self, callback: Callable[[bool], None]):
        self._pause_callbacks.append(callback)

    @property
    def is_active(self) -> bool:
        return self._enabled and not self._paused and not self._killed

    @property
    def is_killed(self) -> bool:
        return self._killed

    def get_stats(self) -> dict:
        now = time.time()
        elapsed = now - self._session_start
        window = sum(
            1 for t in self._action_times if t > now - 1.0
        )
        return {
            "enabled": self._enabled,
            "paused": self._paused,
            "killed": self._killed,
            "session_minutes": round(elapsed / 60, 1),
            "session_limit_hours": self.session_limit_hours,
            "total_actions": self._total_actions,
            "throttled_actions": self._throttled_actions,
            "current_aps": window,
            "max_aps": self.max_aps,
            "min_reaction_ms": self.min_reaction_ms,
            "audit_entries": len(self._audit_log),
        }

    def _calc_next_pause(self) -> float:
        import random
        lo, hi = self._micro_pause_range
        return time.time() + random.uniform(lo, hi)

    def _audit(
        self, action_type: str, details: dict,
        latency_ms: float, throttled: bool = False,
    ):
        entry = InputAuditEntry(
            timestamp=time.time(),
            action_type=action_type,
            details=details,
            latency_ms=latency_ms,
            was_throttled=throttled,
        )
        self._audit_log.append(entry)
        if len(self._audit_log) >= 1000:
            self._flush_audit()

    def _flush_audit(self):
        if not self._audit_path or not self._audit_log:
            return
        try:
            with open(self._audit_path, "a") as f:
                for entry in self._audit_log:
                    f.write(json.dumps({
                        "t": entry.timestamp,
                        "type": entry.action_type,
                        "d": entry.details,
                        "ms": entry.latency_ms,
                        "throttled": entry.was_throttled,
                    }) + "\n")
            self._audit_log.clear()
        except Exception as e:
            logger.error(f"Audit flush failed: {e}")
