"""Frame History — Temporal memory of recent game frames for pattern analysis.

Stores a rolling buffer of FrameSnapshots with structured game state data.
Used by the cognition engine and probability engine to detect patterns,
predict enemy movements, and learn from recent gameplay.
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from loguru import logger


@dataclass
class FrameSnapshot:
    """A single frame's game state snapshot."""
    timestamp: float
    frame_id: str
    game_state: Dict[str, Any] = field(default_factory=dict)
    game_phase: str = "unknown"
    player_hp: Optional[int] = None
    player_armor: Optional[int] = None
    player_pos: Optional[tuple] = None
    enemies_seen: List[dict] = field(default_factory=list)
    allies_alive: int = 0
    enemies_alive: int = 0
    economy: Optional[Dict[str, Any]] = None
    round_time: Optional[float] = None
    events: List[dict] = field(default_factory=list)
    threat_level: str = "low"
    detections: List[dict] = field(default_factory=list)


class FrameHistory:
    """Rolling buffer of recent frame snapshots for temporal analysis.

    Features:
    - Fixed-size circular buffer (default 1800 frames = ~60s at 30fps)
    - Query by time window, game phase, threat level
    - Death location tracking for learning
    - Phase transition detection
    - Trend analysis (health declining, enemies increasing, etc.)
    """

    def __init__(self, max_frames: int = 1800):
        self._buffer: deque[FrameSnapshot] = deque(maxlen=max_frames)
        self._death_locations: deque[dict] = deque(maxlen=50)
        self._phase_transitions: deque[dict] = deque(maxlen=100)
        self._last_phase: str = "unknown"

    def add(self, snapshot: FrameSnapshot):
        """Add a frame snapshot to the buffer."""
        # Detect phase transitions
        if snapshot.game_phase != self._last_phase and self._last_phase != "unknown":
            self._phase_transitions.append({
                "from": self._last_phase,
                "to": snapshot.game_phase,
                "timestamp": snapshot.timestamp,
                "frame_id": snapshot.frame_id,
            })
        self._last_phase = snapshot.game_phase

        # Detect death (HP drops to 0)
        if (snapshot.player_hp is not None and snapshot.player_hp <= 0 and
                self._buffer and self._buffer[-1].player_hp and self._buffer[-1].player_hp > 0):
            self._death_locations.append({
                "timestamp": snapshot.timestamp,
                "position": snapshot.player_pos,
                "game_phase": snapshot.game_phase,
                "threat_level": snapshot.threat_level,
                "enemies_seen": len(snapshot.enemies_seen),
                "hp_before": self._buffer[-1].player_hp,
            })

        self._buffer.append(snapshot)

    def recent(self, seconds: float = 5.0) -> List[FrameSnapshot]:
        """Get frames from the last N seconds."""
        cutoff = time.time() - seconds
        return [s for s in self._buffer if s.timestamp >= cutoff]

    def by_phase(self, phase: str) -> List[FrameSnapshot]:
        """Get all frames in a specific game phase."""
        return [s for s in self._buffer if s.game_phase == phase]

    def by_threat(self, level: str) -> List[FrameSnapshot]:
        """Get all frames at a specific threat level."""
        return [s for s in self._buffer if s.threat_level == level]

    def death_locations(self, count: int = 10) -> List[dict]:
        """Get recent death locations for learning."""
        return list(self._death_locations)[-count:]

    def phase_transitions(self, count: int = 20) -> List[dict]:
        """Get recent game phase transitions."""
        return list(self._phase_transitions)[-count:]

    def health_trend(self, frames: int = 30) -> str:
        """Analyze health trend over recent frames."""
        recent = list(self._buffer)[-frames:]
        hps = [s.player_hp for s in recent if s.player_hp is not None]
        if len(hps) < 5:
            return "unknown"
        first_half = sum(hps[:len(hps) // 2]) / (len(hps) // 2)
        second_half = sum(hps[len(hps) // 2:]) / (len(hps) - len(hps) // 2)
        diff = second_half - first_half
        if diff < -10:
            return "declining"
        elif diff > 10:
            return "recovering"
        return "stable"

    def enemy_count_trend(self, frames: int = 30) -> str:
        """Analyze enemy count trend."""
        recent = list(self._buffer)[-frames:]
        counts = [s.enemies_alive for s in recent]
        if len(counts) < 5:
            return "unknown"
        first_half = sum(counts[:len(counts) // 2]) / (len(counts) // 2)
        second_half = sum(counts[len(counts) // 2:]) / (len(counts) - len(counts) // 2)
        diff = second_half - first_half
        if diff > 0.5:
            return "increasing"
        elif diff < -0.5:
            return "decreasing"
        return "stable"

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def latest(self) -> Optional[FrameSnapshot]:
        return self._buffer[-1] if self._buffer else None
