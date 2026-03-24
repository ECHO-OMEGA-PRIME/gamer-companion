"""Input Recorder — Record human input for learning and replay.

Captures all mouse/keyboard input during a session, timestamps it,
and stores it for later analysis or replay. Used by ObservationLearner
to learn from human gameplay.
"""

from __future__ import annotations
import time
import json
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger


@dataclass
class InputEvent:
    """A single input event (mouse or keyboard)."""
    timestamp: float
    event_type: str           # "mouse_move", "mouse_click", "key_down", "key_up", "scroll"
    x: int = 0                # Mouse position
    y: int = 0
    key: str = ""             # Key name for keyboard events
    button: str = ""          # "left", "right", "middle" for mouse
    scroll_delta: int = 0     # Scroll amount
    modifiers: List[str] = field(default_factory=list)  # ["shift", "ctrl", "alt"]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "x": self.x, "y": self.y,
            "key": self.key,
            "button": self.button,
            "scroll_delta": self.scroll_delta,
            "modifiers": self.modifiers,
        }


@dataclass
class RecordingSession:
    """A recorded input session."""
    session_id: str
    game_id: str
    start_time: float
    end_time: float = 0
    event_count: int = 0
    duration_s: float = 0
    notes: str = ""


class InputRecorder:
    """Record and replay human input events.

    Modes:
    1. RECORD: Capture all input events with timestamps
    2. REPLAY: Play back recorded events with timing
    3. ANALYZE: Compute input statistics (APM, movement patterns, etc.)

    Storage: SQLite for durability, with session-based organization.
    """

    def __init__(self, db_path: str = "input_recordings.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()
        self._recording = False
        self._session_id: Optional[str] = None
        self._game_id: str = ""
        self._start_time: float = 0
        self._buffer: List[InputEvent] = []
        self._flush_threshold = 200
        self._event_count = 0

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                game_id TEXT,
                start_time REAL,
                end_time REAL,
                event_count INTEGER DEFAULT 0,
                notes TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                x INTEGER DEFAULT 0,
                y INTEGER DEFAULT 0,
                key TEXT DEFAULT '',
                button TEXT DEFAULT '',
                scroll_delta INTEGER DEFAULT 0,
                modifiers TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        """)
        self._conn.commit()

    def start_recording(self, game_id: str = "", session_id: str = None) -> str:
        """Start recording input events."""
        self._session_id = session_id or f"rec_{int(time.time())}"
        self._game_id = game_id
        self._start_time = time.time()
        self._recording = True
        self._buffer.clear()
        self._event_count = 0

        self._conn.execute(
            "INSERT INTO sessions (session_id, game_id, start_time) VALUES (?, ?, ?)",
            (self._session_id, game_id, self._start_time),
        )
        self._conn.commit()

        logger.info(f"Input recording started: {self._session_id}")
        return self._session_id

    def stop_recording(self) -> RecordingSession:
        """Stop recording and return session summary."""
        self._recording = False
        self._flush_buffer()

        end_time = time.time()
        duration = end_time - self._start_time

        self._conn.execute(
            "UPDATE sessions SET end_time = ?, event_count = ? WHERE session_id = ?",
            (end_time, self._event_count, self._session_id),
        )
        self._conn.commit()

        session = RecordingSession(
            session_id=self._session_id,
            game_id=self._game_id,
            start_time=self._start_time,
            end_time=end_time,
            event_count=self._event_count,
            duration_s=round(duration, 1),
        )

        logger.info(f"Recording stopped: {self._event_count} events in {duration:.1f}s")
        return session

    def record_event(self, event: InputEvent):
        """Record a single input event."""
        if not self._recording:
            return
        self._buffer.append(event)
        self._event_count += 1
        if len(self._buffer) >= self._flush_threshold:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer:
            return
        rows = [
            (
                self._session_id,
                e.timestamp,
                e.event_type,
                e.x, e.y,
                e.key, e.button,
                e.scroll_delta,
                json.dumps(e.modifiers),
            )
            for e in self._buffer
        ]
        self._conn.executemany(
            "INSERT INTO events (session_id, timestamp, event_type, x, y, "
            "key, button, scroll_delta, modifiers) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._buffer.clear()

    def get_session_events(
        self, session_id: str, event_type: str = None, limit: int = 1000,
    ) -> List[InputEvent]:
        """Get events from a session."""
        if event_type:
            rows = self._conn.execute(
                "SELECT timestamp, event_type, x, y, key, button, scroll_delta, modifiers "
                "FROM events WHERE session_id = ? AND event_type = ? "
                "ORDER BY timestamp LIMIT ?",
                (session_id, event_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT timestamp, event_type, x, y, key, button, scroll_delta, modifiers "
                "FROM events WHERE session_id = ? "
                "ORDER BY timestamp LIMIT ?",
                (session_id, limit),
            ).fetchall()

        return [
            InputEvent(
                timestamp=r[0], event_type=r[1],
                x=r[2], y=r[3],
                key=r[4], button=r[5],
                scroll_delta=r[6],
                modifiers=json.loads(r[7]),
            )
            for r in rows
        ]

    def get_apm(self, session_id: str) -> float:
        """Calculate actions per minute for a session."""
        session = self._conn.execute(
            "SELECT start_time, end_time, event_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not session or session[1] <= session[0]:
            return 0.0
        duration_min = (session[1] - session[0]) / 60.0
        return round(session[2] / max(0.01, duration_min), 1)

    def get_movement_heatmap(
        self, session_id: str, grid_size: int = 20,
    ) -> Dict[Tuple[int, int], int]:
        """Get mouse movement heatmap (grid cell → count)."""
        rows = self._conn.execute(
            "SELECT x, y FROM events WHERE session_id = ? AND event_type = 'mouse_move'",
            (session_id,),
        ).fetchall()

        heatmap: Dict[Tuple[int, int], int] = {}
        for x, y in rows:
            gx = x // grid_size
            gy = y // grid_size
            key = (gx, gy)
            heatmap[key] = heatmap.get(key, 0) + 1

        return heatmap

    def get_key_frequency(self, session_id: str) -> Dict[str, int]:
        """Get key press frequency."""
        rows = self._conn.execute(
            "SELECT key, COUNT(*) FROM events "
            "WHERE session_id = ? AND event_type = 'key_down' AND key != '' "
            "GROUP BY key ORDER BY COUNT(*) DESC",
            (session_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def list_sessions(self, game_id: str = None) -> List[RecordingSession]:
        """List recording sessions."""
        if game_id:
            rows = self._conn.execute(
                "SELECT session_id, game_id, start_time, end_time, event_count, notes "
                "FROM sessions WHERE game_id = ? ORDER BY start_time DESC",
                (game_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT session_id, game_id, start_time, end_time, event_count, notes "
                "FROM sessions ORDER BY start_time DESC",
            ).fetchall()

        return [
            RecordingSession(
                session_id=r[0], game_id=r[1],
                start_time=r[2], end_time=r[3] or 0,
                event_count=r[4],
                duration_s=round((r[3] or r[2]) - r[2], 1),
                notes=r[5] or "",
            )
            for r in rows
        ]

    def get_stats(self) -> dict:
        total_sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_events = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "total_sessions": total_sessions,
            "total_events": total_events,
            "recording": self._recording,
            "current_session": self._session_id if self._recording else None,
            "buffer_size": len(self._buffer),
        }

    def close(self):
        if self._recording:
            self.stop_recording()
        self._conn.close()
