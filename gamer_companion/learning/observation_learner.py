"""Observation Learner — Learn from watching human gameplay.

Records human actions during OBSERVE mode and extracts patterns:
- What actions do humans take in specific situations?
- What's the timing between perception and action?
- What's the mouse movement style (speed, path shape)?
- What strategies do they use at different game phases?
"""

from __future__ import annotations
import time
import json
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from loguru import logger


@dataclass
class ObservedAction:
    """A single observed human action with context."""
    timestamp: float
    action_type: str          # "mouse_move", "click", "key_press", "key_release"
    action_detail: str        # "left_click", "W", "mouse1"
    game_state: str           # Serialized game state snapshot
    screen_region: str        # Which region triggered the action ("enemy_spotted", "minimap", "hud")
    reaction_time_ms: float   # Time from stimulus to action
    mouse_path: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 1.0   # How sure we are this was intentional


@dataclass
class LearnedBehavior:
    """A pattern extracted from observations."""
    behavior_id: str
    trigger: str              # "enemy_visible", "low_health", "bomb_planted"
    action_sequence: List[str]  # ["aim_at_head", "click", "strafe_left"]
    avg_reaction_ms: float
    success_rate: float
    sample_count: int
    context: str = ""         # "close_range", "long_range", "eco_round"


class ObservationLearner:
    """Learn gameplay patterns from human observation.

    Modes:
    1. RECORD: Capture human inputs + game state pairs
    2. ANALYZE: Extract patterns from recorded sessions
    3. APPLY: Use learned patterns to inform AI decisions

    Storage: SQLite for durability across sessions.
    """

    def __init__(self, db_path: str = "observation_data.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()
        self._recording = False
        self._session_id: Optional[str] = None
        self._buffer: List[ObservedAction] = []
        self._flush_threshold = 100

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                action_type TEXT NOT NULL,
                action_detail TEXT NOT NULL,
                game_state TEXT,
                screen_region TEXT,
                reaction_time_ms REAL,
                mouse_path TEXT,
                confidence REAL DEFAULT 1.0
            );
            CREATE TABLE IF NOT EXISTS learned_behaviors (
                behavior_id TEXT PRIMARY KEY,
                trigger TEXT NOT NULL,
                action_sequence TEXT NOT NULL,
                avg_reaction_ms REAL,
                success_rate REAL,
                sample_count INTEGER DEFAULT 0,
                context TEXT,
                updated_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_id);
            CREATE INDEX IF NOT EXISTS idx_obs_action ON observations(action_type);
            CREATE INDEX IF NOT EXISTS idx_behavior_trigger ON learned_behaviors(trigger);
        """)
        self._conn.commit()

    def start_recording(self, session_id: str = None):
        """Start recording human gameplay."""
        self._session_id = session_id or f"obs_{int(time.time())}"
        self._recording = True
        self._buffer.clear()
        logger.info(f"Observation recording started: {self._session_id}")

    def stop_recording(self) -> int:
        """Stop recording and flush buffer. Returns total observations."""
        self._recording = False
        self._flush_buffer()
        count = self._conn.execute(
            "SELECT COUNT(*) FROM observations WHERE session_id = ?",
            (self._session_id,),
        ).fetchone()[0]
        logger.info(f"Recording stopped: {count} observations in session {self._session_id}")
        return count

    def record(self, action: ObservedAction):
        """Record a single observed action."""
        if not self._recording:
            return
        self._buffer.append(action)
        if len(self._buffer) >= self._flush_threshold:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer:
            return
        rows = [
            (
                self._session_id,
                a.timestamp,
                a.action_type,
                a.action_detail,
                a.game_state,
                a.screen_region,
                a.reaction_time_ms,
                json.dumps(a.mouse_path),
                a.confidence,
            )
            for a in self._buffer
        ]
        self._conn.executemany(
            "INSERT INTO observations (session_id, timestamp, action_type, action_detail, "
            "game_state, screen_region, reaction_time_ms, mouse_path, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._buffer.clear()

    def analyze_session(self, session_id: str = None) -> List[LearnedBehavior]:
        """Analyze a recorded session to extract behavioral patterns."""
        sid = session_id or self._session_id
        if not sid:
            return []

        rows = self._conn.execute(
            "SELECT action_type, action_detail, game_state, screen_region, reaction_time_ms "
            "FROM observations WHERE session_id = ? ORDER BY timestamp",
            (sid,),
        ).fetchall()

        if not rows:
            return []

        # Group by screen_region (trigger context)
        region_actions: Dict[str, list] = {}
        reaction_times: Dict[str, list] = {}

        for action_type, action_detail, game_state, region, rt_ms in rows:
            region = region or "unknown"
            if region not in region_actions:
                region_actions[region] = []
                reaction_times[region] = []
            region_actions[region].append(f"{action_type}:{action_detail}")
            if rt_ms and rt_ms > 0:
                reaction_times[region].append(rt_ms)

        behaviors = []
        for region, actions in region_actions.items():
            # Find most common action sequences (bigrams)
            if len(actions) < 2:
                continue

            bigrams: Dict[str, int] = {}
            for i in range(len(actions) - 1):
                bg = f"{actions[i]}|{actions[i+1]}"
                bigrams[bg] = bigrams.get(bg, 0) + 1

            top_bigram = max(bigrams, key=bigrams.get)
            seq = top_bigram.split("|")

            rts = reaction_times.get(region, [])
            avg_rt = sum(rts) / len(rts) if rts else 200.0

            behavior = LearnedBehavior(
                behavior_id=f"obs_{region}_{int(time.time())}",
                trigger=region,
                action_sequence=seq,
                avg_reaction_ms=round(avg_rt, 1),
                success_rate=0.5,  # Unknown until outcome data
                sample_count=bigrams[top_bigram],
                context=sid,
            )
            behaviors.append(behavior)

            # Store in DB
            self._conn.execute(
                "INSERT OR REPLACE INTO learned_behaviors "
                "(behavior_id, trigger, action_sequence, avg_reaction_ms, "
                "success_rate, sample_count, context, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    behavior.behavior_id,
                    behavior.trigger,
                    json.dumps(behavior.action_sequence),
                    behavior.avg_reaction_ms,
                    behavior.success_rate,
                    behavior.sample_count,
                    behavior.context,
                    time.time(),
                ),
            )
        self._conn.commit()

        return behaviors

    def get_behavior_for(self, trigger: str) -> Optional[LearnedBehavior]:
        """Get the best learned behavior for a trigger."""
        row = self._conn.execute(
            "SELECT behavior_id, trigger, action_sequence, avg_reaction_ms, "
            "success_rate, sample_count, context FROM learned_behaviors "
            "WHERE trigger = ? ORDER BY sample_count DESC LIMIT 1",
            (trigger,),
        ).fetchone()

        if not row:
            return None

        return LearnedBehavior(
            behavior_id=row[0],
            trigger=row[1],
            action_sequence=json.loads(row[2]),
            avg_reaction_ms=row[3],
            success_rate=row[4],
            sample_count=row[5],
            context=row[6] or "",
        )

    def get_stats(self) -> dict:
        total_obs = self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        total_behaviors = self._conn.execute("SELECT COUNT(*) FROM learned_behaviors").fetchone()[0]
        sessions = self._conn.execute("SELECT COUNT(DISTINCT session_id) FROM observations").fetchone()[0]
        return {
            "total_observations": total_obs,
            "total_behaviors": total_behaviors,
            "sessions_recorded": sessions,
            "recording": self._recording,
        }

    def close(self):
        self._flush_buffer()
        self._conn.close()
