"""Experience Replay — SQLite-backed buffer for learning from gameplay.

Stores state -> action -> reward tuples. Supports random sampling,
prioritized replay (high reward/surprise weighted), context-specific
queries, and temporal decay. Persisted to SQLite for cross-session learning.
"""

from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from loguru import logger


@dataclass
class Experience:
    """A single experience: state -> action -> reward -> next_state."""
    state_hash: str
    game_phase: str  # "buy", "live", "post_round"
    action_taken: str  # "peek_a_main", "hold_angle", "rotate_b"
    action_confidence: float
    reward: float  # +1 (kill), -0.5 (death), +1.5 (round win)
    next_state_hash: str
    context: str  # "cs2_dust2_ct_round5"
    timestamp: float
    metadata: dict = None


class ExperienceReplayBuffer:
    """SQLite-backed experience replay buffer for learning.

    Features:
    1. Random batch sampling for strategy evaluation
    2. Prioritized replay: high reward/surprise weighted
    3. Temporal decay: recent experiences weighted higher
    4. Context-specific queries: "what worked on dust2 CT side?"
    """

    def __init__(
        self,
        db_path: str = "learning/experience_replay.db",
        max_size: int = 100000,
    ):
        self._db_path = db_path
        self._max_size = max_size
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_hash TEXT,
                game_phase TEXT,
                action_taken TEXT,
                action_confidence REAL,
                reward REAL,
                next_state_hash TEXT,
                context TEXT,
                timestamp REAL,
                metadata TEXT,
                priority REAL DEFAULT 1.0
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context "
            "ON experiences(context)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reward "
            "ON experiences(reward)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_priority "
            "ON experiences(priority DESC)"
        )
        self._conn.commit()

    def add(self, exp: Experience):
        """Add an experience to the buffer."""
        self._conn.execute(
            "INSERT INTO experiences "
            "(state_hash, game_phase, action_taken, action_confidence, "
            "reward, next_state_hash, context, timestamp, metadata, priority) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exp.state_hash, exp.game_phase, exp.action_taken,
                exp.action_confidence, exp.reward, exp.next_state_hash,
                exp.context, exp.timestamp,
                json.dumps(exp.metadata) if exp.metadata else None,
                abs(exp.reward) + 1,
            ),
        )
        self._conn.commit()
        self._prune()

    def sample_random(self, n: int = 32) -> List[Experience]:
        """Sample N random experiences."""
        rows = self._conn.execute(
            "SELECT * FROM experiences ORDER BY RANDOM() LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_exp(r) for r in rows]

    def sample_prioritized(self, n: int = 32) -> List[Experience]:
        """Sample N experiences weighted by priority."""
        rows = self._conn.execute(
            "SELECT * FROM experiences "
            "ORDER BY priority * RANDOM() DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [self._row_to_exp(r) for r in rows]

    def query_context(
        self, context: str, limit: int = 100,
    ) -> List[Experience]:
        """Get experiences for a specific context."""
        rows = self._conn.execute(
            "SELECT * FROM experiences WHERE context = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (context, limit),
        ).fetchall()
        return [self._row_to_exp(r) for r in rows]

    def best_actions(
        self, context: str, game_phase: str, top_n: int = 5,
    ) -> List[dict]:
        """Find the best-performing actions for a context and phase."""
        rows = self._conn.execute(
            "SELECT action_taken, AVG(reward) as avg_reward, "
            "COUNT(*) as count "
            "FROM experiences WHERE context LIKE ? AND game_phase = ? "
            "GROUP BY action_taken HAVING count >= 3 "
            "ORDER BY avg_reward DESC LIMIT ?",
            (f"%{context}%", game_phase, top_n),
        ).fetchall()
        return [
            {"action": r[0], "avg_reward": round(r[1], 3), "count": r[2]}
            for r in rows
        ]

    @property
    def size(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()[0]

    def close(self):
        """Close the database connection."""
        self._conn.close()

    def _prune(self):
        current_size = self.size
        if current_size > self._max_size:
            excess = current_size - self._max_size
            self._conn.execute(
                "DELETE FROM experiences WHERE id IN "
                "(SELECT id FROM experiences "
                "ORDER BY priority ASC, timestamp ASC "
                f"LIMIT {excess})"
            )
            self._conn.commit()

    def _row_to_exp(self, row) -> Experience:
        return Experience(
            state_hash=row[1], game_phase=row[2],
            action_taken=row[3], action_confidence=row[4],
            reward=row[5], next_state_hash=row[6],
            context=row[7], timestamp=row[8],
            metadata=json.loads(row[9]) if row[9] else None,
        )
