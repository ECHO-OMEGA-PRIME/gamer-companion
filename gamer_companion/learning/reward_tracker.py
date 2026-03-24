"""Reward Tracker — Track action → outcome mappings with rewards.

The fundamental feedback signal for all learning systems. Maps every
action the AI takes to its eventual outcome, enabling reinforcement
learning style improvement.
"""

from __future__ import annotations
import time
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger


@dataclass
class TrackedAction:
    """An action with its context and eventual reward."""
    action_id: str
    action: str               # "peek_a", "flash_b", "rotate_mid"
    context: str              # Serialized game state
    timestamp: float = field(default_factory=time.time)
    reward: Optional[float] = None  # Assigned after outcome
    outcome: str = ""         # "kill", "death", "trade", "nothing"
    resolved: bool = False


class RewardTracker:
    """Track actions and assign rewards when outcomes are known.

    Design: Actions are logged immediately. Rewards are assigned later
    when the outcome is known (end of engagement, round, or match).
    This handles delayed reward attribution — a common RL challenge.

    Storage: SQLite for persistence.
    """

    def __init__(self, db_path: str = "reward_data.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()
        self._pending: Dict[str, TrackedAction] = {}
        self._action_counter = 0

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS rewards (
                action_id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                context TEXT,
                timestamp REAL NOT NULL,
                reward REAL,
                outcome TEXT,
                resolved INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_rewards_action ON rewards(action);
            CREATE INDEX IF NOT EXISTS idx_rewards_resolved ON rewards(resolved);
        """)
        self._conn.commit()

    def log_action(self, action: str, context: str = "") -> str:
        """Log an action. Returns action_id for later reward assignment."""
        self._action_counter += 1
        action_id = f"act_{int(time.time())}_{self._action_counter}"

        tracked = TrackedAction(
            action_id=action_id,
            action=action,
            context=context,
        )
        self._pending[action_id] = tracked

        self._conn.execute(
            "INSERT INTO rewards (action_id, action, context, timestamp, resolved) "
            "VALUES (?, ?, ?, ?, 0)",
            (action_id, action, context, tracked.timestamp),
        )
        self._conn.commit()

        return action_id

    def assign_reward(self, action_id: str, reward: float, outcome: str = ""):
        """Assign a reward to a previously logged action."""
        tracked = self._pending.pop(action_id, None)

        self._conn.execute(
            "UPDATE rewards SET reward = ?, outcome = ?, resolved = 1 "
            "WHERE action_id = ?",
            (reward, outcome, action_id),
        )
        self._conn.commit()

    def assign_bulk_reward(self, action_ids: List[str], reward: float, outcome: str = ""):
        """Assign the same reward to multiple actions (round-end attribution)."""
        for aid in action_ids:
            self.assign_reward(aid, reward, outcome)

    def get_action_value(self, action: str) -> Tuple[float, int]:
        """Get average reward and count for an action type.

        Returns: (avg_reward, count)
        """
        row = self._conn.execute(
            "SELECT AVG(reward), COUNT(*) FROM rewards "
            "WHERE action = ? AND resolved = 1",
            (action,),
        ).fetchone()

        return (row[0] or 0.0, row[1] or 0)

    def get_action_rankings(self, limit: int = 20) -> List[Dict]:
        """Get actions ranked by average reward."""
        rows = self._conn.execute(
            "SELECT action, AVG(reward) as avg_r, COUNT(*) as cnt "
            "FROM rewards WHERE resolved = 1 "
            "GROUP BY action HAVING cnt >= 2 "
            "ORDER BY avg_r DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [
            {"action": r[0], "avg_reward": round(r[1], 3), "count": r[2]}
            for r in rows
        ]

    def get_context_rewards(self, context_pattern: str) -> List[Dict]:
        """Get rewards for actions in a specific context."""
        rows = self._conn.execute(
            "SELECT action, reward, outcome FROM rewards "
            "WHERE context LIKE ? AND resolved = 1 "
            "ORDER BY reward DESC LIMIT 20",
            (f"%{context_pattern}%",),
        ).fetchall()

        return [
            {"action": r[0], "reward": r[1], "outcome": r[2]}
            for r in rows
        ]

    def get_pending_count(self) -> int:
        """Get number of actions awaiting reward assignment."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM rewards WHERE resolved = 0"
        ).fetchone()
        return row[0]

    def expire_pending(self, max_age_seconds: float = 300):
        """Expire old pending actions with neutral reward."""
        cutoff = time.time() - max_age_seconds
        self._conn.execute(
            "UPDATE rewards SET reward = 0.0, outcome = 'expired', resolved = 1 "
            "WHERE resolved = 0 AND timestamp < ?",
            (cutoff,),
        )
        self._conn.commit()

        # Clean in-memory pending
        expired = [
            aid for aid, a in self._pending.items()
            if a.timestamp < cutoff
        ]
        for aid in expired:
            self._pending.pop(aid, None)

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM rewards").fetchone()[0]
        resolved = self._conn.execute("SELECT COUNT(*) FROM rewards WHERE resolved = 1").fetchone()[0]
        avg_reward = self._conn.execute("SELECT AVG(reward) FROM rewards WHERE resolved = 1").fetchone()[0]
        return {
            "total_actions": total,
            "resolved": resolved,
            "pending": total - resolved,
            "avg_reward": round(avg_reward or 0, 3),
        }

    def close(self):
        self._conn.close()
