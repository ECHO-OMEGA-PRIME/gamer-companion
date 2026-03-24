"""Skill Memory — Persistent storage for learned skills and behaviors.

Unlike experience replay (raw episodes), skill memory stores distilled
knowledge: "In situation X, action Y works Z% of the time." This is
the long-term memory that persists across sessions and games.
"""

from __future__ import annotations
import time
import json
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


@dataclass
class Skill:
    """A learned skill with performance metrics."""
    skill_id: str
    name: str
    category: str             # "aim", "movement", "strategy", "utility", "economy"
    game_id: str              # "cs2", "valorant", "universal"
    description: str = ""
    proficiency: float = 0.0  # 0.0 to 1.0
    practice_count: int = 0
    success_count: int = 0
    last_practiced: float = 0
    parameters: Dict[str, float] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.practice_count)

    @property
    def mastery_level(self) -> str:
        if self.proficiency >= 0.9:
            return "master"
        if self.proficiency >= 0.7:
            return "advanced"
        if self.proficiency >= 0.4:
            return "intermediate"
        if self.proficiency >= 0.1:
            return "beginner"
        return "untrained"


class SkillMemory:
    """Persistent skill storage with proficiency tracking.

    Features:
    - SQLite-backed for cross-session persistence
    - Proficiency decay (skills degrade without practice)
    - Skill tree dependencies
    - Cross-game skill transfer (universal skills)
    - Proficiency history for progress tracking
    """

    def __init__(self, db_path: str = "skill_memory.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                game_id TEXT NOT NULL,
                description TEXT DEFAULT '',
                proficiency REAL DEFAULT 0.0,
                practice_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_practiced REAL DEFAULT 0,
                parameters TEXT DEFAULT '{}',
                prerequisites TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS skill_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                proficiency REAL,
                timestamp REAL,
                event TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_skills_game ON skills(game_id);
            CREATE INDEX IF NOT EXISTS idx_skills_category ON skills(category);
            CREATE INDEX IF NOT EXISTS idx_history_skill ON skill_history(skill_id);
        """)
        self._conn.commit()

    def learn(self, skill: Skill):
        """Store or update a skill."""
        self._conn.execute(
            "INSERT OR REPLACE INTO skills "
            "(skill_id, name, category, game_id, description, proficiency, "
            "practice_count, success_count, last_practiced, parameters, prerequisites, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                skill.skill_id, skill.name, skill.category, skill.game_id,
                skill.description, skill.proficiency, skill.practice_count,
                skill.success_count, skill.last_practiced,
                json.dumps(skill.parameters), json.dumps(skill.prerequisites),
                json.dumps(skill.tags),
            ),
        )
        self._conn.execute(
            "INSERT INTO skill_history (skill_id, proficiency, timestamp, event) "
            "VALUES (?, ?, ?, ?)",
            (skill.skill_id, skill.proficiency, time.time(), "learn"),
        )
        self._conn.commit()

    def practice(self, skill_id: str, success: bool) -> Optional[float]:
        """Record a practice attempt. Returns new proficiency."""
        row = self._conn.execute(
            "SELECT proficiency, practice_count, success_count FROM skills WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()

        if not row:
            return None

        prof, practice, successes = row
        practice += 1
        if success:
            successes += 1
            # Proficiency increases on success, diminishing returns
            prof = min(1.0, prof + (1.0 - prof) * 0.05)
        else:
            # Small decrease on failure
            prof = max(0.0, prof - 0.01)

        self._conn.execute(
            "UPDATE skills SET proficiency = ?, practice_count = ?, "
            "success_count = ?, last_practiced = ? WHERE skill_id = ?",
            (round(prof, 4), practice, successes, time.time(), skill_id),
        )
        self._conn.execute(
            "INSERT INTO skill_history (skill_id, proficiency, timestamp, event) "
            "VALUES (?, ?, ?, ?)",
            (skill_id, prof, time.time(), "success" if success else "failure"),
        )
        self._conn.commit()

        return prof

    def apply_decay(self, decay_rate: float = 0.001, max_age_hours: float = 24):
        """Apply proficiency decay to skills not practiced recently."""
        cutoff = time.time() - (max_age_hours * 3600)
        rows = self._conn.execute(
            "SELECT skill_id, proficiency, last_practiced FROM skills "
            "WHERE last_practiced < ? AND proficiency > 0.01",
            (cutoff,),
        ).fetchall()

        for skill_id, prof, last_prac in rows:
            hours_since = (time.time() - last_prac) / 3600
            new_prof = max(0.0, prof - decay_rate * hours_since)
            self._conn.execute(
                "UPDATE skills SET proficiency = ? WHERE skill_id = ?",
                (round(new_prof, 4), skill_id),
            )

        if rows:
            self._conn.commit()

        return len(rows)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        row = self._conn.execute(
            "SELECT skill_id, name, category, game_id, description, proficiency, "
            "practice_count, success_count, last_practiced, parameters, prerequisites, tags "
            "FROM skills WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()

        if not row:
            return None

        return Skill(
            skill_id=row[0], name=row[1], category=row[2], game_id=row[3],
            description=row[4], proficiency=row[5], practice_count=row[6],
            success_count=row[7], last_practiced=row[8],
            parameters=json.loads(row[9]), prerequisites=json.loads(row[10]),
            tags=json.loads(row[11]),
        )

    def get_skills_for_game(self, game_id: str) -> List[Skill]:
        """Get all skills for a specific game (including universal)."""
        rows = self._conn.execute(
            "SELECT skill_id FROM skills WHERE game_id IN (?, 'universal') "
            "ORDER BY proficiency DESC",
            (game_id,),
        ).fetchall()

        return [self.get_skill(r[0]) for r in rows if r[0]]

    def get_weakest(self, game_id: str = None, limit: int = 5) -> List[Skill]:
        """Get skills with lowest proficiency (practice targets)."""
        if game_id:
            rows = self._conn.execute(
                "SELECT skill_id FROM skills WHERE game_id IN (?, 'universal') "
                "AND practice_count > 0 ORDER BY proficiency ASC LIMIT ?",
                (game_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT skill_id FROM skills WHERE practice_count > 0 "
                "ORDER BY proficiency ASC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self.get_skill(r[0]) for r in rows if r[0]]

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        avg_prof = self._conn.execute("SELECT AVG(proficiency) FROM skills").fetchone()[0]
        mastered = self._conn.execute(
            "SELECT COUNT(*) FROM skills WHERE proficiency >= 0.9"
        ).fetchone()[0]
        return {
            "total_skills": total,
            "avg_proficiency": round(avg_prof or 0, 3),
            "mastered": mastered,
        }

    def close(self):
        self._conn.close()
