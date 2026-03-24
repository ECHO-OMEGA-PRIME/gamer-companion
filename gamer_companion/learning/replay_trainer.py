"""Replay Trainer — Self-improve from own gameplay replays.

Analyzes completed matches to identify mistakes, missed opportunities,
and optimal plays. Feeds findings back into the learning system.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from loguru import logger


@dataclass
class ReplayMoment:
    """A significant moment in a replay for learning."""
    timestamp: float
    event_type: str           # "death", "kill", "whiff", "good_trade", "bad_position"
    description: str
    game_state: dict = field(default_factory=dict)
    action_taken: str = ""
    optimal_action: str = ""
    reward: float = 0.0       # Positive = good, negative = mistake
    tags: List[str] = field(default_factory=list)


@dataclass
class TrainingLesson:
    """A lesson extracted from replay analysis."""
    lesson_id: str
    category: str             # "aim", "positioning", "timing", "economy", "utility"
    situation: str            # Description of the game situation
    mistake: str              # What went wrong
    correction: str           # What should have been done
    frequency: int = 1        # How often this mistake occurs
    severity: float = 0.5     # 0=minor, 1=game-losing
    drill_suggestion: str = ""  # Suggested practice drill


class ReplayTrainer:
    """Analyze own replays to extract improvement lessons.

    Process:
    1. Parse replay events (from ReplayParser)
    2. Identify negative moments (deaths, missed shots, bad rotations)
    3. Analyze what led to each negative moment
    4. Generate correction lessons
    5. Feed lessons into practice recommendations
    """

    def __init__(self):
        self._moments: List[ReplayMoment] = []
        self._lessons: Dict[str, TrainingLesson] = {}
        self._analysis_history: List[dict] = []

    def add_moment(self, moment: ReplayMoment):
        """Add a replay moment for analysis."""
        self._moments.append(moment)

    def add_moments(self, moments: List[ReplayMoment]):
        """Add multiple replay moments."""
        self._moments.extend(moments)

    def analyze(self) -> List[TrainingLesson]:
        """Analyze all moments and extract lessons."""
        if not self._moments:
            return []

        lessons = []

        # Group moments by event type
        by_type: Dict[str, List[ReplayMoment]] = {}
        for m in self._moments:
            if m.event_type not in by_type:
                by_type[m.event_type] = []
            by_type[m.event_type].append(m)

        # Deaths → positioning/aim lessons
        deaths = by_type.get("death", [])
        if deaths:
            lesson = self._analyze_deaths(deaths)
            if lesson:
                lessons.append(lesson)

        # Whiffs → aim lessons
        whiffs = by_type.get("whiff", [])
        if whiffs:
            lesson = self._analyze_whiffs(whiffs)
            if lesson:
                lessons.append(lesson)

        # Bad positions → positioning lessons
        bad_pos = by_type.get("bad_position", [])
        if bad_pos:
            lesson = self._analyze_positioning(bad_pos)
            if lesson:
                lessons.append(lesson)

        # Store lessons
        for lesson in lessons:
            existing = self._lessons.get(lesson.lesson_id)
            if existing:
                existing.frequency += lesson.frequency
                existing.severity = max(existing.severity, lesson.severity)
            else:
                self._lessons[lesson.lesson_id] = lesson

        self._analysis_history.append({
            "timestamp": time.time(),
            "moments_analyzed": len(self._moments),
            "lessons_generated": len(lessons),
        })

        return lessons

    def _analyze_deaths(self, deaths: List[ReplayMoment]) -> Optional[TrainingLesson]:
        """Extract lessons from death events."""
        if not deaths:
            return None

        # Find patterns in deaths
        avg_reward = sum(d.reward for d in deaths) / len(deaths)

        # Common death causes
        causes: Dict[str, int] = {}
        for d in deaths:
            for tag in d.tags:
                causes[tag] = causes.get(tag, 0) + 1

        top_cause = max(causes, key=causes.get) if causes else "unknown"

        return TrainingLesson(
            lesson_id=f"death_pattern_{top_cause}",
            category="positioning" if "position" in top_cause else "aim",
            situation=f"Died {len(deaths)} times, primary cause: {top_cause}",
            mistake=f"Repeated {top_cause} deaths ({causes.get(top_cause, 0)} times)",
            correction=self._suggest_correction(top_cause),
            frequency=len(deaths),
            severity=min(1.0, len(deaths) * 0.15),
            drill_suggestion=self._suggest_drill(top_cause),
        )

    def _analyze_whiffs(self, whiffs: List[ReplayMoment]) -> Optional[TrainingLesson]:
        if not whiffs:
            return None

        return TrainingLesson(
            lesson_id="aim_whiff_pattern",
            category="aim",
            situation=f"Missed {len(whiffs)} shots in key moments",
            mistake=f"Whiffed {len(whiffs)} times — likely crosshair placement or spray control",
            correction="Practice crosshair placement at head level, learn spray patterns",
            frequency=len(whiffs),
            severity=min(1.0, len(whiffs) * 0.1),
            drill_suggestion="aim_trainer_flick_heads",
        )

    def _analyze_positioning(self, bad_pos: List[ReplayMoment]) -> Optional[TrainingLesson]:
        if not bad_pos:
            return None

        return TrainingLesson(
            lesson_id="positioning_pattern",
            category="positioning",
            situation=f"Bad positioning {len(bad_pos)} times",
            mistake="Exposed to multiple angles or no cover",
            correction="Hold one angle at a time, keep cover nearby, don't wide peek",
            frequency=len(bad_pos),
            severity=min(1.0, len(bad_pos) * 0.12),
            drill_suggestion="positioning_workshop",
        )

    def _suggest_correction(self, cause: str) -> str:
        corrections = {
            "exposed": "Use cover. Don't stand in the open.",
            "flanked": "Watch minimap. Clear corners before pushing.",
            "out_aimed": "Avoid fair aim duels. Use utility to gain advantage.",
            "rushed": "Slow down. Don't repeek the same angle.",
            "utility": "Learn smoke/flash lineups for common engagements.",
            "economy": "Don't force buy. Save for full buy rounds.",
        }
        return corrections.get(cause, "Review the death and identify what you could have done differently.")

    def _suggest_drill(self, cause: str) -> str:
        drills = {
            "exposed": "cover_usage_workshop",
            "flanked": "minimap_awareness_drill",
            "out_aimed": "aim_trainer_heads",
            "rushed": "patience_timing_drill",
            "utility": "utility_lineup_practice",
            "economy": "economy_management_sim",
        }
        return drills.get(cause, "general_review")

    def get_top_lessons(self, limit: int = 5) -> List[TrainingLesson]:
        """Get the most important lessons sorted by severity * frequency."""
        sorted_lessons = sorted(
            self._lessons.values(),
            key=lambda l: l.severity * l.frequency,
            reverse=True,
        )
        return sorted_lessons[:limit]

    def clear_moments(self):
        """Clear analyzed moments (keep lessons)."""
        self._moments.clear()

    def get_stats(self) -> dict:
        return {
            "pending_moments": len(self._moments),
            "total_lessons": len(self._lessons),
            "analyses_run": len(self._analysis_history),
            "top_category": self._top_category(),
        }

    def _top_category(self) -> str:
        if not self._lessons:
            return "none"
        cats: Dict[str, int] = {}
        for l in self._lessons.values():
            cats[l.category] = cats.get(l.category, 0) + l.frequency
        return max(cats, key=cats.get) if cats else "none"
