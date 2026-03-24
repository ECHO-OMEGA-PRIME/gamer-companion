"""Movement Trainer — Practice movement mechanics.

Generates movement challenges for different games:
- Strafe patterns (counter-strafe for FPS)
- Bunny hop timing
- Peek timing (jiggle peek, wide peek, shoulder peek)
- Rotation speed
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class MovementChallenge:
    """A movement training challenge."""
    name: str
    description: str
    drill_type: str           # "strafe", "peek", "bunny_hop", "rotation", "positioning"
    target_time_ms: float     # Target completion time
    steps: List[str]          # Ordered key/movement instructions
    success_criteria: str     # What counts as success
    difficulty: float = 0.5   # 0=easy, 1=pro


@dataclass
class MovementResult:
    """Result of a movement drill."""
    challenge_name: str
    completion_time_ms: float
    target_time_ms: float
    accuracy: float           # 0-1 how precisely the movement was executed
    score: float              # 0-100 overall score
    timestamp: float = field(default_factory=time.time)

    @property
    def time_ratio(self) -> float:
        return self.completion_time_ms / max(1, self.target_time_ms)


# Pre-built movement challenges
CHALLENGES: Dict[str, MovementChallenge] = {
    "counter_strafe": MovementChallenge(
        name="counter_strafe",
        description="Strafe left, counter-strafe, shoot. Core FPS mechanic.",
        drill_type="strafe",
        target_time_ms=300,
        steps=["press_a", "release_a", "press_d", "release_d", "shoot"],
        success_criteria="Velocity < 5 units at time of shot",
        difficulty=0.4,
    ),
    "jiggle_peek": MovementChallenge(
        name="jiggle_peek",
        description="Quick peek around corner to gather info without committing.",
        drill_type="peek",
        target_time_ms=200,
        steps=["press_d", "release_d", "press_a", "release_a"],
        success_criteria="Expose < 200ms, return to cover",
        difficulty=0.5,
    ),
    "wide_peek": MovementChallenge(
        name="wide_peek",
        description="Wide swing to clear an angle. High risk, high info.",
        drill_type="peek",
        target_time_ms=400,
        steps=["press_d", "counter_strafe", "aim", "shoot"],
        success_criteria="Clear the angle and land a shot",
        difficulty=0.6,
    ),
    "bunny_hop_chain": MovementChallenge(
        name="bunny_hop_chain",
        description="Chain 5 bunny hops without losing speed.",
        drill_type="bunny_hop",
        target_time_ms=2500,
        steps=["jump", "strafe_air", "jump_on_land"] * 5,
        success_criteria="Maintain >250 velocity for 5 hops",
        difficulty=0.8,
    ),
    "shoulder_peek": MovementChallenge(
        name="shoulder_peek",
        description="Expose shoulder to bait a shot, then counter-peek.",
        drill_type="peek",
        target_time_ms=500,
        steps=["press_d", "release_d", "wait_shot", "press_d", "aim", "shoot"],
        success_criteria="Bait enemy shot, then trade",
        difficulty=0.7,
    ),
    "rotation_speed": MovementChallenge(
        name="rotation_speed",
        description="Rotate from A to B site as fast as possible.",
        drill_type="rotation",
        target_time_ms=5000,
        steps=["sprint", "navigate_obstacles", "check_angles", "arrive"],
        success_criteria="Arrive at B with full HP",
        difficulty=0.3,
    ),
}


class MovementTrainer:
    """Generate and evaluate movement training drills."""

    def __init__(self):
        self._challenges = {**CHALLENGES}
        self._results: List[MovementResult] = []

    def get_challenge(self, name: str) -> Optional[MovementChallenge]:
        return self._challenges.get(name)

    def record_result(
        self,
        challenge_name: str,
        completion_time_ms: float,
        accuracy: float = 1.0,
    ) -> MovementResult:
        """Record the result of a movement drill."""
        challenge = self._challenges.get(challenge_name)
        target = challenge.target_time_ms if challenge else 1000

        # Score: time accuracy (50%) + execution accuracy (50%)
        time_score = max(0, 50 * (1 - abs(completion_time_ms - target) / target))
        acc_score = accuracy * 50
        score = round(time_score + acc_score, 1)

        result = MovementResult(
            challenge_name=challenge_name,
            completion_time_ms=round(completion_time_ms, 1),
            target_time_ms=target,
            accuracy=round(accuracy, 3),
            score=score,
        )

        self._results.append(result)
        return result

    def get_weakness(self) -> str:
        """Find the weakest movement type."""
        if not self._results:
            return "no_data"

        type_scores: Dict[str, List[float]] = {}
        for r in self._results:
            ch = self._challenges.get(r.challenge_name)
            if ch:
                dtype = ch.drill_type
                if dtype not in type_scores:
                    type_scores[dtype] = []
                type_scores[dtype].append(r.score)

        if not type_scores:
            return "no_data"

        avg = {k: sum(v) / len(v) for k, v in type_scores.items()}
        return min(avg, key=avg.get)

    def list_challenges(self) -> List[dict]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "type": c.drill_type,
                "difficulty": c.difficulty,
            }
            for c in self._challenges.values()
        ]

    def get_stats(self) -> dict:
        if not self._results:
            return {"drills_completed": 0}
        return {
            "drills_completed": len(self._results),
            "avg_score": round(sum(r.score for r in self._results) / len(self._results), 1),
            "weakness": self.get_weakness(),
        }
