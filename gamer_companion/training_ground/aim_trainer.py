"""Aim Trainer — Self-practice aim routines.

Generates aim training scenarios the AI can run to improve
its aim parameters. Works with any aim trainer application
(Aim Lab, Kovaak's) or custom in-game scenarios.
"""

from __future__ import annotations
import random
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger


@dataclass
class AimTarget:
    """A target in an aim training scenario."""
    x: float
    y: float
    radius: float = 20.0
    lifetime_ms: float = 2000  # Time before target expires
    spawn_time: float = 0
    hit: bool = False
    hit_time: float = 0


@dataclass
class AimDrillResult:
    """Results from an aim training drill."""
    drill_name: str
    targets_total: int
    targets_hit: int
    accuracy: float
    avg_reaction_ms: float
    avg_ttk_ms: float         # Time to kill
    best_reaction_ms: float
    worst_reaction_ms: float
    score: float              # Normalized 0-100 score
    timestamp: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        return self.targets_hit / max(1, self.targets_total)


@dataclass
class AimDrill:
    """Configuration for an aim training drill."""
    name: str
    description: str
    target_count: int = 30
    target_radius: float = 20.0
    target_lifetime_ms: float = 2000.0
    spawn_area: Tuple[float, float, float, float] = (200, 100, 1720, 980)  # x1, y1, x2, y2
    moving_targets: bool = False
    target_speed: float = 0.0
    drill_type: str = "flick"  # "flick", "tracking", "switching", "micro"


# Pre-built drill configurations
DRILLS: Dict[str, AimDrill] = {
    "flick_easy": AimDrill(
        name="flick_easy",
        description="Large targets, generous time. Warm-up.",
        target_count=20, target_radius=30, target_lifetime_ms=3000,
        drill_type="flick",
    ),
    "flick_hard": AimDrill(
        name="flick_hard",
        description="Small targets, fast expiry. Competition-grade.",
        target_count=40, target_radius=12, target_lifetime_ms=1200,
        drill_type="flick",
    ),
    "tracking": AimDrill(
        name="tracking",
        description="Follow moving targets smoothly.",
        target_count=15, target_radius=25, target_lifetime_ms=4000,
        moving_targets=True, target_speed=200,
        drill_type="tracking",
    ),
    "micro_adjust": AimDrill(
        name="micro_adjust",
        description="Tiny targets close to crosshair. Precision practice.",
        target_count=30, target_radius=8, target_lifetime_ms=2500,
        spawn_area=(800, 400, 1120, 680),  # Near center
        drill_type="micro",
    ),
    "target_switch": AimDrill(
        name="target_switch",
        description="Two targets at once — switch between them.",
        target_count=20, target_radius=18, target_lifetime_ms=1800,
        drill_type="switching",
    ),
    "headshot_only": AimDrill(
        name="headshot_only",
        description="Head-sized targets only. Crosshair placement drill.",
        target_count=30, target_radius=10, target_lifetime_ms=2000,
        spawn_area=(300, 200, 1620, 600),  # Upper half (head level)
        drill_type="flick",
    ),
}


class AimTrainer:
    """Generate and evaluate aim training scenarios.

    This doesn't control the actual aim trainer application — it generates
    target sequences and evaluates performance, which can be used to:
    1. Benchmark the AI's current aim ability
    2. Identify weaknesses (flick vs tracking vs micro)
    3. Generate practice recommendations
    4. Track improvement over time
    """

    def __init__(self):
        self._drills = {**DRILLS}
        self._results: List[AimDrillResult] = []
        self._screen_center = (960, 540)

    def generate_targets(self, drill_name: str) -> List[AimTarget]:
        """Generate a sequence of targets for a drill."""
        drill = self._drills.get(drill_name)
        if not drill:
            logger.warning(f"Unknown drill: {drill_name}")
            return []

        targets = []
        x1, y1, x2, y2 = drill.spawn_area

        for i in range(drill.target_count):
            x = random.uniform(x1, x2)
            y = random.uniform(y1, y2)

            target = AimTarget(
                x=round(x, 1),
                y=round(y, 1),
                radius=drill.target_radius,
                lifetime_ms=drill.target_lifetime_ms,
            )
            targets.append(target)

        return targets

    def evaluate(
        self,
        drill_name: str,
        targets: List[AimTarget],
        reaction_times_ms: List[float],
    ) -> AimDrillResult:
        """Evaluate performance on a completed drill."""
        hits = sum(1 for t in targets if t.hit)
        total = len(targets)
        accuracy = hits / max(1, total)

        valid_rts = [rt for rt in reaction_times_ms if rt > 0]
        avg_rt = sum(valid_rts) / max(1, len(valid_rts))
        best_rt = min(valid_rts) if valid_rts else 0
        worst_rt = max(valid_rts) if valid_rts else 0

        # Score calculation (0-100)
        # Weight: 50% accuracy, 30% speed, 20% consistency
        acc_score = accuracy * 50
        speed_score = max(0, (1.0 - avg_rt / 1000)) * 30  # Sub-1s = good
        consistency = 1.0 - (worst_rt - best_rt) / max(1, worst_rt) if valid_rts else 0
        consist_score = consistency * 20

        score = round(acc_score + speed_score + consist_score, 1)

        result = AimDrillResult(
            drill_name=drill_name,
            targets_total=total,
            targets_hit=hits,
            accuracy=round(accuracy, 3),
            avg_reaction_ms=round(avg_rt, 1),
            avg_ttk_ms=round(avg_rt * 1.2, 1),  # Approximate
            best_reaction_ms=round(best_rt, 1),
            worst_reaction_ms=round(worst_rt, 1),
            score=score,
        )

        self._results.append(result)
        return result

    def get_weakness(self) -> str:
        """Identify the weakest aim area from drill results."""
        if not self._results:
            return "no_data"

        type_scores: Dict[str, List[float]] = {}
        for r in self._results:
            drill = self._drills.get(r.drill_name)
            if drill:
                dtype = drill.drill_type
                if dtype not in type_scores:
                    type_scores[dtype] = []
                type_scores[dtype].append(r.score)

        if not type_scores:
            return "no_data"

        avg_scores = {k: sum(v) / len(v) for k, v in type_scores.items()}
        return min(avg_scores, key=avg_scores.get)

    def get_recommendation(self) -> str:
        """Get a practice recommendation based on results."""
        weakness = self.get_weakness()
        recs = {
            "flick": "Practice flick_hard drill — focus on fast target acquisition",
            "tracking": "Practice tracking drill — keep crosshair on moving targets",
            "micro": "Practice micro_adjust drill — improve fine aim control",
            "switching": "Practice target_switch drill — faster target transitions",
            "no_data": "Run all drills to establish a baseline",
        }
        return recs.get(weakness, "Continue varied practice across all drill types")

    def list_drills(self) -> List[dict]:
        """List available drills."""
        return [
            {"name": d.name, "description": d.description, "type": d.drill_type}
            for d in self._drills.values()
        ]

    def get_stats(self) -> dict:
        if not self._results:
            return {"drills_completed": 0}

        return {
            "drills_completed": len(self._results),
            "avg_score": round(sum(r.score for r in self._results) / len(self._results), 1),
            "avg_accuracy": round(sum(r.accuracy for r in self._results) / len(self._results), 3),
            "weakness": self.get_weakness(),
            "best_score": round(max(r.score for r in self._results), 1),
        }
