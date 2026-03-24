"""Benchmark Runner — Measure AI performance over time.

Runs standardized test suites and tracks scores to measure improvement.
The "AI Olympics" for the game companion.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


@dataclass
class BenchmarkTest:
    """A single benchmark test."""
    test_id: str
    name: str
    category: str             # "aim", "movement", "decision", "reaction", "strategy"
    description: str = ""
    max_score: float = 100.0
    weight: float = 1.0       # Importance weight in overall score


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_id: str
    score: float
    max_score: float
    details: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def pct(self) -> float:
        return round(self.score / max(1, self.max_score) * 100, 1)


@dataclass
class BenchmarkRun:
    """A complete benchmark run (all tests)."""
    run_id: str
    results: List[BenchmarkResult] = field(default_factory=list)
    overall_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    game_id: str = ""
    notes: str = ""


# Standard benchmark suite
STANDARD_TESTS: List[BenchmarkTest] = [
    BenchmarkTest("aim_flick", "Flick Aim", "aim", "30 targets, 1.5s each, random positions", 100, 1.5),
    BenchmarkTest("aim_tracking", "Tracking", "aim", "Track moving target for 30s", 100, 1.0),
    BenchmarkTest("aim_spray", "Spray Control", "aim", "AK-47 spray 30 bullets, measure grouping", 100, 1.2),
    BenchmarkTest("reaction_visual", "Visual Reaction", "reaction", "React to visual stimulus 20 times", 100, 1.0),
    BenchmarkTest("reaction_audio", "Audio Reaction", "reaction", "React to audio cue 20 times", 100, 0.8),
    BenchmarkTest("movement_strafe", "Strafe Accuracy", "movement", "Counter-strafe and shoot 20 times", 100, 1.0),
    BenchmarkTest("movement_peek", "Peek Timing", "movement", "Jiggle peek, wide peek, shoulder peek", 100, 0.8),
    BenchmarkTest("decision_combat", "Combat Decision", "decision", "20 combat scenarios, choose action", 100, 1.5),
    BenchmarkTest("decision_economy", "Economy Decision", "decision", "15 buy/save scenarios", 100, 0.8),
    BenchmarkTest("strategy_rotation", "Rotation Timing", "strategy", "React to info and rotate", 100, 1.0),
]


class BenchmarkRunner:
    """Run standardized benchmark suites and track improvement.

    Features:
    - Standardized test battery (same tests every time)
    - Historical tracking (measure improvement over days/weeks)
    - Category breakdown (aim vs movement vs decision)
    - Percentile ranking against historical data
    - JSON export for reporting
    """

    def __init__(self, persist_path: str = None):
        self._tests = {t.test_id: t for t in STANDARD_TESTS}
        self._history: List[BenchmarkRun] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    def start_run(self, game_id: str = "") -> BenchmarkRun:
        """Start a new benchmark run."""
        run = BenchmarkRun(
            run_id=f"bench_{int(time.time())}",
            game_id=game_id,
        )
        return run

    def record_test(
        self,
        run: BenchmarkRun,
        test_id: str,
        score: float,
        details: Dict[str, float] = None,
    ) -> Optional[BenchmarkResult]:
        """Record a test result within a run."""
        test = self._tests.get(test_id)
        if not test:
            logger.warning(f"Unknown benchmark test: {test_id}")
            return None

        result = BenchmarkResult(
            test_id=test_id,
            score=min(test.max_score, max(0, score)),
            max_score=test.max_score,
            details=details or {},
        )
        run.results.append(result)
        return result

    def finish_run(self, run: BenchmarkRun) -> float:
        """Finish a benchmark run and compute overall score."""
        if not run.results:
            return 0.0

        total_weighted = 0.0
        total_weight = 0.0

        for result in run.results:
            test = self._tests.get(result.test_id)
            weight = test.weight if test else 1.0
            total_weighted += result.pct * weight
            total_weight += weight

        run.overall_score = round(total_weighted / max(1, total_weight), 1)

        self._history.append(run)
        self._save()

        return run.overall_score

    def get_improvement(self, last_n: int = 5) -> Dict:
        """Calculate improvement trend over last N runs."""
        if len(self._history) < 2:
            return {"trend": "insufficient_data", "runs": len(self._history)}

        recent = self._history[-last_n:]
        scores = [r.overall_score for r in recent]

        if len(scores) < 2:
            return {"trend": "insufficient_data", "runs": len(scores)}

        first_half = sum(scores[:len(scores)//2]) / max(1, len(scores)//2)
        second_half = sum(scores[len(scores)//2:]) / max(1, len(scores) - len(scores)//2)

        change = second_half - first_half

        return {
            "trend": "improving" if change > 2 else "stable" if change > -2 else "declining",
            "change": round(change, 1),
            "current_score": round(scores[-1], 1),
            "best_score": round(max(r.overall_score for r in self._history), 1),
            "runs_total": len(self._history),
        }

    def get_category_scores(self, run: BenchmarkRun = None) -> Dict[str, float]:
        """Get average scores by category."""
        if run is None and self._history:
            run = self._history[-1]
        if not run:
            return {}

        cats: Dict[str, List[float]] = {}
        for result in run.results:
            test = self._tests.get(result.test_id)
            if test:
                if test.category not in cats:
                    cats[test.category] = []
                cats[test.category].append(result.pct)

        return {k: round(sum(v) / len(v), 1) for k, v in cats.items()}

    def get_weakest_category(self) -> str:
        """Get the weakest category from the latest run."""
        scores = self.get_category_scores()
        if not scores:
            return "no_data"
        return min(scores, key=scores.get)

    def list_tests(self) -> List[dict]:
        return [
            {
                "test_id": t.test_id,
                "name": t.name,
                "category": t.category,
                "weight": t.weight,
            }
            for t in self._tests.values()
        ]

    def get_stats(self) -> dict:
        return {
            "total_runs": len(self._history),
            "tests_available": len(self._tests),
            "latest_score": round(self._history[-1].overall_score, 1) if self._history else 0,
            "best_score": round(max((r.overall_score for r in self._history), default=0), 1),
            "improvement": self.get_improvement(),
        }

    def _save(self):
        if not self._persist_path:
            return
        data = {
            "runs": [
                {
                    "run_id": r.run_id,
                    "overall_score": r.overall_score,
                    "timestamp": r.timestamp,
                    "game_id": r.game_id,
                    "results": [
                        {
                            "test_id": res.test_id,
                            "score": res.score,
                            "max_score": res.max_score,
                            "pct": res.pct,
                        }
                        for res in r.results
                    ],
                }
                for r in self._history[-50:]  # Keep last 50 runs
            ]
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for run_data in data.get("runs", []):
                run = BenchmarkRun(
                    run_id=run_data["run_id"],
                    overall_score=run_data["overall_score"],
                    timestamp=run_data["timestamp"],
                    game_id=run_data.get("game_id", ""),
                )
                for res_data in run_data.get("results", []):
                    run.results.append(BenchmarkResult(
                        test_id=res_data["test_id"],
                        score=res_data["score"],
                        max_score=res_data["max_score"],
                        timestamp=run_data["timestamp"],
                    ))
                self._history.append(run)
        except Exception as e:
            logger.warning(f"Failed to load benchmark history: {e}")
