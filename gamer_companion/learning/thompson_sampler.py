"""Thompson Sampler — Multi-armed bandit for strategy selection.

Instead of hard-coded "always do X in situation Y," the AI explores
different strategies and converges on what actually works through
experience. Uses Beta distribution sampling (Thompson Sampling) to
balance exploration vs exploitation.
"""

from __future__ import annotations
import random
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


@dataclass
class StrategyArm:
    """A strategy option (arm in the multi-armed bandit)."""
    name: str
    description: str = ""
    successes: int = 0
    failures: int = 0
    total_reward: float = 0
    last_used: float = 0

    @property
    def sample(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)."""
        return random.betavariate(
            max(1, self.successes + 1),
            max(1, self.failures + 1),
        )

    @property
    def win_rate(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5
        return self.successes / total

    @property
    def confidence(self) -> float:
        total = self.successes + self.failures
        if total < 5:
            return 0.1
        return min(0.95, 1 - 1 / math.sqrt(total))


class ThompsonSampler:
    """Multi-armed bandit for strategy selection.

    Example arms for CS2 T-side:
    - "rush_a": Fast A execute
    - "rush_b": Fast B execute
    - "slow_default": Spread out, gather info, decide late
    - "fake_a_go_b": Utility at A, rotate B
    - "mid_split": Control mid, split to weaker site

    Each round, Thompson Sampling picks the strategy with the highest
    sampled win probability. Over time, the best strategies naturally
    get selected more often.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._arms: Dict[str, Dict[str, StrategyArm]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    def add_strategy(
        self, context: str, name: str, description: str = "",
    ):
        """Add a strategy option for a given context."""
        if context not in self._arms:
            self._arms[context] = {}
        if name not in self._arms[context]:
            self._arms[context][name] = StrategyArm(
                name=name, description=description,
            )

    def select(self, context: str) -> Optional[str]:
        """Select the best strategy for a context via Thompson Sampling."""
        arms = self._arms.get(context, {})
        if not arms:
            return None

        samples = {name: arm.sample for name, arm in arms.items()}
        selected = max(samples, key=samples.get)

        self._arms[context][selected].last_used = time.time()
        logger.debug(
            f"Strategy selected: {selected} (context: {context}, "
            f"sampled: {samples[selected]:.3f}, "
            f"win_rate: {arms[selected].win_rate:.2f})"
        )
        return selected

    def update(self, context: str, strategy_name: str, reward: float):
        """Update a strategy's success/failure based on outcome.

        reward: 1.0 = full win, 0.0 = full loss, 0.5 = draw
        """
        arms = self._arms.get(context, {})
        if strategy_name not in arms:
            return

        arm = arms[strategy_name]
        if reward > 0.5:
            arm.successes += 1
        elif reward < 0.5:
            arm.failures += 1
        else:
            arm.successes += 0.5
            arm.failures += 0.5
        arm.total_reward += reward
        self._save()

    def get_stats(self, context: str) -> List[dict]:
        """Get stats for all strategies in a context."""
        arms = self._arms.get(context, {})
        return sorted(
            [
                {
                    "name": arm.name,
                    "win_rate": round(arm.win_rate, 3),
                    "confidence": round(arm.confidence, 3),
                    "total_plays": arm.successes + arm.failures,
                    "successes": arm.successes,
                    "failures": arm.failures,
                }
                for arm in arms.values()
            ],
            key=lambda x: x["win_rate"],
            reverse=True,
        )

    def _save(self):
        if not self._persist_path:
            return
        data = {}
        for ctx, arms in self._arms.items():
            data[ctx] = {
                name: {
                    "s": arm.successes, "f": arm.failures,
                    "r": arm.total_reward,
                }
                for name, arm in arms.items()
            }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for ctx, arms in data.items():
                self._arms[ctx] = {}
                for name, stats in arms.items():
                    self._arms[ctx][name] = StrategyArm(
                        name=name,
                        successes=stats.get("s", 0),
                        failures=stats.get("f", 0),
                        total_reward=stats.get("r", 0),
                    )
            total = sum(len(a) for a in self._arms.values())
            logger.info(
                f"Loaded strategy memory: {total} strategies "
                f"across {len(self._arms)} contexts"
            )
        except Exception as e:
            logger.warning(f"Failed to load strategy memory: {e}")
