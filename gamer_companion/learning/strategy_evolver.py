"""Strategy Evolver — Evolve strategies over time using evolutionary algorithms.

Combines Thompson Sampling with strategy mutation to discover new strategies
the AI hasn't tried before. Existing strategies that work get refined;
failing strategies get replaced with mutations of successful ones.
"""

from __future__ import annotations
import random
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class Strategy:
    """A game strategy with performance tracking."""
    strategy_id: str
    name: str
    actions: List[str]        # Ordered action plan
    parameters: Dict[str, float] = field(default_factory=dict)  # Tunable params
    wins: int = 0
    losses: int = 0
    total_reward: float = 0.0
    generation: int = 0       # Evolutionary generation
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def games_played(self) -> int:
        return self.wins + self.losses

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.games_played)


class StrategyEvolver:
    """Evolve strategies through selection, mutation, and crossover.

    Lifecycle:
    1. Initialize with seed strategies
    2. Select strategy for each round (Thompson Sampling)
    3. Record outcome
    4. Periodically evolve: kill weak strategies, mutate strong ones
    5. Inject random mutations to explore

    Parameters mutated: aggression, timing, site_preference, utility_usage, etc.
    """

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.2,
        evolve_every: int = 10,
    ):
        self._population: Dict[str, Strategy] = {}
        self._pop_size = population_size
        self._mutation_rate = mutation_rate
        self._evolve_every = evolve_every
        self._round_count = 0
        self._generation = 0
        self._history: List[dict] = []

    def add_strategy(self, strategy: Strategy):
        """Add a strategy to the population."""
        self._population[strategy.strategy_id] = strategy

    def seed(self, strategies: List[Strategy]):
        """Seed initial population."""
        for s in strategies:
            self._population[s.strategy_id] = s

    def select(self) -> Strategy:
        """Select a strategy using Thompson Sampling."""
        if not self._population:
            raise ValueError("No strategies in population")

        best_sample = -1.0
        best_strategy = None

        for strategy in self._population.values():
            alpha = max(1, strategy.wins + 1)
            beta = max(1, strategy.losses + 1)
            sample = random.betavariate(alpha, beta)

            if sample > best_sample:
                best_sample = sample
                best_strategy = strategy

        return best_strategy

    def record_outcome(self, strategy_id: str, won: bool, reward: float = 0.0):
        """Record the outcome of using a strategy."""
        strategy = self._population.get(strategy_id)
        if not strategy:
            return

        if won:
            strategy.wins += 1
        else:
            strategy.losses += 1
        strategy.total_reward += reward

        self._round_count += 1
        self._history.append({
            "round": self._round_count,
            "strategy": strategy_id,
            "won": won,
            "reward": reward,
            "timestamp": time.time(),
        })

        # Evolve periodically
        if self._round_count % self._evolve_every == 0:
            self.evolve()

    def evolve(self):
        """Run one evolution cycle: cull weak, mutate strong, inject random."""
        if len(self._population) < 3:
            return

        self._generation += 1
        strategies = list(self._population.values())

        # Sort by fitness (win_rate * log(games_played + 1))
        import math
        strategies.sort(
            key=lambda s: s.win_rate * math.log(s.games_played + 2),
            reverse=True,
        )

        # Keep top 60%, cull bottom 40%
        keep_count = max(3, int(len(strategies) * 0.6))
        survivors = strategies[:keep_count]
        culled = strategies[keep_count:]

        # Remove culled
        for s in culled:
            self._population.pop(s.strategy_id, None)

        # Mutate top performers to fill population
        while len(self._population) < self._pop_size:
            parent = random.choice(survivors[:max(1, len(survivors) // 2)])
            child = self._mutate(parent)
            self._population[child.strategy_id] = child

        logger.debug(
            f"Evolution gen {self._generation}: "
            f"kept {keep_count}, culled {len(culled)}, "
            f"pop={len(self._population)}"
        )

    def _mutate(self, parent: Strategy) -> Strategy:
        """Create a mutated copy of a strategy."""
        child_id = f"strat_g{self._generation}_{random.randint(1000, 9999)}"

        # Mutate actions (swap, insert, remove)
        child_actions = list(parent.actions)
        if child_actions and random.random() < self._mutation_rate:
            mutation_type = random.choice(["swap", "remove", "duplicate"])
            if mutation_type == "swap" and len(child_actions) >= 2:
                i, j = random.sample(range(len(child_actions)), 2)
                child_actions[i], child_actions[j] = child_actions[j], child_actions[i]
            elif mutation_type == "remove" and len(child_actions) > 1:
                child_actions.pop(random.randint(0, len(child_actions) - 1))
            elif mutation_type == "duplicate":
                idx = random.randint(0, len(child_actions) - 1)
                child_actions.insert(idx, child_actions[idx])

        # Mutate parameters (gaussian perturbation)
        child_params = dict(parent.parameters)
        for key in child_params:
            if random.random() < self._mutation_rate:
                child_params[key] += random.gauss(0, 0.1)
                child_params[key] = max(0.0, min(1.0, child_params[key]))

        return Strategy(
            strategy_id=child_id,
            name=f"{parent.name}_mut",
            actions=child_actions,
            parameters=child_params,
            generation=self._generation,
            parent_id=parent.strategy_id,
        )

    def get_top(self, limit: int = 5) -> List[Strategy]:
        """Get top performing strategies."""
        return sorted(
            self._population.values(),
            key=lambda s: s.win_rate if s.games_played >= 3 else 0,
            reverse=True,
        )[:limit]

    def get_stats(self) -> dict:
        played = [s for s in self._population.values() if s.games_played > 0]
        return {
            "population_size": len(self._population),
            "generation": self._generation,
            "rounds_played": self._round_count,
            "avg_win_rate": round(
                sum(s.win_rate for s in played) / max(1, len(played)), 3
            ),
            "best_win_rate": round(
                max((s.win_rate for s in played), default=0), 3
            ),
        }
