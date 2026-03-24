"""Pattern Extractor — Extract reusable action patterns from gameplay.

Identifies recurring sequences of actions that lead to positive outcomes,
then packages them as reusable templates the AI can apply in similar situations.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter
from loguru import logger


@dataclass
class ActionEvent:
    """A single action in a sequence."""
    timestamp: float
    action: str               # "peek_left", "shoot", "flash", "rotate_b"
    target: str = ""          # "enemy_1", "site_a"
    outcome: str = ""         # "hit", "miss", "kill", "traded"
    game_phase: str = ""      # "early", "mid", "late", "post_plant"


@dataclass
class ActionPattern:
    """A reusable pattern of actions."""
    pattern_id: str
    name: str
    actions: List[str]        # Ordered action sequence
    trigger: str              # When to use this pattern
    avg_duration_ms: float    # How long the pattern takes
    success_rate: float       # Win rate when using this pattern
    occurrences: int          # How many times observed
    game_phase: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Confidence based on sample size."""
        if self.occurrences < 3:
            return 0.1
        if self.occurrences < 10:
            return 0.3 + (self.occurrences / 10) * 0.4
        return min(0.95, 0.7 + (self.occurrences / 100) * 0.25)


class PatternExtractor:
    """Extract and store reusable action patterns.

    Process:
    1. Receive streams of ActionEvents from gameplay
    2. Segment into episodes (round boundaries, death/respawn)
    3. Find common subsequences across episodes
    4. Score patterns by success rate
    5. Store patterns for reuse
    """

    def __init__(self, min_pattern_length: int = 2, max_pattern_length: int = 8):
        self._min_len = min_pattern_length
        self._max_len = max_pattern_length
        self._episodes: List[List[ActionEvent]] = []
        self._current_episode: List[ActionEvent] = []
        self._patterns: Dict[str, ActionPattern] = {}
        self._episode_outcomes: List[float] = []  # reward per episode

    def add_event(self, event: ActionEvent):
        """Add an action event to the current episode."""
        self._current_episode.append(event)

    def end_episode(self, reward: float = 0.0):
        """End current episode (round end, death, etc.)."""
        if self._current_episode:
            self._episodes.append(self._current_episode)
            self._episode_outcomes.append(reward)
            self._current_episode = []

    def extract(self) -> List[ActionPattern]:
        """Extract patterns from all episodes."""
        if len(self._episodes) < 2:
            return []

        # Convert episodes to action strings
        action_episodes = [
            [e.action for e in ep] for ep in self._episodes
        ]

        # Find common n-grams across episodes
        ngram_counts: Counter = Counter()
        ngram_rewards: Dict[str, List[float]] = {}

        for ep_idx, actions in enumerate(action_episodes):
            reward = self._episode_outcomes[ep_idx] if ep_idx < len(self._episode_outcomes) else 0.0

            seen_in_episode = set()
            for n in range(self._min_len, min(self._max_len + 1, len(actions) + 1)):
                for i in range(len(actions) - n + 1):
                    ngram = tuple(actions[i:i + n])
                    ngram_key = "|".join(ngram)

                    if ngram_key not in seen_in_episode:
                        ngram_counts[ngram_key] += 1
                        seen_in_episode.add(ngram_key)

                        if ngram_key not in ngram_rewards:
                            ngram_rewards[ngram_key] = []
                        ngram_rewards[ngram_key].append(reward)

        # Filter to patterns appearing 2+ times
        new_patterns = []
        for ngram_key, count in ngram_counts.most_common(50):
            if count < 2:
                break

            actions = ngram_key.split("|")
            rewards = ngram_rewards.get(ngram_key, [])
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            success_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0

            # Estimate duration from episode data
            durations = self._estimate_durations(actions)
            avg_duration = sum(durations) / len(durations) if durations else 1000.0

            pattern = ActionPattern(
                pattern_id=f"pat_{hash(ngram_key) & 0xFFFFFFFF:08x}",
                name=f"{'_'.join(actions[:3])}{'_...' if len(actions) > 3 else ''}",
                actions=actions,
                trigger=actions[0],  # First action as trigger
                avg_duration_ms=round(avg_duration, 1),
                success_rate=round(success_rate, 3),
                occurrences=count,
                tags=[f"len_{len(actions)}"],
            )

            # Merge with existing or add new
            existing = self._patterns.get(pattern.pattern_id)
            if existing:
                existing.occurrences += count
                existing.success_rate = round(
                    (existing.success_rate + success_rate) / 2, 3
                )
            else:
                self._patterns[pattern.pattern_id] = pattern
                new_patterns.append(pattern)

        return new_patterns

    def _estimate_durations(self, actions: List[str]) -> List[float]:
        """Estimate action sequence duration from episode timestamps."""
        durations = []
        for episode in self._episodes:
            ep_actions = [e.action for e in episode]
            # Find this subsequence in the episode
            for i in range(len(ep_actions) - len(actions) + 1):
                if ep_actions[i:i + len(actions)] == actions:
                    start_t = episode[i].timestamp
                    end_t = episode[i + len(actions) - 1].timestamp
                    dur_ms = (end_t - start_t) * 1000
                    if dur_ms > 0:
                        durations.append(dur_ms)
                    break
        return durations

    def get_patterns_for(self, trigger: str, min_success: float = 0.3) -> List[ActionPattern]:
        """Get patterns matching a trigger action."""
        return sorted(
            [
                p for p in self._patterns.values()
                if p.trigger == trigger and p.success_rate >= min_success
            ],
            key=lambda p: p.success_rate * p.confidence,
            reverse=True,
        )

    def get_best_pattern(self, trigger: str) -> Optional[ActionPattern]:
        """Get the single best pattern for a trigger."""
        patterns = self.get_patterns_for(trigger)
        return patterns[0] if patterns else None

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def get_stats(self) -> dict:
        return {
            "episodes": len(self._episodes),
            "patterns": len(self._patterns),
            "avg_pattern_length": round(
                sum(len(p.actions) for p in self._patterns.values()) / max(1, len(self._patterns)), 1
            ),
            "top_success_rate": round(
                max((p.success_rate for p in self._patterns.values()), default=0), 3
            ),
        }
