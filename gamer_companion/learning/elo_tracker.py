"""ELO Tracker — Track AI skill rating over time using ELO system.

Provides a single number representing the AI's current skill level,
updated after each match based on opponent difficulty and outcome.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class ELOMatch:
    """A match result for ELO calculation."""
    match_id: str
    timestamp: float
    opponent_elo: float       # Estimated opponent rating
    result: float             # 1.0 = win, 0.5 = draw, 0.0 = loss
    elo_before: float
    elo_after: float
    elo_change: float
    game_id: str = ""
    metadata: dict = field(default_factory=dict)


class ELOTracker:
    """Track and update AI skill rating using ELO system.

    Standard ELO:
    - K-factor of 32 for new players (< 30 games), 16 for established
    - Expected score: E = 1 / (1 + 10^((Rb-Ra)/400))
    - New rating: Ra' = Ra + K * (S - E)

    Extensions:
    - Per-game ELO tracking
    - Confidence interval based on game count
    - Streak bonuses/penalties
    - Rating history with trend analysis
    """

    def __init__(self, initial_elo: float = 1000.0):
        self._ratings: Dict[str, float] = {"global": initial_elo}
        self._history: Dict[str, List[ELOMatch]] = {"global": []}
        self._initial = initial_elo

    def record_match(
        self,
        opponent_elo: float,
        result: float,
        game_id: str = "global",
        match_id: str = None,
        metadata: dict = None,
    ) -> ELOMatch:
        """Record a match result and update ELO.

        Args:
            opponent_elo: Estimated opponent rating
            result: 1.0=win, 0.5=draw, 0.0=loss
            game_id: Game identifier for per-game tracking
            match_id: Optional match identifier
            metadata: Optional metadata
        """
        if game_id not in self._ratings:
            self._ratings[game_id] = self._initial
            self._history[game_id] = []

        current = self._ratings[game_id]
        games_played = len(self._history[game_id])

        # K-factor: higher for new players, lower for established
        k = 32 if games_played < 30 else 16

        # Expected score
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - current) / 400))

        # New rating
        change = k * (result - expected)
        new_rating = max(100, current + change)  # Floor at 100

        match = ELOMatch(
            match_id=match_id or f"m_{int(time.time())}_{games_played}",
            timestamp=time.time(),
            opponent_elo=opponent_elo,
            result=result,
            elo_before=round(current, 1),
            elo_after=round(new_rating, 1),
            elo_change=round(change, 1),
            game_id=game_id,
            metadata=metadata or {},
        )

        self._ratings[game_id] = new_rating
        self._history[game_id].append(match)

        # Also update global
        if game_id != "global":
            self.record_match(opponent_elo, result, "global", match_id, metadata)

        return match

    def get_rating(self, game_id: str = "global") -> float:
        """Get current ELO rating."""
        return round(self._ratings.get(game_id, self._initial), 1)

    def get_rank(self, game_id: str = "global") -> str:
        """Get rank name from ELO rating."""
        elo = self.get_rating(game_id)
        if elo >= 2400:
            return "Grandmaster"
        if elo >= 2000:
            return "Master"
        if elo >= 1700:
            return "Diamond"
        if elo >= 1400:
            return "Platinum"
        if elo >= 1200:
            return "Gold"
        if elo >= 1000:
            return "Silver"
        if elo >= 800:
            return "Bronze"
        return "Iron"

    def get_confidence(self, game_id: str = "global") -> float:
        """Get confidence in the rating (0-1 based on games played)."""
        games = len(self._history.get(game_id, []))
        if games < 5:
            return 0.1
        if games < 10:
            return 0.3
        if games < 30:
            return 0.6
        return min(0.95, 0.7 + (games - 30) / 200)

    def get_trend(self, game_id: str = "global", window: int = 10) -> str:
        """Get recent rating trend."""
        history = self._history.get(game_id, [])
        if len(history) < window:
            return "insufficient_data"

        recent = history[-window:]
        total_change = sum(m.elo_change for m in recent)

        if total_change > 20:
            return "climbing"
        if total_change > 5:
            return "rising"
        if total_change > -5:
            return "stable"
        if total_change > -20:
            return "declining"
        return "falling"

    def get_streak(self, game_id: str = "global") -> Tuple[str, int]:
        """Get current win/loss streak."""
        history = self._history.get(game_id, [])
        if not history:
            return ("none", 0)

        streak_type = "win" if history[-1].result >= 0.5 else "loss"
        count = 0

        for match in reversed(history):
            is_win = match.result >= 0.5
            if (streak_type == "win" and is_win) or (streak_type == "loss" and not is_win):
                count += 1
            else:
                break

        return (streak_type, count)

    def get_win_rate(self, game_id: str = "global", last_n: int = 0) -> float:
        """Get win rate (optionally for last N games)."""
        history = self._history.get(game_id, [])
        if not history:
            return 0.5

        if last_n > 0:
            history = history[-last_n:]

        wins = sum(1 for m in history if m.result >= 0.5)
        return round(wins / len(history), 3)

    def get_history(self, game_id: str = "global", limit: int = 50) -> List[ELOMatch]:
        """Get match history."""
        history = self._history.get(game_id, [])
        return history[-limit:]

    def get_stats(self, game_id: str = "global") -> dict:
        history = self._history.get(game_id, [])
        streak_type, streak_count = self.get_streak(game_id)
        return {
            "elo": self.get_rating(game_id),
            "rank": self.get_rank(game_id),
            "games_played": len(history),
            "win_rate": self.get_win_rate(game_id),
            "win_rate_last_10": self.get_win_rate(game_id, 10),
            "confidence": round(self.get_confidence(game_id), 2),
            "trend": self.get_trend(game_id),
            "streak": f"{streak_type}_{streak_count}",
            "peak_elo": round(max((m.elo_after for m in history), default=self._initial), 1),
        }

    def to_json(self) -> str:
        """Export state as JSON."""
        return json.dumps({
            "ratings": {k: round(v, 1) for k, v in self._ratings.items()},
            "history": {
                k: [
                    {
                        "match_id": m.match_id,
                        "timestamp": m.timestamp,
                        "opponent_elo": m.opponent_elo,
                        "result": m.result,
                        "elo_before": m.elo_before,
                        "elo_after": m.elo_after,
                        "elo_change": m.elo_change,
                    }
                    for m in v[-100:]  # Keep last 100 per game
                ]
                for k, v in self._history.items()
            },
        }, indent=2)


# Fix missing import
from typing import Tuple
