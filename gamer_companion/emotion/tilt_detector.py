"""Tilt Detector — Performance-based tilt detection + adaptive coaching.

Monitors performance metrics (KDA, accuracy, reaction time, economy decisions)
over sliding windows. Detects tilt from statistical deviations, then adjusts
coaching tone from encouraging to calming to suggesting breaks.
"""

from __future__ import annotations
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
from loguru import logger


@dataclass
class TiltReading:
    """A single tilt measurement snapshot."""
    timestamp: float
    tilt_level: float  # 0.0 = zen, 1.0 = full tilt
    performance_score: float  # 0.0 = terrible, 1.0 = peak
    contributing_factors: List[str]
    coaching_tone: str  # "encouraging", "neutral", "calming", "urgent", "break"


class TiltDetector:
    """Detect player tilt from performance metrics.

    Tilt indicators (ranked by weight):
    1. Death streak (3+ consecutive deaths)
    2. Performance decline (recent K/D below session average)
    3. Reaction time increase (slowing down = frustration or fatigue)
    4. Economy mismanagement (buying when should save or vice versa)
    5. Reckless plays (pushing without info, rushing alone)
    6. Session duration (fatigue after 45+ minutes)
    7. Input pattern changes (faster, more erratic inputs = frustration)

    The detector computes a 0.0-1.0 tilt score and selects an appropriate
    coaching tone. High tilt triggers break suggestions and simpler strategies.
    """

    TILT_THRESHOLDS = {
        "zen": (0.0, 0.15),
        "focused": (0.15, 0.35),
        "frustrated": (0.35, 0.55),
        "tilted": (0.55, 0.75),
        "full_tilt": (0.75, 1.0),
    }

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._death_times: deque = deque(maxlen=50)
        self._kill_times: deque = deque(maxlen=50)
        self._performance_scores: deque = deque(maxlen=100)
        self._reaction_times: deque = deque(maxlen=100)
        self._session_start = time.time()
        self._current_tilt = 0.0
        self._tilt_history: deque[TiltReading] = deque(maxlen=200)
        self._death_streak = 0
        self._consecutive_round_losses = 0

    def on_kill(self):
        """Record a kill event."""
        self._kill_times.append(time.time())
        self._death_streak = 0
        self._performance_scores.append(0.8)

    def on_death(self):
        """Record a death event."""
        self._death_times.append(time.time())
        self._death_streak += 1
        self._performance_scores.append(0.2 - min(0.15, self._death_streak * 0.03))

    def on_round_result(self, won: bool):
        """Record round win/loss."""
        if won:
            self._consecutive_round_losses = 0
            self._performance_scores.append(0.7)
        else:
            self._consecutive_round_losses += 1
            self._performance_scores.append(0.3)

    def on_reaction_time(self, rt_ms: float):
        """Record a reaction time measurement."""
        self._reaction_times.append(rt_ms)

    def assess(self) -> TiltReading:
        """Compute current tilt level."""
        now = time.time()
        factors = []
        weights = []

        # Factor 1: Death streak (weight 0.25)
        death_score = min(1.0, self._death_streak * 0.2)
        if self._death_streak >= 3:
            factors.append(f"death_streak={self._death_streak}")
        weights.append(("death_streak", death_score, 0.25))

        # Factor 2: Recent KD ratio vs session average (weight 0.2)
        recent_kills = len([
            t for t in self._kill_times if now - t < 300
        ])
        recent_deaths = len([
            t for t in self._death_times if now - t < 300
        ])
        recent_kd = recent_kills / max(recent_deaths, 1)
        session_kills = len(self._kill_times)
        session_deaths = len(self._death_times)
        session_kd = session_kills / max(session_deaths, 1)

        kd_decline = 0
        if session_kd > 0 and recent_kd < session_kd * 0.6:
            kd_decline = min(1.0, (session_kd - recent_kd) / max(session_kd, 0.5))
            factors.append(f"kd_decline={recent_kd:.1f}vs{session_kd:.1f}")
        weights.append(("kd_decline", kd_decline, 0.2))

        # Factor 3: Reaction time increase (weight 0.15)
        rt_score = 0
        if len(self._reaction_times) >= 10:
            recent_rt = list(self._reaction_times)[-10:]
            baseline_rt = list(self._reaction_times)[:max(10, len(self._reaction_times) // 2)]
            avg_recent = sum(recent_rt) / len(recent_rt)
            avg_baseline = sum(baseline_rt) / len(baseline_rt)
            if avg_recent > avg_baseline * 1.2:
                rt_score = min(1.0, (avg_recent - avg_baseline) / avg_baseline)
                factors.append(f"rt_slow={avg_recent:.0f}vs{avg_baseline:.0f}ms")
        weights.append(("reaction_time", rt_score, 0.15))

        # Factor 4: Consecutive round losses (weight 0.15)
        loss_score = min(1.0, self._consecutive_round_losses * 0.15)
        if self._consecutive_round_losses >= 3:
            factors.append(f"loss_streak={self._consecutive_round_losses}")
        weights.append(("round_losses", loss_score, 0.15))

        # Factor 5: Session fatigue (weight 0.1)
        session_minutes = (now - self._session_start) / 60
        fatigue_score = 0
        if session_minutes > 45:
            fatigue_score = min(
                0.8, 0.15 * math.log(1 + (session_minutes - 45) / 30)
            )
            factors.append(f"fatigue={session_minutes:.0f}min")
        weights.append(("fatigue", fatigue_score, 0.1))

        # Factor 6: Recent performance trend (weight 0.15)
        perf_score = 0
        if len(self._performance_scores) >= 5:
            recent = list(self._performance_scores)[-5:]
            avg_perf = sum(recent) / len(recent)
            if avg_perf < 0.4:
                perf_score = 1.0 - avg_perf * 2
                factors.append(f"perf_low={avg_perf:.2f}")
        weights.append(("performance", perf_score, 0.15))

        # Weighted sum
        tilt = sum(score * weight for _, score, weight in weights)
        tilt = max(0.0, min(1.0, tilt))

        # Smooth with previous reading (EMA)
        alpha = 0.3
        self._current_tilt = alpha * tilt + (1 - alpha) * self._current_tilt

        # Determine coaching tone
        if self._current_tilt < 0.15:
            tone = "encouraging"
        elif self._current_tilt < 0.35:
            tone = "neutral"
        elif self._current_tilt < 0.55:
            tone = "calming"
        elif self._current_tilt < 0.75:
            tone = "urgent"
        else:
            tone = "break"

        # Performance score (inverse of tilt, roughly)
        perf = 1.0 - self._current_tilt

        reading = TiltReading(
            timestamp=now,
            tilt_level=round(self._current_tilt, 3),
            performance_score=round(perf, 3),
            contributing_factors=factors,
            coaching_tone=tone,
        )
        self._tilt_history.append(reading)
        return reading

    def get_coaching_message(self) -> Optional[str]:
        """Get a coaching message based on current tilt state."""
        reading = self.assess()
        if reading.coaching_tone == "break":
            return (
                "You've been playing for a while and performance is declining. "
                "Take a 5-minute break — stretch, hydrate, reset."
            )
        elif reading.coaching_tone == "urgent":
            return (
                "Rough stretch. Switch to simpler plays. "
                "Don't force anything — play for info, not kills."
            )
        elif reading.coaching_tone == "calming":
            return (
                "Stay focused. Take a breath before each round. "
                "One play at a time."
            )
        elif reading.coaching_tone == "encouraging":
            return "You're in the zone. Keep it up."
        return None

    @property
    def tilt_level(self) -> float:
        return self._current_tilt

    @property
    def tilt_state(self) -> str:
        for state, (low, high) in self.TILT_THRESHOLDS.items():
            if low <= self._current_tilt < high:
                return state
        return "full_tilt"

    def get_stats(self) -> dict:
        return {
            "tilt_level": round(self._current_tilt, 3),
            "tilt_state": self.tilt_state,
            "death_streak": self._death_streak,
            "consecutive_losses": self._consecutive_round_losses,
            "session_minutes": round(
                (time.time() - self._session_start) / 60, 1
            ),
            "total_kills": len(self._kill_times),
            "total_deaths": len(self._death_times),
            "readings": len(self._tilt_history),
        }
