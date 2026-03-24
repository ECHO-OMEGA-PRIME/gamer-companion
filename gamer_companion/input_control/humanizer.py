"""Humanizer — Fitts' Law speed, micro-corrections, fatigue simulation.

Makes AI input indistinguishable from human input by applying:
1. Fitts' Law movement time (distance/size → speed)
2. Log-normal reaction time distribution
3. Micro-correction movements near targets
4. Fatigue model (performance degrades over session)
5. Random micro-pauses (human hesitation)
6. Accuracy degradation under pressure
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass
from typing import Tuple
from loguru import logger


@dataclass
class HumanProfile:
    """Configurable human-like performance profile.

    Based on published HCI research for FPS players:
    - Novice:    reaction ~300ms, accuracy 0.6, fatigue 0.7
    - Average:   reaction ~220ms, accuracy 0.75, fatigue 0.85
    - Skilled:   reaction ~180ms, accuracy 0.88, fatigue 0.92
    - Pro:       reaction ~150ms, accuracy 0.95, fatigue 0.97
    """
    reaction_mean_ms: float = 200.0
    reaction_std_ms: float = 40.0
    accuracy_base: float = 0.85
    fatigue_resistance: float = 0.9  # 1.0 = no fatigue
    micro_pause_chance: float = 0.05
    micro_pause_ms: Tuple[float, float] = (50.0, 200.0)
    overshoot_chance: float = 0.12
    tremor_amplitude: float = 1.5  # pixels of hand tremor


PROFILES = {
    "novice": HumanProfile(
        reaction_mean_ms=300, reaction_std_ms=60,
        accuracy_base=0.6, fatigue_resistance=0.7,
        overshoot_chance=0.25, tremor_amplitude=3.0,
    ),
    "average": HumanProfile(
        reaction_mean_ms=220, reaction_std_ms=45,
        accuracy_base=0.75, fatigue_resistance=0.85,
    ),
    "skilled": HumanProfile(
        reaction_mean_ms=180, reaction_std_ms=35,
        accuracy_base=0.88, fatigue_resistance=0.92,
        overshoot_chance=0.08, tremor_amplitude=1.0,
    ),
    "pro": HumanProfile(
        reaction_mean_ms=150, reaction_std_ms=25,
        accuracy_base=0.95, fatigue_resistance=0.97,
        overshoot_chance=0.05, tremor_amplitude=0.5,
    ),
}


class Humanizer:
    """Apply human-like imperfections to AI actions.

    Used by the input control system to ensure all actions look
    statistically indistinguishable from human play.
    """

    def __init__(self, profile: str = "skilled"):
        self._profile = PROFILES.get(profile, PROFILES["skilled"])
        self._session_start = time.time()
        self._action_count = 0
        self._last_reaction = 0

    def reaction_delay(self) -> float:
        """Generate a log-normal reaction time in milliseconds.

        Log-normal distribution matches published human RT data better
        than gaussian (right-skewed, floor ~100ms, occasional slow ones).
        """
        mu = math.log(self._profile.reaction_mean_ms)
        sigma = self._profile.reaction_std_ms / self._profile.reaction_mean_ms
        rt = random.lognormvariate(mu, sigma)

        # Apply fatigue
        fatigue = self._fatigue_factor()
        rt *= fatigue

        # Floor
        rt = max(90.0, rt)

        self._last_reaction = rt
        self._action_count += 1
        return rt

    def aim_offset(self, target_size: float = 20.0) -> Tuple[float, float]:
        """Generate a human-like aim offset (miss amount).

        Smaller targets = bigger miss. Fatigue = bigger miss.
        """
        accuracy = self._profile.accuracy_base
        fatigue = self._fatigue_factor()
        effective_accuracy = accuracy / fatigue

        # Standard deviation of miss in pixels
        miss_std = target_size * (1 - effective_accuracy) * 0.5

        # Add tremor
        tremor = self._profile.tremor_amplitude * self._fatigue_factor()

        dx = random.gauss(0, miss_std + tremor)
        dy = random.gauss(0, miss_std + tremor)

        return (dx, dy)

    def should_micro_pause(self) -> bool:
        """Should the AI inject a brief hesitation pause?"""
        return random.random() < self._profile.micro_pause_chance

    def micro_pause_duration(self) -> float:
        """Get a random micro-pause duration in ms."""
        lo, hi = self._profile.micro_pause_ms
        return random.uniform(lo, hi)

    def should_overshoot(self) -> bool:
        """Should the mouse overshoot the target?"""
        return random.random() < self._profile.overshoot_chance

    def movement_time_ms(
        self, distance: float, target_size: float = 20.0,
    ) -> float:
        """Fitts' Law movement time with human variance.

        T = a + b * log2(1 + D/W) + noise
        """
        if target_size <= 0:
            target_size = 1
        a = 100  # base time ms
        b = 150  # scaling factor ms
        fitts = a + b * math.log2(1 + distance / target_size)

        # Add variance (±15%)
        noise = random.gauss(1.0, 0.15)
        fitts *= max(0.5, noise)

        # Apply fatigue
        fitts *= self._fatigue_factor()

        return max(50.0, fitts)

    def click_duration_ms(self) -> float:
        """Human-like click duration (press to release)."""
        return max(30.0, random.gauss(85.0, 20.0))

    def inter_key_delay_ms(self) -> float:
        """Delay between sequential key presses."""
        return max(15.0, random.gauss(45.0, 12.0))

    def set_profile(self, profile: str):
        """Switch to a different skill profile."""
        if profile in PROFILES:
            self._profile = PROFILES[profile]

    def _fatigue_factor(self) -> float:
        """Compute fatigue multiplier (>1.0 = degraded performance).

        Performance degrades logarithmically over session duration.
        """
        minutes = (time.time() - self._session_start) / 60
        if minutes < 15:
            return 1.0
        resistance = self._profile.fatigue_resistance
        degradation = (1 - resistance) * math.log(1 + (minutes - 15) / 30)
        return 1.0 + degradation

    def get_stats(self) -> dict:
        return {
            "profile": next(
                (k for k, v in PROFILES.items() if v is self._profile),
                "custom",
            ),
            "fatigue_factor": round(self._fatigue_factor(), 3),
            "session_minutes": round(
                (time.time() - self._session_start) / 60, 1
            ),
            "actions": self._action_count,
            "last_reaction_ms": round(self._last_reaction, 1),
        }
