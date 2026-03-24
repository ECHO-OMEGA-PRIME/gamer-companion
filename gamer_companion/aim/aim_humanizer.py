"""Aim Humanizer — Skill-scaled inaccuracy, micro-corrections, and jitter.

Makes AI aim look human by adding realistic imperfections:
- Skill-based inaccuracy envelope (worse players miss more)
- Micro-corrections (small adjustments near target)
- Natural hand tremor (subtle oscillation)
- Fatigue modeling (accuracy degrades over time)
- First-shot jitter (slight offset on first shot)
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from loguru import logger


@dataclass
class HumanizerProfile:
    """Parameters controlling aim humanization."""
    name: str
    skill_level: float = 0.85      # 0-1, maps to inaccuracy envelope
    tremor_amplitude: float = 0.5  # Pixels of hand tremor
    tremor_frequency: float = 8.0  # Hz of tremor oscillation
    micro_correction_chance: float = 0.3  # Chance of micro-correction per frame
    fatigue_rate: float = 0.001    # Accuracy loss per minute
    first_shot_jitter: float = 2.0 # Extra pixels of jitter on first shot


HUMANIZER_PROFILES = {
    "noob": HumanizerProfile(
        name="noob", skill_level=0.3,
        tremor_amplitude=2.0, tremor_frequency=6,
        micro_correction_chance=0.5, fatigue_rate=0.003,
        first_shot_jitter=8.0,
    ),
    "casual": HumanizerProfile(
        name="casual", skill_level=0.5,
        tremor_amplitude=1.2, tremor_frequency=7,
        micro_correction_chance=0.4, fatigue_rate=0.002,
        first_shot_jitter=5.0,
    ),
    "skilled": HumanizerProfile(
        name="skilled", skill_level=0.75,
        tremor_amplitude=0.6, tremor_frequency=8,
        micro_correction_chance=0.25, fatigue_rate=0.001,
        first_shot_jitter=2.5,
    ),
    "pro": HumanizerProfile(
        name="pro", skill_level=0.92,
        tremor_amplitude=0.3, tremor_frequency=10,
        micro_correction_chance=0.15, fatigue_rate=0.0005,
        first_shot_jitter=1.0,
    ),
    "inhuman": HumanizerProfile(
        name="inhuman", skill_level=0.99,
        tremor_amplitude=0.1, tremor_frequency=12,
        micro_correction_chance=0.05, fatigue_rate=0.0,
        first_shot_jitter=0.3,
    ),
}


class AimHumanizer:
    """Apply human-like imperfections to aim output.

    Takes raw aim deltas (dx, dy) from the aim engine and
    adds realistic noise, corrections, and tremor.

    Usage:
        humanizer = AimHumanizer("skilled")
        raw_dx, raw_dy = aim_engine.compute(target)
        final_dx, final_dy = humanizer.apply(raw_dx, raw_dy)
    """

    def __init__(self, profile_name: str = "skilled"):
        self._profile = HUMANIZER_PROFILES.get(
            profile_name, HUMANIZER_PROFILES["skilled"]
        )
        self._session_start = time.time()
        self._shot_count = 0
        self._is_first_shot = True
        self._tremor_phase = random.uniform(0, 2 * math.pi)
        self._fatigue_factor = 0.0
        self._last_update = time.time()
        self._corrections_made = 0
        self._total_applied = 0

    def set_profile(self, name: str) -> bool:
        profile = HUMANIZER_PROFILES.get(name)
        if not profile:
            return False
        self._profile = profile
        return True

    def apply(self, dx: float, dy: float, is_shooting: bool = False) -> Tuple[float, float]:
        """Apply humanization to aim deltas.

        Args:
            dx: Raw aim delta X from aim engine
            dy: Raw aim delta Y from aim engine
            is_shooting: Whether currently firing (affects micro-corrections)

        Returns:
            (humanized_dx, humanized_dy)
        """
        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        self._total_applied += 1

        p = self._profile

        # Update fatigue
        elapsed_min = (now - self._session_start) / 60
        self._fatigue_factor = min(0.3, p.fatigue_rate * elapsed_min)

        # 1. Inaccuracy envelope (gaussian noise scaled by skill)
        inaccuracy = (1.0 - p.skill_level + self._fatigue_factor) * 8.0
        noise_dx = random.gauss(0, inaccuracy)
        noise_dy = random.gauss(0, inaccuracy)

        # 2. Hand tremor (sinusoidal oscillation)
        self._tremor_phase += p.tremor_frequency * dt * 2 * math.pi
        tremor_dx = math.sin(self._tremor_phase) * p.tremor_amplitude
        tremor_dy = math.cos(self._tremor_phase * 1.3) * p.tremor_amplitude * 0.7

        # 3. First shot jitter
        first_shot_dx = 0.0
        first_shot_dy = 0.0
        if self._is_first_shot and is_shooting:
            first_shot_dx = random.gauss(0, p.first_shot_jitter)
            first_shot_dy = random.gauss(0, p.first_shot_jitter)
            self._is_first_shot = False
            self._shot_count += 1

        # 4. Micro-corrections (small adjustments when close to target)
        correction_dx = 0.0
        correction_dy = 0.0
        dist = math.hypot(dx, dy)
        if dist < 30 and random.random() < p.micro_correction_chance:
            # Overshoot correction
            correction_dx = -dx * random.uniform(0.05, 0.15)
            correction_dy = -dy * random.uniform(0.05, 0.15)
            self._corrections_made += 1

        # Combine all components
        final_dx = dx + noise_dx + tremor_dx + first_shot_dx + correction_dx
        final_dy = dy + noise_dy + tremor_dy + first_shot_dy + correction_dy

        return (round(final_dx, 2), round(final_dy, 2))

    def reset_shot(self):
        """Reset first-shot state (e.g., after burst pause)."""
        self._is_first_shot = True

    def reset_fatigue(self):
        """Reset fatigue (e.g., after a break)."""
        self._session_start = time.time()
        self._fatigue_factor = 0.0

    @property
    def fatigue(self) -> float:
        return round(self._fatigue_factor, 4)

    @property
    def profile(self) -> HumanizerProfile:
        return self._profile

    def list_profiles(self) -> List[str]:
        return list(HUMANIZER_PROFILES.keys())

    def get_stats(self) -> dict:
        return {
            "profile": self._profile.name,
            "skill_level": self._profile.skill_level,
            "fatigue": self.fatigue,
            "shots_fired": self._shot_count,
            "corrections_made": self._corrections_made,
            "total_applied": self._total_applied,
            "session_minutes": round((time.time() - self._session_start) / 60, 1),
        }
