"""Anti-Detection Layer — Statistical evasion for human-like behavior.

Wraps ALL input actions and injects human-like statistical noise into
timing, trajectories, and behavior patterns.

Does NOT: read game memory, inject DLLs, hook DirectX, modify files,
or intercept network packets. All inputs are external screen capture +
simulated hardware input.
"""

from __future__ import annotations
import random
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class TimingProfile:
    """Human timing characteristics derived from research."""
    mean_reaction_ms: float = 220     # Average human FPS reaction
    std_reaction_ms: float = 45       # Standard deviation
    mean_aps: float = 4.5             # Average actions per second (casual)
    max_aps: float = 12               # Peak human APS (pro players)
    fatigue_onset_min: float = 45     # Minutes before fatigue manifests
    fatigue_multiplier: float = 1.15  # Reaction slows 15% when fatigued
    pause_probability: float = 0.03   # Chance of random micro-pause per action
    pause_duration_ms: tuple = (80, 400)
    double_click_interval_ms: tuple = (40, 120)
    key_hold_variance_ms: float = 30


class AntiDetectionLayer:
    """Statistical evasion system.

    Anti-cheat systems detect bots via:
    1. Timing regularity (bots act at fixed intervals)
    2. Reaction time distribution (bots have flat distributions, humans are log-normal)
    3. Action correlation (bots react identically to identical stimuli)
    4. Input pattern entropy (bots have low entropy)
    5. Fatigue absence (bots don't get tired)
    6. Mouse path analysis (bots move in straight lines)
    7. APS consistency (bots maintain constant APS)

    This layer injects human-like noise to defeat all 7 detection vectors.
    """

    def __init__(self, profile: TimingProfile = None, skill: float = 0.7):
        self._profile = profile or TimingProfile()
        self._skill = skill
        self._session_start = time.time()
        self._action_times: deque = deque(maxlen=1000)
        self._reaction_times: deque = deque(maxlen=200)
        self._current_aps_window: deque = deque(maxlen=60)
        self._fatigue_level: float = 0.0
        self._frustration_level: float = 0.0
        self._death_streak: int = 0
        self._last_action_time: float = 0
        self._micro_pause_due: bool = False

        # Log-normal distribution params for reaction times
        self._rt_mu = math.log(self._profile.mean_reaction_ms)
        self._rt_sigma = self._profile.std_reaction_ms / self._profile.mean_reaction_ms

    def reaction_delay(self, base_ms: float = 0) -> float:
        """Generate a human-realistic reaction delay in milliseconds.

        Uses log-normal distribution (matches empirical human RT data).
        Adjusted for: skill level, fatigue, frustration, time pressure.
        """
        rt = random.lognormvariate(self._rt_mu, self._rt_sigma)

        # Skill: pros react 30% faster
        skill_factor = 1.0 - (self._skill * 0.3)
        rt *= skill_factor

        # Fatigue: reactions slow 5-20% over session
        self._update_fatigue()
        rt *= (1.0 + self._fatigue_level * 0.2)

        # Frustration: inconsistent
        if self._frustration_level > 0.5:
            if random.random() < 0.3:
                rt *= 0.7  # Panic rush
            else:
                rt *= 1.3  # Hesitation
        elif self._frustration_level > 0.2:
            rt *= 1.1

        # Random micro-pause
        if random.random() < self._profile.pause_probability:
            pause = random.uniform(*self._profile.pause_duration_ms)
            rt += pause

        # Minimum human reaction (~100ms for visual stimuli)
        rt = max(rt, 100 + random.gauss(0, 10))

        return rt + base_ms

    def should_act(self) -> bool:
        """Rate limiter: prevent inhuman APS spikes."""
        now = time.time()

        self._current_aps_window.append(now)
        recent = [t for t in self._current_aps_window if now - t < 1.0]
        current_aps = len(recent)

        max_allowed = self._profile.max_aps * self._skill
        if current_aps >= max_allowed:
            return False

        min_gap = 1000.0 / max_allowed / 1000.0
        min_gap += random.gauss(0, min_gap * 0.15)

        if now - self._last_action_time < max(min_gap, 0.02):
            return False

        self._last_action_time = now
        self._action_times.append(now)
        return True

    def jitter_mouse_target(self, target_x: int, target_y: int,
                            target_size: int = 30) -> tuple:
        """Add human-like inaccuracy to mouse targeting (Fitts' Law)."""
        scatter_px = max(3, 30 / max(target_size, 10)) * (1.0 - self._skill * 0.7)
        scatter_px *= (1.0 + self._fatigue_level * 0.3)

        dx = random.gauss(0, scatter_px)
        dy = random.gauss(0, scatter_px)

        return (int(target_x + dx), int(target_y + dy))

    def add_path_noise(self, path: List[tuple]) -> List[tuple]:
        """Add micro-jitter to a mouse movement path."""
        if len(path) < 3:
            return path

        noisy = [path[0]]
        for i in range(1, len(path) - 1):
            x, y = path[i]
            jx = random.gauss(0, 1.2)
            jy = random.gauss(0, 1.2)
            noisy.append((int(x + jx), int(y + jy)))

            # Occasional micro-correction
            if random.random() < 0.05:
                err_x = random.gauss(0, 4)
                err_y = random.gauss(0, 4)
                noisy.append((int(x + err_x), int(y + err_y)))

        noisy.append(path[-1])
        return noisy

    def should_micro_pause(self) -> Optional[float]:
        """Determine if a human-like micro-pause is due. Returns seconds or None."""
        if self._micro_pause_due:
            self._micro_pause_due = False
            return random.uniform(0.1, 0.5)

        now = time.time()
        interval = random.uniform(30, 75)
        recent_actions = [t for t in self._action_times if now - t < interval]
        if len(recent_actions) > 10 and random.random() < 0.02:
            return random.uniform(0.2, 0.8)

        return None

    def on_kill(self):
        """Update state after getting a kill."""
        self._death_streak = 0
        self._frustration_level = max(0, self._frustration_level - 0.1)
        if random.random() < 0.3:
            self._micro_pause_due = True

    def on_death(self):
        """Update state after dying."""
        self._death_streak += 1
        self._frustration_level = min(1.0, self._frustration_level + 0.15)
        self._micro_pause_due = True

    def _update_fatigue(self):
        """Logarithmic fatigue curve based on session duration."""
        session_minutes = (time.time() - self._session_start) / 60
        onset = self._profile.fatigue_onset_min
        if session_minutes < onset:
            self._fatigue_level = 0.0
        else:
            self._fatigue_level = min(0.8, 0.15 * math.log(1 + (session_minutes - onset) / 30))

    def get_stats(self) -> dict:
        now = time.time()
        recent_1s = [t for t in self._action_times if now - t < 1.0]
        recent_10s = [t for t in self._action_times if now - t < 10.0]
        return {
            "session_minutes": round((now - self._session_start) / 60, 1),
            "fatigue_level": round(self._fatigue_level, 3),
            "frustration_level": round(self._frustration_level, 3),
            "death_streak": self._death_streak,
            "current_aps": len(recent_1s),
            "avg_aps_10s": round(len(recent_10s) / 10, 1),
            "total_actions": len(self._action_times),
        }
