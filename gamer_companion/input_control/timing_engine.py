"""Timing Engine — Frame-perfect vs human-realistic timing control.

Controls when actions execute relative to game frames:
- Frame-perfect mode: Execute at exact frame boundaries (speedrun/TAS)
- Human mode: Add natural timing variance (anti-detection)
- Rhythm mode: Maintain consistent tempo for combos
- Reactive mode: Time actions relative to stimuli
"""

from __future__ import annotations
import time
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from loguru import logger


@dataclass
class TimingProfile:
    """Timing parameters for different modes."""
    name: str
    base_delay_ms: float = 0.0       # Base delay before action
    jitter_ms: float = 0.0           # Random variance added
    min_interval_ms: float = 16.0    # Minimum between actions (~60fps)
    reaction_floor_ms: float = 100.0 # Minimum human reaction time
    rhythm_bpm: float = 0.0          # For rhythm mode (beats per minute)
    burst_window_ms: float = 0.0     # Window for burst inputs


TIMING_PROFILES: Dict[str, TimingProfile] = {
    "frame_perfect": TimingProfile(
        name="frame_perfect",
        base_delay_ms=0,
        jitter_ms=0,
        min_interval_ms=16.67,  # 60fps
    ),
    "human_casual": TimingProfile(
        name="human_casual",
        base_delay_ms=50,
        jitter_ms=30,
        min_interval_ms=33,
        reaction_floor_ms=200,
    ),
    "human_focused": TimingProfile(
        name="human_focused",
        base_delay_ms=20,
        jitter_ms=15,
        min_interval_ms=16,
        reaction_floor_ms=140,
    ),
    "pro_player": TimingProfile(
        name="pro_player",
        base_delay_ms=8,
        jitter_ms=8,
        min_interval_ms=16,
        reaction_floor_ms=120,
    ),
    "rhythm_120bpm": TimingProfile(
        name="rhythm_120bpm",
        base_delay_ms=0,
        jitter_ms=10,
        min_interval_ms=16,
        rhythm_bpm=120,
    ),
}


@dataclass
class ScheduledAction:
    """An action scheduled for future execution."""
    action_id: str
    execute_at: float              # Absolute time to execute
    callback: Optional[Callable] = None
    args: tuple = ()
    priority: int = 0             # Lower = higher priority
    executed: bool = False
    cancelled: bool = False


class TimingEngine:
    """Control action timing with frame-awareness and human realism.

    Features:
    - Multiple timing profiles (frame-perfect, human, rhythm)
    - Action scheduling with priority queue
    - Frame-aligned execution for consistency
    - Combo timing with rhythm enforcement
    - Cooldown tracking per action type
    """

    def __init__(self, profile_name: str = "human_focused"):
        self._profile = TIMING_PROFILES.get(profile_name, TIMING_PROFILES["human_focused"])
        self._scheduled: List[ScheduledAction] = []
        self._cooldowns: Dict[str, float] = {}  # action_type → next_allowed_time
        self._last_action_time: float = 0
        self._combo_start: float = 0
        self._combo_beat_index: int = 0
        self._frame_duration_ms: float = 16.67  # Default 60fps
        self._action_counter: int = 0

    def set_profile(self, profile_name: str) -> bool:
        """Switch timing profile."""
        profile = TIMING_PROFILES.get(profile_name)
        if not profile:
            return False
        self._profile = profile
        return True

    def set_fps(self, fps: float):
        """Set game framerate for frame-aligned timing."""
        self._frame_duration_ms = 1000.0 / max(1, fps)

    def get_action_delay_ms(self) -> float:
        """Get the delay before the next action should execute."""
        delay = self._profile.base_delay_ms

        # Add jitter (gaussian, clamped to positive)
        if self._profile.jitter_ms > 0:
            jitter = random.gauss(0, self._profile.jitter_ms / 2)
            delay += abs(jitter)

        # Enforce minimum interval
        elapsed = (time.time() - self._last_action_time) * 1000
        remaining_interval = self._profile.min_interval_ms - elapsed
        if remaining_interval > 0:
            delay = max(delay, remaining_interval)

        return round(max(0, delay), 1)

    def get_reaction_time_ms(self, stimulus_urgency: float = 0.5) -> float:
        """Get a realistic reaction time in milliseconds.

        Args:
            stimulus_urgency: 0.0 (relaxed) to 1.0 (urgent/reflex)

        Returns:
            Reaction time drawn from log-normal distribution.
        """
        floor = self._profile.reaction_floor_ms

        # Log-normal distribution: most reactions near the mean, some slow outliers
        # Urgency reduces the mean
        mean_ms = floor + (1.0 - stimulus_urgency) * 80
        sigma = 0.2 + (1.0 - stimulus_urgency) * 0.3

        rt = random.lognormvariate(math.log(mean_ms), sigma)
        return round(max(floor, rt), 1)

    def get_combo_timing_ms(self, step_index: int) -> float:
        """Get timing for a combo step based on rhythm.

        For rhythm mode, calculates the ideal time for each beat.
        For human mode, adds natural variance to combo timing.
        """
        if self._profile.rhythm_bpm > 0:
            # Beat interval in ms
            beat_ms = 60000.0 / self._profile.rhythm_bpm
            ideal_ms = step_index * beat_ms

            # Add human-like swing/groove
            swing = random.gauss(0, beat_ms * 0.05)
            return round(ideal_ms + swing, 1)

        # Non-rhythm: consistent but slightly variable spacing
        base = 50.0 + step_index * 30
        jitter = random.gauss(0, 10)
        return round(max(0, base + jitter), 1)

    def schedule_action(
        self,
        delay_ms: float,
        callback: Callable = None,
        args: tuple = (),
        priority: int = 0,
    ) -> str:
        """Schedule an action for future execution."""
        self._action_counter += 1
        action_id = f"ta_{self._action_counter}"

        action = ScheduledAction(
            action_id=action_id,
            execute_at=time.time() + delay_ms / 1000.0,
            callback=callback,
            args=args,
            priority=priority,
        )
        self._scheduled.append(action)
        self._scheduled.sort(key=lambda a: (a.execute_at, a.priority))

        return action_id

    def cancel_action(self, action_id: str) -> bool:
        """Cancel a scheduled action."""
        for action in self._scheduled:
            if action.action_id == action_id and not action.executed:
                action.cancelled = True
                return True
        return False

    def get_ready_actions(self) -> List[ScheduledAction]:
        """Get actions that are ready to execute now."""
        now = time.time()
        ready = []

        for action in self._scheduled:
            if action.cancelled or action.executed:
                continue
            if action.execute_at <= now:
                action.executed = True
                ready.append(action)

        # Clean up executed/cancelled
        self._scheduled = [
            a for a in self._scheduled
            if not a.executed and not a.cancelled
        ]

        self._last_action_time = now if ready else self._last_action_time
        return ready

    def set_cooldown(self, action_type: str, cooldown_ms: float):
        """Set a cooldown for an action type."""
        self._cooldowns[action_type] = time.time() + cooldown_ms / 1000.0

    def is_on_cooldown(self, action_type: str) -> bool:
        """Check if an action type is on cooldown."""
        return time.time() < self._cooldowns.get(action_type, 0)

    def get_cooldown_remaining_ms(self, action_type: str) -> float:
        """Get remaining cooldown in ms."""
        remaining = self._cooldowns.get(action_type, 0) - time.time()
        return round(max(0, remaining * 1000), 1)

    def align_to_frame(self, time_ms: float) -> float:
        """Align a time to the nearest frame boundary."""
        frame_ms = self._frame_duration_ms
        return round(math.ceil(time_ms / frame_ms) * frame_ms, 1)

    def mark_action(self):
        """Mark that an action was just executed (for interval tracking)."""
        self._last_action_time = time.time()

    @property
    def profile(self) -> TimingProfile:
        return self._profile

    def list_profiles(self) -> List[str]:
        return list(TIMING_PROFILES.keys())

    def get_stats(self) -> dict:
        return {
            "profile": self._profile.name,
            "pending_actions": len([a for a in self._scheduled if not a.executed and not a.cancelled]),
            "active_cooldowns": sum(1 for v in self._cooldowns.values() if v > time.time()),
            "frame_duration_ms": round(self._frame_duration_ms, 2),
        }
