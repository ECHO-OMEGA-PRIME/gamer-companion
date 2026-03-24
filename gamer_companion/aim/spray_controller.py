"""Spray Controller — Per-weapon recoil compensation patterns.

Stores spray patterns as (dx, dy) offsets per shot. During a spray,
the aim engine subtracts these offsets to compensate for recoil.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from loguru import logger


@dataclass
class SprayPattern:
    """A weapon's recoil pattern as per-shot offsets."""
    weapon_name: str
    offsets: List[Tuple[float, float]] = field(default_factory=list)
    magazine_size: int = 30
    fire_rate_rpm: float = 600
    description: str = ""

    @property
    def shot_interval_ms(self) -> float:
        """Time between shots in milliseconds."""
        return 60000 / self.fire_rate_rpm if self.fire_rate_rpm > 0 else 100

    @property
    def pattern_length(self) -> int:
        return len(self.offsets)


def _generate_t_pattern(
    mag_size: int, max_vertical: float = 8.0,
    horizontal_drift: float = 2.0, pattern: str = "up_then_side",
) -> List[Tuple[float, float]]:
    """Generate a realistic spray pattern.

    Most FPS recoil: shots 1-10 go up, shots 10-20 drift sideways,
    shots 20+ oscillate.
    """
    offsets = []
    for i in range(mag_size):
        t = i / max(mag_size - 1, 1)

        if pattern == "up_then_side":
            # Vertical: ramps up then plateaus
            dy = max_vertical * min(1.0, t * 3) * (1.0 - t * 0.3)
            # Horizontal: starts centered, drifts after shot 10
            if i < 10:
                dx = 0
            elif i < 20:
                dx = horizontal_drift * math.sin((i - 10) * 0.5)
            else:
                dx = horizontal_drift * math.sin((i - 10) * 0.7) * 1.2
        else:
            dy = max_vertical * t
            dx = horizontal_drift * math.sin(t * math.pi * 2)

        offsets.append((round(dx, 2), round(dy, 2)))

    return offsets


# Pre-computed spray patterns for common weapons
SPRAY_PATTERNS: Dict[str, SprayPattern] = {
    "ak47": SprayPattern(
        weapon_name="ak47",
        offsets=_generate_t_pattern(30, max_vertical=9.0, horizontal_drift=2.5),
        magazine_size=30, fire_rate_rpm=600,
        description="AK-47: Strong upward pull, lateral drift after 10 shots",
    ),
    "m4a4": SprayPattern(
        weapon_name="m4a4",
        offsets=_generate_t_pattern(30, max_vertical=7.0, horizontal_drift=2.0),
        magazine_size=30, fire_rate_rpm=666,
        description="M4A4: Moderate upward pull, tighter than AK",
    ),
    "m4a1s": SprayPattern(
        weapon_name="m4a1s",
        offsets=_generate_t_pattern(25, max_vertical=5.0, horizontal_drift=1.5),
        magazine_size=25, fire_rate_rpm=600,
        description="M4A1-S: Minimal recoil, slight upward",
    ),
    "galil": SprayPattern(
        weapon_name="galil",
        offsets=_generate_t_pattern(35, max_vertical=6.0, horizontal_drift=2.0),
        magazine_size=35, fire_rate_rpm=666,
        description="Galil: Moderate spray, larger magazine",
    ),
    "famas": SprayPattern(
        weapon_name="famas",
        offsets=_generate_t_pattern(25, max_vertical=6.5, horizontal_drift=1.8),
        magazine_size=25, fire_rate_rpm=666,
        description="FAMAS: Burst-friendly, moderate spray",
    ),
}


class SprayController:
    """Manage spray compensation during automatic fire.

    Usage:
        sc = SprayController()
        sc.start_spray("ak47")
        for each shot:
            offset = sc.next_compensation()
            # Apply offset to aim
        sc.stop_spray()
    """

    def __init__(self, custom_patterns: Dict[str, SprayPattern] = None):
        self._patterns = {**SPRAY_PATTERNS}
        if custom_patterns:
            self._patterns.update(custom_patterns)
        self._active_pattern: Optional[SprayPattern] = None
        self._shot_index = 0
        self._spraying = False

    def start_spray(self, weapon: str) -> bool:
        """Start spray compensation for a weapon."""
        pattern = self._patterns.get(weapon.lower())
        if not pattern:
            logger.debug(f"No spray pattern for: {weapon}")
            return False
        self._active_pattern = pattern
        self._shot_index = 0
        self._spraying = True
        return True

    def next_compensation(self) -> Tuple[float, float]:
        """Get the next spray compensation offset (dx, dy).

        Returns (0, 0) if no active spray or past magazine end.
        """
        if not self._spraying or not self._active_pattern:
            return (0.0, 0.0)

        if self._shot_index >= self._active_pattern.pattern_length:
            return (0.0, 0.0)

        offset = self._active_pattern.offsets[self._shot_index]
        self._shot_index += 1
        return offset

    def stop_spray(self):
        """Stop spray compensation (release trigger or switch weapon)."""
        self._spraying = False
        self._shot_index = 0
        self._active_pattern = None

    @property
    def is_spraying(self) -> bool:
        return self._spraying

    @property
    def shots_fired(self) -> int:
        return self._shot_index

    def get_pattern(self, weapon: str) -> Optional[SprayPattern]:
        return self._patterns.get(weapon.lower())

    def list_weapons(self) -> List[str]:
        return list(self._patterns.keys())
