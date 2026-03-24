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
    # --- Valorant ---
    "vandal": SprayPattern(
        weapon_name="vandal",
        offsets=_generate_t_pattern(25, max_vertical=8.5, horizontal_drift=2.8),
        magazine_size=25, fire_rate_rpm=585,
        description="Vandal: High vertical pull, wide lateral drift",
    ),
    "phantom": SprayPattern(
        weapon_name="phantom",
        offsets=_generate_t_pattern(30, max_vertical=6.0, horizontal_drift=2.0),
        magazine_size=30, fire_rate_rpm=660,
        description="Phantom: Moderate spray, tighter than Vandal",
    ),
    "spectre": SprayPattern(
        weapon_name="spectre",
        offsets=_generate_t_pattern(30, max_vertical=4.5, horizontal_drift=1.5),
        magazine_size=30, fire_rate_rpm=780,
        description="Spectre: Fast SMG, manageable recoil",
    ),
    # --- Call of Duty ---
    "cod_m4": SprayPattern(
        weapon_name="cod_m4",
        offsets=_generate_t_pattern(30, max_vertical=5.5, horizontal_drift=1.8),
        magazine_size=30, fire_rate_rpm=750,
        description="CoD M4: Low vertical, minimal lateral",
    ),
    "cod_ak47": SprayPattern(
        weapon_name="cod_ak47",
        offsets=_generate_t_pattern(30, max_vertical=7.0, horizontal_drift=2.2),
        magazine_size=30, fire_rate_rpm=682,
        description="CoD AK-47: Moderate climb, lateral bounce",
    ),
    "cod_mp5": SprayPattern(
        weapon_name="cod_mp5",
        offsets=_generate_t_pattern(30, max_vertical=3.5, horizontal_drift=1.0),
        magazine_size=30, fire_rate_rpm=833,
        description="CoD MP5: Very low recoil, tight grouping",
    ),
    "cod_kastov762": SprayPattern(
        weapon_name="cod_kastov762",
        offsets=_generate_t_pattern(30, max_vertical=7.5, horizontal_drift=2.5),
        magazine_size=30, fire_rate_rpm=660,
        description="CoD Kastov 762: Heavy pull, strong drift",
    ),
    "cod_mcw": SprayPattern(
        weapon_name="cod_mcw",
        offsets=_generate_t_pattern(30, max_vertical=5.0, horizontal_drift=1.5),
        magazine_size=30, fire_rate_rpm=720,
        description="CoD MCW (ACR): Very manageable recoil",
    ),
    "cod_striker": SprayPattern(
        weapon_name="cod_striker",
        offsets=_generate_t_pattern(30, max_vertical=3.0, horizontal_drift=0.8),
        magazine_size=30, fire_rate_rpm=900,
        description="CoD Striker SMG: Laser beam, minimal recoil",
    ),
    "cod_holger556": SprayPattern(
        weapon_name="cod_holger556",
        offsets=_generate_t_pattern(30, max_vertical=4.5, horizontal_drift=1.2),
        magazine_size=30, fire_rate_rpm=750,
        description="CoD Holger 556: Balanced AR, low drift",
    ),
    "cod_svamm_sniper": SprayPattern(
        weapon_name="cod_svamm_sniper",
        offsets=_generate_t_pattern(5, max_vertical=12.0, horizontal_drift=0.5),
        magazine_size=5, fire_rate_rpm=55,
        description="CoD SVA 545 Sniper: One-shot, massive vertical kick",
    ),
    # --- Apex Legends ---
    "r301": SprayPattern(
        weapon_name="r301",
        offsets=_generate_t_pattern(28, max_vertical=5.0, horizontal_drift=1.5),
        magazine_size=28, fire_rate_rpm=810,
        description="R-301 Carbine: Easiest AR spray in Apex",
    ),
    "flatline": SprayPattern(
        weapon_name="flatline",
        offsets=_generate_t_pattern(30, max_vertical=7.0, horizontal_drift=3.0),
        magazine_size=30, fire_rate_rpm=600,
        description="VK-47 Flatline: Heavy horizontal oscillation",
    ),
    "r99": SprayPattern(
        weapon_name="r99",
        offsets=_generate_t_pattern(27, max_vertical=6.0, horizontal_drift=2.0),
        magazine_size=27, fire_rate_rpm=1080,
        description="R-99: Very fast fire, aggressive vertical pull",
    ),
    "volt": SprayPattern(
        weapon_name="volt",
        offsets=_generate_t_pattern(28, max_vertical=3.5, horizontal_drift=1.0),
        magazine_size=28, fire_rate_rpm=720,
        description="Volt SMG: Minimal recoil, easy to control",
    ),
    "havoc": SprayPattern(
        weapon_name="havoc",
        offsets=_generate_t_pattern(32, max_vertical=8.0, horizontal_drift=2.5),
        magazine_size=32, fire_rate_rpm=672,
        description="Havoc: High recoil after charge-up delay",
    ),
    # --- Rainbow Six Siege ---
    "r6_l85a2": SprayPattern(
        weapon_name="r6_l85a2",
        offsets=_generate_t_pattern(30, max_vertical=5.0, horizontal_drift=1.2),
        magazine_size=30, fire_rate_rpm=670,
        description="R6 L85A2: Easy AR, good for beginners",
    ),
    "r6_f2": SprayPattern(
        weapon_name="r6_f2",
        offsets=_generate_t_pattern(30, max_vertical=9.0, horizontal_drift=3.5),
        magazine_size=30, fire_rate_rpm=980,
        description="R6 F2: Very fast, extreme vertical recoil",
    ),
    "r6_smg11": SprayPattern(
        weapon_name="r6_smg11",
        offsets=_generate_t_pattern(17, max_vertical=10.0, horizontal_drift=4.0),
        magazine_size=17, fire_rate_rpm=1270,
        description="R6 SMG-11: Highest ROF, brutal recoil",
    ),
    # --- PUBG ---
    "pubg_m416": SprayPattern(
        weapon_name="pubg_m416",
        offsets=_generate_t_pattern(30, max_vertical=6.0, horizontal_drift=2.0),
        magazine_size=30, fire_rate_rpm=680,
        description="PUBG M416: Versatile, moderate recoil with grip",
    ),
    "pubg_akm": SprayPattern(
        weapon_name="pubg_akm",
        offsets=_generate_t_pattern(30, max_vertical=9.5, horizontal_drift=3.0),
        magazine_size=30, fire_rate_rpm=600,
        description="PUBG AKM: Hard-hitting but wild spray",
    ),
    "pubg_ump45": SprayPattern(
        weapon_name="pubg_ump45",
        offsets=_generate_t_pattern(25, max_vertical=3.0, horizontal_drift=0.8),
        magazine_size=25, fire_rate_rpm=650,
        description="PUBG UMP45: Very stable SMG",
    ),
    # --- Rust ---
    "rust_ak47": SprayPattern(
        weapon_name="rust_ak47",
        offsets=_generate_t_pattern(30, max_vertical=10.0, horizontal_drift=4.0,
                                     pattern="up_then_side"),
        magazine_size=30, fire_rate_rpm=450,
        description="Rust AK-47: Notorious S-pattern recoil",
    ),
    "rust_lr300": SprayPattern(
        weapon_name="rust_lr300",
        offsets=_generate_t_pattern(30, max_vertical=6.0, horizontal_drift=2.0),
        magazine_size=30, fire_rate_rpm=500,
        description="Rust LR-300: Easier alternative to AK",
    ),
    "rust_mp5": SprayPattern(
        weapon_name="rust_mp5",
        offsets=_generate_t_pattern(30, max_vertical=4.0, horizontal_drift=1.5),
        magazine_size=30, fire_rate_rpm=600,
        description="Rust MP5A4: Low recoil close-range",
    ),
    # --- Battlefield ---
    "bf_m5a3": SprayPattern(
        weapon_name="bf_m5a3",
        offsets=_generate_t_pattern(30, max_vertical=5.0, horizontal_drift=1.5),
        magazine_size=30, fire_rate_rpm=750,
        description="BF2042 M5A3: Standard AR, moderate recoil",
    ),
    # --- Halo ---
    "halo_ar": SprayPattern(
        weapon_name="halo_ar",
        offsets=_generate_t_pattern(36, max_vertical=3.0, horizontal_drift=2.5),
        magazine_size=36, fire_rate_rpm=600,
        description="Halo MA40 AR: Wide spread, low vertical",
    ),
    # --- Tarkov ---
    "tarkov_m4a1": SprayPattern(
        weapon_name="tarkov_m4a1",
        offsets=_generate_t_pattern(30, max_vertical=6.5, horizontal_drift=2.0),
        magazine_size=30, fire_rate_rpm=800,
        description="Tarkov M4A1: Settles after initial kick",
    ),
    "tarkov_ak74": SprayPattern(
        weapon_name="tarkov_ak74",
        offsets=_generate_t_pattern(30, max_vertical=7.5, horizontal_drift=2.5),
        magazine_size=30, fire_rate_rpm=650,
        description="Tarkov AK-74: Classic upward then side drift",
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
