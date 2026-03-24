"""Target Prioritizer — Decide which enemy to shoot first.

Ranks targets by composite threat score considering:
- Distance (closer = higher threat)
- Health (low HP = easy kill)
- Weapon (AWP > AK > pistol)
- Visibility (fully visible > partially)
- Aim angle required (smaller flick = faster kill)
- Is aiming at us (immediate danger)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger


@dataclass
class PrioritizedTarget:
    """A target with computed priority score."""
    entity_id: str
    screen_x: float
    screen_y: float
    distance: float
    health: int
    weapon: str
    is_visible: bool
    is_aiming_at_us: bool
    threat_score: float = 0
    kill_priority: float = 0
    aim_angle_deg: float = 0


# Weapon threat multipliers
WEAPON_THREAT = {
    "awp": 2.0, "scout": 1.5,
    "ak47": 1.3, "m4a4": 1.2, "m4a1s": 1.2,
    "galil": 1.0, "famas": 1.0,
    "mp9": 0.8, "mac10": 0.8, "ump": 0.85,
    "deagle": 1.1, "usp": 0.7, "glock": 0.6, "p250": 0.7,
    "shotgun": 1.4,  # Close range lethal
    "knife": 0.3,
    "unknown": 1.0,
}


class TargetPrioritizer:
    """Rank targets by threat and kill priority.

    Two scoring systems:
    1. Threat Score: Who is most dangerous to ME?
       → Prioritize neutralizing highest threats first (defensive)
    2. Kill Priority: Who is easiest to kill?
       → Prioritize easy frags for momentum (aggressive)

    The active scoring depends on playstyle aggression setting.
    """

    def __init__(
        self,
        screen_center: Tuple[int, int] = (960, 540),
        aggression: float = 0.5,
    ):
        self._screen_center = screen_center
        self.aggression = aggression  # 0 = threat-first, 1 = kill-first

    def prioritize(
        self, targets: List[PrioritizedTarget],
    ) -> List[PrioritizedTarget]:
        """Score and rank all targets. Returns sorted list (best first)."""
        if not targets:
            return []

        cx, cy = self._screen_center

        for t in targets:
            # Aim angle from crosshair
            dx = t.screen_x - cx
            dy = t.screen_y - cy
            t.aim_angle_deg = math.degrees(
                math.atan2(math.hypot(dx, dy), 1000)
            )

            # Threat score (who's dangerous)
            threat = 0.0

            # Distance factor (closer = more threatening)
            if t.distance > 0:
                dist_factor = max(0.1, 1.0 - t.distance / 3000)
            else:
                dist_factor = 0.5
            threat += dist_factor * 0.3

            # Weapon factor
            weapon_mult = WEAPON_THREAT.get(t.weapon.lower(), 1.0)
            threat += weapon_mult * 0.2

            # Aiming at us = immediate danger
            if t.is_aiming_at_us:
                threat += 0.3

            # Visibility factor
            if t.is_visible:
                threat += 0.1

            t.threat_score = min(1.0, threat)

            # Kill priority (who's easiest to kill)
            kill = 0.0

            # Low HP = easy kill
            hp_factor = 1.0 - (t.health / 100)
            kill += hp_factor * 0.35

            # Small aim angle = faster kill
            angle_factor = max(0.0, 1.0 - t.aim_angle_deg / 45)
            kill += angle_factor * 0.3

            # Visible = can shoot now
            if t.is_visible:
                kill += 0.2

            # Close = easier to hit
            kill += dist_factor * 0.15

            t.kill_priority = min(1.0, kill)

        # Combined score based on aggression
        def combined_score(t: PrioritizedTarget) -> float:
            return (
                t.threat_score * (1 - self.aggression)
                + t.kill_priority * self.aggression
            )

        targets.sort(key=combined_score, reverse=True)

        return targets

    def get_best(
        self, targets: List[PrioritizedTarget],
    ) -> Optional[PrioritizedTarget]:
        """Get the single best target to engage."""
        ranked = self.prioritize(targets)
        return ranked[0] if ranked else None
