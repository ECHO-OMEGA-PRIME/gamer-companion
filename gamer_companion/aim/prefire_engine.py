"""Prefire Engine — Pre-aim common angles from learned positions.

Manages a database of common enemy positions per map/zone and
pre-aims at them when approaching. This is what good players do
naturally — aim at head-height at the angle enemies are most
likely to appear.

Learned from:
- Pro demo analysis (preset data)
- Self-play observation (what angles get kills)
- Death analysis (where enemies were when we died)
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger


@dataclass
class PrefireAngle:
    """A prefire position on a map."""
    angle_id: str
    map_id: str
    zone: str                 # "a_site", "mid", "b_long"
    x: float                  # Screen position to pre-aim
    y: float
    hit_rate: float = 0.0    # Historical hit rate when pre-aiming here
    frequency: float = 0.0   # How often enemies are seen here (0-1)
    kills: int = 0
    whiffs: int = 0
    total_checks: int = 0
    priority: float = 0.5    # Higher = check this angle first
    head_height_y: float = 0  # Head-level Y coordinate


@dataclass
class PrefireSequence:
    """An ordered sequence of angles to check when entering a zone."""
    zone: str
    map_id: str
    angles: List[PrefireAngle]
    current_index: int = 0


class PrefireEngine:
    """Pre-aim common enemy positions.

    Usage flow:
    1. Player enters a zone (e.g., "a_site")
    2. Engine returns ordered list of angles to check
    3. AI pre-aims each angle in priority order
    4. After clearing, reports results (hit/whiff)
    5. Over time, priorities adjust based on actual enemy positions

    The engine tracks per-map, per-zone angle data and
    adapts to opponent tendencies.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._angles: Dict[str, Dict[str, List[PrefireAngle]]] = {}  # map → zone → angles
        self._persist_path = Path(persist_path) if persist_path else None
        self._active_sequence: Optional[PrefireSequence] = None
        self._angle_counter = 0
        self._total_checks = 0
        self._total_hits = 0
        self._load()

    def add_angle(
        self,
        map_id: str,
        zone: str,
        x: float,
        y: float,
        priority: float = 0.5,
        head_height_y: float = 0,
    ) -> PrefireAngle:
        """Add a prefire angle to the database."""
        self._angle_counter += 1
        angle = PrefireAngle(
            angle_id=f"pf_{self._angle_counter}",
            map_id=map_id,
            zone=zone,
            x=x, y=y,
            priority=priority,
            head_height_y=head_height_y or y,
        )

        if map_id not in self._angles:
            self._angles[map_id] = {}
        if zone not in self._angles[map_id]:
            self._angles[map_id][zone] = []
        self._angles[map_id][zone].append(angle)

        return angle

    def get_angles(self, map_id: str, zone: str) -> List[PrefireAngle]:
        """Get prefire angles for a zone, sorted by priority."""
        angles = self._angles.get(map_id, {}).get(zone, [])
        return sorted(angles, key=lambda a: a.priority, reverse=True)

    def start_sequence(self, map_id: str, zone: str) -> Optional[PrefireSequence]:
        """Start a prefire sequence for entering a zone."""
        angles = self.get_angles(map_id, zone)
        if not angles:
            return None

        self._active_sequence = PrefireSequence(
            zone=zone,
            map_id=map_id,
            angles=angles,
            current_index=0,
        )
        return self._active_sequence

    def get_next_angle(self) -> Optional[PrefireAngle]:
        """Get the next angle to check in the active sequence."""
        if not self._active_sequence:
            return None
        seq = self._active_sequence
        if seq.current_index >= len(seq.angles):
            self._active_sequence = None
            return None
        return seq.angles[seq.current_index]

    def advance(self, hit: bool = False):
        """Mark current angle as checked and advance."""
        if not self._active_sequence:
            return
        seq = self._active_sequence
        if seq.current_index < len(seq.angles):
            angle = seq.angles[seq.current_index]
            angle.total_checks += 1
            self._total_checks += 1
            if hit:
                angle.kills += 1
                self._total_hits += 1
            else:
                angle.whiffs += 1

            # Update hit rate
            total = angle.kills + angle.whiffs
            angle.hit_rate = angle.kills / max(1, total)

            # Adapt priority based on hit rate
            angle.priority = 0.3 + angle.hit_rate * 0.7

            seq.current_index += 1
            if seq.current_index >= len(seq.angles):
                self._active_sequence = None
                self._save()

    def record_enemy_position(self, map_id: str, zone: str, x: float, y: float):
        """Record where an enemy was actually seen.

        If close to an existing angle, boost that angle's frequency.
        If no nearby angle exists, add a new one.
        """
        angles = self._angles.get(map_id, {}).get(zone, [])
        import math

        closest = None
        closest_dist = float('inf')
        for angle in angles:
            dist = math.hypot(angle.x - x, angle.y - y)
            if dist < closest_dist:
                closest_dist = dist
                closest = angle

        if closest and closest_dist < 50:
            # Boost existing angle
            closest.frequency = min(1.0, closest.frequency + 0.05)
            closest.priority = min(1.0, closest.priority + 0.02)
        else:
            # New angle discovered
            self.add_angle(map_id, zone, x, y, priority=0.4, head_height_y=y)

    def decay_priorities(self, factor: float = 0.98):
        """Slowly decay all priorities (prevents stale data)."""
        for map_angles in self._angles.values():
            for zone_angles in map_angles.values():
                for angle in zone_angles:
                    angle.priority *= factor
                    angle.frequency *= factor

    def _save(self):
        if not self._persist_path:
            return
        data = {}
        for map_id, zones in self._angles.items():
            data[map_id] = {}
            for zone, angles in zones.items():
                data[map_id][zone] = [
                    {
                        "x": a.x, "y": a.y,
                        "kills": a.kills, "whiffs": a.whiffs,
                        "priority": round(a.priority, 3),
                        "frequency": round(a.frequency, 3),
                        "head_y": a.head_height_y,
                    }
                    for a in angles
                ]
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save prefire data: {e}")

    def _load(self):
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for map_id, zones in data.items():
                for zone, angles in zones.items():
                    for a in angles:
                        angle = self.add_angle(
                            map_id, zone,
                            a["x"], a["y"],
                            priority=a.get("priority", 0.5),
                            head_height_y=a.get("head_y", a["y"]),
                        )
                        angle.kills = a.get("kills", 0)
                        angle.whiffs = a.get("whiffs", 0)
                        angle.frequency = a.get("frequency", 0)
                        total = angle.kills + angle.whiffs
                        angle.hit_rate = angle.kills / max(1, total)
        except Exception as e:
            logger.warning(f"Failed to load prefire data: {e}")

    @property
    def is_sequencing(self) -> bool:
        return self._active_sequence is not None

    def get_stats(self) -> dict:
        total_angles = sum(
            len(angles)
            for zones in self._angles.values()
            for angles in zones.values()
        )
        return {
            "maps": len(self._angles),
            "total_angles": total_angles,
            "total_checks": self._total_checks,
            "total_hits": self._total_hits,
            "hit_rate": round(self._total_hits / max(1, self._total_checks), 3),
            "active_sequence": self._active_sequence is not None,
        }
