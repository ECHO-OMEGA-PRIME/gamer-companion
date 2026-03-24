"""Objective Tracker — Track and pursue game objectives.

Maintains awareness of:
- Primary objectives (plant bomb, capture point, destroy nexus)
- Secondary objectives (economy, map control, vision)
- Time-sensitive objectives (bomb timer, dragon spawn, zone closure)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from loguru import logger


class ObjectivePriority(Enum):
    CRITICAL = "critical"    # Must do NOW (defuse bomb, escape zone)
    HIGH = "high"            # Should do soon (take site, contest dragon)
    MEDIUM = "medium"        # Standard play (map control, economy)
    LOW = "low"              # Nice to have (deep wards, enemy econ)
    BACKGROUND = "background"  # Passive tracking (score, KDA)


class ObjectiveStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Objective:
    """A tracked game objective."""
    obj_id: str
    name: str
    description: str = ""
    priority: ObjectivePriority = ObjectivePriority.MEDIUM
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    deadline: Optional[float] = None  # Absolute time (None = no deadline)
    progress: float = 0.0  # 0.0 to 1.0
    location: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class ObjectiveTracker:
    """Track multiple game objectives with priority and deadlines.

    Manages a priority queue of objectives. Higher priority and sooner
    deadlines bubble to the top. The AI always knows what it should
    be doing next.
    """

    def __init__(self):
        self._objectives: Dict[str, Objective] = {}
        self._completed: List[Objective] = []

    def add(self, objective: Objective):
        """Add or update an objective."""
        self._objectives[objective.obj_id] = objective

    def remove(self, obj_id: str):
        self._objectives.pop(obj_id, None)

    def complete(self, obj_id: str):
        """Mark an objective as completed."""
        obj = self._objectives.get(obj_id)
        if obj:
            obj.status = ObjectiveStatus.COMPLETED
            obj.progress = 1.0
            obj.completed_at = time.time()
            self._completed.append(obj)
            del self._objectives[obj_id]

    def fail(self, obj_id: str):
        """Mark an objective as failed."""
        obj = self._objectives.get(obj_id)
        if obj:
            obj.status = ObjectiveStatus.FAILED
            self._completed.append(obj)
            del self._objectives[obj_id]

    def update_progress(self, obj_id: str, progress: float):
        """Update objective progress (0.0 to 1.0)."""
        obj = self._objectives.get(obj_id)
        if obj:
            obj.progress = max(0.0, min(1.0, progress))
            if obj.progress >= 1.0:
                self.complete(obj_id)

    def get_current(self) -> Optional[Objective]:
        """Get the highest-priority active objective."""
        active = self.get_active()
        return active[0] if active else None

    def get_active(self) -> List[Objective]:
        """Get all active objectives sorted by priority and urgency."""
        now = time.time()

        # Expire objectives past deadline
        expired = []
        for obj_id, obj in self._objectives.items():
            if obj.deadline and now > obj.deadline:
                obj.status = ObjectiveStatus.EXPIRED
                expired.append(obj_id)
        for obj_id in expired:
            obj = self._objectives.pop(obj_id)
            self._completed.append(obj)

        # Sort by priority (critical first), then by urgency (sooner deadline)
        priority_order = {
            ObjectivePriority.CRITICAL: 0,
            ObjectivePriority.HIGH: 1,
            ObjectivePriority.MEDIUM: 2,
            ObjectivePriority.LOW: 3,
            ObjectivePriority.BACKGROUND: 4,
        }

        active = list(self._objectives.values())
        active.sort(key=lambda o: (
            priority_order.get(o.priority, 5),
            o.deadline or float("inf"),
        ))

        return active

    def get_by_priority(
        self, priority: ObjectivePriority,
    ) -> List[Objective]:
        """Get objectives at a specific priority level."""
        return [
            o for o in self._objectives.values()
            if o.priority == priority
        ]

    def clear_round(self):
        """Clear round-scoped objectives (keep match-scoped ones)."""
        round_objs = [
            oid for oid, o in self._objectives.items()
            if o.metadata.get("scope") == "round"
        ]
        for oid in round_objs:
            self.remove(oid)

    @property
    def count(self) -> int:
        return len(self._objectives)

    def get_stats(self) -> dict:
        priority_counts = {}
        for o in self._objectives.values():
            p = o.priority.value
            priority_counts[p] = priority_counts.get(p, 0) + 1
        return {
            "active": len(self._objectives),
            "completed": len([
                c for c in self._completed
                if c.status == ObjectiveStatus.COMPLETED
            ]),
            "failed": len([
                c for c in self._completed
                if c.status == ObjectiveStatus.FAILED
            ]),
            "by_priority": priority_counts,
        }
