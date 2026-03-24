"""Movement Engine — Pathfinding and in-game movement planning.

Plans and executes movement through game environments:
- A* pathfinding on grid/nav-mesh representations
- Waypoint-based route planning
- Threat-weighted pathfinding (avoid danger zones)
- Movement optimization (shortest path vs safest path)
- Dynamic re-routing when threats change
"""

from __future__ import annotations
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from loguru import logger


@dataclass
class Waypoint:
    """A point in the game world."""
    name: str
    x: float
    y: float
    z: float = 0.0
    zone: str = ""            # "a_site", "mid", "b_long", etc.
    is_cover: bool = False    # Can take cover here
    is_choke: bool = False    # Chokepoint (high risk)
    danger_level: float = 0.0  # 0=safe, 1=very dangerous
    connections: List[str] = field(default_factory=list)  # Connected waypoint names


@dataclass
class MovementPlan:
    """A planned movement route through waypoints."""
    plan_id: str
    waypoints: List[Waypoint]
    total_distance: float
    estimated_time_s: float
    danger_score: float       # Average danger along route
    has_cover: bool           # Route has cover positions
    created_at: float = field(default_factory=time.time)

    @property
    def waypoint_names(self) -> List[str]:
        return [w.name for w in self.waypoints]


class MovementEngine:
    """Plan and optimize in-game movement.

    Features:
    - A* pathfinding on waypoint graph
    - Threat-weighted routing (prefers safe paths)
    - Multiple route options (fast vs safe)
    - Dynamic re-routing when danger zones change
    - Cover-to-cover movement planning

    Works with map data from game profiles or auto-discovered waypoints.
    """

    def __init__(self, move_speed: float = 250.0):
        self._waypoints: Dict[str, Waypoint] = {}
        self._move_speed = move_speed  # Units per second
        self._plans: List[MovementPlan] = []
        self._plan_counter = 0

    def add_waypoint(self, waypoint: Waypoint):
        """Add a waypoint to the navigation graph."""
        self._waypoints[waypoint.name] = waypoint

    def add_waypoints(self, waypoints: List[Waypoint]):
        """Add multiple waypoints."""
        for wp in waypoints:
            self._waypoints[wp.name] = wp

    def connect(self, name_a: str, name_b: str):
        """Connect two waypoints bidirectionally."""
        a = self._waypoints.get(name_a)
        b = self._waypoints.get(name_b)
        if not a or not b:
            return
        if name_b not in a.connections:
            a.connections.append(name_b)
        if name_a not in b.connections:
            b.connections.append(name_a)

    def _distance(self, a: Waypoint, b: Waypoint) -> float:
        """Euclidean distance between waypoints."""
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        )

    def find_path(
        self,
        start_name: str,
        end_name: str,
        danger_weight: float = 0.5,
    ) -> Optional[MovementPlan]:
        """Find path using A* with danger-weighted costs.

        Args:
            start_name: Starting waypoint name
            end_name: Destination waypoint name
            danger_weight: 0.0 = shortest path, 1.0 = safest path
        """
        start = self._waypoints.get(start_name)
        end = self._waypoints.get(end_name)
        if not start or not end:
            return None

        # A* search
        open_set: List[Tuple[float, str]] = [(0, start_name)]
        came_from: Dict[str, str] = {}
        g_score: Dict[str, float] = {start_name: 0}
        f_score: Dict[str, float] = {start_name: self._distance(start, end)}

        while open_set:
            _, current_name = heapq.heappop(open_set)

            if current_name == end_name:
                # Reconstruct path
                path = [current_name]
                while current_name in came_from:
                    current_name = came_from[current_name]
                    path.append(current_name)
                path.reverse()

                return self._build_plan(path, danger_weight)

            current = self._waypoints[current_name]
            for neighbor_name in current.connections:
                neighbor = self._waypoints.get(neighbor_name)
                if not neighbor:
                    continue

                dist = self._distance(current, neighbor)
                # Cost = distance + danger penalty
                danger_cost = neighbor.danger_level * danger_weight * dist
                choke_cost = 50.0 if neighbor.is_choke and danger_weight > 0.3 else 0.0
                cost = dist + danger_cost + choke_cost

                tentative_g = g_score[current_name] + cost
                if tentative_g < g_score.get(neighbor_name, float('inf')):
                    came_from[neighbor_name] = current_name
                    g_score[neighbor_name] = tentative_g
                    f_score[neighbor_name] = tentative_g + self._distance(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor_name], neighbor_name))

        return None  # No path found

    def _build_plan(self, path_names: List[str], danger_weight: float) -> MovementPlan:
        """Build a MovementPlan from waypoint names."""
        waypoints = [self._waypoints[n] for n in path_names]

        total_dist = 0.0
        for i in range(len(waypoints) - 1):
            total_dist += self._distance(waypoints[i], waypoints[i + 1])

        avg_danger = sum(w.danger_level for w in waypoints) / max(1, len(waypoints))
        has_cover = any(w.is_cover for w in waypoints[1:-1])  # Exclude start/end

        self._plan_counter += 1
        plan = MovementPlan(
            plan_id=f"mv_{self._plan_counter}",
            waypoints=waypoints,
            total_distance=round(total_dist, 1),
            estimated_time_s=round(total_dist / self._move_speed, 2),
            danger_score=round(avg_danger, 3),
            has_cover=has_cover,
        )

        self._plans.append(plan)
        return plan

    def find_safest_path(self, start: str, end: str) -> Optional[MovementPlan]:
        """Find the safest path (maximum danger avoidance)."""
        return self.find_path(start, end, danger_weight=1.0)

    def find_fastest_path(self, start: str, end: str) -> Optional[MovementPlan]:
        """Find the fastest path (ignoring danger)."""
        return self.find_path(start, end, danger_weight=0.0)

    def compare_routes(
        self, start: str, end: str,
    ) -> List[MovementPlan]:
        """Compare fastest vs safest routes."""
        plans = []
        for dw in [0.0, 0.3, 0.5, 0.7, 1.0]:
            plan = self.find_path(start, end, danger_weight=dw)
            if plan:
                plans.append(plan)
        return plans

    def update_danger(self, waypoint_name: str, danger_level: float):
        """Update danger level for a waypoint (enemy spotted, etc.)."""
        wp = self._waypoints.get(waypoint_name)
        if wp:
            wp.danger_level = max(0.0, min(1.0, danger_level))

    def find_cover(self, near: str, max_distance: float = 500) -> List[Waypoint]:
        """Find nearby cover positions."""
        origin = self._waypoints.get(near)
        if not origin:
            return []

        cover = []
        for wp in self._waypoints.values():
            if wp.is_cover and wp.name != near:
                dist = self._distance(origin, wp)
                if dist <= max_distance:
                    cover.append(wp)

        cover.sort(key=lambda w: self._distance(origin, w))
        return cover

    def get_stats(self) -> dict:
        return {
            "waypoints": len(self._waypoints),
            "connections": sum(len(w.connections) for w in self._waypoints.values()) // 2,
            "plans_computed": len(self._plans),
            "cover_positions": sum(1 for w in self._waypoints.values() if w.is_cover),
            "choke_points": sum(1 for w in self._waypoints.values() if w.is_choke),
        }
