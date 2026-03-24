"""Resource Manager — In-game resource optimization.

Manages all in-game resources and makes optimal allocation decisions:
- Economy (money, gold, credits)
- Ammunition / ability charges
- Health / shield / armor
- Cooldowns
- Inventory slots
- Team resource pooling

Provides buy recommendations, resource tracking, and efficiency scoring.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger


@dataclass
class Resource:
    """A tracked in-game resource."""
    name: str
    current: float
    maximum: float
    regen_rate: float = 0.0   # Per second
    cost: float = 0.0         # Purchase cost
    priority: float = 0.5     # Importance 0-1
    category: str = ""        # "economy", "health", "ammo", "ability", "item"
    cooldown_s: float = 0.0   # Remaining cooldown
    last_updated: float = field(default_factory=time.time)

    @property
    def pct(self) -> float:
        return round(self.current / max(1, self.maximum) * 100, 1)

    @property
    def is_full(self) -> bool:
        return self.current >= self.maximum

    @property
    def is_empty(self) -> bool:
        return self.current <= 0

    @property
    def is_low(self) -> bool:
        return self.pct < 25


@dataclass
class BuyOption:
    """A purchasable item/upgrade."""
    name: str
    cost: float
    category: str             # "weapon", "armor", "utility", "consumable"
    value_score: float = 0.5  # How valuable is this purchase (0-1)
    priority: int = 0         # Lower = buy first
    requires: List[str] = field(default_factory=list)  # Prerequisites


@dataclass
class BuyRecommendation:
    """A recommended purchase plan."""
    items: List[BuyOption]
    total_cost: float
    remaining_money: float
    value_score: float
    reason: str


class ResourceManager:
    """Track and optimize in-game resource allocation.

    Features:
    - Real-time resource tracking with regen modeling
    - Optimal buy order computation (knapsack-style)
    - Team economy coordination
    - Resource efficiency scoring
    - Cooldown tracking
    - Predictive resource modeling
    """

    def __init__(self):
        self._resources: Dict[str, Resource] = {}
        self._buy_options: Dict[str, BuyOption] = {}
        self._history: List[Dict[str, float]] = []
        self._team_resources: Dict[str, Dict[str, float]] = {}  # player → {resource → value}

    def track(self, name: str, current: float, maximum: float, **kwargs) -> Resource:
        """Start tracking a resource."""
        resource = Resource(
            name=name,
            current=current,
            maximum=maximum,
            **kwargs,
        )
        self._resources[name] = resource
        return resource

    def update(self, name: str, current: float) -> Optional[Resource]:
        """Update a resource's current value."""
        r = self._resources.get(name)
        if not r:
            return None
        r.current = max(0, min(r.maximum, current))
        r.last_updated = time.time()
        return r

    def get(self, name: str) -> Optional[Resource]:
        return self._resources.get(name)

    def get_value(self, name: str) -> float:
        """Get current value of a resource."""
        r = self._resources.get(name)
        return r.current if r else 0.0

    def get_pct(self, name: str) -> float:
        """Get percentage of a resource."""
        r = self._resources.get(name)
        return r.pct if r else 0.0

    def apply_regen(self, dt_seconds: float = 1.0):
        """Apply regeneration to all resources."""
        for r in self._resources.values():
            if r.regen_rate > 0 and not r.is_full:
                r.current = min(r.maximum, r.current + r.regen_rate * dt_seconds)
                r.last_updated = time.time()

    def apply_cooldowns(self, dt_seconds: float = 1.0):
        """Reduce cooldowns."""
        for r in self._resources.values():
            if r.cooldown_s > 0:
                r.cooldown_s = max(0, r.cooldown_s - dt_seconds)

    def add_buy_option(self, option: BuyOption):
        """Add a purchasable item."""
        self._buy_options[option.name] = option

    def recommend_buy(self, budget: float) -> BuyRecommendation:
        """Recommend optimal purchases within budget.

        Uses greedy value/cost ratio selection.
        """
        available = [
            opt for opt in self._buy_options.values()
            if opt.cost <= budget
        ]

        # Sort by value/cost ratio (greedy approach)
        available.sort(key=lambda o: o.value_score / max(1, o.cost), reverse=True)

        selected = []
        remaining = budget
        total_value = 0.0

        for opt in available:
            if opt.cost <= remaining:
                # Check prerequisites
                prereqs_met = all(
                    any(s.name == req for s in selected)
                    for req in opt.requires
                )
                if not prereqs_met and opt.requires:
                    continue

                selected.append(opt)
                remaining -= opt.cost
                total_value += opt.value_score

        total_cost = budget - remaining
        reason = "full_buy" if remaining < 500 else "partial_buy" if selected else "save"

        return BuyRecommendation(
            items=selected,
            total_cost=total_cost,
            remaining_money=remaining,
            value_score=round(total_value / max(1, len(selected)), 3) if selected else 0,
            reason=reason,
        )

    def should_save(self, current_money: float, full_buy_cost: float, rounds_until_half: int = 3) -> bool:
        """Should the team save this round?

        Simple economy logic:
        - Save if can't afford a full buy
        - But force if team money is very low and it's late in the half
        """
        if current_money >= full_buy_cost:
            return False
        if rounds_until_half <= 1:
            return False  # Force buy on last round
        return True

    def get_low_resources(self) -> List[Resource]:
        """Get all resources that are low."""
        return [r for r in self._resources.values() if r.is_low and not r.is_empty]

    def get_empty_resources(self) -> List[Resource]:
        """Get all empty resources."""
        return [r for r in self._resources.values() if r.is_empty]

    def snapshot(self) -> Dict[str, float]:
        """Take a snapshot of all resource values."""
        snap = {r.name: r.current for r in self._resources.values()}
        self._history.append(snap)
        if len(self._history) > 100:
            self._history.pop(0)
        return snap

    def set_team_resource(self, player: str, resource_name: str, value: float):
        """Track a teammate's resource."""
        if player not in self._team_resources:
            self._team_resources[player] = {}
        self._team_resources[player][resource_name] = value

    def get_team_total(self, resource_name: str) -> float:
        """Get total of a resource across the team."""
        total = 0.0
        for player_res in self._team_resources.values():
            total += player_res.get(resource_name, 0)
        # Include self
        r = self._resources.get(resource_name)
        if r:
            total += r.current
        return total

    def get_stats(self) -> dict:
        low = self.get_low_resources()
        return {
            "tracked_resources": len(self._resources),
            "buy_options": len(self._buy_options),
            "low_resources": [r.name for r in low],
            "team_players": len(self._team_resources),
            "snapshots": len(self._history),
        }
