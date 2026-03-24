"""Navigation Engine — Navigate menus, shops, and inventories.

Automates complex navigation sequences in game menus:
- Buy menu navigation (CS2 buy menu, LoL item shop)
- Settings menu navigation
- Inventory management
- Character selection
- Queue/matchmaking flows

Uses detected UI elements from ElementDetector to plan click sequences.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from loguru import logger


@dataclass
class NavAction:
    """A single navigation action."""
    action_type: str          # "click", "type", "scroll", "wait", "drag"
    target: str = ""          # Element text/ID to interact with
    x: int = 0
    y: int = 0
    text: str = ""            # For "type" actions
    scroll_amount: int = 0    # For "scroll" actions
    wait_ms: float = 0        # For "wait" actions
    drag_to: Tuple[int, int] = (0, 0)  # For "drag" actions
    completed: bool = False
    success: bool = False


@dataclass
class NavRoute:
    """A planned navigation route (sequence of actions)."""
    route_id: str
    name: str
    description: str = ""
    actions: List[NavAction] = field(default_factory=list)
    current_step: int = 0
    completed: bool = False
    success: bool = False
    start_time: float = 0
    end_time: float = 0

    @property
    def progress(self) -> float:
        if not self.actions:
            return 1.0
        return self.current_step / len(self.actions)


# Pre-built navigation routes for common tasks
PRESET_ROUTES: Dict[str, List[dict]] = {
    "cs2_buy_ak47": [
        {"action_type": "click", "target": "buy_menu", "wait_ms": 100},
        {"action_type": "click", "target": "rifles", "wait_ms": 50},
        {"action_type": "click", "target": "ak47", "wait_ms": 50},
    ],
    "cs2_buy_awp": [
        {"action_type": "click", "target": "buy_menu", "wait_ms": 100},
        {"action_type": "click", "target": "rifles", "wait_ms": 50},
        {"action_type": "click", "target": "awp", "wait_ms": 50},
    ],
    "cs2_buy_armor_helmet": [
        {"action_type": "click", "target": "buy_menu", "wait_ms": 100},
        {"action_type": "click", "target": "equipment", "wait_ms": 50},
        {"action_type": "click", "target": "kevlar_helmet", "wait_ms": 50},
    ],
    "cs2_buy_smoke_flash": [
        {"action_type": "click", "target": "buy_menu", "wait_ms": 100},
        {"action_type": "click", "target": "grenades", "wait_ms": 50},
        {"action_type": "click", "target": "smoke", "wait_ms": 50},
        {"action_type": "click", "target": "flash", "wait_ms": 50},
    ],
    "settings_lower_sensitivity": [
        {"action_type": "click", "target": "settings", "wait_ms": 200},
        {"action_type": "click", "target": "mouse", "wait_ms": 100},
        {"action_type": "click", "target": "sensitivity", "wait_ms": 50},
        {"action_type": "type", "text": "1.2", "wait_ms": 50},
    ],
    # --- Call of Duty ---
    "cod_loadout_select": [
        {"action_type": "wait", "wait_ms": 500},
        {"action_type": "click", "target": "loadout_menu", "wait_ms": 200},
        {"action_type": "click", "target": "loadout_1", "wait_ms": 100},
    ],
    "cod_buy_station_loadout": [
        {"action_type": "click", "target": "buy_station", "wait_ms": 200},
        {"action_type": "click", "target": "buy_loadout_drop", "wait_ms": 100},
        {"action_type": "click", "target": "confirm", "wait_ms": 50},
    ],
    "cod_buy_station_uav": [
        {"action_type": "click", "target": "buy_station", "wait_ms": 200},
        {"action_type": "click", "target": "killstreaks", "wait_ms": 100},
        {"action_type": "click", "target": "uav", "wait_ms": 50},
    ],
    "cod_buy_station_plates": [
        {"action_type": "click", "target": "buy_station", "wait_ms": 200},
        {"action_type": "click", "target": "armor_plates", "wait_ms": 100},
    ],
    # --- Valorant ---
    "val_buy_vandal": [
        {"action_type": "click", "target": "buy_menu", "wait_ms": 100},
        {"action_type": "click", "target": "rifles", "wait_ms": 50},
        {"action_type": "click", "target": "vandal", "wait_ms": 50},
    ],
    "val_buy_phantom": [
        {"action_type": "click", "target": "buy_menu", "wait_ms": 100},
        {"action_type": "click", "target": "rifles", "wait_ms": 50},
        {"action_type": "click", "target": "phantom", "wait_ms": 50},
    ],
    "val_buy_armor_full": [
        {"action_type": "click", "target": "buy_menu", "wait_ms": 100},
        {"action_type": "click", "target": "armor", "wait_ms": 50},
        {"action_type": "click", "target": "heavy_shields", "wait_ms": 50},
    ],
    # --- League of Legends ---
    "lol_buy_item_1": [
        {"action_type": "click", "target": "shop_button", "wait_ms": 200},
        {"action_type": "click", "target": "recommended_1", "wait_ms": 100},
        {"action_type": "click", "target": "buy_button", "wait_ms": 50},
        {"action_type": "click", "target": "shop_close", "wait_ms": 50},
    ],
    "lol_trinket_swap": [
        {"action_type": "click", "target": "shop_button", "wait_ms": 200},
        {"action_type": "click", "target": "search_bar", "wait_ms": 50},
        {"action_type": "type", "text": "oracle", "wait_ms": 100},
        {"action_type": "click", "target": "result_1", "wait_ms": 50},
        {"action_type": "click", "target": "buy_button", "wait_ms": 50},
    ],
    # --- Dota 2 ---
    "dota_buy_bkb": [
        {"action_type": "click", "target": "shop_button", "wait_ms": 200},
        {"action_type": "click", "target": "search_bar", "wait_ms": 50},
        {"action_type": "type", "text": "black king bar", "wait_ms": 100},
        {"action_type": "click", "target": "result_1", "wait_ms": 50},
        {"action_type": "click", "target": "buy_button", "wait_ms": 50},
    ],
    # --- Fortnite ---
    "fn_queue_solo": [
        {"action_type": "click", "target": "game_mode", "wait_ms": 200},
        {"action_type": "click", "target": "solo", "wait_ms": 100},
        {"action_type": "click", "target": "play", "wait_ms": 50},
    ],
    # --- Overwatch ---
    "ow_hero_select": [
        {"action_type": "click", "target": "hero_grid", "wait_ms": 200},
        {"action_type": "click", "target": "preferred_hero", "wait_ms": 100},
        {"action_type": "click", "target": "confirm", "wait_ms": 100},
    ],
    # --- Hearthstone ---
    "hs_end_turn": [
        {"action_type": "click", "target": "end_turn_button", "wait_ms": 200},
    ],
    "hs_play_card": [
        {"action_type": "drag", "target": "hand_card_1", "wait_ms": 100},
    ],
    # --- Rocket League ---
    "rl_quick_chat_nice_shot": [
        {"action_type": "press", "target": "d_pad_right", "wait_ms": 50},
        {"action_type": "press", "target": "d_pad_up", "wait_ms": 50},
    ],
    # --- SC2 ---
    "sc2_build_worker": [
        {"action_type": "click", "target": "town_hall", "wait_ms": 50},
        {"action_type": "press", "target": "S", "wait_ms": 30},
    ],
}


class NavigationEngine:
    """Navigate game menus and UI using detected elements.

    Process:
    1. Receive navigation goal (e.g., "buy AK47")
    2. Look up or compute route
    3. Execute actions step by step
    4. Verify each step succeeded before proceeding
    5. Handle failures with retry/fallback

    Integration: Uses ElementDetector for finding click targets.
    """

    def __init__(self):
        self._routes: Dict[str, NavRoute] = {}
        self._active_route: Optional[NavRoute] = None
        self._history: List[NavRoute] = []
        self._route_counter = 0

    def create_route(self, name: str, actions: List[NavAction], description: str = "") -> NavRoute:
        """Create a navigation route."""
        self._route_counter += 1
        route = NavRoute(
            route_id=f"route_{self._route_counter}",
            name=name,
            description=description,
            actions=actions,
        )
        self._routes[route.route_id] = route
        return route

    def load_preset(self, preset_name: str) -> Optional[NavRoute]:
        """Load a preset navigation route."""
        preset = PRESET_ROUTES.get(preset_name)
        if not preset:
            logger.warning(f"Unknown preset: {preset_name}")
            return None

        actions = [NavAction(**step) for step in preset]
        return self.create_route(preset_name, actions, f"Preset: {preset_name}")

    def start_route(self, route_id: str) -> bool:
        """Start executing a navigation route."""
        route = self._routes.get(route_id)
        if not route:
            return False

        route.current_step = 0
        route.completed = False
        route.success = False
        route.start_time = time.time()
        self._active_route = route
        return True

    def get_next_action(self) -> Optional[NavAction]:
        """Get the next action to execute."""
        if not self._active_route:
            return None

        route = self._active_route
        if route.current_step >= len(route.actions):
            route.completed = True
            route.success = True
            route.end_time = time.time()
            self._history.append(route)
            self._active_route = None
            return None

        return route.actions[route.current_step]

    def advance(self, success: bool = True):
        """Mark current action as done and advance to next."""
        if not self._active_route:
            return

        route = self._active_route
        if route.current_step < len(route.actions):
            route.actions[route.current_step].completed = True
            route.actions[route.current_step].success = success
            route.current_step += 1

            if route.current_step >= len(route.actions):
                route.completed = True
                route.success = all(a.success for a in route.actions)
                route.end_time = time.time()
                self._history.append(route)
                self._active_route = None

    def abort_route(self):
        """Abort the current route."""
        if self._active_route:
            self._active_route.completed = True
            self._active_route.success = False
            self._active_route.end_time = time.time()
            self._history.append(self._active_route)
            self._active_route = None

    def is_navigating(self) -> bool:
        return self._active_route is not None

    @property
    def active_route(self) -> Optional[NavRoute]:
        return self._active_route

    def list_presets(self) -> List[str]:
        return list(PRESET_ROUTES.keys())

    def get_stats(self) -> dict:
        completed = [r for r in self._history if r.completed]
        successful = [r for r in completed if r.success]
        return {
            "total_routes": len(self._routes),
            "active": self._active_route.name if self._active_route else None,
            "completed": len(completed),
            "success_rate": round(len(successful) / max(1, len(completed)), 3),
            "presets_available": len(PRESET_ROUTES),
        }
