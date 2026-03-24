"""Declarative Game Profile System — the foundation of GGI.

Instead of hardcoding game-specific logic, games are described by JSON profiles.
The AI adapts its behavior using the profile. Profiles can be built-in,
auto-generated, community-contributed, or plugin-extended.
"""

from __future__ import annotations
import json
import ctypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger


@dataclass
class ScreenRegion:
    """A named region of interest on the game screen."""
    name: str                   # "minimap", "health_bar", "ammo", "kill_feed", "radar"
    x_pct: float               # Left edge as % of screen width (0.0 - 1.0)
    y_pct: float               # Top edge as % of screen height
    w_pct: float               # Width as % of screen width
    h_pct: float               # Height as % of screen height
    capture_fps: int = 5       # How often to capture this region specifically
    ocr_enabled: bool = False  # Run OCR on this region
    model_id: Optional[str] = None  # Specific YOLO model for this region


@dataclass
class KeyBind:
    """A game keybind mapping."""
    action: str                # "move_forward", "shoot", "ability_1", "buy_menu", "crouch"
    key: str                   # "w", "mouse1", "q", "b", "ctrl"
    hold: bool = False         # True if action requires holding the key
    cooldown_ms: float = 0     # Minimum time between uses (for abilities)


@dataclass
class WeaponProfile:
    """Weapon characteristics for aim/spray compensation."""
    name: str
    weapon_type: str           # "rifle", "smg", "pistol", "shotgun", "sniper", "melee"
    fire_rate_rpm: float       # Rounds per minute
    damage_body: int
    damage_head: int
    spray_pattern: List[Tuple[float, float]] = field(default_factory=list)
    movement_penalty: float = 0.0
    cost: int = 0
    magazine_size: int = 30
    reload_time_ms: float = 2500


@dataclass
class MapData:
    """Per-map strategic data."""
    name: str
    callout_zones: Dict[str, dict] = field(default_factory=dict)
    default_positions: Dict[str, List[str]] = field(default_factory=dict)
    rotation_paths: List[dict] = field(default_factory=list)
    common_angles: List[dict] = field(default_factory=list)


@dataclass
class GameProfile:
    """Complete declarative profile for a game.

    This is the key innovation: instead of writing game-specific code,
    you write a JSON profile. The AI adapts its behavior using the profile.
    """
    game_id: str               # "cs2", "valorant", "league_of_legends"
    display_name: str          # "Counter-Strike 2"
    genre: str                 # "tactical_fps", "battle_royale", "moba", "rts", "rpg"
    sub_genre: str = ""

    # Detection
    process_names: List[str] = field(default_factory=list)
    window_titles: List[str] = field(default_factory=list)
    icon_hash: Optional[str] = None
    splash_fingerprint: Optional[str] = None

    # Screen regions
    regions: Dict[str, ScreenRegion] = field(default_factory=dict)
    resolution_base: Tuple[int, int] = (1920, 1080)

    # Controls
    keybinds: Dict[str, KeyBind] = field(default_factory=dict)
    sensitivity: float = 1.0
    invert_y: bool = False
    raw_input: bool = True

    # Game mechanics
    max_players_per_team: int = 5
    round_based: bool = True
    economy_system: bool = True
    max_health: int = 100
    max_armor: int = 100
    respawn: bool = False
    friendly_fire: bool = True

    # Weapons
    weapons: Dict[str, WeaponProfile] = field(default_factory=dict)

    # Maps
    maps: Dict[str, MapData] = field(default_factory=dict)

    # AI tuning
    analysis_fps: int = 15
    reflex_threshold_ms: float = 150
    strategic_interval_s: float = 3.0
    default_skill_level: float = 0.7
    default_aggression: float = 0.5

    # FSM
    game_phases: List[str] = field(default_factory=list)
    fsm_module: Optional[str] = None

    # Overlay
    overlay_widgets: List[str] = field(default_factory=list)

    # Learning
    reward_signals: Dict[str, float] = field(default_factory=dict)

    # Plugins
    plugins: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        import dataclasses
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> "GameProfile":
        d = json.loads(data)
        if "regions" in d:
            d["regions"] = {k: ScreenRegion(**v) for k, v in d["regions"].items()}
        if "keybinds" in d:
            d["keybinds"] = {k: KeyBind(**v) for k, v in d["keybinds"].items()}
        if "weapons" in d:
            d["weapons"] = {k: WeaponProfile(**v) for k, v in d["weapons"].items()}
        if "maps" in d:
            d["maps"] = {k: MapData(**v) for k, v in d["maps"].items()}
        if "resolution_base" in d and isinstance(d["resolution_base"], list):
            d["resolution_base"] = tuple(d["resolution_base"])
        return cls(**d)


class GameDetector:
    """Auto-detect which game is running and load its profile.

    Detection chain:
    1. Enumerate visible windows -> match process name
    2. Match window title patterns
    3. Hash window icon -> compare to known fingerprints
    4. If no match: analyze splash screen with vision AI -> auto-generate profile
    """

    def __init__(self, profiles_dir: str = "game_profiles"):
        self.profiles_dir = Path(profiles_dir)
        self._profiles: Dict[str, GameProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        if not self.profiles_dir.exists():
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            return
        for f in self.profiles_dir.glob("*.json"):
            try:
                profile = GameProfile.from_json(f.read_text(encoding="utf-8"))
                self._profiles[profile.game_id] = profile
                logger.debug(f"Loaded profile: {profile.display_name}")
            except Exception as e:
                logger.warning(f"Failed to load profile {f.name}: {e}")
        logger.info(f"Loaded {len(self._profiles)} game profiles")

    def get_profile(self, game_id: str) -> Optional[GameProfile]:
        """Get a profile by game ID."""
        return self._profiles.get(game_id)

    def detect(self) -> Optional[GameProfile]:
        """Detect running game and return its profile."""
        windows = self._enumerate_windows()
        for title, pid, exe_name in windows:
            for profile in self._profiles.values():
                if exe_name.lower() in [p.lower() for p in profile.process_names]:
                    logger.info(f"Detected game: {profile.display_name} (via process name)")
                    return profile
                for pattern in profile.window_titles:
                    if pattern.lower() in title.lower():
                        logger.info(f"Detected game: {profile.display_name} (via window title)")
                        return profile
        return None

    def _enumerate_windows(self) -> List[Tuple[str, int, str]]:
        """Get all visible windows with titles and process info."""
        from ctypes import wintypes
        results = []

        def callback(hwnd, _):
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    pid = wintypes.DWORD()
                    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    try:
                        import psutil
                        proc = psutil.Process(pid.value)
                        exe = proc.name()
                    except Exception:
                        exe = "unknown"
                    results.append((title, pid.value, exe))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        ctypes.windll.user32.EnumWindows(WNDENUMPROC(callback), 0)
        return results

    def auto_generate_profile(self, window_title: str) -> GameProfile:
        """Auto-generate a basic profile from window title and genre conventions."""
        return GameProfile(
            game_id=window_title.lower().replace(" ", "_")[:32],
            display_name=window_title,
            genre="unknown",
        )
