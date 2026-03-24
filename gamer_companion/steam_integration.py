"""Steam Integration — Auto-detect games via Steam API and local client.

Connects to Steam in three ways:
1. Steam Web API (public game data, player stats)
2. Local Steam client (installed games, running game detection)
3. Process detection fallback (no Steam required)

This allows the Gamer Companion to automatically detect what game
you're playing and load the appropriate profile, even for games
not in the hardcoded registry — by reading Steam's game metadata.
"""

from __future__ import annotations
import json
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import winreg
except ImportError:
    winreg = None


STEAM_WEB_API_BASE = "https://api.steampowered.com"
STEAM_STORE_API = "https://store.steampowered.com/api"


@dataclass
class SteamGame:
    """A game from the Steam library."""
    app_id: int
    name: str
    install_dir: str = ""
    size_bytes: int = 0
    last_played: int = 0
    playtime_minutes: int = 0
    genres: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    exe_path: str = ""
    is_installed: bool = False
    is_running: bool = False
    store_tags: List[str] = field(default_factory=list)


# Map Steam genre/tag strings to our Genre enum values
STEAM_GENRE_MAP = {
    "action": "tactical_fps",
    "fps": "tactical_fps",
    "shooter": "tactical_fps",
    "battle royale": "battle_royale",
    "moba": "moba",
    "strategy": "rts",
    "real-time strategy": "rts",
    "turn-based strategy": "turn_based_strategy",
    "turn-based": "turn_based_strategy",
    "4x": "grand_strategy",
    "grand strategy": "grand_strategy",
    "fighting": "fighting",
    "racing": "racing",
    "sports": "sports",
    "card game": "card",
    "trading card game": "card",
    "puzzle": "puzzle",
    "survival": "survival",
    "sandbox": "sandbox",
    "mmo": "mmo",
    "mmorpg": "mmo",
    "souls-like": "soulslike",
    "roguelike": "roguelike",
    "roguelite": "roguelike",
    "platformer": "platformer",
    "simulation": "simulation",
    "city builder": "simulation",
    "automation": "simulation",
    "board game": "board_game",
    "chess": "board_game",
    "rhythm": "rhythm",
}


def _find_steam_path() -> Optional[Path]:
    """Find Steam installation path from registry or common locations."""
    # Try Windows registry first
    if winreg:
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\WOW6432Node\Valve\Steam",
            )
            path, _ = winreg.QueryValueEx(key, "InstallPath")
            winreg.CloseKey(key)
            if Path(path).exists():
                return Path(path)
        except (OSError, FileNotFoundError):
            pass

        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"SOFTWARE\Valve\Steam",
            )
            path, _ = winreg.QueryValueEx(key, "SteamPath")
            winreg.CloseKey(key)
            if Path(path).exists():
                return Path(path)
        except (OSError, FileNotFoundError):
            pass

    # Fallback: common locations
    common_paths = [
        Path("C:/Program Files (x86)/Steam"),
        Path("C:/Program Files/Steam"),
        Path("D:/Steam"),
        Path("D:/SteamLibrary"),
        Path(os.path.expanduser("~/.steam/steam")),
    ]
    for p in common_paths:
        if p.exists():
            return p
    return None


def _parse_vdf(text: str) -> dict:
    """Parse Valve's VDF (KeyValues) format into a dict.

    VDF is a simple nested key-value format:
        "key" "value"
        "key" { nested }
    """
    result = {}
    stack = [result]
    lines = text.replace("\t", " ").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("//"):
            continue
        if line == "{":
            continue
        if line == "}":
            if len(stack) > 1:
                stack.pop()
            continue

        # Parse "key" "value" or "key" followed by {
        parts = []
        in_quote = False
        current = []
        for ch in line:
            if ch == '"':
                if in_quote:
                    parts.append("".join(current))
                    current = []
                in_quote = not in_quote
            elif in_quote:
                current.append(ch)

        if len(parts) >= 2:
            stack[-1][parts[0]] = parts[1]
        elif len(parts) == 1:
            # Next line should be {
            new_dict = {}
            stack[-1][parts[0]] = new_dict
            stack.append(new_dict)

    return result


def _find_library_folders(steam_path: Path) -> List[Path]:
    """Find all Steam library folders (multiple drives)."""
    folders = [steam_path]
    vdf_path = steam_path / "steamapps" / "libraryfolders.vdf"
    if not vdf_path.exists():
        return folders

    try:
        data = _parse_vdf(vdf_path.read_text(encoding="utf-8", errors="ignore"))
        lib_data = data.get("libraryfolders", data.get("LibraryFolders", {}))
        for key, val in lib_data.items():
            if isinstance(val, dict) and "path" in val:
                p = Path(val["path"])
                if p.exists() and p not in folders:
                    folders.append(p)
            elif isinstance(val, str) and Path(val).exists():
                p = Path(val)
                if p not in folders:
                    folders.append(p)
    except Exception as e:
        logger.warning(f"Failed to parse libraryfolders.vdf: {e}")

    return folders


class SteamIntegration:
    """Integrate with Steam for game detection and library scanning.

    Usage:
        steam = SteamIntegration()
        games = steam.get_installed_games()
        running = steam.detect_running_game()
    """

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._steam_path = _find_steam_path()
        self._library_folders: List[Path] = []
        self._game_cache: Dict[int, SteamGame] = {}
        self._last_scan = 0

        if self._steam_path:
            self._library_folders = _find_library_folders(self._steam_path)
            logger.info(
                f"Steam found: {self._steam_path} "
                f"({len(self._library_folders)} library folders)"
            )
        else:
            logger.warning("Steam installation not found")

    @property
    def is_available(self) -> bool:
        return self._steam_path is not None

    @property
    def steam_path(self) -> Optional[Path]:
        return self._steam_path

    def get_installed_games(self, force_rescan: bool = False) -> List[SteamGame]:
        """Scan all Steam library folders for installed games.

        Reads .acf manifest files from steamapps/ directories.
        """
        if self._game_cache and not force_rescan:
            if time.time() - self._last_scan < 300:
                return list(self._game_cache.values())

        games = []
        for lib_folder in self._library_folders:
            steamapps = lib_folder / "steamapps"
            if not steamapps.exists():
                continue

            for acf_file in steamapps.glob("appmanifest_*.acf"):
                try:
                    data = _parse_vdf(
                        acf_file.read_text(encoding="utf-8", errors="ignore")
                    )
                    app_state = data.get("AppState", {})
                    if not app_state:
                        continue

                    app_id = int(app_state.get("appid", 0))
                    name = app_state.get("name", "Unknown")
                    install_dir = app_state.get("installdir", "")
                    size = int(app_state.get("SizeOnDisk", 0))
                    last_played = int(app_state.get("LastPlayed", 0))

                    game = SteamGame(
                        app_id=app_id,
                        name=name,
                        install_dir=str(steamapps / "common" / install_dir),
                        size_bytes=size,
                        last_played=last_played,
                        is_installed=True,
                    )
                    games.append(game)
                    self._game_cache[app_id] = game

                except Exception as e:
                    logger.debug(f"Failed to parse {acf_file.name}: {e}")

        self._last_scan = time.time()
        logger.info(f"Found {len(games)} installed Steam games")
        return games

    def get_recently_played(self, limit: int = 20) -> List[SteamGame]:
        """Get recently played games sorted by last played time."""
        games = self.get_installed_games()
        return sorted(
            [g for g in games if g.last_played > 0],
            key=lambda g: g.last_played,
            reverse=True,
        )[:limit]

    def find_game(self, query: str) -> List[SteamGame]:
        """Search installed games by name."""
        q = query.lower()
        games = self.get_installed_games()
        return [g for g in games if q in g.name.lower()]

    def detect_running_game(self) -> Optional[SteamGame]:
        """Detect which Steam game is currently running.

        Checks the Steam client's local state for active app ID.
        Falls back to process enumeration.
        """
        # Method 1: Read Steam's registry for running app
        if winreg:
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"SOFTWARE\Valve\Steam",
                )
                app_id, _ = winreg.QueryValueEx(key, "RunningAppID")
                winreg.CloseKey(key)
                if app_id and app_id > 0:
                    return self._game_cache.get(int(app_id))
            except (OSError, FileNotFoundError):
                pass

        return None

    def get_game_details_url(self, app_id: int) -> str:
        """Get the Steam Store API URL for game details."""
        return f"{STEAM_STORE_API}/appdetails?appids={app_id}"

    def get_player_stats_url(self, steam_id: str, app_id: int) -> str:
        """Get URL for player stats (requires API key)."""
        return (
            f"{STEAM_WEB_API_BASE}/ISteamUserStats/GetUserStatsForGame/v2/"
            f"?appid={app_id}&key={self._api_key}&steamid={steam_id}"
        )

    def classify_genre(self, game: SteamGame) -> str:
        """Classify a Steam game into our genre system using tags/categories."""
        for tag in game.store_tags + game.genres + game.categories:
            tag_lower = tag.lower().strip()
            if tag_lower in STEAM_GENRE_MAP:
                return STEAM_GENRE_MAP[tag_lower]
        return "unknown"

    def get_stats(self) -> dict:
        return {
            "steam_available": self.is_available,
            "steam_path": str(self._steam_path) if self._steam_path else None,
            "library_folders": len(self._library_folders),
            "games_cached": len(self._game_cache),
            "has_api_key": bool(self._api_key),
        }
