"""Per-Game Settings Persistence — Remember settings for each game.

Stores and loads user preferences per game:
- Preferred play mode (observe, assist, autonomous, etc.)
- Skill level, aggression, sensitivity overrides
- Active pro profile / mimic target
- Overlay widget preferences
- Custom keybind overrides
- Auto-start behavior (which mode to enter when game launches)

Settings are stored as JSON files in a user-configurable directory.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from pathlib import Path
from loguru import logger


@dataclass
class GameSettings:
    """Per-game user settings."""
    game_id: str
    display_name: str = ""

    # Auto-launch behavior
    auto_activate: bool = True
    default_mode: str = "observe"       # PlayMode value to enter on game launch
    auto_mode_confirmed: bool = False   # If True, skip confirmation for dangerous modes

    # AI tuning
    skill_level: float = 0.7
    aggression: float = 0.5
    reaction_time_ms: float = 150.0

    # Sensitivity
    sensitivity_override: Optional[float] = None
    dpi: int = 800

    # Pro mimic
    active_pro_profile: str = ""

    # Overlay
    overlay_enabled: bool = True
    overlay_widgets: List[str] = field(default_factory=lambda: [
        "crosshair", "minimap", "stats",
    ])
    overlay_opacity: float = 0.8

    # Input
    keybind_overrides: Dict[str, str] = field(default_factory=dict)

    # Session
    max_session_hours: float = 4.0
    auto_pause_on_alt_tab: bool = True

    # Stats tracking
    total_sessions: int = 0
    total_play_minutes: float = 0.0
    last_played: float = 0.0


class GameSettingsStore:
    """Persist and retrieve per-game settings.

    Usage:
        store = GameSettingsStore()
        settings = store.get("cs2")
        settings.default_mode = "assist"
        settings.skill_level = 0.8
        store.save(settings)
    """

    def __init__(self, settings_dir: Optional[str] = None):
        if settings_dir:
            self._dir = Path(settings_dir)
        else:
            self._dir = Path.home() / ".gamer-companion" / "game_settings"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, GameSettings] = {}
        self._load_all()

    def _load_all(self):
        for f in self._dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                game_id = data.get("game_id", f.stem)
                self._cache[game_id] = GameSettings(**data)
            except Exception as e:
                logger.warning(f"Failed to load settings {f.name}: {e}")
        logger.info(f"Loaded settings for {len(self._cache)} games")

    def get(self, game_id: str) -> GameSettings:
        """Get settings for a game. Creates defaults if none exist."""
        if game_id not in self._cache:
            self._cache[game_id] = GameSettings(game_id=game_id)
        return self._cache[game_id]

    def save(self, settings: GameSettings):
        """Save settings to disk."""
        self._cache[settings.game_id] = settings
        path = self._dir / f"{settings.game_id}.json"
        try:
            path.write_text(
                json.dumps(asdict(settings), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save settings for {settings.game_id}: {e}")

    def delete(self, game_id: str):
        """Delete settings for a game."""
        self._cache.pop(game_id, None)
        path = self._dir / f"{game_id}.json"
        if path.exists():
            path.unlink()

    def list_games(self) -> List[str]:
        """List all games with saved settings."""
        return list(self._cache.keys())

    def get_stats(self) -> dict:
        return {
            "settings_dir": str(self._dir),
            "games_configured": len(self._cache),
            "games": {
                gid: {
                    "mode": s.default_mode,
                    "skill": s.skill_level,
                    "sessions": s.total_sessions,
                }
                for gid, s in self._cache.items()
            },
        }
