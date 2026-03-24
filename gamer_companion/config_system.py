"""TOML-based configuration with hot-reload and validation."""

from __future__ import annotations
import copy
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List
from loguru import logger

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


DEFAULT_CONFIG = {
    "general": {
        "mode": "observe",              # observe|assist|copilot|autonomous|training|coach
        "language": "en",
    },
    "ai": {
        "skill_level": 0.7,            # 0.0-1.0 (affects aim accuracy, reaction time)
        "aggression": 0.5,             # 0.0=passive, 1.0=hyper-aggressive
        "risk_tolerance": 0.5,
        "llm_provider": "gemini",      # gemini|gpt|grok|bravo|claude
        "llm_interval_s": 3.0,        # Strategic LLM query frequency
    },
    "performance": {
        "target_fps": 30,              # Main loop target
        "yolo_enabled": True,
        "ocr_enabled": True,
        "frame_diff_enabled": True,
        "gpu_device": "cuda",          # cuda|cpu
    },
    "audio": {
        "enabled": True,
        "gunshot_detection": True,
        "footstep_detection": True,
        "spatial_audio": True,
    },
    "overlay": {
        "enabled": True,
        "opacity": 0.8,
        "show_detections": True,
        "show_minimap_enhanced": True,
        "show_economy": True,
        "show_predictions": True,
        "font_size": 14,
    },
    "coach": {
        "personality": "tactical",     # tactical|aggressive|supportive|drill_sergeant|chill
        "voice_enabled": True,
        "voice_personality": "Echo",
        "callout_frequency": "normal", # minimal|normal|verbose
    },
    "safety": {
        "max_session_hours": 4,
        "max_aps": 15,
        "min_reaction_ms": 150,
        "kill_switch_key": "F12",
        "micro_pause_interval_s": 45,
        "audit_log": True,
    },
    "server": {
        "enabled": True,
        "port": 9600,
        "host": "127.0.0.1",
    },
    "discord": {
        "enabled": False,
        "client_id": "",
    },
    "paths": {
        "profiles_dir": "game_profiles",
        "models_dir": "models",
        "data_dir": "data",
        "logs_dir": "logs",
    },
    "updates": {
        "auto_check": True,
        "auto_download_models": True,
        "auto_download_profiles": True,
        "registry_url": "https://echo-gamer-companion-api.bmcii1976.workers.dev",
    },
}


class GGIConfig:
    """TOML-based configuration with hot-reload and validation.

    Features:
    - Loads from TOML file (human-readable, comments allowed)
    - Falls back to defaults for missing keys
    - Hot-reload: watches file for changes, applies without restart
    - Validation: type-checks values, clamps ranges
    - Callbacks: subsystems register for change notifications
    - CLI override: --key=value on command line
    """

    def __init__(self, path: str = "config/settings.toml"):
        self._path = Path(path)
        self._data: Dict[str, Any] = {}
        self._callbacks: List[Callable[[str, Any], None]] = []
        self._last_mtime: float = 0
        self._watcher_running = False
        self._load()
        self._start_watcher()

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value by dotted key (e.g., 'ai.skill_level')."""
        keys = dotted_key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, dotted_key: str, value: Any):
        """Set a config value and save."""
        keys = dotted_key.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        old = d.get(keys[-1])
        d[keys[-1]] = value
        if old != value:
            self._save()
            self._notify(dotted_key, value)

    def on_change(self, callback: Callable[[str, Any], None]):
        """Register callback for config changes."""
        self._callbacks.append(callback)

    def _load(self):
        """Load config from TOML file, merge with defaults."""
        self._data = copy.deepcopy(DEFAULT_CONFIG)

        if self._path.exists() and tomllib:
            try:
                with open(self._path, "rb") as f:
                    user_config = tomllib.load(f)
                self._deep_merge(self._data, user_config)
                self._last_mtime = self._path.stat().st_mtime
                logger.debug(f"Config loaded from {self._path}")
            except Exception as e:
                logger.warning(f"Config load error: {e}. Using defaults.")
        elif not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._save()
            logger.info(f"Default config created at {self._path}")

        self._validate()

    def _save(self):
        """Save current config to TOML file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# GGI APEX PREDATOR Configuration",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        self._dict_to_toml(self._data, lines, "")
        self._path.write_text("\n".join(lines), encoding="utf-8")

    def _dict_to_toml(self, d: dict, lines: list, prefix: str):
        """Convert nested dict to TOML format."""
        for key, val in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                lines.append(f"\n[{full_key}]")
                self._dict_to_toml(val, lines, full_key)
            elif isinstance(val, bool):
                lines.append(f"{key} = {'true' if val else 'false'}")
            elif isinstance(val, str):
                lines.append(f'{key} = "{val}"')
            elif isinstance(val, (int, float)):
                lines.append(f"{key} = {val}")

    def _deep_merge(self, base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def _validate(self):
        """Clamp values to valid ranges."""
        clamps = {
            "ai.skill_level": (0.0, 1.0),
            "ai.aggression": (0.0, 1.0),
            "ai.risk_tolerance": (0.0, 1.0),
            "performance.target_fps": (1, 144),
            "safety.max_session_hours": (0.5, 12),
            "safety.max_aps": (1, 20),
            "safety.min_reaction_ms": (50, 500),
            "overlay.opacity": (0.1, 1.0),
            "overlay.font_size": (8, 32),
            "server.port": (1024, 65535),
        }
        for key, (lo, hi) in clamps.items():
            val = self.get(key)
            if val is not None and isinstance(val, (int, float)):
                clamped = max(lo, min(hi, val))
                if clamped != val:
                    self.set(key, clamped)

    def _notify(self, key: str, value: Any):
        for cb in self._callbacks:
            try:
                cb(key, value)
            except Exception as e:
                logger.error(f"Config callback error: {e}")

    def _start_watcher(self):
        """Watch config file for external changes (hot-reload)."""
        self._watcher_running = True

        def watch():
            while self._watcher_running:
                try:
                    if self._path.exists():
                        mtime = self._path.stat().st_mtime
                        if mtime > self._last_mtime:
                            logger.info("Config file changed — reloading")
                            self._load()
                            self._notify("__reload__", None)
                except Exception:
                    pass
                time.sleep(2)

        t = threading.Thread(target=watch, daemon=True)
        t.start()

    def stop(self):
        self._watcher_running = False
