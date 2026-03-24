"""Test game watcher daemon, per-game settings, and auto-activation."""

import json
import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from gamer_companion.daemon.game_settings import GameSettings, GameSettingsStore
from gamer_companion.daemon.game_watcher import (
    GameWatcherDaemon, DaemonConfig, ActiveSession, _FSM_GENRE_MAP,
)
from gamer_companion.game_registry import GameRegistry, GAME_REGISTRY
from gamer_companion.autonomous.mode_manager import PlayMode


# =============================================================================
# PER-GAME SETTINGS
# =============================================================================
class TestGameSettings:
    def test_default_settings(self):
        s = GameSettings(game_id="cs2")
        assert s.game_id == "cs2"
        assert s.default_mode == "observe"
        assert s.auto_activate is True
        assert s.skill_level == 0.7
        assert s.total_sessions == 0

    def test_custom_settings(self):
        s = GameSettings(
            game_id="league",
            default_mode="assist",
            skill_level=0.9,
            aggression=0.8,
            active_pro_profile="faker",
        )
        assert s.default_mode == "assist"
        assert s.skill_level == 0.9
        assert s.active_pro_profile == "faker"


class TestGameSettingsStore:
    def test_store_save_and_load(self, tmp_path):
        store = GameSettingsStore(settings_dir=str(tmp_path / "settings"))
        settings = store.get("cs2")
        settings.default_mode = "autonomous"
        settings.skill_level = 0.95
        settings.auto_mode_confirmed = True
        store.save(settings)

        # Verify file exists
        assert (tmp_path / "settings" / "cs2.json").exists()

        # Load fresh store from same dir
        store2 = GameSettingsStore(settings_dir=str(tmp_path / "settings"))
        loaded = store2.get("cs2")
        assert loaded.default_mode == "autonomous"
        assert loaded.skill_level == 0.95
        assert loaded.auto_mode_confirmed is True

    def test_store_creates_defaults(self, tmp_path):
        store = GameSettingsStore(settings_dir=str(tmp_path / "settings"))
        settings = store.get("unknown_game")
        assert settings.game_id == "unknown_game"
        assert settings.default_mode == "observe"

    def test_store_delete(self, tmp_path):
        store = GameSettingsStore(settings_dir=str(tmp_path / "settings"))
        settings = store.get("chess")
        store.save(settings)
        assert "chess" in store.list_games()

        store.delete("chess")
        assert "chess" not in store.list_games()
        assert not (tmp_path / "settings" / "chess.json").exists()

    def test_store_list_games(self, tmp_path):
        store = GameSettingsStore(settings_dir=str(tmp_path / "settings"))
        for gid in ["cs2", "league", "chess"]:
            store.save(GameSettings(game_id=gid))
        assert set(store.list_games()) == {"cs2", "league", "chess"}

    def test_store_stats(self, tmp_path):
        store = GameSettingsStore(settings_dir=str(tmp_path / "settings"))
        store.save(GameSettings(game_id="cs2", default_mode="autonomous"))
        stats = store.get_stats()
        assert stats["games_configured"] == 1
        assert "cs2" in stats["games"]


# =============================================================================
# DAEMON CONFIG
# =============================================================================
class TestDaemonConfig:
    def test_defaults(self):
        cfg = DaemonConfig()
        assert cfg.poll_interval_s == 3.0
        assert cfg.steam_enabled is True
        assert cfg.auto_activate is True
        assert cfg.default_mode == "observe"

    def test_custom(self):
        cfg = DaemonConfig(
            poll_interval_s=1.0,
            steam_enabled=False,
            default_mode="assist",
        )
        assert cfg.poll_interval_s == 1.0
        assert cfg.steam_enabled is False


# =============================================================================
# FSM GENRE MAP
# =============================================================================
class TestFSMGenreMap:
    def test_all_major_genres_mapped(self):
        expected = [
            "tactical_fps", "battle_royale", "moba", "rts",
            "fighting", "racing", "sports", "card", "survival",
            "mmo", "soulslike", "board_game", "puzzle",
            "turn_based_strategy", "grand_strategy",
        ]
        for genre in expected:
            assert genre in _FSM_GENRE_MAP, f"Missing FSM mapping for {genre}"


# =============================================================================
# GAME WATCHER DAEMON
# =============================================================================
class TestGameWatcherDaemon:
    def test_init(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        assert daemon.is_running is False
        assert daemon.active_game is None
        assert daemon.active_mode is None

    def test_stats_when_idle(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()
        stats = daemon.get_stats()
        assert stats["daemon_running"] is False
        assert stats["active_session"] is None
        assert stats["games_in_registry"] >= 45

    def test_force_activate(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        assert daemon.force_activate("cs2")
        assert daemon.active_game == "Counter-Strike 2"
        assert daemon.active_mode == "observe"

        stats = daemon.get_stats()
        assert stats["active_session"] is not None
        assert stats["active_session"]["game_id"] == "cs2"

    def test_force_activate_unknown_game(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        assert daemon.force_activate("nonexistent_game") is False

    def test_force_deactivate(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        daemon.force_activate("chess")
        assert daemon.active_game == "Chess"

        daemon.force_deactivate()
        assert daemon.active_game is None

    def test_switch_mode(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()
        daemon.force_activate("cs2")

        assert daemon.switch_mode("assist")
        assert daemon.active_mode == "assist"

        assert daemon.switch_mode("coach")
        assert daemon.active_mode == "coach"

        # Dangerous mode without confirmation
        assert daemon.switch_mode("autonomous") is False
        assert daemon.active_mode == "coach"

        # Dangerous mode with confirmation
        assert daemon.switch_mode("autonomous", confirmed=True)
        assert daemon.active_mode == "autonomous"

    def test_mode_no_active_session(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        assert daemon.switch_mode("assist") is False

    def test_game_detection_by_process(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)

        # Simulate finding cs2.exe in process list
        result = daemon._detect_game({"cs2.exe", "explorer.exe", "chrome.exe"})
        assert result is not None
        assert result[0].game_id == "cs2"

    def test_game_detection_lol(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)

        result = daemon._detect_game({"league of legends.exe", "explorer.exe"})
        assert result is not None
        assert result[0].game_id == "league"

    def test_game_detection_no_match(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        result = daemon._detect_game({"explorer.exe", "notepad.exe"})
        assert result is None

    def test_callbacks_fired(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        detected_games = []
        exited_games = []

        daemon.on_game_detected(lambda e, s: detected_games.append(e.game_id))
        daemon.on_game_exited(lambda e, d: exited_games.append(e.game_id))

        daemon.force_activate("valorant")
        assert detected_games == ["valorant"]

        daemon.force_deactivate()
        assert exited_games == ["valorant"]

    def test_update_settings(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)

        daemon.update_settings("cs2", skill_level=0.99, default_mode="autonomous")
        settings = daemon._settings.get("cs2")
        assert settings.skill_level == 0.99
        assert settings.default_mode == "autonomous"

    def test_session_stats_persist(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        daemon.force_activate("chess")
        daemon.force_deactivate()

        settings = daemon._settings.get("chess")
        assert settings.total_sessions == 1
        assert settings.total_play_minutes >= 0

    def test_auto_mode_confirmed_settings(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        # Pre-configure autonomous with confirmation
        daemon.update_settings(
            "cs2",
            default_mode="autonomous",
            auto_mode_confirmed=True,
        )

        daemon.force_activate("cs2")
        assert daemon.active_mode == "autonomous"

    def test_dangerous_mode_falls_back(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        # Set autonomous WITHOUT confirmation
        daemon.update_settings(
            "cs2",
            default_mode="autonomous",
            auto_mode_confirmed=False,
        )

        daemon.force_activate("cs2")
        # Should fall back to observe since not confirmed
        assert daemon.active_mode == "observe"

    def test_multiple_game_switches(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        daemon.force_activate("cs2")
        assert daemon.active_game == "Counter-Strike 2"

        # Switch to another game (auto-deactivates first)
        daemon.force_activate("chess")
        assert daemon.active_game == "Chess"

        daemon.force_deactivate()
        assert daemon.active_game is None

    def test_fsm_created_for_genre(self, tmp_path):
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        daemon.force_activate("cs2")
        assert daemon._active_session.fsm is not None
        assert "FPS" in type(daemon._active_session.fsm).__name__

        daemon.force_activate("chess")
        assert daemon._active_session.fsm is not None
        assert "Board" in type(daemon._active_session.fsm).__name__

        daemon.force_activate("league")
        assert daemon._active_session.fsm is not None
        assert "MOBA" in type(daemon._active_session.fsm).__name__

    def test_poll_detects_and_deactivates(self, tmp_path):
        """Simulate a full poll cycle: detect game, then detect exit."""
        cfg = DaemonConfig(
            steam_enabled=False,
            settings_dir=str(tmp_path / "settings"),
        )
        daemon = GameWatcherDaemon(config=cfg)
        daemon._start_time = time.time()

        # Mock process list with CS2 running
        with patch.object(daemon, "_get_running_processes", return_value={"cs2.exe", "explorer.exe"}):
            daemon._poll()
        assert daemon.active_game == "Counter-Strike 2"

        # CS2 no longer running
        with patch.object(daemon, "_get_running_processes", return_value={"explorer.exe"}):
            daemon._poll()
        assert daemon.active_game is None
