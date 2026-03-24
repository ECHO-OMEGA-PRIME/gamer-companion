"""Test the Steam Overlay control panel web server."""

import json
import time
import pytest
from unittest.mock import patch
from http.client import HTTPConnection

from gamer_companion.daemon.game_watcher import GameWatcherDaemon, DaemonConfig
from gamer_companion.daemon.control_panel import start_control_panel


@pytest.fixture
def daemon_and_panel(tmp_path):
    """Create a daemon + control panel server for testing."""
    cfg = DaemonConfig(
        steam_enabled=False,
        settings_dir=str(tmp_path / "settings"),
    )
    daemon = GameWatcherDaemon(config=cfg)
    daemon._start_time = time.time()

    # Start on a random high port to avoid conflicts
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = start_control_panel(daemon, port=port)
    time.sleep(0.2)  # Let server thread start

    yield daemon, port, server

    server.shutdown()


def _get(port, path):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    return resp.status, body


def _post(port, path, data):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(data)
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    result = resp.read().decode("utf-8")
    conn.close()
    return resp.status, json.loads(result)


class TestControlPanel:
    def test_health(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, body = _get(port, "/health")
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "ok"

    def test_serves_html(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, body = _get(port, "/")
        assert status == 200
        assert "GGI APEX PREDATOR" in body
        assert "Control Panel" in body
        assert "mode-btn" in body

    def test_panel_path(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, body = _get(port, "/panel")
        assert status == 200
        assert "GGI APEX PREDATOR" in body

    def test_api_status_idle(self, daemon_and_panel):
        daemon, port, _ = daemon_and_panel
        status, body = _get(port, "/api/status")
        assert status == 200
        data = json.loads(body)
        assert data["daemon_running"] is False
        assert data["active_session"] is None
        assert data["games_in_registry"] >= 45

    def test_api_games_list(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, body = _get(port, "/api/games")
        assert status == 200
        data = json.loads(body)
        assert data["total"] >= 45
        names = [g["name"] for g in data["games"]]
        assert "Counter-Strike 2" in names
        assert "Chess" in names

    def test_api_activate_deactivate(self, daemon_and_panel):
        daemon, port, _ = daemon_and_panel

        # Activate CS2
        status, data = _post(port, "/api/activate", {"game_id": "cs2"})
        assert status == 200
        assert data["success"] is True
        assert data["active_game"] == "Counter-Strike 2"

        # Check status reflects active game
        st, body = _get(port, "/api/status")
        status_data = json.loads(body)
        assert status_data["active_session"] is not None
        assert status_data["active_session"]["game"] == "Counter-Strike 2"

        # Deactivate
        status, data = _post(port, "/api/deactivate", {})
        assert status == 200
        assert data["success"] is True

    def test_api_mode_switch(self, daemon_and_panel):
        daemon, port, _ = daemon_and_panel

        # Activate a game first
        _post(port, "/api/activate", {"game_id": "chess"})

        # Switch to assist
        status, data = _post(port, "/api/mode", {"mode": "assist"})
        assert status == 200
        assert data["success"] is True
        assert data["mode"] == "assist"

        # Switch to autonomous without confirmation
        status, data = _post(port, "/api/mode", {"mode": "autonomous"})
        assert data["success"] is False

        # With confirmation
        status, data = _post(port, "/api/mode", {"mode": "autonomous", "confirmed": True})
        assert data["success"] is True
        assert data["mode"] == "autonomous"

    def test_api_settings_get(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, body = _get(port, "/api/settings?game=cs2")
        assert status == 200
        data = json.loads(body)
        assert data["game_id"] == "cs2"
        assert "skill_level" in data
        assert "default_mode" in data

    def test_api_settings_update(self, daemon_and_panel):
        daemon, port, _ = daemon_and_panel

        # Update settings
        status, data = _post(port, "/api/settings", {
            "game_id": "cs2",
            "skill_level": 0.95,
            "default_mode": "assist",
        })
        assert status == 200
        assert data["success"] is True

        # Verify persisted
        st, body = _get(port, "/api/settings?game=cs2")
        settings = json.loads(body)
        assert settings["skill_level"] == 0.95
        assert settings["default_mode"] == "assist"

    def test_api_404(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, _ = _get(port, "/api/nonexistent")
        assert status == 404

    def test_api_settings_no_game(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, body = _get(port, "/api/settings")
        assert status == 400

    def test_html_contains_steam_instructions(self, daemon_and_panel):
        _, port, _ = daemon_and_panel
        status, body = _get(port, "/")
        assert "Steam Overlay" in body
        assert "Shift+Tab" in body
        assert "localhost:27060" in body
