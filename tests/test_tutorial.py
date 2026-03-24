"""Test the interactive tutorial system."""

import json
import time
import pytest

from gamer_companion.daemon.tutorial import (
    TutorialStore,
    TutorialProgress,
    get_all_tutorials,
    get_tutorial_by_id,
    get_tutorial_page_html,
)
from gamer_companion.daemon.game_watcher import GameWatcherDaemon, DaemonConfig
from gamer_companion.daemon.control_panel import start_control_panel
from http.client import HTTPConnection


class TestTutorialData:
    def test_all_tutorials_loaded(self):
        tutorials = get_all_tutorials()
        assert len(tutorials) >= 7

    def test_tutorials_have_steps(self):
        for t in get_all_tutorials():
            assert len(t.steps) >= 3, f"Tutorial {t.id} has too few steps"

    def test_tutorial_ids_unique(self):
        tutorials = get_all_tutorials()
        ids = [t.id for t in tutorials]
        assert len(ids) == len(set(ids))

    def test_step_ids_unique_within_tutorial(self):
        for t in get_all_tutorials():
            step_ids = [s.id for s in t.steps]
            assert len(step_ids) == len(set(step_ids)), f"Duplicate step IDs in {t.id}"

    def test_get_tutorial_by_id(self):
        t = get_tutorial_by_id("getting-started")
        assert t is not None
        assert t.title == "Getting Started"

    def test_get_tutorial_by_id_not_found(self):
        t = get_tutorial_by_id("nonexistent")
        assert t is None

    def test_categories_present(self):
        cats = set(t.category for t in get_all_tutorials())
        assert "Basics" in cats
        assert "Play Modes" in cats
        assert "Safety" in cats

    def test_difficulties_valid(self):
        valid = {"beginner", "intermediate", "advanced"}
        for t in get_all_tutorials():
            assert t.difficulty in valid, f"{t.id} has invalid difficulty: {t.difficulty}"


class TestTutorialProgress:
    def test_default_progress(self):
        p = TutorialProgress()
        assert p.completed_tutorials == []
        assert p.current_tutorial is None
        assert p.current_step == 0

    def test_complete_step(self):
        p = TutorialProgress()
        p.complete_step("getting-started", 0)
        p.complete_step("getting-started", 1)
        assert p.step_completions["getting-started"] == [0, 1]

    def test_complete_step_idempotent(self):
        p = TutorialProgress()
        p.complete_step("getting-started", 0)
        p.complete_step("getting-started", 0)
        assert p.step_completions["getting-started"] == [0]

    def test_complete_tutorial(self):
        p = TutorialProgress()
        p.complete_tutorial("getting-started")
        assert p.is_completed("getting-started")
        assert p.current_tutorial is None

    def test_progress_pct(self):
        p = TutorialProgress()
        p.complete_step("getting-started", 0)
        p.complete_step("getting-started", 1)
        assert p.get_progress_pct("getting-started", 7) == 28

    def test_progress_pct_zero_steps(self):
        p = TutorialProgress()
        assert p.get_progress_pct("whatever", 0) == 100


class TestTutorialStore:
    def test_save_and_load(self, tmp_path):
        store = TutorialStore(settings_dir=str(tmp_path))
        p = TutorialProgress()
        p.complete_step("getting-started", 0)
        p.complete_step("getting-started", 1)
        p.complete_tutorial("getting-started")
        store.save(p)

        loaded = store.load()
        assert loaded.is_completed("getting-started")
        assert loaded.step_completions["getting-started"] == [0, 1]

    def test_load_missing_file(self, tmp_path):
        store = TutorialStore(settings_dir=str(tmp_path))
        p = store.load()
        assert p.completed_tutorials == []

    def test_default_dir(self):
        store = TutorialStore()
        assert store._path.name == "tutorial_progress.json"


class TestTutorialHTML:
    def test_html_contains_tutorials(self):
        html = get_tutorial_page_html()
        assert "Getting Started" in html
        assert "Mastering Play Modes" in html
        assert "Safety Layer" in html
        assert "TUTORIAL_DATA_PLACEHOLDER" not in html

    def test_html_contains_json_data(self):
        html = get_tutorial_page_html()
        assert '"getting-started"' in html
        assert '"play-modes"' in html


@pytest.fixture
def tutorial_panel(tmp_path):
    """Create a daemon + control panel with tutorial support."""
    cfg = DaemonConfig(
        steam_enabled=False,
        settings_dir=str(tmp_path / "settings"),
    )
    daemon = GameWatcherDaemon(config=cfg)
    daemon._start_time = time.time()

    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = start_control_panel(daemon, port=port)
    time.sleep(0.2)

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


class TestTutorialAPI:
    def test_tutorial_page_serves(self, tutorial_panel):
        _, port, _ = tutorial_panel
        status, body = _get(port, "/tutorial")
        assert status == 200
        assert "Interactive Tutorials" in body
        assert "Getting Started" in body

    def test_tutorial_progress_starts_empty_or_valid(self, tutorial_panel):
        _, port, _ = tutorial_panel
        status, body = _get(port, "/api/tutorial/progress")
        assert status == 200
        data = json.loads(body)
        assert isinstance(data["completed_tutorials"], list)

    def test_complete_step_and_tutorial(self, tutorial_panel):
        _, port, _ = tutorial_panel

        # Complete a step
        status, data = _post(port, "/api/tutorial/complete-step", {
            "tutorial_id": "getting-started",
            "step": 0,
        })
        assert status == 200
        assert data["success"] is True

        # Check progress
        status, body = _get(port, "/api/tutorial/progress")
        progress = json.loads(body)
        assert 0 in progress["step_completions"]["getting-started"]

        # Complete tutorial
        status, data = _post(port, "/api/tutorial/complete", {
            "tutorial_id": "getting-started",
        })
        assert status == 200
        assert data["success"] is True

        # Verify completed
        status, body = _get(port, "/api/tutorial/progress")
        progress = json.loads(body)
        assert "getting-started" in progress["completed_tutorials"]

    def test_panel_has_tutorial_link(self, tutorial_panel):
        _, port, _ = tutorial_panel
        status, body = _get(port, "/")
        assert status == 200
        assert "/tutorial" in body
        assert "Tutorials" in body
