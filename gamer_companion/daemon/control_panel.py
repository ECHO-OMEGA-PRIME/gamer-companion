"""Control Panel — Local web server for Steam Overlay integration.

Serves a Steam-styled control panel at http://localhost:27060 that can be
opened inside Steam's overlay browser (Shift+Tab → Web Browser).

Features:
- Real-time game detection status
- Mode switching (observe/assist/coach/copilot/autonomous)
- Per-game settings editor
- Live FSM state display
- Session stats and safety metrics
- One-click force-activate for any game

No extra dependencies — uses Python's built-in http.server.
"""

from __future__ import annotations
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from gamer_companion.daemon.game_watcher import GameWatcherDaemon

# The daemon reference — set by start_control_panel()
_daemon: Optional["GameWatcherDaemon"] = None
_tutorial_store = None  # Set by start_control_panel()

DEFAULT_PORT = 27060


class ControlPanelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the control panel."""

    def log_message(self, format, *args):
        # Suppress default stderr logging
        pass

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/panel":
            self._html_response(PANEL_HTML)
        elif path == "/tutorial":
            self._serve_tutorial()
        elif path == "/api/tutorial/progress":
            self._api_tutorial_progress()
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/games":
            self._api_games()
        elif path == "/api/settings":
            game_id = params.get("game", [None])[0]
            self._api_get_settings(game_id)
        elif path == "/health":
            self._json_response({"status": "ok", "uptime": time.time()})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_len > 0:
            raw = self.rfile.read(content_len)
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._json_response({"error": "invalid json"}, 400)
                return

        if path == "/api/mode":
            self._api_set_mode(body)
        elif path == "/api/activate":
            self._api_activate(body)
        elif path == "/api/deactivate":
            self._api_deactivate()
        elif path == "/api/settings":
            self._api_set_settings(body)
        elif path == "/api/tutorial/complete-step":
            self._api_tutorial_complete_step(body)
        elif path == "/api/tutorial/complete":
            self._api_tutorial_complete(body)
        else:
            self._json_response({"error": "not found"}, 404)

    # ── API handlers ────────────────────────────────────────────────────

    def _api_status(self):
        if not _daemon:
            self._json_response({"error": "daemon not connected"}, 503)
            return
        self._json_response(_daemon.get_stats())

    def _api_games(self):
        from gamer_companion.game_registry import GAME_REGISTRY
        games = []
        for gid, entry in GAME_REGISTRY.items():
            games.append({
                "game_id": gid,
                "name": entry.display_name,
                "genre": entry.genre.value,
                "esport_tier": entry.esport_tier,
            })
        games.sort(key=lambda g: g["name"])
        self._json_response({"games": games, "total": len(games)})

    def _api_get_settings(self, game_id: Optional[str]):
        if not _daemon or not game_id:
            self._json_response({"error": "game_id required"}, 400)
            return
        from dataclasses import asdict
        settings = _daemon._settings.get(game_id)
        self._json_response(asdict(settings))

    def _api_set_mode(self, body: dict):
        if not _daemon:
            self._json_response({"error": "daemon not connected"}, 503)
            return
        mode = body.get("mode", "")
        confirmed = body.get("confirmed", False)
        ok = _daemon.switch_mode(mode, confirmed=confirmed)
        self._json_response({
            "success": ok,
            "mode": _daemon.active_mode,
        })

    def _api_activate(self, body: dict):
        if not _daemon:
            self._json_response({"error": "daemon not connected"}, 503)
            return
        game_id = body.get("game_id", "")
        ok = _daemon.force_activate(game_id)
        self._json_response({
            "success": ok,
            "active_game": _daemon.active_game,
        })

    def _api_deactivate(self):
        if not _daemon:
            self._json_response({"error": "daemon not connected"}, 503)
            return
        _daemon.force_deactivate()
        self._json_response({"success": True, "active_game": None})

    def _api_set_settings(self, body: dict):
        if not _daemon:
            self._json_response({"error": "daemon not connected"}, 503)
            return
        game_id = body.pop("game_id", "")
        if not game_id:
            self._json_response({"error": "game_id required"}, 400)
            return
        _daemon.update_settings(game_id, **body)
        self._json_response({"success": True, "game_id": game_id})

    # ── Tutorial handlers ──────────────────────────────────────────────

    def _serve_tutorial(self):
        from gamer_companion.daemon.tutorial import get_tutorial_page_html
        self._html_response(get_tutorial_page_html())

    def _api_tutorial_progress(self):
        if not _tutorial_store:
            self._json_response({})
            return
        from dataclasses import asdict
        progress = _tutorial_store.load()
        self._json_response(asdict(progress))

    def _api_tutorial_complete_step(self, body: dict):
        if not _tutorial_store:
            self._json_response({"error": "tutorial store not initialized"}, 503)
            return
        tid = body.get("tutorial_id", "")
        step = body.get("step", 0)
        progress = _tutorial_store.load()
        progress.complete_step(tid, step)
        _tutorial_store.save(progress)
        self._json_response({"success": True})

    def _api_tutorial_complete(self, body: dict):
        if not _tutorial_store:
            self._json_response({"error": "tutorial store not initialized"}, 503)
            return
        tid = body.get("tutorial_id", "")
        progress = _tutorial_store.load()
        progress.complete_tutorial(tid)
        _tutorial_store.save(progress)
        self._json_response({"success": True})


def start_control_panel(
    daemon: "GameWatcherDaemon",
    port: int = DEFAULT_PORT,
) -> HTTPServer:
    """Start the control panel web server in a background thread.

    Returns the HTTPServer instance for shutdown.
    """
    global _daemon, _tutorial_store
    _daemon = daemon

    from gamer_companion.daemon.tutorial import TutorialStore
    settings_dir = getattr(daemon, '_config', None)
    settings_dir = getattr(settings_dir, 'settings_dir', None) if settings_dir else None
    _tutorial_store = TutorialStore(settings_dir=settings_dir)

    server = HTTPServer(("127.0.0.1", port), ControlPanelHandler)

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="ControlPanel",
    )
    thread.start()
    logger.info(
        f"Control Panel running at http://localhost:{port}\n"
        f"  Open in Steam Overlay browser (Shift+Tab) or any browser"
    )
    return server


# =============================================================================
# STEAM-STYLED SINGLE-PAGE CONTROL PANEL
# =============================================================================
PANEL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GGI APEX PREDATOR — Control Panel</title>
<style>
  :root {
    --bg-darkest: #171a21;
    --bg-dark: #1b2838;
    --bg-card: #1e2a3a;
    --bg-hover: #2a475e;
    --bg-input: #32404f;
    --accent: #66c0f4;
    --accent-glow: #4fc3f7;
    --accent-dim: #3b8aba;
    --green: #4caf50;
    --green-glow: #69f0ae;
    --orange: #ff9800;
    --red: #f44336;
    --text: #c7d5e0;
    --text-dim: #8f98a0;
    --text-bright: #ffffff;
    --border: #2a3f50;
    --radius: 4px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Motiva Sans', Arial, Helvetica, sans-serif;
    background: var(--bg-darkest);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Header ─────────────────────────────────────────── */
  .header {
    background: linear-gradient(180deg, var(--bg-dark) 0%, var(--bg-darkest) 100%);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .header h1 {
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 1px;
  }
  .header h1 span { color: var(--text-dim); font-weight: 300; }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
  }
  .status-pill.active { background: rgba(76,175,80,0.2); color: var(--green); }
  .status-pill.idle { background: rgba(143,152,160,0.2); color: var(--text-dim); }
  .status-pill .dot {
    width: 8px; height: 8px; border-radius: 50%;
    animation: pulse 2s infinite;
  }
  .status-pill.active .dot { background: var(--green); box-shadow: 0 0 6px var(--green-glow); }
  .status-pill.idle .dot { background: var(--text-dim); animation: none; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* ── Layout ─────────────────────────────────────────── */
  .container { max-width: 960px; margin: 0 auto; padding: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .grid.full { grid-template-columns: 1fr; }

  /* ── Cards ──────────────────────────────────────────── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
  }
  .card h2 {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }

  /* ── Active Game Card ───────────────────────────────── */
  .game-banner {
    background: linear-gradient(135deg, var(--bg-hover) 0%, var(--bg-card) 100%);
    border: 1px solid var(--accent-dim);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
    text-align: center;
  }
  .game-banner.none { border-color: var(--border); }
  .game-banner .game-name {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-bright);
    margin-bottom: 4px;
  }
  .game-banner .game-genre {
    font-size: 13px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .game-banner .game-meta {
    margin-top: 12px;
    display: flex;
    justify-content: center;
    gap: 24px;
    font-size: 12px;
    color: var(--text-dim);
  }
  .game-banner .game-meta strong { color: var(--text); }
  .no-game {
    font-size: 16px;
    color: var(--text-dim);
    padding: 24px 0;
  }

  /* ── Mode Buttons ───────────────────────────────────── */
  .mode-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  .mode-btn {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    padding: 10px 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }
  .mode-btn:hover { background: var(--bg-hover); border-color: var(--accent-dim); }
  .mode-btn.active {
    background: rgba(102,192,244,0.15);
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 8px rgba(102,192,244,0.1);
  }
  .mode-btn.danger { border-color: var(--orange); }
  .mode-btn.danger.active {
    background: rgba(255,152,0,0.15);
    border-color: var(--orange);
    color: var(--orange);
  }

  /* ── Stats ──────────────────────────────────────────── */
  .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid rgba(42,63,80,0.5);
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-row .label { color: var(--text-dim); }
  .stat-row .value { color: var(--text); font-weight: 500; }
  .stat-row .value.green { color: var(--green); }
  .stat-row .value.orange { color: var(--orange); }
  .stat-row .value.red { color: var(--red); }
  .stat-row .value.accent { color: var(--accent); }

  /* ── Game List ──────────────────────────────────────── */
  .game-list { max-height: 300px; overflow-y: auto; }
  .game-list::-webkit-scrollbar { width: 6px; }
  .game-list::-webkit-scrollbar-track { background: var(--bg-dark); }
  .game-list::-webkit-scrollbar-thumb { background: var(--bg-hover); border-radius: 3px; }
  .game-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    border-radius: var(--radius);
    cursor: pointer;
    font-size: 13px;
    transition: background 0.1s;
  }
  .game-item:hover { background: var(--bg-hover); }
  .game-item .name { color: var(--text); }
  .game-item .genre {
    font-size: 11px;
    color: var(--text-dim);
    background: var(--bg-input);
    padding: 2px 8px;
    border-radius: 8px;
  }

  /* ── Search ─────────────────────────────────────────── */
  .search-box {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    padding: 8px 12px;
    font-size: 13px;
    margin-bottom: 10px;
    outline: none;
  }
  .search-box:focus { border-color: var(--accent-dim); }
  .search-box::placeholder { color: var(--text-dim); }

  /* ── Settings Panel ─────────────────────────────────── */
  .settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .setting-item { display: flex; flex-direction: column; gap: 4px; }
  .setting-item label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
  }
  .setting-item input, .setting-item select {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    padding: 6px 10px;
    font-size: 13px;
    outline: none;
  }
  .setting-item input:focus, .setting-item select:focus {
    border-color: var(--accent-dim);
  }
  .save-btn {
    margin-top: 12px;
    background: linear-gradient(135deg, var(--accent-dim), var(--accent));
    border: none;
    border-radius: var(--radius);
    color: var(--bg-darkest);
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 1px;
    width: 100%;
  }
  .save-btn:hover { box-shadow: 0 0 12px rgba(102,192,244,0.3); }

  /* ── FSM State ──────────────────────────────────────── */
  .fsm-state {
    display: inline-block;
    background: rgba(102,192,244,0.15);
    border: 1px solid var(--accent-dim);
    border-radius: 12px;
    padding: 4px 14px;
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* ── Footer ─────────────────────────────────────────── */
  .footer {
    text-align: center;
    padding: 16px;
    font-size: 11px;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
    margin-top: 20px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>GGI APEX PREDATOR <span>Control Panel</span></h1>
  <div style="display:flex;align-items:center;gap:12px">
    <a href="/tutorial" style="color:var(--accent);text-decoration:none;font-size:13px;font-weight:600;padding:6px 14px;border:1px solid var(--accent);border-radius:var(--radius)">Tutorials</a>
    <div id="daemon-status" class="status-pill idle">
      <div class="dot"></div>
      <span>IDLE</span>
    </div>
  </div>
</div>

<div class="container">

  <!-- Active Game Banner -->
  <div id="game-banner" class="game-banner none">
    <div class="no-game">Watching for games...</div>
  </div>

  <div class="grid">
    <!-- Mode Selector -->
    <div class="card">
      <h2>Play Mode</h2>
      <div class="mode-grid" id="mode-grid">
        <button class="mode-btn active" data-mode="observe">Observe</button>
        <button class="mode-btn" data-mode="assist">Assist</button>
        <button class="mode-btn" data-mode="coach">Coach</button>
        <button class="mode-btn" data-mode="copilot">Copilot</button>
        <button class="mode-btn danger" data-mode="autonomous">Auto</button>
        <button class="mode-btn" data-mode="training">Training</button>
        <button class="mode-btn danger" data-mode="mimic">Mimic</button>
        <button class="mode-btn danger" data-mode="swarm">Swarm</button>
      </div>
    </div>

    <!-- Live Stats -->
    <div class="card">
      <h2>Session Stats</h2>
      <div id="stats-panel">
        <div class="stat-row"><span class="label">Duration</span><span class="value" id="stat-duration">—</span></div>
        <div class="stat-row"><span class="label">FSM State</span><span class="value accent" id="stat-fsm">—</span></div>
        <div class="stat-row"><span class="label">Skill Level</span><span class="value" id="stat-skill">—</span></div>
        <div class="stat-row"><span class="label">Safety</span><span class="value green" id="stat-safety">—</span></div>
        <div class="stat-row"><span class="label">Actions</span><span class="value" id="stat-actions">—</span></div>
        <div class="stat-row"><span class="label">Throttled</span><span class="value" id="stat-throttled">—</span></div>
        <div class="stat-row"><span class="label">APS</span><span class="value" id="stat-aps">—</span></div>
        <div class="stat-row"><span class="label">Games Detected</span><span class="value" id="stat-detected">—</span></div>
      </div>
    </div>
  </div>

  <div class="grid">
    <!-- Game Settings -->
    <div class="card">
      <h2>Game Settings <span id="settings-game" style="color:var(--accent)"></span></h2>
      <div class="settings-grid" id="settings-form">
        <div class="setting-item">
          <label>Skill Level</label>
          <input type="range" id="set-skill" min="0" max="1" step="0.05" value="0.7">
        </div>
        <div class="setting-item">
          <label>Aggression</label>
          <input type="range" id="set-aggression" min="0" max="1" step="0.05" value="0.5">
        </div>
        <div class="setting-item">
          <label>Default Mode</label>
          <select id="set-mode">
            <option value="observe">Observe</option>
            <option value="assist">Assist</option>
            <option value="coach">Coach</option>
            <option value="copilot">Copilot</option>
            <option value="autonomous">Autonomous</option>
            <option value="training">Training</option>
            <option value="mimic">Mimic</option>
          </select>
        </div>
        <div class="setting-item">
          <label>Session Limit (hrs)</label>
          <input type="number" id="set-session" min="0.5" max="24" step="0.5" value="4">
        </div>
        <div class="setting-item">
          <label>Reaction Time (ms)</label>
          <input type="number" id="set-reaction" min="50" max="500" step="10" value="150">
        </div>
        <div class="setting-item">
          <label>DPI</label>
          <input type="number" id="set-dpi" min="100" max="6400" step="100" value="800">
        </div>
      </div>
      <button class="save-btn" onclick="saveSettings()">Save Settings</button>
    </div>

    <!-- Game Library -->
    <div class="card">
      <h2>Game Library</h2>
      <input class="search-box" type="text" id="game-search" placeholder="Search games..." oninput="filterGames()">
      <div class="game-list" id="game-list"></div>
    </div>
  </div>

</div>

<div class="footer">
  GGI APEX PREDATOR v1.0.0 &mdash; ECHO OMEGA PRIME &mdash;
  Steam Overlay: Shift+Tab &rarr; Web Browser &rarr; localhost:27060
</div>

<script>
const API = '';  // same origin
let currentGameId = null;
let allGames = [];

// ── Polling ──────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch(API + '/api/status');
    const data = await r.json();
    updateUI(data);
  } catch (e) {
    document.getElementById('daemon-status').className = 'status-pill idle';
    document.getElementById('daemon-status').querySelector('span').textContent = 'OFFLINE';
  }
}

function updateUI(data) {
  const pill = document.getElementById('daemon-status');
  const banner = document.getElementById('game-banner');

  if (data.active_session) {
    const s = data.active_session;
    pill.className = 'status-pill active';
    pill.querySelector('span').textContent = 'ACTIVE';

    banner.className = 'game-banner';
    banner.innerHTML = `
      <div class="game-name">${s.game}</div>
      <div class="game-genre">${s.genre}</div>
      <div class="game-meta">
        <span>Mode: <strong>${s.mode || '—'}</strong></span>
        <span>Duration: <strong>${s.duration_min} min</strong></span>
        <span>Skill: <strong>${s.skill_level}</strong></span>
        <span>FSM: <strong class="fsm-state">${s.fsm_state || '—'}</strong></span>
      </div>
    `;

    // Update mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === s.mode);
    });

    // Update stats
    el('stat-duration', s.duration_min + ' min');
    el('stat-fsm', s.fsm_state || '—');
    el('stat-skill', s.skill_level);
    if (s.safety) {
      const sf = s.safety;
      el('stat-safety', sf.killed ? 'KILLED' : sf.paused ? 'PAUSED' : 'ACTIVE');
      document.getElementById('stat-safety').className =
        'value ' + (sf.killed ? 'red' : sf.paused ? 'orange' : 'green');
      el('stat-actions', sf.total_actions);
      el('stat-throttled', sf.throttled_actions);
      el('stat-aps', sf.current_aps + ' / ' + sf.max_aps);
    }

    // Load settings for active game
    if (currentGameId !== s.game_id) {
      currentGameId = s.game_id;
      loadSettings(s.game_id);
    }
  } else {
    pill.className = 'status-pill idle';
    pill.querySelector('span').textContent = 'WATCHING';
    banner.className = 'game-banner none';
    banner.innerHTML = '<div class="no-game">Watching for games... (' +
      data.games_in_registry + ' in registry' +
      (data.steam_available ? ', Steam connected' : '') + ')</div>';
    currentGameId = null;
  }

  el('stat-detected', data.total_games_detected);
}

function el(id, text) {
  const e = document.getElementById(id);
  if (e) e.textContent = text;
}

// ── Mode Switching ───────────────────────────────────────────────────
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const mode = btn.dataset.mode;
    const isDanger = btn.classList.contains('danger');
    if (isDanger && !confirm('Enable ' + mode.toUpperCase() + ' mode? This gives AI full control.')) return;

    const r = await fetch(API + '/api/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ mode, confirmed: isDanger }),
    });
    const data = await r.json();
    if (data.success) {
      document.querySelectorAll('.mode-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === data.mode));
    }
  });
});

// ── Game Library ─────────────────────────────────────────────────────
async function loadGames() {
  const r = await fetch(API + '/api/games');
  const data = await r.json();
  allGames = data.games;
  renderGames(allGames);
}

function renderGames(games) {
  const list = document.getElementById('game-list');
  list.innerHTML = games.map(g => `
    <div class="game-item" onclick="activateGame('${g.game_id}')">
      <span class="name">${g.name}</span>
      <span class="genre">${g.genre}</span>
    </div>
  `).join('');
}

function filterGames() {
  const q = document.getElementById('game-search').value.toLowerCase();
  renderGames(allGames.filter(g =>
    g.name.toLowerCase().includes(q) || g.genre.toLowerCase().includes(q)
  ));
}

async function activateGame(gameId) {
  if (!confirm('Force-activate ' + gameId + '?')) return;
  await fetch(API + '/api/activate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ game_id: gameId }),
  });
  poll();
}

// ── Settings ─────────────────────────────────────────────────────────
async function loadSettings(gameId) {
  const r = await fetch(API + '/api/settings?game=' + gameId);
  const s = await r.json();
  document.getElementById('settings-game').textContent = '(' + gameId + ')';
  document.getElementById('set-skill').value = s.skill_level || 0.7;
  document.getElementById('set-aggression').value = s.aggression || 0.5;
  document.getElementById('set-mode').value = s.default_mode || 'observe';
  document.getElementById('set-session').value = s.max_session_hours || 4;
  document.getElementById('set-reaction').value = s.reaction_time_ms || 150;
  document.getElementById('set-dpi').value = s.dpi || 800;
}

async function saveSettings() {
  const gameId = currentGameId;
  if (!gameId) { alert('No active game'); return; }
  await fetch(API + '/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      game_id: gameId,
      skill_level: parseFloat(document.getElementById('set-skill').value),
      aggression: parseFloat(document.getElementById('set-aggression').value),
      default_mode: document.getElementById('set-mode').value,
      max_session_hours: parseFloat(document.getElementById('set-session').value),
      reaction_time_ms: parseFloat(document.getElementById('set-reaction').value),
      dpi: parseInt(document.getElementById('set-dpi').value),
    }),
  });
  // Visual feedback
  const btn = document.querySelector('.save-btn');
  btn.textContent = 'SAVED';
  setTimeout(() => btn.textContent = 'SAVE SETTINGS', 1500);
}

// ── Init ─────────────────────────────────────────────────────────────
loadGames();
poll();
setInterval(poll, 2000);
</script>
</body>
</html>"""
