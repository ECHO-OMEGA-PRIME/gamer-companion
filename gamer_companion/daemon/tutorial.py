"""Interactive Tutorial System for GGI APEX PREDATOR.

Step-by-step guided walkthroughs for every feature of the gaming companion.
Serves an interactive tutorial page at /tutorial on the control panel.
Tracks tutorial progress per-user in a local JSON file.

Tutorial categories:
  1. Getting Started (setup, first launch, connecting Steam)
  2. Play Modes (each of the 8 modes with hands-on exercises)
  3. Control Panel (navigating the overlay, changing settings)
  4. Per-Game Setup (configuring settings for specific games)
  5. Safety & Ethics (understanding limits, kill switch, APS)
  6. Advanced (pro mimicry, spray training, swarm coordination)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from loguru import logger


@dataclass
class TutorialStep:
    """A single step within a tutorial."""
    id: str
    title: str
    content: str  # HTML content
    action: Optional[str] = None  # 'click_button', 'change_mode', 'activate_game', etc.
    action_target: Optional[str] = None  # CSS selector or identifier
    validation: Optional[str] = None  # JS expression to validate completion
    tip: Optional[str] = None


@dataclass
class Tutorial:
    """A complete tutorial with multiple steps."""
    id: str
    title: str
    description: str
    category: str
    difficulty: str  # beginner, intermediate, advanced
    estimated_minutes: int
    steps: list[TutorialStep] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class TutorialProgress:
    """Tracks user progress through tutorials."""
    completed_tutorials: list[str] = field(default_factory=list)
    current_tutorial: Optional[str] = None
    current_step: int = 0
    step_completions: dict[str, list[int]] = field(default_factory=dict)

    def complete_step(self, tutorial_id: str, step_idx: int):
        if tutorial_id not in self.step_completions:
            self.step_completions[tutorial_id] = []
        if step_idx not in self.step_completions[tutorial_id]:
            self.step_completions[tutorial_id].append(step_idx)

    def complete_tutorial(self, tutorial_id: str):
        if tutorial_id not in self.completed_tutorials:
            self.completed_tutorials.append(tutorial_id)
        self.current_tutorial = None
        self.current_step = 0

    def is_completed(self, tutorial_id: str) -> bool:
        return tutorial_id in self.completed_tutorials

    def get_progress_pct(self, tutorial_id: str, total_steps: int) -> int:
        if total_steps == 0:
            return 100
        done = len(self.step_completions.get(tutorial_id, []))
        return int((done / total_steps) * 100)


class TutorialStore:
    """Persists tutorial progress to disk."""

    def __init__(self, settings_dir: Optional[str] = None):
        base = Path(settings_dir) if settings_dir else Path.home() / ".gamer-companion"
        self._path = base / "tutorial_progress.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> TutorialProgress:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                return TutorialProgress(**data)
            except Exception:
                pass
        return TutorialProgress()

    def save(self, progress: TutorialProgress):
        self._path.write_text(json.dumps(asdict(progress), indent=2))


# ── Tutorial Definitions ─────────────────────────────────────────────────────

def _build_tutorials() -> list[Tutorial]:
    """Build all tutorial definitions."""
    return [
        Tutorial(
            id="getting-started",
            title="Getting Started",
            description="Install, launch, and connect your first game in under 5 minutes.",
            category="Basics",
            difficulty="beginner",
            estimated_minutes=5,
            steps=[
                TutorialStep(
                    id="gs-1",
                    title="Welcome to GGI Apex Predator",
                    content="<p>Welcome, Commander. GGI Apex Predator is an AI gaming companion that auto-detects your games, reads the screen in real-time, and helps you play better — or plays for you entirely.</p><p>This tutorial will walk you through setup, your first game session, and the control panel.</p>",
                    tip="You can press Escape at any time to exit the tutorial and come back later.",
                ),
                TutorialStep(
                    id="gs-2",
                    title="Verify Installation",
                    content="<p>The fact that you're seeing this means the daemon is running and the control panel is active. Let's check the status.</p><p>Look at the <strong>Status</strong> section at the top of the main panel. You should see:</p><ul><li>Daemon: <span style='color:#10b981'>Running</span></li><li>Games in Registry: <strong>45+</strong></li></ul>",
                    action="navigate",
                    action_target="/",
                    tip="If you see 'Not Running', try restarting with: gamer-companion --debug",
                ),
                TutorialStep(
                    id="gs-3",
                    title="Understanding the Game Library",
                    content="<p>Scroll down to the <strong>Game Library</strong> section. You'll see all 45+ supported games organized by genre.</p><p>Each game card shows:</p><ul><li>Game name and genre</li><li>Your current skill level setting</li><li>Default play mode</li><li>A button to force-activate</li></ul>",
                    tip="Use the search box to quickly find a specific game.",
                ),
                TutorialStep(
                    id="gs-4",
                    title="Force-Activate a Game",
                    content="<p>Let's try activating a game manually. Find <strong>Chess</strong> in the game library and click the <strong>Activate</strong> button.</p><p>This simulates what happens automatically when the daemon detects a game launch.</p>",
                    action="click_button",
                    action_target="activate-chess",
                    validation="status.active_session !== null",
                ),
                TutorialStep(
                    id="gs-5",
                    title="Check Active Session",
                    content="<p>Look at the top of the panel. You should now see an <strong>Active Game</strong> banner showing:</p><ul><li>Game: Chess</li><li>Mode: observe (default)</li><li>Session duration counting up</li><li>Safety metrics (APS, throttle count)</li></ul><p>The AI is now watching the game state.</p>",
                ),
                TutorialStep(
                    id="gs-6",
                    title="Deactivate the Session",
                    content="<p>Click the <strong>Deactivate</strong> button to end the session. In normal use, this happens automatically when you close the game.</p>",
                    action="click_button",
                    action_target="deactivate",
                    validation="status.active_session === null",
                ),
                TutorialStep(
                    id="gs-7",
                    title="Setup Complete!",
                    content="<p>You've completed the basics. The daemon will now automatically detect games when you launch them.</p><p><strong>Next steps:</strong></p><ul><li>Try the <em>Play Modes</em> tutorial to learn all 8 modes</li><li>Configure per-game settings for your favorite games</li><li>Learn about the Safety Layer</li></ul>",
                ),
            ],
        ),
        Tutorial(
            id="play-modes",
            title="Mastering Play Modes",
            description="Learn each of the 8 play modes — from passive observation to full AI autonomy.",
            category="Play Modes",
            difficulty="beginner",
            estimated_minutes=10,
            prerequisites=["getting-started"],
            steps=[
                TutorialStep(
                    id="pm-1",
                    title="The 8 Play Modes",
                    content="<p>GGI has 8 distinct play modes, each offering a different level of AI involvement:</p><ol><li><strong>Observe</strong> — AI watches silently, learns your patterns</li><li><strong>Assist</strong> — Contextual tips and recommendations</li><li><strong>Coach</strong> — Active real-time coaching</li><li><strong>Copilot</strong> — AI handles secondary tasks</li><li><strong>Autonomous</strong> — Full AI control (dangerous)</li><li><strong>Training</strong> — Structured practice drills</li><li><strong>Mimic</strong> — Replicates a pro player's style</li><li><strong>Swarm</strong> — Multi-agent team coordination</li></ol>",
                ),
                TutorialStep(
                    id="pm-2",
                    title="Safe Modes (No Risk)",
                    content="<p>These modes never touch your game inputs:</p><ul><li><strong>Observe:</strong> Pure data collection. The AI builds a profile of your play style for future coaching.</li><li><strong>Assist:</strong> Overlay tips appear (buy suggestions, rotation timers, enemy positions). You make all decisions.</li><li><strong>Coach:</strong> Voice or text coaching. 'Rotate A site', 'Save this round', 'Crosshair higher'. Active guidance.</li><li><strong>Training:</strong> Dedicated drill mode. Spray patterns, build speed, APM exercises with scoring.</li></ul>",
                    tip="Start with Observe or Assist until you're comfortable with the AI's analysis.",
                ),
                TutorialStep(
                    id="pm-3",
                    title="Try Switching to Assist Mode",
                    content="<p>First, activate a game. Then click the <strong>Assist</strong> button in the mode panel.</p><p>Watch how the status updates to show the new mode.</p>",
                    action="change_mode",
                    action_target="assist",
                ),
                TutorialStep(
                    id="pm-4",
                    title="Elevated Modes (Caution)",
                    content="<p>These modes control your mouse and keyboard:</p><ul><li><strong>Copilot:</strong> AI handles secondary tasks (ability combos, resource buys, map pings) while you handle primary combat.</li><li><strong>Mimic:</strong> Loads a pro player profile. Replicates their crosshair placement, movement patterns, and decision-making.</li></ul><p>These use human-like input patterns (Gaussian jitter, Bezier curves) to appear natural.</p>",
                ),
                TutorialStep(
                    id="pm-5",
                    title="Dangerous Modes (Confirmation Required)",
                    content="<p>These modes require explicit confirmation:</p><ul><li><strong>Autonomous:</strong> Full AI control. Plays the game at superhuman speed with perfect mechanics.</li><li><strong>Swarm:</strong> Coordinates multiple agents across team members.</li></ul><p>When you try to activate these, you'll see a confirmation dialog. This is the Safety Layer protecting you.</p>",
                    tip="Press F12 at ANY time to instantly kill all AI input (the kill switch).",
                ),
                TutorialStep(
                    id="pm-6",
                    title="Mode Switching Complete",
                    content="<p>You now understand all 8 modes. Key takeaways:</p><ul><li>Start with <strong>Observe</strong> or <strong>Assist</strong> for zero risk</li><li><strong>Coach</strong> and <strong>Training</strong> actively improve your skills</li><li><strong>Copilot</strong> and <strong>Mimic</strong> provide AI-assisted play</li><li><strong>Autonomous</strong> and <strong>Swarm</strong> are full AI control</li><li>The <strong>F12 kill switch</strong> stops everything instantly</li></ul>",
                ),
            ],
        ),
        Tutorial(
            id="control-panel",
            title="Control Panel Deep Dive",
            description="Master every feature of the Steam Overlay control panel.",
            category="Control Panel",
            difficulty="beginner",
            estimated_minutes=7,
            prerequisites=["getting-started"],
            steps=[
                TutorialStep(
                    id="cp-1",
                    title="Accessing the Panel",
                    content="<p>The control panel runs at <code>http://localhost:27060</code>. Access it via:</p><ul><li><strong>In-game:</strong> Shift+Tab → Web Browser → localhost:27060</li><li><strong>Desktop:</strong> Open any browser → localhost:27060</li></ul><p>The panel auto-refreshes every 2 seconds, so you always see live data.</p>",
                ),
                TutorialStep(
                    id="cp-2",
                    title="Status Dashboard",
                    content="<p>The top section shows:</p><ul><li><strong>Daemon status</strong> — Running/Stopped</li><li><strong>Active game</strong> — Currently detected game (or 'No game active')</li><li><strong>Current mode</strong> — Which of the 8 modes is active</li><li><strong>Session duration</strong> — How long the current game has been running</li><li><strong>Safety metrics</strong> — Actions per second, throttle count, safety status</li></ul>",
                ),
                TutorialStep(
                    id="cp-3",
                    title="Mode Buttons",
                    content="<p>The mode panel shows 8 colored buttons — one for each play mode. The active mode is highlighted.</p><p>Click any button to switch modes instantly. Dangerous modes (Autonomous, Swarm) show a confirmation popup first.</p>",
                ),
                TutorialStep(
                    id="cp-4",
                    title="Per-Game Settings",
                    content="<p>When a game is active, the settings panel shows:</p><ul><li><strong>Skill Level</strong> (0.0 - 1.0) — How skilled the AI behaves</li><li><strong>Aggression</strong> (0.0 - 1.0) — How aggressive the play style</li><li><strong>DPI</strong> — Mouse sensitivity for input simulation</li><li><strong>Reaction Time</strong> (ms) — Simulated human reaction time</li><li><strong>Default Mode</strong> — Mode to start in when game is detected</li><li><strong>Session Limit</strong> (hours) — Auto-stop after this duration</li></ul><p>Changes save automatically per-game.</p>",
                ),
                TutorialStep(
                    id="cp-5",
                    title="Game Library",
                    content="<p>The bottom section lists all 45+ supported games. Each card shows:</p><ul><li>Game name and genre tag</li><li>A quick-activate button</li></ul><p>Use the <strong>search box</strong> to filter games by name. Click <strong>Activate</strong> to manually start a session without the game running.</p>",
                ),
                TutorialStep(
                    id="cp-6",
                    title="Panel Mastered!",
                    content="<p>You now know every section of the control panel:</p><ul><li>Status dashboard for live monitoring</li><li>Mode buttons for instant switching</li><li>Settings panel for per-game configuration</li><li>Game library for manual activation</li></ul><p>All of this works inside the Steam Overlay too.</p>",
                ),
            ],
        ),
        Tutorial(
            id="game-settings",
            title="Per-Game Configuration",
            description="Configure AI behavior for each game — skill, aggression, timing, and more.",
            category="Settings",
            difficulty="intermediate",
            estimated_minutes=8,
            prerequisites=["control-panel"],
            steps=[
                TutorialStep(
                    id="gs2-1",
                    title="Why Per-Game Settings?",
                    content="<p>Every game is different. The skill level and aggression that works for Chess won't work for CS2.</p><p>GGI stores separate settings for each game at <code>~/.gamer-companion/game_settings/</code>. Each file is a JSON document that persists between sessions.</p>",
                ),
                TutorialStep(
                    id="gs2-2",
                    title="Skill Level",
                    content="<p><strong>Skill Level (0.0 — 1.0)</strong></p><ul><li><strong>0.0 — 0.3:</strong> Casual. Slow reactions, basic strategies.</li><li><strong>0.3 — 0.6:</strong> Average. Solid fundamentals, standard play.</li><li><strong>0.6 — 0.8:</strong> Advanced. Fast reactions, complex strategies.</li><li><strong>0.8 — 1.0:</strong> Pro. Superhuman speed, optimal decisions.</li></ul><p>In Mimic mode, this is overridden by the pro player profile.</p>",
                    tip="Start at 0.5 and adjust based on your rank and comfort level.",
                ),
                TutorialStep(
                    id="gs2-3",
                    title="Aggression",
                    content="<p><strong>Aggression (0.0 — 1.0)</strong></p><ul><li><strong>Low (0.0 — 0.3):</strong> Defensive play, safe rotations, economy saving.</li><li><strong>Medium (0.3 — 0.7):</strong> Balanced approach, contextual aggression.</li><li><strong>High (0.7 — 1.0):</strong> Push-heavy, rush strategies, aggressive peeks.</li></ul>",
                ),
                TutorialStep(
                    id="gs2-4",
                    title="Reaction Time & DPI",
                    content="<p><strong>Reaction Time (ms):</strong> How fast the AI reacts. Default is 180ms (human average). Lower = faster but more suspicious. Minimum 80ms.</p><p><strong>DPI:</strong> Mouse sensitivity for input simulation. Match this to your actual in-game sensitivity for accurate Bezier mouse movements.</p>",
                ),
                TutorialStep(
                    id="gs2-5",
                    title="Session Limits",
                    content="<p><strong>Max Session Hours:</strong> The Safety Layer auto-deactivates after this many hours. Default is 8. Set to 0 for unlimited.</p><p>This prevents marathon AI sessions that could flag your account.</p>",
                ),
                TutorialStep(
                    id="gs2-6",
                    title="Saving Settings",
                    content="<p>Settings save automatically when you change them in the control panel. You can also edit the JSON files directly:</p><p><code>~/.gamer-companion/game_settings/cs2.json</code></p><p>Settings persist across daemon restarts.</p>",
                ),
            ],
        ),
        Tutorial(
            id="safety-layer",
            title="Safety Layer & Ethics",
            description="Understand the safety systems that protect you while using AI assistance.",
            category="Safety",
            difficulty="intermediate",
            estimated_minutes=6,
            prerequisites=["play-modes"],
            steps=[
                TutorialStep(
                    id="sl-1",
                    title="The Safety Layer",
                    content="<p>Every action the AI takes passes through the Safety Layer. It enforces:</p><ul><li><strong>APS Limits:</strong> Maximum actions per second (prevents inhuman speed)</li><li><strong>Kill Switch:</strong> F12 instantly stops ALL AI input</li><li><strong>Session Caps:</strong> Auto-stop after configured hours</li><li><strong>Audit Trail:</strong> Every action is logged with timestamp</li><li><strong>Mode Gates:</strong> Dangerous modes require confirmation</li></ul>",
                ),
                TutorialStep(
                    id="sl-2",
                    title="The Kill Switch (F12)",
                    content="<p>Press <strong>F12</strong> at any time to immediately halt all AI input. This is a hardware-level interrupt that:</p><ul><li>Stops all mouse movement</li><li>Releases all held keys</li><li>Pauses the autonomous controller</li><li>Logs the kill switch activation</li></ul><p>The AI remains connected but takes zero actions until you re-enable it from the control panel.</p>",
                    tip="Memorize F12. It's your emergency brake.",
                ),
                TutorialStep(
                    id="sl-3",
                    title="APS (Actions Per Second)",
                    content="<p>Humans average 2-5 meaningful actions per second. The Safety Layer caps the AI at configurable limits:</p><ul><li><strong>Default:</strong> 4 APS</li><li><strong>Training mode:</strong> 8 APS (drills need speed)</li><li><strong>Autonomous:</strong> 6 APS (above human but not robotic)</li></ul><p>If the AI exceeds the limit, it's throttled. The throttle count appears in the status panel.</p>",
                ),
                TutorialStep(
                    id="sl-4",
                    title="Mode Confirmation Gates",
                    content="<p>Switching to dangerous modes requires confirmation:</p><ul><li><strong>Autonomous:</strong> 'This mode takes full control of your input. Continue?'</li><li><strong>Swarm:</strong> 'This mode coordinates with other agents. Continue?'</li><li><strong>Mimic:</strong> 'This mode replicates a pro player profile. Continue?'</li></ul><p>Or set <code>auto_mode_confirmed: true</code> in per-game settings to skip the prompt.</p>",
                ),
                TutorialStep(
                    id="sl-5",
                    title="Responsible Use",
                    content="<p>AI gaming assistance exists in a gray area. Key guidelines:</p><ul><li>Observe, Assist, Coach, and Training modes are generally safe</li><li>Copilot and higher modes may violate game ToS</li><li>Use at your own discretion and risk</li><li>Never use in competitive ranked games without understanding the risks</li><li>The Safety Layer protects you mechanically, not legally</li></ul>",
                ),
            ],
        ),
        Tutorial(
            id="advanced-features",
            title="Advanced Features",
            description="Pro mimicry, spray training, and multi-agent swarm coordination.",
            category="Advanced",
            difficulty="advanced",
            estimated_minutes=12,
            prerequisites=["play-modes", "safety-layer"],
            steps=[
                TutorialStep(
                    id="af-1",
                    title="Pro Player Mimicry",
                    content="<p>Mimic mode loads a pro player profile and replicates their:</p><ul><li>Crosshair placement patterns</li><li>Movement and positioning preferences</li><li>Economy decisions and buy patterns</li><li>Rotation timing and map control</li><li>Utility usage (smokes, flashes, etc.)</li></ul><p>Profiles are stored as JSON files with statistical models of each behavior.</p>",
                ),
                TutorialStep(
                    id="af-2",
                    title="Loading a Pro Profile",
                    content="<p>To use Mimic mode:</p><ol><li>Activate a game (e.g., CS2)</li><li>Switch to Mimic mode</li><li>Select a pro player profile from the dropdown</li><li>The AI calibrates to their play style</li></ol><p>Available profiles include top players from CS2, Valorant, and League of Legends. Community profiles can be shared as JSON files.</p>",
                ),
                TutorialStep(
                    id="af-3",
                    title="Spray Pattern Training",
                    content="<p>Training mode includes interactive recoil control drills:</p><ol><li>A target appears on screen</li><li>The drill starts spraying with a specific weapon</li><li>You control the mouse to counteract recoil</li><li>Real-time accuracy percentage shown</li><li>After each drill, a heatmap shows your spray vs. ideal</li></ol><p>Weapons include: AK-47, M4A4, M4A1-S, Phantom, Vandal, and more.</p>",
                ),
                TutorialStep(
                    id="af-4",
                    title="Swarm Mode Basics",
                    content="<p>Swarm mode coordinates multiple GGI instances across team members:</p><ul><li>Synchronized pushes (everyone enters at the same time)</li><li>Coordinated utility (flash + smoke + entry)</li><li>Call-out sharing (enemy positions broadcast to all agents)</li><li>Role assignment (entry, support, AWP, lurk)</li></ul><p>Requires all team members to run GGI with Swarm mode enabled.</p>",
                ),
                TutorialStep(
                    id="af-5",
                    title="The Learning System",
                    content="<p>Every session feeds into the learning system:</p><ul><li><strong>Action logs:</strong> Every AI action recorded with context</li><li><strong>Outcome tracking:</strong> Win/loss/KDA correlated with actions</li><li><strong>Pattern extraction:</strong> Identifies your strengths and weaknesses</li><li><strong>Model updates:</strong> The AI improves after each game</li></ul><p>Over time, the AI adapts to your specific play style and the opponents you face.</p>",
                ),
                TutorialStep(
                    id="af-6",
                    title="Advanced Mastery Complete",
                    content="<p>You're now an advanced user. You understand:</p><ul><li>Pro player profile loading and mimicry</li><li>Spray pattern training drills</li><li>Multi-agent swarm coordination</li><li>The continuous learning system</li></ul><p>Continue exploring. The AI gets better the more you use it.</p>",
                ),
            ],
        ),
        Tutorial(
            id="steam-integration",
            title="Steam Integration",
            description="Set up Steam Overlay browser access and game library detection.",
            category="Steam",
            difficulty="beginner",
            estimated_minutes=5,
            steps=[
                TutorialStep(
                    id="si-1",
                    title="Steam Overlay Setup",
                    content="<p>Steam's built-in overlay includes a web browser. GGI uses this to give you in-game access to the control panel.</p><p><strong>Requirements:</strong></p><ul><li>Steam must be running</li><li>Steam Overlay must be enabled (Steam → Settings → In Game → Enable Steam Overlay)</li><li>GGI daemon must be running</li></ul>",
                ),
                TutorialStep(
                    id="si-2",
                    title="Opening the Panel In-Game",
                    content="<p>While in any game:</p><ol><li>Press <strong>Shift+Tab</strong> to open Steam Overlay</li><li>Click <strong>Web Browser</strong> (bottom bar)</li><li>Type <strong>localhost:27060</strong> in the address bar</li><li>Press Enter</li></ol><p>The full control panel loads right inside the overlay. You can switch modes, change settings, and activate games without alt-tabbing.</p>",
                    tip="Bookmark localhost:27060 in the Steam browser for one-click access.",
                ),
                TutorialStep(
                    id="si-3",
                    title="Steam Game Detection",
                    content="<p>GGI queries the Steam Web API for additional game detection. This helps identify games that don't have obvious process names.</p><p>Steam integration is enabled by default. Disable it with <code>--no-steam</code> if needed.</p>",
                ),
                TutorialStep(
                    id="si-4",
                    title="Non-Steam Games",
                    content="<p>Games not in Steam still work. The daemon detects them by process name matching against the 45+ game registry.</p><p>You can also force-activate any game from the control panel or CLI:</p><p><code>gamer-companion --game chess</code></p>",
                ),
            ],
        ),
    ]


# ── Tutorial HTML Generator ──────────────────────────────────────────────────

def get_all_tutorials() -> list[Tutorial]:
    return _build_tutorials()


def get_tutorial_by_id(tutorial_id: str) -> Optional[Tutorial]:
    for t in _build_tutorials():
        if t.id == tutorial_id:
            return t
    return None


def generate_tutorial_html() -> str:
    """Generate the full interactive tutorial page HTML."""
    tutorials = _build_tutorials()
    return TUTORIAL_HTML


# ── Tutorial Page HTML ───────────────────────────────────────────────────────

TUTORIAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GGI APEX PREDATOR — Interactive Tutorials</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#1b2838;--bg2:#171a21;--card:#1e2a3a;--border:#2a475e;
  --text:#c7d5e0;--text2:#8f98a0;--accent:#66c0f4;--accent2:#1a9fff;
  --green:#5ba32b;--red:#c0392b;--orange:#f39c12;--purple:#9b59b6;
}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}
a{color:var(--accent);text-decoration:none}
.container{max-width:900px;margin:0 auto;padding:20px}
.header{text-align:center;padding:30px 0;border-bottom:1px solid var(--border)}
.header h1{font-size:28px;color:#fff;margin-bottom:5px}
.header p{color:var(--text2);font-size:14px}
.back-link{display:inline-block;margin-bottom:20px;font-size:13px;color:var(--accent)}
.progress-bar{background:var(--bg2);border-radius:8px;padding:15px 20px;margin:20px 0;display:flex;align-items:center;gap:15px;border:1px solid var(--border)}
.progress-bar .pct{font-size:24px;font-weight:800;color:var(--accent);min-width:60px;text-align:center}
.progress-bar .info{flex:1}
.progress-bar .bar{height:6px;background:var(--border);border-radius:3px;margin-top:5px}
.progress-bar .bar-fill{height:100%;background:var(--accent);border-radius:3px;transition:width 0.3s}
.cat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:15px;margin-top:20px}
.cat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;cursor:pointer;transition:border-color 0.2s}
.cat-card:hover{border-color:var(--accent)}
.cat-card.completed{border-color:var(--green)}
.cat-card h3{font-size:16px;color:#fff;margin-bottom:5px}
.cat-card p{font-size:12px;color:var(--text2);margin-bottom:10px}
.cat-card .meta{display:flex;gap:10px;font-size:11px;color:var(--text2)}
.cat-card .meta span{display:flex;align-items:center;gap:4px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;text-transform:uppercase}
.badge-beginner{background:rgba(91,163,43,0.2);color:var(--green)}
.badge-intermediate{background:rgba(243,156,18,0.2);color:var(--orange)}
.badge-advanced{background:rgba(155,89,182,0.2);color:var(--purple)}
.badge-complete{background:rgba(91,163,43,0.2);color:var(--green)}
.step-nav{display:flex;justify-content:space-between;align-items:center;margin:20px 0}
.step-counter{font-size:13px;color:var(--text2)}
.step-dots{display:flex;gap:6px}
.step-dot{width:10px;height:10px;border-radius:50%;background:var(--border);cursor:pointer;transition:all 0.2s}
.step-dot.active{background:var(--accent);transform:scale(1.3)}
.step-dot.done{background:var(--green)}
.step-content{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:30px;margin:15px 0;min-height:250px}
.step-content h2{font-size:22px;color:#fff;margin-bottom:15px}
.step-content p{margin-bottom:12px;font-size:14px}
.step-content ul,.step-content ol{margin:10px 0 10px 20px;font-size:14px}
.step-content li{margin-bottom:6px}
.step-content code{background:var(--bg2);padding:2px 6px;border-radius:4px;font-size:13px;color:var(--accent)}
.step-content strong{color:#fff}
.tip-box{background:rgba(102,192,244,0.1);border:1px solid rgba(102,192,244,0.3);border-radius:8px;padding:12px 16px;margin-top:15px;font-size:13px}
.tip-box::before{content:'TIP: ';font-weight:700;color:var(--accent)}
.btn{padding:10px 24px;border-radius:8px;border:none;font-size:14px;font-weight:600;cursor:pointer;transition:all 0.2s}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover{background:var(--accent2)}
.btn-secondary{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn-secondary:hover{border-color:var(--accent)}
.btn-green{background:var(--green);color:#fff}
.btn-group{display:flex;gap:10px;margin-top:20px;justify-content:flex-end}
#tutorial-view{display:none}
</style>
</head>
<body>
<div class="container">
  <!-- Tutorial List View -->
  <div id="list-view">
    <div class="header">
      <h1>GGI APEX PREDATOR</h1>
      <p>Interactive Tutorials — Learn every feature step by step</p>
    </div>
    <div class="progress-bar">
      <div class="pct" id="total-pct">0%</div>
      <div class="info">
        <div style="font-size:13px;color:#fff;font-weight:600">Overall Progress</div>
        <div style="font-size:11px;color:var(--text2)" id="total-label">0 of 7 tutorials completed</div>
        <div class="bar"><div class="bar-fill" id="total-bar" style="width:0%"></div></div>
      </div>
    </div>
    <div class="cat-grid" id="tutorial-grid"></div>
  </div>

  <!-- Tutorial Step View -->
  <div id="tutorial-view">
    <a href="#" class="back-link" onclick="showList(); return false;">← Back to Tutorials</a>
    <div class="step-nav">
      <div class="step-counter" id="step-counter">Step 1 of 7</div>
      <div class="step-dots" id="step-dots"></div>
    </div>
    <div class="step-content" id="step-content"></div>
    <div class="btn-group">
      <button class="btn btn-secondary" id="btn-prev" onclick="prevStep()">Previous</button>
      <button class="btn btn-primary" id="btn-next" onclick="nextStep()">Next</button>
    </div>
  </div>
</div>

<script>
const TUTORIALS = TUTORIAL_DATA_PLACEHOLDER;

let progress = {};
let currentTutorial = null;
let currentStep = 0;

// Load progress from API
async function loadProgress() {
  try {
    const res = await fetch('/api/tutorial/progress');
    if (res.ok) progress = await res.json();
  } catch(e) {}
  renderList();
}

function renderList() {
  const grid = document.getElementById('tutorial-grid');
  const completed = (progress.completed_tutorials || []).length;
  const total = TUTORIALS.length;
  document.getElementById('total-pct').textContent = Math.round((completed/total)*100) + '%';
  document.getElementById('total-label').textContent = completed + ' of ' + total + ' tutorials completed';
  document.getElementById('total-bar').style.width = Math.round((completed/total)*100) + '%';

  grid.innerHTML = TUTORIALS.map(t => {
    const done = (progress.completed_tutorials || []).includes(t.id);
    const stepsDone = ((progress.step_completions || {})[t.id] || []).length;
    const pct = Math.round((stepsDone / t.steps.length) * 100);
    return `
      <div class="cat-card ${done ? 'completed' : ''}" onclick="startTutorial('${t.id}')">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <span class="badge badge-${t.difficulty}">${t.difficulty}</span>
          ${done ? '<span class="badge badge-complete">Complete</span>' : ''}
        </div>
        <h3>${t.title}</h3>
        <p>${t.description}</p>
        <div class="bar" style="margin-bottom:8px"><div class="bar-fill" style="width:${pct}%;background:${done ? 'var(--green)' : 'var(--accent)'}"></div></div>
        <div class="meta">
          <span>${t.steps.length} steps</span>
          <span>${t.estimated_minutes} min</span>
          <span>${t.category}</span>
        </div>
      </div>`;
  }).join('');
}

function startTutorial(id) {
  currentTutorial = TUTORIALS.find(t => t.id === id);
  if (!currentTutorial) return;
  const stepsDone = (progress.step_completions || {})[id] || [];
  currentStep = stepsDone.length < currentTutorial.steps.length ? stepsDone.length : 0;
  document.getElementById('list-view').style.display = 'none';
  document.getElementById('tutorial-view').style.display = 'block';
  renderStep();
}

function showList() {
  document.getElementById('list-view').style.display = 'block';
  document.getElementById('tutorial-view').style.display = 'none';
  loadProgress();
}

function renderStep() {
  if (!currentTutorial) return;
  const step = currentTutorial.steps[currentStep];
  const total = currentTutorial.steps.length;
  document.getElementById('step-counter').textContent = `Step ${currentStep+1} of ${total} — ${currentTutorial.title}`;

  // Dots
  const dots = document.getElementById('step-dots');
  const stepsDone = (progress.step_completions || {})[currentTutorial.id] || [];
  dots.innerHTML = currentTutorial.steps.map((s, i) => {
    const cls = i === currentStep ? 'active' : stepsDone.includes(i) ? 'done' : '';
    return `<div class="step-dot ${cls}" onclick="goToStep(${i})"></div>`;
  }).join('');

  // Content
  let html = `<h2>${step.title}</h2>${step.content}`;
  if (step.tip) html += `<div class="tip-box">${step.tip}</div>`;
  document.getElementById('step-content').innerHTML = html;

  // Buttons
  document.getElementById('btn-prev').style.display = currentStep === 0 ? 'none' : '';
  const isLast = currentStep === total - 1;
  const btn = document.getElementById('btn-next');
  btn.textContent = isLast ? 'Complete Tutorial' : 'Next';
  btn.className = isLast ? 'btn btn-green' : 'btn btn-primary';
}

function goToStep(idx) {
  currentStep = idx;
  renderStep();
}

async function nextStep() {
  if (!currentTutorial) return;
  // Mark step complete
  await fetch('/api/tutorial/complete-step', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tutorial_id: currentTutorial.id, step: currentStep})
  });

  // Update local progress
  if (!progress.step_completions) progress.step_completions = {};
  if (!progress.step_completions[currentTutorial.id]) progress.step_completions[currentTutorial.id] = [];
  if (!progress.step_completions[currentTutorial.id].includes(currentStep))
    progress.step_completions[currentTutorial.id].push(currentStep);

  if (currentStep < currentTutorial.steps.length - 1) {
    currentStep++;
    renderStep();
  } else {
    // Complete tutorial
    await fetch('/api/tutorial/complete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tutorial_id: currentTutorial.id})
    });
    if (!progress.completed_tutorials) progress.completed_tutorials = [];
    if (!progress.completed_tutorials.includes(currentTutorial.id))
      progress.completed_tutorials.push(currentTutorial.id);
    showList();
  }
}

function prevStep() {
  if (currentStep > 0) { currentStep--; renderStep(); }
}

loadProgress();
</script>
</body>
</html>"""


def get_tutorial_page_html() -> str:
    """Return the tutorial HTML with tutorial data injected."""
    tutorials = _build_tutorials()
    data = []
    for t in tutorials:
        data.append({
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "category": t.category,
            "difficulty": t.difficulty,
            "estimated_minutes": t.estimated_minutes,
            "prerequisites": t.prerequisites,
            "steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "content": s.content,
                    "tip": s.tip,
                    "action": s.action,
                    "action_target": s.action_target,
                }
                for s in t.steps
            ],
        })
    json_data = json.dumps(data, indent=2)
    return TUTORIAL_HTML.replace("TUTORIAL_DATA_PLACEHOLDER", json_data)
