# GGI APEX PREDATOR — AI Architecture Manual

**Version 1.0.0 | Technical Reference for Developers and AI Engineers**
**By Echo Prime Technologies**

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Module Inventory](#2-module-inventory)
3. [Perception Pipeline](#3-perception-pipeline)
4. [Cognition Engine (3-Layer)](#4-cognition-engine)
5. [Action Pipeline](#5-action-pipeline)
6. [State Machine Framework](#6-state-machine-framework)
7. [Game Registry & Profile System](#7-game-registry--profile-system)
8. [Autonomous Controller Loop](#8-autonomous-controller-loop)
9. [Safety Layer Architecture](#9-safety-layer-architecture)
10. [Learning System](#10-learning-system)
11. [Steam Integration](#11-steam-integration)
12. [Daemon Architecture](#12-daemon-architecture)
13. [Control Panel API](#13-control-panel-api)
14. [Extension Points](#14-extension-points)
15. [Data Flow Diagrams](#15-data-flow-diagrams)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAME WATCHER DAEMON                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Process   │  │ Steam        │  │ Control Panel             │ │
│  │ Monitor   │→→│ Integration  │  │ (localhost:27060)         │ │
│  └────┬─────┘  └──────┬───────┘  └───────────────────────────┘ │
│       ↓               ↓                                        │
│  ┌────────────────────────┐                                     │
│  │    GAME REGISTRY       │  45+ games, 23 genres               │
│  │    + Game Settings     │  Per-game persistence               │
│  └──────────┬─────────────┘                                     │
│             ↓                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AUTONOMOUS CONTROLLER                        │   │
│  │                                                           │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │   │
│  │  │ PERCEIVE│ → │  THINK  │ → │   ACT   │ → │  LEARN  │ │   │
│  │  │         │   │         │   │         │   │         │ │   │
│  │  │ Screen  │   │ Reflex  │   │ Safety  │   │ Replay  │ │   │
│  │  │ Audio   │   │ Tactical│   │ Layer   │   │ Buffer  │ │   │
│  │  │ OCR     │   │ Strategy│   │ Input   │   │ Weights │ │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘ │   │
│  │                                                           │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ MODE MANAGER (8 modes) ←→ SAFETY LAYER (hard limits)│ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   GAME-SPECIFIC DATA                      │   │
│  │  FSM │ Spray Patterns │ Pro Profiles │ Drills │ Macros   │   │
│  │  19 FSMs │ 35 weapons  │ 22 pros     │ 50+    │ 23+     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Inventory

### 10 Packages, 67 Modules, ~16,000 Lines

| Package | Modules | Lines | Purpose |
|---------|---------|-------|---------|
| `foundation` | 3 | ~900 | Game profiles, screen capture, frame buffer |
| `temporal` | 4 | ~1,100 | Time-series analysis, event correlation |
| `predictive` | 4 | ~1,000 | Bayesian prediction, Monte Carlo tree search |
| `audio_intel` | 3 | ~750 | Audio capture, spatial audio, sound classification |
| `emotion` | 3 | ~800 | Tilt detection, momentum tracking, adaptation |
| `squad` | 3 | ~900 | Multi-agent coordination, role assignment |
| `replay_ai` | 3 | ~850 | Replay recording, pattern extraction, coaching |
| `learning` | 5 | ~1,400 | Thompson sampling, experience replay, Q-learning |
| `state_machine` | 2 | ~1,200 | Base FSM + 19 genre-specific FSMs |
| `autonomous` | 4 | ~950 | Controller, cognition, safety, mode manager |
| `input_control` | 3 | ~850 | Mouse/keyboard simulation, macro engine |
| `game_intelligence` | 5 | ~1,500 | Object detection, navigation, mini-map, parser |
| `aim` | 4 | ~1,100 | Humanized aim, spray compensation, pre-fire |
| `playstyle` | 3 | ~900 | Pro mimicry, personality mapping, style engine |
| `training_ground` | 3 | ~850 | Drill system, aim trainer, game-specific drills |
| `daemon` | 4 | ~1,200 | Auto-launch, settings, control panel, watcher |
| Root | 3 | ~200 | Init, main, orchestrator |

---

## 3. Perception Pipeline

The perception system captures the game state from the screen.

### Pipeline Steps

```
Screen Capture (mss, 15-30 FPS)
     ↓
Frame Buffer (circular, 300 frames)
     ↓
┌──────────────┬──────────────┬──────────────┐
│ Region Crop  │ Full Frame   │ Audio Capture │
│ (minimap,    │ Analysis     │ (spatial,     │
│  health,     │ (YOLO/OCR)   │  footsteps,   │
│  ammo, etc.) │              │  gunshots)    │
└──────┬───────┴──────┬───────┴──────┬────────┘
       ↓              ↓              ↓
   Region OCR    Object Detect   Sound Class
       ↓              ↓              ↓
       └──────────────┼──────────────┘
                      ↓
              GameState dataclass
              {detections, ocr_text, audio_events,
               health, ammo, game_phase, minimap}
```

### Key Classes

- **`ScreenCapture`** (`foundation/screen_capture.py`) — Uses `mss` for high-speed captures. Region-based for efficiency.
- **`FrameBuffer`** (`foundation/frame_buffer.py`) — Circular buffer of 300 frames for temporal analysis and replay.
- **`ElementDetector`** (`game_intelligence/element_detector.py`) — YOLO-based object detection for game elements (enemies, items, UI).
- **`UniversalParser`** (`game_intelligence/universal_parser.py`) — OCR + template matching for reading UI elements. 14 genre-specific region hint profiles.
- **`SpatialAudioEngine`** (`audio_intel/spatial_audio.py`) — Directional audio analysis for footstep/gunshot localization.

---

## 4. Cognition Engine (3-Layer)

**File:** `autonomous/cognition_engine.py`

The brain of the system uses a 3-layer decision architecture:

### Layer 1: Reflex Layer (< 50ms)
- Pre-programmed reactions to immediate threats
- Enemy in crosshair → fire
- Flash detected → look away
- Health critical → retreat
- No deliberation, pattern matching only

### Layer 2: Tactical Layer (50-500ms)
- Short-term planning within the current game phase
- Positioning relative to objectives
- Ability/utility usage timing
- Economy decisions
- Map rotation choices
- Uses FSM state + minimap data + team state

### Layer 3: Strategic Layer (500ms-5s)
- Long-term planning across game phases
- Team composition analysis
- Win condition evaluation
- Resource management (economy, loadout)
- Monte Carlo Tree Search for deep planning
- Updated every N seconds (configurable per game)

### Decision Priority

```
Reflex decisions OVERRIDE Tactical decisions OVERRIDE Strategic decisions.

If a reflex detects "enemy in crosshair", it fires immediately —
regardless of what the tactical or strategic layers want.
```

---

## 5. Action Pipeline

### Input Control Architecture

```
ActionPlan (from Cognition)
     ↓
Safety Layer Check
  ├── APS limit (max 15/s)
  ├── Reaction time floor (min 90ms)
  ├── Session limit check
  ├── Kill switch check
  ├── Micro-pause injection
  └── Full audit logging
     ↓ (if approved)
Input Controller
  ├── MouseController (Bezier curves, jitter, DPI-aware)
  ├── KeyboardController (hold timing, chord support)
  └── MacroEngine (multi-step sequences with timing)
     ↓
Win32 API (SendInput)
```

### Humanization

All inputs are humanized:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Aim curve | Bezier with 2 control points | Natural mouse path |
| Overshoot | 3-8% of distance | Humans overshoot, then correct |
| Tremor | ±0.5-2px at 8-12 Hz | Natural hand shake |
| Reaction variance | ±20% of base time | Not robotic |
| Fatigue | +5% reaction per hour | Humans slow down |
| Micro-pauses | 100-400ms every 30-60s | Hesitation |

---

## 6. State Machine Framework

### Base FSM

**File:** `state_machine/game_fsm.py`

```python
class GameFSM:
    def __init__(self):
        self._states: Dict[str, dict] = {}  # state_name → {transitions: [...]}
        self._current_state: str = ""

    def transition(self, target: str) -> bool
    def add_state(self, name: str, transitions: List[str])
```

### 19 Genre-Specific FSMs

| FSM | States | Example Flow |
|-----|--------|-------------|
| `TacticalFPSFSM` | 6 | buy → spawn → alive → combat → clutch → dead |
| `BattleRoyaleFSM` | 7 | drop → loot → rotate → combat → heal → endgame → dead |
| `MOBAFSM` | 6 | laning → roaming → teamfight → objective → base_defense → pushing |
| `CoDMultiplayerFSM` | 8 | pre_match → spawning → patrolling → combat → killstreak → objective → dead → match_end |
| `HeroShooterFSM` | 6 | hero_select → setup → pushing/defending → teamfight → regrouping → overtime |
| `ArenaShooterFSM` | 5 | spawning → hunting → combat → power_up → dead |
| `RTSFSM` | 6 | opening → expanding → aggression → defending → mid_game → late_game |
| `FightingGameFSM` | 6 | neutral → pressure → defending → combo → knockdown → round_end |
| `RacingFSM` | 7 | pre_race → starting_grid → racing → battling → pitting → incident → finish |
| `SportsFSM` | 6 | kickoff → offense → defense → set_piece → transition → timeout |
| `CardGameFSM` | 5 | mulligan → early_game → mid_game → late_game → combo_turn |
| `AutoBattlerFSM` | 5 | planning → shopping → positioning → combat → carousel |
| `SurvivalFSM` | 7 | spawning → gathering → building → crafting → exploring → combat → dead |
| `MMOFSM` | 7 | idle → questing → combat → dungeon → boss_fight → pvp → crafting_gathering |
| `SoulslikeFSM` | 7 | exploring → combat → boss_approach → boss_fight → resting → dead → victory |
| `ExtractionShooterFSM` | 7 | planning → infiltrating → looting → combat → extracting → healing → dead |
| `BoardGameFSM` | 5 | waiting → thinking → moving → endgame → game_over |
| `TurnBasedStrategyFSM` | 7 | turn_start → production → research → military → diplomacy → exploration → turn_end |
| `PuzzleGameFSM` | 5 | analyzing → planning → executing → danger → game_over |

---

## 7. Game Registry & Profile System

### GameRegistry (`game_registry.py`)

Central catalog of all supported games:

```python
@dataclass
class GameEntry:
    game_id: str           # "cs2", "chess", "league"
    display_name: str      # "Counter-Strike 2"
    genre: Genre           # Genre.TACTICAL_FPS
    process_names: list    # ["cs2.exe"]
    window_titles: list    # ["Counter-Strike 2"]
    capabilities: set      # {AIM_ASSIST, AUTONOMOUS_PLAY, ...}
    fsm_class: str         # "TacticalFPSFSM"
    esport_tier: int       # 0-5 (5 = highest)
    has_spray_patterns: bool
    has_pro_profiles: bool
    has_drills: bool
    has_macros: bool
    has_nav_routes: bool
```

### 23 Genre Classifications

```
TACTICAL_FPS, BATTLE_ROYALE, MOBA, COD_MULTIPLAYER, HERO_SHOOTER,
ARENA_SHOOTER, RTS, FIGHTING, RACING, SPORTS, CARD, AUTO_BATTLER,
SURVIVAL, SANDBOX, MMO, SOULSLIKE, EXTRACTION_SHOOTER, BOARD_GAME,
TURN_BASED_STRATEGY, GRAND_STRATEGY, PUZZLE, ROGUELIKE, PLATFORMER,
SIMULATION, RHYTHM
```

### 21 AI Capabilities

```
AIM_ASSIST, SPRAY_CONTROL, RECOIL_COMPENSATION, UTILITY_TIMING,
MAP_AWARENESS, ECONOMY_MANAGEMENT, TEAM_COORDINATION, AUTONOMOUS_PLAY,
COACH_MODE, ENEMY_TRACKING, AUDIO_RADAR, CARD_STRATEGY, HAND_EVALUATION,
BUILD_ORDER, UNIT_MICRO, BOSS_PATTERNS, COMBO_EXECUTION,
BOARD_EVALUATION, TURN_PLANNING, PUZZLE_SOLVING, MENU_NAVIGATION
```

### GameProfile System (`foundation/game_profile.py`)

Declarative JSON profiles for detailed game configuration:

```json
{
  "game_id": "cs2",
  "display_name": "Counter-Strike 2",
  "genre": "tactical_fps",
  "process_names": ["cs2.exe"],
  "regions": {
    "minimap": {"x_pct": 0.01, "y_pct": 0.72, "w_pct": 0.18, "h_pct": 0.26},
    "health": {"x_pct": 0.02, "y_pct": 0.95, "w_pct": 0.08, "h_pct": 0.04, "ocr_enabled": true},
    "ammo": {"x_pct": 0.88, "y_pct": 0.95, "w_pct": 0.10, "h_pct": 0.04, "ocr_enabled": true}
  },
  "keybinds": {
    "move_forward": {"key": "w", "hold": true},
    "shoot": {"key": "mouse1"},
    "buy_menu": {"key": "b"}
  },
  "weapons": {
    "ak47": {"fire_rate_rpm": 600, "damage_head": 111, "magazine_size": 30}
  },
  "analysis_fps": 15,
  "default_skill_level": 0.7
}
```

---

## 8. Autonomous Controller Loop

**File:** `autonomous/controller.py`

The main loop runs at `target_fps` (default 30):

```
while running and not killed:
    tick_start = perf_counter()

    # 1. PERCEIVE (capture screen + audio → GameState)
    game_state = await perceive()

    # 2. THINK (analyze state → plan actions)
    if mode.allows_input:
        actions = await think(game_state)

    # 3. ACT (execute through safety gate)
    for action in sorted(actions, key=priority, reverse=True):
        if safety.check_action(action):
            result = await execute(action)

    # 4. LEARN (record experience, update weights)
    if results:
        await learn(game_state, actions, results)

    # Frame pacing
    sleep(tick_interval - elapsed)
```

### Performance Budget

| Phase | Target | Description |
|-------|--------|-------------|
| Perceive | < 10ms | Screen capture + region crop |
| Think | < 20ms | Reflex + tactical decision |
| Act | < 5ms | Input simulation |
| Learn | < 5ms | Buffer append |
| **Total** | < 33ms | 30 FPS target |

---

## 9. Safety Layer Architecture

**File:** `autonomous/safety_layer.py`

Every input passes through `safety.check_action()` before reaching the OS:

```python
def check_action(action_type, details) -> bool:
    if killed: return False
    if paused: return False
    if session_expired: kill("session_limit"); return False
    if aps_exceeded: audit(throttled=True); return False
    if reaction_too_fast: audit(throttled=True); return False
    if micro_pause_due: return False

    # Permitted
    audit_log.append(action)
    return True
```

### Kill Switch

F12 activates the kill switch. This is **permanent** — the companion must be restarted. No code path can bypass a kill.

### Audit Trail

Every action (permitted or throttled) is logged as structured JSON:

```json
{"t": 1711234567.89, "type": "mouse_move", "d": {"dx": 45, "dy": -12}, "ms": 156.3, "throttled": false}
```

---

## 10. Learning System

### Components

| Module | Algorithm | Purpose |
|--------|-----------|---------|
| `ThompsonSampler` | Beta-Bernoulli sampling | Strategy selection (explore vs. exploit) |
| `ExperienceReplay` | Priority queue | Replay important experiences for training |
| `QLearning` | Tabular Q-learning | State-action value estimation |
| `StrategyOptimizer` | Gradient-free optimization | Tune strategy parameters per game |
| `SkillTracker` | Exponential smoothing | Track improving/declining performance |

### Learning Loop

```
1. Record (state, action, reward) tuples
2. Store in ExperienceReplay buffer (priority-weighted)
3. Periodically sample batch → update Q-values
4. Thompson sampling selects strategies proportional to expected reward
5. SkillTracker adjusts difficulty/aggression based on performance trend
```

---

## 11. Steam Integration

**File:** `steam_integration.py`

### Detection Chain

```
1. Windows Registry: HKLM\SOFTWARE\WOW6432Node\Valve\Steam\InstallPath
2. Windows Registry: HKCU\SOFTWARE\Valve\Steam\SteamPath
3. Fallback paths: C:\Program Files (x86)\Steam, D:\Steam, etc.
4. Parse libraryfolders.vdf → discover all library folders
5. Scan steamapps/appmanifest_*.acf → enumerate installed games
6. Read HKCU\SOFTWARE\Valve\Steam\RunningAppID → detect running game
```

### VDF Parser

Valve's KeyValues format is parsed into nested Python dicts:

```python
_parse_vdf('"AppState" { "appid" "730" "name" "CS2" }')
# → {"AppState": {"appid": "730", "name": "CS2"}}
```

### Genre Classification

Steam store tags are mapped to internal genres via `STEAM_GENRE_MAP` (30+ mappings).

---

## 12. Daemon Architecture

**File:** `daemon/game_watcher.py`

```
GameWatcherDaemon
  ├── GameRegistry (45+ games)
  ├── SteamIntegration (library scanning)
  ├── GameSettingsStore (per-game JSON persistence)
  ├── ActiveSession
  │     ├── GameEntry (from registry)
  │     ├── GameSettings (from store)
  │     ├── GameFSM (genre-specific)
  │     ├── ModeManager (8 modes)
  │     ├── SafetyLayer (hard limits)
  │     └── AutonomousController (See→Think→Act→Learn)
  └── ControlPanelHandler (HTTP server)
```

### Lifecycle

```
start() → background thread → poll loop (every 3s)
  ├── _get_running_processes() → Set[str]
  ├── _detect_game(processes) → (GameEntry, process_name) | None
  ├── _activate_game(entry, process)
  │     ├── Load settings from GameSettingsStore
  │     ├── Create FSM for genre
  │     ├── Create SafetyLayer
  │     ├── Create ModeManager → set default mode
  │     ├── Create AutonomousController
  │     └── Fire on_game_detected callbacks
  └── _deactivate_game()
        ├── Stop controller
        ├── Save session stats
        └── Fire on_game_exited callbacks
```

---

## 13. Control Panel API

**File:** `daemon/control_panel.py`

HTTP server on `localhost:27060` using Python's `http.server`:

| Route | Method | Request | Response |
|-------|--------|---------|----------|
| `/` | GET | — | Full HTML control panel |
| `/health` | GET | — | `{"status": "ok"}` |
| `/api/status` | GET | — | Daemon stats + active session |
| `/api/games` | GET | — | All games in registry |
| `/api/settings?game=X` | GET | game query param | Per-game settings JSON |
| `/api/mode` | POST | `{"mode": "assist", "confirmed": false}` | Mode switch result |
| `/api/activate` | POST | `{"game_id": "cs2"}` | Force-activation result |
| `/api/deactivate` | POST | — | Deactivation result |
| `/api/settings` | POST | `{"game_id": "cs2", "skill_level": 0.9, ...}` | Settings update result |

---

## 14. Extension Points

### Adding a New Game

1. Add entry to `GAME_REGISTRY` in `game_registry.py`:
```python
"my_game": GameEntry(
    game_id="my_game",
    display_name="My Game",
    genre=Genre.SURVIVAL,
    process_names=["mygame.exe"],
    capabilities={AICapability.AUTONOMOUS_PLAY},
    fsm_class="SurvivalFSM",
)
```

2. Optionally add spray patterns, pro profiles, drills, macros, nav routes.

### Adding a New FSM

1. Subclass `GameFSM` in `state_machine/game_fsm.py`:
```python
class MyGenreFSM(GameFSM):
    def __init__(self):
        super().__init__()
        self.add_state("state1", ["state2", "state3"])
        self.add_state("state2", ["state1", "state3"])
        self._current_state = "state1"
```

2. Add genre mapping in `daemon/game_watcher.py`'s `_FSM_GENRE_MAP`.

### Adding a New Play Mode

1. Add to `PlayMode` enum and `MODE_CONFIGS` in `mode_manager.py`.
2. Add to `DANGEROUS_MODES` if it grants AI input control.

---

## 15. Data Flow Diagrams

### Game Detection Flow

```
Process List → Registry Match? ─yes→ Activate
                    │
                    no
                    ↓
              Steam Running? ─yes→ Classify Genre → Activate
                    │
                    no
                    ↓
                  Idle (poll again in 3s)
```

### Mode Transition Safety

```
User requests mode change
     ↓
Mode in DANGEROUS_MODES? ─no→ Switch immediately
     ↓ yes
confirmed=True? ─no→ Reject (return False)
     ↓ yes
Switch mode, fire callbacks
```

### Settings Persistence Flow

```
Game detected → load settings from ~/.gamer-companion/game_settings/{id}.json
     ↓
Session active → settings used for SafetyLayer, ModeManager, Controller config
     ↓
Game exits → update total_sessions, total_play_minutes, last_played → save to disk
```

---

*GGI APEX PREDATOR — AI Architecture Manual*
*Echo Prime Technologies | https://echo-ept.com/gamer-companion*
