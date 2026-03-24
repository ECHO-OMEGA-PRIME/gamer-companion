# GGI APEX PREDATOR — User Manual

**Version 1.0.0 | AI Gaming Companion**
**By Echo Prime Technologies**

---

## Table of Contents

1. [What Is GGI Apex Predator?](#1-what-is-ggi-apex-predator)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Quick Start (5 Minutes)](#4-quick-start)
5. [The 8 Play Modes](#5-the-8-play-modes)
6. [Supported Games (45+)](#6-supported-games)
7. [Steam Integration](#7-steam-integration)
8. [Control Panel (Steam Overlay)](#8-control-panel)
9. [Per-Game Settings](#9-per-game-settings)
10. [Auto-Launch Daemon](#10-auto-launch-daemon)
11. [Spray Pattern Compensation](#11-spray-pattern-compensation)
12. [Pro Player Mimicry](#12-pro-player-mimicry)
13. [Training Drills](#13-training-drills)
14. [Macro Engine](#14-macro-engine)
15. [Safety System](#15-safety-system)
16. [Frequently Asked Questions](#16-faq)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What Is GGI Apex Predator?

GGI Apex Predator is an AI-powered gaming companion that watches your screen, understands what's happening in your game, and helps you play better — or plays for you entirely. It works with **45+ games** across every major genre: FPS, MOBA, RTS, fighting, racing, sports, card games, board games, survival, MMO, puzzle, and more.

**It is not a cheat.** It does not inject into game memory, modify game files, or intercept network traffic. It works entirely through screen analysis (computer vision) and standard input simulation — the same way a human would play, just faster and smarter.

### What It Can Do

- **Watch and coach** — Analyze your gameplay in real-time and give you voice tips
- **Assist with suggestions** — Overlay suggestions for positioning, economy, builds
- **Co-pilot** — AI handles movement/abilities while you handle aiming
- **Play autonomously** — Full AI control for grinding, farming, or just watching it play
- **Mimic pros** — Play in the exact style of professional players (TenZ, s1mple, Faker, etc.)
- **Run training drills** — Guided practice for aim, movement, game sense, economy
- **Coordinate squads** — Multi-AI swarm mode for team games

---

## 2. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | 4 cores, 2.5 GHz | 6+ cores, 3.5 GHz |
| RAM | 8 GB | 16 GB |
| GPU | Not required | Any (used for game, not AI) |
| Python | 3.11+ | 3.11+ |
| Disk | 500 MB | 1 GB |
| Display | 1920x1080 | 1920x1080 or 2560x1440 |

The AI runs on CPU. Your GPU is free for the game itself.

---

## 3. Installation

### From PyPI (Recommended)

```bash
pip install gamer-companion
```

### From Source

```bash
git clone https://github.com/ECHO-OMEGA-PRIME/gamer-companion.git
cd gamer-companion
pip install -e .
```

### Verify Installation

```bash
gamer-companion --version
# Output: GGI APEX PREDATOR v1.0.0
```

---

## 4. Quick Start

### Step 1: Launch the Daemon

```bash
gamer-companion
```

That's it. The daemon now watches for game launches. When you start any supported game, the companion auto-activates.

### Step 2: Open the Control Panel

Open your browser (or Steam Overlay browser via Shift+Tab) and go to:

```
http://localhost:27060
```

You'll see the full control panel with game status, mode switcher, settings, and game library.

### Step 3: Play a Game

Launch any supported game. The companion detects it within 3 seconds, loads the correct profile, and starts in **Observe** mode (no AI input — analysis only).

### Step 4: Switch Modes

From the control panel, click any mode button:
- **Assist** — Tips via overlay
- **Coach** — Real-time voice coaching
- **Autonomous** — Full AI control (requires confirmation)

---

## 5. The 8 Play Modes

| Mode | AI Input | Description | Best For |
|------|----------|-------------|----------|
| **Observe** | None | Watch and analyze only. Collects data, builds strategy. | Learning the AI, first launch |
| **Assist** | None (overlay only) | Suggestions displayed on-screen and via voice. | Improving your gameplay |
| **Coach** | None (voice only) | Real-time voice coaching: "Enemy flanking left", "Save for AWP" | Active learning while playing |
| **Copilot** | Partial | AI handles movement, utility, abilities. You handle aiming and shooting. | Tactical advantage without full AI |
| **Autonomous** | Full | Full AI control. Perceive → Think → Act loop at 30 FPS. | AFK farming, grinding, bot matches |
| **Training** | Full (drills) | AI runs structured drills: aim trainers, economy practice, movement courses. | Skill building |
| **Mimic** | Full | Play in the exact style of a selected pro player. | Learning from the best |
| **Swarm** | Full (multi-agent) | Multiple AI agents coordinate as a squad. | Team games, bot lobbies |

### Mode Safety

Modes that give the AI mouse/keyboard control (**Autonomous**, **Mimic**, **Swarm**) are flagged as "dangerous" and require explicit confirmation before activation. This prevents accidental AI input during ranked matches.

---

## 6. Supported Games (45+)

### First-Person Shooters
| Game | Process | Spray Patterns | Pro Profiles | Drills | Macros |
|------|---------|----------------|--------------|--------|--------|
| Counter-Strike 2 | cs2.exe | 5 weapons | s1mple, ZywOo, NiKo, ropz | 5 | 3 |
| Valorant | VALORANT-Win64-Shipping.exe | 3 weapons | TenZ, Aspas, Demon1 | 2 | 1 |
| Call of Duty (MP) | cod.exe | 8 weapons | Shotzzy, Simp, Cellium, Dashy | 5 | 4 |
| Call of Duty (Warzone) | cod.exe | — | — | 3 | — |
| Apex Legends | r5apex.exe | 5 weapons | ImperialHal, Genburten | 3 | 2 |
| Rainbow Six Siege | RainbowSix.exe | 3 weapons | Beaulo | 2 | — |
| Overwatch 2 | Overwatch.exe | — | Profit | 3 | — |
| Halo Infinite | HaloInfinite.exe | 1 weapon | — | — | — |
| Battlefield 2042 | BF2042.exe | 1 weapon | — | — | — |
| Escape from Tarkov | EscapeFromTarkov.exe | 2 weapons | — | — | — |
| PUBG | TslGame.exe | 3 weapons | — | 1 | — |

### Battle Royale
| Game | Process | Drills |
|------|---------|--------|
| Fortnite | FortniteClient-Win64-Shipping.exe | 3 |

### MOBA
| Game | Process | Pro Profiles | Drills |
|------|---------|--------------|--------|
| League of Legends | League of Legends.exe | Faker | 3 |
| Dota 2 | dota2.exe | Yatoro | 2 |
| Smite 2 | SmiteEternalEmpire.exe | — | — |

### Strategy
| Game | Process | Pro Profiles | Drills |
|------|---------|--------------|--------|
| StarCraft II | SC2_x64.exe | Serral, Maru | 3 |
| Age of Empires IV | RelicCardinal.exe | — | — |
| Civilization VI | CivilizationVI.exe | — | — |
| Civilization VII | CivilizationVII.exe | — | — |
| XCOM 2 | XCOM2.exe | — | — |
| Total War: Warhammer III | Warhammer3.exe | — | — |
| Humankind | Humankind.exe | — | — |

### Fighting
| Game | Process | Pro Profiles | Drills |
|------|---------|--------------|--------|
| Street Fighter 6 | StreetFighter6.exe | Daigo | 3 |
| Tekken 8 | Tekken8.exe | Knee | 2 |
| Mortal Kombat 1 | MK12.exe | — | — |

### Racing & Sports
| Game | Process | Pro Profiles | Drills |
|------|---------|--------------|--------|
| Forza Motorsport | ForzaMotorsport.exe | — | — |
| Rocket League | RocketLeague.exe | JSTN | 3 |
| iRacing | iRacingUI.exe | — | — |
| EA FC 25 | FC25.exe | — | 2 |
| Madden NFL | Madden25.exe | — | — |
| NBA 2K | NBA2K25.exe | — | — |

### Card & Auto Battler
| Game | Process | Drills |
|------|---------|--------|
| Hearthstone | Hearthstone.exe | 2 |
| MTG Arena | MTGA.exe | — |
| Teamfight Tactics | League of Legends.exe | — |
| Slay the Spire | SlayTheSpire.exe | — |
| Balatro | Balatro.exe | — |

### Board Games
| Game | Process |
|------|---------|
| Chess | — |
| Checkers | — |
| Backgammon | — |

### Survival & Sandbox
| Game | Process | Drills |
|------|---------|--------|
| Minecraft | javaw.exe | 2 |
| Rust | RustClient.exe | 1 |

### MMO
| Game | Process | Drills |
|------|---------|--------|
| World of Warcraft | Wow.exe | 2 |
| Final Fantasy XIV | ffxiv_dx11.exe | — |

### Soulslike
| Game | Process | Drills |
|------|---------|--------|
| Elden Ring | eldenring.exe | 2 |
| Dark Souls III | DarkSoulsIII.exe | — |

### Puzzle & Indie
| Game | Process |
|------|---------|
| Tetris | — |
| Baba Is You | — |
| Celeste | — |
| Hollow Knight | — |
| Factorio | — |
| Satisfactory | — |
| Cities: Skylines II | — |
| Hades 2 | — |

---

## 7. Steam Integration

The companion automatically integrates with your Steam library:

### How It Works

1. **Library Discovery** — Reads Steam's `libraryfolders.vdf` and `appmanifest_*.acf` files to find all installed games across all drives
2. **Running Game Detection** — Checks the Windows registry key `HKCU\SOFTWARE\Valve\Steam\RunningAppID` to detect which Steam game is running
3. **Genre Classification** — Maps Steam store tags (FPS, MOBA, Strategy, etc.) to the companion's internal genre system

### Steam Overlay Integration

The control panel is designed to work inside Steam's overlay browser:

1. Launch your game through Steam
2. Press **Shift+Tab** to open the Steam overlay
3. Click **Web Browser**
4. Navigate to `http://localhost:27060`
5. You now have full control of the companion without leaving your game

**Tip:** Bookmark `localhost:27060` in Steam's overlay browser for instant access.

---

## 8. Control Panel

The control panel is a web-based UI that runs locally at `http://localhost:27060`.

### Sections

| Section | Purpose |
|---------|---------|
| **Game Banner** | Shows the currently detected game, genre, FSM state, and session duration |
| **Play Mode** | 8 mode buttons. Click to switch. Dangerous modes prompt for confirmation |
| **Session Stats** | Real-time safety metrics, actions per second, throttle rate |
| **Game Settings** | Per-game configuration: skill level, aggression, DPI, reaction time, session limit |
| **Game Library** | Searchable list of all 45+ games. Click to force-activate any game |

### REST API

For advanced users and scripting:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Full daemon status + active session |
| `/api/games` | GET | List all games in registry |
| `/api/settings?game=cs2` | GET | Get per-game settings |
| `/api/settings` | POST | Update per-game settings |
| `/api/mode` | POST | Switch play mode |
| `/api/activate` | POST | Force-activate a game |
| `/api/deactivate` | POST | Deactivate companion |
| `/health` | GET | Health check |

---

## 9. Per-Game Settings

Every game remembers its own settings between sessions. Settings are stored in:

```
~/.gamer-companion/game_settings/{game_id}.json
```

### Available Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `default_mode` | observe | Mode to activate when game launches |
| `auto_activate` | true | Whether to activate when game is detected |
| `auto_mode_confirmed` | false | Pre-confirm dangerous modes (skip dialog) |
| `skill_level` | 0.7 | AI skill (0.0 = intentionally bad, 1.0 = maximum) |
| `aggression` | 0.5 | How aggressively the AI plays (0.0 = passive, 1.0 = hyper-aggressive) |
| `reaction_time_ms` | 150 | Minimum AI reaction time in milliseconds |
| `sensitivity_override` | null | Override in-game mouse sensitivity |
| `dpi` | 800 | Mouse DPI for aim calculations |
| `active_pro_profile` | "" | Pro player to mimic (e.g. "tenz", "s1mple") |
| `overlay_enabled` | true | Show the AI overlay |
| `max_session_hours` | 4.0 | Auto-stop after this many hours |

### Example: Configure CS2 for Autonomous Play

```bash
curl -X POST http://localhost:27060/api/settings \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "cs2",
    "default_mode": "autonomous",
    "auto_mode_confirmed": true,
    "skill_level": 0.85,
    "aggression": 0.6,
    "reaction_time_ms": 120
  }'
```

---

## 10. Auto-Launch Daemon

The daemon runs in the background and watches for game processes.

### How It Works

1. Polls running processes every 3 seconds (configurable with `--poll`)
2. Matches process names against the 45+ game registry
3. Falls back to Steam running-game detection if no process match
4. When a game is found: loads settings, creates the genre-specific FSM, initializes safety layer, activates companion
5. When the game exits: saves session stats, deactivates, returns to idle

### CLI Options

```bash
gamer-companion                     # Default: watch for games
gamer-companion --game cs2          # Force-activate for CS2 immediately
gamer-companion --mode assist       # Default to assist mode
gamer-companion --poll 1.0          # Check every 1 second
gamer-companion --no-steam          # Disable Steam integration
gamer-companion --no-panel          # Disable web control panel
gamer-companion --port 8080         # Control panel on different port
gamer-companion --debug             # Verbose logging
gamer-companion --legacy            # Use old orchestrator (pre-daemon)
```

---

## 11. Spray Pattern Compensation

For FPS games, the companion includes recoil compensation data for **35+ weapons** across CS2, Valorant, CoD, Apex, R6 Siege, PUBG, Rust, Tarkov, Battlefield, and Halo.

Each pattern includes:
- Per-bullet X/Y offset vectors
- Magazine size and fire rate
- Compensation timing synced to RPM

The AI uses these patterns in **Copilot** and **Autonomous** modes to counter recoil automatically, simulating a human learning the spray pattern over time.

---

## 12. Pro Player Mimicry

**22 professional players** are modeled with behavioral profiles:

Each profile captures:
- **Aggression** — How often they push vs. hold
- **Consistency** — Shot-to-shot variance
- **Utility usage** — Grenade/ability timing
- **Positioning** — Default angles and rotations
- **Movement style** — Peek patterns, strafe speed
- **Economy decisions** — When to force-buy, eco, full-buy

Example: Setting `active_pro_profile: "tenz"` makes the AI play with TenZ's aggressive peek style, high sensitivity flicks, and ability-first approach.

---

## 13. Training Drills

**50+ structured drills** across 21 games covering:

| Category | Examples |
|----------|---------|
| **Aim** | Flick training, tracking, spray control, headshot practice |
| **Movement** | Strafe shooting, bunny hopping, slide canceling, wall bouncing |
| **Utility** | Smoke lineups, flash timings, ability combos |
| **Economy** | Buy round optimization, eco round strats, investment timing |
| **Game Sense** | Map rotation, timing pushes, reading opponent patterns |

Drills are game-specific and adapt to your skill level.

---

## 14. Macro Engine

**23+ pre-built macros** for complex input sequences:

| Game | Macros |
|------|--------|
| CoD | Slide Cancel, Bunny Hop, YY Cancel, Dolphin Dive |
| Apex | Superglide, Tap Strafe |
| Valorant | Jett Dash-Shoot |
| LoL | Fast Combo, Ward Hop |
| Fortnite | 90s Build, Edit Reset |
| Rocket League | Fast Aerial, Half Flip |
| StarCraft II | Inject Cycle |
| Street Fighter 6 | Hadouken, Shoryuken |
| WoW | Rotation Burst |
| Minecraft | Speed Bridge |

Macros execute frame-perfect timing through the safety layer with human-like variance injected.

---

## 15. Safety System

The safety layer is **non-bypassable**. Every AI input passes through it.

### Hard Limits

| Rule | Value | Purpose |
|------|-------|---------|
| Max APS | 15 | Human maximum is ~12 actions/second |
| Min Reaction | 90ms | Human minimum is ~150ms |
| Session Limit | 4 hours | Configurable per-game |
| Micro-Pauses | Every 30-60s | Simulates human hesitation |
| Kill Switch | F12 | Permanent disable until restart |
| Audit Trail | Every action | Full JSON log of all AI inputs |

### Anti-Detection Design

- All inputs route through standard Windows API calls (same as human input)
- Random micro-pauses prevent pattern detection
- Reaction times are randomized within a human-like range
- Actions per second stay below human physiological limits
- Mouse movements follow Bezier curves with natural jitter

---

## 16. FAQ

**Q: Is this a cheat?**
A: No. It does not modify game memory, inject DLLs, or intercept network traffic. It reads the screen and presses keys — exactly like a human would.

**Q: Will I get banned?**
A: We cannot guarantee anything. Use at your own risk in competitive modes. The safety system is designed to make AI input indistinguishable from human input, but no system is 100% undetectable. We recommend using Autonomous mode only in bot matches, casual lobbies, or single-player.

**Q: Does it work with any game?**
A: It has optimized profiles for 45+ games. For unlisted games, the Steam integration will auto-detect and classify them, and the AI will use genre-appropriate defaults.

**Q: Does it need internet?**
A: No. Everything runs locally on your PC. The only network access is optional Steam API calls for game metadata.

**Q: Can it play chess?**
A: Yes. Chess, checkers, and backgammon are supported with the BoardGameFSM. In Autonomous mode, it evaluates board positions and makes moves through the screen.

**Q: Does it use my GPU?**
A: No. All AI processing is CPU-based. Your GPU is 100% available for the game itself.

---

## 17. Troubleshooting

### Game Not Detected

1. Check the game's process name matches the registry: `curl http://localhost:27060/api/games | jq '.games[] | select(.name | contains("CS"))'`
2. If using Steam, ensure Steam is running
3. Try `--debug` mode: `gamer-companion --debug`
4. Force-activate: `gamer-companion --game cs2`

### Control Panel Not Loading

1. Check port isn't in use: `netstat -ano | findstr 27060`
2. Try a different port: `gamer-companion --port 8080`
3. Disable and re-enable: `gamer-companion --no-panel` then restart

### High CPU Usage

1. Reduce poll interval: `gamer-companion --poll 5.0`
2. Use **Observe** mode instead of **Autonomous** (no action processing)
3. Close the control panel if not needed: `--no-panel`

### AI Playing Poorly

1. Check skill level is appropriate: `/api/settings?game=cs2`
2. Ensure the correct game is detected (check game banner)
3. Verify screen resolution matches expected (1920x1080 base)
4. Run training drills first to calibrate

---

*GGI APEX PREDATOR — Built by Echo Prime Technologies*
*https://echo-ept.com/gamer-companion*
