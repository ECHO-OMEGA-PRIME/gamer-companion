"""Game Watcher Daemon — Auto-detect game launches and activate companion.

Runs as a background daemon that:
1. Polls running processes every N seconds
2. Matches against GameRegistry + Steam integration
3. On game detected: loads per-game settings, initializes the correct FSM,
   configures subsystems, and activates the companion
4. On game exit: saves session stats, deactivates, returns to idle
5. Handles multiple game switches gracefully

This is the top-level entry point that makes the companion fully autonomous.
"""

from __future__ import annotations
import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Set
from loguru import logger

from gamer_companion.game_registry import GameRegistry, GameEntry, GAME_REGISTRY
from gamer_companion.steam_integration import SteamIntegration
from gamer_companion.daemon.game_settings import GameSettingsStore, GameSettings
from gamer_companion.state_machine.game_fsm import GameFSM
from gamer_companion.autonomous.mode_manager import ModeManager, PlayMode
from gamer_companion.autonomous.safety_layer import SafetyLayer
from gamer_companion.autonomous.controller import AutonomousController


# Maps genre strings to FSM class names for dynamic instantiation
_FSM_GENRE_MAP = {
    "tactical_fps": "TacticalFPSFSM",
    "battle_royale": "BattleRoyaleFSM",
    "moba": "MOBAFSM",
    "cod_multiplayer": "CoDMultiplayerFSM",
    "hero_shooter": "HeroShooterFSM",
    "arena_shooter": "ArenaShooterFSM",
    "rts": "RTSFSM",
    "fighting": "FightingGameFSM",
    "racing": "RacingFSM",
    "sports": "SportsFSM",
    "card": "CardGameFSM",
    "auto_battler": "AutoBattlerFSM",
    "survival": "SurvivalFSM",
    "sandbox": "SurvivalFSM",
    "mmo": "MMOFSM",
    "soulslike": "SoulslikeFSM",
    "extraction_shooter": "ExtractionShooterFSM",
    "board_game": "BoardGameFSM",
    "turn_based_strategy": "TurnBasedStrategyFSM",
    "grand_strategy": "TurnBasedStrategyFSM",
    "puzzle": "PuzzleGameFSM",
    "roguelike": "PuzzleGameFSM",
    "platformer": "PuzzleGameFSM",
    "simulation": "TurnBasedStrategyFSM",
    "rhythm": "PuzzleGameFSM",
}


@dataclass
class ActiveSession:
    """Tracks an active game session."""
    game_entry: GameEntry
    settings: GameSettings
    fsm: Optional[GameFSM] = None
    mode_manager: Optional[ModeManager] = None
    controller: Optional[AutonomousController] = None
    safety: Optional[SafetyLayer] = None
    start_time: float = 0.0
    process_name: str = ""


@dataclass
class DaemonConfig:
    """Configuration for the game watcher daemon."""
    poll_interval_s: float = 3.0        # How often to check for games
    steam_enabled: bool = True           # Use Steam integration
    auto_activate: bool = True           # Auto-activate on game detection
    default_mode: str = "observe"        # Default mode if no per-game setting
    settings_dir: Optional[str] = None   # Override settings directory
    tray_icon: bool = True               # Show system tray icon
    notifications: bool = True           # Desktop notifications on events


class GameWatcherDaemon:
    """Background daemon that watches for game launches.

    Lifecycle:
        daemon = GameWatcherDaemon()
        daemon.start()    # Non-blocking, runs in background thread
        ...
        daemon.stop()     # Graceful shutdown

    Or async:
        daemon = GameWatcherDaemon()
        await daemon.run()  # Blocking async loop
    """

    def __init__(self, config: Optional[DaemonConfig] = None):
        self.config = config or DaemonConfig()
        self._registry = GameRegistry()
        self._steam = SteamIntegration() if self.config.steam_enabled else None
        self._settings = GameSettingsStore(self.config.settings_dir)
        self._active_session: Optional[ActiveSession] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks
        self._on_game_detected: List[Callable] = []
        self._on_game_exited: List[Callable] = []
        self._on_mode_changed: List[Callable] = []

        # Tracking
        self._seen_processes: Set[str] = set()
        self._poll_count = 0
        self._games_detected = 0
        self._start_time = 0.0

        logger.info("GameWatcherDaemon initialized")

    # ── Event registration ──────────────────────────────────────────────

    def on_game_detected(self, callback: Callable[[GameEntry, GameSettings], None]):
        """Register callback for when a game is detected."""
        self._on_game_detected.append(callback)

    def on_game_exited(self, callback: Callable[[GameEntry, float], None]):
        """Register callback for when a game exits. Receives (entry, duration_s)."""
        self._on_game_exited.append(callback)

    def on_mode_changed(self, callback: Callable[[str, str], None]):
        """Register callback for mode transitions."""
        self._on_mode_changed.append(callback)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self):
        """Start the daemon in a background thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._run_in_thread, daemon=True, name="GameWatcher",
        )
        self._thread.start()
        logger.info("GameWatcherDaemon started (background thread)")

    def stop(self):
        """Stop the daemon gracefully."""
        self._running = False
        if self._active_session:
            self._deactivate_game()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("GameWatcherDaemon stopped")

    def _run_in_thread(self):
        """Thread target — create event loop and run."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self.run())
        finally:
            self._loop.close()

    async def run(self):
        """Main async polling loop."""
        self._running = True
        self._start_time = time.time()
        logger.info(
            f"Game watcher polling every {self.config.poll_interval_s}s "
            f"({len(GAME_REGISTRY)} games in registry)"
        )

        while self._running:
            try:
                self._poll_count += 1
                self._poll()
            except Exception as e:
                logger.error(f"Poll error: {e}")

            await asyncio.sleep(self.config.poll_interval_s)

    # ── Core polling logic ──────────────────────────────────────────────

    def _poll(self):
        """Single poll cycle: check for game processes."""
        running_processes = self._get_running_processes()

        if self._active_session:
            # Check if active game is still running
            if self._active_session.process_name not in running_processes:
                logger.info(
                    f"Game exited: {self._active_session.game_entry.display_name}"
                )
                self._deactivate_game()
        else:
            # Scan for a new game
            detected = self._detect_game(running_processes)
            if detected:
                self._activate_game(detected[0], detected[1])

    def _get_running_processes(self) -> Set[str]:
        """Get set of currently running process names."""
        try:
            import psutil
            procs = set()
            for proc in psutil.process_iter(["name"]):
                try:
                    name = proc.info["name"]
                    if name:
                        procs.add(name.lower())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return procs
        except ImportError:
            # Fallback: use ctypes/WMI
            return self._get_processes_fallback()

    def _get_processes_fallback(self) -> Set[str]:
        """Fallback process enumeration without psutil."""
        import ctypes
        from ctypes import wintypes

        TH32CS_SNAPPROCESS = 0x00000002
        procs = set()

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        kernel32 = ctypes.windll.kernel32
        snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == -1:
            return procs

        pe = PROCESSENTRY32()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32)

        if kernel32.Process32First(snap, ctypes.byref(pe)):
            while True:
                name = pe.szExeFile.decode("utf-8", errors="ignore").lower()
                procs.add(name)
                if not kernel32.Process32Next(snap, ctypes.byref(pe)):
                    break

        kernel32.CloseHandle(snap)
        return procs

    def _detect_game(
        self, running_processes: Set[str],
    ) -> Optional[tuple]:
        """Try to detect a game from running processes.

        Returns (GameEntry, matched_process_name) or None.
        """
        # Method 1: Match against GameRegistry process names
        for game_id, entry in GAME_REGISTRY.items():
            for proc_name in entry.process_names:
                if proc_name.lower() in running_processes:
                    return (entry, proc_name.lower())

        # Method 2: Steam running game detection
        if self._steam and self._steam.is_available:
            steam_game = self._steam.detect_running_game()
            if steam_game:
                genre = self._steam.classify_genre(steam_game)
                # Try to find in registry by app_id or name
                reg_entry = self._registry.get(
                    str(steam_game.app_id)
                ) or self._registry.search(steam_game.name)
                if reg_entry:
                    if isinstance(reg_entry, list):
                        reg_entry = reg_entry[0] if reg_entry else None
                    if reg_entry:
                        return (reg_entry, steam_game.name.lower())

        return None

    # ── Activation / Deactivation ───────────────────────────────────────

    def _activate_game(self, entry: GameEntry, process_name: str):
        """Activate the companion for a detected game."""
        self._games_detected += 1
        logger.info(
            f"Game detected: {entry.display_name} "
            f"(genre={entry.genre.value}, process={process_name})"
        )

        # Load per-game settings
        settings = self._settings.get(entry.game_id)
        settings.display_name = entry.display_name
        settings.total_sessions += 1
        settings.last_played = time.time()

        # Don't activate if user disabled auto-activate for this game
        if not settings.auto_activate and not self.config.auto_activate:
            logger.info(f"Auto-activate disabled for {entry.display_name}")
            return

        # Create the FSM for this genre
        fsm = self._create_fsm(entry)

        # Create safety layer
        safety = SafetyLayer(
            min_reaction_ms=settings.reaction_time_ms,
            session_limit_hours=settings.max_session_hours,
        )

        # Create mode manager and set the game's default mode
        mode_manager = ModeManager()
        target_mode = PlayMode(settings.default_mode)
        if target_mode in ModeManager.DANGEROUS_MODES:
            if settings.auto_mode_confirmed:
                mode_manager.switch(target_mode, confirmed=True)
            else:
                # Fall back to OBSERVE for unconfirmed dangerous modes
                logger.info(
                    f"Dangerous mode {target_mode.value} not confirmed, "
                    "starting in OBSERVE"
                )
                mode_manager.switch(PlayMode.OBSERVE)
        else:
            mode_manager.switch(target_mode)

        # Wire mode change callbacks
        def _on_mode(old, new):
            for cb in self._on_mode_changed:
                try:
                    cb(old.value, new.value)
                except Exception:
                    pass
        mode_manager.on_change(_on_mode)

        # Create autonomous controller
        controller = AutonomousController(
            target_fps=15.0,
            safety=safety,
            mode_manager=mode_manager,
        )

        # Store active session
        self._active_session = ActiveSession(
            game_entry=entry,
            settings=settings,
            fsm=fsm,
            mode_manager=mode_manager,
            controller=controller,
            safety=safety,
            start_time=time.time(),
            process_name=process_name,
        )

        # Save updated settings
        self._settings.save(settings)

        # Fire callbacks
        for cb in self._on_game_detected:
            try:
                cb(entry, settings)
            except Exception as e:
                logger.error(f"Game detected callback error: {e}")

        logger.info(
            f"Companion ACTIVATED for {entry.display_name} "
            f"| Mode: {mode_manager.mode.value} "
            f"| FSM: {type(fsm).__name__ if fsm else 'none'} "
            f"| Skill: {settings.skill_level}"
        )

    def _deactivate_game(self):
        """Deactivate the companion when game exits."""
        if not self._active_session:
            return

        session = self._active_session
        duration = time.time() - session.start_time

        # Stop controller if running
        if session.controller and session.controller.is_running:
            session.controller.stop()

        # Update session stats
        session.settings.total_play_minutes += duration / 60.0
        self._settings.save(session.settings)

        # Fire callbacks
        for cb in self._on_game_exited:
            try:
                cb(session.game_entry, duration)
            except Exception as e:
                logger.error(f"Game exited callback error: {e}")

        logger.info(
            f"Companion DEACTIVATED for {session.game_entry.display_name} "
            f"| Duration: {duration/60:.1f} min "
            f"| Sessions: {session.settings.total_sessions}"
        )

        self._active_session = None

    def _create_fsm(self, entry: GameEntry) -> Optional[GameFSM]:
        """Create the correct FSM for a game's genre."""
        from gamer_companion.state_machine import game_fsm as fsm_module

        # Use the fsm_class name from the registry entry if specified
        fsm_class_name = entry.fsm_class
        if not fsm_class_name:
            # Fall back to genre mapping
            fsm_class_name = _FSM_GENRE_MAP.get(entry.genre.value, "")

        if fsm_class_name:
            cls = getattr(fsm_module, fsm_class_name, None)
            if cls:
                return cls()
            logger.warning(f"FSM class {fsm_class_name} not found")

        # Final fallback
        return GameFSM()

    # ── Manual controls ─────────────────────────────────────────────────

    def switch_mode(self, mode: str, confirmed: bool = False) -> bool:
        """Manually switch play mode for the active game."""
        if not self._active_session or not self._active_session.mode_manager:
            return False
        try:
            target = PlayMode(mode)
        except ValueError:
            logger.error(f"Invalid mode: {mode}")
            return False
        return self._active_session.mode_manager.switch(target, confirmed=confirmed)

    def update_settings(self, game_id: str, **kwargs) -> bool:
        """Update settings for a game."""
        settings = self._settings.get(game_id)
        for k, v in kwargs.items():
            if hasattr(settings, k):
                setattr(settings, k, v)
        self._settings.save(settings)
        return True

    def force_activate(self, game_id: str) -> bool:
        """Manually force-activate companion for a game (even if not detected)."""
        entry = self._registry.get(game_id)
        if not entry:
            logger.error(f"Unknown game: {game_id}")
            return False
        if self._active_session:
            self._deactivate_game()
        self._activate_game(entry, "manual")
        return True

    def force_deactivate(self):
        """Manually deactivate the companion."""
        if self._active_session:
            self._deactivate_game()

    # ── Queries ─────────────────────────────────────────────────────────

    @property
    def active_game(self) -> Optional[str]:
        if self._active_session:
            return self._active_session.game_entry.display_name
        return None

    @property
    def active_mode(self) -> Optional[str]:
        if self._active_session and self._active_session.mode_manager:
            return self._active_session.mode_manager.mode.value
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        session_info = None
        if self._active_session:
            s = self._active_session
            session_info = {
                "game": s.game_entry.display_name,
                "game_id": s.game_entry.game_id,
                "genre": s.game_entry.genre.value,
                "mode": s.mode_manager.mode.value if s.mode_manager else None,
                "fsm_state": s.fsm.state if s.fsm else None,
                "duration_min": round((time.time() - s.start_time) / 60, 1),
                "skill_level": s.settings.skill_level,
                "safety": s.safety.get_stats() if s.safety else None,
            }
        return {
            "daemon_running": self._running,
            "uptime_min": round(uptime / 60, 1),
            "poll_count": self._poll_count,
            "poll_interval_s": self.config.poll_interval_s,
            "games_in_registry": len(GAME_REGISTRY),
            "games_configured": len(self._settings.list_games()),
            "total_games_detected": self._games_detected,
            "steam_available": self._steam.is_available if self._steam else False,
            "active_session": session_info,
        }
