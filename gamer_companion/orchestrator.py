"""Master Orchestrator — the single entry point that runs the entire system.

Architecture:
1. BOOT: Load config -> detect game -> load profile -> initialize subsystems
2. LOOP: Capture -> Perceive -> Think -> Act -> Learn -> Repeat
3. SHUTDOWN: Save state -> close connections -> clean exit
"""

from __future__ import annotations
import asyncio
import signal
import time
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

# Configure structured logging
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
logger.add("logs/ggi_{time:YYYY-MM-DD}.jsonl", serialize=True, rotation="50 MB", retention="30 days")


@dataclass
class OrchestratorState:
    """Global state shared across all subsystems."""
    running: bool = True
    mode: str = "observe"               # observe|assist|copilot|autonomous|training|coach
    game_detected: bool = False
    game_id: str = ""
    game_profile: Any = None
    perception_fps: float = 0.0
    cognition_latency_ms: float = 0.0
    action_count: int = 0
    session_start: float = field(default_factory=time.time)
    errors: int = 0
    last_perception: Any = None
    last_cognition: Any = None
    tilt_level: float = 0.0
    skill_estimate: float = 0.5


class GGIOrchestrator:
    """Master orchestrator — runs BOOT -> LOOP -> SHUTDOWN lifecycle.

    The orchestrator owns the event loop. Every subsystem registers callbacks.
    Nothing runs independently — the orchestrator controls timing, priority,
    and resource allocation for every component.
    """

    def __init__(self, config_path: str = "config/settings.toml"):
        self.state = OrchestratorState()
        self._config_path = config_path
        self._config = None
        self._force_game: Optional[str] = None

        # Subsystem references (initialized in boot)
        self._game_detector = None
        self._screen_capture = None
        self._frame_count = 0
        self._loop_task: Optional[asyncio.Task] = None

    async def boot(self) -> bool:
        """Phase 1: Initialize everything."""
        logger.info("=== GGI APEX PREDATOR v1.0.0 — BOOTING ===")

        # 1. Load config
        from .config_system import GGIConfig
        self._config = GGIConfig(self._config_path)
        self._config.on_change(self._on_config_change)
        if self.state.mode == "observe":
            self.state.mode = self._config.get("general.mode", "observe")
        logger.info(f"Config loaded: mode={self.state.mode}")

        # 2. Screen capture
        try:
            import mss
            self._screen_capture = mss.mss()
            logger.info("Screen capture initialized (mss)")
        except ImportError:
            logger.error("mss not installed. Run: pip install mss")
            return False

        # 3. Detect game
        from .foundation.game_profile import GameDetector
        profiles_dir = self._config.get("paths.profiles_dir", "game_profiles")
        self._game_detector = GameDetector(profiles_dir)

        if self._force_game:
            profile = self._game_detector.get_profile(self._force_game)
            if profile:
                self.state.game_detected = True
                self.state.game_id = profile.game_id
                self.state.game_profile = profile
                logger.info(f"Forced game profile: {profile.display_name}")
            else:
                logger.warning(f"Forced game '{self._force_game}' not found in profiles")
        else:
            profile = self._game_detector.detect()
            if profile:
                self.state.game_detected = True
                self.state.game_id = profile.game_id
                self.state.game_profile = profile
                logger.info(f"Game detected: {profile.display_name}")
            else:
                logger.warning("No game detected. Running in standby mode.")

        # 4. Register signal handlers (Unix-style — Windows uses different approach)
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_shutdown)
        except (NotImplementedError, AttributeError):
            # Windows doesn't support add_signal_handler for all signals
            pass

        logger.info(f"=== BOOT COMPLETE — {self.state.game_id or 'standby'} | mode={self.state.mode} ===")
        return True

    async def run(self):
        """Phase 2: The main loop."""
        if not await self.boot():
            logger.error("Boot failed. Exiting.")
            return

        target_fps = self._config.get("performance.target_fps", 30)
        frame_interval = 1.0 / target_fps

        logger.info(f"Main loop starting at {target_fps} fps target")
        while self.state.running:
            loop_start = time.monotonic()

            try:
                # Check for game if not detected
                if not self.state.game_detected:
                    profile = self._game_detector.detect()
                    if profile:
                        self.state.game_detected = True
                        self.state.game_id = profile.game_id
                        self.state.game_profile = profile
                        logger.info(f"Game detected mid-session: {profile.display_name}")
                    else:
                        await asyncio.sleep(2)  # Poll every 2s
                        continue

                # === CAPTURE ===
                frame = self._capture_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                # === PERCEIVE (stub — Phase 2 builds perception pipeline) ===
                perception = self._basic_perceive(frame)
                self.state.last_perception = perception

                # === THINK (stub — Phase 2 builds cognition engine) ===
                cognition_start = time.monotonic()
                decision = self._basic_think(perception)
                self.state.cognition_latency_ms = (time.monotonic() - cognition_start) * 1000
                self.state.last_cognition = decision

                # === PERFORMANCE ===
                loop_ms = (time.monotonic() - loop_start) * 1000
                self.state.perception_fps = 1000 / max(loop_ms, 1)
                self._frame_count += 1

                # === SESSION LIMIT ===
                session_hours = (time.time() - self.state.session_start) / 3600
                max_hours = self._config.get("safety.max_session_hours", 4)
                if session_hours >= max_hours:
                    logger.warning(f"Session limit reached ({max_hours}h). Auto-stopping.")
                    self.state.running = False

            except KeyboardInterrupt:
                self.state.running = False
            except Exception as e:
                self.state.errors += 1
                logger.error(f"Main loop error #{self.state.errors}: {e}")
                if self.state.errors > 50:
                    logger.critical("Too many errors. Shutting down.")
                    self.state.running = False

            # Frame pacing
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        await self.shutdown()

    def _capture_frame(self):
        """Capture the primary monitor."""
        try:
            import numpy as np
            monitor = self._screen_capture.monitors[1]
            shot = self._screen_capture.grab(monitor)
            return np.array(shot)[:, :, :3]  # Drop alpha channel
        except Exception:
            return None

    def _basic_perceive(self, frame) -> dict:
        """Basic perception stub — returns frame metadata.

        Full perception pipeline (YOLO, OCR, region analysis) comes in Phase 2.
        """
        return {
            "frame_id": self._frame_count,
            "timestamp": time.time(),
            "frame_shape": frame.shape if frame is not None else None,
            "game_id": self.state.game_id,
            "game_phase": "unknown",
            "health": None,
            "armor": None,
            "money": None,
            "enemies_visible": 0,
            "detections": [],
        }

    def _basic_think(self, perception: dict) -> dict:
        """Basic cognition stub — returns observe decision.

        Full cognition (LLM, MCTS, probability engine) comes in Phase 3.
        """
        return {
            "action": "observe",
            "reasoning": "Phase 1 — perception only, no decision engine yet",
            "confidence": 0.0,
            "layer": "stub",
        }

    async def shutdown(self):
        """Phase 3: Clean exit."""
        logger.info("=== SHUTTING DOWN ===")
        if self._config:
            self._config.stop()
        session_duration = time.time() - self.state.session_start
        logger.info(
            f"Session: {session_duration/60:.1f}min | {self._frame_count} frames | "
            f"{self.state.action_count} actions | {self.state.errors} errors"
        )
        logger.info("=== SHUTDOWN COMPLETE ===")

    def _signal_shutdown(self):
        self.state.running = False

    def _on_config_change(self, key: str, value):
        logger.info(f"Config changed: {key} = {value}")
        if key == "general.mode":
            self.state.mode = value


def main():
    """Entry point: python -m gamer_companion"""
    orchestrator = GGIOrchestrator()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
