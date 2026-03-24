"""Entry point: python -m gamer_companion

Two operating modes:
  gamer-companion              → Daemon mode (watches for games, auto-activates)
  gamer-companion --legacy     → Legacy orchestrator mode
"""

import asyncio
import sys
import argparse
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        prog="gamer-companion",
        description="GGI APEX PREDATOR — AI Gaming Companion",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "observe", "assist", "coach", "copilot",
            "autonomous", "training", "mimic", "swarm",
        ],
        default="observe",
        help="Default play mode (default: observe)",
    )
    parser.add_argument("--game", help="Force-activate for a specific game (e.g. cs2, chess, league)")
    parser.add_argument("--poll", type=float, default=3.0, help="Process poll interval in seconds (default: 3)")
    parser.add_argument("--no-steam", action="store_true", help="Disable Steam integration")
    parser.add_argument("--settings-dir", help="Override per-game settings directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--legacy", action="store_true", help="Use legacy orchestrator mode")
    parser.add_argument("--version", action="version", version="GGI APEX PREDATOR v1.0.0")
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logger.remove()
        logger.add(
            sys.stderr, level="DEBUG",
            format="{time:HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}",
        )
    else:
        logger.remove()
        logger.add(
            sys.stderr, level="INFO",
            format="{time:HH:mm:ss} | {level:<7} | {message}",
        )

    if args.legacy:
        _run_legacy(args)
        return

    # Daemon mode — auto-detect games and activate
    from gamer_companion.daemon.game_watcher import GameWatcherDaemon, DaemonConfig

    config = DaemonConfig(
        poll_interval_s=args.poll,
        steam_enabled=not args.no_steam,
        default_mode=args.mode,
        settings_dir=args.settings_dir,
    )
    daemon = GameWatcherDaemon(config=config)

    # Register event handlers for logging
    daemon.on_game_detected(
        lambda entry, settings: logger.info(
            f">>> GAME DETECTED: {entry.display_name} | "
            f"Mode: {settings.default_mode} | Skill: {settings.skill_level}"
        )
    )
    daemon.on_game_exited(
        lambda entry, duration: logger.info(
            f"<<< GAME EXITED: {entry.display_name} | "
            f"Duration: {duration/60:.1f} min"
        )
    )

    # Force-activate a specific game if requested
    if args.game:
        daemon.start()
        import time
        time.sleep(0.5)  # Let daemon thread initialize
        if not daemon.force_activate(args.game):
            logger.error(f"Unknown game: {args.game}")
            daemon.stop()
            sys.exit(1)
        logger.info(f"Force-activated: {args.game}")
        try:
            while daemon.is_running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        daemon.stop()
    else:
        # Normal daemon mode — watch for games
        logger.info(
            "GGI APEX PREDATOR — Game Watcher Daemon started\n"
            "  Watching for game launches... (Ctrl+C to stop)"
        )
        try:
            asyncio.run(daemon.run())
        except KeyboardInterrupt:
            pass
        daemon.stop()

    logger.info("Daemon shut down.")


def _run_legacy(args):
    """Run the legacy orchestrator (pre-daemon mode)."""
    from .orchestrator import GGIOrchestrator
    orchestrator = GGIOrchestrator(config_path="config/settings.toml")
    if args.mode:
        orchestrator.state.mode = args.mode
    if args.game:
        orchestrator._force_game = args.game
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
