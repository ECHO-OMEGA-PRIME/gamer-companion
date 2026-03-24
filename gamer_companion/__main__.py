"""Entry point: python -m gamer_companion"""

import asyncio
import sys
import argparse
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        prog="gamer-companion",
        description="GGI APEX PREDATOR — AI Gaming Companion"
    )
    parser.add_argument("--config", default="config/settings.toml",
                        help="Path to config file")
    parser.add_argument("--mode", choices=["observe", "assist", "copilot", "autonomous", "training", "coach"],
                        help="Override operating mode")
    parser.add_argument("--game", help="Force game profile (e.g. cs2, valorant)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="version", version="GGI APEX PREDATOR v1.0.0")
    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG",
                   format="{time:HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}")

    from .orchestrator import GGIOrchestrator
    orchestrator = GGIOrchestrator(config_path=args.config)

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
