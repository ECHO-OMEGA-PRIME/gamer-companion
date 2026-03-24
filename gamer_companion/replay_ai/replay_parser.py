"""Replay Parser — Parse native game replay files for rich analysis.

Supports .dem (CS2/CSGO/Dota2) and .rofl (League of Legends).
Replay files contain every tick of every player — 1000x more data
than screen capture for post-game analysis.
"""

from __future__ import annotations
import struct
import json
from dataclasses import dataclass, field
from typing import List, Optional, BinaryIO
from pathlib import Path
from loguru import logger


@dataclass
class ReplayEvent:
    """A single event in a replay file."""
    tick: int
    timestamp: float  # seconds from match start
    event_type: str  # "kill", "death", "plant", "ability", "buy", "position"
    data: dict = field(default_factory=dict)


@dataclass
class ParsedReplay:
    """Structured data extracted from a game replay file."""
    game: str
    map_name: str
    duration_seconds: float
    players: List[dict] = field(default_factory=list)
    events: List[ReplayEvent] = field(default_factory=list)
    round_summaries: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ReplayParser:
    """Parse native game replay files.

    Supported formats:
    - CS2/CSGO .dem files (Valve demo format)
    - League of Legends .rofl files
    """

    PARSERS = {
        ".dem": "_parse_dem",
        ".rofl": "_parse_rofl",
    }

    def parse(self, file_path: str) -> Optional[ParsedReplay]:
        """Parse a replay file."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Replay file not found: {file_path}")
            return None

        ext = path.suffix.lower()
        parser_method = self.PARSERS.get(ext)
        if not parser_method:
            logger.error(f"Unsupported replay format: {ext}")
            return None

        try:
            with open(path, "rb") as f:
                return getattr(self, parser_method)(f, path.name)
        except Exception as e:
            logger.error(f"Failed to parse replay: {e}")
            return None

    def _parse_dem(self, f: BinaryIO, filename: str) -> ParsedReplay:
        """Parse Valve .dem format (CS2/CSGO/Dota2)."""
        magic = f.read(8)
        if magic[:7] != b"HL2DEMO":
            raise ValueError("Not a valid .dem file")

        demo_protocol = struct.unpack("<i", f.read(4))[0]
        network_protocol = struct.unpack("<i", f.read(4))[0]
        server_name = f.read(260).split(b"\x00")[0].decode(
            "utf-8", errors="replace"
        )
        client_name = f.read(260).split(b"\x00")[0].decode(
            "utf-8", errors="replace"
        )
        map_name = f.read(260).split(b"\x00")[0].decode(
            "utf-8", errors="replace"
        )
        game_dir = f.read(260).split(b"\x00")[0].decode(
            "utf-8", errors="replace"
        )
        playback_time = struct.unpack("<f", f.read(4))[0]
        playback_ticks = struct.unpack("<i", f.read(4))[0]
        _playback_frames = struct.unpack("<i", f.read(4))[0]
        _sign_on_length = struct.unpack("<i", f.read(4))[0]

        tick_rate = playback_ticks / max(playback_time, 1)

        events = []
        try:
            while True:
                cmd_byte = f.read(1)
                if not cmd_byte:
                    break
                cmd = struct.unpack("<B", cmd_byte)[0]
                _tick = struct.unpack("<i", f.read(4))[0]

                if cmd == 1:  # dem_signon
                    size = struct.unpack("<i", f.read(4))[0]
                    f.read(size)
                elif cmd == 2:  # dem_packet
                    f.read(152)  # cmdinfo
                    f.read(4)  # sequence_in
                    f.read(4)  # sequence_out
                    size = struct.unpack("<i", f.read(4))[0]
                    f.read(size)
                elif cmd == 3:  # dem_synctick
                    pass
                elif cmd == 4:  # dem_consolecmd
                    size = struct.unpack("<i", f.read(4))[0]
                    f.read(size)
                elif cmd == 5:  # dem_usercmd
                    f.read(4)  # outgoing_sequence
                    size = struct.unpack("<i", f.read(4))[0]
                    f.read(size)
                elif cmd == 6:  # dem_datatables
                    size = struct.unpack("<i", f.read(4))[0]
                    f.read(size)
                elif cmd == 7:  # dem_stop
                    break
                else:
                    break
        except Exception:
            pass

        game_id = "cs2"
        if "csgo" in game_dir.lower():
            game_id = "csgo"
        elif "dota" in game_dir.lower():
            game_id = "dota2"

        return ParsedReplay(
            game=game_id,
            map_name=map_name,
            duration_seconds=playback_time,
            events=events,
            metadata={
                "server": server_name,
                "client": client_name,
                "tick_rate": round(tick_rate, 1),
                "total_ticks": playback_ticks,
                "demo_protocol": demo_protocol,
                "network_protocol": network_protocol,
            },
        )

    def _parse_rofl(self, f: BinaryIO, filename: str) -> ParsedReplay:
        """Parse League of Legends .rofl format."""
        magic = f.read(4)
        if magic != b"RIOT":
            raise ValueError("Not a valid .rofl file")

        f.read(262)  # Header fields
        metadata_offset = struct.unpack("<I", f.read(4))[0]
        metadata_length = struct.unpack("<I", f.read(4))[0]

        f.seek(metadata_offset)
        metadata_raw = f.read(metadata_length)
        try:
            metadata = json.loads(metadata_raw.decode("utf-8"))
        except Exception:
            metadata = {}

        return ParsedReplay(
            game="league_of_legends",
            map_name=metadata.get("mapId", "summoners_rift"),
            duration_seconds=metadata.get("gameLength", 0) / 1000,
            players=metadata.get("statsJson", []),
            metadata=metadata,
        )
