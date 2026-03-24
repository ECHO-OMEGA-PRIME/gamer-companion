"""Pro Mimic — Mimic specific professional player playstyles.

Stores playstyle profiles extracted from pro player analysis,
allowing the AI to "play like" a specific pro.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


@dataclass
class ProProfile:
    """A professional player's playstyle profile."""
    player_name: str
    game: str                 # "cs2", "valorant", "league"
    role: str                 # "entry", "awper", "lurker", "igl", "support"
    team: str = ""

    # Playstyle parameters (0.0 to 1.0)
    aggression: float = 0.5
    risk_tolerance: float = 0.5
    patience: float = 0.5
    aim_speed: float = 0.5
    spray_control: float = 0.5
    headshot_rate: float = 0.5
    utility_usage: float = 0.5
    positioning: float = 0.5
    team_play: float = 0.5
    clutch_factor: float = 0.5
    consistency: float = 0.5
    creativity: float = 0.5

    # Signature moves
    signature_plays: List[str] = field(default_factory=list)

    # Stat context
    rating: float = 0.0      # HLTV rating, op.gg rank, etc.
    maps_strong: List[str] = field(default_factory=list)
    maps_weak: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "game": self.game,
            "role": self.role,
            "team": self.team,
            "aggression": self.aggression,
            "risk_tolerance": self.risk_tolerance,
            "patience": self.patience,
            "aim_speed": self.aim_speed,
            "spray_control": self.spray_control,
            "headshot_rate": self.headshot_rate,
            "utility_usage": self.utility_usage,
            "positioning": self.positioning,
            "team_play": self.team_play,
            "clutch_factor": self.clutch_factor,
            "consistency": self.consistency,
            "creativity": self.creativity,
            "rating": self.rating,
        }


# Pre-built pro profiles (CS2 archetypes)
PRO_PROFILES: Dict[str, ProProfile] = {
    "s1mple": ProProfile(
        player_name="s1mple", game="cs2", role="awper", team="NAVI",
        aggression=0.75, risk_tolerance=0.7, patience=0.4,
        aim_speed=0.99, spray_control=0.9, headshot_rate=0.65,
        utility_usage=0.5, positioning=0.85, team_play=0.6,
        clutch_factor=0.95, consistency=0.9, creativity=0.85,
        rating=1.25,
        signature_plays=["aggressive_awp_peek", "no_scope", "deagle_ace"],
        maps_strong=["mirage", "inferno", "nuke"],
    ),
    "zywoo": ProProfile(
        player_name="ZywOo", game="cs2", role="awper", team="Vitality",
        aggression=0.5, risk_tolerance=0.4, patience=0.7,
        aim_speed=0.98, spray_control=0.85, headshot_rate=0.55,
        utility_usage=0.6, positioning=0.95, team_play=0.7,
        clutch_factor=0.9, consistency=0.95, creativity=0.7,
        rating=1.28,
        signature_plays=["passive_awp_hold", "clutch_spray_transfer"],
        maps_strong=["dust2", "overpass", "vertigo"],
    ),
    "niko": ProProfile(
        player_name="NiKo", game="cs2", role="entry", team="G2",
        aggression=0.85, risk_tolerance=0.75, patience=0.3,
        aim_speed=0.95, spray_control=0.95, headshot_rate=0.7,
        utility_usage=0.5, positioning=0.7, team_play=0.5,
        clutch_factor=0.85, consistency=0.8, creativity=0.6,
        rating=1.18,
        signature_plays=["one_tap_headshots", "deagle_specialist"],
        maps_strong=["mirage", "dust2", "inferno"],
    ),
    "ropz": ProProfile(
        player_name="ropz", game="cs2", role="lurker", team="FaZe",
        aggression=0.3, risk_tolerance=0.25, patience=0.95,
        aim_speed=0.85, spray_control=0.9, headshot_rate=0.55,
        utility_usage=0.7, positioning=0.95, team_play=0.4,
        clutch_factor=0.8, consistency=0.95, creativity=0.6,
        rating=1.12,
        signature_plays=["late_lurk_flank", "patient_post_plant"],
        maps_strong=["overpass", "ancient", "nuke"],
    ),
    "faker": ProProfile(
        player_name="Faker", game="league", role="mid", team="T1",
        aggression=0.7, risk_tolerance=0.5, patience=0.6,
        aim_speed=0.9, spray_control=0.9, headshot_rate=0.0,
        utility_usage=0.95, positioning=0.9, team_play=0.8,
        clutch_factor=0.95, consistency=0.95, creativity=0.9,
        rating=0.0,
        signature_plays=["mechanical_outplay", "lane_dominance", "team_fight_carry"],
    ),
}


class ProMimic:
    """Mimic professional player playstyles.

    Usage:
    1. Select a pro to mimic
    2. Get their playstyle profile
    3. Feed profile into StyleEngine, AggressionController, AimEngine
    4. AI adjusts behavior to match the pro's tendencies
    """

    def __init__(self, custom_profiles_path: Optional[str] = None):
        self._profiles = {**PRO_PROFILES}
        self._active: Optional[ProProfile] = None
        self._custom_path = Path(custom_profiles_path) if custom_profiles_path else None
        self._load_custom()

    def set_active(self, player_name: str) -> bool:
        """Activate a pro player's style."""
        # Case-insensitive lookup
        for key, profile in self._profiles.items():
            if key.lower() == player_name.lower() or profile.player_name.lower() == player_name.lower():
                self._active = profile
                logger.info(f"Mimicking: {profile.player_name} ({profile.role}, {profile.team})")
                return True

        logger.warning(f"Unknown pro player: {player_name}")
        return False

    def clear_active(self):
        """Stop mimicking."""
        self._active = None

    @property
    def active(self) -> Optional[ProProfile]:
        return self._active

    @property
    def is_mimicking(self) -> bool:
        return self._active is not None

    def get_playstyle_params(self) -> dict:
        """Get active pro's params formatted for StyleEngine."""
        if not self._active:
            return {}

        return {
            "aggression": self._active.aggression,
            "risk_tolerance": self._active.risk_tolerance,
            "patience": self._active.patience,
            "aim_speed": self._active.aim_speed,
            "spray_discipline": self._active.spray_control,
            "headshot_focus": self._active.headshot_rate,
            "creativity": self._active.creativity,
            "team_play": self._active.team_play,
        }

    def list_pros(self, game: str = None) -> List[dict]:
        """List available pro profiles."""
        profiles = self._profiles.values()
        if game:
            profiles = [p for p in profiles if p.game == game]

        return [
            {
                "name": p.player_name,
                "game": p.game,
                "role": p.role,
                "team": p.team,
                "rating": p.rating,
            }
            for p in profiles
        ]

    def add_custom(self, profile: ProProfile):
        """Add a custom pro profile."""
        key = profile.player_name.lower().replace(" ", "_")
        self._profiles[key] = profile
        self._save_custom()

    def _save_custom(self):
        if not self._custom_path:
            return
        custom = {
            k: v.to_dict()
            for k, v in self._profiles.items()
            if k not in PRO_PROFILES
        }
        self._custom_path.parent.mkdir(parents=True, exist_ok=True)
        self._custom_path.write_text(json.dumps(custom, indent=2))

    def _load_custom(self):
        if not self._custom_path or not self._custom_path.exists():
            return
        try:
            data = json.loads(self._custom_path.read_text())
            for name, vals in data.items():
                self._profiles[name] = ProProfile(**vals)
        except Exception as e:
            logger.warning(f"Failed to load custom pro profiles: {e}")
