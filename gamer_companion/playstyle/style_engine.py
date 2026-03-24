"""Style Engine — Configurable playstyle parameters.

Defines the AI's personality as a player: how aggressive, how risky,
how patient, how team-oriented. These parameters flow into combat,
aim, movement, and communication systems.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
from loguru import logger


@dataclass
class PlaystyleProfile:
    """A complete playstyle configuration."""
    name: str
    description: str = ""

    # Core personality axes (0.0 to 1.0)
    aggression: float = 0.5      # 0=passive, 1=hyper-aggro
    risk_tolerance: float = 0.5  # 0=play safe, 1=yolo
    patience: float = 0.5        # 0=rush, 1=wait forever
    team_play: float = 0.5       # 0=solo, 1=pure support
    creativity: float = 0.5      # 0=textbook, 1=unpredictable
    economy: float = 0.5         # 0=force every round, 1=save religiously

    # Aim behavior
    aim_speed: float = 0.7       # 0=slow methodical, 1=instant flick
    spray_discipline: float = 0.7  # 0=spray everything, 1=tap/burst only
    headshot_focus: float = 0.5  # 0=body shots, 1=head only

    # Communication
    callout_frequency: float = 0.5  # 0=silent, 1=constant comms
    toxicity: float = 0.0        # 0=positive, 1=toxic (for realism)

    # Adaptability
    tilt_resistance: float = 0.7  # 0=tilts easily, 1=unbreakable mental

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "aggression": self.aggression,
            "risk_tolerance": self.risk_tolerance,
            "patience": self.patience,
            "team_play": self.team_play,
            "creativity": self.creativity,
            "economy": self.economy,
            "aim_speed": self.aim_speed,
            "spray_discipline": self.spray_discipline,
            "headshot_focus": self.headshot_focus,
            "callout_frequency": self.callout_frequency,
            "tilt_resistance": self.tilt_resistance,
        }


# Pre-built playstyle profiles modeled after real archetypes
PRESET_STYLES: Dict[str, PlaystyleProfile] = {
    "entry_fragger": PlaystyleProfile(
        name="entry_fragger",
        description="First in, creates space. High aggression, fast aim.",
        aggression=0.85, risk_tolerance=0.8, patience=0.2,
        team_play=0.6, creativity=0.4, economy=0.3,
        aim_speed=0.9, spray_discipline=0.5, headshot_focus=0.7,
        callout_frequency=0.8, tilt_resistance=0.6,
    ),
    "lurker": PlaystyleProfile(
        name="lurker",
        description="Sneaky flanker. Patient, creative, solo-oriented.",
        aggression=0.4, risk_tolerance=0.5, patience=0.9,
        team_play=0.2, creativity=0.9, economy=0.6,
        aim_speed=0.6, spray_discipline=0.8, headshot_focus=0.8,
        callout_frequency=0.3, tilt_resistance=0.8,
    ),
    "awper": PlaystyleProfile(
        name="awper",
        description="Sniper specialist. Precise, patient, positional.",
        aggression=0.3, risk_tolerance=0.3, patience=0.8,
        team_play=0.5, creativity=0.3, economy=0.8,
        aim_speed=0.95, spray_discipline=1.0, headshot_focus=0.3,
        callout_frequency=0.7, tilt_resistance=0.7,
    ),
    "igl": PlaystyleProfile(
        name="igl",
        description="In-game leader. Tactical, team-first, calls strats.",
        aggression=0.4, risk_tolerance=0.3, patience=0.7,
        team_play=0.95, creativity=0.7, economy=0.9,
        aim_speed=0.5, spray_discipline=0.7, headshot_focus=0.5,
        callout_frequency=1.0, tilt_resistance=0.9,
    ),
    "support": PlaystyleProfile(
        name="support",
        description="Utility player. Flashes, smokes, trades, saves.",
        aggression=0.3, risk_tolerance=0.2, patience=0.6,
        team_play=0.9, creativity=0.4, economy=0.7,
        aim_speed=0.5, spray_discipline=0.6, headshot_focus=0.4,
        callout_frequency=0.8, tilt_resistance=0.8,
    ),
    "bot_easy": PlaystyleProfile(
        name="bot_easy",
        description="Easy bot. Slow, predictable, bad aim.",
        aggression=0.2, risk_tolerance=0.2, patience=0.3,
        team_play=0.1, creativity=0.1, economy=0.3,
        aim_speed=0.2, spray_discipline=0.2, headshot_focus=0.1,
        callout_frequency=0.1, tilt_resistance=0.3,
    ),
    "bot_hard": PlaystyleProfile(
        name="bot_hard",
        description="Hard bot. Fast, aggressive, good but not inhuman.",
        aggression=0.8, risk_tolerance=0.7, patience=0.4,
        team_play=0.5, creativity=0.6, economy=0.5,
        aim_speed=0.85, spray_discipline=0.8, headshot_focus=0.7,
        callout_frequency=0.6, tilt_resistance=0.8,
    ),
}


class StyleEngine:
    """Manage and apply playstyle profiles.

    The active style flows into all AI subsystems:
    - CombatEngine reads aggression + risk_tolerance
    - AimEngine reads aim_speed + spray_discipline + headshot_focus
    - Humanizer reads from profile for timing distributions
    - CommunicationAI reads callout_frequency
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._styles = {**PRESET_STYLES}
        self._active: PlaystyleProfile = PRESET_STYLES["entry_fragger"]
        self._persist_path = Path(persist_path) if persist_path else None
        self._load_custom()

    def set_active(self, name: str) -> bool:
        """Switch to a named playstyle."""
        style = self._styles.get(name)
        if not style:
            logger.warning(f"Unknown playstyle: {name}")
            return False
        self._active = style
        logger.info(f"Playstyle: {name} (aggro={style.aggression})")
        return True

    def add_custom(self, profile: PlaystyleProfile):
        """Add a custom playstyle profile."""
        self._styles[profile.name] = profile
        self._save_custom()

    @property
    def active(self) -> PlaystyleProfile:
        return self._active

    @property
    def aggression(self) -> float:
        return self._active.aggression

    @property
    def risk_tolerance(self) -> float:
        return self._active.risk_tolerance

    def list_styles(self) -> list:
        return [
            {"name": s.name, "description": s.description}
            for s in self._styles.values()
        ]

    def _save_custom(self):
        if not self._persist_path:
            return
        custom = {
            k: v.to_dict()
            for k, v in self._styles.items()
            if k not in PRESET_STYLES
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(custom, indent=2))

    def _load_custom(self):
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for name, vals in data.items():
                self._styles[name] = PlaystyleProfile(**vals)
        except Exception as e:
            logger.warning(f"Failed to load custom styles: {e}")
