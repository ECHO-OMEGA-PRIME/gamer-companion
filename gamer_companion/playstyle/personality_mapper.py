"""Personality Mapper — Map ECHO personalities to playstyles.

Bridges ECHO's 14 AI personalities into game playstyle profiles.
Each ECHO personality has traits that translate into how the AI plays:
- Aggressive personality → entry fragger playstyle
- Analytical personality → IGL/support playstyle
- Creative personality → lurker playstyle
- etc.

Allows the AI to "be" a character while playing, affecting decisions,
communication style, risk tolerance, and combat behavior.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


@dataclass
class PersonalityTraits:
    """Core personality traits that influence playstyle."""
    name: str
    description: str = ""

    # Personality axes (0.0 to 1.0)
    aggression: float = 0.5
    analytical: float = 0.5
    creative: float = 0.5
    leadership: float = 0.5
    patience: float = 0.5
    social: float = 0.5
    confidence: float = 0.5
    adaptability: float = 0.5
    discipline: float = 0.5
    humor: float = 0.0       # Affects communication tone


@dataclass
class PlaystyleMapping:
    """Maps personality traits to playstyle parameters."""
    personality_name: str
    playstyle_name: str       # Target playstyle (entry_fragger, lurker, etc.)
    aggression: float = 0.5
    risk_tolerance: float = 0.5
    patience: float = 0.5
    team_play: float = 0.5
    creativity: float = 0.5
    economy: float = 0.5
    aim_speed: float = 0.7
    spray_discipline: float = 0.7
    headshot_focus: float = 0.5
    callout_frequency: float = 0.5
    toxicity: float = 0.0
    tilt_resistance: float = 0.7


# Pre-built personality-to-playstyle mappings
PERSONALITY_MAP: Dict[str, PlaystyleMapping] = {
    "commander": PlaystyleMapping(
        personality_name="commander",
        playstyle_name="igl",
        aggression=0.5, risk_tolerance=0.4, patience=0.7,
        team_play=0.95, creativity=0.7, economy=0.8,
        aim_speed=0.6, callout_frequency=1.0, tilt_resistance=0.95,
    ),
    "sentinel": PlaystyleMapping(
        personality_name="sentinel",
        playstyle_name="awper",
        aggression=0.3, risk_tolerance=0.2, patience=0.9,
        team_play=0.5, creativity=0.3, economy=0.8,
        aim_speed=0.95, spray_discipline=1.0, headshot_focus=0.3,
        callout_frequency=0.6, tilt_resistance=0.85,
    ),
    "shadow": PlaystyleMapping(
        personality_name="shadow",
        playstyle_name="lurker",
        aggression=0.4, risk_tolerance=0.5, patience=0.85,
        team_play=0.2, creativity=0.9, economy=0.6,
        aim_speed=0.7, headshot_focus=0.8,
        callout_frequency=0.2, tilt_resistance=0.8,
    ),
    "berserker": PlaystyleMapping(
        personality_name="berserker",
        playstyle_name="entry_fragger",
        aggression=0.95, risk_tolerance=0.9, patience=0.1,
        team_play=0.4, creativity=0.3, economy=0.2,
        aim_speed=0.9, spray_discipline=0.3, headshot_focus=0.5,
        callout_frequency=0.7, tilt_resistance=0.4, toxicity=0.2,
    ),
    "oracle": PlaystyleMapping(
        personality_name="oracle",
        playstyle_name="support",
        aggression=0.3, risk_tolerance=0.3, patience=0.8,
        team_play=0.9, creativity=0.6, economy=0.9,
        aim_speed=0.5, spray_discipline=0.7,
        callout_frequency=0.9, tilt_resistance=0.9,
    ),
    "tactician": PlaystyleMapping(
        personality_name="tactician",
        playstyle_name="igl",
        aggression=0.4, risk_tolerance=0.3, patience=0.7,
        team_play=0.85, creativity=0.8, economy=0.85,
        aim_speed=0.6, spray_discipline=0.8,
        callout_frequency=0.95, tilt_resistance=0.9,
    ),
    "maverick": PlaystyleMapping(
        personality_name="maverick",
        playstyle_name="entry_fragger",
        aggression=0.8, risk_tolerance=0.85, patience=0.2,
        team_play=0.3, creativity=0.95, economy=0.3,
        aim_speed=0.85, headshot_focus=0.7,
        callout_frequency=0.5, tilt_resistance=0.6,
    ),
    "guardian": PlaystyleMapping(
        personality_name="guardian",
        playstyle_name="support",
        aggression=0.2, risk_tolerance=0.15, patience=0.7,
        team_play=0.95, creativity=0.4, economy=0.8,
        aim_speed=0.5, spray_discipline=0.8,
        callout_frequency=0.85, tilt_resistance=0.85,
    ),
    "scholar": PlaystyleMapping(
        personality_name="scholar",
        playstyle_name="support",
        aggression=0.2, risk_tolerance=0.2, patience=0.9,
        team_play=0.7, creativity=0.5, economy=0.95,
        aim_speed=0.4, spray_discipline=0.9,
        callout_frequency=0.8, tilt_resistance=0.95,
    ),
    "jester": PlaystyleMapping(
        personality_name="jester",
        playstyle_name="lurker",
        aggression=0.6, risk_tolerance=0.7, patience=0.4,
        team_play=0.5, creativity=1.0, economy=0.4,
        aim_speed=0.7, headshot_focus=0.6,
        callout_frequency=0.8, tilt_resistance=0.5,
        toxicity=0.1,
    ),
}


class PersonalityMapper:
    """Map personalities to game playstyles.

    Provides a bridge between ECHO's personality system and
    the playstyle engine. Each personality maps to a set of
    playstyle parameters that affect all AI subsystems.

    Supports custom personality→playstyle mappings.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._mappings: Dict[str, PlaystyleMapping] = {**PERSONALITY_MAP}
        self._active: Optional[PlaystyleMapping] = None
        self._persist_path = Path(persist_path) if persist_path else None
        self._load_custom()

    def set_personality(self, name: str) -> bool:
        """Activate a personality mapping."""
        mapping = self._mappings.get(name.lower())
        if not mapping:
            logger.warning(f"Unknown personality: {name}")
            return False
        self._active = mapping
        logger.info(
            f"Personality: {name} → playstyle: {mapping.playstyle_name}"
        )
        return True

    def get_mapping(self, name: str) -> Optional[PlaystyleMapping]:
        return self._mappings.get(name.lower())

    def add_custom(self, mapping: PlaystyleMapping):
        """Add a custom personality mapping."""
        self._mappings[mapping.personality_name.lower()] = mapping
        self._save_custom()

    @property
    def active(self) -> Optional[PlaystyleMapping]:
        return self._active

    @property
    def active_name(self) -> str:
        return self._active.personality_name if self._active else "none"

    def get_playstyle_params(self) -> Dict[str, float]:
        """Get the active personality's playstyle parameters as a dict."""
        if not self._active:
            return {}
        m = self._active
        return {
            "aggression": m.aggression,
            "risk_tolerance": m.risk_tolerance,
            "patience": m.patience,
            "team_play": m.team_play,
            "creativity": m.creativity,
            "economy": m.economy,
            "aim_speed": m.aim_speed,
            "spray_discipline": m.spray_discipline,
            "headshot_focus": m.headshot_focus,
            "callout_frequency": m.callout_frequency,
            "toxicity": m.toxicity,
            "tilt_resistance": m.tilt_resistance,
        }

    def from_traits(self, traits: PersonalityTraits) -> PlaystyleMapping:
        """Generate a playstyle mapping from raw personality traits.

        Uses heuristic mapping from personality axes to playstyle params.
        """
        # Determine base role from dominant trait
        if traits.aggression > 0.7 and traits.confidence > 0.6:
            role = "entry_fragger"
        elif traits.leadership > 0.7 and traits.social > 0.6:
            role = "igl"
        elif traits.patience > 0.7 and traits.creative > 0.6:
            role = "lurker"
        elif traits.analytical > 0.7 and traits.discipline > 0.6:
            role = "awper"
        else:
            role = "support"

        mapping = PlaystyleMapping(
            personality_name=traits.name,
            playstyle_name=role,
            aggression=traits.aggression,
            risk_tolerance=traits.confidence * 0.8,
            patience=traits.patience,
            team_play=traits.social * 0.8 + traits.leadership * 0.2,
            creativity=traits.creative,
            economy=traits.discipline * 0.7 + traits.patience * 0.3,
            aim_speed=0.5 + traits.confidence * 0.4,
            spray_discipline=traits.discipline,
            headshot_focus=traits.analytical * 0.6 + traits.confidence * 0.4,
            callout_frequency=traits.social,
            toxicity=max(0, traits.aggression - traits.social) * 0.2,
            tilt_resistance=traits.discipline * 0.5 + traits.adaptability * 0.5,
        )

        return mapping

    def list_personalities(self) -> List[Dict[str, str]]:
        return [
            {
                "personality": m.personality_name,
                "playstyle": m.playstyle_name,
            }
            for m in self._mappings.values()
        ]

    def _save_custom(self):
        if not self._persist_path:
            return
        custom = {}
        for name, mapping in self._mappings.items():
            if name not in PERSONALITY_MAP:
                custom[name] = {
                    "personality_name": mapping.personality_name,
                    "playstyle_name": mapping.playstyle_name,
                    "aggression": mapping.aggression,
                    "risk_tolerance": mapping.risk_tolerance,
                    "patience": mapping.patience,
                    "team_play": mapping.team_play,
                    "creativity": mapping.creativity,
                    "economy": mapping.economy,
                    "aim_speed": mapping.aim_speed,
                    "spray_discipline": mapping.spray_discipline,
                    "headshot_focus": mapping.headshot_focus,
                    "callout_frequency": mapping.callout_frequency,
                    "toxicity": mapping.toxicity,
                    "tilt_resistance": mapping.tilt_resistance,
                }
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(custom, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save custom mappings: {e}")

    def _load_custom(self):
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for name, vals in data.items():
                self._mappings[name] = PlaystyleMapping(**vals)
        except Exception as e:
            logger.warning(f"Failed to load custom mappings: {e}")

    def get_stats(self) -> dict:
        return {
            "total_personalities": len(self._mappings),
            "preset_count": len(PERSONALITY_MAP),
            "custom_count": len(self._mappings) - len(PERSONALITY_MAP),
            "active": self.active_name,
            "active_role": self._active.playstyle_name if self._active else "none",
        }
