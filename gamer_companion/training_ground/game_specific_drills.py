"""Game-Specific Drills — Custom training drills per game.

Each game has unique mechanics that need specific practice:
- CS2: Spray control, smoke lineups, flash angles
- Valorant: Ability usage, agent-specific mechanics
- LoL: Last hitting, jungle clear, wave management
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class Drill:
    """A specific training drill."""
    drill_id: str
    game_id: str
    name: str
    description: str
    category: str             # "aim", "utility", "movement", "economy", "game_sense"
    difficulty: float = 0.5
    duration_seconds: int = 60
    steps: List[str] = field(default_factory=list)
    success_metric: str = ""
    best_score: float = 0.0
    attempts: int = 0


@dataclass
class DrillResult:
    """Result of completing a drill."""
    drill_id: str
    score: float              # 0-100
    completion_time_s: float
    notes: str = ""
    timestamp: float = field(default_factory=time.time)


# Pre-built game-specific drills
GAME_DRILLS: Dict[str, List[Drill]] = {
    "cs2": [
        Drill(
            drill_id="cs2_ak_spray", game_id="cs2",
            name="AK-47 Spray Control",
            description="Control AK spray on a wall for 30 rounds. Keep grouping tight.",
            category="aim", difficulty=0.6, duration_seconds=60,
            steps=["equip_ak47", "aim_wall", "spray_30_bullets", "check_grouping"],
            success_metric="grouping_diameter < 50px at 15m",
        ),
        Drill(
            drill_id="cs2_smoke_mirage_a", game_id="cs2",
            name="Mirage A Smokes",
            description="Throw 4 key A-site smokes: CT, stairs, jungle, connector.",
            category="utility", difficulty=0.5, duration_seconds=30,
            steps=["buy_smokes", "throw_ct_smoke", "throw_stairs_smoke",
                   "throw_jungle_smoke", "throw_connector_smoke"],
            success_metric="all 4 smokes land correctly",
        ),
        Drill(
            drill_id="cs2_counter_strafe", game_id="cs2",
            name="Counter-Strafe Shooting",
            description="Strafe, stop, one-tap. Repeat 20 times.",
            category="movement", difficulty=0.5, duration_seconds=45,
            steps=["strafe_left", "counter_strafe", "one_tap"] * 20,
            success_metric="hit_rate > 60% with velocity < 5",
        ),
        Drill(
            drill_id="cs2_eco_management", game_id="cs2",
            name="Economy Decisions",
            description="Practice buy/save decisions across 15 round scenarios.",
            category="economy", difficulty=0.4, duration_seconds=90,
            success_metric="team_economy_optimal > 80%",
        ),
        Drill(
            drill_id="cs2_retake", game_id="cs2",
            name="Retake Scenarios",
            description="1v1, 2v1, 3v2 retake situations on A/B sites.",
            category="game_sense", difficulty=0.7, duration_seconds=120,
            success_metric="retake_success > 50%",
        ),
    ],
    "valorant": [
        Drill(
            drill_id="val_spray_transfer", game_id="valorant",
            name="Spray Transfer",
            description="Kill two targets in one spray. Transfer between targets.",
            category="aim", difficulty=0.7, duration_seconds=60,
            success_metric="transfer_time < 300ms",
        ),
        Drill(
            drill_id="val_ability_combo", game_id="valorant",
            name="Ability Combo Practice",
            description="Chain abilities: flash + peek + shoot.",
            category="utility", difficulty=0.5, duration_seconds=45,
            success_metric="combo_execution_time < 500ms",
        ),
    ],
    "league": [
        Drill(
            drill_id="lol_last_hit", game_id="league",
            name="Last Hit Practice",
            description="Get 10 CS per minute in practice tool for 5 minutes.",
            category="aim", difficulty=0.4, duration_seconds=300,
            success_metric="cs_per_min >= 8",
        ),
        Drill(
            drill_id="lol_jungle_clear", game_id="league",
            name="Jungle Full Clear",
            description="Full jungle clear in under 3:15.",
            category="movement", difficulty=0.6, duration_seconds=200,
            success_metric="clear_time < 3:15",
        ),
        Drill(
            drill_id="lol_wave_management", game_id="league",
            name="Wave Management",
            description="Freeze, slow push, fast push, and crash waves.",
            category="game_sense", difficulty=0.7, duration_seconds=180,
            success_metric="correct_wave_state > 80%",
        ),
    ],
}


class GameSpecificDrills:
    """Manage and run game-specific training drills."""

    def __init__(self):
        self._drills: Dict[str, Dict[str, Drill]] = {}
        self._results: List[DrillResult] = []

        # Load pre-built drills
        for game_id, drills in GAME_DRILLS.items():
            self._drills[game_id] = {}
            for drill in drills:
                self._drills[game_id][drill.drill_id] = drill

    def get_drills(self, game_id: str) -> List[Drill]:
        """Get all drills for a game."""
        game_drills = self._drills.get(game_id, {})
        return list(game_drills.values())

    def get_drill(self, drill_id: str) -> Optional[Drill]:
        """Get a specific drill by ID."""
        for game_drills in self._drills.values():
            if drill_id in game_drills:
                return game_drills[drill_id]
        return None

    def record_result(self, drill_id: str, score: float, completion_time_s: float, notes: str = "") -> Optional[DrillResult]:
        """Record a drill result."""
        drill = self.get_drill(drill_id)
        if not drill:
            return None

        drill.attempts += 1
        drill.best_score = max(drill.best_score, score)

        result = DrillResult(
            drill_id=drill_id,
            score=round(score, 1),
            completion_time_s=round(completion_time_s, 1),
            notes=notes,
        )
        self._results.append(result)
        return result

    def add_custom_drill(self, drill: Drill):
        """Add a custom drill."""
        if drill.game_id not in self._drills:
            self._drills[drill.game_id] = {}
        self._drills[drill.game_id][drill.drill_id] = drill

    def get_recommended(self, game_id: str, limit: int = 3) -> List[Drill]:
        """Get recommended drills (weakest areas first)."""
        drills = self.get_drills(game_id)
        if not drills:
            return []

        # Sort by: lowest best_score first, then least attempted
        return sorted(
            drills,
            key=lambda d: (d.best_score, d.attempts),
        )[:limit]

    def list_games(self) -> List[str]:
        """List games with available drills."""
        return list(self._drills.keys())

    def get_stats(self, game_id: str = None) -> dict:
        if game_id:
            drills = self.get_drills(game_id)
            results = [r for r in self._results if self.get_drill(r.drill_id) and self.get_drill(r.drill_id).game_id == game_id]
        else:
            drills = []
            for g in self._drills.values():
                drills.extend(g.values())
            results = self._results

        return {
            "total_drills": len(drills),
            "total_attempts": sum(d.attempts for d in drills),
            "results_recorded": len(results),
            "avg_score": round(sum(r.score for r in results) / max(1, len(results)), 1),
            "games": self.list_games(),
        }
