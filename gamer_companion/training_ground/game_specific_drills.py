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
    # --- Call of Duty ---
    "cod_mp": [
        Drill(
            drill_id="cod_slide_cancel", game_id="cod_mp",
            name="Slide Cancel Drill",
            description="Execute 20 consecutive slide cancels while maintaining sprint momentum.",
            category="movement", difficulty=0.5, duration_seconds=45,
            steps=["tac_sprint", "slide", "cancel", "repeat"] * 20,
            success_metric="20_consecutive_clean_cancels",
        ),
        Drill(
            drill_id="cod_centering", game_id="cod_mp",
            name="Pre-Aim Centering",
            description="Walk through map keeping crosshair at head height on common angles.",
            category="aim", difficulty=0.4, duration_seconds=60,
            steps=["load_map", "walk_lane", "check_crosshair_height", "adjust"],
            success_metric="head_height_accuracy > 75%",
        ),
        Drill(
            drill_id="cod_ar_spray", game_id="cod_mp",
            name="AR Recoil Control",
            description="Control M4 and AK spray on wall target at 20m. Full mag.",
            category="aim", difficulty=0.6, duration_seconds=60,
            steps=["equip_ar", "aim_wall", "full_spray", "check_grouping"],
            success_metric="grouping_diameter < 60px at 20m",
        ),
        Drill(
            drill_id="cod_spawn_awareness", game_id="cod_mp",
            name="Spawn Prediction",
            description="Predict next enemy spawn based on minimap teammate positions.",
            category="game_sense", difficulty=0.7, duration_seconds=120,
            success_metric="spawn_prediction_accuracy > 60%",
        ),
        Drill(
            drill_id="cod_sniper_quickscope", game_id="cod_mp",
            name="Quickscope Practice",
            description="ADS and fire within 300ms. Hit 15/20 quickscopes.",
            category="aim", difficulty=0.8, duration_seconds=90,
            success_metric="15/20 hits within 300ms ADS",
        ),
    ],
    "cod_warzone": [
        Drill(
            drill_id="wz_drop_loot", game_id="cod_warzone",
            name="Hot Drop Loot Speed",
            description="Land, loot weapon+plates in under 15 seconds.",
            category="movement", difficulty=0.5, duration_seconds=30,
            steps=["drop", "land_building", "grab_weapon", "plate_up"],
            success_metric="full_kit < 15s",
        ),
        Drill(
            drill_id="wz_rotation", game_id="cod_warzone",
            name="Zone Rotation",
            description="Practice rotating to next zone using cover and vehicles.",
            category="game_sense", difficulty=0.6, duration_seconds=120,
            success_metric="arrive_in_zone_alive_with_plates",
        ),
        Drill(
            drill_id="wz_buy_station", game_id="cod_warzone",
            name="Buy Station Speed",
            description="Navigate buy station menu and purchase loadout in <3 seconds.",
            category="economy", difficulty=0.3, duration_seconds=30,
            success_metric="purchase_time < 3s",
        ),
    ],
    # --- Apex Legends ---
    "apex": [
        Drill(
            drill_id="apex_recoil_r301", game_id="apex",
            name="R-301 Recoil Control",
            description="One-clip a dummy at 50m with R-301. Purple mag.",
            category="aim", difficulty=0.5, duration_seconds=60,
            success_metric="one_clip_at_50m",
        ),
        Drill(
            drill_id="apex_wall_bounce", game_id="apex",
            name="Wall Bounce Movement",
            description="Execute 10 wall bounces in firing range.",
            category="movement", difficulty=0.7, duration_seconds=60,
            steps=["sprint_at_wall", "jump", "air_strafe", "jump_off_wall"] * 10,
            success_metric="10_clean_wall_bounces",
        ),
        Drill(
            drill_id="apex_armor_swap", game_id="apex",
            name="Armor Swap Speed",
            description="Kill a dummy and armor swap from deathbox in <1.5s.",
            category="game_sense", difficulty=0.6, duration_seconds=30,
            success_metric="armor_swap_time < 1.5s",
        ),
    ],
    # --- Overwatch 2 ---
    "overwatch2": [
        Drill(
            drill_id="ow2_tracking_aim", game_id="overwatch2",
            name="Tracking Aim (Soldier/Tracer)",
            description="Track moving bot for 10 seconds. Maintain 50%+ accuracy.",
            category="aim", difficulty=0.6, duration_seconds=60,
            success_metric="tracking_accuracy > 50%",
        ),
        Drill(
            drill_id="ow2_ult_economy", game_id="overwatch2",
            name="Ultimate Economy",
            description="Build ult charge efficiently. Don't waste ult on lost fights.",
            category="game_sense", difficulty=0.7, duration_seconds=180,
            success_metric="ult_value_score > 70%",
        ),
        Drill(
            drill_id="ow2_ability_cooldowns", game_id="overwatch2",
            name="Ability Cooldown Management",
            description="Track enemy abilities. Punish when on cooldown.",
            category="utility", difficulty=0.8, duration_seconds=120,
            success_metric="punish_rate > 50%",
        ),
    ],
    # --- Rainbow Six Siege ---
    "r6siege": [
        Drill(
            drill_id="r6_one_tap", game_id="r6siege",
            name="One-Tap Headshot",
            description="One-tap terrorist hunt headshots only. 20 kills.",
            category="aim", difficulty=0.7, duration_seconds=120,
            success_metric="20_headshot_kills",
        ),
        Drill(
            drill_id="r6_drone_intel", game_id="r6siege",
            name="Drone Intelligence",
            description="Drone out a site. Call out all 5 defenders positions.",
            category="game_sense", difficulty=0.5, duration_seconds=45,
            success_metric="5/5_positions_called",
        ),
    ],
    # --- StarCraft II ---
    "starcraft2": [
        Drill(
            drill_id="sc2_build_order", game_id="starcraft2",
            name="Build Order Execution",
            description="Execute a 2-base build order hitting benchmarks within 5 seconds.",
            category="economy", difficulty=0.6, duration_seconds=300,
            success_metric="all_benchmarks_within_5s",
        ),
        Drill(
            drill_id="sc2_multitask", game_id="starcraft2",
            name="Multitask Drill",
            description="Macro at home while harassing with 8 units simultaneously.",
            category="game_sense", difficulty=0.9, duration_seconds=180,
            steps=["build_workers", "produce_units", "expand", "harass_enemy_workers"],
            success_metric="no_supply_block + constant_production + harassment_damage",
        ),
        Drill(
            drill_id="sc2_marine_split", game_id="starcraft2",
            name="Marine Split vs Banelings",
            description="Split 20 marines against baneling attack. Lose <5.",
            category="aim", difficulty=0.8, duration_seconds=30,
            success_metric="marines_surviving > 15",
        ),
    ],
    # --- Dota 2 ---
    "dota2": [
        Drill(
            drill_id="dota_last_hit", game_id="dota2",
            name="Last Hit Under Tower",
            description="Last hit creeps under tower for 5 minutes. Include denies.",
            category="aim", difficulty=0.6, duration_seconds=300,
            success_metric="cs+denies > 80%",
        ),
        Drill(
            drill_id="dota_stacking", game_id="dota2",
            name="Camp Stacking",
            description="Stack 3 jungle camps simultaneously at x:55 timing.",
            category="game_sense", difficulty=0.5, duration_seconds=60,
            success_metric="3_camps_stacked",
        ),
    ],
    # --- Street Fighter 6 ---
    "street_fighter_6": [
        Drill(
            drill_id="sf6_confirm_combo", game_id="street_fighter_6",
            name="Hit Confirm into Combo",
            description="Confirm light attack into full combo. React to block/hit.",
            category="aim", difficulty=0.7, duration_seconds=60,
            success_metric="confirm_rate > 80%",
        ),
        Drill(
            drill_id="sf6_anti_air", game_id="street_fighter_6",
            name="Anti-Air Reactions",
            description="React to jump-ins with DP or anti-air normal. 20 reps.",
            category="movement", difficulty=0.6, duration_seconds=45,
            success_metric="anti_air_rate > 75%",
        ),
        Drill(
            drill_id="sf6_drive_impact", game_id="street_fighter_6",
            name="Drive Impact React",
            description="React to Drive Impact with own DI or parry. 15 reps.",
            category="game_sense", difficulty=0.8, duration_seconds=30,
            success_metric="react_rate > 60%",
        ),
    ],
    # --- Tekken 8 ---
    "tekken8": [
        Drill(
            drill_id="tk8_punish", game_id="tekken8",
            name="Punish Drill",
            description="Punish blocked moves with correct frame-data punisher.",
            category="game_sense", difficulty=0.7, duration_seconds=90,
            success_metric="correct_punish > 70%",
        ),
        Drill(
            drill_id="tk8_combo_execution", game_id="tekken8",
            name="Combo Execution",
            description="Land a launcher combo 10 times in a row without dropping.",
            category="aim", difficulty=0.6, duration_seconds=120,
            success_metric="10_consecutive_combos",
        ),
    ],
    # --- Rocket League ---
    "rocket_league": [
        Drill(
            drill_id="rl_aerial", game_id="rocket_league",
            name="Aerial Training",
            description="Hit 10 aerial shots in custom training pack.",
            category="aim", difficulty=0.7, duration_seconds=120,
            success_metric="10_aerial_hits",
        ),
        Drill(
            drill_id="rl_fast_aerial", game_id="rocket_league",
            name="Fast Aerial Drill",
            description="Execute fast aerial (jump+boost, tilt, jump) consistently.",
            category="movement", difficulty=0.6, duration_seconds=60,
            steps=["jump", "boost+tilt_back", "second_jump", "fly_to_ball"] * 10,
            success_metric="10_clean_fast_aerials",
        ),
        Drill(
            drill_id="rl_dribbling", game_id="rocket_league",
            name="Ball Carry Dribble",
            description="Carry ball on car roof across field. Flick at end.",
            category="movement", difficulty=0.8, duration_seconds=60,
            success_metric="full_field_carry + flick",
        ),
    ],
    # --- Fortnite ---
    "fortnite": [
        Drill(
            drill_id="fn_90s", game_id="fortnite",
            name="90s Building",
            description="Build 10 layers of 90s as fast as possible.",
            category="movement", difficulty=0.6, duration_seconds=30,
            steps=["wall", "ramp", "turn_90", "wall", "ramp"] * 10,
            success_metric="10_layers_in_<15s",
        ),
        Drill(
            drill_id="fn_edit_course", game_id="fortnite",
            name="Edit Course Speed",
            description="Complete an edit course. Wall edits, floor edits, cone edits.",
            category="movement", difficulty=0.7, duration_seconds=60,
            success_metric="course_time < 45s",
        ),
        Drill(
            drill_id="fn_box_fight", game_id="fortnite",
            name="Box Fight Mechanics",
            description="Edit-peek-shoot-reset cycle. 10 reps.",
            category="aim", difficulty=0.8, duration_seconds=60,
            success_metric="10_clean_edit_peeks",
        ),
    ],
    # --- Minecraft ---
    "minecraft": [
        Drill(
            drill_id="mc_speed_bridge", game_id="minecraft",
            name="Speed Bridging",
            description="Speed bridge 20 blocks without falling.",
            category="movement", difficulty=0.5, duration_seconds=30,
            success_metric="20_blocks_no_fall",
        ),
        Drill(
            drill_id="mc_pvp_combo", game_id="minecraft",
            name="PvP Combo (1.8 Style)",
            description="Land 5-hit combo on practice dummy with strafing.",
            category="aim", difficulty=0.6, duration_seconds=30,
            success_metric="5_hit_combo",
        ),
    ],
    # --- Hearthstone ---
    "hearthstone": [
        Drill(
            drill_id="hs_lethal_puzzle", game_id="hearthstone",
            name="Lethal Puzzle",
            description="Find lethal from given board state. 10 puzzles.",
            category="game_sense", difficulty=0.6, duration_seconds=300,
            success_metric="8/10_lethals_found",
        ),
        Drill(
            drill_id="hs_mulligan", game_id="hearthstone",
            name="Mulligan Decision",
            description="Choose correct mulligan based on matchup for 20 scenarios.",
            category="economy", difficulty=0.5, duration_seconds=120,
            success_metric="correct_mulligan > 80%",
        ),
    ],
    # --- EA FC ---
    "ea_fc": [
        Drill(
            drill_id="fc_skill_moves", game_id="ea_fc",
            name="Skill Move Chains",
            description="Chain 5-star skill moves past 3 defenders.",
            category="movement", difficulty=0.7, duration_seconds=60,
            success_metric="beat_3_defenders_with_skills",
        ),
        Drill(
            drill_id="fc_timed_finishing", game_id="ea_fc",
            name="Timed Finishing",
            description="Score 10 goals with green-timed shots.",
            category="aim", difficulty=0.6, duration_seconds=120,
            success_metric="10_green_timed_goals",
        ),
    ],
    # --- WoW ---
    "wow": [
        Drill(
            drill_id="wow_rotation", game_id="wow",
            name="DPS Rotation Practice",
            description="Execute optimal rotation on target dummy for 3 minutes.",
            category="aim", difficulty=0.5, duration_seconds=180,
            success_metric="dps_within_5%_of_sim",
        ),
        Drill(
            drill_id="wow_interrupt", game_id="wow",
            name="Interrupt Timing",
            description="Interrupt casts within reaction window. 20 casts.",
            category="game_sense", difficulty=0.6, duration_seconds=60,
            success_metric="interrupt_rate > 90%",
        ),
    ],
    # --- Elden Ring / Soulslike ---
    "elden_ring": [
        Drill(
            drill_id="er_dodge_timing", game_id="elden_ring",
            name="Dodge i-Frame Timing",
            description="Roll through 20 boss attacks using i-frames.",
            category="movement", difficulty=0.7, duration_seconds=120,
            success_metric="dodge_rate > 85%",
        ),
        Drill(
            drill_id="er_parry", game_id="elden_ring",
            name="Parry Timing",
            description="Successfully parry 10 enemy attacks in a row.",
            category="game_sense", difficulty=0.8, duration_seconds=60,
            success_metric="10_consecutive_parries",
        ),
    ],
    # --- PUBG ---
    "pubg": [
        Drill(
            drill_id="pubg_spray_m416", game_id="pubg",
            name="M416 Spray Transfer",
            description="Spray two targets at 50m. Transfer between them.",
            category="aim", difficulty=0.7, duration_seconds=60,
            success_metric="both_targets_hit_in_one_mag",
        ),
    ],
    # --- Rust ---
    "rust": [
        Drill(
            drill_id="rust_ak_spray", game_id="rust",
            name="Rust AK Spray Pattern",
            description="Master the S-pattern. Hit wall target at 75m full spray.",
            category="aim", difficulty=0.9, duration_seconds=120,
            success_metric="grouping < 100px at 75m",
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
