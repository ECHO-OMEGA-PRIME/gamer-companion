"""Game Registry — Master catalog of all supported games.

Central registry of 30+ games across 10 genres. Each entry defines:
- Game identity (ID, name, genre, process names)
- Capabilities (what the AI can do in this game)
- FSM type to use for phase tracking
- Default configuration values

The GameDetector in foundation/game_profile.py uses process_names and
window_titles for auto-detection. This registry provides the canonical
list of what's supported.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Genre(Enum):
    TACTICAL_FPS = "tactical_fps"
    BATTLE_ROYALE = "battle_royale"
    ARENA_SHOOTER = "arena_shooter"
    HERO_SHOOTER = "hero_shooter"
    MOBA = "moba"
    RTS = "rts"
    TURN_BASED_STRATEGY = "turn_based_strategy"
    FIGHTING = "fighting"
    RACING = "racing"
    SPORTS = "sports"
    CARD = "card"
    BOARD_GAME = "board_game"
    PUZZLE = "puzzle"
    SURVIVAL = "survival"
    MMO = "mmo"
    SOULSLIKE = "soulslike"
    SANDBOX = "sandbox"
    AUTO_BATTLER = "auto_battler"
    RHYTHM = "rhythm"
    GRAND_STRATEGY = "grand_strategy"
    ROGUELIKE = "roguelike"
    PLATFORMER = "platformer"
    SIMULATION = "simulation"


class AICapability(Enum):
    AIM_ASSIST = "aim_assist"
    SPRAY_CONTROL = "spray_control"
    MOVEMENT_TECH = "movement_tech"
    ECONOMY_MANAGEMENT = "economy_management"
    MAP_AWARENESS = "map_awareness"
    ABILITY_USAGE = "ability_usage"
    MACRO_STRATEGY = "macro_strategy"
    MICRO_CONTROL = "micro_control"
    BUILD_ORDER = "build_order"
    COMBO_EXECUTION = "combo_execution"
    RESOURCE_MANAGEMENT = "resource_management"
    TEAM_COORDINATION = "team_coordination"
    OBJECTIVE_TRACKING = "objective_tracking"
    LOOT_MANAGEMENT = "loot_management"
    CARD_STRATEGY = "card_strategy"
    DRAFT_PICK = "draft_pick"
    VEHICLE_CONTROL = "vehicle_control"
    WAVE_MANAGEMENT = "wave_management"
    TRADING = "trading"
    BOSS_PATTERNS = "boss_patterns"
    AUTONOMOUS_PLAY = "autonomous_play"


@dataclass
class GameEntry:
    """A game in the registry."""
    game_id: str
    display_name: str
    genre: Genre
    sub_genre: str = ""
    process_names: List[str] = field(default_factory=list)
    window_titles: List[str] = field(default_factory=list)
    capabilities: List[AICapability] = field(default_factory=list)
    fsm_class: str = ""
    max_players_per_team: int = 5
    round_based: bool = False
    economy_system: bool = False
    respawn: bool = False
    has_spray_patterns: bool = False
    has_pro_profiles: bool = False
    has_drills: bool = False
    has_macros: bool = False
    has_nav_routes: bool = False
    esport_tier: int = 0  # 0=none, 1=minor, 2=major, 3=top-tier
    platforms: List[str] = field(default_factory=lambda: ["pc"])
    notes: str = ""


# Master game registry
GAME_REGISTRY: Dict[str, GameEntry] = {
    # =========================================================================
    # TACTICAL FPS
    # =========================================================================
    "cs2": GameEntry(
        game_id="cs2", display_name="Counter-Strike 2",
        genre=Genre.TACTICAL_FPS, process_names=["cs2.exe"],
        window_titles=["Counter-Strike 2"],
        fsm_class="TacticalFPSFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.MOVEMENT_TECH, AICapability.ECONOMY_MANAGEMENT,
            AICapability.MAP_AWARENESS, AICapability.TEAM_COORDINATION,
            AICapability.OBJECTIVE_TRACKING, AICapability.AUTONOMOUS_PLAY,
        ],
        round_based=True, economy_system=True, has_spray_patterns=True,
        has_pro_profiles=True, has_drills=True, has_macros=True,
        has_nav_routes=True, esport_tier=3,
    ),
    "valorant": GameEntry(
        game_id="valorant", display_name="Valorant",
        genre=Genre.TACTICAL_FPS, process_names=["VALORANT-Win64-Shipping.exe"],
        window_titles=["VALORANT"],
        fsm_class="TacticalFPSFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.ABILITY_USAGE, AICapability.ECONOMY_MANAGEMENT,
            AICapability.MAP_AWARENESS, AICapability.TEAM_COORDINATION,
            AICapability.AUTONOMOUS_PLAY,
        ],
        round_based=True, economy_system=True, has_spray_patterns=True,
        has_pro_profiles=True, has_drills=True, has_macros=True,
        esport_tier=3,
    ),
    "r6siege": GameEntry(
        game_id="r6siege", display_name="Rainbow Six Siege",
        genre=Genre.TACTICAL_FPS, process_names=["RainbowSix.exe"],
        window_titles=["Rainbow Six"],
        fsm_class="TacticalFPSFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.MAP_AWARENESS, AICapability.TEAM_COORDINATION,
            AICapability.ABILITY_USAGE, AICapability.AUTONOMOUS_PLAY,
        ],
        round_based=True, has_spray_patterns=True, has_pro_profiles=True,
        has_drills=True, esport_tier=2,
    ),
    "tarkov": GameEntry(
        game_id="tarkov", display_name="Escape from Tarkov",
        genre=Genre.TACTICAL_FPS, sub_genre="extraction_shooter",
        process_names=["EscapeFromTarkov.exe"],
        window_titles=["EscapeFromTarkov"],
        fsm_class="ExtractionShooterFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.LOOT_MANAGEMENT, AICapability.MAP_AWARENESS,
            AICapability.RESOURCE_MANAGEMENT, AICapability.AUTONOMOUS_PLAY,
        ],
        has_spray_patterns=True, has_drills=True, has_macros=True,
        respawn=False,
    ),

    # =========================================================================
    # CALL OF DUTY (both tactical and BR)
    # =========================================================================
    "cod_mp": GameEntry(
        game_id="cod_mp", display_name="Call of Duty: Multiplayer",
        genre=Genre.TACTICAL_FPS, sub_genre="arcade_fps",
        process_names=["cod.exe", "ModernWarfare.exe", "BlackOps.exe"],
        window_titles=["Call of Duty"],
        fsm_class="CoDMultiplayerFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.MOVEMENT_TECH, AICapability.MAP_AWARENESS,
            AICapability.TEAM_COORDINATION, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=6, round_based=False, respawn=True,
        has_spray_patterns=True, has_pro_profiles=True, has_drills=True,
        has_macros=True, has_nav_routes=True, esport_tier=3,
        platforms=["pc", "playstation", "xbox"],
    ),
    "cod_warzone": GameEntry(
        game_id="cod_warzone", display_name="Call of Duty: Warzone",
        genre=Genre.BATTLE_ROYALE, sub_genre="br_fps",
        process_names=["cod.exe", "ModernWarfare.exe"],
        window_titles=["Call of Duty"],
        fsm_class="BattleRoyaleFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.MOVEMENT_TECH, AICapability.LOOT_MANAGEMENT,
            AICapability.MAP_AWARENESS, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=4, has_spray_patterns=True,
        has_pro_profiles=True, has_drills=True, has_macros=True,
        has_nav_routes=True, esport_tier=2,
        platforms=["pc", "playstation", "xbox"],
    ),

    # =========================================================================
    # BATTLE ROYALE
    # =========================================================================
    "fortnite": GameEntry(
        game_id="fortnite", display_name="Fortnite",
        genre=Genre.BATTLE_ROYALE, process_names=["FortniteClient-Win64-Shipping.exe"],
        window_titles=["Fortnite"],
        fsm_class="BattleRoyaleFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.MOVEMENT_TECH,
            AICapability.LOOT_MANAGEMENT, AICapability.MAP_AWARENESS,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=4, has_pro_profiles=True, has_drills=True,
        has_macros=True, esport_tier=2,
        platforms=["pc", "playstation", "xbox", "switch", "mobile"],
        notes="Building mechanics are core (90-degree walls, ramp rushes, edits)",
    ),
    "apex": GameEntry(
        game_id="apex", display_name="Apex Legends",
        genre=Genre.BATTLE_ROYALE, sub_genre="hero_br",
        process_names=["r5apex.exe"],
        window_titles=["Apex Legends"],
        fsm_class="BattleRoyaleFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.MOVEMENT_TECH, AICapability.ABILITY_USAGE,
            AICapability.LOOT_MANAGEMENT, AICapability.TEAM_COORDINATION,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=3, has_spray_patterns=True,
        has_pro_profiles=True, has_drills=True, has_macros=True,
        esport_tier=2, platforms=["pc", "playstation", "xbox"],
    ),
    "pubg": GameEntry(
        game_id="pubg", display_name="PUBG: Battlegrounds",
        genre=Genre.BATTLE_ROYALE, process_names=["TslGame.exe"],
        window_titles=["PUBG"],
        fsm_class="BattleRoyaleFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.VEHICLE_CONTROL, AICapability.MAP_AWARENESS,
            AICapability.LOOT_MANAGEMENT, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=4, has_spray_patterns=True, has_drills=True,
        esport_tier=2,
    ),

    # =========================================================================
    # HERO / ARENA SHOOTERS
    # =========================================================================
    "overwatch2": GameEntry(
        game_id="overwatch2", display_name="Overwatch 2",
        genre=Genre.HERO_SHOOTER, process_names=["Overwatch.exe"],
        window_titles=["Overwatch"],
        fsm_class="HeroShooterFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.ABILITY_USAGE,
            AICapability.TEAM_COORDINATION, AICapability.OBJECTIVE_TRACKING,
            AICapability.MAP_AWARENESS, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=5, respawn=True, has_pro_profiles=True,
        has_drills=True, has_macros=True, esport_tier=2,
    ),
    "halo_infinite": GameEntry(
        game_id="halo_infinite", display_name="Halo Infinite",
        genre=Genre.ARENA_SHOOTER, process_names=["HaloInfinite.exe"],
        window_titles=["Halo Infinite"],
        fsm_class="ArenaShooterFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.MOVEMENT_TECH,
            AICapability.MAP_AWARENESS, AICapability.OBJECTIVE_TRACKING,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=4, respawn=True, has_drills=True,
        esport_tier=1,
    ),
    "battlefield": GameEntry(
        game_id="battlefield", display_name="Battlefield 2042",
        genre=Genre.ARENA_SHOOTER, sub_genre="large_scale_fps",
        process_names=["BF2042.exe"],
        window_titles=["Battlefield"],
        fsm_class="ArenaShooterFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.VEHICLE_CONTROL, AICapability.OBJECTIVE_TRACKING,
            AICapability.TEAM_COORDINATION, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=64, respawn=True,
        has_spray_patterns=True, has_drills=True,
    ),

    # =========================================================================
    # MOBA
    # =========================================================================
    "league": GameEntry(
        game_id="league", display_name="League of Legends",
        genre=Genre.MOBA, process_names=["League of Legends.exe"],
        window_titles=["League of Legends"],
        fsm_class="MOBAFSM",
        capabilities=[
            AICapability.MICRO_CONTROL, AICapability.MACRO_STRATEGY,
            AICapability.WAVE_MANAGEMENT, AICapability.OBJECTIVE_TRACKING,
            AICapability.TEAM_COORDINATION, AICapability.DRAFT_PICK,
            AICapability.ABILITY_USAGE, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=5, has_pro_profiles=True, has_drills=True,
        has_macros=True, has_nav_routes=True, esport_tier=3,
    ),
    "dota2": GameEntry(
        game_id="dota2", display_name="Dota 2",
        genre=Genre.MOBA, process_names=["dota2.exe"],
        window_titles=["Dota 2"],
        fsm_class="MOBAFSM",
        capabilities=[
            AICapability.MICRO_CONTROL, AICapability.MACRO_STRATEGY,
            AICapability.RESOURCE_MANAGEMENT, AICapability.OBJECTIVE_TRACKING,
            AICapability.TEAM_COORDINATION, AICapability.DRAFT_PICK,
            AICapability.ABILITY_USAGE, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=5, has_pro_profiles=True, has_drills=True,
        esport_tier=3,
    ),
    "smite": GameEntry(
        game_id="smite", display_name="Smite 2",
        genre=Genre.MOBA, process_names=["Smite.exe"],
        window_titles=["SMITE"],
        fsm_class="MOBAFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.ABILITY_USAGE,
            AICapability.OBJECTIVE_TRACKING, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=5, has_drills=True, esport_tier=1,
    ),

    # =========================================================================
    # RTS
    # =========================================================================
    "starcraft2": GameEntry(
        game_id="starcraft2", display_name="StarCraft II",
        genre=Genre.RTS, process_names=["SC2_x64.exe"],
        window_titles=["StarCraft II"],
        fsm_class="RTSFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.MICRO_CONTROL,
            AICapability.BUILD_ORDER, AICapability.RESOURCE_MANAGEMENT,
            AICapability.MAP_AWARENESS, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_pro_profiles=True, has_drills=True,
        has_macros=True, esport_tier=2,
    ),
    "aoe4": GameEntry(
        game_id="aoe4", display_name="Age of Empires IV",
        genre=Genre.RTS, process_names=["RelicCardinal.exe"],
        window_titles=["Age of Empires IV"],
        fsm_class="RTSFSM",
        capabilities=[
            AICapability.BUILD_ORDER, AICapability.RESOURCE_MANAGEMENT,
            AICapability.MACRO_STRATEGY, AICapability.MICRO_CONTROL,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True, has_macros=True,
        esport_tier=1,
    ),

    # =========================================================================
    # FIGHTING GAMES
    # =========================================================================
    "street_fighter_6": GameEntry(
        game_id="street_fighter_6", display_name="Street Fighter 6",
        genre=Genre.FIGHTING, process_names=["StreetFighter6.exe"],
        window_titles=["Street Fighter 6"],
        fsm_class="FightingGameFSM",
        capabilities=[
            AICapability.COMBO_EXECUTION, AICapability.MOVEMENT_TECH,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, round_based=True, has_pro_profiles=True,
        has_drills=True, has_macros=True, esport_tier=2,
    ),
    "tekken8": GameEntry(
        game_id="tekken8", display_name="Tekken 8",
        genre=Genre.FIGHTING, process_names=["Tekken8.exe", "TEKKEN 8.exe"],
        window_titles=["TEKKEN 8"],
        fsm_class="FightingGameFSM",
        capabilities=[
            AICapability.COMBO_EXECUTION, AICapability.MOVEMENT_TECH,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, round_based=True, has_pro_profiles=True,
        has_drills=True, has_macros=True, esport_tier=2,
    ),
    "mortal_kombat": GameEntry(
        game_id="mortal_kombat", display_name="Mortal Kombat 1",
        genre=Genre.FIGHTING, process_names=["MK12.exe"],
        window_titles=["Mortal Kombat"],
        fsm_class="FightingGameFSM",
        capabilities=[
            AICapability.COMBO_EXECUTION, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, round_based=True, has_drills=True,
        has_macros=True, esport_tier=1,
    ),

    # =========================================================================
    # RACING
    # =========================================================================
    "forza_motorsport": GameEntry(
        game_id="forza_motorsport", display_name="Forza Motorsport",
        genre=Genre.RACING, process_names=["ForzaMotorsport.exe"],
        window_titles=["Forza Motorsport"],
        fsm_class="RacingFSM",
        capabilities=[
            AICapability.VEHICLE_CONTROL, AICapability.MAP_AWARENESS,
            AICapability.AUTONOMOUS_PLAY,
        ],
        has_drills=True, esport_tier=1,
        platforms=["pc", "xbox"],
    ),
    "rocket_league": GameEntry(
        game_id="rocket_league", display_name="Rocket League",
        genre=Genre.RACING, sub_genre="car_soccer",
        process_names=["RocketLeague.exe"],
        window_titles=["Rocket League"],
        fsm_class="RacingFSM",
        capabilities=[
            AICapability.VEHICLE_CONTROL, AICapability.MOVEMENT_TECH,
            AICapability.TEAM_COORDINATION, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=3, has_pro_profiles=True, has_drills=True,
        has_macros=True, esport_tier=2,
    ),
    "iracing": GameEntry(
        game_id="iracing", display_name="iRacing",
        genre=Genre.RACING, process_names=["iRacingUI.exe"],
        window_titles=["iRacing"],
        fsm_class="RacingFSM",
        capabilities=[
            AICapability.VEHICLE_CONTROL, AICapability.MAP_AWARENESS,
            AICapability.AUTONOMOUS_PLAY,
        ],
        has_drills=True, esport_tier=1,
    ),

    # =========================================================================
    # SPORTS
    # =========================================================================
    "ea_fc": GameEntry(
        game_id="ea_fc", display_name="EA Sports FC 25",
        genre=Genre.SPORTS, sub_genre="soccer",
        process_names=["FC25.exe", "FIFA.exe"],
        window_titles=["EA SPORTS FC"],
        fsm_class="SportsFSM",
        capabilities=[
            AICapability.MICRO_CONTROL, AICapability.MACRO_STRATEGY,
            AICapability.TEAM_COORDINATION, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=11, has_drills=True, has_macros=True,
        esport_tier=2, platforms=["pc", "playstation", "xbox"],
    ),
    "madden": GameEntry(
        game_id="madden", display_name="Madden NFL 26",
        genre=Genre.SPORTS, sub_genre="football",
        process_names=["Madden26.exe"],
        window_titles=["Madden NFL"],
        fsm_class="SportsFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.MICRO_CONTROL,
            AICapability.AUTONOMOUS_PLAY,
        ],
        has_drills=True, has_macros=True,
        platforms=["pc", "playstation", "xbox"],
    ),
    "nba2k": GameEntry(
        game_id="nba2k", display_name="NBA 2K26",
        genre=Genre.SPORTS, sub_genre="basketball",
        process_names=["NBA2K26.exe"],
        window_titles=["NBA 2K"],
        fsm_class="SportsFSM",
        capabilities=[
            AICapability.MICRO_CONTROL, AICapability.MACRO_STRATEGY,
            AICapability.AUTONOMOUS_PLAY,
        ],
        has_drills=True, has_macros=True,
        platforms=["pc", "playstation", "xbox"],
    ),

    # =========================================================================
    # CARD / AUTO-BATTLER
    # =========================================================================
    "hearthstone": GameEntry(
        game_id="hearthstone", display_name="Hearthstone",
        genre=Genre.CARD, process_names=["Hearthstone.exe"],
        window_titles=["Hearthstone"],
        fsm_class="CardGameFSM",
        capabilities=[
            AICapability.CARD_STRATEGY, AICapability.RESOURCE_MANAGEMENT,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True, esport_tier=1,
    ),
    "mtg_arena": GameEntry(
        game_id="mtg_arena", display_name="Magic: The Gathering Arena",
        genre=Genre.CARD, process_names=["MTGA.exe"],
        window_titles=["MTGA", "Magic: The Gathering Arena"],
        fsm_class="CardGameFSM",
        capabilities=[
            AICapability.CARD_STRATEGY, AICapability.RESOURCE_MANAGEMENT,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
    ),
    "tft": GameEntry(
        game_id="tft", display_name="Teamfight Tactics",
        genre=Genre.AUTO_BATTLER, process_names=["League of Legends.exe"],
        window_titles=["Teamfight Tactics"],
        fsm_class="AutoBattlerFSM",
        capabilities=[
            AICapability.RESOURCE_MANAGEMENT, AICapability.MACRO_STRATEGY,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, economy_system=True, has_drills=True,
    ),

    # =========================================================================
    # SURVIVAL / SANDBOX
    # =========================================================================
    "minecraft": GameEntry(
        game_id="minecraft", display_name="Minecraft",
        genre=Genre.SANDBOX, process_names=["javaw.exe"],
        window_titles=["Minecraft"],
        fsm_class="SurvivalFSM",
        capabilities=[
            AICapability.RESOURCE_MANAGEMENT, AICapability.MAP_AWARENESS,
            AICapability.AUTONOMOUS_PLAY,
        ],
        has_drills=True, has_macros=True,
        platforms=["pc", "playstation", "xbox", "switch", "mobile"],
    ),
    "rust": GameEntry(
        game_id="rust", display_name="Rust",
        genre=Genre.SURVIVAL, process_names=["RustClient.exe"],
        window_titles=["Rust"],
        fsm_class="SurvivalFSM",
        capabilities=[
            AICapability.AIM_ASSIST, AICapability.SPRAY_CONTROL,
            AICapability.RESOURCE_MANAGEMENT, AICapability.MAP_AWARENESS,
            AICapability.AUTONOMOUS_PLAY,
        ],
        has_spray_patterns=True, has_drills=True,
    ),

    # =========================================================================
    # MMO
    # =========================================================================
    "wow": GameEntry(
        game_id="wow", display_name="World of Warcraft",
        genre=Genre.MMO, process_names=["Wow.exe", "WowClassic.exe"],
        window_titles=["World of Warcraft"],
        fsm_class="MMOFSM",
        capabilities=[
            AICapability.ABILITY_USAGE, AICapability.RESOURCE_MANAGEMENT,
            AICapability.MACRO_STRATEGY, AICapability.TEAM_COORDINATION,
            AICapability.AUTONOMOUS_PLAY,
        ],
        has_drills=True, has_macros=True,
    ),
    "ffxiv": GameEntry(
        game_id="ffxiv", display_name="Final Fantasy XIV",
        genre=Genre.MMO, process_names=["ffxiv_dx11.exe"],
        window_titles=["FINAL FANTASY XIV"],
        fsm_class="MMOFSM",
        capabilities=[
            AICapability.ABILITY_USAGE, AICapability.BOSS_PATTERNS,
            AICapability.TEAM_COORDINATION, AICapability.AUTONOMOUS_PLAY,
        ],
        has_drills=True, has_macros=True,
    ),

    # =========================================================================
    # SOULSLIKE / ACTION RPG
    # =========================================================================
    "elden_ring": GameEntry(
        game_id="elden_ring", display_name="Elden Ring",
        genre=Genre.SOULSLIKE, process_names=["eldenring.exe"],
        window_titles=["ELDEN RING"],
        fsm_class="SoulslikeFSM",
        capabilities=[
            AICapability.BOSS_PATTERNS, AICapability.MOVEMENT_TECH,
            AICapability.RESOURCE_MANAGEMENT, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
    ),
    "dark_souls_3": GameEntry(
        game_id="dark_souls_3", display_name="Dark Souls III",
        genre=Genre.SOULSLIKE, process_names=["DarkSoulsIII.exe"],
        window_titles=["DARK SOULS III"],
        fsm_class="SoulslikeFSM",
        capabilities=[
            AICapability.BOSS_PATTERNS, AICapability.MOVEMENT_TECH,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
    ),

    # =========================================================================
    # BOARD GAMES (Chess, Checkers, Backgammon)
    # =========================================================================
    "chess": GameEntry(
        game_id="chess", display_name="Chess",
        genre=Genre.BOARD_GAME, sub_genre="abstract_strategy",
        process_names=["lichess.exe"],
        window_titles=["Chess", "Lichess", "Chess.com"],
        fsm_class="BoardGameFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
        notes="Uses Stockfish-style evaluation. Reads board via screen OCR or API.",
    ),
    "checkers": GameEntry(
        game_id="checkers", display_name="Checkers / Draughts",
        genre=Genre.BOARD_GAME, sub_genre="abstract_strategy",
        window_titles=["Checkers", "Draughts"],
        fsm_class="BoardGameFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
    ),
    "backgammon": GameEntry(
        game_id="backgammon", display_name="Backgammon",
        genre=Genre.BOARD_GAME, sub_genre="dice_strategy",
        window_titles=["Backgammon"],
        fsm_class="BoardGameFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
        notes="Uses GNU Backgammon style probability-based evaluation.",
    ),

    # =========================================================================
    # TURN-BASED STRATEGY / GRAND STRATEGY (Civilization, XCOM, etc.)
    # =========================================================================
    "civilization6": GameEntry(
        game_id="civilization6", display_name="Civilization VI",
        genre=Genre.GRAND_STRATEGY, sub_genre="4x",
        process_names=["CivilizationVI.exe", "CivilizationVI_DX12.exe"],
        window_titles=["Civilization VI", "Sid Meier"],
        fsm_class="TurnBasedStrategyFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.RESOURCE_MANAGEMENT,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
    ),
    "civilization7": GameEntry(
        game_id="civilization7", display_name="Civilization VII",
        genre=Genre.GRAND_STRATEGY, sub_genre="4x",
        process_names=["CivilizationVII.exe"],
        window_titles=["Civilization VII"],
        fsm_class="TurnBasedStrategyFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.RESOURCE_MANAGEMENT,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
    ),
    "xcom2": GameEntry(
        game_id="xcom2", display_name="XCOM 2",
        genre=Genre.TURN_BASED_STRATEGY,
        process_names=["XCOM2.exe"],
        window_titles=["XCOM 2"],
        fsm_class="TurnBasedStrategyFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.MICRO_CONTROL,
            AICapability.RESOURCE_MANAGEMENT, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
    ),
    "total_war_warhammer3": GameEntry(
        game_id="total_war_warhammer3", display_name="Total War: Warhammer III",
        genre=Genre.GRAND_STRATEGY, sub_genre="rts_hybrid",
        process_names=["Warhammer3.exe"],
        window_titles=["Total War: WARHAMMER III"],
        fsm_class="TurnBasedStrategyFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.MICRO_CONTROL,
            AICapability.RESOURCE_MANAGEMENT, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
    "humankind": GameEntry(
        game_id="humankind", display_name="Humankind",
        genre=Genre.GRAND_STRATEGY, sub_genre="4x",
        process_names=["Humankind.exe"],
        window_titles=["HUMANKIND"],
        fsm_class="TurnBasedStrategyFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.RESOURCE_MANAGEMENT,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),

    # =========================================================================
    # PUZZLE GAMES
    # =========================================================================
    "tetris": GameEntry(
        game_id="tetris", display_name="Tetris Effect / Tetris 99",
        genre=Genre.PUZZLE,
        process_names=["TetrisEffect.exe"],
        window_titles=["Tetris"],
        fsm_class="PuzzleGameFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1, has_drills=True,
        notes="T-spin, perfect clears, back-to-back bonus optimization.",
    ),
    "puyo_puyo_tetris": GameEntry(
        game_id="puyo_puyo_tetris", display_name="Puyo Puyo Tetris 2",
        genre=Genre.PUZZLE,
        process_names=["PuyoPuyoTetris2.exe"],
        window_titles=["Puyo Puyo"],
        fsm_class="PuzzleGameFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
    "baba_is_you": GameEntry(
        game_id="baba_is_you", display_name="Baba Is You",
        genre=Genre.PUZZLE,
        process_names=["Baba Is You.exe"],
        window_titles=["Baba Is You"],
        fsm_class="PuzzleGameFSM",
        capabilities=[
            AICapability.MACRO_STRATEGY, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),

    # =========================================================================
    # ROGUELIKE / ACTION ROGUELIKE
    # =========================================================================
    "hades2": GameEntry(
        game_id="hades2", display_name="Hades II",
        genre=Genre.ROGUELIKE,
        process_names=["Hades2.exe"],
        window_titles=["Hades II"],
        fsm_class="SoulslikeFSM",
        capabilities=[
            AICapability.MOVEMENT_TECH, AICapability.ABILITY_USAGE,
            AICapability.RESOURCE_MANAGEMENT, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
    "slay_the_spire": GameEntry(
        game_id="slay_the_spire", display_name="Slay the Spire",
        genre=Genre.ROGUELIKE, sub_genre="deck_builder",
        process_names=["SlayTheSpire.exe"],
        window_titles=["Slay the Spire"],
        fsm_class="CardGameFSM",
        capabilities=[
            AICapability.CARD_STRATEGY, AICapability.RESOURCE_MANAGEMENT,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
    "balatro": GameEntry(
        game_id="balatro", display_name="Balatro",
        genre=Genre.ROGUELIKE, sub_genre="poker_roguelike",
        process_names=["Balatro.exe"],
        window_titles=["Balatro"],
        fsm_class="CardGameFSM",
        capabilities=[
            AICapability.CARD_STRATEGY, AICapability.RESOURCE_MANAGEMENT,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),

    # =========================================================================
    # PLATFORMER
    # =========================================================================
    "celeste": GameEntry(
        game_id="celeste", display_name="Celeste",
        genre=Genre.PLATFORMER,
        process_names=["Celeste.exe"],
        window_titles=["Celeste"],
        fsm_class="PuzzleGameFSM",
        capabilities=[
            AICapability.MOVEMENT_TECH, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
    "hollow_knight": GameEntry(
        game_id="hollow_knight", display_name="Hollow Knight",
        genre=Genre.PLATFORMER, sub_genre="metroidvania",
        process_names=["hollow_knight.exe"],
        window_titles=["Hollow Knight"],
        fsm_class="SoulslikeFSM",
        capabilities=[
            AICapability.MOVEMENT_TECH, AICapability.BOSS_PATTERNS,
            AICapability.MAP_AWARENESS, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),

    # =========================================================================
    # SIMULATION
    # =========================================================================
    "factorio": GameEntry(
        game_id="factorio", display_name="Factorio",
        genre=Genre.SIMULATION, sub_genre="automation",
        process_names=["factorio.exe"],
        window_titles=["Factorio"],
        fsm_class="SurvivalFSM",
        capabilities=[
            AICapability.RESOURCE_MANAGEMENT, AICapability.MACRO_STRATEGY,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
    "satisfactory": GameEntry(
        game_id="satisfactory", display_name="Satisfactory",
        genre=Genre.SIMULATION, sub_genre="automation",
        process_names=["FactoryGame-Win64-Shipping.exe"],
        window_titles=["Satisfactory"],
        fsm_class="SurvivalFSM",
        capabilities=[
            AICapability.RESOURCE_MANAGEMENT, AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
    "cities_skylines2": GameEntry(
        game_id="cities_skylines2", display_name="Cities: Skylines II",
        genre=Genre.SIMULATION, sub_genre="city_builder",
        process_names=["Cities2.exe"],
        window_titles=["Cities: Skylines II"],
        fsm_class="TurnBasedStrategyFSM",
        capabilities=[
            AICapability.RESOURCE_MANAGEMENT, AICapability.MACRO_STRATEGY,
            AICapability.AUTONOMOUS_PLAY,
        ],
        max_players_per_team=1,
    ),
}


class GameRegistry:
    """Query and manage the game registry."""

    def __init__(self):
        self._games = dict(GAME_REGISTRY)

    def get(self, game_id: str) -> Optional[GameEntry]:
        return self._games.get(game_id)

    def list_all(self) -> List[GameEntry]:
        return list(self._games.values())

    def list_by_genre(self, genre: Genre) -> List[GameEntry]:
        return [g for g in self._games.values() if g.genre == genre]

    def list_esports(self, min_tier: int = 1) -> List[GameEntry]:
        return [g for g in self._games.values() if g.esport_tier >= min_tier]

    def list_autonomous(self) -> List[GameEntry]:
        return [
            g for g in self._games.values()
            if AICapability.AUTONOMOUS_PLAY in g.capabilities
        ]

    def search(self, query: str) -> List[GameEntry]:
        q = query.lower()
        return [
            g for g in self._games.values()
            if q in g.game_id or q in g.display_name.lower()
        ]

    def detect_by_process(self, process_name: str) -> Optional[GameEntry]:
        pn = process_name.lower()
        for game in self._games.values():
            if pn in [p.lower() for p in game.process_names]:
                return game
        return None

    def register(self, entry: GameEntry):
        self._games[entry.game_id] = entry

    def get_stats(self) -> dict:
        genres = {}
        for g in self._games.values():
            genres[g.genre.value] = genres.get(g.genre.value, 0) + 1
        return {
            "total_games": len(self._games),
            "genres": genres,
            "esports": len(self.list_esports()),
            "autonomous_capable": len(self.list_autonomous()),
            "with_spray_patterns": sum(1 for g in self._games.values() if g.has_spray_patterns),
            "with_pro_profiles": sum(1 for g in self._games.values() if g.has_pro_profiles),
            "with_drills": sum(1 for g in self._games.values() if g.has_drills),
        }
