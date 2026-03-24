"""Test ALL GAMES expansion — registry, FSMs, spray patterns, drills, macros, Steam."""

import pytest
from gamer_companion.game_registry import (
    GameRegistry, Genre, AICapability, GAME_REGISTRY,
)
from gamer_companion.steam_integration import (
    SteamIntegration, _parse_vdf, STEAM_GENRE_MAP,
)
from gamer_companion.state_machine.game_fsm import (
    GameFSM, TacticalFPSFSM, BattleRoyaleFSM, MOBAFSM,
    CoDMultiplayerFSM, HeroShooterFSM, ArenaShooterFSM,
    RTSFSM, FightingGameFSM, RacingFSM, SportsFSM,
    CardGameFSM, AutoBattlerFSM, SurvivalFSM, MMOFSM,
    SoulslikeFSM, ExtractionShooterFSM,
    BoardGameFSM, TurnBasedStrategyFSM, PuzzleGameFSM,
)
from gamer_companion.aim.spray_controller import SPRAY_PATTERNS, SprayController
from gamer_companion.playstyle.pro_mimic import PRO_PROFILES, ProMimic
from gamer_companion.training_ground.game_specific_drills import GAME_DRILLS, GameSpecificDrills
from gamer_companion.input_control.macro_engine import (
    MacroEngine, COD_SLIDE_CANCEL, COD_BUNNY_HOP, COD_YY_CANCEL,
    APEX_SUPERGLIDE, APEX_TAP_STRAFE, VAL_JETT_DASH_SHOOT,
    LOL_FAST_COMBO, FN_90S_BUILD, RL_FAST_AERIAL,
    SC2_INJECT_CYCLE, SF6_HADOUKEN, WOW_ROTATION_BURST, MC_BRIDGE,
)
from gamer_companion.game_intelligence.navigation_engine import PRESET_ROUTES
from gamer_companion.game_intelligence.universal_parser import REGION_HINTS


# =============================================================================
# GAME REGISTRY
# =============================================================================
class TestGameRegistry:
    def test_registry_has_45_plus_games(self):
        assert len(GAME_REGISTRY) >= 45

    def test_registry_class(self):
        reg = GameRegistry()
        assert reg.get("cs2") is not None
        assert reg.get("chess") is not None
        assert reg.get("civilization6") is not None
        assert reg.get("tetris") is not None

    def test_genre_coverage(self):
        reg = GameRegistry()
        genres = set(g.genre for g in reg.list_all())
        assert Genre.TACTICAL_FPS in genres
        assert Genre.BATTLE_ROYALE in genres
        assert Genre.MOBA in genres
        assert Genre.RTS in genres
        assert Genre.FIGHTING in genres
        assert Genre.RACING in genres
        assert Genre.SPORTS in genres
        assert Genre.CARD in genres
        assert Genre.BOARD_GAME in genres
        assert Genre.SURVIVAL in genres
        assert Genre.MMO in genres
        assert Genre.SOULSLIKE in genres
        assert Genre.GRAND_STRATEGY in genres
        assert Genre.PUZZLE in genres

    def test_all_autonomous(self):
        reg = GameRegistry()
        auto_games = reg.list_autonomous()
        assert len(auto_games) >= 40

    def test_esports(self):
        reg = GameRegistry()
        top_tier = reg.list_esports(min_tier=3)
        names = [g.game_id for g in top_tier]
        assert "cs2" in names
        assert "league" in names
        assert "dota2" in names
        assert "valorant" in names
        assert "cod_mp" in names

    def test_search(self):
        reg = GameRegistry()
        results = reg.search("call")
        assert any("cod" in g.game_id for g in results)

    def test_process_detection(self):
        reg = GameRegistry()
        assert reg.detect_by_process("cs2.exe") is not None
        assert reg.detect_by_process("League of Legends.exe") is not None

    def test_board_games_registered(self):
        reg = GameRegistry()
        assert reg.get("chess") is not None
        assert reg.get("checkers") is not None
        assert reg.get("backgammon") is not None

    def test_civ_registered(self):
        reg = GameRegistry()
        assert reg.get("civilization6") is not None
        assert reg.get("civilization7") is not None

    def test_stats(self):
        reg = GameRegistry()
        stats = reg.get_stats()
        assert stats["total_games"] >= 45
        assert stats["autonomous_capable"] >= 40


# =============================================================================
# FSM COVERAGE
# =============================================================================
class TestAllFSMs:
    @pytest.mark.parametrize("fsm_class", [
        TacticalFPSFSM, BattleRoyaleFSM, MOBAFSM,
        CoDMultiplayerFSM, HeroShooterFSM, ArenaShooterFSM,
        RTSFSM, FightingGameFSM, RacingFSM, SportsFSM,
        CardGameFSM, AutoBattlerFSM, SurvivalFSM, MMOFSM,
        SoulslikeFSM, ExtractionShooterFSM,
        BoardGameFSM, TurnBasedStrategyFSM, PuzzleGameFSM,
    ])
    def test_fsm_has_states(self, fsm_class):
        fsm = fsm_class()
        assert fsm.state is not None
        assert len(fsm._states) >= 3

    def test_cod_mp_fsm_flow(self):
        fsm = CoDMultiplayerFSM()
        assert fsm.state == "pre_match"
        assert fsm.transition("spawning")
        assert fsm.transition("patrolling")
        assert fsm.transition("combat")
        assert fsm.transition("killstreak")
        assert fsm.transition("dead")
        assert fsm.transition("spawning")

    def test_board_game_fsm_flow(self):
        fsm = BoardGameFSM()
        assert fsm.state == "waiting"
        assert fsm.transition("thinking")
        assert fsm.transition("moving")
        assert fsm.transition("endgame")
        assert fsm.transition("game_over")

    def test_turn_based_strategy_fsm_flow(self):
        fsm = TurnBasedStrategyFSM()
        assert fsm.state == "turn_start"
        assert fsm.transition("production")
        assert fsm.transition("military")
        assert fsm.transition("turn_end")
        assert fsm.transition("turn_start")

    def test_puzzle_fsm_flow(self):
        fsm = PuzzleGameFSM()
        assert fsm.state == "analyzing"
        assert fsm.transition("planning")
        assert fsm.transition("executing")
        assert fsm.transition("danger")
        assert fsm.transition("game_over")


# =============================================================================
# SPRAY PATTERNS — 30+ weapons
# =============================================================================
class TestSprayExpansion:
    def test_pattern_count(self):
        assert len(SPRAY_PATTERNS) >= 30

    def test_cod_weapons(self):
        assert "cod_m4" in SPRAY_PATTERNS
        assert "cod_ak47" in SPRAY_PATTERNS
        assert "cod_mp5" in SPRAY_PATTERNS
        assert "cod_kastov762" in SPRAY_PATTERNS
        assert "cod_mcw" in SPRAY_PATTERNS
        assert "cod_striker" in SPRAY_PATTERNS

    def test_valorant_weapons(self):
        assert "vandal" in SPRAY_PATTERNS
        assert "phantom" in SPRAY_PATTERNS

    def test_apex_weapons(self):
        assert "r301" in SPRAY_PATTERNS
        assert "flatline" in SPRAY_PATTERNS
        assert "r99" in SPRAY_PATTERNS

    def test_r6_weapons(self):
        assert "r6_f2" in SPRAY_PATTERNS
        assert "r6_smg11" in SPRAY_PATTERNS

    def test_all_patterns_have_offsets(self):
        sc = SprayController()
        for weapon in sc.list_weapons():
            pattern = sc.get_pattern(weapon)
            assert len(pattern.offsets) > 0
            assert pattern.magazine_size > 0
            assert pattern.fire_rate_rpm > 0


# =============================================================================
# PRO PROFILES — 20+ pros
# =============================================================================
class TestProExpansion:
    def test_profile_count(self):
        assert len(PRO_PROFILES) >= 20

    def test_cod_pros(self):
        mimic = ProMimic()
        assert mimic.set_active("shotzzy")
        assert mimic.active.game == "cod_mp"
        assert mimic.set_active("simp")
        assert mimic.set_active("cellium")
        assert mimic.set_active("dashy")

    def test_valorant_pros(self):
        mimic = ProMimic()
        assert mimic.set_active("tenz")
        assert mimic.set_active("aspas")

    def test_apex_pros(self):
        mimic = ProMimic()
        assert mimic.set_active("imperialhal")

    def test_sc2_pros(self):
        mimic = ProMimic()
        assert mimic.set_active("serral")
        assert mimic.active.game == "starcraft2"

    def test_fighting_pros(self):
        mimic = ProMimic()
        assert mimic.set_active("daigo")
        assert mimic.set_active("knee")

    def test_all_pros_have_valid_params(self):
        mimic = ProMimic()
        for name, profile in PRO_PROFILES.items():
            assert 0 <= profile.aggression <= 1
            assert 0 <= profile.consistency <= 1
            assert profile.game != ""


# =============================================================================
# DRILLS — 15+ games
# =============================================================================
class TestDrillExpansion:
    def test_drill_game_count(self):
        assert len(GAME_DRILLS) >= 15

    def test_cod_drills(self):
        assert "cod_mp" in GAME_DRILLS
        assert len(GAME_DRILLS["cod_mp"]) >= 4
        assert "cod_warzone" in GAME_DRILLS
        assert len(GAME_DRILLS["cod_warzone"]) >= 2

    def test_apex_drills(self):
        assert "apex" in GAME_DRILLS
        assert len(GAME_DRILLS["apex"]) >= 3

    def test_fighting_drills(self):
        assert "street_fighter_6" in GAME_DRILLS
        assert "tekken8" in GAME_DRILLS

    def test_strategy_drills(self):
        assert "starcraft2" in GAME_DRILLS
        assert len(GAME_DRILLS["starcraft2"]) >= 3

    def test_fortnite_drills(self):
        assert "fortnite" in GAME_DRILLS
        assert len(GAME_DRILLS["fortnite"]) >= 3

    def test_all_drills_valid(self):
        gsd = GameSpecificDrills()
        for game in gsd.list_games():
            drills = gsd.get_drills(game)
            assert len(drills) > 0
            for drill in drills:
                assert drill.drill_id != ""
                assert drill.name != ""
                assert drill.category in (
                    "aim", "utility", "movement", "economy", "game_sense"
                )


# =============================================================================
# MACROS — all games
# =============================================================================
class TestMacroExpansion:
    def test_cod_macros_exist(self):
        assert COD_SLIDE_CANCEL.game == "cod_mp"
        assert COD_BUNNY_HOP.game == "cod_mp"
        assert COD_YY_CANCEL.game == "cod_mp"
        assert len(COD_SLIDE_CANCEL.steps) >= 4

    def test_apex_macros(self):
        assert APEX_SUPERGLIDE.game == "apex"
        assert APEX_TAP_STRAFE.game == "apex"

    def test_lol_macros(self):
        assert LOL_FAST_COMBO.game == "league"

    def test_fortnite_macros(self):
        assert FN_90S_BUILD.game == "fortnite"

    def test_rocket_league_macros(self):
        assert RL_FAST_AERIAL.game == "rocket_league"

    def test_sc2_macros(self):
        assert SC2_INJECT_CYCLE.game == "starcraft2"

    def test_fighting_macros(self):
        assert SF6_HADOUKEN.game == "street_fighter_6"

    def test_macro_engine_loads(self):
        engine = MacroEngine()
        for macro in [
            COD_SLIDE_CANCEL, COD_BUNNY_HOP, APEX_SUPERGLIDE,
            LOL_FAST_COMBO, FN_90S_BUILD, RL_FAST_AERIAL,
            SC2_INJECT_CYCLE, SF6_HADOUKEN, WOW_ROTATION_BURST,
            MC_BRIDGE,
        ]:
            engine.add_macro(macro)
        assert engine.get_stats()["total_macros"] >= 10

        cod_macros = engine.list_macros(game="cod_mp")
        assert len(cod_macros) >= 2


# =============================================================================
# NAVIGATION ROUTES
# =============================================================================
class TestNavExpansion:
    def test_preset_count(self):
        assert len(PRESET_ROUTES) >= 15

    def test_cod_routes(self):
        assert "cod_loadout_select" in PRESET_ROUTES
        assert "cod_buy_station_loadout" in PRESET_ROUTES
        assert "cod_buy_station_uav" in PRESET_ROUTES

    def test_val_routes(self):
        assert "val_buy_vandal" in PRESET_ROUTES
        assert "val_buy_phantom" in PRESET_ROUTES

    def test_lol_routes(self):
        assert "lol_buy_item_1" in PRESET_ROUTES


# =============================================================================
# REGION HINTS
# =============================================================================
class TestRegionHints:
    def test_genre_count(self):
        assert len(REGION_HINTS) >= 12

    def test_all_genres(self):
        expected = [
            "fps", "moba", "battle_royale", "hero_shooter",
            "arena_shooter", "rts", "fighting", "racing",
            "sports", "card", "survival", "mmo", "soulslike",
            "auto_battler",
        ]
        for genre in expected:
            assert genre in REGION_HINTS, f"Missing region hints for {genre}"


# =============================================================================
# STEAM INTEGRATION
# =============================================================================
class TestSteamIntegration:
    def test_vdf_parser(self):
        vdf = '''
        "AppState"
        {
            "appid"     "730"
            "name"      "Counter-Strike 2"
            "installdir"    "Counter-Strike Global Offensive"
        }
        '''
        result = _parse_vdf(vdf)
        assert "AppState" in result
        assert result["AppState"]["appid"] == "730"
        assert result["AppState"]["name"] == "Counter-Strike 2"

    def test_genre_map(self):
        assert STEAM_GENRE_MAP["fps"] == "tactical_fps"
        assert STEAM_GENRE_MAP["moba"] == "moba"
        assert STEAM_GENRE_MAP["chess"] == "board_game"
        assert STEAM_GENRE_MAP["turn-based strategy"] == "turn_based_strategy"

    def test_steam_integration_init(self):
        steam = SteamIntegration()
        stats = steam.get_stats()
        assert "steam_available" in stats
        assert "games_cached" in stats

    def test_steam_classify(self):
        from gamer_companion.steam_integration import SteamGame
        steam = SteamIntegration()
        game = SteamGame(app_id=730, name="CS2", store_tags=["FPS", "Shooter"])
        genre = steam.classify_genre(game)
        assert genre == "tactical_fps"
