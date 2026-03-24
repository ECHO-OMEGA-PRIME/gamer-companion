"""Game FSM — Base finite state machine + genre-specific implementations.

Tracks game phases (buy/execute/post-plant, laning/teamfight/objective, etc.)
and provides phase-specific AI advice. Supports hierarchical FSMs where
game FSM contains round FSM contains encounter FSM.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Set
from loguru import logger


@dataclass
class FSMTransition:
    """A transition between states."""
    from_state: str
    to_state: str
    condition: str  # Human-readable condition description
    priority: int = 0


@dataclass
class FSMState:
    """A state in the finite state machine."""
    name: str
    display_name: str
    advice: str  # Default advice for this state
    allowed_transitions: Set[str] = field(default_factory=set)
    entry_actions: List[str] = field(default_factory=list)
    exit_actions: List[str] = field(default_factory=list)


class GameFSM:
    """Base finite state machine for game phase tracking.

    Supports:
    - State registration with metadata
    - Transition rules with conditions
    - Entry/exit actions per state
    - State duration tracking
    - Transition history for pattern analysis
    - Callbacks on state change
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._states: Dict[str, FSMState] = {}
        self._transitions: List[FSMTransition] = []
        self._current_state: Optional[str] = None
        self._state_start_time: float = 0
        self._history: List[dict] = []
        self._callbacks: List[Callable] = []

    def add_state(
        self, name: str, display_name: str = "",
        advice: str = "", transitions_to: List[str] = None,
    ):
        """Register a state."""
        self._states[name] = FSMState(
            name=name,
            display_name=display_name or name,
            advice=advice,
            allowed_transitions=set(transitions_to or []),
        )
        if self._current_state is None:
            self._current_state = name
            self._state_start_time = time.time()

    def add_transition(
        self, from_state: str, to_state: str,
        condition: str = "", priority: int = 0,
    ):
        """Register a transition rule."""
        self._transitions.append(FSMTransition(
            from_state=from_state, to_state=to_state,
            condition=condition, priority=priority,
        ))
        if from_state in self._states:
            self._states[from_state].allowed_transitions.add(to_state)

    def on_change(self, callback: Callable[[str, str], None]):
        """Register callback for state changes. Args: (old_state, new_state)."""
        self._callbacks.append(callback)

    def transition(self, new_state: str) -> bool:
        """Attempt to transition to a new state."""
        if new_state not in self._states:
            logger.warning(f"FSM '{self.name}': unknown state '{new_state}'")
            return False

        if self._current_state and new_state not in self._states[self._current_state].allowed_transitions:
            if self._states[self._current_state].allowed_transitions:
                logger.debug(
                    f"FSM '{self.name}': transition {self._current_state} -> "
                    f"{new_state} not allowed"
                )
                return False

        old_state = self._current_state
        duration = time.time() - self._state_start_time

        self._history.append({
            "from": old_state,
            "to": new_state,
            "duration": round(duration, 2),
            "timestamp": time.time(),
        })

        self._current_state = new_state
        self._state_start_time = time.time()

        for cb in self._callbacks:
            try:
                cb(old_state, new_state)
            except Exception as e:
                logger.error(f"FSM callback error: {e}")

        return True

    @property
    def state(self) -> Optional[str]:
        return self._current_state

    @property
    def state_info(self) -> Optional[FSMState]:
        if self._current_state:
            return self._states.get(self._current_state)
        return None

    @property
    def state_duration(self) -> float:
        return time.time() - self._state_start_time

    @property
    def advice(self) -> str:
        info = self.state_info
        return info.advice if info else ""

    @property
    def history(self) -> List[dict]:
        return self._history

    def state_distribution(self) -> Dict[str, float]:
        """Get time spent in each state as percentage."""
        totals: Dict[str, float] = {}
        for entry in self._history:
            state = entry["from"]
            if state:
                totals[state] = totals.get(state, 0) + entry["duration"]
        grand_total = sum(totals.values()) or 1
        return {
            k: round(v / grand_total, 3) for k, v in totals.items()
        }


class TacticalFPSFSM(GameFSM):
    """FSM for tactical FPS games (CS2, Valorant, R6 Siege)."""

    def __init__(self):
        super().__init__(name="tactical_fps")
        self.add_state("warmup", "Warmup",
                       "Practice crosshair placement. Check angles.",
                       ["freeze_time"])
        self.add_state("freeze_time", "Freeze Time / Buy Phase",
                       "Buy weapons + utility. Discuss strategy.",
                       ["live"])
        self.add_state("live", "Live Round",
                       "Execute strategy. Check corners. Trade kills.",
                       ["post_plant", "round_end", "clutch"])
        self.add_state("post_plant", "Post-Plant",
                       "Play time. Hold crossfires. Listen for defuse.",
                       ["round_end", "clutch"])
        self.add_state("clutch", "Clutch Situation",
                       "Stay calm. Play for info. Use utility wisely.",
                       ["round_end"])
        self.add_state("round_end", "Round End",
                       "Save weapons if losing. Manage economy.",
                       ["freeze_time", "half_time", "match_end"])
        self.add_state("half_time", "Half Time",
                       "Review first half. Adjust strategy for side switch.",
                       ["freeze_time"])
        self.add_state("match_end", "Match End",
                       "Review performance. Check stats.")


class BattleRoyaleFSM(GameFSM):
    """FSM for battle royale games (Fortnite, Apex, PUBG, Warzone)."""

    def __init__(self):
        super().__init__(name="battle_royale")
        self.add_state("lobby", "Lobby", "Pick landing spot.",
                       ["dropping"])
        self.add_state("dropping", "Dropping",
                       "Watch for other teams. Optimize drop path.",
                       ["early_game"])
        self.add_state("early_game", "Early Game / Looting",
                       "Loot fast. Get loadout. Watch for enemies.",
                       ["mid_game", "combat"])
        self.add_state("combat", "In Combat",
                       "Use cover. Focus one target. Communicate.",
                       ["early_game", "mid_game", "late_game", "eliminated"])
        self.add_state("mid_game", "Mid Game / Rotating",
                       "Move to zone. Position for circle. Loot upgrades.",
                       ["combat", "late_game"])
        self.add_state("late_game", "Late Game / Endgame",
                       "Play zone. Hold high ground. Save utility.",
                       ["combat", "victory", "eliminated"])
        self.add_state("victory", "Victory!")
        self.add_state("eliminated", "Eliminated",
                       "Review what happened. Could you have repositioned?")


class MOBAFSM(GameFSM):
    """FSM for MOBA games (League of Legends, Dota 2)."""

    def __init__(self):
        super().__init__(name="moba")
        self.add_state("loading", "Loading", "Plan lane matchup.",
                       ["laning"])
        self.add_state("laning", "Laning Phase",
                       "Focus CS. Trade when advantaged. Ward river.",
                       ["roaming", "teamfight", "objective"])
        self.add_state("roaming", "Roaming",
                       "Look for ganks. Track enemy jungler. Deep ward.",
                       ["laning", "teamfight", "objective"])
        self.add_state("teamfight", "Teamfight",
                       "Focus priority targets. Peel for carries. Use cooldowns.",
                       ["objective", "laning", "roaming", "base_defense"])
        self.add_state("objective", "Objective Control",
                       "Set up vision. Group up. Zone enemy team.",
                       ["teamfight", "laning", "roaming", "base_siege"])
        self.add_state("base_siege", "Sieging",
                       "Poke safely. Wait for engage. Don't dive.",
                       ["teamfight", "objective", "victory"])
        self.add_state("base_defense", "Defending Base",
                       "Clear waves. Don't get picked. Wait for mistake.",
                       ["teamfight", "defeat"])
        self.add_state("victory", "Victory!")
        self.add_state("defeat", "Defeat",
                       "Review what could have been different.")


class CoDMultiplayerFSM(GameFSM):
    """FSM for Call of Duty multiplayer (respawn modes)."""

    def __init__(self):
        super().__init__(name="cod_mp")
        self.add_state("pre_match", "Pre-Match",
                       "Select loadout. Check map. Plan spawns.",
                       ["spawning"])
        self.add_state("spawning", "Spawning",
                       "Pick lane. Check minimap for teammates.",
                       ["patrolling", "combat"])
        self.add_state("patrolling", "Patrolling",
                       "Control power positions. Watch lanes. Pre-aim angles.",
                       ["combat", "killstreak", "objective"])
        self.add_state("combat", "In Combat",
                       "Strafe. Aim center mass. Use movement tech.",
                       ["patrolling", "dead", "killstreak", "objective"])
        self.add_state("killstreak", "Killstreak Active",
                       "Use streak wisely. Call in at good timing.",
                       ["patrolling", "combat", "dead"])
        self.add_state("objective", "Playing Objective",
                       "Cap flag/hardpoint. Plant bomb. Hold position.",
                       ["combat", "patrolling", "dead"])
        self.add_state("dead", "Dead / Respawning",
                       "Watch killcam. Adjust class if needed.",
                       ["spawning"])
        self.add_state("match_end", "Match End",
                       "Review scoreboard. Check stats.")


class HeroShooterFSM(GameFSM):
    """FSM for hero shooters (Overwatch 2)."""

    def __init__(self):
        super().__init__(name="hero_shooter")
        self.add_state("hero_select", "Hero Select",
                       "Pick hero based on team comp and map.",
                       ["setup"])
        self.add_state("setup", "Setup Phase",
                       "Position with team. Plan ultimate combos.",
                       ["pushing", "defending"])
        self.add_state("pushing", "Pushing / Attacking",
                       "Group up. Use abilities together. Focus targets.",
                       ["teamfight", "regrouping", "overtime"])
        self.add_state("defending", "Defending",
                       "Hold high ground. Use cooldowns wisely. Peel.",
                       ["teamfight", "regrouping", "overtime"])
        self.add_state("teamfight", "Teamfight",
                       "Use ultimate. Focus supports. Track abilities.",
                       ["pushing", "defending", "regrouping", "overtime"])
        self.add_state("regrouping", "Regrouping",
                       "Wait for team. Don't trickle in. Swap hero if needed.",
                       ["pushing", "defending", "hero_select"])
        self.add_state("overtime", "Overtime",
                       "All in. Touch point. Don't die off objective.",
                       ["teamfight", "victory", "defeat"])
        self.add_state("victory", "Victory!")
        self.add_state("defeat", "Defeat",
                       "Review hero matchups. Check damage stats.")


class ArenaShooterFSM(GameFSM):
    """FSM for arena/large-scale shooters (Halo, Battlefield)."""

    def __init__(self):
        super().__init__(name="arena_shooter")
        self.add_state("spawning", "Spawning",
                       "Pick spawn location. Note objectives.",
                       ["roaming"])
        self.add_state("roaming", "Roaming",
                       "Control power weapons/vehicles. Map awareness.",
                       ["combat", "objective", "vehicle"])
        self.add_state("combat", "In Combat",
                       "Use cover. Control engagement range. Strafe.",
                       ["roaming", "dead", "objective"])
        self.add_state("objective", "Playing Objective",
                       "Cap point. Carry flag. Arm/defuse.",
                       ["combat", "roaming", "dead"])
        self.add_state("vehicle", "In Vehicle",
                       "Support infantry. Watch for rockets. Reposition.",
                       ["combat", "roaming", "dead"])
        self.add_state("dead", "Dead / Respawning",
                       "Check scoreboard. Pick next spawn.",
                       ["spawning"])
        self.add_state("match_end", "Match End",
                       "Review stats.")


class RTSFSM(GameFSM):
    """FSM for RTS games (StarCraft II, Age of Empires)."""

    def __init__(self):
        super().__init__(name="rts")
        self.add_state("opening", "Opening / Build Order",
                       "Execute build order. Scout opponent.",
                       ["expanding", "aggression", "defending"])
        self.add_state("expanding", "Expanding Economy",
                       "Take bases. Macro up. Tech up. Wall off.",
                       ["aggression", "defending", "mid_game"])
        self.add_state("aggression", "Attacking / Harassing",
                       "Multi-prong. Deny expansions. Pressure.",
                       ["expanding", "mid_game", "defending"])
        self.add_state("defending", "Defending",
                       "Wall up. Build defenses. Counter-attack.",
                       ["expanding", "mid_game", "aggression"])
        self.add_state("mid_game", "Mid Game / Army",
                       "Composition matters. Control map. Take fights.",
                       ["late_game", "aggression", "defending"])
        self.add_state("late_game", "Late Game",
                       "Max army. Multi-pronged attacks. Decisive fights.",
                       ["victory", "defeat"])
        self.add_state("victory", "Victory!")
        self.add_state("defeat", "Defeat",
                       "Review build order timing. Check supply blocks.")


class FightingGameFSM(GameFSM):
    """FSM for fighting games (SF6, Tekken, MK)."""

    def __init__(self):
        super().__init__(name="fighting")
        self.add_state("neutral", "Neutral",
                       "Control space. Whiff punish. Anti-air.",
                       ["pressure", "defending", "combo", "knockdown"])
        self.add_state("pressure", "Applying Pressure",
                       "Mix up. Frame trap. Throw. Command grab.",
                       ["neutral", "combo", "defending"])
        self.add_state("defending", "Blocking / Defending",
                       "Block string. Punish gaps. Escape pressure.",
                       ["neutral", "pressure", "combo", "knockdown"])
        self.add_state("combo", "Combo Execution",
                       "Confirm into max damage. Optimize route.",
                       ["knockdown", "neutral"])
        self.add_state("knockdown", "Knockdown / Oki",
                       "Set up okizeme. Meaty. Shimmy. Safe jump.",
                       ["pressure", "neutral", "combo"])
        self.add_state("round_end", "Round End",
                       "Adapt strategy. Note opponent patterns.",
                       ["neutral"])
        self.add_state("match_end", "Match End",
                       "Review frame data. Note matchup knowledge.")


class RacingFSM(GameFSM):
    """FSM for racing games (Forza, iRacing, Rocket League)."""

    def __init__(self):
        super().__init__(name="racing")
        self.add_state("pre_race", "Pre-Race",
                       "Choose car/tune. Study track. Check conditions.",
                       ["starting_grid"])
        self.add_state("starting_grid", "Starting Grid",
                       "Launch timing. React to lights. Avoid lap 1 chaos.",
                       ["racing"])
        self.add_state("racing", "Racing",
                       "Hit apexes. Manage tires. Defend/attack position.",
                       ["battling", "pitting", "finish"])
        self.add_state("battling", "Battling for Position",
                       "DRS/slipstream. Late brake. Defend inside line.",
                       ["racing", "pitting", "incident"])
        self.add_state("pitting", "Pit Stop",
                       "Pit window. Tire strategy. Undercut/overcut.",
                       ["racing"])
        self.add_state("incident", "Incident / Off Track",
                       "Rejoin safely. Don't lose more time.",
                       ["racing"])
        self.add_state("finish", "Finish",
                       "Review lap times. Note improvements.")


class SportsFSM(GameFSM):
    """FSM for sports games (EA FC, Madden, NBA 2K)."""

    def __init__(self):
        super().__init__(name="sports")
        self.add_state("pre_game", "Pre-Game",
                       "Set formation/playbook. Check matchups.",
                       ["offense"])
        self.add_state("offense", "On Offense",
                       "Build play. Find openings. Score.",
                       ["defense", "set_piece", "timeout"])
        self.add_state("defense", "On Defense",
                       "Mark up. Pressure ball. Intercept/tackle.",
                       ["offense", "set_piece", "timeout"])
        self.add_state("set_piece", "Set Piece / Special Teams",
                       "Execute set play. Free kick/punt/penalty.",
                       ["offense", "defense"])
        self.add_state("timeout", "Timeout / Break",
                       "Adjust tactics. Sub players. Change formation.",
                       ["offense", "defense"])
        self.add_state("halftime", "Halftime",
                       "Review stats. Adjust strategy for 2nd half.",
                       ["offense", "defense"])
        self.add_state("game_end", "Game End",
                       "Review performance. Check stats.")


class CardGameFSM(GameFSM):
    """FSM for card games (Hearthstone, MTG Arena)."""

    def __init__(self):
        super().__init__(name="card_game")
        self.add_state("mulligan", "Mulligan",
                       "Keep/mulligan based on matchup and curve.",
                       ["early_game"])
        self.add_state("early_game", "Early Game",
                       "Develop board. Play on curve. Set up synergies.",
                       ["mid_game", "combo_turn"])
        self.add_state("mid_game", "Mid Game",
                       "Board control. Value trades. Build advantage.",
                       ["late_game", "combo_turn"])
        self.add_state("late_game", "Late Game",
                       "Top-deck mode. Play for value. Close out game.",
                       ["combo_turn", "victory", "defeat"])
        self.add_state("combo_turn", "Combo Turn",
                       "Execute lethal combo. Calculate damage.",
                       ["mid_game", "late_game", "victory"])
        self.add_state("victory", "Victory!")
        self.add_state("defeat", "Defeat",
                       "Review key turns. Adjust deck.")


class AutoBattlerFSM(GameFSM):
    """FSM for auto-battlers (TFT, Dota Underlords)."""

    def __init__(self):
        super().__init__(name="auto_battler")
        self.add_state("planning", "Planning Phase",
                       "Buy units. Position board. Manage economy.",
                       ["combat", "carousel"])
        self.add_state("combat", "Combat Phase",
                       "Observe fight. Note what won/lost.",
                       ["planning", "carousel"])
        self.add_state("carousel", "Carousel / Armory",
                       "Pick best item. Prioritize for comp.",
                       ["planning"])
        self.add_state("victory", "Victory!")
        self.add_state("eliminated", "Eliminated",
                       "Review economy decisions and comp choices.")


class SurvivalFSM(GameFSM):
    """FSM for survival games (Minecraft, Rust, ARK)."""

    def __init__(self):
        super().__init__(name="survival")
        self.add_state("spawning", "Fresh Spawn",
                       "Gather basic resources. Find shelter location.",
                       ["gathering"])
        self.add_state("gathering", "Gathering Resources",
                       "Farm wood/stone/ore. Manage inventory.",
                       ["building", "exploring", "combat", "crafting"])
        self.add_state("building", "Building Base",
                       "Place structures. Defend entry. Organize storage.",
                       ["gathering", "crafting", "exploring"])
        self.add_state("crafting", "Crafting / Upgrading",
                       "Craft tools/weapons/armor. Smelt ores.",
                       ["gathering", "building", "exploring"])
        self.add_state("exploring", "Exploring",
                       "Discover new areas. Find loot. Map terrain.",
                       ["gathering", "combat", "crafting"])
        self.add_state("combat", "In Combat",
                       "Fight mobs/players. Use terrain. Manage health.",
                       ["gathering", "exploring", "dead"])
        self.add_state("dead", "Dead",
                       "Recover items if possible. Rebuild.",
                       ["spawning"])


class MMOFSM(GameFSM):
    """FSM for MMOs (WoW, FFXIV)."""

    def __init__(self):
        super().__init__(name="mmo")
        self.add_state("idle", "Idle / Town",
                       "Manage inventory. Repair. Check auction house.",
                       ["questing", "dungeon", "pvp", "crafting_gathering"])
        self.add_state("questing", "Questing",
                       "Follow objectives. Kill mobs. Turn in quests.",
                       ["combat", "idle", "exploring"])
        self.add_state("combat", "In Combat",
                       "Execute rotation. Manage cooldowns. Position.",
                       ["questing", "dungeon", "exploring", "dead"])
        self.add_state("dungeon", "Dungeon / Raid",
                       "Follow mechanics. DPS rotation. Heal/tank duty.",
                       ["boss_fight", "idle"])
        self.add_state("boss_fight", "Boss Fight",
                       "Execute mechanics. Dodge AOE. Burn phase.",
                       ["dungeon", "dead"])
        self.add_state("pvp", "PvP Combat",
                       "Burst windows. CC chain. Peel for team.",
                       ["idle", "dead"])
        self.add_state("crafting_gathering", "Crafting / Gathering",
                       "Gather nodes. Craft items. Level professions.",
                       ["idle", "combat"])
        self.add_state("exploring", "Exploring World",
                       "Discover areas. Find treasures. Complete achievements.",
                       ["combat", "idle", "questing"])
        self.add_state("dead", "Dead",
                       "Run back. Check repair costs.",
                       ["idle"])


class SoulslikeFSM(GameFSM):
    """FSM for Soulslike games (Elden Ring, Dark Souls)."""

    def __init__(self):
        super().__init__(name="soulslike")
        self.add_state("exploring", "Exploring",
                       "Check corners. Manage stamina. Find items.",
                       ["combat", "boss_approach", "resting"])
        self.add_state("combat", "Fighting Enemies",
                       "Dodge. Punish openings. Manage stamina.",
                       ["exploring", "dead", "resting"])
        self.add_state("boss_approach", "Boss Arena",
                       "Buff up. Study moveset. Be patient.",
                       ["boss_fight"])
        self.add_state("boss_fight", "Boss Fight",
                       "Learn patterns. Dodge. Punish windows. Don't greed.",
                       ["victory", "dead"])
        self.add_state("resting", "At Bonfire / Grace",
                       "Level up. Equip. Allocate flasks. Plan route.",
                       ["exploring"])
        self.add_state("dead", "You Died",
                       "Recover souls. Adjust strategy. Try again.",
                       ["exploring", "resting"])
        self.add_state("victory", "Boss Defeated!",
                       "Collect reward. Explore new area.",
                       ["exploring", "resting"])


class BoardGameFSM(GameFSM):
    """FSM for board games (Chess, Checkers, Backgammon)."""

    def __init__(self):
        super().__init__(name="board_game")
        self.add_state("waiting", "Waiting for Turn",
                       "Analyze board. Plan next 3-5 moves ahead.",
                       ["thinking"])
        self.add_state("thinking", "Thinking",
                       "Evaluate positions. Calculate threats. Find best move.",
                       ["moving"])
        self.add_state("moving", "Making Move",
                       "Execute chosen move. Watch opponent's reaction.",
                       ["waiting", "endgame", "game_over"])
        self.add_state("endgame", "Endgame",
                       "Simplify. Convert advantage. Avoid stalemate traps.",
                       ["thinking", "moving", "game_over"])
        self.add_state("game_over", "Game Over",
                       "Review key moments. Analyze mistakes.")


class TurnBasedStrategyFSM(GameFSM):
    """FSM for turn-based strategy (Civilization, XCOM, Total War)."""

    def __init__(self):
        super().__init__(name="turn_based_strategy")
        self.add_state("turn_start", "Turn Start",
                       "Check notifications. Assess situation. Plan priorities.",
                       ["production", "research", "military", "diplomacy", "exploration"])
        self.add_state("production", "Production / Building",
                       "Queue builds. Manage cities/bases. Optimize output.",
                       ["research", "military", "diplomacy", "turn_end"])
        self.add_state("research", "Research / Tech",
                       "Choose technology. Plan tech tree path.",
                       ["production", "military", "turn_end"])
        self.add_state("military", "Military / Combat",
                       "Move units. Attack enemies. Defend positions.",
                       ["production", "diplomacy", "turn_end"])
        self.add_state("diplomacy", "Diplomacy",
                       "Trade. Negotiate. Alliance management.",
                       ["production", "military", "turn_end"])
        self.add_state("exploration", "Exploration",
                       "Scout map. Discover resources. Find opponents.",
                       ["production", "military", "turn_end"])
        self.add_state("turn_end", "End Turn",
                       "Verify all units moved. Confirm end turn.",
                       ["turn_start", "victory", "defeat"])
        self.add_state("victory", "Victory!")
        self.add_state("defeat", "Defeat",
                       "Review strategy decisions.")


class PuzzleGameFSM(GameFSM):
    """FSM for puzzle games (Tetris, Baba Is You, etc.)."""

    def __init__(self):
        super().__init__(name="puzzle")
        self.add_state("analyzing", "Analyzing Board",
                       "Study current state. Identify patterns.",
                       ["planning"])
        self.add_state("planning", "Planning Move",
                       "Calculate best placement. Consider future pieces.",
                       ["executing"])
        self.add_state("executing", "Executing Move",
                       "Place piece. Clear lines. Chain combos.",
                       ["analyzing", "danger", "game_over"])
        self.add_state("danger", "Danger / Pressure",
                       "Board is filling. Prioritize survival. Clear garbage.",
                       ["analyzing", "executing", "game_over"])
        self.add_state("game_over", "Game Over",
                       "Review score. Analyze missed opportunities.")


class ExtractionShooterFSM(GameFSM):
    """FSM for extraction shooters (Tarkov, DMZ)."""

    def __init__(self):
        super().__init__(name="extraction_shooter")
        self.add_state("loadout", "Loadout / Stash",
                       "Build kit. Check ammo. Insure gear.",
                       ["deploying"])
        self.add_state("deploying", "Deploying / Loading",
                       "Plan route to loot spots. Know extract locations.",
                       ["looting"])
        self.add_state("looting", "Looting",
                       "Search containers. Check bodies. Manage space.",
                       ["combat", "extracting", "sneaking"])
        self.add_state("sneaking", "Sneaking / Avoiding",
                       "Slow walk. Listen. Avoid hotspots.",
                       ["looting", "combat", "extracting"])
        self.add_state("combat", "In Combat",
                       "Identify threat. Use cover. Flank. Aim for head.",
                       ["looting", "extracting", "dead"])
        self.add_state("extracting", "Extracting",
                       "Move to extract. Watch for campers. Timer check.",
                       ["combat", "extracted"])
        self.add_state("extracted", "Extracted Successfully",
                       "Sell loot. Repair gear. Profit check.")
        self.add_state("dead", "Dead / MIA",
                       "Gear lost. Rebuild kit. Learn from death.",
                       ["loadout"])
