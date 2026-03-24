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
