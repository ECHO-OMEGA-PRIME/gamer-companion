"""Macro Engine — Multi-step action sequences with timing control.

Defines and executes complex action sequences like:
- Buy sequences: B → 1 → 4 → 2 (buy AK47 in CS2)
- Ability combos: E → Q → W → R (League combo)
- Movement patterns: W+A+Space (strafe jump)

Each step has configurable timing, hold duration, and conditions.
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from loguru import logger


@dataclass
class MacroStep:
    """A single step in a macro sequence."""
    action: str  # "press", "hold", "release", "click", "move", "wait"
    key: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    duration_ms: float = 0  # For "hold" and "wait"
    delay_after_ms: float = 50  # Delay before next step
    jitter_ms: float = 15  # Random timing variance
    button: str = "left"  # For "click"
    condition: Optional[str] = None  # Optional condition key


@dataclass
class Macro:
    """A complete macro sequence."""
    name: str
    description: str = ""
    steps: List[MacroStep] = field(default_factory=list)
    game: str = ""  # Empty = universal
    context: str = ""  # e.g., "buy_phase", "combat"
    cooldown_ms: float = 0
    last_executed: float = 0
    times_executed: int = 0


class MacroEngine:
    """Store, manage, and execute macro sequences.

    Macros are executed through callback functions — the engine itself
    doesn't directly call Win32 APIs. This keeps it testable and
    decoupled from the input layer.
    """

    def __init__(self):
        self._macros: Dict[str, Macro] = {}
        self._on_press: Optional[Callable] = None
        self._on_hold: Optional[Callable] = None
        self._on_release: Optional[Callable] = None
        self._on_click: Optional[Callable] = None
        self._on_move: Optional[Callable] = None
        self._executing = False

    def register_handlers(
        self,
        on_press: Callable = None,
        on_hold: Callable = None,
        on_release: Callable = None,
        on_click: Callable = None,
        on_move: Callable = None,
    ):
        """Register input handler callbacks."""
        self._on_press = on_press
        self._on_hold = on_hold
        self._on_release = on_release
        self._on_click = on_click
        self._on_move = on_move

    def add_macro(self, macro: Macro):
        """Register a macro."""
        self._macros[macro.name] = macro

    def remove_macro(self, name: str):
        self._macros.pop(name, None)

    def get_macro(self, name: str) -> Optional[Macro]:
        return self._macros.get(name)

    def list_macros(
        self, game: str = None, context: str = None,
    ) -> List[Macro]:
        """List macros, optionally filtered by game/context."""
        macros = list(self._macros.values())
        if game:
            macros = [m for m in macros if not m.game or m.game == game]
        if context:
            macros = [m for m in macros if not m.context or m.context == context]
        return macros

    def execute(self, name: str) -> bool:
        """Execute a macro by name.

        Returns True if executed, False if on cooldown or not found.
        """
        macro = self._macros.get(name)
        if not macro:
            logger.warning(f"Macro not found: {name}")
            return False

        now = time.time()
        if macro.cooldown_ms > 0:
            since_last = (now - macro.last_executed) * 1000
            if since_last < macro.cooldown_ms:
                return False

        self._executing = True
        try:
            for step in macro.steps:
                if not self._executing:
                    break
                self._execute_step(step)
        finally:
            self._executing = False

        macro.last_executed = time.time()
        macro.times_executed += 1
        return True

    def cancel(self):
        """Cancel currently executing macro."""
        self._executing = False

    def _execute_step(self, step: MacroStep):
        """Execute a single macro step."""
        # Add jitter to timing
        jitter = random.gauss(0, step.jitter_ms) if step.jitter_ms > 0 else 0

        if step.action == "press" and step.key:
            if self._on_press:
                self._on_press(step.key, step.duration_ms)
        elif step.action == "hold" and step.key:
            if self._on_hold:
                self._on_hold(step.key)
            if step.duration_ms > 0:
                time.sleep((step.duration_ms + jitter) / 1000)
        elif step.action == "release" and step.key:
            if self._on_release:
                self._on_release(step.key)
        elif step.action == "click":
            if self._on_click:
                self._on_click(step.button, step.x, step.y)
        elif step.action == "move" and step.x is not None:
            if self._on_move:
                self._on_move(step.x, step.y or 0)
        elif step.action == "wait":
            wait_ms = step.duration_ms + jitter
            if wait_ms > 0:
                time.sleep(wait_ms / 1000)

        # Post-step delay
        delay = step.delay_after_ms + jitter
        if delay > 0:
            time.sleep(max(0, delay / 1000))

    @property
    def is_executing(self) -> bool:
        return self._executing

    def get_stats(self) -> dict:
        return {
            "total_macros": len(self._macros),
            "executing": self._executing,
            "macros": [
                {
                    "name": m.name,
                    "steps": len(m.steps),
                    "game": m.game,
                    "times_executed": m.times_executed,
                }
                for m in self._macros.values()
            ],
        }


# Pre-built macros for common games
CS2_BUY_AK47 = Macro(
    name="cs2_buy_ak47",
    description="Buy AK-47 in CS2",
    game="cs2",
    context="buy_phase",
    steps=[
        MacroStep(action="press", key="B", delay_after_ms=100),
        MacroStep(action="press", key="1", delay_after_ms=80),
        MacroStep(action="press", key="4", delay_after_ms=80),
        MacroStep(action="press", key="2", delay_after_ms=50),
    ],
    cooldown_ms=1000,
)

CS2_BUY_ARMOR_HELMET = Macro(
    name="cs2_buy_armor_helmet",
    description="Buy kevlar + helmet in CS2",
    game="cs2",
    context="buy_phase",
    steps=[
        MacroStep(action="press", key="B", delay_after_ms=100),
        MacroStep(action="press", key="2", delay_after_ms=80),
        MacroStep(action="press", key="2", delay_after_ms=50),
    ],
    cooldown_ms=1000,
)

STRAFE_JUMP = Macro(
    name="strafe_jump",
    description="W + A + Space strafe jump",
    game="",
    context="movement",
    steps=[
        MacroStep(action="hold", key="W"),
        MacroStep(action="hold", key="A", delay_after_ms=10),
        MacroStep(action="press", key="space", duration_ms=50),
        MacroStep(action="wait", duration_ms=200),
        MacroStep(action="release", key="A"),
        MacroStep(action="release", key="W"),
    ],
)

# --- Call of Duty macros ---
COD_SLIDE_CANCEL = Macro(
    name="cod_slide_cancel",
    description="Tac sprint → slide → crouch cancel (CoD movement tech)",
    game="cod_mp",
    context="movement",
    steps=[
        MacroStep(action="press", key="shift", duration_ms=50, delay_after_ms=80),
        MacroStep(action="hold", key="W"),
        MacroStep(action="press", key="shift", duration_ms=50, delay_after_ms=150),
        MacroStep(action="press", key="ctrl", duration_ms=100, delay_after_ms=80),
        MacroStep(action="press", key="ctrl", duration_ms=50),
        MacroStep(action="release", key="W"),
    ],
    cooldown_ms=300,
)

COD_BUNNY_HOP = Macro(
    name="cod_bunny_hop",
    description="Jump → slide → jump chain for momentum (CoD/Warzone)",
    game="cod_mp",
    context="movement",
    steps=[
        MacroStep(action="hold", key="W"),
        MacroStep(action="press", key="space", duration_ms=50, delay_after_ms=150),
        MacroStep(action="press", key="ctrl", duration_ms=80, delay_after_ms=100),
        MacroStep(action="press", key="space", duration_ms=50, delay_after_ms=150),
        MacroStep(action="press", key="ctrl", duration_ms=80, delay_after_ms=100),
        MacroStep(action="press", key="space", duration_ms=50),
        MacroStep(action="release", key="W"),
    ],
    cooldown_ms=200,
)

COD_YY_CANCEL = Macro(
    name="cod_yy_cancel",
    description="Double weapon swap to cancel animation (YY cancel)",
    game="cod_mp",
    context="combat",
    steps=[
        MacroStep(action="press", key="Y", duration_ms=30, delay_after_ms=60),
        MacroStep(action="press", key="Y", duration_ms=30),
    ],
    cooldown_ms=500,
)

COD_DOLPHIN_DIVE = Macro(
    name="cod_dolphin_dive",
    description="Sprint → prone → dive animation",
    game="cod_mp",
    context="movement",
    steps=[
        MacroStep(action="hold", key="W"),
        MacroStep(action="press", key="shift", duration_ms=50, delay_after_ms=200),
        MacroStep(action="press", key="ctrl", duration_ms=200),
        MacroStep(action="release", key="W"),
    ],
    cooldown_ms=500,
)

# --- Apex Legends macros ---
APEX_SUPERGLIDE = Macro(
    name="apex_superglide",
    description="Mantle → crouch+jump at top for speed boost (Apex superglide)",
    game="apex",
    context="movement",
    steps=[
        MacroStep(action="hold", key="W"),
        MacroStep(action="press", key="space", duration_ms=50, delay_after_ms=350),
        MacroStep(action="press", key="space", duration_ms=20, jitter_ms=5),
        MacroStep(action="press", key="ctrl", duration_ms=20, jitter_ms=5),
        MacroStep(action="release", key="W"),
    ],
    cooldown_ms=1000,
)

APEX_TAP_STRAFE = Macro(
    name="apex_tap_strafe",
    description="Scroll wheel W-taps mid-air for direction change (Apex tap strafe)",
    game="apex",
    context="movement",
    steps=[
        MacroStep(action="press", key="space", duration_ms=50, delay_after_ms=50),
        MacroStep(action="press", key="W", duration_ms=10, delay_after_ms=10),
        MacroStep(action="press", key="W", duration_ms=10, delay_after_ms=10),
        MacroStep(action="press", key="W", duration_ms=10, delay_after_ms=10),
        MacroStep(action="press", key="W", duration_ms=10, delay_after_ms=10),
        MacroStep(action="press", key="W", duration_ms=10),
    ],
    cooldown_ms=500,
)

# --- Valorant macros ---
VAL_JETT_DASH_SHOOT = Macro(
    name="val_jett_dash_shoot",
    description="Jett dash → instant right-click (Valorant)",
    game="valorant",
    context="combat",
    steps=[
        MacroStep(action="press", key="E", duration_ms=50, delay_after_ms=100),
        MacroStep(action="click", button="right", duration_ms=50),
    ],
    cooldown_ms=2000,
)

# --- League of Legends macros ---
LOL_FAST_COMBO = Macro(
    name="lol_fast_combo",
    description="Q → auto → W → auto weave (generic LoL combo)",
    game="league",
    context="combat",
    steps=[
        MacroStep(action="press", key="Q", duration_ms=30, delay_after_ms=100),
        MacroStep(action="click", button="right", delay_after_ms=200),
        MacroStep(action="press", key="W", duration_ms=30, delay_after_ms=100),
        MacroStep(action="click", button="right"),
    ],
    cooldown_ms=500,
)

LOL_WARD_HOP = Macro(
    name="lol_ward_hop",
    description="Place ward → Lee Sin W to ward (LoL ward hop)",
    game="league",
    context="movement",
    steps=[
        MacroStep(action="press", key="4", duration_ms=30),
        MacroStep(action="click", button="left", delay_after_ms=50),
        MacroStep(action="press", key="W", duration_ms=30),
        MacroStep(action="click", button="left"),
    ],
    cooldown_ms=1000,
)

# --- Fortnite macros ---
FN_90S_BUILD = Macro(
    name="fn_90s_build",
    description="Wall → ramp → 90 degree turn (Fortnite building)",
    game="fortnite",
    context="combat",
    steps=[
        MacroStep(action="click", button="left", delay_after_ms=30),
        MacroStep(action="press", key="F1", duration_ms=20, delay_after_ms=30),
        MacroStep(action="click", button="left", delay_after_ms=30),
        MacroStep(action="press", key="F2", duration_ms=20, delay_after_ms=30),
        MacroStep(action="click", button="left", delay_after_ms=30),
        MacroStep(action="press", key="space", duration_ms=50),
        MacroStep(action="move", x=300, y=0),
    ],
    cooldown_ms=100,
)

FN_EDIT_RESET = Macro(
    name="fn_edit_reset",
    description="Open edit → right-click reset → confirm (Fortnite instant reset)",
    game="fortnite",
    context="combat",
    steps=[
        MacroStep(action="press", key="G", duration_ms=20, delay_after_ms=30),
        MacroStep(action="click", button="right", delay_after_ms=20),
        MacroStep(action="press", key="G", duration_ms=20),
    ],
    cooldown_ms=100,
)

# --- Rocket League macros ---
RL_FAST_AERIAL = Macro(
    name="rl_fast_aerial",
    description="Jump+boost → tilt back → second jump (Rocket League fast aerial)",
    game="rocket_league",
    context="movement",
    steps=[
        MacroStep(action="hold", key="space"),
        MacroStep(action="hold", key="shift", delay_after_ms=50),
        MacroStep(action="press", key="down", duration_ms=80, delay_after_ms=50),
        MacroStep(action="press", key="space", duration_ms=50),
        MacroStep(action="release", key="shift"),
    ],
    cooldown_ms=500,
)

RL_HALF_FLIP = Macro(
    name="rl_half_flip",
    description="Backflip cancel → air roll 180 (Rocket League half flip)",
    game="rocket_league",
    context="movement",
    steps=[
        MacroStep(action="press", key="space", duration_ms=50, delay_after_ms=80),
        MacroStep(action="press", key="S", duration_ms=30),
        MacroStep(action="press", key="space", duration_ms=50, delay_after_ms=50),
        MacroStep(action="hold", key="W", duration_ms=100),
        MacroStep(action="press", key="Q", duration_ms=150),
    ],
    cooldown_ms=1000,
)

# --- StarCraft 2 macros ---
SC2_INJECT_CYCLE = Macro(
    name="sc2_inject_cycle",
    description="Select queens → inject larva on all hatches (SC2 Zerg)",
    game="starcraft2",
    context="economy",
    steps=[
        MacroStep(action="press", key="5", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="V", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="backspace", duration_ms=30, delay_after_ms=30),
        MacroStep(action="click", button="left", delay_after_ms=30),
        MacroStep(action="press", key="backspace", duration_ms=30, delay_after_ms=30),
        MacroStep(action="click", button="left", delay_after_ms=30),
        MacroStep(action="press", key="backspace", duration_ms=30, delay_after_ms=30),
        MacroStep(action="click", button="left"),
    ],
    cooldown_ms=25000,
)

# --- Fighting game macros ---
SF6_HADOUKEN = Macro(
    name="sf6_hadouken",
    description="Quarter circle forward + punch (SF6 fireball)",
    game="street_fighter_6",
    context="combat",
    steps=[
        MacroStep(action="press", key="down", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="down+right", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="right", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="U", duration_ms=50),
    ],
    cooldown_ms=300,
)

SF6_SHORYUKEN = Macro(
    name="sf6_shoryuken",
    description="Forward → down → down-forward + punch (SF6 DP)",
    game="street_fighter_6",
    context="combat",
    steps=[
        MacroStep(action="press", key="right", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="down", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="down+right", duration_ms=30, delay_after_ms=30),
        MacroStep(action="press", key="U", duration_ms=50),
    ],
    cooldown_ms=500,
)

# --- WoW macros ---
WOW_ROTATION_BURST = Macro(
    name="wow_rotation_burst",
    description="Pop cooldowns → burst rotation opener (WoW)",
    game="wow",
    context="combat",
    steps=[
        MacroStep(action="press", key="1", duration_ms=30, delay_after_ms=50),
        MacroStep(action="press", key="2", duration_ms=30, delay_after_ms=50),
        MacroStep(action="press", key="shift+1", duration_ms=30, delay_after_ms=1500),
        MacroStep(action="press", key="3", duration_ms=30, delay_after_ms=50),
        MacroStep(action="press", key="4", duration_ms=30, delay_after_ms=50),
        MacroStep(action="press", key="5", duration_ms=30),
    ],
    cooldown_ms=120000,
)

# --- Minecraft macros ---
MC_BRIDGE = Macro(
    name="mc_bridge",
    description="Shift → look down → place block → walk (Minecraft bridging)",
    game="minecraft",
    context="movement",
    steps=[
        MacroStep(action="hold", key="shift"),
        MacroStep(action="hold", key="S"),
        MacroStep(action="click", button="right", delay_after_ms=150),
        MacroStep(action="click", button="right", delay_after_ms=150),
        MacroStep(action="click", button="right", delay_after_ms=150),
        MacroStep(action="click", button="right", delay_after_ms=150),
        MacroStep(action="click", button="right"),
        MacroStep(action="release", key="S"),
        MacroStep(action="release", key="shift"),
    ],
    cooldown_ms=200,
)

# --- EA FC / FIFA macros ---
FC_SKILL_CHAIN = Macro(
    name="fc_skill_chain",
    description="Ball roll → La Croqueta → shot (EA FC skill chain)",
    game="ea_fc",
    context="combat",
    steps=[
        MacroStep(action="hold", key="right_stick_right", duration_ms=200, delay_after_ms=50),
        MacroStep(action="press", key="L1+right_stick_right", duration_ms=150, delay_after_ms=50),
        MacroStep(action="press", key="circle", duration_ms=80),
    ],
    cooldown_ms=500,
)
