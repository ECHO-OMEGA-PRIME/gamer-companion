"""Cognition Engine — 3-layer decision hierarchy.

Processes perception data and produces action plans:
- Layer 1: REFLEX — Instant reactions (enemy on crosshair → shoot)
- Layer 2: TACTICAL — Short-term plans (push site, hold angle, rotate)
- Layer 3: STRATEGIC — Long-term strategy (economy, map control, team role)

Each layer can override lower layers. Reflex is fastest, Strategic is smartest.
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from enum import Enum
from loguru import logger


class CognitionLayer(Enum):
    REFLEX = "reflex"       # <16ms — hardcoded reactions
    TACTICAL = "tactical"   # <100ms — situational decisions
    STRATEGIC = "strategic"  # <500ms — round/game-level planning


@dataclass
class Decision:
    """A decision produced by the cognition engine."""
    action_type: str          # "shoot", "move", "use_ability", "buy", "rotate", "hold"
    params: Dict = field(default_factory=dict)
    layer: CognitionLayer = CognitionLayer.TACTICAL
    priority: float = 0.5    # 0-1, higher = more urgent
    confidence: float = 0.5  # 0-1, how sure we are
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    expires_ms: float = 500  # Decision expires after this many ms


@dataclass
class ReflexRule:
    """A hardcoded reflex rule — bypasses thinking."""
    name: str
    condition: Callable  # (perception_frame) → bool
    action_type: str
    params: Dict = field(default_factory=dict)
    priority: float = 0.9
    cooldown_ms: float = 100
    last_fired: float = 0


@dataclass
class TacticalState:
    """Current tactical situation assessment."""
    is_in_combat: bool = False
    enemies_visible: int = 0
    allies_alive: int = 5
    enemies_alive: int = 5
    our_advantage: float = 0.0   # -1 (losing) to +1 (winning)
    time_pressure: float = 0.0   # 0 (no pressure) to 1 (bomb planted/timer low)
    economy_state: str = "full"  # "eco", "force", "full"
    map_control: float = 0.5    # 0 (none) to 1 (full)


class CognitionEngine:
    """3-layer decision engine: Reflex → Tactical → Strategic.

    The engine evaluates all layers each tick and returns a prioritized
    list of decisions. Higher layers can suppress lower-layer decisions
    when they conflict.

    Reflex rules are checked first and produce instant decisions.
    Tactical analysis runs every tick.
    Strategic planning runs periodically (every few seconds).
    """

    def __init__(self):
        self._reflex_rules: List[ReflexRule] = []
        self._tactical_evaluators: List[Callable] = []
        self._strategic_evaluators: List[Callable] = []
        self._tactical_state = TacticalState()
        self._current_strategy: str = "default"
        self._decision_history: List[Decision] = []
        self._max_history = 200
        self._strategic_interval_s = 3.0
        self._last_strategic_time = 0

    def add_reflex(self, rule: ReflexRule):
        """Add a reflex rule (instant reaction)."""
        self._reflex_rules.append(rule)
        self._reflex_rules.sort(key=lambda r: r.priority, reverse=True)

    def add_tactical_evaluator(self, fn: Callable):
        """Add a tactical evaluation function.

        fn(tactical_state, perception_frame) → List[Decision]
        """
        self._tactical_evaluators.append(fn)

    def add_strategic_evaluator(self, fn: Callable):
        """Add a strategic planning function.

        fn(tactical_state, decision_history) → List[Decision]
        """
        self._strategic_evaluators.append(fn)

    def think(self, perception_frame) -> List[Decision]:
        """Process perception and produce decisions.

        Runs all 3 layers and returns merged, prioritized decisions.
        """
        decisions: List[Decision] = []
        now = time.time()

        # Layer 1: REFLEX — check all rules
        for rule in self._reflex_rules:
            if now - rule.last_fired < rule.cooldown_ms / 1000:
                continue
            try:
                if rule.condition(perception_frame):
                    decisions.append(Decision(
                        action_type=rule.action_type,
                        params=dict(rule.params),
                        layer=CognitionLayer.REFLEX,
                        priority=rule.priority,
                        confidence=0.95,
                        reasoning=f"reflex:{rule.name}",
                    ))
                    rule.last_fired = now
            except Exception:
                pass

        # Update tactical state from perception
        self._update_tactical_state(perception_frame)

        # Layer 2: TACTICAL — run all evaluators
        for evaluator in self._tactical_evaluators:
            try:
                tac_decisions = evaluator(self._tactical_state, perception_frame)
                if tac_decisions:
                    decisions.extend(tac_decisions)
            except Exception:
                pass

        # Layer 3: STRATEGIC — run periodically
        if now - self._last_strategic_time >= self._strategic_interval_s:
            self._last_strategic_time = now
            for evaluator in self._strategic_evaluators:
                try:
                    strat_decisions = evaluator(
                        self._tactical_state, self._decision_history[-50:]
                    )
                    if strat_decisions:
                        decisions.extend(strat_decisions)
                except Exception:
                    pass

        # Deduplicate and resolve conflicts
        decisions = self._resolve_conflicts(decisions)

        # Record history
        for d in decisions:
            self._decision_history.append(d)
        if len(self._decision_history) > self._max_history:
            self._decision_history = self._decision_history[-self._max_history:]

        return decisions

    def _update_tactical_state(self, frame):
        """Update tactical state from perception data."""
        if frame is None:
            return

        if hasattr(frame, 'detections'):
            enemies = [d for d in frame.detections if d.get("label") == "enemy"]
            self._tactical_state.enemies_visible = len(enemies)
            self._tactical_state.is_in_combat = len(enemies) > 0

        if hasattr(frame, 'ocr'):
            if "timer" in frame.ocr:
                try:
                    timer_val = float(frame.ocr["timer"])
                    self._tactical_state.time_pressure = max(
                        0, min(1, 1.0 - timer_val / 120)
                    )
                except (ValueError, TypeError):
                    pass

    def _resolve_conflicts(self, decisions: List[Decision]) -> List[Decision]:
        """Resolve conflicting decisions from different layers.

        Higher layer wins when actions conflict. Same-layer decisions
        are sorted by priority.
        """
        if not decisions:
            return []

        # Group by action_type
        by_type: Dict[str, List[Decision]] = {}
        for d in decisions:
            by_type.setdefault(d.action_type, []).append(d)

        resolved = []
        layer_rank = {
            CognitionLayer.STRATEGIC: 3,
            CognitionLayer.TACTICAL: 2,
            CognitionLayer.REFLEX: 1,
        }

        for action_type, group in by_type.items():
            if len(group) == 1:
                resolved.append(group[0])
            else:
                # Highest layer wins; within same layer, highest priority
                best = max(
                    group,
                    key=lambda d: (layer_rank[d.layer], d.priority),
                )
                resolved.append(best)

        # Sort by priority descending
        resolved.sort(key=lambda d: d.priority, reverse=True)
        return resolved

    def update_tactical(self, **kwargs):
        """Manually update tactical state fields."""
        for key, val in kwargs.items():
            if hasattr(self._tactical_state, key):
                setattr(self._tactical_state, key, val)

    @property
    def tactical_state(self) -> TacticalState:
        return self._tactical_state

    @property
    def strategy(self) -> str:
        return self._current_strategy

    @strategy.setter
    def strategy(self, value: str):
        self._current_strategy = value

    def get_stats(self) -> dict:
        layer_counts = {l.value: 0 for l in CognitionLayer}
        for d in self._decision_history[-50:]:
            layer_counts[d.layer.value] += 1
        return {
            "reflex_rules": len(self._reflex_rules),
            "tactical_evaluators": len(self._tactical_evaluators),
            "strategic_evaluators": len(self._strategic_evaluators),
            "decision_history": len(self._decision_history),
            "layer_distribution": layer_counts,
            "current_strategy": self._current_strategy,
            "in_combat": self._tactical_state.is_in_combat,
            "enemies_visible": self._tactical_state.enemies_visible,
        }
