"""Risk Assessor — Calculate risk/reward for potential actions.

Quantifies how risky an action is given the current game state,
allowing the AI to make informed decisions about aggression level.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class RiskAssessment:
    """Risk/reward analysis for an action."""
    action: str
    risk_score: float         # 0.0 (safe) to 1.0 (very risky)
    reward_score: float       # 0.0 (low value) to 1.0 (high value)
    ev: float                 # Expected value: reward - risk
    factors: Dict[str, float]  # Individual risk factors
    recommendation: str       # "go", "caution", "abort"


@dataclass
class GameContext:
    """Simplified game context for risk assessment."""
    health: int = 100
    armor: int = 0
    weapon_value: int = 2700  # Money value of current weapon
    teammates_alive: int = 4
    enemies_alive: int = 5
    round_time_left: float = 115.0
    score_diff: int = 0       # Positive = winning
    economy: int = 4000
    has_info: bool = False     # Do we know enemy positions?
    bomb_planted: bool = False
    is_ct: bool = True


class RiskAssessor:
    """Assess risk/reward for combat actions.

    Risk factors:
    - Health risk: low HP = high risk
    - Economy risk: expensive weapon = high risk to lose
    - Numbers: outnumbered = high risk
    - Info deficit: unknown enemy positions = high risk
    - Time pressure: low time = forced high risk

    Reward factors:
    - Kill value: enemy with AWP = high reward
    - Round impact: closing out a round = high reward
    - Economy damage: forcing enemy eco = medium reward
    - Map control: taking key position = medium reward
    """

    def __init__(self, risk_tolerance: float = 0.5):
        self.risk_tolerance = risk_tolerance  # 0 = very cautious, 1 = yolo

    def assess(self, action: str, ctx: GameContext) -> RiskAssessment:
        """Assess risk/reward for an action in context."""
        risk_factors = {}
        reward_factors = {}

        # Health risk
        health_risk = 1.0 - (ctx.health / 100)
        if ctx.armor > 0:
            health_risk *= 0.7
        risk_factors["health"] = round(health_risk, 3)

        # Economy risk (losing expensive weapon)
        econ_risk = min(1.0, ctx.weapon_value / 5000)
        if ctx.economy > 6000:
            econ_risk *= 0.5  # Can rebuy
        risk_factors["economy"] = round(econ_risk, 3)

        # Numbers risk
        if ctx.enemies_alive > 0:
            numbers_ratio = (ctx.teammates_alive + 1) / ctx.enemies_alive
            numbers_risk = max(0.0, 1.0 - numbers_ratio / 2)
        else:
            numbers_risk = 0.0
        risk_factors["numbers"] = round(numbers_risk, 3)

        # Info risk
        info_risk = 0.0 if ctx.has_info else 0.4
        risk_factors["information"] = round(info_risk, 3)

        # Action-specific risk modifiers
        action_risk = self._action_risk(action)
        risk_factors["action_inherent"] = round(action_risk, 3)

        # Time pressure (low time = must act)
        if ctx.round_time_left < 15:
            risk_factors["time_forced"] = 0.3  # Force action regardless
        elif ctx.round_time_left < 30:
            risk_factors["time_pressure"] = 0.15

        # Compute total risk
        total_risk = min(1.0, sum(risk_factors.values()) / len(risk_factors))

        # Reward calculation
        if ctx.enemies_alive <= 2:
            reward_factors["round_close"] = 0.8
        if ctx.bomb_planted:
            reward_factors["bomb_context"] = 0.6
        if ctx.enemies_alive > 0:
            reward_factors["kill_value"] = min(1.0, 1.0 / ctx.enemies_alive)
        else:
            reward_factors["free_round"] = 1.0

        # Action-specific reward
        reward_factors["action_value"] = self._action_reward(action)

        total_reward = min(1.0, sum(reward_factors.values()) / max(1, len(reward_factors)))

        # Expected value adjusted by risk tolerance
        ev = total_reward - total_risk * (1.0 - self.risk_tolerance)

        # Recommendation
        if ev > 0.3:
            rec = "go"
        elif ev > 0.0:
            rec = "caution"
        else:
            rec = "abort"

        return RiskAssessment(
            action=action,
            risk_score=round(total_risk, 3),
            reward_score=round(total_reward, 3),
            ev=round(ev, 3),
            factors={**risk_factors, **{f"reward_{k}": v for k, v in reward_factors.items()}},
            recommendation=rec,
        )

    def _action_risk(self, action: str) -> float:
        """Inherent risk of an action type."""
        risks = {
            "engage": 0.6,
            "peek": 0.4,
            "flash_peek": 0.3,
            "hold": 0.1,
            "retreat": 0.1,
            "reposition": 0.3,
            "rotate": 0.2,
            "save": 0.05,
            "trade": 0.5,
            "bait": 0.7,
            "rush": 0.8,
            "wide_peek": 0.7,
            "jiggle_peek": 0.2,
        }
        return risks.get(action, 0.5)

    def _action_reward(self, action: str) -> float:
        """Potential reward of an action type."""
        rewards = {
            "engage": 0.7,
            "peek": 0.4,
            "flash_peek": 0.5,
            "hold": 0.3,
            "retreat": 0.1,
            "reposition": 0.4,
            "rotate": 0.3,
            "save": 0.2,
            "trade": 0.6,
            "bait": 0.5,
            "rush": 0.6,
            "wide_peek": 0.5,
            "jiggle_peek": 0.3,
        }
        return rewards.get(action, 0.4)

    def compare_actions(
        self, actions: List[str], ctx: GameContext,
    ) -> List[RiskAssessment]:
        """Compare risk/reward across multiple actions."""
        assessments = [self.assess(action, ctx) for action in actions]
        assessments.sort(key=lambda a: a.ev, reverse=True)
        return assessments

    def get_best_action(
        self, actions: List[str], ctx: GameContext,
    ) -> RiskAssessment:
        """Get the action with highest expected value."""
        ranked = self.compare_actions(actions, ctx)
        return ranked[0]
