"""Communication AI — Generate in-game callouts and team communication.

Produces context-aware callouts for team coordination:
- Enemy position callouts ("2 A long")
- Utility requests ("flash me into site")
- Strategy calls ("let's go B")
- Economy calls ("full save this round")
- Encouragement/morale ("nice shot", "we got this")
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class Callout:
    """A generated callout message."""
    text: str
    priority: int             # 0=critical, 1=high, 2=medium, 3=low
    category: str             # "enemy", "utility", "strategy", "economy", "morale"
    timestamp: float = field(default_factory=time.time)
    cooldown_key: str = ""    # Prevent spam of same callout


class CommunicationAI:
    """Generate contextual in-game callouts.

    Respects:
    - Callout frequency (configurable)
    - Cooldowns (don't spam the same info)
    - Priority (critical info first)
    - Toxicity level (for realism dial)
    """

    def __init__(
        self,
        callout_frequency: float = 0.5,  # 0=silent, 1=constant
        toxicity: float = 0.0,           # 0=positive only, 1=toxic
    ):
        self.callout_frequency = callout_frequency
        self.toxicity = toxicity
        self._cooldowns: Dict[str, float] = {}
        self._cooldown_duration = 5.0  # seconds
        self._queue: List[Callout] = []
        self._history: List[Callout] = []

    def enemy_spotted(
        self,
        count: int,
        location: str,
        weapons: List[str] = None,
    ) -> Optional[Callout]:
        """Generate enemy position callout."""
        if not self._should_callout("enemy"):
            return None

        weapon_info = ""
        if weapons:
            awps = sum(1 for w in weapons if w.lower() == "awp")
            if awps:
                weapon_info = f" ({awps} AWP{'s' if awps > 1 else ''})"

        text = f"{count} {location}{weapon_info}"
        callout = Callout(
            text=text,
            priority=0 if count >= 3 else 1,
            category="enemy",
            cooldown_key=f"enemy_{location}",
        )

        return self._emit(callout)

    def request_utility(self, utility_type: str, location: str) -> Optional[Callout]:
        """Request utility from teammates."""
        if not self._should_callout("utility"):
            return None

        templates = {
            "flash": [f"flash me {location}", f"flash {location} please", f"need flash {location}"],
            "smoke": [f"smoke {location}", f"need smoke {location}"],
            "molly": [f"molly {location}", f"burn {location}"],
        }

        options = templates.get(utility_type, [f"need {utility_type} {location}"])
        text = random.choice(options)

        callout = Callout(
            text=text,
            priority=2,
            category="utility",
            cooldown_key=f"util_{utility_type}_{location}",
        )

        return self._emit(callout)

    def strategy_call(self, strategy: str) -> Optional[Callout]:
        """Call a strategy."""
        if not self._should_callout("strategy"):
            return None

        templates = {
            "rush_a": ["let's go A fast", "rush A", "A execute"],
            "rush_b": ["B rush", "go B", "B execute"],
            "default": ["play default", "spread out", "let's default"],
            "eco": ["eco round", "save", "full save"],
            "force": ["force buy", "let's force", "force up"],
            "rotate": ["rotate", "let's rotate", "switch sites"],
        }

        options = templates.get(strategy, [strategy])
        text = random.choice(options)

        callout = Callout(
            text=text,
            priority=1,
            category="strategy",
            cooldown_key=f"strat_{strategy}",
        )

        return self._emit(callout)

    def economy_call(self, team_money: int, recommendation: str) -> Optional[Callout]:
        """Economy-related callout."""
        if not self._should_callout("economy"):
            return None

        if recommendation == "full_save":
            text = f"save this round (${team_money})"
        elif recommendation == "force":
            text = f"force buy — ${team_money}"
        elif recommendation == "full_buy":
            text = "full buy"
        else:
            text = recommendation

        callout = Callout(
            text=text,
            priority=2,
            category="economy",
            cooldown_key="econ_call",
        )

        return self._emit(callout)

    def morale(self, context: str = "neutral") -> Optional[Callout]:
        """Morale/encouragement callout."""
        if not self._should_callout("morale"):
            return None

        if self.toxicity > 0.7:
            positive = ["nt", "whatever", "just play"]
            negative = ["come on guys", "are you serious", "wake up"]
        elif self.toxicity > 0.3:
            positive = ["nice", "good job", "let's go"]
            negative = ["unlucky", "np we got next", "focus up"]
        else:
            positive = ["nice shot!", "great round!", "let's keep it up!", "gg well played"]
            negative = ["no worries, we got this", "nice try", "next round is ours", "stay positive"]

        if context in ("kill", "round_win", "clutch"):
            text = random.choice(positive)
        else:
            text = random.choice(negative)

        callout = Callout(
            text=text,
            priority=3,
            category="morale",
            cooldown_key="morale",
        )

        return self._emit(callout)

    def _should_callout(self, category: str) -> bool:
        """Check if we should generate a callout based on frequency."""
        # Critical callouts always go through
        if category == "enemy":
            return True

        # Random check based on frequency setting
        return random.random() < self.callout_frequency

    def _check_cooldown(self, key: str) -> bool:
        """Check if a callout is on cooldown."""
        if not key:
            return False
        last = self._cooldowns.get(key, 0)
        return (time.time() - last) < self._cooldown_duration

    def _emit(self, callout: Callout) -> Optional[Callout]:
        """Emit a callout if not on cooldown."""
        if self._check_cooldown(callout.cooldown_key):
            return None

        if callout.cooldown_key:
            self._cooldowns[callout.cooldown_key] = time.time()

        self._queue.append(callout)
        self._history.append(callout)

        return callout

    def get_queue(self, clear: bool = True) -> List[Callout]:
        """Get pending callouts sorted by priority."""
        queue = sorted(self._queue, key=lambda c: c.priority)
        if clear:
            self._queue.clear()
        return queue

    def get_stats(self) -> dict:
        cats: Dict[str, int] = {}
        for c in self._history:
            cats[c.category] = cats.get(c.category, 0) + 1
        return {
            "total_callouts": len(self._history),
            "queue_size": len(self._queue),
            "by_category": cats,
            "frequency": self.callout_frequency,
            "toxicity": self.toxicity,
        }
