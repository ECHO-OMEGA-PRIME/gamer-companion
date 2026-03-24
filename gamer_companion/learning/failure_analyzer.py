"""Failure Analyzer — Deep analysis of gameplay failures.

Goes beyond simple death tracking to understand WHY failures happen:
- Decision chain analysis (what led to the failure?)
- Counterfactual reasoning (what should have been done?)
- Pattern clustering (are these the same mistake repeated?)
- Root cause taxonomy (aim? positioning? timing? info?)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter
from loguru import logger


class FailureCategory:
    AIM = "aim"                   # Missed shots, wrong target
    POSITIONING = "positioning"    # Bad angle, exposed, no cover
    TIMING = "timing"             # Too early, too late, impatient
    INFORMATION = "information"    # Didn't check minimap, no callout
    ECONOMY = "economy"           # Bad buy, force when should save
    UTILITY = "utility"           # Wasted utility, didn't use utility
    TEAMPLAY = "teamplay"         # No trade, no flash, solo push
    MENTAL = "mental"             # Tilt, panic, rage play
    MECHANICAL = "mechanical"     # Movement error, wrong key, misclick


@dataclass
class FailureEvent:
    """A gameplay failure with context."""
    failure_id: str
    timestamp: float
    category: str             # From FailureCategory
    description: str
    game_state: dict = field(default_factory=dict)
    decision_chain: List[str] = field(default_factory=list)  # Actions leading to failure
    impact: float = 0.5       # 0=minor, 1=game-losing
    round_number: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class FailurePattern:
    """A recurring failure pattern."""
    pattern_id: str
    category: str
    description: str
    occurrences: int
    avg_impact: float
    example_chains: List[List[str]]  # Example decision chains
    suggested_fix: str
    first_seen: float
    last_seen: float


class FailureAnalyzer:
    """Analyze gameplay failures to find root causes and patterns.

    Process:
    1. Collect failure events during gameplay
    2. Categorize by root cause
    3. Cluster similar failures into patterns
    4. Generate actionable improvement suggestions
    5. Track pattern frequency over time (are we improving?)
    """

    def __init__(self):
        self._failures: List[FailureEvent] = []
        self._patterns: Dict[str, FailurePattern] = {}
        self._session_failures: List[FailureEvent] = []

    def record_failure(self, failure: FailureEvent):
        """Record a failure event."""
        self._failures.append(failure)
        self._session_failures.append(failure)

    def analyze(self) -> List[FailurePattern]:
        """Analyze all failures and extract patterns."""
        if not self._failures:
            return []

        # Group by category
        by_category: Dict[str, List[FailureEvent]] = {}
        for f in self._failures:
            if f.category not in by_category:
                by_category[f.category] = []
            by_category[f.category].append(f)

        patterns = []
        for category, failures in by_category.items():
            if len(failures) < 2:
                continue

            # Sub-cluster by decision chain similarity
            chain_groups = self._cluster_by_chains(failures)

            for group_key, group_failures in chain_groups.items():
                if len(group_failures) < 2:
                    continue

                pattern = FailurePattern(
                    pattern_id=f"fp_{category}_{hash(group_key) & 0xFFFF:04x}",
                    category=category,
                    description=self._describe_pattern(category, group_failures),
                    occurrences=len(group_failures),
                    avg_impact=sum(f.impact for f in group_failures) / len(group_failures),
                    example_chains=[f.decision_chain[:5] for f in group_failures[:3]],
                    suggested_fix=self._suggest_fix(category, group_key),
                    first_seen=min(f.timestamp for f in group_failures),
                    last_seen=max(f.timestamp for f in group_failures),
                )

                self._patterns[pattern.pattern_id] = pattern
                patterns.append(pattern)

        return sorted(patterns, key=lambda p: p.avg_impact * p.occurrences, reverse=True)

    def _cluster_by_chains(
        self, failures: List[FailureEvent],
    ) -> Dict[str, List[FailureEvent]]:
        """Cluster failures by similar decision chains."""
        groups: Dict[str, List[FailureEvent]] = {}

        for f in failures:
            # Use first 2 actions as cluster key (simplified)
            key = "|".join(f.decision_chain[:2]) if f.decision_chain else "no_chain"
            if key not in groups:
                groups[key] = []
            groups[key].append(f)

        return groups

    def _describe_pattern(self, category: str, failures: List[FailureEvent]) -> str:
        """Generate a human-readable description of a failure pattern."""
        tag_counts = Counter()
        for f in failures:
            for tag in f.tags:
                tag_counts[tag] += 1

        top_tags = [t for t, _ in tag_counts.most_common(3)]
        tags_str = ", ".join(top_tags) if top_tags else "various situations"

        return f"Recurring {category} failure ({len(failures)}x) in {tags_str}"

    def _suggest_fix(self, category: str, chain_key: str) -> str:
        """Suggest a fix for a failure pattern."""
        fixes = {
            FailureCategory.AIM: "Practice aim in training mode. Focus on crosshair placement before engagements.",
            FailureCategory.POSITIONING: "Review common positions. Never expose to multiple angles simultaneously.",
            FailureCategory.TIMING: "Wait for utility before peeking. Count to 2 before committing.",
            FailureCategory.INFORMATION: "Check minimap every 5 seconds. Call out enemy positions.",
            FailureCategory.ECONOMY: "Follow team economy decisions. Don't force buy solo.",
            FailureCategory.UTILITY: "Pre-plan utility usage. Have a default lineup for each position.",
            FailureCategory.TEAMPLAY: "Stay close enough to trade. Flash for teammates.",
            FailureCategory.MENTAL: "Take a break after 2 consecutive losses. Breathe between rounds.",
            FailureCategory.MECHANICAL: "Warm up before ranked. Practice movement in workshop maps.",
        }
        return fixes.get(category, "Review gameplay footage to identify the root cause.")

    def get_top_issues(self, limit: int = 5) -> List[FailurePattern]:
        """Get most impactful failure patterns."""
        return sorted(
            self._patterns.values(),
            key=lambda p: p.avg_impact * p.occurrences,
            reverse=True,
        )[:limit]

    def get_category_breakdown(self) -> Dict[str, int]:
        """Get failure count by category."""
        breakdown: Dict[str, int] = {}
        for f in self._failures:
            breakdown[f.category] = breakdown.get(f.category, 0) + 1
        return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))

    def get_improvement_trend(self) -> Dict[str, float]:
        """Check if failure frequency is decreasing over time."""
        if len(self._failures) < 10:
            return {"trend": "insufficient_data"}

        mid = len(self._failures) // 2
        first_half_rate = mid  # Count in first half
        second_half_rate = len(self._failures) - mid  # Count in second half

        # Normalize by time span
        first_span = self._failures[mid - 1].timestamp - self._failures[0].timestamp
        second_span = self._failures[-1].timestamp - self._failures[mid].timestamp

        if first_span <= 0 or second_span <= 0:
            return {"trend": "insufficient_data"}

        first_rate = first_half_rate / first_span
        second_rate = second_half_rate / second_span

        improvement = (first_rate - second_rate) / first_rate if first_rate > 0 else 0

        return {
            "trend": "improving" if improvement > 0.1 else "stable" if improvement > -0.1 else "declining",
            "improvement_pct": round(improvement * 100, 1),
            "first_half_rate": round(first_rate, 3),
            "second_half_rate": round(second_rate, 3),
        }

    def clear_session(self):
        """Clear session failures (keep lifetime patterns)."""
        self._session_failures.clear()

    def get_stats(self) -> dict:
        return {
            "total_failures": len(self._failures),
            "session_failures": len(self._session_failures),
            "patterns_found": len(self._patterns),
            "category_breakdown": self.get_category_breakdown(),
        }
