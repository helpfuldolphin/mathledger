"""
TDA Pattern Classification from Window Metrics

Provides windowed pattern classification for P5 integration:
- classify_windows(): Classify each window in a sequence
- aggregate_pattern_summary(): Compute summary statistics across windows

See: docs/system_law/TDA_PhaseX_Binding.md Section 10

SHADOW MODE CONTRACT:
- All classifications are observational only
- No governance modification based on pattern detection
- Classifications are logged for analysis, never enforced
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.tda.metrics import TDAWindowMetrics
from backend.tda.monitor import TDASummary
from backend.tda.pattern_classifier import (
    RTTSPattern,
    PatternClassification,
    TDAPatternClassifier,
    PATTERN_PRIORITY,
)

__all__ = [
    "classify_windows",
    "aggregate_pattern_summary",
    "attach_windowed_patterns_to_evidence",
    "extract_windowed_patterns_status",
    "get_top_events_digest",
    "attach_signals_tda_windowed_patterns",
    "WindowPatternResult",
    "PatternAggregateSummary",
    "WindowedPatternsStatus",
    "TopEventDigest",
    "DEFAULT_TOP_EVENTS",
]


# Default maximum windows to include in evidence attachment
DEFAULT_MAX_WINDOWS = 50


@dataclass
class WindowPatternResult:
    """
    Result of classifying a single window.

    SHADOW MODE: Observational only.
    """
    window_index: int
    window_id: str
    cycle_range: Tuple[int, int]
    classification: PatternClassification
    mode: str = "SHADOW"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "window_index": self.window_index,
            "window_id": self.window_id,
            "cycle_range": list(self.cycle_range),
            "pattern": self.classification.pattern.value,
            "confidence": round(self.classification.confidence, 4),
            "primary_triggers": self.classification.primary_triggers,
            "secondary_triggers": self.classification.secondary_triggers,
            "mode": self.mode,
        }


@dataclass
class PatternAggregateSummary:
    """
    Aggregate summary of pattern classifications across windows.

    SHADOW MODE: Observational only.
    """
    # Total windows analyzed
    total_windows: int = 0

    # Pattern counts
    pattern_counts: Dict[str, int] = field(default_factory=dict)

    # Dominant pattern (excluding NONE)
    dominant_pattern: str = "NONE"
    dominant_pattern_count: int = 0
    dominant_pattern_ratio: float = 0.0

    # High confidence events (confidence >= 0.75, excluding NONE)
    high_confidence_events: int = 0
    high_confidence_windows: List[int] = field(default_factory=list)

    # Pattern streaks
    max_streak_pattern: str = "NONE"
    max_streak_length: int = 0
    current_streak_pattern: str = "NONE"
    current_streak_length: int = 0

    # Transitions
    pattern_transitions: int = 0

    # Time range
    first_window_index: int = 0
    last_window_index: int = 0
    first_cycle: int = 0
    last_cycle: int = 0

    # SHADOW MODE marker
    mode: str = "SHADOW"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "schema_version": "1.0.0",
            "total_windows": self.total_windows,
            "pattern_counts": dict(self.pattern_counts),
            "dominant_pattern": self.dominant_pattern,
            "dominant_pattern_count": self.dominant_pattern_count,
            "dominant_pattern_ratio": round(self.dominant_pattern_ratio, 4),
            "high_confidence_events": self.high_confidence_events,
            "high_confidence_windows": self.high_confidence_windows[:10],  # Limit to first 10
            "streaks": {
                "max_pattern": self.max_streak_pattern,
                "max_length": self.max_streak_length,
                "current_pattern": self.current_streak_pattern,
                "current_length": self.current_streak_length,
            },
            "pattern_transitions": self.pattern_transitions,
            "time_range": {
                "first_window": self.first_window_index,
                "last_window": self.last_window_index,
                "first_cycle": self.first_cycle,
                "last_cycle": self.last_cycle,
            },
            "mode": self.mode,
        }


def classify_windows(
    tda_windows_p3: List[TDAWindowMetrics],
    tda_windows_p4: Optional[List[TDAWindowMetrics]] = None,
    classifier: Optional[TDAPatternClassifier] = None,
    history_depth: int = 3,
) -> List[WindowPatternResult]:
    """
    Classify RTTS patterns for each window in a sequence.

    Each window is classified using its metrics plus a sliding history
    of previous windows for trend detection (deltas, slopes).

    Args:
        tda_windows_p3: List of P3 window metrics (required)
        tda_windows_p4: Optional list of P4 window metrics (for DRS)
        classifier: Optional pre-configured classifier (creates new if None)
        history_depth: Number of previous windows to include for trend analysis

    Returns:
        List of WindowPatternResult, one per window, in deterministic order

    SHADOW MODE: All classifications are observational only.
    """
    if not tda_windows_p3:
        return []

    if classifier is None:
        classifier = TDAPatternClassifier()

    results: List[WindowPatternResult] = []

    # Align P4 windows with P3 by index (if provided)
    p4_by_index: Dict[int, TDAWindowMetrics] = {}
    if tda_windows_p4:
        for w in tda_windows_p4:
            p4_by_index[w.window_index] = w

    # Process each P3 window
    for i, p3_window in enumerate(tda_windows_p3):
        # Build P3 summary from window
        p3_summary = _window_to_summary(p3_window)

        # Get corresponding P4 window if available
        p4_summary: Optional[TDASummary] = None
        p4_window = p4_by_index.get(p3_window.window_index)
        if p4_window:
            p4_summary = _window_to_summary(p4_window)

        # Build history for trend analysis
        history_start = max(0, i - history_depth)
        window_history = tda_windows_p3[history_start:i + 1]

        # Generate window ID
        window_id = f"window_{p3_window.window_index:04d}"

        # Cycle range
        cycle_range = (p3_window.window_start_cycle, p3_window.window_end_cycle)

        # Classify
        classification = classifier.classify(
            p3_tda=p3_summary,
            p4_tda=p4_summary,
            window_history=window_history if len(window_history) > 1 else None,
            window_id=window_id,
            cycle_range=cycle_range,
        )

        result = WindowPatternResult(
            window_index=p3_window.window_index,
            window_id=window_id,
            cycle_range=cycle_range,
            classification=classification,
        )
        results.append(result)

    return results


def aggregate_pattern_summary(
    window_results: List[WindowPatternResult],
) -> PatternAggregateSummary:
    """
    Compute aggregate statistics from window classification results.

    Computes:
    - Pattern counts and dominant pattern
    - High confidence events
    - Pattern streaks (max and current)
    - Transition count

    Args:
        window_results: List of WindowPatternResult from classify_windows()

    Returns:
        PatternAggregateSummary with aggregate statistics

    SHADOW MODE: Summary is observational only.
    """
    if not window_results:
        return PatternAggregateSummary()

    # Initialize counts for all patterns
    pattern_counts: Dict[str, int] = {p.value: 0 for p in RTTSPattern}

    # Track high confidence events
    high_confidence_events = 0
    high_confidence_windows: List[int] = []

    # Track streaks
    max_streak_pattern = RTTSPattern.NONE.value
    max_streak_length = 0
    current_streak_pattern = RTTSPattern.NONE.value
    current_streak_length = 0

    # Track transitions
    transitions = 0
    prev_pattern: Optional[str] = None

    # Process each result in order
    for result in window_results:
        pattern = result.classification.pattern.value
        confidence = result.classification.confidence

        # Count pattern
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Track high confidence events (excluding NONE)
        if confidence >= 0.75 and pattern != RTTSPattern.NONE.value:
            high_confidence_events += 1
            high_confidence_windows.append(result.window_index)

        # Track transitions
        if prev_pattern is not None and prev_pattern != pattern:
            transitions += 1

        # Track streaks
        if pattern == current_streak_pattern:
            current_streak_length += 1
        else:
            # Check if previous streak was longer
            if current_streak_length > max_streak_length:
                max_streak_length = current_streak_length
                max_streak_pattern = current_streak_pattern
            # Start new streak
            current_streak_pattern = pattern
            current_streak_length = 1

        prev_pattern = pattern

    # Final streak check
    if current_streak_length > max_streak_length:
        max_streak_length = current_streak_length
        max_streak_pattern = current_streak_pattern

    # Find dominant pattern (excluding NONE)
    # Only consider patterns with at least one occurrence
    non_none_counts = {k: v for k, v in pattern_counts.items() if k != RTTSPattern.NONE.value and v > 0}
    if non_none_counts:
        dominant_pattern = max(non_none_counts, key=lambda k: non_none_counts[k])
        dominant_count = non_none_counts[dominant_pattern]
    else:
        dominant_pattern = RTTSPattern.NONE.value
        dominant_count = pattern_counts.get(RTTSPattern.NONE.value, 0)

    total_windows = len(window_results)
    dominant_ratio = dominant_count / total_windows if total_windows > 0 else 0.0

    # Time range
    first_window = window_results[0]
    last_window = window_results[-1]

    return PatternAggregateSummary(
        total_windows=total_windows,
        pattern_counts=pattern_counts,
        dominant_pattern=dominant_pattern,
        dominant_pattern_count=dominant_count,
        dominant_pattern_ratio=dominant_ratio,
        high_confidence_events=high_confidence_events,
        high_confidence_windows=high_confidence_windows,
        max_streak_pattern=max_streak_pattern,
        max_streak_length=max_streak_length,
        current_streak_pattern=current_streak_pattern,
        current_streak_length=current_streak_length,
        pattern_transitions=transitions,
        first_window_index=first_window.window_index,
        last_window_index=last_window.window_index,
        first_cycle=first_window.cycle_range[0],
        last_cycle=last_window.cycle_range[1],
    )


def attach_windowed_patterns_to_evidence(
    evidence: Dict[str, Any],
    window_results: List[WindowPatternResult],
    aggregate_summary: Optional[PatternAggregateSummary] = None,
    max_windows: int = DEFAULT_MAX_WINDOWS,
    include_per_window: bool = True,
    include_triggers: bool = False,
) -> Dict[str, Any]:
    """
    Attach windowed pattern classifications to evidence pack.

    Extends governance.tda.patterns with:
    - per_window_classifications: List of per-window results (bounded)
    - aggregate_summary: Summary statistics across all windows
    - windowed_metadata: Info about windowing parameters

    Args:
        evidence: Evidence pack dict to augment
        window_results: List of WindowPatternResult from classify_windows()
        aggregate_summary: Optional pre-computed summary (computes if None)
        max_windows: Maximum windows to include in per_window list
        include_per_window: Whether to include per-window classifications
        include_triggers: Whether to include trigger details in per-window

    Returns:
        Modified evidence dict with windowed patterns section added

    SHADOW MODE: Observational only.
    """
    # Ensure governance.tda.patterns structure exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    if "tda" not in evidence["governance"]:
        evidence["governance"]["tda"] = {"mode": "SHADOW"}
    if "patterns" not in evidence["governance"]["tda"]:
        evidence["governance"]["tda"]["patterns"] = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
        }

    patterns_section = evidence["governance"]["tda"]["patterns"]

    # Compute aggregate if not provided
    if aggregate_summary is None and window_results:
        aggregate_summary = aggregate_pattern_summary(window_results)

    # Add aggregate summary
    if aggregate_summary:
        patterns_section["aggregate_summary"] = aggregate_summary.to_dict()

    # Add per-window classifications (bounded)
    if include_per_window and window_results:
        # Determine which windows to include
        total_windows = len(window_results)
        if total_windows <= max_windows:
            selected_windows = window_results
            truncated = False
        else:
            # Include first and last windows, plus evenly distributed middle
            # This ensures we capture start, end, and representative middle
            selected_indices = _select_representative_indices(total_windows, max_windows)
            selected_windows = [window_results[i] for i in selected_indices]
            truncated = True

        # Build per-window list
        per_window_list = []
        for wr in selected_windows:
            entry: Dict[str, Any] = {
                "window_index": wr.window_index,
                "window_id": wr.window_id,
                "cycle_range": list(wr.cycle_range),
                "pattern": wr.classification.pattern.value,
                "confidence": round(wr.classification.confidence, 4),
            }
            if include_triggers:
                entry["primary_triggers"] = wr.classification.primary_triggers
                entry["secondary_triggers"] = wr.classification.secondary_triggers
            per_window_list.append(entry)

        patterns_section["per_window_classifications"] = per_window_list
        patterns_section["windowed_metadata"] = {
            "total_windows": total_windows,
            "included_windows": len(selected_windows),
            "truncated": truncated,
            "max_windows_setting": max_windows,
            "include_triggers": include_triggers,
        }

    # Ensure SHADOW mode marker
    patterns_section["mode"] = "SHADOW"
    patterns_section["verifier_note"] = (
        "Windowed pattern classification is SHADOW-ONLY. "
        "No enforcement triggered. Per-window data may be truncated."
    )

    return evidence


def _window_to_summary(window: TDAWindowMetrics) -> TDASummary:
    """Convert TDAWindowMetrics to TDASummary for classification."""
    return TDASummary(
        total_cycles=window.window_end_cycle - window.window_start_cycle + 1,
        sns_mean=window.sns_mean,
        sns_max=window.sns_max,
        pcs_mean=window.pcs_mean,
        pcs_min=window.pcs_min,
        hss_mean=window.hss_mean,
        hss_min=window.hss_min,
        envelope_occupancy=window.envelope_occupancy_rate,
        envelope_exit_total=window.envelope_exit_count,
        max_envelope_exit_streak=window.max_envelope_exit_streak,
        total_red_flags=(
            window.tda_sns_anomaly_flags +
            window.tda_pcs_collapse_flags +
            window.tda_hss_degradation_flags +
            window.tda_envelope_exit_flags
        ),
    )


def _select_representative_indices(total: int, max_count: int) -> List[int]:
    """
    Select representative indices for truncation.

    Strategy: Include first 5, last 5, and evenly distributed middle.
    This ensures start/end visibility while sampling the middle.
    """
    if total <= max_count:
        return list(range(total))

    indices: List[int] = []

    # Include first 5
    first_count = min(5, max_count // 3)
    indices.extend(range(first_count))

    # Include last 5
    last_count = min(5, max_count // 3)
    last_start = total - last_count
    indices.extend(range(last_start, total))

    # Fill remaining with evenly distributed middle
    remaining = max_count - len(indices)
    if remaining > 0:
        middle_start = first_count
        middle_end = last_start
        middle_range = middle_end - middle_start

        if middle_range > 0 and remaining > 0:
            step = middle_range / (remaining + 1)
            for i in range(remaining):
                idx = int(middle_start + step * (i + 1))
                if idx not in indices and idx < middle_end:
                    indices.append(idx)

    # Sort and deduplicate
    indices = sorted(set(indices))

    return indices[:max_count]


# =============================================================================
# Status Extraction for signals.tda_windowed_patterns
# =============================================================================

@dataclass
class WindowedPatternsStatus:
    """
    Status summary for signals.tda_windowed_patterns.

    Provides a compact view of windowed pattern analysis results
    suitable for status panels and cross-panel correlation.

    SHADOW MODE: Observational only.
    """
    # Core summary
    dominant_pattern: str = "NONE"
    max_streak_pattern: str = "NONE"
    max_streak_length: int = 0
    high_confidence_count: int = 0

    # Window coverage
    total_windows: int = 0
    windows_with_patterns: int = 0  # Non-NONE patterns

    # Pattern distribution (top 3)
    top_patterns: List[Tuple[str, int]] = field(default_factory=list)

    # Transition rate
    transition_rate: float = 0.0  # transitions per window

    # Time bounds
    first_cycle: int = 0
    last_cycle: int = 0

    # SHADOW MODE marker
    mode: str = "SHADOW"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary for signals."""
        return {
            "dominant_pattern": self.dominant_pattern,
            "max_streak": {
                "pattern": self.max_streak_pattern,
                "length": self.max_streak_length,
            },
            "high_confidence_count": self.high_confidence_count,
            "coverage": {
                "total_windows": self.total_windows,
                "windows_with_patterns": self.windows_with_patterns,
            },
            "top_patterns": [
                {"pattern": p, "count": c} for p, c in self.top_patterns
            ],
            "transition_rate": round(self.transition_rate, 4),
            "time_range": {
                "first_cycle": self.first_cycle,
                "last_cycle": self.last_cycle,
            },
            "mode": self.mode,
        }


@dataclass
class TopEventDigest:
    """
    Single event in the top-5 events digest.

    Represents a high-confidence non-NONE pattern classification
    for human review.

    SHADOW MODE: Observational only.
    """
    window_index: int
    pattern: str
    confidence: float
    cycle_range: Tuple[int, int]
    primary_triggers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "window_index": self.window_index,
            "pattern": self.pattern,
            "confidence": round(self.confidence, 4),
            "cycle_range": list(self.cycle_range),
            "primary_triggers": self.primary_triggers,
        }


def extract_windowed_patterns_status(
    window_results: List[WindowPatternResult],
    aggregate_summary: Optional[PatternAggregateSummary] = None,
) -> WindowedPatternsStatus:
    """
    Extract status summary for signals.tda_windowed_patterns.

    Produces a compact status suitable for status panels and
    cross-panel correlation.

    Args:
        window_results: List of WindowPatternResult from classify_windows()
        aggregate_summary: Optional pre-computed summary (computes if None)

    Returns:
        WindowedPatternsStatus for signals attachment

    SHADOW MODE: Observational only.
    """
    if not window_results:
        return WindowedPatternsStatus()

    # Compute summary if not provided
    if aggregate_summary is None:
        aggregate_summary = aggregate_pattern_summary(window_results)

    # Count windows with non-NONE patterns
    windows_with_patterns = sum(
        1 for r in window_results
        if r.classification.pattern != RTTSPattern.NONE
    )

    # Compute transition rate
    transition_rate = (
        aggregate_summary.pattern_transitions / aggregate_summary.total_windows
        if aggregate_summary.total_windows > 0
        else 0.0
    )

    # Get top 3 patterns (excluding NONE, sorted by count desc)
    pattern_counts = aggregate_summary.pattern_counts
    non_none_patterns = [
        (pattern, count)
        for pattern, count in pattern_counts.items()
        if pattern != RTTSPattern.NONE.value and count > 0
    ]
    # Sort by count descending, then by pattern name for determinism
    non_none_patterns.sort(key=lambda x: (-x[1], x[0]))
    top_patterns = non_none_patterns[:3]

    return WindowedPatternsStatus(
        dominant_pattern=aggregate_summary.dominant_pattern,
        max_streak_pattern=aggregate_summary.max_streak_pattern,
        max_streak_length=aggregate_summary.max_streak_length,
        high_confidence_count=aggregate_summary.high_confidence_events,
        total_windows=aggregate_summary.total_windows,
        windows_with_patterns=windows_with_patterns,
        top_patterns=top_patterns,
        transition_rate=transition_rate,
        first_cycle=aggregate_summary.first_cycle,
        last_cycle=aggregate_summary.last_cycle,
    )


# Default maximum events in top-5 digest
DEFAULT_TOP_EVENTS = 5


def get_top_events_digest(
    window_results: List[WindowPatternResult],
    max_events: int = DEFAULT_TOP_EVENTS,
    min_confidence: float = 0.5,
) -> List[TopEventDigest]:
    """
    Get top N highest-confidence non-NONE events for human review.

    Produces a bounded, human-readable digest of the most significant
    pattern classifications.

    Args:
        window_results: List of WindowPatternResult from classify_windows()
        max_events: Maximum events to include (default 5)
        min_confidence: Minimum confidence threshold (default 0.5)

    Returns:
        List of TopEventDigest, sorted by confidence descending,
        deterministic ordering for equal confidence

    SHADOW MODE: Observational only.
    """
    if not window_results:
        return []

    # Filter to non-NONE patterns with minimum confidence
    candidates = [
        r for r in window_results
        if r.classification.pattern != RTTSPattern.NONE
        and r.classification.confidence >= min_confidence
    ]

    # Sort by confidence (desc), then window_index (asc) for determinism
    candidates.sort(
        key=lambda r: (-r.classification.confidence, r.window_index)
    )

    # Take top N
    top_events = candidates[:max_events]

    # Convert to digest entries
    return [
        TopEventDigest(
            window_index=r.window_index,
            pattern=r.classification.pattern.value,
            confidence=r.classification.confidence,
            cycle_range=r.cycle_range,
            primary_triggers=r.classification.primary_triggers,
        )
        for r in top_events
    ]


def attach_signals_tda_windowed_patterns(
    signals: Dict[str, Any],
    window_results: List[WindowPatternResult],
    aggregate_summary: Optional[PatternAggregateSummary] = None,
    include_top_events: bool = True,
    max_top_events: int = DEFAULT_TOP_EVENTS,
) -> Dict[str, Any]:
    """
    Attach windowed patterns status to signals dict.

    Creates signals.tda_windowed_patterns with:
    - status: WindowedPatternsStatus summary
    - top_events: Top N highest-confidence events (optional)

    Args:
        signals: Signals dict to augment
        window_results: List of WindowPatternResult
        aggregate_summary: Optional pre-computed summary
        include_top_events: Whether to include top events digest
        max_top_events: Maximum events in digest

    Returns:
        Modified signals dict

    SHADOW MODE: Observational only.
    """
    # Extract status
    status = extract_windowed_patterns_status(window_results, aggregate_summary)

    # Build windowed patterns section
    windowed_patterns: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": status.to_dict(),
        "mode": "SHADOW",
    }

    # Add top events if requested
    if include_top_events and window_results:
        top_events = get_top_events_digest(window_results, max_top_events)
        windowed_patterns["top_events"] = [e.to_dict() for e in top_events]
        windowed_patterns["top_events_count"] = len(top_events)

    # Attach to signals
    signals["tda_windowed_patterns"] = windowed_patterns

    return signals
