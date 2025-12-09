# PHASE II — NOT RUN IN PHASE I
"""
U2 Trace Health Snapshot & Multi-Run Trend Analysis

STATUS: PHASE II — NOT RUN IN PHASE I

Provides a multi-run telemetry insight layer:
- Trace health snapshots (single-run view)
- Cross-run trace trends (historical analysis)
- Global health integration (summary for dashboards)

INVARIANTS:
- Read-only: Never modifies input data
- Deterministic: Same inputs always produce same output
- Zero-semantics: Analysis does not affect experiment behavior

Usage:
    from experiments.u2.trace_health import (
        build_trace_health_snapshot,
        build_trace_trend,
        summarize_trace_for_global_health,
    )
    
    # Single run snapshot
    snapshot = build_trace_health_snapshot(corr_summary, hotspot_report)
    
    # Multi-run trend analysis
    trend = build_trace_trend([snapshot1, snapshot2, snapshot3])
    
    # Global health summary
    global_health = summarize_trace_for_global_health(snapshot, trend)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set

from .inspector import HotspotReport
from .trace_correlator import CorrelationSummary

# Schema version for trace health snapshots
TRACE_HEALTH_SCHEMA_VERSION = "trace-health-1.0.0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — Trace Health Snapshot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class TraceHealthSnapshot:
    """
    Unified health snapshot for a single trace run.
    
    Combines correlation summary and hotspot data into a single view
    suitable for historical tracking and trend analysis.
    """
    schema_version: str
    trace_status: Literal["OK", "WARN", "ERROR"]
    coverage_percentage: float
    max_cycle_duration_ms: float
    cycles_with_errors: int
    total_error_count: int
    top_hotspot_cycles: List[int]  # e.g., top 3 cycles by duration
    run_id: Optional[str] = None
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "trace_status": self.trace_status,
            "coverage_percentage": self.coverage_percentage,
            "max_cycle_duration_ms": self.max_cycle_duration_ms,
            "cycles_with_errors": self.cycles_with_errors,
            "total_error_count": self.total_error_count,
            "top_hotspot_cycles": self.top_hotspot_cycles,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
        }


def build_trace_health_snapshot(
    corr_summary: CorrelationSummary,
    hotspots: HotspotReport,
    *,
    top_n_hotspots: int = 3,
    run_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a unified health snapshot from correlation summary and hotspot report.
    
    Args:
        corr_summary: Correlation summary from TraceCorrelator.
        hotspots: Hotspot report from TraceLogInspector.
        top_n_hotspots: Number of top hotspot cycles to include.
        run_id: Optional run identifier for tracking.
        timestamp: Optional timestamp for the snapshot.
    
    Returns:
        Dictionary with trace health snapshot data.
    
    Example:
        >>> snapshot = build_trace_health_snapshot(corr_summary, hotspots)
        >>> print(snapshot["trace_status"])
        'OK'
    """
    # Extract top hotspot cycles
    top_hotspot_cycles = [
        entry.cycle for entry in hotspots.entries[:top_n_hotspots]
    ]
    
    snapshot = TraceHealthSnapshot(
        schema_version=TRACE_HEALTH_SCHEMA_VERSION,
        trace_status=corr_summary.status,
        coverage_percentage=corr_summary.coverage_percentage,
        max_cycle_duration_ms=hotspots.max_duration_ms,
        cycles_with_errors=hotspots.cycles_with_errors,
        total_error_count=hotspots.total_error_count,
        top_hotspot_cycles=top_hotspot_cycles,
        run_id=run_id,
        timestamp=timestamp,
    )
    
    return snapshot.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — Cross-Run Trace Trends
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


TrendDirection = Literal["IMPROVING", "STABLE", "DEGRADING", "UNKNOWN"]


@dataclass
class TraceTrend:
    """
    Cross-run trend analysis for trace health metrics.
    
    Tracks how metrics evolve across multiple runs to detect
    performance degradation or improvement patterns.
    """
    schema_version: str
    run_count: int
    
    # Duration trends
    duration_trend: TrendDirection
    duration_values: List[float]  # Historical max durations
    duration_change_pct: Optional[float]  # % change first to last
    
    # Error trends
    error_trend: TrendDirection
    error_counts: List[int]  # Historical error counts
    error_change_pct: Optional[float]  # % change first to last
    
    # Hotspot stability
    hotspot_stability: float  # 0.0 = completely different, 1.0 = identical
    persistent_hotspots: List[int]  # Cycles that appear in multiple runs
    hotspot_frequency: Dict[int, int]  # cycle -> number of runs it appeared
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "run_count": self.run_count,
            "duration_trend": self.duration_trend,
            "duration_values": self.duration_values,
            "duration_change_pct": self.duration_change_pct,
            "error_trend": self.error_trend,
            "error_counts": self.error_counts,
            "error_change_pct": self.error_change_pct,
            "hotspot_stability": self.hotspot_stability,
            "persistent_hotspots": self.persistent_hotspots,
            "hotspot_frequency": self.hotspot_frequency,
        }


def _compute_trend_direction(
    values: List[float],
    threshold_pct: float = 10.0,
) -> TrendDirection:
    """
    Compute trend direction from a sequence of values.
    
    Args:
        values: Historical values (oldest to newest).
        threshold_pct: Minimum % change to be considered a trend.
    
    Returns:
        TrendDirection indicating improvement, stability, or degradation.
    """
    if len(values) < 2:
        return "UNKNOWN"
    
    first = values[0]
    last = values[-1]
    
    if first == 0:
        if last == 0:
            return "STABLE"
        elif last > 0:
            return "DEGRADING"  # Went from 0 to positive (for errors)
        else:
            return "IMPROVING"  # Went from 0 to negative (shouldn't happen)
    
    change_pct = ((last - first) / abs(first)) * 100
    
    if abs(change_pct) < threshold_pct:
        return "STABLE"
    elif change_pct < 0:
        return "IMPROVING"  # Values decreased (fewer errors, shorter durations)
    else:
        return "DEGRADING"  # Values increased


def _compute_change_pct(values: List[float]) -> Optional[float]:
    """Compute percentage change from first to last value."""
    if len(values) < 2:
        return None
    
    first = values[0]
    last = values[-1]
    
    if first == 0:
        if last == 0:
            return 0.0
        return None  # Can't compute % change from 0
    
    return ((last - first) / abs(first)) * 100


def _compute_hotspot_stability(
    hotspot_lists: List[List[int]],
) -> tuple[float, List[int], Dict[int, int]]:
    """
    Compute hotspot stability across runs.
    
    Args:
        hotspot_lists: List of top hotspot cycles from each run.
    
    Returns:
        Tuple of (stability_score, persistent_hotspots, frequency_map).
        
        - stability_score: 0.0 = completely different, 1.0 = identical
        - persistent_hotspots: Cycles appearing in >50% of runs
        - frequency_map: cycle -> count of runs it appeared in
    """
    if not hotspot_lists:
        return 0.0, [], {}
    
    if len(hotspot_lists) == 1:
        # Single run, no stability measure
        return 1.0, hotspot_lists[0], {c: 1 for c in hotspot_lists[0]}
    
    # Count frequency of each cycle across runs
    frequency: Counter[int] = Counter()
    for hotspots in hotspot_lists:
        frequency.update(hotspots)
    
    frequency_map = dict(frequency)
    num_runs = len(hotspot_lists)
    
    # Persistent hotspots = appear in >50% of runs
    persistent = [
        cycle for cycle, count in frequency.items()
        if count > num_runs / 2
    ]
    persistent.sort()  # Deterministic ordering
    
    # Stability score: Jaccard similarity across consecutive pairs
    if len(hotspot_lists) < 2:
        return 1.0, persistent, frequency_map
    
    similarities = []
    for i in range(len(hotspot_lists) - 1):
        set_a = set(hotspot_lists[i])
        set_b = set(hotspot_lists[i + 1])
        
        if not set_a and not set_b:
            similarities.append(1.0)
        elif not set_a or not set_b:
            similarities.append(0.0)
        else:
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            similarities.append(intersection / union)
    
    stability = sum(similarities) / len(similarities) if similarities else 0.0
    
    return stability, persistent, frequency_map


def build_trace_trend(
    history: Sequence[Dict[str, Any]],
    *,
    trend_threshold_pct: float = 10.0,
) -> Dict[str, Any]:
    """
    Build cross-run trend analysis from historical snapshots.
    
    Args:
        history: Sequence of trace health snapshots (oldest to newest).
        trend_threshold_pct: Minimum % change to be considered a trend.
    
    Returns:
        Dictionary with trend analysis data.
    
    Example:
        >>> trend = build_trace_trend([snapshot1, snapshot2, snapshot3])
        >>> print(trend["duration_trend"])
        'STABLE'
        >>> print(trend["persistent_hotspots"])
        [42, 100]
    """
    if not history:
        return TraceTrend(
            schema_version=TRACE_HEALTH_SCHEMA_VERSION,
            run_count=0,
            duration_trend="UNKNOWN",
            duration_values=[],
            duration_change_pct=None,
            error_trend="UNKNOWN",
            error_counts=[],
            error_change_pct=None,
            hotspot_stability=0.0,
            persistent_hotspots=[],
            hotspot_frequency={},
        ).to_dict()
    
    # Extract historical values
    duration_values = [
        float(s.get("max_cycle_duration_ms", 0.0)) for s in history
    ]
    error_counts = [
        int(s.get("total_error_count", 0)) for s in history
    ]
    hotspot_lists = [
        list(s.get("top_hotspot_cycles", [])) for s in history
    ]
    
    # Compute trends
    duration_trend = _compute_trend_direction(duration_values, trend_threshold_pct)
    error_trend = _compute_trend_direction(
        [float(e) for e in error_counts],
        trend_threshold_pct,
    )
    
    # Compute hotspot stability
    stability, persistent, frequency = _compute_hotspot_stability(hotspot_lists)
    
    trend = TraceTrend(
        schema_version=TRACE_HEALTH_SCHEMA_VERSION,
        run_count=len(history),
        duration_trend=duration_trend,
        duration_values=duration_values,
        duration_change_pct=_compute_change_pct(duration_values),
        error_trend=error_trend,
        error_counts=error_counts,
        error_change_pct=_compute_change_pct([float(e) for e in error_counts]),
        hotspot_stability=stability,
        persistent_hotspots=persistent,
        hotspot_frequency=frequency,
    )
    
    return trend.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — Global Health Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class GlobalTraceHealth:
    """
    Summarized trace health for global health dashboards.
    
    Provides a high-level view suitable for system-wide monitoring
    and alerting integration.
    """
    schema_version: str
    trace_ok: bool
    trace_status: Literal["OK", "WARN", "ERROR"]
    persistent_hotspots: List[int]
    hotspot_stability: float
    error_trend: TrendDirection
    duration_trend: TrendDirection
    coverage_percentage: float
    max_cycle_duration_ms: float
    cycles_with_errors: int
    run_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "trace_ok": self.trace_ok,
            "trace_status": self.trace_status,
            "persistent_hotspots": self.persistent_hotspots,
            "hotspot_stability": self.hotspot_stability,
            "error_trend": self.error_trend,
            "duration_trend": self.duration_trend,
            "coverage_percentage": self.coverage_percentage,
            "max_cycle_duration_ms": self.max_cycle_duration_ms,
            "cycles_with_errors": self.cycles_with_errors,
            "run_count": self.run_count,
        }


def summarize_trace_for_global_health(
    trace_snapshot: Dict[str, Any],
    trace_trend: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize trace health for global health dashboard integration.
    
    Combines current snapshot and historical trend into a unified
    summary suitable for system-wide health monitoring.
    
    Args:
        trace_snapshot: Current trace health snapshot.
        trace_trend: Optional cross-run trend analysis.
    
    Returns:
        Dictionary with global health summary.
    
    Example:
        >>> summary = summarize_trace_for_global_health(snapshot, trend)
        >>> print(summary["trace_ok"])
        True
        >>> print(summary["error_trend"])
        'STABLE'
    """
    trace_status = trace_snapshot.get("trace_status", "UNKNOWN")
    trace_ok = trace_status == "OK"
    
    # Extract from trend if available
    if trace_trend:
        persistent_hotspots = trace_trend.get("persistent_hotspots", [])
        hotspot_stability = trace_trend.get("hotspot_stability", 0.0)
        error_trend = trace_trend.get("error_trend", "UNKNOWN")
        duration_trend = trace_trend.get("duration_trend", "UNKNOWN")
        run_count = trace_trend.get("run_count", 1)
    else:
        persistent_hotspots = trace_snapshot.get("top_hotspot_cycles", [])
        hotspot_stability = 1.0  # Single run = stable
        error_trend = "UNKNOWN"
        duration_trend = "UNKNOWN"
        run_count = 1
    
    summary = GlobalTraceHealth(
        schema_version=TRACE_HEALTH_SCHEMA_VERSION,
        trace_ok=trace_ok,
        trace_status=trace_status,
        persistent_hotspots=persistent_hotspots,
        hotspot_stability=hotspot_stability,
        error_trend=error_trend,
        duration_trend=duration_trend,
        coverage_percentage=trace_snapshot.get("coverage_percentage", 0.0),
        max_cycle_duration_ms=trace_snapshot.get("max_cycle_duration_ms", 0.0),
        cycles_with_errors=trace_snapshot.get("cycles_with_errors", 0),
        run_count=run_count,
    )
    
    return summary.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience functions for integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_trace_health(
    corr_summary: CorrelationSummary,
    hotspots: HotspotReport,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    run_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One-shot analysis: build snapshot, trend, and global summary.
    
    Args:
        corr_summary: Correlation summary from TraceCorrelator.
        hotspots: Hotspot report from TraceLogInspector.
        history: Optional historical snapshots for trend analysis.
        run_id: Optional run identifier.
        timestamp: Optional timestamp.
    
    Returns:
        Dictionary with:
        - snapshot: Current trace health snapshot
        - trend: Cross-run trend analysis (if history provided)
        - global_health: Global health summary
    """
    snapshot = build_trace_health_snapshot(
        corr_summary,
        hotspots,
        run_id=run_id,
        timestamp=timestamp,
    )
    
    # Build trend if history is provided
    if history:
        # Include current snapshot in trend
        full_history = list(history) + [snapshot]
        trend = build_trace_trend(full_history)
    else:
        trend = None
    
    global_health = summarize_trace_for_global_health(snapshot, trend)
    
    return {
        "snapshot": snapshot,
        "trend": trend,
        "global_health": global_health,
    }

