"""
Curriculum Integration Module for Phase X P3/P4 and Evidence Packs.

Provides attachment functions for integrating curriculum governance signals into:
- P3 stability reports (First-Light Integration)
- Evidence packs
- Uplift council views

SHADOW MODE CONTRACT:
- All functions are pure (no side effects) and non-mutating
- Governance signals are observational only
- No control flow modifications based on curriculum status
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from curriculum.enforcement import (
    DriftSeverity,
    DriftStatus,
    GovernanceSignal,
    DriftTimelineEvent,
    GovernanceSignalType,
)


# -----------------------------------------------------------------------------
# Type Aliases
# -----------------------------------------------------------------------------

CouncilStatus = Literal["OK", "WARN", "BLOCK"]


# -----------------------------------------------------------------------------
# P3 Stability Hook
# -----------------------------------------------------------------------------

def attach_curriculum_governance_to_p3(
    stability_report: Dict[str, Any],
    governance_signal: GovernanceSignal,
) -> Dict[str, Any]:
    """
    Attach curriculum governance signal to P3 stability report.

    Adds curriculum governance summary under stability_report["curriculum_governance"].

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Curriculum governance is observational only
    - Does not influence other stability metrics

    Args:
        stability_report: P3 stability report dictionary
        governance_signal: Curriculum governance signal from GovernanceSignalBuilder

    Returns:
        Updated stability report with curriculum_governance field
    """
    # Extract key fields from governance signal
    severity = governance_signal.severity
    status = governance_signal.status
    signal_type = governance_signal.signal_type

    # Compute curriculum health score (0.0 - 1.0)
    if severity == DriftSeverity.NONE:
        curriculum_health_score = 1.0
    elif severity == DriftSeverity.PARAMETRIC:
        curriculum_health_score = 0.7
    else:  # SEMANTIC
        curriculum_health_score = 0.3

    # Determine curriculum status light
    if status == DriftStatus.OK:
        status_light = "GREEN"
    elif status == DriftStatus.WARN:
        status_light = "YELLOW"
    else:  # BLOCK
        status_light = "RED"

    # Count violations by type
    violation_counts: Dict[str, int] = {}
    for v in governance_signal.violations:
        prefix = v.code.split("_")[0] if "_" in v.code else v.code
        violation_counts[prefix] = violation_counts.get(prefix, 0) + 1

    # Build curriculum governance summary
    curriculum_governance = {
        "schema_version": "1.0.0",
        "phase": governance_signal.phase,
        "mode": governance_signal.mode,
        "signal_id": governance_signal.signal_id,
        "signal_type": signal_type.value,
        "curriculum_fingerprint": governance_signal.curriculum_fingerprint,
        "active_slice": governance_signal.active_slice,
        "severity": severity.value,
        "status": status.value,
        "status_light": status_light,
        "curriculum_health_score": curriculum_health_score,
        "violation_count": len(governance_signal.violations),
        "violation_counts_by_type": violation_counts,
        "governance_action": governance_signal.governance_action,
        "hypothetical": governance_signal.hypothetical,
        "timestamp": governance_signal.timestamp,
    }

    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["curriculum_governance"] = curriculum_governance

    return updated_report


def attach_curriculum_timeline_to_p3(
    stability_report: Dict[str, Any],
    drift_events: List[DriftTimelineEvent],
) -> Dict[str, Any]:
    """
    Attach curriculum drift timeline summary to P3 stability report.

    Adds timeline summary under stability_report["curriculum_drift_timeline"].

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Timeline is observational only

    Args:
        stability_report: P3 stability report dictionary
        drift_events: List of drift timeline events

    Returns:
        Updated stability report with curriculum_drift_timeline field
    """
    if not drift_events:
        timeline_summary = {
            "schema_version": "1.0.0",
            "event_count": 0,
            "max_severity": "NONE",
            "overall_status": "OK",
            "events": [],
        }
    else:
        # Compute summary metrics
        severities = [e.drift_severity for e in drift_events]
        max_severity = max(severities, key=lambda s: {"NONE": 0, "PARAMETRIC": 1, "SEMANTIC": 2}.get(s.value, 0))

        statuses = [e.drift_status for e in drift_events]
        worst_status = max(statuses, key=lambda s: {"OK": 0, "WARN": 1, "BLOCK": 2}.get(s.value, 0))

        # Build event summaries (limit to last 10)
        event_summaries = [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "severity": e.drift_severity.value,
                "status": e.drift_status.value,
                "changed_param_count": len(e.changed_params),
                "mono_violation_count": len(e.monotonicity_violations),
                "gate_violation_count": len(e.gate_evolution_violations),
            }
            for e in drift_events[-10:]
        ]

        timeline_summary = {
            "schema_version": "1.0.0",
            "event_count": len(drift_events),
            "max_severity": max_severity.value,
            "overall_status": worst_status.value,
            "semantic_violations": sum(1 for e in drift_events if e.drift_severity == DriftSeverity.SEMANTIC),
            "parametric_changes": sum(1 for e in drift_events if e.drift_severity == DriftSeverity.PARAMETRIC),
            "events": event_summaries,
        }

    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["curriculum_drift_timeline"] = timeline_summary

    return updated_report


# -----------------------------------------------------------------------------
# Evidence Attachment
# -----------------------------------------------------------------------------

def attach_curriculum_to_evidence(
    evidence: Dict[str, Any],
    governance_signal: GovernanceSignal,
    drift_timeline: Optional[List[DriftTimelineEvent]] = None,
) -> Dict[str, Any]:
    """
    Attach curriculum governance to evidence pack.

    Stores under evidence["governance"]["curriculum"] with:
    - signal: Full governance signal
    - timeline: Drift timeline summary
    - council_status: OK/WARN/BLOCK classification

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - All data is observational only

    Args:
        evidence: Evidence pack dictionary
        governance_signal: Curriculum governance signal
        drift_timeline: Optional list of drift timeline events

    Returns:
        Updated evidence with governance.curriculum field
    """
    # Compute council status
    council_status = council_classify_curriculum(governance_signal, drift_timeline)

    # Build timeline summary
    timeline_summary: Dict[str, Any] = {"event_count": 0}
    if drift_timeline:
        severities = [e.drift_severity for e in drift_timeline]
        max_sev = max(severities, key=lambda s: {"NONE": 0, "PARAMETRIC": 1, "SEMANTIC": 2}.get(s.value, 0))

        timeline_summary = {
            "event_count": len(drift_timeline),
            "max_severity": max_sev.value,
            "semantic_count": sum(1 for e in drift_timeline if e.drift_severity == DriftSeverity.SEMANTIC),
            "parametric_count": sum(1 for e in drift_timeline if e.drift_severity == DriftSeverity.PARAMETRIC),
            "first_event_id": drift_timeline[0].event_id if drift_timeline else None,
            "last_event_id": drift_timeline[-1].event_id if drift_timeline else None,
        }

    # Build curriculum governance block
    curriculum_block = {
        "schema_version": "1.0.0",
        "signal": governance_signal.to_dict(),
        "timeline_summary": timeline_summary,
        "council_status": council_status,
        "council_classification": {
            "status": council_status,
            "severity": governance_signal.severity.value,
            "violation_count": len(governance_signal.violations),
            "would_block_uplift": council_status == "BLOCK",
            "would_warn_uplift": council_status == "WARN",
        },
    }

    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)

    # Ensure governance key exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])

    updated_evidence["governance"]["curriculum"] = curriculum_block

    return updated_evidence


# -----------------------------------------------------------------------------
# Council Classification
# -----------------------------------------------------------------------------

def council_classify_curriculum(
    governance_signal: GovernanceSignal,
    drift_timeline: Optional[List[DriftTimelineEvent]] = None,
) -> CouncilStatus:
    """
    Reduce curriculum governance signals to OK/WARN/BLOCK for Uplift Council.

    Classification rules:
    - BLOCK: Any SEMANTIC severity OR signal_type is INVARIANT_VIOLATION
    - WARN: Any PARAMETRIC severity OR non-empty violations
    - OK: NONE severity AND no violations

    This is a pure helper function with no side effects.

    Args:
        governance_signal: Curriculum governance signal
        drift_timeline: Optional list of drift timeline events for additional context

    Returns:
        CouncilStatus: "OK", "WARN", or "BLOCK"
    """
    # Check signal severity
    if governance_signal.severity == DriftSeverity.SEMANTIC:
        return "BLOCK"

    # Check signal type
    if governance_signal.signal_type == GovernanceSignalType.INVARIANT_VIOLATION:
        return "BLOCK"

    # Check for blocking violations
    if governance_signal.violations:
        blocking_codes = [v.code for v in governance_signal.violations if "SEMANTIC" in v.code or "REGRESS" in v.code]
        if blocking_codes:
            return "BLOCK"

    # Check hypothetical blocking
    if governance_signal.hypothetical:
        if not governance_signal.hypothetical.get("would_allow_transition", True):
            return "BLOCK"

    # Check drift timeline for semantic violations
    if drift_timeline:
        semantic_events = [e for e in drift_timeline if e.drift_severity == DriftSeverity.SEMANTIC]
        if semantic_events:
            return "BLOCK"

    # Check for WARN conditions
    if governance_signal.severity == DriftSeverity.PARAMETRIC:
        return "WARN"

    if governance_signal.violations:
        return "WARN"

    if governance_signal.status == DriftStatus.WARN:
        return "WARN"

    # Check timeline for parametric changes
    if drift_timeline:
        parametric_events = [e for e in drift_timeline if e.drift_severity == DriftSeverity.PARAMETRIC]
        if parametric_events:
            return "WARN"

    # All clear
    return "OK"


def council_classify_curriculum_from_dict(
    signal_dict: Dict[str, Any],
    timeline_dicts: Optional[List[Dict[str, Any]]] = None,
) -> CouncilStatus:
    """
    Classify curriculum status from serialized signal/timeline dictionaries.

    Convenience function for when signals are already serialized to dicts.

    Args:
        signal_dict: Serialized governance signal (from signal.to_dict())
        timeline_dicts: Optional list of serialized drift events

    Returns:
        CouncilStatus: "OK", "WARN", or "BLOCK"
    """
    # Extract severity
    severity_str = signal_dict.get("severity", "NONE")
    if severity_str == "SEMANTIC":
        return "BLOCK"

    # Check signal type
    signal_type = signal_dict.get("signal_type", "")
    if signal_type == "INVARIANT_VIOLATION":
        return "BLOCK"

    # Check violations
    violations = signal_dict.get("violations", [])
    if violations:
        blocking_codes = [v.get("code", "") for v in violations if "SEMANTIC" in v.get("code", "") or "REGRESS" in v.get("code", "")]
        if blocking_codes:
            return "BLOCK"

    # Check hypothetical
    hypothetical = signal_dict.get("hypothetical", {})
    if hypothetical and not hypothetical.get("would_allow_transition", True):
        return "BLOCK"

    # Check timeline
    if timeline_dicts:
        for event in timeline_dicts:
            if event.get("drift_severity") == "SEMANTIC":
                return "BLOCK"

    # Check WARN conditions
    if severity_str == "PARAMETRIC":
        return "WARN"

    if violations:
        return "WARN"

    status_str = signal_dict.get("status", "OK")
    if status_str == "WARN":
        return "WARN"

    if timeline_dicts:
        for event in timeline_dicts:
            if event.get("drift_severity") == "PARAMETRIC":
                return "WARN"

    return "OK"


# -----------------------------------------------------------------------------
# Batch Operations
# -----------------------------------------------------------------------------

def summarize_curriculum_for_council(
    governance_signals: List[GovernanceSignal],
    drift_events: Optional[List[DriftTimelineEvent]] = None,
) -> Dict[str, Any]:
    """
    Summarize multiple curriculum signals for council view.

    Aggregates signals and provides overall curriculum health assessment.

    Args:
        governance_signals: List of governance signals
        drift_events: Optional list of all drift events

    Returns:
        Summary dict suitable for council integration
    """
    if not governance_signals:
        return {
            "schema_version": "1.0.0",
            "signal_count": 0,
            "overall_status": "OK",
            "curriculum_health": "HEALTHY",
            "violations_total": 0,
            "semantic_violations": 0,
            "parametric_changes": 0,
        }

    # Classify each signal
    statuses = [council_classify_curriculum(s, drift_events) for s in governance_signals]

    # Determine overall status (worst-case)
    if "BLOCK" in statuses:
        overall_status: CouncilStatus = "BLOCK"
    elif "WARN" in statuses:
        overall_status = "WARN"
    else:
        overall_status = "OK"

    # Count violations
    all_violations = []
    for s in governance_signals:
        all_violations.extend(s.violations)

    semantic_count = sum(1 for s in governance_signals if s.severity == DriftSeverity.SEMANTIC)
    parametric_count = sum(1 for s in governance_signals if s.severity == DriftSeverity.PARAMETRIC)

    # Determine curriculum health
    if overall_status == "BLOCK":
        curriculum_health = "CRITICAL"
    elif overall_status == "WARN":
        curriculum_health = "DEGRADED"
    else:
        curriculum_health = "HEALTHY"

    return {
        "schema_version": "1.0.0",
        "signal_count": len(governance_signals),
        "overall_status": overall_status,
        "curriculum_health": curriculum_health,
        "violations_total": len(all_violations),
        "semantic_violations": semantic_count,
        "parametric_changes": parametric_count,
        "block_signals": statuses.count("BLOCK"),
        "warn_signals": statuses.count("WARN"),
        "ok_signals": statuses.count("OK"),
    }


# -----------------------------------------------------------------------------
# CTRPK (Curriculum Transition Requests Per 1K Cycles)
# -----------------------------------------------------------------------------

StatusLight = Literal["GREEN", "YELLOW", "RED"]
TrendDirection = Literal["IMPROVING", "STABLE", "DEGRADING"]


def compute_ctrpk(transition_requests: int, total_cycles: int) -> float:
    """
    Compute Curriculum Transition Requests Per 1K Cycles.

    CTRPK = (transition_requests / total_cycles) * 1000

    Args:
        transition_requests: Count of TRANSITION_REQUESTED governance signals
        total_cycles: Total derivation cycles in measurement window

    Returns:
        CTRPK value (float). Returns 0.0 if total_cycles is 0.
    """
    if total_cycles <= 0:
        return 0.0
    return (transition_requests / total_cycles) * 1000


def ctrpk_to_status_light(ctrpk: float) -> StatusLight:
    """
    Convert CTRPK value to status light color.

    Thresholds:
    - GREEN:  CTRPK < 1.0
    - YELLOW: CTRPK 1.0 - 5.0
    - RED:    CTRPK > 5.0

    Args:
        ctrpk: CTRPK value

    Returns:
        StatusLight: "GREEN", "YELLOW", or "RED"
    """
    if ctrpk < 1.0:
        return "GREEN"
    elif ctrpk <= 5.0:
        return "YELLOW"
    else:
        return "RED"


def compute_ctrpk_trend(
    ctrpk_1h: float,
    ctrpk_24h: float,
    delta_threshold: float = 0.5,
) -> TrendDirection:
    """
    Compute CTRPK trend direction.

    Trend is determined by comparing short-term (1h) vs long-term (24h) CTRPK:
    - IMPROVING: delta < -threshold (CTRPK decreasing)
    - DEGRADING: delta > threshold (CTRPK increasing)
    - STABLE: otherwise

    Args:
        ctrpk_1h: CTRPK over last 1 hour
        ctrpk_24h: CTRPK over last 24 hours
        delta_threshold: Threshold for trend detection (default 0.5)

    Returns:
        TrendDirection: "IMPROVING", "STABLE", or "DEGRADING"
    """
    delta = ctrpk_1h - ctrpk_24h

    if delta < -delta_threshold:
        return "IMPROVING"
    elif delta > delta_threshold:
        return "DEGRADING"
    else:
        return "STABLE"


def council_classify_ctrpk(
    ctrpk: float,
    semantic_violations: int = 0,
    blocked_requests: int = 0,
    trend_direction: TrendDirection = "STABLE",
) -> CouncilStatus:
    """
    Classify CTRPK for uplift council.

    Classification rules:
    - BLOCK: semantic_violations > 0, CTRPK > 5.0, or (DEGRADING + CTRPK > 3.0)
    - WARN: CTRPK > 1.0 or blocked_requests > 0
    - OK: otherwise

    Args:
        ctrpk: CTRPK value
        semantic_violations: Count of SEMANTIC severity violations
        blocked_requests: Count of transition requests that would be blocked
        trend_direction: Current trend direction

    Returns:
        CouncilStatus: "OK", "WARN", or "BLOCK"
    """
    # Hard blocks
    if semantic_violations > 0:
        return "BLOCK"
    if ctrpk > 5.0:
        return "BLOCK"
    if trend_direction == "DEGRADING" and ctrpk > 3.0:
        return "BLOCK"

    # Warnings
    if ctrpk > 1.0:
        return "WARN"
    if blocked_requests > 0:
        return "WARN"

    return "OK"


def build_ctrpk_summary(
    transition_requests: int,
    total_cycles: int,
    measurement_window_minutes: int = 60,
    blocked_requests: int = 0,
    successful_transitions: int = 0,
    ctrpk_1h: Optional[float] = None,
    ctrpk_24h: Optional[float] = None,
    semantic_violations: int = 0,
) -> Dict[str, Any]:
    """
    Build a complete CTRPK summary for tiles and evidence.

    Args:
        transition_requests: Count of TRANSITION_REQUESTED signals
        total_cycles: Total cycles in measurement window
        measurement_window_minutes: Duration of window in minutes
        blocked_requests: Requests that would be blocked
        successful_transitions: Transitions that passed all gates
        ctrpk_1h: Optional 1-hour CTRPK for trend
        ctrpk_24h: Optional 24-hour CTRPK for trend
        semantic_violations: Count of semantic violations

    Returns:
        Complete CTRPK summary dict
    """
    ctrpk = compute_ctrpk(transition_requests, total_cycles)
    status_light = ctrpk_to_status_light(ctrpk)

    # Compute trend if historical data available
    trend_direction: TrendDirection = "STABLE"
    if ctrpk_1h is not None and ctrpk_24h is not None:
        trend_direction = compute_ctrpk_trend(ctrpk_1h, ctrpk_24h)

    # Compute council status
    council_status = council_classify_ctrpk(
        ctrpk=ctrpk,
        semantic_violations=semantic_violations,
        blocked_requests=blocked_requests,
        trend_direction=trend_direction,
    )

    summary: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "ctrpk": ctrpk,
        "status_light": status_light,
        "transition_requests": transition_requests,
        "total_cycles": total_cycles,
        "measurement_window_minutes": measurement_window_minutes,
        "blocked_requests": blocked_requests,
        "successful_transitions": successful_transitions,
        "council_status": council_status,
        "trend": {
            "direction": trend_direction,
        },
    }

    # Add historical data if available
    if ctrpk_1h is not None:
        summary["trend"]["ctrpk_1h"] = ctrpk_1h
    if ctrpk_24h is not None:
        summary["trend"]["ctrpk_24h"] = ctrpk_24h
        summary["trend"]["delta_vs_baseline"] = ctrpk - ctrpk_24h

    return summary


def build_ctrpk_compact(
    transition_requests: int,
    total_cycles: int,
    trend_direction: TrendDirection = "STABLE",
    semantic_violations: int = 0,
    blocked_requests: int = 0,
) -> Dict[str, Any]:
    """
    Build compact CTRPK field for evidence packs.

    Args:
        transition_requests: Count of TRANSITION_REQUESTED signals
        total_cycles: Total cycles in measurement window
        trend_direction: Trend direction
        semantic_violations: Count of semantic violations
        blocked_requests: Count of blocked requests

    Returns:
        Compact CTRPK dict for evidence["governance"]["curriculum"]["ctrpk"]
    """
    ctrpk = compute_ctrpk(transition_requests, total_cycles)
    council_status = council_classify_ctrpk(
        ctrpk=ctrpk,
        semantic_violations=semantic_violations,
        blocked_requests=blocked_requests,
        trend_direction=trend_direction,
    )

    return {
        "value": ctrpk,
        "status": council_status,
        "window_cycles": total_cycles,
        "transition_requests": transition_requests,
        "trend": trend_direction,
    }


def attach_ctrpk_to_evidence(
    evidence: Dict[str, Any],
    transition_requests: int,
    total_cycles: int,
    trend_direction: TrendDirection = "STABLE",
    semantic_violations: int = 0,
    blocked_requests: int = 0,
) -> Dict[str, Any]:
    """
    Attach CTRPK compact field to evidence pack.

    Stores under evidence["governance"]["curriculum"]["ctrpk"].

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - CTRPK is observational only

    Args:
        evidence: Evidence pack dictionary
        transition_requests: Count of TRANSITION_REQUESTED signals
        total_cycles: Total cycles in measurement window
        trend_direction: Trend direction
        semantic_violations: Count of semantic violations
        blocked_requests: Count of blocked requests

    Returns:
        Updated evidence with CTRPK compact field
    """
    ctrpk_compact = build_ctrpk_compact(
        transition_requests=transition_requests,
        total_cycles=total_cycles,
        trend_direction=trend_direction,
        semantic_violations=semantic_violations,
        blocked_requests=blocked_requests,
    )

    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)

    # Ensure governance.curriculum path exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])

    if "curriculum" not in updated_evidence["governance"]:
        updated_evidence["governance"]["curriculum"] = {}
    else:
        updated_evidence["governance"]["curriculum"] = dict(
            updated_evidence["governance"]["curriculum"]
        )

    updated_evidence["governance"]["curriculum"]["ctrpk"] = ctrpk_compact

    return updated_evidence


# -----------------------------------------------------------------------------
# Module Exports
# -----------------------------------------------------------------------------

__all__ = [
    # P3 Stability Hook
    "attach_curriculum_governance_to_p3",
    "attach_curriculum_timeline_to_p3",
    # Evidence Attachment
    "attach_curriculum_to_evidence",
    # Council Classification
    "council_classify_curriculum",
    "council_classify_curriculum_from_dict",
    "summarize_curriculum_for_council",
    # CTRPK
    "compute_ctrpk",
    "ctrpk_to_status_light",
    "compute_ctrpk_trend",
    "council_classify_ctrpk",
    "build_ctrpk_summary",
    "build_ctrpk_compact",
    "attach_ctrpk_to_evidence",
]
