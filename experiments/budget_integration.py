#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Budget ↔ Metric Integration & Uplift Readiness
═══════════════════════════════════════════════════════════════════════════════

Connects budget observability with metric conformance and uplift readiness.
Read-only analysis tool for governance and decision support.

PHASE III — BUDGET↔METRIC INTEGRATION & UPLIFT READINESS
PHASE IV — CROSS-EXPERIMENT BUDGET GOVERNANCE & UPLIFT SCHEDULING

This module provides:
    1. Budget ↔ Metric Joint View: Cross-reference budget health with metric conformance
    2. Uplift Readiness Signal: Determine if budget state allows uplift experiments
    3. Global Health Summary: Aggregate budget health trends across runs
    4. Cross-Experiment Budget Governance: Aggregate budget health across multiple runs
    5. Uplift Scheduling Advisor: Recommend when to run uplift experiments
    6. Director Panel: High-level budget status dashboard

Usage:
    from experiments.budget_integration import (
        build_budget_metric_joint_view,
        summarize_budget_for_uplift,
        summarize_budget_for_global_health,
    )
    
    # Joint view
    joint = build_budget_metric_joint_view(budget_summaries, metric_snapshots)
    
    # Uplift readiness
    readiness = summarize_budget_for_uplift(trend_report, joint)
    
    # Global health
    global_health = summarize_budget_for_global_health(trend_report)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from experiments.summarize_budget_usage import (
    BudgetSummary,
    BudgetHealthStatus,
    BudgetHealthResult,
    classify_budget_health,
)
from experiments.budget_trends import (
    TrendDirection,
    TrendReport,
    SliceTrend,
)


# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================


class UpliftReadinessStatus(Enum):
    """
    Uplift readiness classification.
    
    OK:    Budget conditions support uplift experiments
    WARN:  Some slices at risk, proceed with caution
    BLOCK: Critical slices STARVED, uplift experiments unreliable
    """
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class MetricLevel(Enum):
    """
    Metric conformance levels (from MaaS/governance).
    
    L0: No conformance data
    L1: Basic conformance
    L2: Intermediate conformance
    L3: Full conformance
    PASS: Simple pass indicator
    FAIL: Simple fail indicator
    """
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


# Slices considered critical for uplift experiments
# If these are STARVED, uplift should be blocked
CRITICAL_UPLIFT_SLICES: Set[str] = {
    "slice_uplift_goal",
    "slice_uplift_sparse",
    "slice_uplift_tree",
    "slice_uplift_dependency",
}


# =============================================================================
# TASK 1: Budget ↔ Metric Joint View
# =============================================================================


@dataclass
class SliceJointStatus:
    """
    Joint budget and metric status for a single slice.
    
    Attributes:
        slice_name: Name of the slice
        budget_status: Budget health (SAFE/TIGHT/STARVED/INVALID)
        metric_status: Metric conformance level
        budget_metrics: Raw budget metrics
        metric_details: Raw metric details (if available)
        flags: List of concern flags for this slice
    """
    slice_name: str
    budget_status: str
    metric_status: str
    budget_metrics: Dict[str, float] = field(default_factory=dict)
    metric_details: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slice_name": self.slice_name,
            "budget_status": self.budget_status,
            "metric_status": self.metric_status,
            "budget_metrics": self.budget_metrics,
            "metric_details": self.metric_details,
            "flags": self.flags,
        }


def _parse_metric_status(metric_data: Optional[Dict[str, Any]]) -> str:
    """
    Parse metric conformance status from metric snapshot data.
    
    Supports multiple formats:
        - {"level": "L2"}
        - {"status": "PASS"}
        - {"conformance": {"level": "L1"}}
    """
    if metric_data is None:
        return MetricLevel.UNKNOWN.value
    
    # Direct level
    if "level" in metric_data:
        level = metric_data["level"]
        if level in [l.value for l in MetricLevel]:
            return level
    
    # Direct status
    if "status" in metric_data:
        status = metric_data["status"]
        if status in [l.value for l in MetricLevel]:
            return status
    
    # Nested conformance
    if "conformance" in metric_data and isinstance(metric_data["conformance"], dict):
        return _parse_metric_status(metric_data["conformance"])
    
    return MetricLevel.UNKNOWN.value


def _identify_flags(budget_status: str, metric_status: str) -> List[str]:
    """
    Identify concern flags based on budget and metric status combination.
    
    Flags identify problematic combinations:
        - STARVED + high metric claims (unreliable data)
        - SAFE budget but weak metrics (missed opportunity)
    """
    flags = []
    
    # STARVED + high metric claims = unreliable
    if budget_status == "STARVED":
        if metric_status in ["L2", "L3", "PASS"]:
            flags.append("STARVED_WITH_HIGH_METRICS")
    
    # SAFE budget but weak metrics = opportunity
    if budget_status == "SAFE":
        if metric_status in ["L0", "L1", "FAIL", "UNKNOWN"]:
            flags.append("SAFE_WITH_WEAK_METRICS")
    
    # TIGHT + high metrics = marginal reliability
    if budget_status == "TIGHT":
        if metric_status in ["L2", "L3"]:
            flags.append("TIGHT_WITH_HIGH_METRICS")
    
    return flags


def build_budget_metric_joint_view(
    budget_summaries: List[BudgetSummary],
    metric_conformance_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build a joint view of budget health and metric conformance per slice.
    
    This function cross-references budget health status with metric conformance
    levels to identify problematic combinations:
        - Slices with STARVED budget + high metric claims (unreliable data)
        - Slices with SAFE budget but weak metrics (missed opportunity)
    
    Args:
        budget_summaries: List of BudgetSummary objects (one per slice)
        metric_conformance_snapshots: Optional dict mapping slice_name to metric data
                                       Format: {"slice_name": {"level": "L2", ...}}
    
    Returns:
        Dictionary with:
            - "slices": List of SliceJointStatus.to_dict()
            - "summary": Aggregate counts
            - "concerns": List of flagged slice names
    
    Example:
        >>> summaries = [parse_log_file(path) for path in log_paths]
        >>> metrics = {"slice_uplift_goal": {"level": "L2"}}
        >>> joint = build_budget_metric_joint_view(summaries, metrics)
        >>> print(joint["concerns"])
        ["slice_uplift_goal"]  # if STARVED + L2
    """
    if metric_conformance_snapshots is None:
        metric_conformance_snapshots = {}
    
    slice_statuses: List[SliceJointStatus] = []
    
    for summary in budget_summaries:
        slice_name = summary.slice_name or "unknown"
        
        # Get budget health
        health = classify_budget_health(summary)
        budget_status = health.status.value
        
        # Get metric status
        metric_data = metric_conformance_snapshots.get(slice_name)
        metric_status = _parse_metric_status(metric_data)
        
        # Identify flags
        flags = _identify_flags(budget_status, metric_status)
        
        slice_statuses.append(SliceJointStatus(
            slice_name=slice_name,
            budget_status=budget_status,
            metric_status=metric_status,
            budget_metrics=health.metrics,
            metric_details=metric_data or {},
            flags=flags,
        ))
    
    # Build summary
    budget_counts = {
        "SAFE": sum(1 for s in slice_statuses if s.budget_status == "SAFE"),
        "TIGHT": sum(1 for s in slice_statuses if s.budget_status == "TIGHT"),
        "STARVED": sum(1 for s in slice_statuses if s.budget_status == "STARVED"),
        "INVALID": sum(1 for s in slice_statuses if s.budget_status == "INVALID"),
    }
    
    metric_counts = {}
    for s in slice_statuses:
        metric_counts[s.metric_status] = metric_counts.get(s.metric_status, 0) + 1
    
    # Identify all flagged slices
    flagged_slices = [s.slice_name for s in slice_statuses if s.flags]
    starved_with_high = [s.slice_name for s in slice_statuses if "STARVED_WITH_HIGH_METRICS" in s.flags]
    safe_with_weak = [s.slice_name for s in slice_statuses if "SAFE_WITH_WEAK_METRICS" in s.flags]
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "slices": [s.to_dict() for s in slice_statuses],
        "summary": {
            "total_slices": len(slice_statuses),
            "budget_counts": budget_counts,
            "metric_counts": metric_counts,
        },
        "concerns": {
            "flagged_slices": flagged_slices,
            "starved_with_high_metrics": starved_with_high,
            "safe_with_weak_metrics": safe_with_weak,
        },
    }


# =============================================================================
# TASK 2: Uplift Readiness Signal
# =============================================================================


def summarize_budget_for_uplift(
    budget_trend: Optional[TrendReport],
    joint_view: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Determine if budget state supports uplift experiments.
    
    Evaluates budget health across critical uplift slices to determine
    whether uplift experiments can proceed reliably.
    
    Decision logic:
        BLOCK: Any critical slice is STARVED with DEGRADING trend
        WARN:  Any critical slice is TIGHT or has flags
        OK:    All critical slices are SAFE
    
    Args:
        budget_trend: TrendReport from budget_trends.py (or None)
        joint_view: Output from build_budget_metric_joint_view (or None)
    
    Returns:
        Dictionary with:
            - "budget_ready_for_uplift": bool
            - "at_risk_slices": List of slice names at risk
            - "status": OK|WARN|BLOCK
            - "reasons": List of human-readable reasons
    
    Example:
        >>> trend = analyze_trends([path1, path2])
        >>> joint = build_budget_metric_joint_view(summaries, metrics)
        >>> readiness = summarize_budget_for_uplift(trend, joint)
        >>> if readiness["status"] == "BLOCK":
        ...     print("Uplift experiments unreliable")
    """
    at_risk_slices: List[str] = []
    reasons: List[str] = []
    status = UpliftReadinessStatus.OK
    
    # Extract slice statuses from joint view
    slice_statuses: Dict[str, str] = {}
    slice_flags: Dict[str, List[str]] = {}
    
    if joint_view is not None:
        for s in joint_view.get("slices", []):
            slice_statuses[s["slice_name"]] = s["budget_status"]
            slice_flags[s["slice_name"]] = s.get("flags", [])
    
    # Extract trends
    slice_trends: Dict[str, str] = {}
    if budget_trend is not None:
        for st in budget_trend.slices:
            slice_trends[st.slice_name] = st.trend.value
    
    # Evaluate each critical slice
    for slice_name in CRITICAL_UPLIFT_SLICES:
        budget_status = slice_statuses.get(slice_name, "UNKNOWN")
        trend = slice_trends.get(slice_name, "UNKNOWN")
        flags = slice_flags.get(slice_name, [])
        
        # BLOCK conditions
        if budget_status == "STARVED":
            at_risk_slices.append(slice_name)
            if trend == "DEGRADING":
                status = UpliftReadinessStatus.BLOCK
                reasons.append(f"{slice_name}: STARVED + DEGRADING trend")
            elif status != UpliftReadinessStatus.BLOCK:
                status = UpliftReadinessStatus.WARN
                reasons.append(f"{slice_name}: STARVED budget")
        
        # WARN conditions
        elif budget_status == "TIGHT":
            if status == UpliftReadinessStatus.OK:
                status = UpliftReadinessStatus.WARN
            at_risk_slices.append(slice_name)
            reasons.append(f"{slice_name}: TIGHT budget")
        
        # Flag-based concerns
        if "STARVED_WITH_HIGH_METRICS" in flags:
            if status == UpliftReadinessStatus.OK:
                status = UpliftReadinessStatus.WARN
            reasons.append(f"{slice_name}: STARVED with high metric claims (unreliable)")
    
    # Check for non-critical slices with degrading trends
    if budget_trend is not None:
        degrading_count = sum(1 for st in budget_trend.slices if st.trend == TrendDirection.DEGRADING)
        if degrading_count > len(budget_trend.slices) // 2:
            if status == UpliftReadinessStatus.OK:
                status = UpliftReadinessStatus.WARN
            reasons.append(f"Majority of slices ({degrading_count}/{len(budget_trend.slices)}) showing DEGRADING trend")
    
    # Default reason if OK
    if status == UpliftReadinessStatus.OK and not reasons:
        reasons.append("All critical uplift slices have healthy budget")
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "budget_ready_for_uplift": status != UpliftReadinessStatus.BLOCK,
        "at_risk_slices": list(set(at_risk_slices)),
        "status": status.value,
        "reasons": reasons,
        "critical_slices_checked": list(CRITICAL_UPLIFT_SLICES),
    }


# =============================================================================
# TASK 3: Global Health Budget Summary
# =============================================================================


def summarize_budget_for_global_health(
    budget_trend: Optional[TrendReport],
    recent_runs: int = 5,
) -> Dict[str, Any]:
    """
    Summarize budget health trends for global health dashboard.
    
    Aggregates budget health across all slices and runs to provide
    a high-level health signal.
    
    Args:
        budget_trend: TrendReport from budget_trends.py (or None)
        recent_runs: Number of recent runs to consider for status counts
    
    Returns:
        Dictionary with:
            - "trend_status": IMPROVING/STABLE/DEGRADING/UNKNOWN
            - "slice_counts": {"SAFE": N, "TIGHT": N, "STARVED": N}
            - "trend_counts": {"IMPROVING": N, "STABLE": N, "DEGRADING": N}
            - "health_score": 0-100 numeric score
    
    Example:
        >>> trend = analyze_trends([path1, path2, path3])
        >>> health = summarize_budget_for_global_health(trend)
        >>> print(f"Global trend: {health['trend_status']}")
    """
    if budget_trend is None or not budget_trend.slices:
        return {
            "phase": "PHASE II — NOT USED IN PHASE I",
            "trend_status": TrendDirection.UNKNOWN.value,
            "slice_counts": {"SAFE": 0, "TIGHT": 0, "STARVED": 0, "INVALID": 0},
            "trend_counts": {"IMPROVING": 0, "STABLE": 0, "DEGRADING": 0, "UNKNOWN": 0},
            "health_score": 0,
            "total_slices": 0,
            "total_runs": 0,
        }
    
    # Count slice trends
    trend_counts = {
        "IMPROVING": 0,
        "STABLE": 0,
        "DEGRADING": 0,
        "UNKNOWN": 0,
    }
    for st in budget_trend.slices:
        trend_counts[st.trend.value] += 1
    
    # Count latest status per slice (last status in sequence)
    slice_counts = {
        "SAFE": 0,
        "TIGHT": 0,
        "STARVED": 0,
        "INVALID": 0,
    }
    for st in budget_trend.slices:
        if st.status_sequence:
            last_status = st.status_sequence[-1]
            if last_status in slice_counts:
                slice_counts[last_status] += 1
    
    # Determine overall trend status
    total_slices = len(budget_trend.slices)
    improving = trend_counts["IMPROVING"]
    degrading = trend_counts["DEGRADING"]
    stable = trend_counts["STABLE"]
    
    if improving > degrading and improving >= total_slices // 3:
        overall_trend = TrendDirection.IMPROVING
    elif degrading > improving and degrading >= total_slices // 3:
        overall_trend = TrendDirection.DEGRADING
    elif stable >= total_slices // 2:
        overall_trend = TrendDirection.STABLE
    else:
        # Mixed signals
        if improving > degrading:
            overall_trend = TrendDirection.IMPROVING
        elif degrading > improving:
            overall_trend = TrendDirection.DEGRADING
        else:
            overall_trend = TrendDirection.STABLE
    
    # Calculate health score (0-100)
    # Formula: 100 * (SAFE + 0.5*TIGHT) / total - penalty for STARVED
    if total_slices > 0:
        safe_weight = slice_counts["SAFE"]
        tight_weight = slice_counts["TIGHT"] * 0.5
        starved_penalty = slice_counts["STARVED"] * 0.5
        health_score = max(0, min(100, 100 * (safe_weight + tight_weight - starved_penalty) / total_slices))
    else:
        health_score = 0
    
    # Trend bonus/penalty
    if overall_trend == TrendDirection.IMPROVING:
        health_score = min(100, health_score + 10)
    elif overall_trend == TrendDirection.DEGRADING:
        health_score = max(0, health_score - 10)
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "trend_status": overall_trend.value,
        "slice_counts": slice_counts,
        "trend_counts": trend_counts,
        "health_score": round(health_score, 1),
        "total_slices": total_slices,
        "total_runs": len(budget_trend.inputs),
    }


# =============================================================================
# PHASE IV: CROSS-EXPERIMENT BUDGET GOVERNANCE & UPLIFT SCHEDULING
# =============================================================================


def build_cross_experiment_budget_view(
    run_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a cross-experiment budget governance view.
    
    Aggregates budget health across multiple experiment runs to identify:
        - Slices with consistent budget health patterns
        - Slices frequently STARVED across runs
        - Runs with BLOCK uplift status
    
    Args:
        run_summaries: Sequence of run summary dictionaries. Each dict should have:
            - "run_id": Unique identifier for the run
            - "slices": List of slice data with "slice_name" and "budget_status"
            - "uplift_readiness": Optional dict with "status" (OK/WARN/BLOCK)
    
    Returns:
        Dictionary with:
            - "experiments_count": Total number of runs analyzed
            - "slices_frequently_starved": List of slice names starved in >50% of runs
            - "per_slice": Dict mapping slice_name to:
                - "classification_distribution": {"SAFE": N, "TIGHT": N, "STARVED": N}
                - "block_count": Number of runs with BLOCK status for this slice
                - "total_runs": Number of runs this slice appeared in
            - "global_block_count": Total number of runs with BLOCK uplift status
    
    Example:
        >>> runs = [
        ...     {"run_id": "run1", "slices": [{"slice_name": "slice_uplift_goal", "budget_status": "SAFE"}]},
        ...     {"run_id": "run2", "slices": [{"slice_name": "slice_uplift_goal", "budget_status": "STARVED"}]},
        ... ]
        >>> view = build_cross_experiment_budget_view(runs)
        >>> print(view["per_slice"]["slice_uplift_goal"]["classification_distribution"])
        {"SAFE": 1, "TIGHT": 0, "STARVED": 1}
    """
    experiments_count = len(run_summaries)
    
    # Aggregate per-slice data
    per_slice: Dict[str, Dict[str, Any]] = {}
    global_block_count = 0
    
    for run in run_summaries:
        run_id = run.get("run_id", "unknown")
        slices = run.get("slices", [])
        uplift_status = run.get("uplift_readiness", {}).get("status", "OK")
        
        # Count BLOCK runs
        if uplift_status == "BLOCK":
            global_block_count += 1
        
        # Process each slice in this run
        for slice_data in slices:
            slice_name = slice_data.get("slice_name", "unknown")
            budget_status = slice_data.get("budget_status", "UNKNOWN")
            
            if slice_name not in per_slice:
                per_slice[slice_name] = {
                    "classification_distribution": {
                        "SAFE": 0,
                        "TIGHT": 0,
                        "STARVED": 0,
                        "INVALID": 0,
                    },
                    "block_count": 0,
                    "total_runs": 0,
                }
            
            # Update classification distribution
            if budget_status in per_slice[slice_name]["classification_distribution"]:
                per_slice[slice_name]["classification_distribution"][budget_status] += 1
            
            # Count BLOCK runs for this slice
            if uplift_status == "BLOCK":
                # Check if this slice was in the at_risk_slices
                at_risk = run.get("uplift_readiness", {}).get("at_risk_slices", [])
                if slice_name in at_risk:
                    per_slice[slice_name]["block_count"] += 1
            
            per_slice[slice_name]["total_runs"] += 1
    
    # Identify slices frequently starved (>50% of runs)
    slices_frequently_starved: List[str] = []
    for slice_name, data in per_slice.items():
        total = data["total_runs"]
        starved_count = data["classification_distribution"].get("STARVED", 0)
        if total > 0 and (starved_count / total) > 0.5:
            slices_frequently_starved.append(slice_name)
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "experiments_count": experiments_count,
        "slices_frequently_starved": sorted(slices_frequently_starved),
        "per_slice": {
            slice_name: {
                "classification_distribution": data["classification_distribution"],
                "block_count": data["block_count"],
                "total_runs": data["total_runs"],
                "starved_percentage": round(
                    100.0 * data["classification_distribution"].get("STARVED", 0) / data["total_runs"]
                    if data["total_runs"] > 0 else 0.0,
                    1,
                ),
            }
            for slice_name, data in sorted(per_slice.items())
        },
        "global_block_count": global_block_count,
    }


def plan_uplift_runs(cross_view: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate uplift scheduling recommendations based on cross-experiment budget view.
    
    Analyzes budget health patterns to recommend:
        - Slices ready for uplift experiments
        - Slices needing budget tuning before uplift
        - Scheduling hints for experiment planning
    
    Args:
        cross_view: Output from build_cross_experiment_budget_view()
    
    Returns:
        Dictionary with:
            - "slices_ready_for_uplift": List of slice names with consistent SAFE status
            - "slices_needing_budget_tuning": List of slice names frequently STARVED
            - "schedule_recommendations": List of neutral scheduling hints
    
    Example:
        >>> view = build_cross_experiment_budget_view(runs)
        >>> plan = plan_uplift_runs(view)
        >>> print(plan["slices_ready_for_uplift"])
        ["slice_uplift_goal"]
    """
    slices_ready_for_uplift: List[str] = []
    slices_needing_budget_tuning: List[str] = []
    schedule_recommendations: List[str] = []
    
    per_slice = cross_view.get("per_slice", {})
    experiments_count = cross_view.get("experiments_count", 0)
    
    # Threshold: slice is "ready" if SAFE in >=70% of runs
    ready_threshold = 0.7
    # Threshold: slice "needs tuning" if STARVED in >50% of runs
    tuning_threshold = 0.5
    
    for slice_name, data in per_slice.items():
        total_runs = data.get("total_runs", 0)
        if total_runs == 0:
            continue
        
        dist = data.get("classification_distribution", {})
        safe_count = dist.get("SAFE", 0)
        starved_count = dist.get("STARVED", 0)
        
        safe_ratio = safe_count / total_runs
        starved_ratio = starved_count / total_runs
        
        # Ready for uplift: consistently SAFE
        if safe_ratio >= ready_threshold:
            slices_ready_for_uplift.append(slice_name)
            schedule_recommendations.append(
                f"Run uplift on {slice_name} in next window"
            )
        
        # Needs tuning: frequently STARVED
        elif starved_ratio > tuning_threshold:
            slices_needing_budget_tuning.append(slice_name)
            schedule_recommendations.append(
                f"Delay uplift on {slice_name} until budget reconfiguration"
            )
        
        # Marginal: TIGHT in most runs
        elif dist.get("TIGHT", 0) / total_runs >= 0.5:
            schedule_recommendations.append(
                f"Consider budget adjustment for {slice_name} before uplift"
            )
    
    # Global recommendation if many blocks
    global_block_count = cross_view.get("global_block_count", 0)
    if global_block_count > experiments_count * 0.3:  # >30% of runs blocked
        schedule_recommendations.append(
            "High frequency of BLOCK status across runs — review budget configuration"
        )
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "slices_ready_for_uplift": sorted(slices_ready_for_uplift),
        "slices_needing_budget_tuning": sorted(slices_needing_budget_tuning),
        "schedule_recommendations": schedule_recommendations,
        "total_slices_analyzed": len(per_slice),
    }


def build_budget_director_panel(
    cross_view: Dict[str, Any],
    uplift_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a high-level budget status panel for director-level decision making.
    
    Provides a simplified, executive-friendly view of budget health and its
    impact on uplift experiment readiness.
    
    Args:
        cross_view: Output from build_cross_experiment_budget_view()
        uplift_plan: Output from plan_uplift_runs()
    
    Returns:
        Dictionary with:
            - "budget_status_light": GREEN | YELLOW | RED
            - "critical_slices_at_risk": List of critical slice names at risk
            - "ready_slices": List of slice names ready for uplift
            - "message": Short neutral text summarizing budget's role in uplift posture
    
    Status Light Logic:
        GREEN: No critical slices frequently starved, <20% runs blocked
        YELLOW: Some critical slices at risk, 20-50% runs blocked
        RED: Critical slices frequently starved, >50% runs blocked
    """
    # Extract data
    slices_frequently_starved = set(cross_view.get("slices_frequently_starved", []))
    global_block_count = cross_view.get("global_block_count", 0)
    experiments_count = cross_view.get("experiments_count", 1)
    block_percentage = (global_block_count / experiments_count) * 100.0 if experiments_count > 0 else 0.0
    
    slices_ready = set(uplift_plan.get("slices_ready_for_uplift", []))
    slices_needing_tuning = set(uplift_plan.get("slices_needing_budget_tuning", []))
    
    # Identify critical slices at risk
    critical_at_risk = [
        slice_name
        for slice_name in CRITICAL_UPLIFT_SLICES
        if slice_name in slices_frequently_starved or slice_name in slices_needing_tuning
    ]
    
    # Determine status light
    critical_starved = any(s in slices_frequently_starved for s in CRITICAL_UPLIFT_SLICES)
    
    if not critical_starved and block_percentage < 20.0:
        status_light = "GREEN"
    elif critical_starved or block_percentage >= 50.0:
        status_light = "RED"
    else:
        # YELLOW: block_percentage in [20, 50) OR non-critical slices starved
        status_light = "YELLOW"
    
    # Build message
    ready_count = len(slices_ready)
    at_risk_count = len(critical_at_risk)
    
    if status_light == "GREEN":
        message = (
            f"Budget health supports uplift experiments. "
            f"{ready_count} slice(s) ready for uplift. "
            f"Block rate: {block_percentage:.1f}%."
        )
    elif status_light == "YELLOW":
        message = (
            f"Budget health requires attention. "
            f"{at_risk_count} critical slice(s) at risk. "
            f"{ready_count} slice(s) ready. "
            f"Block rate: {block_percentage:.1f}%."
        )
    else:  # RED
        message = (
            f"Budget health limits uplift experiment reliability. "
            f"{at_risk_count} critical slice(s) frequently starved. "
            f"Block rate: {block_percentage:.1f}%. "
            f"Budget reconfiguration recommended."
        )
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "budget_status_light": status_light,
        "critical_slices_at_risk": sorted(critical_at_risk),
        "ready_slices": sorted(list(slices_ready)),
        "message": message,
        "summary": {
            "experiments_analyzed": experiments_count,
            "block_percentage": round(block_percentage, 1),
            "slices_frequently_starved_count": len(slices_frequently_starved),
            "ready_slices_count": len(slices_ready),
        },
    }


# =============================================================================
# UPLIFT COUNCIL: MULTI-DIMENSIONAL UPLIFT DECISION MAKING
# =============================================================================


class UpliftCouncilStatus(Enum):
    """
    Uplift Council decision status.
    
    OK:    All dimensions support uplift experiments
    WARN:  Non-critical slices at risk, proceed with caution
    BLOCK: Critical slices blocked by one or more dimensions
    """
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


def build_uplift_council_view(
    budget_cross_view: Dict[str, Any],
    perf_trend: Optional[Dict[str, Any]] = None,
    metric_conformance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a unified uplift council view combining budget, performance, and metrics.
    
    The Uplift Council makes multi-dimensional decisions about whether uplift
    experiments can proceed reliably. A slice is blocked if ANY dimension
    (budget, performance, or metrics) indicates it should not proceed.
    
    Args:
        budget_cross_view: Output from build_cross_experiment_budget_view()
        perf_trend: Optional performance trend data. Expected format:
                    {"slices": [{"slice_name": "...", "status": "OK|WARN|BLOCK", ...}]}
        metric_conformance: Optional metric conformance data. Expected format:
                           {"slices": [{"slice_name": "...", "ready": bool, ...}]}
    
    Returns:
        Dictionary with:
            - "slices_ready_for_uplift": List of slice names ready across all dimensions
            - "slices_blocked_by_budget": List of slice names blocked by budget
            - "slices_blocked_by_perf": List of slice names blocked by performance
            - "slices_blocked_by_metrics": List of slice names blocked by metrics
            - "council_status": OK|WARN|BLOCK
            - "per_slice": Detailed status per slice
    
    Council Rules:
        BLOCK: Any critical slice blocked by ANY dimension
        WARN:  Non-critical slices at risk
        OK:    All critical slices ready across all dimensions
    """
    # Extract budget data
    slices_frequently_starved = set(budget_cross_view.get("slices_frequently_starved", []))
    budget_per_slice = budget_cross_view.get("per_slice", {})
    
    # Extract performance data
    perf_slices: Dict[str, Dict[str, Any]] = {}
    if perf_trend is not None:
        for perf_slice in perf_trend.get("slices", []):
            slice_name = perf_slice.get("slice_name", "unknown")
            perf_slices[slice_name] = perf_slice
    
    # Extract metrics data
    metrics_slices: Dict[str, Dict[str, Any]] = {}
    if metric_conformance is not None:
        for metric_slice in metric_conformance.get("slices", []):
            slice_name = metric_slice.get("slice_name", "unknown")
            metrics_slices[slice_name] = metric_slice
    
    # Analyze each slice
    slices_ready: List[str] = []
    slices_blocked_by_budget: List[str] = []
    slices_blocked_by_perf: List[str] = []
    slices_blocked_by_metrics: List[str] = []
    per_slice_status: Dict[str, Dict[str, Any]] = {}
    
    # Get all slice names from all sources
    all_slice_names = set()
    all_slice_names.update(budget_per_slice.keys())
    all_slice_names.update(perf_slices.keys())
    all_slice_names.update(metrics_slices.keys())
    
    for slice_name in all_slice_names:
        slice_status = {
            "budget": "UNKNOWN",
            "perf": "UNKNOWN",
            "metrics": "UNKNOWN",
            "overall": "UNKNOWN",
        }
        
        # Budget dimension
        if slice_name in slices_frequently_starved:
            slice_status["budget"] = "BLOCK"
            slices_blocked_by_budget.append(slice_name)
        elif slice_name in budget_per_slice:
            dist = budget_per_slice[slice_name].get("classification_distribution", {})
            safe_ratio = dist.get("SAFE", 0) / max(budget_per_slice[slice_name].get("total_runs", 1), 1)
            if safe_ratio >= 0.7:
                slice_status["budget"] = "OK"
            elif safe_ratio >= 0.5:
                slice_status["budget"] = "WARN"
            else:
                slice_status["budget"] = "BLOCK"
                slices_blocked_by_budget.append(slice_name)
        else:
            slice_status["budget"] = "OK"  # No data = assume OK
        
        # Performance dimension
        if slice_name in perf_slices:
            perf_status = perf_slices[slice_name].get("status", "UNKNOWN")
            slice_status["perf"] = perf_status
            if perf_status == "BLOCK":
                slices_blocked_by_perf.append(slice_name)
        else:
            slice_status["perf"] = "OK"  # No data = assume OK
        
        # Metrics dimension
        if slice_name in metrics_slices:
            metrics_ready = metrics_slices[slice_name].get("ready", True)
            if not metrics_ready:
                slice_status["metrics"] = "BLOCK"
                slices_blocked_by_metrics.append(slice_name)
            else:
                slice_status["metrics"] = "OK"
        else:
            slice_status["metrics"] = "OK"  # No data = assume OK
        
        # Overall status (worst case wins)
        statuses = [slice_status["budget"], slice_status["perf"], slice_status["metrics"]]
        if "BLOCK" in statuses:
            slice_status["overall"] = "BLOCK"
        elif "WARN" in statuses:
            slice_status["overall"] = "WARN"
        else:
            slice_status["overall"] = "OK"
            slices_ready.append(slice_name)
        
        per_slice_status[slice_name] = slice_status
    
    # Determine council status
    critical_blocked = [
        s for s in CRITICAL_UPLIFT_SLICES
        if s in slices_blocked_by_budget
        or s in slices_blocked_by_perf
        or s in slices_blocked_by_metrics
    ]
    
    if critical_blocked:
        council_status = UpliftCouncilStatus.BLOCK
    elif (
        slices_blocked_by_budget
        or slices_blocked_by_perf
        or slices_blocked_by_metrics
    ):
        council_status = UpliftCouncilStatus.WARN
    else:
        council_status = UpliftCouncilStatus.OK
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "slices_ready_for_uplift": sorted(slices_ready),
        "slices_blocked_by_budget": sorted(slices_blocked_by_budget),
        "slices_blocked_by_perf": sorted(slices_blocked_by_perf),
        "slices_blocked_by_metrics": sorted(slices_blocked_by_metrics),
        "council_status": council_status.value,
        "critical_slices_blocked": sorted(critical_blocked),
        "per_slice": per_slice_status,
        "summary": {
            "total_slices": len(all_slice_names),
            "ready_count": len(slices_ready),
            "blocked_by_budget_count": len(slices_blocked_by_budget),
            "blocked_by_perf_count": len(slices_blocked_by_perf),
            "blocked_by_metrics_count": len(slices_blocked_by_metrics),
        },
    }


def build_uplift_director_panel(council_view: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a unified director-level panel for uplift decision making.
    
    Provides a single, executive-friendly view combining budget, performance,
    and metrics into one uplift readiness signal.
    
    Args:
        council_view: Output from build_uplift_council_view()
    
    Returns:
        Dictionary with:
            - "status_light": GREEN | YELLOW | RED
            - "headline": Short summary text
            - "ready_slices": List of slice names ready for uplift
            - "blocked_slices": List of slice names blocked (any dimension)
            - "summary": Aggregate statistics
    
    Status Light Mapping:
        GREEN: Council status OK
        YELLOW: Council status WARN
        RED: Council status BLOCK
    """
    council_status = council_view.get("council_status", "UNKNOWN")
    ready_slices = council_view.get("slices_ready_for_uplift", [])
    blocked_budget = council_view.get("slices_blocked_by_budget", [])
    blocked_perf = council_view.get("slices_blocked_by_perf", [])
    blocked_metrics = council_view.get("slices_blocked_by_metrics", [])
    critical_blocked = council_view.get("critical_slices_blocked", [])
    summary = council_view.get("summary", {})
    
    # Map council status to status light
    if council_status == "OK":
        status_light = "GREEN"
    elif council_status == "WARN":
        status_light = "YELLOW"
    else:  # BLOCK
        status_light = "RED"
    
    # Build headline
    ready_count = len(ready_slices)
    blocked_count = len(set(blocked_budget + blocked_perf + blocked_metrics))
    critical_count = len(critical_blocked)
    
    if council_status == "OK":
        headline = (
            f"Uplift experiments ready. {ready_count} slice(s) ready across all dimensions. "
            f"No critical slices blocked."
        )
    elif council_status == "WARN":
        headline = (
            f"Uplift experiments proceed with caution. {ready_count} slice(s) ready. "
            f"{blocked_count} non-critical slice(s) at risk."
        )
    else:  # BLOCK
        headline = (
            f"Uplift experiments blocked. {critical_count} critical slice(s) blocked. "
            f"Review budget, performance, and metrics before proceeding."
        )
    
    # Detailed breakdown
    blocked_breakdown = []
    if blocked_budget:
        blocked_breakdown.append(f"{len(blocked_budget)} by budget")
    if blocked_perf:
        blocked_breakdown.append(f"{len(blocked_perf)} by performance")
    if blocked_metrics:
        blocked_breakdown.append(f"{len(blocked_metrics)} by metrics")
    
    return {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "status_light": status_light,
        "headline": headline,
        "ready_slices": sorted(ready_slices),
        "blocked_slices": sorted(list(set(blocked_budget + blocked_perf + blocked_metrics))),
        "blocked_breakdown": blocked_breakdown,
        "critical_slices_blocked": sorted(critical_blocked),
        "summary": {
            "council_status": council_status,
            "ready_count": ready_count,
            "blocked_count": blocked_count,
            "critical_blocked_count": critical_count,
            "total_slices": summary.get("total_slices", 0),
        },
    }


# =============================================================================
# COUNCIL → GLOBAL CONSOLE & GOVERNANCE INTEGRATION
# =============================================================================


def summarize_uplift_council_for_global_console(
    council_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize uplift council view for global console tile.
    
    Produces a simplified, standardized output format for integration with
    global monitoring/console systems.
    
    Args:
        council_view: Output from build_uplift_council_view()
    
    Returns:
        Dictionary with:
            - "schema_version": "1.0.0"
            - "status_light": GREEN|YELLOW|RED (derived from council_status)
            - "council_status": OK|WARN|BLOCK
            - "critical_slices_blocked": List of critical slice names
            - "blocked_slices": List of all blocked slice names
            - "headline": Short summary text
    
    Status Light Mapping:
        OK → GREEN
        WARN → YELLOW
        BLOCK → RED
    """
    council_status = council_view.get("council_status", "UNKNOWN")
    
    # Map council status to status light
    if council_status == "OK":
        status_light = "GREEN"
    elif council_status == "WARN":
        status_light = "YELLOW"
    else:  # BLOCK or UNKNOWN
        status_light = "RED"
    
    critical_slices_blocked = council_view.get("critical_slices_blocked", [])
    blocked_budget = council_view.get("slices_blocked_by_budget", [])
    blocked_perf = council_view.get("slices_blocked_by_perf", [])
    blocked_metrics = council_view.get("slices_blocked_by_metrics", [])
    
    # Union of all blocked slices
    all_blocked = set(blocked_budget) | set(blocked_perf) | set(blocked_metrics)
    
    # Build headline
    if council_status == "OK":
        headline = "Uplift experiments ready across all dimensions."
    elif council_status == "WARN":
        headline = f"Uplift experiments proceed with caution. {len(all_blocked)} non-critical slice(s) at risk."
    else:  # BLOCK
        headline = (
            f"Uplift experiments blocked. {len(critical_slices_blocked)} critical slice(s) blocked. "
            f"Review required before proceeding."
        )
    
    return {
        "schema_version": "1.0.0",
        "status_light": status_light,
        "council_status": council_status,
        "critical_slices_blocked": sorted(critical_slices_blocked),
        "blocked_slices": sorted(list(all_blocked)),
        "headline": headline,
    }


def build_governance_signal_from_council(
    council_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a CLAUDE-I style governance signal from uplift council view.
    
    Produces a standardized governance signal that can be consumed by
    governance/audit systems.
    
    Args:
        council_view: Output from build_uplift_council_view()
    
    Returns:
        Dictionary with:
            - "signal_type": "UPLIFT_COUNCIL"
            - "status": OK|WARN|BLOCK
            - "blocking_slices": List of slice names blocking uplift
            - "conditions": Dict with dimension-specific conditions
    
    Format matches CLAUDE-I governance signal schema.
    """
    council_status = council_view.get("council_status", "UNKNOWN")
    critical_blocked = council_view.get("critical_slices_blocked", [])
    blocked_budget = council_view.get("slices_blocked_by_budget", [])
    blocked_perf = council_view.get("slices_blocked_by_perf", [])
    blocked_metrics = council_view.get("slices_blocked_by_metrics", [])
    
    # All blocking slices (union)
    blocking_slices = sorted(list(set(blocked_budget) | set(blocked_perf) | set(blocked_metrics)))
    
    # Build conditions dict
    conditions = {
        "budget": {
            "blocked_count": len(blocked_budget),
            "blocked_slices": sorted(blocked_budget),
        },
        "performance": {
            "blocked_count": len(blocked_perf),
            "blocked_slices": sorted(blocked_perf),
        },
        "metrics": {
            "blocked_count": len(blocked_metrics),
            "blocked_slices": sorted(blocked_metrics),
        },
    }
    
    return {
        "signal_type": "UPLIFT_COUNCIL",
        "status": council_status,
        "blocking_slices": blocking_slices,
        "conditions": conditions,
        "critical_slices_blocked": sorted(critical_blocked),
        "timestamp": None,  # Can be set by caller if needed
    }


def validate_slice_naming_contract(
    budget_cross_view: Optional[Dict[str, Any]] = None,
    perf_trend: Optional[Dict[str, Any]] = None,
    metric_conformance: Optional[Dict[str, Any]] = None,
    council_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate slice naming contract between CRITICAL_UPLIFT_SLICES and input sources.
    
    Checks that configured critical slices appear in at least one input source.
    Logs neutral notes for missing critical slices.
    
    Args:
        budget_cross_view: Optional budget cross-view
        perf_trend: Optional performance trend
        metric_conformance: Optional metric conformance
        council_view: Optional council view (for post-council validation)
    
    Returns:
        Dictionary with:
            - "valid": bool
            - "missing_critical_slices": List of critical slices not found in any source
            - "found_critical_slices": List of critical slices found
            - "notes": List of neutral validation notes
    """
    # Collect all slice names from all sources
    all_slice_names = set()
    
    if budget_cross_view:
        all_slice_names.update(budget_cross_view.get("per_slice", {}).keys())
        all_slice_names.update(budget_cross_view.get("slices_frequently_starved", []))
    
    if perf_trend:
        for perf_slice in perf_trend.get("slices", []):
            all_slice_names.add(perf_slice.get("slice_name", "unknown"))
    
    if metric_conformance:
        for metric_slice in metric_conformance.get("slices", []):
            all_slice_names.add(metric_slice.get("slice_name", "unknown"))
    
    if council_view:
        # Extract from council view
        all_slice_names.update(council_view.get("slices_ready_for_uplift", []))
        all_slice_names.update(council_view.get("slices_blocked_by_budget", []))
        all_slice_names.update(council_view.get("slices_blocked_by_perf", []))
        all_slice_names.update(council_view.get("slices_blocked_by_metrics", []))
    
    # Check which critical slices are present
    found_critical = [s for s in CRITICAL_UPLIFT_SLICES if s in all_slice_names]
    missing_critical = [s for s in CRITICAL_UPLIFT_SLICES if s not in all_slice_names]
    
    notes = []
    if missing_critical:
        notes.append(
            f"Note: {len(missing_critical)} configured critical slice(s) not found in input sources: {missing_critical}"
        )
    
    if found_critical:
        notes.append(
            f"Note: {len(found_critical)} configured critical slice(s) found in input sources: {found_critical}"
        )
    
    return {
        "valid": len(missing_critical) == 0,
        "missing_critical_slices": sorted(missing_critical),
        "found_critical_slices": sorted(found_critical),
        "notes": notes,
        "total_critical_slices": len(CRITICAL_UPLIFT_SLICES),
        "total_slices_found": len(all_slice_names),
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """CLI entry point for budget integration analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Budget ↔ Metric Integration & Uplift Readiness Analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Joint view command
    joint_parser = subparsers.add_parser("joint-view", help="Build budget-metric joint view")
    joint_parser.add_argument("--budget-logs", nargs="+", required=True, help="Budget log JSONL files")
    joint_parser.add_argument("--metrics", type=str, help="Metric conformance JSON file")
    joint_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Uplift readiness command
    uplift_parser = subparsers.add_parser("uplift-readiness", help="Check uplift readiness")
    uplift_parser.add_argument("--trend-json", type=str, help="Budget trend JSON file")
    uplift_parser.add_argument("--joint-json", type=str, help="Joint view JSON file")
    uplift_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Global health command
    health_parser = subparsers.add_parser("global-health", help="Summarize global budget health")
    health_parser.add_argument("--trend-json", type=str, required=True, help="Budget trend JSON file")
    health_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Cross-experiment view command (Phase IV)
    cross_parser = subparsers.add_parser("cross-view", help="Build cross-experiment budget view")
    cross_parser.add_argument("--run-summaries", type=str, required=True, help="JSON file with run summaries")
    cross_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Uplift planning command (Phase IV)
    plan_parser = subparsers.add_parser("uplift-plan", help="Generate uplift scheduling recommendations")
    plan_parser.add_argument("--cross-view-json", type=str, required=True, help="Cross-view JSON file")
    plan_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Director panel command (Phase IV)
    panel_parser = subparsers.add_parser("director-panel", help="Build director-level budget panel")
    panel_parser.add_argument("--cross-view-json", type=str, required=True, help="Cross-view JSON file")
    panel_parser.add_argument("--uplift-plan-json", type=str, required=True, help="Uplift plan JSON file")
    panel_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Uplift council command
    council_parser = subparsers.add_parser("uplift-council", help="Build multi-dimensional uplift council view")
    council_parser.add_argument("--budget-cross-view-json", type=str, required=True, help="Budget cross-view JSON file")
    council_parser.add_argument("--perf-trend-json", type=str, help="Performance trend JSON file")
    council_parser.add_argument("--metric-conformance-json", type=str, help="Metric conformance JSON file")
    council_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Uplift director panel command
    uplift_panel_parser = subparsers.add_parser("uplift-director-panel", help="Build unified uplift director panel")
    uplift_panel_parser.add_argument("--council-view-json", type=str, required=True, help="Council view JSON file")
    uplift_panel_parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    print("PHASE II — NOT USED IN PHASE I", file=sys.stderr)
    print("", file=sys.stderr)
    
    if args.command == "joint-view":
        from experiments.summarize_budget_usage import parse_log_file
        
        # Load budget summaries
        summaries = []
        for path in args.budget_logs:
            try:
                summaries.append(parse_log_file(Path(path)))
            except Exception as e:
                print(f"WARNING: Failed to load {path}: {e}", file=sys.stderr)
        
        # Load metric conformance
        metrics = None
        if args.metrics:
            with open(args.metrics, "r") as f:
                metrics = json.load(f)
        
        # Build joint view
        result = build_budget_metric_joint_view(summaries, metrics)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    elif args.command == "uplift-readiness":
        trend = None
        if args.trend_json:
            with open(args.trend_json, "r") as f:
                trend_data = json.load(f)
                # Reconstruct TrendReport (simplified)
                trend = TrendReport(
                    inputs=trend_data.get("inputs", []),
                    slices=[
                        SliceTrend(
                            slice_name=s["slice_name"],
                            runs=[],
                            trend=TrendDirection(s["trend"]),
                            status_sequence=s["status_sequence"],
                        )
                        for s in trend_data.get("slices", [])
                    ],
                    summary=trend_data.get("summary", {}),
                )
        
        joint = None
        if args.joint_json:
            with open(args.joint_json, "r") as f:
                joint = json.load(f)
        
        result = summarize_budget_for_uplift(trend, joint)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    elif args.command == "global-health":
        with open(args.trend_json, "r") as f:
            trend_data = json.load(f)
            trend = TrendReport(
                inputs=trend_data.get("inputs", []),
                slices=[
                    SliceTrend(
                        slice_name=s["slice_name"],
                        runs=[],
                        trend=TrendDirection(s["trend"]),
                        status_sequence=s["status_sequence"],
                    )
                    for s in trend_data.get("slices", [])
                ],
                summary=trend_data.get("summary", {}),
            )
        
        result = summarize_budget_for_global_health(trend)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    elif args.command == "cross-view":
        with open(args.run_summaries, "r") as f:
            run_summaries = json.load(f)
            if not isinstance(run_summaries, list):
                print("ERROR: run_summaries must be a JSON array", file=sys.stderr)
                sys.exit(1)
        
        result = build_cross_experiment_budget_view(run_summaries)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    elif args.command == "uplift-plan":
        with open(args.cross_view_json, "r") as f:
            cross_view = json.load(f)
        
        result = plan_uplift_runs(cross_view)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    elif args.command == "director-panel":
        with open(args.cross_view_json, "r") as f:
            cross_view = json.load(f)
        with open(args.uplift_plan_json, "r") as f:
            uplift_plan = json.load(f)
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    elif args.command == "uplift-council":
        with open(args.budget_cross_view_json, "r") as f:
            budget_cross_view = json.load(f)
        
        perf_trend = None
        if args.perf_trend_json:
            with open(args.perf_trend_json, "r") as f:
                perf_trend = json.load(f)
        
        metric_conformance = None
        if args.metric_conformance_json:
            with open(args.metric_conformance_json, "r") as f:
                metric_conformance = json.load(f)
        
        result = build_uplift_council_view(budget_cross_view, perf_trend, metric_conformance)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    elif args.command == "uplift-director-panel":
        with open(args.council_view_json, "r") as f:
            council_view = json.load(f)
        
        result = build_uplift_director_panel(council_view)
        
        output = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

