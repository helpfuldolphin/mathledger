"""
Minimal Uplift Council for Multi-Dimensional Uplift Decisions.

Provides read-only council aggregation combining budget, performance, and metrics.
All functions are pure (no side effects) and JSON-serializable.

Phase X Integration:
- P3 (synthetic): Budget is typically N/A or stable. Synthetic experiments don't
  enforce real budget constraints, so budget dimension may be omitted or default to OK.
- P4 (shadow): Budget can flag shadow experiments that are too costly or unstable.
  Budget health from real-runner telemetry feeds into council decisions.

Evidence Pack Integration:
- Council view is attached under evidence["governance"]["budget_council"].
- Provides multi-dimensional uplift readiness signal for external consumers.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from experiments.budget_observability import BudgetSummary

# Critical uplift slices (from curriculum_uplift_phase2.yaml)
CRITICAL_UPLIFT_SLICES = {
    "slice_uplift_goal",
    "slice_uplift_sparse",
    "slice_uplift_tree",
    "slice_uplift_dependency",
}


class CouncilStatus(str, Enum):
    """Overall council decision status."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class DimensionStatus(str, Enum):
    """Per-dimension status."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"
    UNKNOWN = "UNKNOWN"


def build_uplift_council_view(
    budget_cross_view: Optional[Dict[str, Any]] = None,
    perf_trend: Optional[Dict[str, Any]] = None,
    metric_conformance: Optional[Dict[str, Any]] = None,
    perf_joint_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build unified uplift council view from three dimensions.
    
    Evaluates each slice across budget, performance, and metrics dimensions.
    Aggregates worst-case status per slice and determines overall council_status.
    
    Council rules:
    - BLOCK: Any critical slice blocked by any dimension
    - WARN: Non-critical slices at risk OR any dimension WARN on critical slice
    - OK: All dimensions OK for all slices
    
    Missing dimension data defaults to OK (assume healthy if no data).
    
    Args:
        budget_cross_view: Budget health view (from budget_integration)
            Format: {"slices": [{"slice_name": str, "status": "SAFE|TIGHT|STARVED", ...}]}
        perf_trend: Performance trend data (legacy format)
            Format: {"slices": [{"slice_name": str, "status": "OK|WARN|BLOCK"}]}
        metric_conformance: Metric conformance data
            Format: {"slices": [{"slice_name": str, "ready": bool}]}
        perf_joint_view: Performance joint governance view (preferred)
            Format: Output from build_perf_joint_governance_view()
            If provided, takes precedence over perf_trend.
    
    Returns:
        Council view with per-slice status and overall council_status
    """
    # Convert perf_joint_view to council format if provided
    if perf_joint_view:
        from experiments.verify_perf_equivalence import adapt_perf_joint_view_for_council
        perf_trend = adapt_perf_joint_view_for_council(perf_joint_view)
    
    # Collect all slice names from all dimensions
    all_slice_names = set()
    
    if budget_cross_view and "slices" in budget_cross_view:
        all_slice_names.update(s.get("slice_name") for s in budget_cross_view["slices"])
    
    if perf_trend:
        # Check for perf_joint_view format (has slices_blocking_uplift)
        if "slices_blocking_uplift" in perf_trend:
            all_slice_names.update(perf_trend.get("slices_blocking_uplift", []))
            all_slice_names.update(perf_trend.get("slices_with_regressions", []))
        elif "slices" in perf_trend:
            all_slice_names.update(s.get("slice_name") for s in perf_trend["slices"])
    
    if metric_conformance and "slices" in metric_conformance:
        all_slice_names.update(s.get("slice_name") for s in metric_conformance["slices"])
    
    # Evaluate each slice across dimensions
    slices_ready_for_uplift: List[str] = []
    slices_blocked_by_budget: List[str] = []
    slices_blocked_by_perf: List[str] = []
    slices_blocked_by_metrics: List[str] = []
    
    per_slice_status: Dict[str, Dict[str, str]] = {}
    
    for slice_name in all_slice_names:
        if not slice_name:
            continue
        
        # Evaluate each dimension
        budget_status = _evaluate_budget_dimension(slice_name, budget_cross_view)
        perf_status = _evaluate_perf_dimension(slice_name, perf_trend)
        metrics_status = _evaluate_metrics_dimension(slice_name, metric_conformance)
        
        # Aggregate worst-case
        overall_status = _aggregate_status([budget_status, perf_status, metrics_status])
        
        per_slice_status[slice_name] = {
            "budget": budget_status.value,
            "perf": perf_status.value,
            "metrics": metrics_status.value,
            "overall": overall_status.value,
        }
        
        # Categorize
        if overall_status == DimensionStatus.BLOCK:
            if budget_status == DimensionStatus.BLOCK:
                slices_blocked_by_budget.append(slice_name)
            if perf_status == DimensionStatus.BLOCK:
                slices_blocked_by_perf.append(slice_name)
            if metrics_status == DimensionStatus.BLOCK:
                slices_blocked_by_metrics.append(slice_name)
        elif overall_status == DimensionStatus.OK:
            slices_ready_for_uplift.append(slice_name)
    
    # Determine council status
    council_status = _determine_council_status(
        per_slice_status,
        slices_blocked_by_budget,
        slices_blocked_by_perf,
        slices_blocked_by_metrics,
    )
    
    return {
        "schema_version": "1.0.0",
        "council_status": council_status.value,
        "slices_ready_for_uplift": sorted(slices_ready_for_uplift),
        "slices_blocked_by_budget": sorted(slices_blocked_by_budget),
        "slices_blocked_by_perf": sorted(slices_blocked_by_perf),
        "slices_blocked_by_metrics": sorted(slices_blocked_by_metrics),
        "per_slice_status": per_slice_status,
    }


def _evaluate_budget_dimension(
    slice_name: str,
    budget_cross_view: Optional[Dict[str, Any]],
) -> DimensionStatus:
    """Evaluate budget dimension for a slice."""
    if not budget_cross_view or "slices" not in budget_cross_view:
        return DimensionStatus.UNKNOWN
    
    for slice_data in budget_cross_view["slices"]:
        if slice_data.get("slice_name") == slice_name:
            health = slice_data.get("health_status", "")
            # Map budget health to dimension status
            if health == "STARVED":
                # Check if frequently starved
                if slice_data.get("frequently_starved", False):
                    return DimensionStatus.BLOCK
                return DimensionStatus.WARN
            elif health == "TIGHT":
                return DimensionStatus.WARN
            elif health == "SAFE":
                return DimensionStatus.OK
            else:
                return DimensionStatus.UNKNOWN
    
    return DimensionStatus.UNKNOWN


def _evaluate_perf_dimension(
    slice_name: str,
    perf_trend: Optional[Dict[str, Any]],
) -> DimensionStatus:
    """
    Evaluate performance dimension for a slice.
    
    Checks if slice is in slices_blocking_uplift (BLOCK) or
    slices_with_regressions (WARN) from perf_joint_view, or
    uses legacy perf_trend format.
    """
    if not perf_trend:
        return DimensionStatus.UNKNOWN
    
    # Check for perf_joint_view format (has slices_blocking_uplift)
    if "slices_blocking_uplift" in perf_trend:
        slices_blocking = perf_trend.get("slices_blocking_uplift", [])
        slices_regressions = perf_trend.get("slices_with_regressions", [])
        
        if slice_name in slices_blocking:
            return DimensionStatus.BLOCK
        elif slice_name in slices_regressions:
            return DimensionStatus.WARN
        else:
            # Check perf_risk for overall assessment
            perf_risk = perf_trend.get("perf_risk", "LOW")
            if perf_risk == "HIGH":
                return DimensionStatus.WARN  # Conservative: WARN on HIGH risk
            return DimensionStatus.OK
    
    # Legacy format: slices array
    if "slices" in perf_trend:
        for slice_data in perf_trend["slices"]:
            if slice_data.get("slice_name") == slice_name:
                status = slice_data.get("status", "").upper()
                if status == "BLOCK":
                    return DimensionStatus.BLOCK
                elif status == "WARN":
                    return DimensionStatus.WARN
                elif status == "OK":
                    return DimensionStatus.OK
    
    return DimensionStatus.UNKNOWN


def _evaluate_metrics_dimension(
    slice_name: str,
    metric_conformance: Optional[Dict[str, Any]],
) -> DimensionStatus:
    """Evaluate metrics dimension for a slice."""
    if not metric_conformance or "slices" not in metric_conformance:
        return DimensionStatus.UNKNOWN
    
    for slice_data in metric_conformance["slices"]:
        if slice_data.get("slice_name") == slice_name:
            ready = slice_data.get("ready", False)
            return DimensionStatus.BLOCK if not ready else DimensionStatus.OK
    
    return DimensionStatus.UNKNOWN


def _aggregate_status(statuses: List[DimensionStatus]) -> DimensionStatus:
    """Aggregate multiple dimension statuses using worst-case logic."""
    # Worst-case ordering: BLOCK > WARN > OK > UNKNOWN
    if DimensionStatus.BLOCK in statuses:
        return DimensionStatus.BLOCK
    if DimensionStatus.WARN in statuses:
        return DimensionStatus.WARN
    if DimensionStatus.OK in statuses:
        return DimensionStatus.OK
    return DimensionStatus.UNKNOWN


def _determine_council_status(
    per_slice_status: Dict[str, Dict[str, str]],
    slices_blocked_by_budget: List[str],
    slices_blocked_by_perf: List[str],
    slices_blocked_by_metrics: List[str],
) -> CouncilStatus:
    """Determine overall council status."""
    # Check if any critical slice is blocked
    critical_blocked = any(
        slice_name in CRITICAL_UPLIFT_SLICES
        and per_slice_status.get(slice_name, {}).get("overall") == "BLOCK"
        for slice_name in (slices_blocked_by_budget + slices_blocked_by_perf + slices_blocked_by_metrics)
    )
    
    if critical_blocked:
        return CouncilStatus.BLOCK
    
    # Check if any slice (critical or non-critical) has WARN
    any_warn = any(
        per_slice_status.get(slice_name, {}).get("overall") == "WARN"
        for slice_name in per_slice_status.keys()
    )
    
    if any_warn:
        return CouncilStatus.WARN
    
    # All OK
    return CouncilStatus.OK


def summarize_uplift_council_for_global_console(
    council_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize council view for global console tile.
    
    Translates multi-dimensional council decision into simple status light
    and headline for executive dashboard.
    
    Args:
        council_view: Output from build_uplift_council_view()
    
    Returns:
        Global console tile with status_light, headline, and breakdown
    """
    council_status = council_view.get("council_status", "OK")
    
    # Map council status to status light
    if council_status == "BLOCK":
        status_light = "RED"
        budget_status = "BLOCK"
    elif council_status == "WARN":
        status_light = "YELLOW"
        budget_status = "WARN"
    else:
        status_light = "GREEN"
        budget_status = "OK"
    
    # Build headline
    blocked_count = len(council_view.get("slices_blocked_by_budget", []))
    ready_count = len(council_view.get("slices_ready_for_uplift", []))
    
    if blocked_count > 0:
        headline = f"{blocked_count} slice(s) blocked, {ready_count} ready"
    else:
        headline = f"{ready_count} slice(s) ready for uplift"
    
    # Identify critical slices blocked
    critical_slices_blocked = [
        s for s in (
            council_view.get("slices_blocked_by_budget", []) +
            council_view.get("slices_blocked_by_perf", []) +
            council_view.get("slices_blocked_by_metrics", [])
        )
        if s in CRITICAL_UPLIFT_SLICES
    ]
    
    return {
        "tile_type": "uplift_council",
        "schema_version": "1.0.0",
        "status_light": status_light,
        "budget_status": budget_status,
        "critical_slices_blocked": sorted(critical_slices_blocked),
        "headline": headline,
        "ready_slices": sorted(council_view.get("slices_ready_for_uplift", [])),
        "blocked_slices": sorted(
            council_view.get("slices_blocked_by_budget", []) +
            council_view.get("slices_blocked_by_perf", []) +
            council_view.get("slices_blocked_by_metrics", [])
        ),
    }


def compute_budget_modulation_for_calibration_window(
    window: Dict[str, Any],
    budget_cross_view: Optional[Dict[str, Any]] = None,
    slice_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute budget-aware modulation fields for calibration window.
    
    Analyzes whether budget constraints are confounding divergence measurements
    in a calibration window. Provides modulation fields for P5 drift-modulation analysis.
    
    Budget Confounding Detection:
    - If budget health is STARVED during window → budget_confounded = True
    - If budget frequently starved (>50% of runs) → persistent drift
    - If budget STARVED in single window → transient drift
    
    Effective LR Adjustment:
    - Estimates learning rate adjustment needed to compensate for budget constraints
    - Formula: effective_lr_adjustment = 1.0 - (budget_exhausted_pct / 100.0)
    - Range: [0.0, 1.0] where 1.0 = no adjustment, 0.0 = full budget exhaustion
    
    Args:
        window: Calibration window dictionary with start_cycle, end_cycle, etc.
        budget_cross_view: Budget health view (from budget_integration)
            Format: {"slices": [{"slice_name": str, "health_status": "SAFE|TIGHT|STARVED", ...}]}
        slice_name: Name of slice for this calibration window (optional)
    
    Returns:
        Modulation dictionary:
        {
            "budget_confounded": bool,
            "effective_lr_adjustment": float,
            "drift_classification": "NONE" | "TRANSIENT" | "PERSISTENT",
            "budget_health_during_window": "SAFE" | "TIGHT" | "STARVED" | "UNKNOWN",
        }
    """
    # Default: no budget confounding
    budget_confounded = False
    effective_lr_adjustment = 1.0
    drift_classification = "NONE"
    budget_health = "UNKNOWN"
    
    if not budget_cross_view or not slice_name:
        return {
            "budget_confounded": False,
            "effective_lr_adjustment": 1.0,
            "drift_classification": "NONE",
            "budget_health_during_window": "UNKNOWN",
        }
    
    # Find budget status for this slice
    for slice_data in budget_cross_view.get("slices", []):
        if slice_data.get("slice_name") == slice_name:
            health_status = slice_data.get("health_status", "")
            frequently_starved = slice_data.get("frequently_starved", False)
            budget_exhausted_pct = slice_data.get("budget_exhausted_pct", 0.0)
            
            budget_health = health_status
            
            # Determine if budget is confounding
            if health_status == "STARVED":
                budget_confounded = True
                if frequently_starved:
                    drift_classification = "PERSISTENT"
                else:
                    drift_classification = "TRANSIENT"
            
            # Compute effective LR adjustment
            # When budget is exhausted, learning rate should be reduced
            # to account for reduced effective sample size
            if budget_exhausted_pct > 0.0:
                # Adjustment: reduce LR proportionally to budget exhaustion
                # More exhaustion → lower effective LR
                effective_lr_adjustment = max(0.0, 1.0 - (budget_exhausted_pct / 100.0))
            else:
                effective_lr_adjustment = 1.0
            
            break
    
    return {
        "budget_confounded": budget_confounded,
        "effective_lr_adjustment": round(effective_lr_adjustment, 4),
        "drift_classification": drift_classification,
        "budget_health_during_window": budget_health,
    }


def annotate_calibration_window_with_exclusion(
    window: Dict[str, Any],
    prng_signal: Optional[Dict[str, Any]] = None,
    topology_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Annotate a single calibration window with exclusion recommendation.
    
    This is a convenience wrapper that:
    1. Ensures window has budget modulation (if not present, computes it)
    2. Adds exclusion recommendation based on cross-signal checks
    
    Args:
        window: Calibration window dictionary (may or may not have budget modulation)
        prng_signal: Optional PRNG signal for cross-check
        topology_signal: Optional topology signal for cross-check
    
    Returns:
        Window with exclusion recommendation fields added
    """
    from experiments.budget_calibration_modulation import (
        compute_calibration_exclusion_recommendation,
    )
    
    # Extract budget modulation (assume it's already computed if present)
    if "budget_confounded" in window:
        budget_modulation = {
            "budget_confounded": window.get("budget_confounded", False),
            "drift_classification": window.get("drift_classification", "NONE"),
        }
    else:
        # No budget modulation present, default to no confounding
        budget_modulation = {
            "budget_confounded": False,
            "drift_classification": "NONE",
        }
    
    exclusion = compute_calibration_exclusion_recommendation(
        budget_modulation=budget_modulation,
        prng_signal=prng_signal,
        topology_signal=topology_signal,
    )
    
    # Add exclusion fields to window
    annotated = dict(window)
    annotated.update(exclusion)
    return annotated


def build_first_light_budget_summary(
    council_view: Dict[str, Any],
    calibration_windows: Optional[List[Dict[str, Any]]] = None,
    budget_cross_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build compact First Light budget summary from council view.
    
    Creates a minimal, evidence-friendly summary for First Light cross-check.
    Includes only essential fields: status, critical_slices_blocked, blocked_slices, ready_slices.
    
    Optionally annotates calibration windows with budget-driven drift classification
    (transient vs persistent) for P5 drift-modulation analysis.
    
    Args:
        council_view: Output from build_uplift_council_view()
        calibration_windows: Optional list of calibration window dictionaries
            Format: [{"start_cycle": int, "end_cycle": int, "slice_name": str, ...}, ...]
        budget_cross_view: Optional budget health view for window annotation
    
    Returns:
        Compact summary dictionary:
        {
            "status": "OK" | "WARN" | "BLOCK",
            "critical_slices_blocked": [...],
            "blocked_slices": [...],
            "ready_slices": [...],
            "calibration_windows_annotated": [...],  # If calibration_windows provided
        }
    """
    council_status = council_view.get("council_status", "OK")
    
    # Collect all blocked slices
    blocked_slices = sorted(
        council_view.get("slices_blocked_by_budget", []) +
        council_view.get("slices_blocked_by_perf", []) +
        council_view.get("slices_blocked_by_metrics", [])
    )
    
    # Identify critical slices blocked
    critical_slices_blocked = [
        s for s in blocked_slices
        if s in CRITICAL_UPLIFT_SLICES
    ]
    
    summary = {
        "status": council_status,
        "critical_slices_blocked": sorted(critical_slices_blocked),
        "blocked_slices": blocked_slices,
        "ready_slices": sorted(council_view.get("slices_ready_for_uplift", [])),
    }
    
    # Annotate calibration windows with budget modulation if provided
    if calibration_windows and budget_cross_view:
        from experiments.budget_calibration_modulation import (
            annotate_calibration_windows_with_budget_modulation,
            annotate_calibration_windows_with_exclusion_recommendations,
        )
        
        # First, add budget modulation
        annotated_windows = annotate_calibration_windows_with_budget_modulation(
            calibration_windows=calibration_windows,
            budget_cross_view=budget_cross_view,
            slice_name=blocked_slices[0] if blocked_slices else None,
        )
        
        # Then, add exclusion recommendations based on cross-signal checks
        # Note: This is advisory only - no automatic filtering
        annotated_windows = annotate_calibration_windows_with_exclusion_recommendations(
            calibration_windows=annotated_windows,
            prng_signal=prng_signal,
            topology_signal=topology_signal,
        )
        
        summary["calibration_windows_annotated"] = annotated_windows
    
    return summary


def budget_summary_to_council_input(
    slice_name: str,
    budget_summary: Any,  # BudgetSummary from budget_observability
    health_status: str,
    frequently_starved: bool = False,
) -> Dict[str, Any]:
    """
    Convert BudgetSummary to council's expected budget dimension input format.
    
    This helper bridges budget observability (BudgetSummary) with council
    evaluation (_evaluate_budget_dimension). It transforms the raw budget
    summary into the structured format expected by build_uplift_council_view().
    
    Phase X Context:
    - P3: Budget summaries may be synthetic or N/A. If provided, use this helper
      to convert to council format, but budget dimension may be omitted entirely.
    - P4: Real-runner budget telemetry should be converted using this helper
      before feeding into council view.
    
    Args:
        slice_name: Name of the slice this budget summary applies to
        budget_summary: BudgetSummary from budget observability
        health_status: BudgetHealthStatus value ("SAFE", "TIGHT", "STARVED", "INVALID")
        frequently_starved: Whether this slice is frequently starved (>50% of runs)
    
    Returns:
        Dictionary in format expected by build_uplift_council_view() budget_cross_view:
        {
            "slice_name": str,
            "health_status": "SAFE" | "TIGHT" | "STARVED" | "INVALID",
            "frequently_starved": bool,
            "budget_exhausted_pct": float,
            "timeout_abstentions_avg": float,
        }
    """
    return {
        "slice_name": slice_name,
        "health_status": health_status,
        "frequently_starved": frequently_starved,
        "budget_exhausted_pct": budget_summary.budget_exhausted_pct,
        "timeout_abstentions_avg": budget_summary.timeout_abstentions_avg,
    }


def attach_budget_council_to_evidence(
    evidence: Dict[str, Any],
    council_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach budget council view to evidence pack.
    
    Non-mutating: copies input evidence dict, attaches council view under
    evidence["governance"]["budget_council"], and returns new dict.
    
    Evidence Pack Structure:
    {
        "governance": {
            "budget_council": {
                "council_status": "OK" | "WARN" | "BLOCK",
                "slices_ready_for_uplift": [...],
                "slices_blocked_by_budget": [...],
                "slices_blocked_by_perf": [...],
                "slices_blocked_by_metrics": [...],
                "per_slice_status": {...},
            }
        }
    }
    
    Args:
        evidence: Evidence pack dictionary (will be copied, not modified)
        council_view: Output from build_uplift_council_view()
    
    Returns:
        New evidence dict with council view attached under governance.budget_council
    """
    # Non-mutating: create new dict
    updated = dict(evidence)
    
    # Ensure governance section exists
    if "governance" not in updated:
        updated["governance"] = {}
    else:
        updated["governance"] = dict(updated["governance"])
    
    # Attach council view
    updated["governance"]["budget_council"] = dict(council_view)
    
    # Attach First Light summary for cross-check
    summary = build_first_light_budget_summary(council_view)
    updated["governance"]["budget_council_summary"] = dict(summary)
    
    return updated

