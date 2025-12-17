"""Performance governance and equivalence verification.

Provides performance ratchet, SLO enforcement, trend analytics, and release governance.
All outputs are advisory and JSON-safe.

PHASE X ALIGNMENT:
Performance governance is one axis of the P3/P4 evidence package, not a standalone gate.
The perf tile is intended to be consumed by the Uplift Council and global health surfaces,
and is expected to appear in the Phase X evidence bundle.

All functions are read-only and advisory. They provide observability and risk assessment
but do not enforce hard gates. The Uplift Council aggregates performance data with
budget and metric dimensions to make unified uplift decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class PerfGovernanceInputError(Exception):
    """Raised when input contract validation fails."""

    pass


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_risk_level(value: str, name: str) -> None:
    """Validate that a risk level is one of the allowed values."""
    allowed = {"LOW", "MEDIUM", "HIGH"}
    if value not in allowed:
        raise PerfGovernanceInputError(
            f"{name} must be one of {allowed}, got {value!r}"
        )


def validate_perf_trend(perf_trend: Dict[str, Any]) -> None:
    """Validate perf_trend structure."""
    if not isinstance(perf_trend, dict):
        raise PerfGovernanceInputError("perf_trend must be a dict")

    # Check required top-level keys
    required_keys = ["schema_version", "runs"]
    for key in required_keys:
        if key not in perf_trend:
            raise PerfGovernanceInputError(f"perf_trend missing required key: {key}")

    # Validate runs
    runs = perf_trend.get("runs", [])
    if not isinstance(runs, list):
        raise PerfGovernanceInputError("perf_trend.runs must be a list")

    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            raise PerfGovernanceInputError(f"perf_trend.runs[{i}] must be a dict")
        if "status" not in run:
            raise PerfGovernanceInputError(f"perf_trend.runs[{i}] missing 'status'")

        # Validate status if present
        status = run.get("status")
        if status is not None and status not in {"PASS", "WARN", "FAIL"}:
            raise PerfGovernanceInputError(
                f"perf_trend.runs[{i}].status must be PASS/WARN/FAIL, got {status!r}"
            )

    # Validate release_risk_level if present
    risk_level = perf_trend.get("release_risk_level")
    if risk_level is not None:
        _validate_risk_level(risk_level, "perf_trend.release_risk_level")


def validate_budget_trend(budget_trend: Dict[str, Any]) -> None:
    """Validate budget_trend structure."""
    if not isinstance(budget_trend, dict):
        raise PerfGovernanceInputError("budget_trend must be a dict")

    # Validate budget_risk if present (allow missing for graceful defaults)
    budget_risk = budget_trend.get("budget_risk")
    if budget_risk is not None:
        _validate_risk_level(budget_risk, "budget_trend.budget_risk")


def validate_metric_conformance(metric_conformance: Dict[str, Any]) -> None:
    """Validate metric_conformance structure."""
    if not isinstance(metric_conformance, dict):
        raise PerfGovernanceInputError("metric_conformance must be a dict")

    # Metric conformance is optional, but if present should have valid structure
    if "status" in metric_conformance:
        status = metric_conformance["status"]
        if status not in {"OK", "WARN", "BLOCK"}:
            raise PerfGovernanceInputError(
                f"metric_conformance.status must be OK/WARN/BLOCK, got {status!r}"
            )


def validate_perf_governance_inputs(
    perf_trend: Optional[Dict[str, Any]] = None,
    budget_trend: Optional[Dict[str, Any]] = None,
    metric_conformance: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate all performance governance inputs."""
    if perf_trend is not None:
        validate_perf_trend(perf_trend)
    if budget_trend is not None:
        validate_budget_trend(budget_trend)
    if metric_conformance is not None:
        validate_metric_conformance(metric_conformance)


# ═══════════════════════════════════════════════════════════════════════════════
# JOINT GOVERNANCE VIEW
# ═══════════════════════════════════════════════════════════════════════════════


def build_perf_joint_governance_view(
    perf_trend: Dict[str, Any],
    budget_trend: Dict[str, Any],
    metric_conformance: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a joint governance view combining perf, budget, and metric data.

    Args:
        perf_trend: Performance trend ledger with runs and risk level.
        budget_trend: Budget trend data with risk assessment.
        metric_conformance: Metric conformance status.

    Returns:
        Joint view with perf_risk, slices_with_regressions, slices_blocking_uplift.
    """
    # Validate inputs (only if they have content - allow empty dicts to default)
    if perf_trend:
        validate_perf_trend(perf_trend)
    if budget_trend:
        validate_budget_trend(budget_trend)
    if metric_conformance:
        validate_metric_conformance(metric_conformance)

    # Extract perf risk from trend ledger
    perf_risk = perf_trend.get("release_risk_level", "LOW")
    if perf_risk not in {"LOW", "MEDIUM", "HIGH"}:
        perf_risk = "LOW"  # Default to LOW if invalid

    # Extract budget risk
    budget_risk = budget_trend.get("budget_risk", "LOW")
    if budget_risk not in {"LOW", "MEDIUM", "HIGH"}:
        budget_risk = "LOW"

    # Identify slices with perf regressions
    slices_with_regressions = []
    runs = perf_trend.get("runs", [])
    for run in runs:
        status = run.get("status")
        if status in {"WARN", "FAIL"}:
            # Try to extract slice_name from run
            slice_name = run.get("slice_name")
            if slice_name:
                if slice_name not in slices_with_regressions:
                    slices_with_regressions.append(slice_name)
            else:
                # Fallback: mark as "all_slices" if slice info missing
                if "all_slices" not in slices_with_regressions:
                    slices_with_regressions.append("all_slices")

    # Determine slices where perf blocks uplift
    # This requires both perf regression AND high budget risk
    slices_blocking_uplift = []
    if perf_risk in {"MEDIUM", "HIGH"} and budget_risk == "HIGH":
        # If we have slice-level data, use it; otherwise mark all
        if slices_with_regressions:
            slices_blocking_uplift = [
                s for s in slices_with_regressions if s != "all_slices"
            ]
            if not slices_blocking_uplift:
                slices_blocking_uplift = ["all_slices"]
        else:
            slices_blocking_uplift = ["all_slices"]

    # Build summary note
    notes = []
    if perf_risk == "HIGH":
        notes.append("High performance risk detected")
    if budget_risk == "HIGH":
        notes.append("High budget risk detected")
    if slices_blocking_uplift:
        notes.append(f"Performance blocking uplift on {len(slices_blocking_uplift)} slice(s)")

    summary_note = "; ".join(notes) if notes else "Performance governance: nominal"

    return {
        "perf_risk": perf_risk,
        "budget_risk": budget_risk,
        "slices_with_regressions": slices_with_regressions,
        "slices_blocking_uplift": slices_blocking_uplift,
        "summary_note": summary_note,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RELEASE BUNDLE ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════


def summarize_perf_for_global_release(
    perf_joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Map perf joint view into release decision format.

    Args:
        perf_joint_view: Output from build_perf_joint_governance_view.

    Returns:
        Release decision with release_ok, status, blocking_components, headline.
    """
    # Validate that perf_joint_view has required keys
    required_keys = ["perf_risk", "budget_risk", "slices_with_regressions", "slices_blocking_uplift"]
    for key in required_keys:
        if key not in perf_joint_view:
            raise PerfGovernanceInputError(
                f"perf_joint_view missing required key: {key}"
            )

    perf_risk = perf_joint_view["perf_risk"]
    budget_risk = perf_joint_view["budget_risk"]
    slices_with_regressions = perf_joint_view["slices_with_regressions"]
    slices_blocking_uplift = perf_joint_view["slices_blocking_uplift"]

    # Decision logic
    release_ok = True
    status = "OK"
    blocking_components = []

    # BLOCK conditions
    if perf_risk == "HIGH" and budget_risk == "HIGH" and slices_blocking_uplift:
        release_ok = False
        status = "BLOCK"
        blocking_components = ["performance", "budget"]
    elif perf_risk == "HIGH" and slices_blocking_uplift:
        release_ok = False
        status = "BLOCK"
        blocking_components = ["performance"]
    elif perf_risk == "HIGH":
        release_ok = False
        status = "BLOCK"
        blocking_components = ["performance"]

    # WARN conditions
    elif perf_risk == "MEDIUM" or (slices_with_regressions and budget_risk == "LOW"):
        release_ok = True  # Still OK to release, but warn
        status = "WARN"
        if slices_with_regressions:
            blocking_components = ["performance"]

    # OK conditions
    else:
        release_ok = True
        status = "OK"

    # Build headline
    if status == "BLOCK":
        headline = f"Performance governance: {status} ({', '.join(blocking_components)})"
    elif status == "WARN":
        headline = f"Performance governance: {status} (monitor {', '.join(blocking_components) if blocking_components else 'trends'})"
    else:
        headline = "Performance governance: OK"

    return {
        "release_ok": release_ok,
        "status": status,
        "blocking_components": blocking_components,
        "headline": headline,
        "perf_risk": perf_risk,
        "budget_risk": budget_risk,
        "slices_with_regressions": slices_with_regressions,
        "slices_blocking_uplift": slices_blocking_uplift,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSOLE ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════


def summarize_perf_for_global_console(
    perf_joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a small JSON tile for global dashboards.

    Status mapping:
    - RED if perf_risk="HIGH" or any critical slice appears in slices_where_perf_blocks_uplift
    - YELLOW if perf_risk="MEDIUM" or regressions exist but not blocking uplift
    - GREEN if perf_risk="LOW" and no regressions on critical slices

    Args:
        perf_joint_view: Output from build_perf_joint_governance_view.

    Returns:
        Global console tile with status_light, perf_risk, budget_risk, etc.
    """
    # Validate that perf_joint_view has required keys
    required_keys = ["perf_risk", "budget_risk", "slices_with_regressions", "slices_blocking_uplift"]
    for key in required_keys:
        if key not in perf_joint_view:
            raise PerfGovernanceInputError(
                f"perf_joint_view missing required key: {key}"
            )

    perf_risk = perf_joint_view["perf_risk"]
    budget_risk = perf_joint_view["budget_risk"]
    slices_with_regressions = perf_joint_view["slices_with_regressions"]
    slices_blocking_uplift = perf_joint_view["slices_blocking_uplift"]

    # Determine status_light
    status_light = "GREEN"
    if perf_risk == "HIGH" or slices_blocking_uplift:
        status_light = "RED"
    elif perf_risk == "MEDIUM" or slices_with_regressions:
        status_light = "YELLOW"

    # Extract critical slices with regressions (non-"all_slices" entries)
    critical_slices_with_regressions = [
        s for s in slices_with_regressions if s != "all_slices"
    ]

    # Build headline
    if status_light == "RED":
        headline = f"Performance risk: {perf_risk}"
    elif status_light == "YELLOW":
        headline = f"Performance risk: {perf_risk} (monitoring)"
    else:
        headline = "Performance: nominal"

    return {
        "schema_version": "1.0.0",
        "tile_type": "uplift_perf_health",
        "status_light": status_light,
        "perf_risk": perf_risk,
        "budget_risk": budget_risk,
        "critical_slices_with_regressions": critical_slices_with_regressions,
        "headline": headline,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE PACK HOOK
# ═══════════════════════════════════════════════════════════════════════════════


def build_uplift_perf_governance_tile(
    perf_joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an evidence tile for the evidence pack.

    This tile exposes perf_risk, slices_blocking_uplift, and neutral rationale strings.
    It is read-only and does not act as a direct gate.

    Args:
        perf_joint_view: Output from build_perf_joint_governance_view.

    Returns:
        Evidence tile with status, perf_risk, slices, headline, notes.
    """
    # Validate that perf_joint_view has required keys
    required_keys = ["perf_risk", "slices_with_regressions", "slices_blocking_uplift", "summary_note"]
    for key in required_keys:
        if key not in perf_joint_view:
            raise PerfGovernanceInputError(
                f"perf_joint_view missing required key: {key}"
            )

    perf_risk = perf_joint_view["perf_risk"]
    slices_with_regressions = perf_joint_view["slices_with_regressions"]
    slices_blocking_uplift = perf_joint_view["slices_blocking_uplift"]
    summary_note = perf_joint_view.get("summary_note", "")

    # Determine status from perf_risk
    if perf_risk == "HIGH" or slices_blocking_uplift:
        status = "BLOCK"
    elif perf_risk == "MEDIUM" or slices_with_regressions:
        status = "WARN"
    else:
        status = "OK"

    # Build notes array
    notes = []
    if slices_with_regressions:
        notes.append(f"Regressions detected on {len(slices_with_regressions)} slice(s)")
    if slices_blocking_uplift:
        notes.append(f"Performance blocking uplift on {len(slices_blocking_uplift)} slice(s)")
    if perf_risk == "HIGH":
        notes.append("High performance risk")
    elif perf_risk == "MEDIUM":
        notes.append("Medium performance risk")

    # Build headline
    if status == "BLOCK":
        headline = "Performance governance: BLOCK"
    elif status == "WARN":
        headline = "Performance governance: WARN"
    else:
        headline = "Performance governance: OK"

    return {
        "tile_type": "uplift_perf_governance",
        "schema_version": "1.0.0",
        "status": status,
        "perf_risk": perf_risk,
        "slices_with_regressions": slices_with_regressions,
        "slices_blocking_uplift": slices_blocking_uplift,
        "headline": headline,
        "notes": notes if notes else [summary_note] if summary_note else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# UPLIFT COUNCIL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def adapt_perf_joint_view_for_council(
    perf_joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adapt perf_joint_view to the format expected by Uplift Council.

    Returns a dict that _evaluate_perf_dimension can process, containing:
    - slices_blocking_uplift: List of slice names that are blocked
    - slices_with_regressions: List of slice names with regressions
    - perf_risk: Overall performance risk level

    Args:
        perf_joint_view: Output from build_perf_joint_governance_view.

    Returns:
        Council-compatible perf_trend dict with slices_blocking_uplift and slices_with_regressions.
    """
    slices_blocking_uplift = perf_joint_view.get("slices_blocking_uplift", [])
    slices_with_regressions = perf_joint_view.get("slices_with_regressions", [])
    perf_risk = perf_joint_view.get("perf_risk", "LOW")

    # Filter out "all_slices" marker
    slices_blocking = [s for s in slices_blocking_uplift if s != "all_slices"]
    slices_regressions = [s for s in slices_with_regressions if s != "all_slices"]

    return {
        "slices_blocking_uplift": slices_blocking,
        "slices_with_regressions": slices_regressions,
        "perf_risk": perf_risk,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DELTA P TRAJECTORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def classify_uplift_shape(delta_p_trajectory: List[float]) -> str:
    """
    Classify uplift shape from Δp trajectory.

    Analyzes a sequence of delta_p values to determine if the trajectory is:
    - "monotonic": Consistently increasing or decreasing
    - "plateau": Relatively stable (low variance)
    - "oscillatory": Alternating increases and decreases

    Args:
        delta_p_trajectory: List of delta_p values over time.

    Returns:
        Shape descriptor: "monotonic", "plateau", or "oscillatory".
    """
    if len(delta_p_trajectory) < 2:
        return "plateau"  # Default for insufficient data

    if len(delta_p_trajectory) == 2:
        # Two points: check if increasing or decreasing
        if delta_p_trajectory[1] > delta_p_trajectory[0]:
            return "monotonic"
        elif delta_p_trajectory[1] < delta_p_trajectory[0]:
            return "monotonic"
        else:
            return "plateau"

    # Compute differences
    diffs = [
        delta_p_trajectory[i + 1] - delta_p_trajectory[i]
        for i in range(len(delta_p_trajectory) - 1)
    ]

    # Count sign changes
    sign_changes = 0
    for i in range(len(diffs) - 1):
        if (diffs[i] > 0 and diffs[i + 1] < 0) or (diffs[i] < 0 and diffs[i + 1] > 0):
            sign_changes += 1

    # Compute variance relative to mean
    mean_val = sum(delta_p_trajectory) / len(delta_p_trajectory)
    variance = sum((x - mean_val) ** 2 for x in delta_p_trajectory) / len(delta_p_trajectory)
    std_dev = variance ** 0.5
    coefficient_of_variation = std_dev / abs(mean_val) if mean_val != 0 else float('inf')

    # Classification logic
    # Plateau: low variance (CV < 0.1) or all differences near zero
    # Check this FIRST to avoid misclassifying stable data as oscillatory
    if coefficient_of_variation < 0.1 or all(abs(d) < 0.01 for d in diffs):
        return "plateau"

    # Oscillatory: frequent sign changes (at least 30% of transitions)
    if sign_changes >= len(diffs) * 0.3:
        return "oscillatory"

    # Monotonic: consistent direction (all positive or all negative)
    all_positive = all(d >= 0 for d in diffs)
    all_negative = all(d <= 0 for d in diffs)
    if all_positive or all_negative:
        return "monotonic"

    # Default: if not clearly oscillatory or plateau, treat as monotonic
    # (slight variations but overall trend)
    return "monotonic"


def map_uplift_shape_to_readiness_hint(
    uplift_shape: str,
    delta_p_trajectory: Optional[List[float]] = None,
) -> Dict[str, str]:
    """
    Map uplift shape to calibration readiness hint with transparency fields.

    Provides advisory hints about calibration readiness based on observed
    uplift shape. This is advisory only and does not gate any decisions.

    Args:
        uplift_shape: Shape descriptor from classify_uplift_shape().
        delta_p_trajectory: Optional trajectory for direction detection.

    Returns:
        Dictionary with:
        - "hint_schema_version": Schema version string (currently "1.0.0")
        - "hint": Readiness hint string ("READY_FOR_EXTENDED_RUN", "NEEDS_PARAMETER_TUNING", or "UNSTABLE_CALIBRATION")
        - "basis": Evidence basis string (e.g., "delta_p_trajectory_shape")
        - "scope_note": Scope clarification (always "ADVISORY_ONLY_NO_GATE" in v1.0.0)
    """
    hint_schema_version = "1.0.0"
    basis = "delta_p_trajectory_shape"
    scope_note = "ADVISORY_ONLY_NO_GATE"

    if uplift_shape == "monotonic":
        # Check if decreasing (improving divergence)
        if delta_p_trajectory and len(delta_p_trajectory) >= 2:
            if delta_p_trajectory[-1] < delta_p_trajectory[0]:
                # Decreasing: divergence improving
                return {
                    "hint_schema_version": hint_schema_version,
                    "hint": "READY_FOR_EXTENDED_RUN",
                    "basis": basis,
                    "scope_note": scope_note,
                }
        # Default for monotonic (increasing or unknown direction)
        return {
            "hint_schema_version": hint_schema_version,
            "hint": "READY_FOR_EXTENDED_RUN",
            "basis": basis,
            "scope_note": scope_note,
        }
    elif uplift_shape == "plateau":
        return {
            "hint_schema_version": hint_schema_version,
            "hint": "NEEDS_PARAMETER_TUNING",
            "basis": basis,
            "scope_note": scope_note,
        }
    elif uplift_shape == "oscillatory":
        return {
            "hint_schema_version": hint_schema_version,
            "hint": "UNSTABLE_CALIBRATION",
            "basis": basis,
            "scope_note": scope_note,
        }
    else:
        # Unknown shape: default to parameter tuning
        return {
            "hint_schema_version": hint_schema_version,
            "hint": "NEEDS_PARAMETER_TUNING",
            "basis": basis,
            "scope_note": scope_note,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST LIGHT PERF SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════


def build_first_light_perf_summary(
    perf_joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact perf summary for First Light evidence.

    This is a cross-check for First Light evidence packs. It provides a compact
    view of performance governance without any gating logic.

    Args:
        perf_joint_view: Output from build_perf_joint_governance_view.

    Returns:
        Compact summary with schema_version, perf_risk, slices_with_regressions,
        and slices_blocking_uplift.
    """
    # Validate that perf_joint_view has required keys
    required_keys = ["perf_risk", "slices_with_regressions", "slices_blocking_uplift"]
    for key in required_keys:
        if key not in perf_joint_view:
            raise PerfGovernanceInputError(
                f"perf_joint_view missing required key: {key}"
            )

    perf_risk = perf_joint_view["perf_risk"]
    slices_with_regressions = perf_joint_view["slices_with_regressions"]
    slices_blocking_uplift = perf_joint_view["slices_blocking_uplift"]

    # Filter out "all_slices" marker for cleaner output
    slices_regressions = [s for s in slices_with_regressions if s != "all_slices"]
    slices_blocking = [s for s in slices_blocking_uplift if s != "all_slices"]

    return {
        "schema_version": "1.0.0",
        "perf_risk": perf_risk,
        "slices_with_regressions": slices_regressions,
        "slices_blocking_uplift": slices_blocking,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE PACK INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def attach_perf_governance_tile_to_evidence(
    evidence_payload: Dict[str, Any],
    perf_joint_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach performance governance tile to an evidence payload.

    This is an advisory-only integration; the tile provides observability
    but does not enforce decisions.

    Args:
        evidence_payload: Evidence payload dictionary (will be modified in-place).
        perf_joint_view: Output from build_perf_joint_governance_view.

    Returns:
        Modified evidence_payload with perf governance tile attached under
        evidence["governance"]["uplift_perf"].
    """
    tile = build_uplift_perf_governance_tile(perf_joint_view)

    # Attach under governance.uplift_perf key
    if "governance" not in evidence_payload:
        evidence_payload["governance"] = {}

    evidence_payload["governance"]["uplift_perf"] = tile

    return evidence_payload


def attach_perf_governance_tile(
    evidence: Dict[str, Any],
    perf_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach performance governance tile to an evidence pack (read-only, additive).

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        perf_tile: Performance governance tile from build_uplift_perf_governance_tile().

    Returns:
        New dict with evidence contents plus uplift_perf tile attached.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> joint_view = build_perf_joint_governance_view(...)
        >>> tile = build_uplift_perf_governance_tile(joint_view)
        >>> enriched = attach_perf_governance_tile(evidence, tile)
        >>> "uplift_perf" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Attach performance governance tile
    if "governance" not in enriched:
        enriched["governance"] = {}
    enriched["governance"]["uplift_perf"] = perf_tile

    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════


def build_perf_calibration_summary(
    cal_exp1_data: Optional[Dict[str, Any]] = None,
    cal_exp2_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build performance calibration summary from CAL-EXP-1 and CAL-EXP-2 data.

    Analyzes Δp trajectory during calibration experiments and classifies
    the uplift shape (monotonic, plateau, oscillatory).

    Args:
        cal_exp1_data: CAL-EXP-1 report data (from cal_exp1_report.json).
            Expected format: {"windows": [{"mean_delta_p": float, ...}, ...]}
        cal_exp2_data: CAL-EXP-2 report data (optional, future).
            Expected format: {"windows": [{"mean_delta_p": float, ...}, ...]}

    Returns:
        Calibration summary with schema_version, experiments, uplift_shapes,
        and delta_p_trajectories.
    """
    summary = {
        "schema_version": "1.0.0",
        "experiments": {},
        "uplift_shapes": {},
        "delta_p_trajectories": {},
        "calibration_readiness_hints": {},
    }

    # Process CAL-EXP-1
    if cal_exp1_data:
        windows = cal_exp1_data.get("windows", [])
        delta_p_values = [w.get("mean_delta_p", 0.0) for w in windows if "mean_delta_p" in w]

        if delta_p_values:
            uplift_shape = classify_uplift_shape(delta_p_values)
            readiness_hint_dict = map_uplift_shape_to_readiness_hint(
                uplift_shape, delta_p_values
            )
            summary["experiments"]["cal_exp1"] = {
                "windows_count": len(windows),
                "trajectory_length": len(delta_p_values),
            }
            summary["uplift_shapes"]["cal_exp1"] = uplift_shape
            summary["delta_p_trajectories"]["cal_exp1"] = delta_p_values
            summary["calibration_readiness_hints"]["cal_exp1"] = readiness_hint_dict

    # Process CAL-EXP-2 (when available)
    if cal_exp2_data:
        windows = cal_exp2_data.get("windows", [])
        delta_p_values = [w.get("mean_delta_p", 0.0) for w in windows if "mean_delta_p" in w]

        if delta_p_values:
            uplift_shape = classify_uplift_shape(delta_p_values)
            readiness_hint_dict = map_uplift_shape_to_readiness_hint(
                uplift_shape, delta_p_values
            )
            summary["experiments"]["cal_exp2"] = {
                "windows_count": len(windows),
                "trajectory_length": len(delta_p_values),
            }
            summary["uplift_shapes"]["cal_exp2"] = uplift_shape
            summary["delta_p_trajectories"]["cal_exp2"] = delta_p_values
            summary["calibration_readiness_hints"]["cal_exp2"] = readiness_hint_dict

    return summary


def _extract_hint_string(hint_value: Any) -> str:
    """
    Extract hint string from hint value (supports both old string and new dict format).

    Args:
        hint_value: Either a string (old format) or dict with "hint" key (new format).

    Returns:
        Hint string.
    """
    if isinstance(hint_value, dict):
        return hint_value.get("hint", "UNKNOWN")
    elif isinstance(hint_value, str):
        # Backward compatibility: old string format
        return hint_value
    else:
        return "UNKNOWN"


def extract_calibration_readiness_status(
    calibration_summary: Dict[str, Any],
) -> str:
    """
    Extract one-line calibration readiness status hint for status JSON.

    Provides a compact, one-line hint about calibration readiness based on
    the most concerning hint across all experiments. This is advisory only.

    Supports both old string format and new dict format for backward compatibility.

    Args:
        calibration_summary: Output from build_perf_calibration_summary().

    Returns:
        One-line status hint string, or "UNKNOWN" if no hints available.
    """
    hints = calibration_summary.get("calibration_readiness_hints", {})
    if not hints:
        return "UNKNOWN"

    # Extract hint strings (handles both old string and new dict format)
    hint_strings = [_extract_hint_string(h) for h in hints.values()]

    # Priority order: UNSTABLE > NEEDS_TUNING > READY
    if any(h == "UNSTABLE_CALIBRATION" for h in hint_strings):
        return "UNSTABLE_CALIBRATION"
    elif any(h == "NEEDS_PARAMETER_TUNING" for h in hint_strings):
        return "NEEDS_PARAMETER_TUNING"
    elif any(h == "READY_FOR_EXTENDED_RUN" for h in hint_strings):
        return "READY_FOR_EXTENDED_RUN"
    else:
        return "UNKNOWN"


def attach_perf_calibration_summary(
    evidence: Dict[str, Any],
    calibration_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach performance calibration summary to an evidence pack (read-only, additive).

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the summary attached.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        calibration_summary: Calibration summary from build_perf_calibration_summary().

    Returns:
        New dict with evidence contents plus perf_calibration_summary attached.

    Example:
        >>> evidence = {"version": "1.0.0", "artifacts": []}
        >>> cal_exp1 = load_cal_exp1_report("cal_exp1_report.json")
        >>> summary = build_perf_calibration_summary(cal_exp1_data=cal_exp1)
        >>> enriched = attach_perf_calibration_summary(evidence, summary)
        >>> "perf_calibration_summary" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Attach performance calibration summary
    if "governance" not in enriched:
        enriched["governance"] = {}
    enriched["governance"]["perf_calibration_summary"] = calibration_summary

    return enriched


def attach_first_light_perf_summary(
    evidence: Dict[str, Any],
    perf_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach First Light perf summary to an evidence pack (read-only, additive).

    This is a lightweight helper for First Light evidence builders. It does not
    modify the input evidence dict, but returns a new dict with the summary attached.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        perf_summary: Performance summary from build_first_light_perf_summary().

    Returns:
        New dict with evidence contents plus uplift_perf_summary attached.

    Example:
        >>> evidence = {"version": "1.0.0", "artifacts": []}
        >>> joint_view = build_perf_joint_governance_view(...)
        >>> summary = build_first_light_perf_summary(joint_view)
        >>> enriched = attach_first_light_perf_summary(evidence, summary)
        >>> "uplift_perf_summary" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Attach First Light perf summary
    if "governance" not in enriched:
        enriched["governance"] = {}
    enriched["governance"]["uplift_perf_summary"] = perf_summary

    return enriched


__all__ = [
    "PerfGovernanceInputError",
    "validate_perf_trend",
    "validate_budget_trend",
    "validate_metric_conformance",
    "validate_perf_governance_inputs",
    "build_perf_joint_governance_view",
    "summarize_perf_for_global_release",
    "summarize_perf_for_global_console",
    "build_uplift_perf_governance_tile",
    "build_first_light_perf_summary",
    "classify_uplift_shape",
    "map_uplift_shape_to_readiness_hint",
    "build_perf_calibration_summary",
    "adapt_perf_joint_view_for_council",
    "attach_perf_governance_tile_to_evidence",
    "attach_perf_governance_tile",
    "attach_first_light_perf_summary",
    "attach_perf_calibration_summary",
    "extract_calibration_readiness_status",
]

