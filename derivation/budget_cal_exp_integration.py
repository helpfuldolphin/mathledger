"""
Budget Storyline Integration for CAL-EXP Reports (P5 Calibration).

This module provides helpers to embed budget_storyline_summary into CAL-EXP-1/2/3
calibration experiment reports. The budget storyline serves as an "Energy Drift Monitor"
for P5 calibration, tracking budget invariant health across calibration windows.

SHADOW MODE CONTRACT:
- All functions are read-only (aside from dict construction)
- Budget storyline data is purely observational
- No gating decisions based on budget storyline in Phase X/P5 POC
- Phase Y only for P5AcceptanceGate integration

Author: Agent B1 (Budget Enforcement Architect)
Date: 2025-01-XX
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from derivation.budget_invariants import (
    build_first_light_budget_storyline,
    build_budget_invariant_timeline,
    build_budget_storyline,
    project_budget_stability_horizon,
)


def attach_budget_storyline_to_cal_exp_report(
    report: Dict[str, Any],
    timeline: Dict[str, Any],
    storyline: Dict[str, Any],
    projection: Dict[str, Any],
    experiment_id: str,
    run_id: str,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Attach budget storyline summary to a CAL-EXP report.
    
    Non-mutating: returns a new dict with budget_storyline_summary attached.
    This embeds the budget storyline as an "Energy Drift Monitor" for P5 calibration,
    tracking budget invariant health across calibration experiment windows.
    
    Args:
        report: Existing CAL-EXP report dict (read-only, not modified).
        timeline: Budget invariant timeline from build_budget_invariant_timeline().
        storyline: Budget storyline from build_budget_storyline().
        projection: BNH-Φ projection from project_budget_stability_horizon().
        experiment_id: Experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3").
        run_id: Run identifier (e.g., "cal_exp1_20250101_120000").
        window_start: Optional window start index (for CAL-EXP-2/3 window bounds).
        window_end: Optional window end index (for CAL-EXP-2/3 window bounds).
    
    Returns:
        New dict with report contents plus budget_storyline_summary attached.
    
    Structure:
        {
            ...existing report fields...,
            "budget_storyline_summary": {
                "schema_version": "1.0.0",
                "experiment_id": "CAL-EXP-1",
                "run_id": "...",
                "window_start": Optional[int],
                "window_end": Optional[int],
                "combined_status": "OK" | "WARN" | "BLOCK",
                "stability_index": float,
                "episodes_count": int,
                "projection_class": "STABLE" | "DRIFTING" | "VOLATILE",
                "key_structural_events": List[str],
            }
        }
    """
    # Create a copy to avoid mutating the original
    enriched = report.copy()
    
    # Build First Light storyline summary
    storyline_summary = build_first_light_budget_storyline(
        timeline=timeline,
        storyline=storyline,
        projection=projection,
    )
    
    # Add experiment metadata
    storyline_summary["experiment_id"] = experiment_id
    storyline_summary["run_id"] = run_id
    if window_start is not None:
        storyline_summary["window_start"] = window_start
    if window_end is not None:
        storyline_summary["window_end"] = window_end
    
    # Attach to report
    enriched["budget_storyline_summary"] = storyline_summary
    
    return enriched


def extract_budget_storyline_from_cal_exp_report(
    report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract budget storyline summary from a CAL-EXP report.
    
    Helper function to retrieve budget_storyline_summary if present in the report.
    
    Args:
        report: CAL-EXP report dict.
    
    Returns:
        Budget storyline summary dict if present, None otherwise.
    """
    return report.get("budget_storyline_summary")


def build_cal_exp_budget_storyline_from_snapshots(
    snapshots: list[Any],  # List of PipelineStats or compatible dicts
    budget_health_history: list[Dict[str, Any]],
    experiment_id: str,
    run_id: str,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build budget storyline summary from snapshots and health history for CAL-EXP report.
    
    Convenience function that builds timeline, storyline, and projection from raw data
    and produces a complete budget storyline summary ready for CAL-EXP report embedding.
    
    Args:
        snapshots: Sequence of budget invariant snapshots (from build_budget_invariant_snapshot).
        budget_health_history: Sequence of budget health objects (chronological).
        experiment_id: Experiment identifier (e.g., "CAL-EXP-1").
        run_id: Run identifier.
        window_start: Optional window start index.
        window_end: Optional window end index.
    
    Returns:
        Budget storyline summary dict ready for attach_budget_storyline_to_cal_exp_report().
    """
    from derivation.budget_invariants import build_budget_invariant_snapshot
    
    # Build timeline from snapshots
    timeline = build_budget_invariant_timeline(snapshots)
    
    # Build storyline
    storyline = build_budget_storyline(timeline, budget_health_history)
    
    # Build projection
    projection = project_budget_stability_horizon(budget_health_history)
    
    # Build First Light summary
    summary = build_first_light_budget_storyline(timeline, storyline, projection)
    
    # Add experiment metadata
    summary["experiment_id"] = experiment_id
    summary["run_id"] = run_id
    if window_start is not None:
        summary["window_start"] = window_start
    if window_end is not None:
        summary["window_end"] = window_end
    
    return summary


def annotate_cal_exp_windows_with_budget_storyline(
    report: Dict[str, Any],
    budget_storyline_summary: Dict[str, Any],
    confound_stability_threshold: float = 0.95,
    strict_confounding: bool = True,
) -> Dict[str, Any]:
    """
    Annotate CAL-EXP report windows with budget storyline data.
    
    For each window in the report, attaches budget confounding annotations:
    - budget_combined_status: "OK" | "WARN" | "BLOCK"
    - budget_stability_index: float
    - budget_projection_class: "STABLE" | "DRIFTING" | "VOLATILE" | "UNKNOWN"
    - budget_confounded: bool
    - budget_confound_reason: str | None (explanation for confounding)
    - confound_stability_threshold: float (threshold used for confounding calculation)
    
    Confounding Logic (strict_confounding=True, default):
        budget_confounded = (budget_combined_status in {"WARN", "BLOCK"}) AND 
                           (budget_stability_index < confound_stability_threshold)
        
        This requires BOTH conditions to be true, preventing over-flagging.
        
    Legacy Logic (strict_confounding=False):
        budget_confounded = (budget_combined_status != "OK") OR 
                           (budget_stability_index < confound_stability_threshold)
        
        This uses the original OR-rule for backward compatibility.
    
    Reason Codes (strict_confounding=True):
        - "STATUS_WARN": status is WARN (stability_index may be >= threshold)
        - "STATUS_BLOCK": status is BLOCK (stability_index may be >= threshold)
        - "LOW_STABILITY_INDEX": stability_index < threshold (status is OK)
        - "STATUS_AND_LOW_STABILITY": both status in {WARN, BLOCK} AND stability_index < threshold
        - None: not confounded (status OK AND stability_index >= threshold)
    
    Reason Codes (strict_confounding=False, legacy):
        - Set when confounded is True (explains which condition triggered)
        - None when confounded is False
    
    Non-mutating: returns a new dict with annotated windows.
    Windows are sorted by window index to ensure deterministic ordering.
    
    Args:
        report: CAL-EXP report dict (read-only, not modified).
        budget_storyline_summary: Budget storyline summary dict.
        confound_stability_threshold: Stability index threshold for confounding (default 0.95).
        strict_confounding: If True, use AND-rule (default). If False, use legacy OR-rule.
    
    Returns:
        New dict with report contents plus annotated windows.
        
    Note:
        The budget_confounded flag indicates potential confounding in calibration
        analysis. When True, downstream analysis should consider budget drift as
        a potential factor in divergence patterns. This is observational only;
        no gating logic is applied. The "confounded" label is for interpretation
        guidance, not exclusion and not a gate.
    """
    # Create a copy to avoid mutating the original
    enriched = report.copy()
    
    # Extract budget fields
    combined_status = budget_storyline_summary.get("combined_status", "OK")
    stability_index = budget_storyline_summary.get("stability_index", 1.0)
    projection_class = budget_storyline_summary.get("projection_class", "UNKNOWN")
    
    # Calculate confounded flag and reason
    if strict_confounding:
        # New strict rule: requires BOTH conditions
        status_concern = combined_status in {"WARN", "BLOCK"}
        stability_low = stability_index < confound_stability_threshold
        
        if status_concern and stability_low:
            budget_confounded = True
            budget_confound_reason = "STATUS_AND_LOW_STABILITY"
        elif status_concern:
            budget_confounded = False  # Not confounded (stability OK)
            budget_confound_reason = "STATUS_WARN" if combined_status == "WARN" else "STATUS_BLOCK"
        elif stability_low:
            budget_confounded = False  # Not confounded (status OK)
            budget_confound_reason = "LOW_STABILITY_INDEX"
        else:
            budget_confounded = False  # Not confounded (both OK)
            budget_confound_reason = None
    else:
        # Legacy OR-rule for backward compatibility
        status_concern = combined_status != "OK"
        stability_low = stability_index < confound_stability_threshold
        
        budget_confounded = status_concern or stability_low
        
        # Set reason based on which condition(s) triggered
        if status_concern and stability_low:
            budget_confound_reason = "STATUS_AND_LOW_STABILITY"
        elif status_concern:
            budget_confound_reason = "STATUS_WARN" if combined_status == "WARN" else "STATUS_BLOCK"
        elif stability_low:
            budget_confound_reason = "LOW_STABILITY_INDEX"
        else:
            budget_confound_reason = None
    
    # Annotate windows
    windows = enriched.get("windows", [])
    annotated_windows = []
    
    # Create list with (index, window) for sorting
    window_list = []
    for idx, window in enumerate(windows):
        window_list.append((idx, window))
    
    # Sort by index for deterministic ordering
    window_list.sort(key=lambda x: x[0])
    
    # Annotate each window
    for idx, window in window_list:
        annotated_window = window.copy()
        annotated_window["budget_combined_status"] = combined_status
        annotated_window["budget_stability_index"] = stability_index
        annotated_window["budget_projection_class"] = projection_class
        annotated_window["budget_confounded"] = budget_confounded
        annotated_window["budget_confound_reason"] = budget_confound_reason
        annotated_window["confound_stability_threshold"] = confound_stability_threshold
        annotated_windows.append(annotated_window)
    
    enriched["windows"] = annotated_windows
    
    return enriched


def build_budget_confounding_truth_table(
    confound_stability_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Build a canonical truth table for budget confounding semantics.
    
    Enumerates all combinations of status and stability_index to produce
    a machine-readable reference for confounding logic in both strict
    (AND-rule) and legacy (OR-rule) modes.
    
    Args:
        confound_stability_threshold: Stability index threshold (default: 0.95).
    
    Returns:
        Dict with schema:
        {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "confound_stability_threshold": float,
            "truth_table": [
                {
                    "status": "OK" | "WARN" | "BLOCK",
                    "stability_index": float,
                    "stability_relation": "above_threshold" | "below_threshold",
                    "strict_confounding_result": bool,
                    "legacy_confounding_result": bool,
                    "strict_confound_reason": str | None,
                    "legacy_confound_reason": str | None,
                },
                ...
            ]
        }
    """
    epsilon = 0.01  # Small offset to represent threshold+ε and threshold-ε
    
    statuses = ["OK", "WARN", "BLOCK"]
    stability_values = [
        (confound_stability_threshold + epsilon, "above_threshold"),
        (confound_stability_threshold - epsilon, "below_threshold"),
    ]
    
    truth_table = []
    
    for status in statuses:
        for stability_index, stability_relation in stability_values:
            # Strict mode (AND-rule)
            status_concern = status in {"WARN", "BLOCK"}
            stability_low = stability_index < confound_stability_threshold
            
            if status_concern and stability_low:
                strict_confounded = True
                strict_reason = "STATUS_AND_LOW_STABILITY"
            elif status_concern:
                strict_confounded = False
                strict_reason = "STATUS_WARN" if status == "WARN" else "STATUS_BLOCK"
            elif stability_low:
                strict_confounded = False
                strict_reason = "LOW_STABILITY_INDEX"
            else:
                strict_confounded = False
                strict_reason = None
            
            # Legacy mode (OR-rule)
            status_concern_legacy = status != "OK"
            legacy_confounded = status_concern_legacy or stability_low
            
            if status_concern_legacy and stability_low:
                legacy_reason = "STATUS_AND_LOW_STABILITY"
            elif status_concern_legacy:
                legacy_reason = "STATUS_WARN" if status == "WARN" else "STATUS_BLOCK"
            elif stability_low:
                legacy_reason = "LOW_STABILITY_INDEX"
            else:
                legacy_reason = None
            
            truth_table.append({
                "status": status,
                "stability_index": stability_index,
                "stability_relation": stability_relation,
                "strict_confounding_result": strict_confounded,
                "legacy_confounding_result": legacy_confounded,
                "strict_confound_reason": strict_reason,
                "legacy_confound_reason": legacy_reason,
            })
    
    return {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "confound_stability_threshold": confound_stability_threshold,
        "truth_table": truth_table,
    }


def validate_budget_confounding_defaults(
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Validate that annotated windows include required budget confounding fields.
    
    Returns a list of structured warnings with codes if validation issues are found.
    This is observational only—no exceptions are raised.
    
    Args:
        report: CAL-EXP report dict (may contain annotated windows).
    
    Returns:
        List of warning dicts with keys: "code", "message", "window_index".
        Empty list if no issues found.
        Warnings are deterministically ordered by window_index.
    """
    warnings: List[Dict[str, Any]] = []
    
    windows = report.get("windows", [])
    if not windows:
        return warnings  # No windows to validate
    
    # Warning codes (stable, machine-readable)
    CODE_MISSING_REASON = "BUDGET-DEF-001"
    CODE_MISSING_THRESHOLD = "BUDGET-DEF-002"
    
    for idx, window in enumerate(windows):
        window_ref = f"window[{idx}]"
        
        # Check for budget_confounded field (if present, reason must be explicit)
        if "budget_confounded" in window:
            if "budget_confound_reason" not in window:
                warnings.append({
                    "code": CODE_MISSING_REASON,
                    "message": f"{window_ref}: budget_confounded present but budget_confound_reason missing",
                    "window_index": idx,
                })
        elif "budget_confound_reason" in window:
            # Reason present without confounded flag is unusual but not invalid
            pass
        
        # Check for confound_stability_threshold (should be present if any budget fields exist)
        has_budget_fields = any(
            key.startswith("budget_") for key in window.keys()
        )
        if has_budget_fields and "confound_stability_threshold" not in window:
            warnings.append({
                "code": CODE_MISSING_THRESHOLD,
                "message": f"{window_ref}: budget fields present but confound_stability_threshold missing",
                "window_index": idx,
            })
    
    # Sort deterministically by window_index (stable ordering)
    warnings.sort(key=lambda w: w["window_index"])
    
    return warnings



