"""
Runtime Profile Calibration Correlation (SHADOW MODE)

Correlates runtime profile metrics (profile_stability, no_run_rate) with
CAL-EXP-1 window metrics (mean_delta_p, divergence_rate) for calibration analysis.

SHADOW MODE CONTRACT:
- All correlations are advisory only
- No gating or threshold enforcement
- Missing data handled gracefully
- Deterministic output (Pearson correlation)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import math


def compute_pearson_correlation(
    x: List[float], y: List[float]
) -> Union[float, Dict[str, Union[str, None]]]:
    """
    Compute Pearson correlation coefficient between two lists.

    Returns correlation value in [-1, 1] or dict with "value": None and "reason" field.

    Args:
        x: First variable values
        y: Second variable values (must be same length as x)

    Returns:
        Either float in [-1, 1] or dict with {"value": None, "reason": "..."}
        Reasons: "INSUFFICIENT_POINTS", "ZERO_VARIANCE_X", "ZERO_VARIANCE_Y", "NON_NUMERIC_INPUT"
    """
    # Validate input types
    try:
        x_float = [float(v) for v in x]
        y_float = [float(v) for v in y]
    except (ValueError, TypeError):
        return {"value": None, "reason": "NON_NUMERIC_INPUT"}

    if len(x_float) != len(y_float) or len(x_float) < 2:
        return {"value": None, "reason": "INSUFFICIENT_POINTS"}

    # Compute means
    mean_x = sum(x_float) / len(x_float)
    mean_y = sum(y_float) / len(y_float)

    # Compute numerator (covariance)
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_float, y_float))

    # Compute denominators (standard deviations)
    var_x = sum((xi - mean_x) ** 2 for xi in x_float)
    var_y = sum((yi - mean_y) ** 2 for yi in y_float)

    # Check for zero variance (with small epsilon for floating point)
    epsilon = 1e-10
    if var_x < epsilon:
        return {"value": None, "reason": "ZERO_VARIANCE_X"}
    if var_y < epsilon:
        return {"value": None, "reason": "ZERO_VARIANCE_Y"}

    # Compute correlation
    denominator = math.sqrt(var_x * var_y)
    if denominator < epsilon:
        return {"value": None, "reason": "ZERO_VARIANCE_X"}  # Fallback (shouldn't happen)

    correlation = numerator / denominator
    # Clamp to [-1, 1] for numerical stability and round to 6 decimals
    correlation = max(-1.0, min(1.0, correlation))
    return round(correlation, 6)


def correlate_runtime_profile_with_cal_windows(
    runtime_profile_snapshot: Dict[str, any],
    cal_exp1_windows: List[Dict[str, any]],
) -> Dict[str, any]:
    """
    Correlate runtime profile metrics with CAL-EXP-1 window metrics.

    SHADOW MODE: Advisory only, no gating.

    Args:
        runtime_profile_snapshot: Runtime profile snapshot with profile_stability, no_run_rate
        cal_exp1_windows: List of CAL-EXP-1 window dictionaries

    Returns:
        Dictionary with windowed correlations and summary
    """
    if not cal_exp1_windows:
        return {
            "schema_version": "1.0.0",
            "windows_analyzed": 0,
            "correlations": {},
            "advisory_note": "No CAL-EXP-1 windows provided",
        }

    profile_stability = runtime_profile_snapshot.get("profile_stability")
    no_run_rate = runtime_profile_snapshot.get("no_run_rate")

    if profile_stability is None and no_run_rate is None:
        return {
            "schema_version": "1.0.0",
            "windows_analyzed": len(cal_exp1_windows),
            "correlations": {},
            "advisory_note": "Runtime profile snapshot missing profile_stability and no_run_rate",
        }

    # Extract window metrics with missingness tracking
    divergence_rates = []
    mean_delta_ps = []
    windows_dropped = []
    missing_fields_by_window: Dict[int, List[str]] = {}

    for idx, window in enumerate(cal_exp1_windows):
        missing_fields = []
        div_rate = window.get("divergence_rate")
        delta_p = window.get("mean_delta_p")

        if div_rate is None:
            missing_fields.append("divergence_rate")
        else:
            try:
                divergence_rates.append(float(div_rate))
            except (ValueError, TypeError):
                missing_fields.append("divergence_rate (non-numeric)")

        if delta_p is None:
            missing_fields.append("mean_delta_p")
        else:
            try:
                mean_delta_ps.append(float(delta_p))
            except (ValueError, TypeError):
                missing_fields.append("mean_delta_p (non-numeric)")

        if missing_fields:
            missing_fields_by_window[idx] = missing_fields
            # Drop window if both fields missing
            if "divergence_rate" in missing_fields and "mean_delta_p" in missing_fields:
                windows_dropped.append(idx)

    correlations: Dict[str, Union[float, Dict[str, Union[str, None]]]] = {}

    # Correlate profile_stability with divergence_rate
    if profile_stability is not None and len(divergence_rates) >= 2:
        # Create constant list for profile_stability (single value for all windows)
        stability_list = [profile_stability] * len(divergence_rates)
        corr_result = compute_pearson_correlation(stability_list, divergence_rates)
        if isinstance(corr_result, dict):
            correlations["profile_stability_vs_divergence_rate"] = corr_result.get("value")
        else:
            correlations["profile_stability_vs_divergence_rate"] = corr_result

    # Correlate profile_stability with mean_delta_p
    if profile_stability is not None and len(mean_delta_ps) >= 2:
        stability_list = [profile_stability] * len(mean_delta_ps)
        corr_result = compute_pearson_correlation(stability_list, mean_delta_ps)
        if isinstance(corr_result, dict):
            correlations["profile_stability_vs_mean_delta_p"] = corr_result.get("value")
        else:
            correlations["profile_stability_vs_mean_delta_p"] = corr_result

    # Correlate no_run_rate with divergence_rate
    if no_run_rate is not None and len(divergence_rates) >= 2:
        no_run_list = [no_run_rate] * len(divergence_rates)
        corr_result = compute_pearson_correlation(no_run_list, divergence_rates)
        if isinstance(corr_result, dict):
            correlations["no_run_rate_vs_divergence_rate"] = corr_result.get("value")
        else:
            correlations["no_run_rate_vs_divergence_rate"] = corr_result

    # Correlate no_run_rate with mean_delta_p
    if no_run_rate is not None and len(mean_delta_ps) >= 2:
        no_run_list = [no_run_rate] * len(mean_delta_ps)
        corr_result = compute_pearson_correlation(no_run_list, mean_delta_ps)
        if isinstance(corr_result, dict):
            correlations["no_run_rate_vs_mean_delta_p"] = corr_result.get("value")
        else:
            correlations["no_run_rate_vs_mean_delta_p"] = corr_result

    # Build correlation reasons summary
    correlation_reasons: Dict[str, str] = {}
    # Re-compute correlations to extract reasons
    if profile_stability is not None and len(divergence_rates) >= 2:
        stability_list = [profile_stability] * len(divergence_rates)
        corr_result = compute_pearson_correlation(stability_list, divergence_rates)
        if isinstance(corr_result, dict) and corr_result.get("value") is None:
            correlation_reasons["profile_stability_vs_divergence_rate"] = corr_result.get("reason", "UNKNOWN")
    
    if profile_stability is not None and len(mean_delta_ps) >= 2:
        stability_list = [profile_stability] * len(mean_delta_ps)
        corr_result = compute_pearson_correlation(stability_list, mean_delta_ps)
        if isinstance(corr_result, dict) and corr_result.get("value") is None:
            correlation_reasons["profile_stability_vs_mean_delta_p"] = corr_result.get("reason", "UNKNOWN")
    
    if no_run_rate is not None and len(divergence_rates) >= 2:
        no_run_list = [no_run_rate] * len(divergence_rates)
        corr_result = compute_pearson_correlation(no_run_list, divergence_rates)
        if isinstance(corr_result, dict) and corr_result.get("value") is None:
            correlation_reasons["no_run_rate_vs_divergence_rate"] = corr_result.get("reason", "UNKNOWN")
    
    if no_run_rate is not None and len(mean_delta_ps) >= 2:
        no_run_list = [no_run_rate] * len(mean_delta_ps)
        corr_result = compute_pearson_correlation(no_run_list, mean_delta_ps)
        if isinstance(corr_result, dict) and corr_result.get("value") is None:
            correlation_reasons["no_run_rate_vs_mean_delta_p"] = corr_result.get("reason", "UNKNOWN")

    return {
        "schema_version": "1.0.0",
        "windows_analyzed": len(cal_exp1_windows),
        "windows_dropped": windows_dropped,
        "missing_fields_by_window": missing_fields_by_window,
        "runtime_profile_metrics": {
            "profile_stability": profile_stability,
            "no_run_rate": no_run_rate,
        },
        "correlations": correlations,
        "correlation_reasons": correlation_reasons,
        "advisory_note": (
            "Correlations computed using single runtime profile snapshot "
            "against all CAL-EXP-1 windows. Constant profile metrics may "
            "result in undefined correlations (see correlation_reasons)."
        ),
    }


def annotate_cal_windows_with_runtime_confounding(
    runtime_profile_snapshot: Dict[str, any],
    cal_exp1_windows: List[Dict[str, any]],
    divergence_spike_threshold: float = 0.8,
    delta_p_threshold: float = 0.05,
) -> Tuple[List[Dict[str, any]], Dict[str, any]]:
    """
    Annotate CAL-EXP-1 windows with runtime_profile_confounded flag.

    A window is confounded if:
    - runtime_profile status_light == "RED" AND
    - (window divergence_rate >= divergence_spike_threshold OR
       window mean_delta_p >= delta_p_threshold)

    SHADOW MODE: Advisory only, no gating.

    Args:
        runtime_profile_snapshot: Runtime profile snapshot with status_light
        cal_exp1_windows: List of CAL-EXP-1 window dictionaries
        divergence_spike_threshold: Threshold for divergence spike detection (default: 0.8)
        delta_p_threshold: Threshold for mean_delta_p spike detection (default: 0.05)

    Returns:
        Tuple of (annotated_windows, summary_dict)
    """
    status_light = runtime_profile_snapshot.get("status_light")
    profile_name = runtime_profile_snapshot.get("profile", "unknown")

    confounded_count = 0
    confounded_windows: List[int] = []
    confound_reasons: Dict[int, List[str]] = {}

    # Check if runtime profile is RED
    is_red = status_light == "RED"

    annotated_windows = []
    for idx, window in enumerate(cal_exp1_windows):
        # Create a copy to avoid mutating original
        annotated_window = dict(window)
        annotated_window["runtime_profile_confounded"] = False

        if is_red:
            confound_reasons_list = []
            divergence_rate = window.get("divergence_rate")
            mean_delta_p = window.get("mean_delta_p")

            # Check divergence_rate threshold
            if divergence_rate is not None:
                try:
                    div_rate_float = float(divergence_rate)
                    if div_rate_float >= divergence_spike_threshold:
                        confound_reasons_list.append("divergence_rate")
                except (ValueError, TypeError):
                    pass  # Non-numeric, skip

            # Check mean_delta_p threshold
            if mean_delta_p is not None:
                try:
                    delta_p_float = float(mean_delta_p)
                    if abs(delta_p_float) >= delta_p_threshold:  # Use absolute value
                        confound_reasons_list.append("mean_delta_p")
                except (ValueError, TypeError):
                    pass  # Non-numeric, skip

            # Window is confounded if either condition met
            if confound_reasons_list:
                annotated_window["runtime_profile_confounded"] = True
                confounded_count += 1
                confounded_windows.append(idx)
                confound_reasons[idx] = confound_reasons_list

        annotated_windows.append(annotated_window)

    # Build advisory note with counterfactual control
    if is_red and confounded_count == 0:
        advisory_note = (
            f"Runtime profile status is RED but no confounding detected "
            f"(divergence_rate < {divergence_spike_threshold} and "
            f"|mean_delta_p| < {delta_p_threshold}). "
            "Runtime instability present but not correlated with calibration instability."
        )
    elif is_red:
        advisory_note = (
            f"Detected {confounded_count} confounded windows "
            f"(status_light=RED with divergence_rate >= {divergence_spike_threshold} "
            f"or |mean_delta_p| >= {delta_p_threshold}). "
            "This is advisory only and does not invalidate calibration results."
        )
    else:
        advisory_note = (
            "No confounding detected (status_light != RED). "
            "This is advisory only and does not invalidate calibration results."
        )

    summary = {
        "schema_version": "1.0.0",
        "profile": profile_name,
        "status_light": status_light,
        "is_red": is_red,
        "confounding_thresholds": {
            "divergence_rate": divergence_spike_threshold,
            "mean_delta_p": delta_p_threshold,
        },
        "total_windows": len(cal_exp1_windows),
        "confounded_windows_count": confounded_count,
        "confounded_window_indices": confounded_windows,
        "confound_reasons": confound_reasons,
        "advisory_note": advisory_note,
    }

    return annotated_windows, summary


def decompose_divergence_components(
    windows: List[Dict[str, Any]],
    state_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Decompose divergence into component rates (state vs outcome divergences).

    SHADOW MODE: Advisory only, no gating.

    This function computes per-window and overall rates for:
    - state_divergence_rate: H/rho/tau/beta deltas exceeding threshold (via mean_delta_p proxy)
    - outcome_divergence_rate_success: Twin predicted success incorrectly (requires per-cycle data)
    - outcome_divergence_rate_omega: Twin predicted omega incorrectly (requires per-cycle data)
    - outcome_divergence_rate_blocked: Twin predicted blocked incorrectly (requires per-cycle data)
    - overall_any_divergence_rate: Legacy behavior (from divergence_rate field)

    Args:
        windows: List of CAL-EXP-1 window dictionaries
        state_threshold: Threshold for state divergence detection (default: 0.05)

    Returns:
        Dictionary with decomposed divergence rates, thresholds used, and missing field lists
    """
    if not windows:
        return {
            "schema_version": "1.0.0",
            "error": "INSUFFICIENT_DATA",
            "missing_fields": ["windows"],
            "advisory_note": "No windows provided for decomposition",
        }

    # Track missing fields
    missing_fields: List[str] = []
    per_window_decompositions: List[Dict[str, Any]] = []

    # Aggregate counters
    total_cycles = 0
    total_state_diverged_cycles = 0
    total_any_diverged_cycles = 0
    windows_with_outcome_data = 0

    for idx, window in enumerate(windows):
        window_decomp: Dict[str, Any] = {
            "window_index": idx,
            "state_divergence_rate": None,
            "outcome_divergence_rate_success": None,
            "outcome_divergence_rate_omega": None,
            "outcome_divergence_rate_blocked": None,
            "overall_any_divergence_rate": None,
        }

        cycles_in_window = window.get("cycles_in_window") or window.get("window_size")
        if cycles_in_window is None:
            missing_fields.append(f"windows[{idx}].cycles_in_window")
            continue

        total_cycles += cycles_in_window

        # State divergence: use mean_delta_p > threshold as proxy
        # Note: This is approximate since mean_delta_p is averaged, but if mean > threshold,
        # many individual cycles likely exceeded it
        mean_delta_p = window.get("mean_delta_p")
        if mean_delta_p is not None:
            try:
                mean_delta_p_float = float(mean_delta_p)
                # Approximate: if mean_delta_p > threshold, assume all cycles diverged
                # This is conservative but avoids recomputing from per-cycle data
                state_diverged = mean_delta_p_float > state_threshold
                window_decomp["state_divergence_rate"] = 1.0 if state_diverged else 0.0
                if state_diverged:
                    total_state_diverged_cycles += cycles_in_window
            except (ValueError, TypeError):
                missing_fields.append(f"windows[{idx}].mean_delta_p (non-numeric)")
        else:
            missing_fields.append(f"windows[{idx}].mean_delta_p")

        # Overall any divergence rate: use divergence_rate field
        divergence_rate = window.get("divergence_rate")
        if divergence_rate is not None:
            try:
                div_rate_float = float(divergence_rate)
                window_decomp["overall_any_divergence_rate"] = div_rate_float
                total_any_diverged_cycles += int(div_rate_float * cycles_in_window)
            except (ValueError, TypeError):
                missing_fields.append(f"windows[{idx}].divergence_rate (non-numeric)")
        else:
            missing_fields.append(f"windows[{idx}].divergence_rate")

        # Outcome divergences: not available in aggregated windows
        # Would require per-cycle divergence snapshots (success_diverged, omega_diverged, blocked_diverged)
        # Mark as None with explicit reason
        window_decomp["outcome_divergence_rate_success"] = None
        window_decomp["outcome_divergence_rate_omega"] = None
        window_decomp["outcome_divergence_rate_blocked"] = None

        per_window_decompositions.append(window_decomp)

    # Compute overall rates
    overall_state_divergence_rate = (
        total_state_diverged_cycles / total_cycles if total_cycles > 0 else 0.0
    )
    overall_any_divergence_rate = (
        total_any_diverged_cycles / total_cycles if total_cycles > 0 else 0.0
    )

    result = {
        "schema_version": "1.0.0",
        "thresholds": {
            "state_threshold": state_threshold,
            "threshold_source": "data_structures_p4.py:483 (hardcoded, PROVISIONAL)",
        },
        "per_window": per_window_decompositions,
        "overall": {
            "total_cycles": total_cycles,
            "state_divergence_rate": round(overall_state_divergence_rate, 6),
            "state_divergence_rate_basis": "proxy_mean_delta_p_threshold",
            "outcome_divergence_rate_success": None,  # Requires per-cycle data
            "outcome_divergence_rate_omega": None,  # Requires per-cycle data
            "outcome_divergence_rate_blocked": None,  # Requires per-cycle data
            "outcome_divergence_basis": "UNAVAILABLE_NO_PER_CYCLE_COMPONENTS",
            "overall_any_divergence_rate": round(overall_any_divergence_rate, 6),
        },
        "missing_fields": missing_fields if missing_fields else None,
        "advisory_note": (
            "State divergence uses mean_delta_p > threshold as proxy (conservative approximation). "
            "Outcome divergences (success/omega/blocked) require per-cycle divergence snapshots "
            "not available in aggregated windows. Set to None with reason documented."
        ),
    }

    # Add outcome divergence note
    if not missing_fields or all(not f.endswith("_divergence_rate") for f in missing_fields):
        result["outcome_divergence_note"] = (
            "Outcome divergence rates (success, omega, blocked) are None because "
            "per-cycle DivergenceSnapshot data (success_diverged, omega_diverged, blocked_diverged) "
            "is not available in aggregated cal_exp1_report.json windows. "
            "To compute outcome rates, use divergence_log.jsonl per-cycle snapshots."
        )

    return result


def build_runtime_profile_calibration_annex(
    runtime_profile_snapshot: Optional[Dict[str, Any]],
    cal_exp1_windows: Optional[List[Dict[str, Any]]],
    divergence_spike_threshold: float = 0.8,
    delta_p_threshold: float = 0.05,
    state_divergence_threshold: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """
    Build runtime profile calibration annex for evidence pack.

    SHADOW MODE: Advisory only, no gating.

    Args:
        runtime_profile_snapshot: Runtime profile snapshot (optional)
        cal_exp1_windows: CAL-EXP-1 windows (optional)
        divergence_spike_threshold: Threshold for divergence spike detection (default: 0.8)
        delta_p_threshold: Threshold for mean_delta_p spike detection (default: 0.05)
        state_divergence_threshold: Threshold for state divergence (default: 0.05)

    Returns:
        Annex dictionary or None if insufficient data
    """
    if not runtime_profile_snapshot or not cal_exp1_windows:
        return None

    # Compute correlations
    correlations = correlate_runtime_profile_with_cal_windows(
        runtime_profile_snapshot, cal_exp1_windows
    )

    # Annotate windows with confounding flags
    annotated_windows, confounding_summary = annotate_cal_windows_with_runtime_confounding(
        runtime_profile_snapshot,
        cal_exp1_windows,
        divergence_spike_threshold,
        delta_p_threshold,
    )

    # Decompose divergence components
    divergence_decomposition = decompose_divergence_components(
        cal_exp1_windows, state_threshold=state_divergence_threshold
    )

    # Determine instrumentation verdict
    overall_any_divergence_rate = divergence_decomposition.get("overall", {}).get(
        "overall_any_divergence_rate"
    )
    instrumentation_verdict = "METER_OK"
    instrumentation_notes: List[str] = []

    if divergence_decomposition.get("error") == "INSUFFICIENT_DATA":
        instrumentation_verdict = "INSUFFICIENT_DATA"
        instrumentation_notes.append(
            f"Insufficient data for decomposition: {divergence_decomposition.get('missing_fields', [])}"
        )
    elif overall_any_divergence_rate is not None:
        # Check for meter saturation
        if overall_any_divergence_rate >= 0.99:  # Near saturation
            instrumentation_verdict = "METER_SATURATED"
            instrumentation_notes.append(
                f"overall_any_divergence_rate saturates at {overall_any_divergence_rate:.3f} "
                f"(>= 0.99). This occurs when any stochastic mismatch is counted as divergence, "
                f"causing the metric to saturate near 1.0 regardless of mean_delta_p improvements. "
                f"Use state_divergence_rate or mean_delta_p for calibration tuning."
            )

        # Check if outcome divergences are missing
        overall = divergence_decomposition.get("overall", {})
        if (
            overall.get("outcome_divergence_rate_success") is None
            or overall.get("outcome_divergence_rate_omega") is None
            or overall.get("outcome_divergence_rate_blocked") is None
        ):
            instrumentation_notes.append(
                "Outcome divergence rates (success, omega, blocked) are unavailable "
                "(require per-cycle DivergenceSnapshot data from divergence_log.jsonl, "
                "not available in aggregated windows)."
            )

    # Build advisory note
    advisory_note_parts = []
    if correlations.get("windows_analyzed", 0) > 0:
        advisory_note_parts.append(
            f"Analyzed {correlations['windows_analyzed']} CAL-EXP-1 windows "
            f"against runtime profile {runtime_profile_snapshot.get('profile', 'unknown')}."
        )

    # Use advisory note from confounding summary (includes counterfactual control)
    advisory_note = (
        confounding_summary.get("advisory_note", "")
        + " All analysis is advisory only (SHADOW MODE)."
    )

    return {
        "schema_version": "1.0.0",
        "profile": runtime_profile_snapshot.get("profile"),
        "status_light": runtime_profile_snapshot.get("status_light"),
        "windowed_correlations": correlations,
        "confounding_summary": confounding_summary,
        "divergence_decomposition": divergence_decomposition,
        "instrumentation_verdict": instrumentation_verdict,
        "instrumentation_notes": instrumentation_notes,
        "annotated_windows": annotated_windows,
        "advisory_note": advisory_note,
        "mode": "SHADOW",
    }

