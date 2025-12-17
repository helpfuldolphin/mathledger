"""Semantic consistency audit module.

STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

Provides semantic graph analysis, drift detection, and cross-correlation
with TDA (Topological Data Analysis) health signals.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Outputs are purely observational and do NOT influence governance decisions
- No control flow depends on these outputs
"""

from typing import Any, Dict, List

# Schema version for semantic-TDA coupling
SEMANTIC_TDA_COUPLING_SCHEMA_VERSION = "1.0.0"


def _validate_input_keys(data: Dict[str, Any], required_keys: List[str], context: str) -> None:
    """Validate that required keys are present in data dict.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required key names
        context: Context string for error messages (e.g., "semantic_timeline")
    
    Raises:
        ValueError: If any required key is missing
    """
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(
            f"{context} missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(data.keys()))}"
        )


def correlate_semantic_and_tda_signals(
    semantic_timeline: Dict[str, Any],
    tda_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Correlate semantic drift signals with TDA health signals.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

    Analyzes alignment between structural drift in semantic graph (symbols/terms)
    and structural drift in TDA topology. Identifies slices where both systems
    signal issues, or where they disagree.

    Args:
        semantic_timeline: From build_semantic_drift_timeline.
            Expected keys: timeline, runs_with_critical_signals, node_disappearance_events, trend
        tda_health: From summarize_tda_for_global_health (backend.health.tda_adapter).
            Expected keys: tda_status, block_rate, hss_trend, governance_signal

    Returns:
        Dictionary with:
        - schema_version
        - correlation_coefficient: float (-1.0 to 1.0)
        - slices_where_both_signal: List of slice/term names where both systems signal issues
        - semantic_only_slices: List of slices/terms with semantic signals but no TDA signals
        - tda_only_slices: List of slices/terms with TDA signals but no semantic signals
        - alignment_note: Descriptive note about correlation strength and direction
    """
    # Validate inputs
    _validate_input_keys(
        semantic_timeline,
        ["timeline", "runs_with_critical_signals", "node_disappearance_events", "trend"],
        "semantic_timeline"
    )
    _validate_input_keys(
        tda_health,
        ["tda_status", "block_rate", "hss_trend", "governance_signal"],
        "tda_health"
    )

    # Normalize statuses to numeric values for correlation
    # Semantic: OK=0, WARN=1, BLOCK=2
    semantic_status_light = semantic_timeline.get("semantic_status_light", "GREEN")
    semantic_numeric = {"GREEN": 0, "YELLOW": 1, "RED": 2}.get(semantic_status_light, 0)
    
    # TDA: OK=0, ATTENTION=1, ALERT=2
    tda_status = tda_health.get("tda_status", "OK")
    tda_numeric = {"OK": 0, "ATTENTION": 1, "ALERT": 2}.get(tda_status, 0)
    
    # Simple correlation: if both are non-zero, positive correlation
    # If one is zero and other is non-zero, negative correlation
    # If both are zero, neutral correlation
    if semantic_numeric == 0 and tda_numeric == 0:
        correlation_coefficient = 0.0
    elif semantic_numeric > 0 and tda_numeric > 0:
        # Both signal issues - positive correlation
        correlation_coefficient = 0.8 + (min(semantic_numeric, tda_numeric) * 0.1)
    elif semantic_numeric > 0 and tda_numeric == 0:
        # Semantic signals but TDA doesn't - negative correlation
        correlation_coefficient = -0.5
    elif semantic_numeric == 0 and tda_numeric > 0:
        # TDA signals but semantic doesn't - negative correlation
        correlation_coefficient = -0.5
    else:
        correlation_coefficient = 0.0
    
    # Clamp to [-1.0, 1.0]
    correlation_coefficient = max(-1.0, min(1.0, correlation_coefficient))

    # Extract slices/terms from semantic timeline
    node_disappearance_events = semantic_timeline.get("node_disappearance_events", [])
    semantic_slices = set()
    for event in node_disappearance_events:
        if isinstance(event, dict):
            term = event.get("term") or event.get("slice")
            if term:
                semantic_slices.add(term)
        elif isinstance(event, str):
            semantic_slices.add(event)
    
    # TDA doesn't track individual slices, so we use a generic marker
    # If TDA signals issues, we mark it as "system_wide" for tda_only
    tda_only_slices = []
    if tda_numeric > 0 and semantic_numeric == 0:
        tda_only_slices = ["system_wide"]
    
    # Find slices where both signal
    slices_where_both_signal = []
    if semantic_numeric > 0 and tda_numeric > 0:
        # If we have semantic slices, use them; otherwise use generic
        if semantic_slices:
            slices_where_both_signal = list(semantic_slices)
        else:
            slices_where_both_signal = ["system_wide"]
    
    # Semantic-only slices
    semantic_only_slices = []
    if semantic_numeric > 0 and tda_numeric == 0:
        if semantic_slices:
            semantic_only_slices = list(semantic_slices)
        else:
            semantic_only_slices = ["system_wide"]

    # Generate alignment note
    if correlation_coefficient >= 0.7:
        alignment_note = (
            f"Strong positive correlation (r={correlation_coefficient:.2f}). "
            "Both semantic and TDA systems signal structural drift, indicating "
            "agreement on system health degradation."
        )
    elif correlation_coefficient <= -0.3:
        alignment_note = (
            f"Negative correlation (r={correlation_coefficient:.2f}). "
            "Semantic and TDA systems disagree: one signals drift while the other "
            "appears stable. This may indicate a false positive or missed signal."
        )
    else:
        alignment_note = (
            f"Weak correlation (r={correlation_coefficient:.2f}). "
            "Semantic and TDA signals are largely independent, or both systems "
            "report stable state."
        )

    return {
        "schema_version": SEMANTIC_TDA_COUPLING_SCHEMA_VERSION,
        "correlation_coefficient": round(correlation_coefficient, 3),
        "slices_where_both_signal": sorted(slices_where_both_signal),
        "semantic_only_slices": sorted(semantic_only_slices),
        "tda_only_slices": sorted(tda_only_slices),
        "alignment_note": alignment_note,
    }


def build_semantic_tda_governance_tile(
    semantic_panel: Dict[str, Any],
    tda_panel: Dict[str, Any],
    correlation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact governance tile combining semantic and TDA signals.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

    Provides a unified view of structural drift in both semantic graph (language/symbols)
    and TDA topology (shape/structure). The tile is designed for global health dashboards.

    Args:
        semantic_panel: From build_semantic_director_panel.
            Expected keys: semantic_status_light, alignment_status, critical_run_ids, headline
        tda_panel: From summarize_tda_for_global_health (backend.health.tda_adapter).
            Expected keys: tda_status, block_rate, hss_trend, governance_signal
        correlation: From correlate_semantic_and_tda_signals.
            Expected keys: correlation_coefficient, slices_where_both_signal, alignment_note

    Returns:
        Dictionary with:
        - schema_version
        - status: "OK" | "ATTENTION" | "BLOCK"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - headline: Short neutral sentence summarizing combined status
        - key_slices: List of slice names requiring attention
    """
    # Validate inputs
    _validate_input_keys(
        semantic_panel,
        ["semantic_status_light", "alignment_status", "critical_run_ids", "headline"],
        "semantic_panel"
    )
    _validate_input_keys(
        tda_panel,
        ["tda_status", "block_rate", "hss_trend", "governance_signal"],
        "tda_panel"
    )
    _validate_input_keys(
        correlation,
        ["correlation_coefficient", "slices_where_both_signal", "alignment_note"],
        "correlation"
    )

    semantic_status_light = semantic_panel.get("semantic_status_light", "GREEN")
    tda_status = tda_panel.get("tda_status", "OK")
    correlation_coeff = correlation.get("correlation_coefficient", 0.0)
    slices_where_both = correlation.get("slices_where_both_signal", [])

    # Determine overall status: BLOCK > ATTENTION > OK
    # BLOCK: Both systems signal critical issues (RED/ALERT) OR strong correlation with issues
    # ATTENTION: One system signals issues OR weak correlation with issues
    # OK: Both systems stable
    
    status = "OK"
    status_light = "GREEN"
    
    if (semantic_status_light == "RED" and tda_status == "ALERT") or \
       (semantic_status_light == "RED" and correlation_coeff >= 0.7) or \
       (tda_status == "ALERT" and correlation_coeff >= 0.7):
        status = "BLOCK"
        status_light = "RED"
    elif (semantic_status_light == "RED" or tda_status == "ALERT") or \
         (semantic_status_light == "YELLOW" and tda_status == "ATTENTION") or \
         (correlation_coeff <= -0.3):  # Disagreement is also attention-worthy
        status = "ATTENTION"
        status_light = "YELLOW"
    else:
        status = "OK"
        status_light = "GREEN"

    # Build headline
    if status == "BLOCK":
        if slices_where_both:
            headline = (
                f"Semantic and TDA systems both signal critical structural drift "
                f"on {len(slices_where_both)} slice(s). Strong agreement (r={correlation_coeff:.2f})."
            )
        else:
            headline = (
                f"Semantic and TDA systems both signal critical structural drift. "
                f"Strong agreement (r={correlation_coeff:.2f})."
            )
    elif status == "ATTENTION":
        if correlation_coeff <= -0.3:
            headline = (
                "Semantic and TDA systems disagree on structural health. "
                "One signals drift while the other appears stable."
            )
        elif semantic_status_light == "RED" or tda_status == "ALERT":
            headline = (
                "One system signals critical structural drift while the other "
                "reports stable state. Investigation recommended."
            )
        else:
            headline = (
                "Semantic and TDA systems report moderate structural drift. "
                "Monitoring recommended."
            )
    else:
        headline = (
            "Semantic and TDA systems report stable structural health. "
            f"Correlation: {correlation_coeff:.2f}."
        )

    # Collect key slices
    key_slices = set()
    if slices_where_both:
        key_slices.update(slices_where_both)
    
    # Add slices from semantic panel if available
    semantic_critical_runs = semantic_panel.get("critical_run_ids", [])
    if semantic_critical_runs and semantic_status_light != "GREEN":
        # Extract slice names from run IDs if possible
        for run_id in semantic_critical_runs:
            # Try to infer slice from run_id (e.g., "run_uplift_slice_alpha" -> "slice_alpha")
            if "slice" in run_id.lower():
                parts = run_id.lower().split("slice")
                if len(parts) > 1:
                    slice_name = "slice" + parts[1].split("_")[0]
                    key_slices.add(slice_name)
    
    correlation_coeff = correlation.get("correlation_coefficient", 0.0)

    return {
        "schema_version": SEMANTIC_TDA_COUPLING_SCHEMA_VERSION,
        "status": status,
        "status_light": status_light,
        "headline": headline,
        "key_slices": sorted(key_slices) if key_slices else sorted(slices_where_both),
        "semantic_status": semantic_panel.get("semantic_status_light", "GREEN"),
        "tda_status": tda_panel.get("tda_status", "OK"),
        "correlation_coefficient": round(correlation_coeff, 3),
    }


__all__ = [
    "SEMANTIC_TDA_COUPLING_SCHEMA_VERSION",
    "correlate_semantic_and_tda_signals",
    "build_semantic_tda_governance_tile",
]

