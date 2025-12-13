"""Semantic-TDA integration adapter for global health.

STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

Provides integration between semantic governance signals and TDA health signals
for the global health surface builder.

INTEGRATION EXPECTATIONS FOR PHASE X:
The semantic-TDA coupling tile is intended to demonstrate structural alignment:

- **P3 (Synthetic Evaluation)**: "The map and the topology agree"
  - Semantic graph (language/symbols) and TDA topology (shape/structure) should
    show consistent drift signals when structural issues occur.
  - High correlation indicates that both systems detect the same underlying
    structural problems, providing cross-validation of governance signals.

- **P4 (Real-Coupled Evaluation)**: "The twin and the real system share structural geometry"
  - Semantic drift in the real system should correlate with TDA topology drift
    in both the real system and its twin (shadow) representation.
  - Agreement between semantic and TDA signals provides evidence that the
    governance model's structural understanding matches operational reality.

This tile is expected to appear in Phase X evidence bundles alongside:
- Δp (pressure differential)
- RSI (relative stability index)
- Ω (USLA safe region)
- Divergence (P4 twin-vs-real)
- PRNG governance
- Budget governance
- Performance governance
- Replay governance

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The semantic_tda tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

from typing import Any, Dict, Optional

from experiments.semantic_consistency_audit import (
    build_semantic_tda_governance_tile,
    correlate_semantic_and_tda_signals,
)
from backend.health.semantic_tda_timeline import (
    extract_correlation_timeline,
    extract_correlation_trends,
)

SEMANTIC_TDA_TILE_SCHEMA_VERSION = "1.0.0"


def _validate_semantic_panel(panel: Dict[str, Any]) -> None:
    """Validate semantic panel structure.
    
    Args:
        panel: Semantic panel dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["semantic_status_light", "alignment_status", "critical_run_ids", "headline"]
    missing = [key for key in required_keys if key not in panel]
    if missing:
        raise ValueError(
            f"semantic_panel missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(panel.keys()))}"
        )


def _validate_tda_panel(panel: Dict[str, Any]) -> None:
    """Validate TDA panel structure.
    
    Args:
        panel: TDA panel dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["tda_status", "block_rate", "hss_trend", "governance_signal"]
    missing = [key for key in required_keys if key not in panel]
    if missing:
        raise ValueError(
            f"tda_panel missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(panel.keys()))}"
        )


def _extract_timeline_from_panel(
    semantic_panel: Dict[str, Any],
    semantic_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract or synthesize semantic timeline for correlation.
    
    If a full semantic_timeline is provided, use it as-is.
    Otherwise, synthesize a minimal timeline from the semantic_panel.
    
    The synthetic timeline is acceptable as a first-pass integration,
    as long as it's documented and deterministic.
    
    Args:
        semantic_panel: Semantic director panel
        semantic_timeline: Optional full timeline (preferred if available)
    
    Returns:
        Semantic timeline dictionary with required structure for correlation
    """
    # If full timeline is available, use it
    if semantic_timeline is not None:
        return semantic_timeline
    
    # Otherwise, synthesize from panel
    semantic_status_light = semantic_panel.get("semantic_status_light", "GREEN")
    critical_run_ids = semantic_panel.get("critical_run_ids", [])
    trend = semantic_panel.get("trend", "STABLE")
    
    # Determine if we have critical signals
    has_critical = semantic_status_light in ("RED", "YELLOW") or len(critical_run_ids) > 0
    
    # Synthesize node disappearance events from critical runs
    node_disappearance_events = []
    for run_id in critical_run_ids:
        # Create a synthetic event for each critical run
        node_disappearance_events.append({
            "run_id": run_id,
            "term": f"term_from_{run_id}",  # Generic term name
        })
    
    # If no critical runs but status is not GREEN, create a generic event
    if has_critical and not node_disappearance_events:
        node_disappearance_events.append({
            "run_id": "synthetic_critical",
            "term": "system_wide",
        })
    
    # Build synthetic timeline
    synthetic_timeline = {
        "timeline": [
            {
                "run_id": run_id,
                "status": "CRITICAL" if run_id in critical_run_ids else "OK",
            }
            for run_id in critical_run_ids
        ] if critical_run_ids else [],
        "runs_with_critical_signals": critical_run_ids,
        "node_disappearance_events": node_disappearance_events,
        "trend": trend,
        "semantic_status_light": semantic_status_light,
    }
    
    return synthetic_timeline


def build_semantic_tda_tile_for_global_health(
    semantic_panel: Dict[str, Any],
    tda_panel: Dict[str, Any],
    semantic_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build semantic-TDA governance tile for global health surface.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

    Integrates semantic governance signals with TDA health signals to produce
    a unified governance tile for the global health dashboard.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents

    Args:
        semantic_panel: Semantic director panel from build_semantic_director_panel.
            Must contain: semantic_status_light, alignment_status, critical_run_ids, headline
        tda_panel: TDA health panel from summarize_tda_for_global_health.
            Must contain: tda_status, block_rate, hss_trend, governance_signal
        semantic_timeline: Optional full semantic timeline. If provided, used for correlation.
            If None, a synthetic timeline is created from semantic_panel.

    Returns:
        Semantic-TDA governance tile dictionary with:
        - schema_version
        - tile_type: "semantic_tda"
        - status: "OK" | "ATTENTION" | "BLOCK"
        - correlation_coefficient: float
        - slices_where_both_signal: List[str]
        - headline: str
        - notes: List[str]
    """
    # Validate inputs
    _validate_semantic_panel(semantic_panel)
    _validate_tda_panel(tda_panel)
    
    # Extract or synthesize timeline
    timeline = _extract_timeline_from_panel(semantic_panel, semantic_timeline)
    
    # Use tda_panel as tda_health for correlation (same structure expected)
    correlation = correlate_semantic_and_tda_signals(timeline, tda_panel)
    
    # Build governance tile
    tile = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)
    
    # Add tile_type and notes for global health integration
    tile["tile_type"] = "semantic_tda"
    tile["notes"] = [
        correlation.get("alignment_note", ""),
        f"Semantic status: {semantic_panel.get('semantic_status_light', 'GREEN')}",
        f"TDA status: {tda_panel.get('tda_status', 'OK')}",
    ]
    
    return tile


def build_semantic_tda_correlation_summary(tile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a minimal, evidence-ready correlation summary from a semantic-TDA tile.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

    Returns a compact summary suitable for First Light evidence bundles,
    allowing external reviewers to quickly see where semantic and TDA systems
    agree or disagree.

    **For External Reviewers:**
    Intended to give a compact, high-level indicator of whether semantic drift
    and TDA drift are aligned on the same slices. When correlation ≈ 1.0 with
    a few key slices, it indicates that the semantic panel and topology panel
    are seeing the same failures, providing strong cross-validation of structural
    problems.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents

    Args:
        tile: Semantic-TDA governance tile from build_semantic_tda_tile_for_global_health().

    Returns:
        Minimal correlation summary dictionary with:
        - schema_version
        - status: Overall tile status
        - correlation_coefficient: Correlation value
        - num_key_slices: Total number of key slices
        - key_slices: First 5 key slices (truncated for evidence readability)
    """
    key_slices = tile.get("key_slices", [])
    return {
        "schema_version": SEMANTIC_TDA_TILE_SCHEMA_VERSION,
        "status": tile.get("status", "OK"),
        "correlation_coefficient": tile.get("correlation_coefficient", 0.0),
        "num_key_slices": len(key_slices),
        "key_slices": key_slices[:5],  # Truncate to first 5 for evidence readability
    }


def attach_semantic_tda_to_evidence(
    evidence: Dict[str, Any],
    semantic_tile: Dict[str, Any],
    include_correlation_summary: bool = False,
) -> Dict[str, Any]:
    """
    Attach semantic-TDA governance tile to an evidence pack (read-only, additive).

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["semantic_tda"].

    Optionally includes a correlation_summary sub-object for First Light evidence
    bundles, providing a compact view of where semantic and TDA systems agree
    or disagree.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        semantic_tile: Semantic-TDA governance tile from build_semantic_tda_tile_for_global_health().
        include_correlation_summary: If True, include a correlation_summary sub-object
            in the attached tile for First Light evidence bundles.

    Returns:
        New dict with evidence contents plus semantic_tda tile attached under governance key.
        If include_correlation_summary is True, the tile will also contain a
        correlation_summary sub-object.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_semantic_tda_tile_for_global_health(panel, tda_panel)
        >>> enriched = attach_semantic_tda_to_evidence(evidence, tile, include_correlation_summary=True)
        >>> "governance" in enriched
        True
        >>> "semantic_tda" in enriched["governance"]
        True
        >>> "correlation_summary" in enriched["governance"]["semantic_tda"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Attach semantic-TDA tile (copy to avoid mutating original)
    tile_copy = semantic_tile.copy()
    
    # Optionally include correlation summary
    if include_correlation_summary:
        tile_copy["correlation_summary"] = build_semantic_tda_correlation_summary(semantic_tile)
    
    # Attach tile to governance
    enriched["governance"] = enriched["governance"].copy()
    enriched["governance"]["semantic_tda"] = tile_copy
    
    return enriched


def extract_correlation_for_pattern_classifier(
    semantic_timeline_history: list[Dict[str, Any]],
    tda_health_history: list[Dict[str, Any]],
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Extract correlation timeline and trends for TDA pattern classifier integration.

    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE (UPGRADE-2)

    Provides correlation timeline and trend statistics for integration with
    TDA pattern classifier. The correlation trends can be used as additional
    inputs to improve pattern classification accuracy.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned data is purely observational
    - No control flow depends on the correlation data

    Args:
        semantic_timeline_history: List of semantic timeline snapshots per cycle
        tda_health_history: List of TDA health snapshots per cycle
        window_size: Number of cycles per window (default: 10)

    Returns:
        Dictionary with:
        - timeline: Correlation timeline from extract_correlation_timeline()
        - trends: Correlation trends from extract_correlation_trends()
        - phase_lag: Phase lag analysis from compute_phase_lag_index()
        - pattern_classifier_input: Formatted input for pattern classifier:
            - correlation_mean: Mean correlation value
            - correlation_slope: Trend direction
            - correlation_regime: Classification regime
            - alignment_strength: Normalized alignment strength [0, 1]
            - phase_lag_index: Phase lag severity index [0, 1]
    """
    from backend.health.semantic_tda_timeline import compute_phase_lag_index

    timeline = extract_correlation_timeline(
        semantic_timeline_history,
        tda_health_history,
        window_size=window_size,
    )

    trends = extract_correlation_trends(timeline)
    phase_lag = compute_phase_lag_index(timeline, trends=trends)

    # Format for pattern classifier input
    # Normalize correlation_mean to [0, 1] for alignment_strength
    # (correlation ranges from -1 to 1, so we map to 0-1)
    correlation_mean = trends.get("correlation_mean", 0.0)
    alignment_strength = max(0.0, min(1.0, (correlation_mean + 1.0) / 2.0))

    pattern_classifier_input = {
        "correlation_mean": trends.get("correlation_mean", 0.0),
        "correlation_slope": trends.get("correlation_slope", 0.0),
        "correlation_regime": trends.get("correlation_regime", "STABLE"),
        "alignment_strength": round(alignment_strength, 3),
        "windows_with_high_correlation": trends.get("windows_with_high_correlation", 0),
        "windows_with_negative_correlation": trends.get("windows_with_negative_correlation", 0),
        "phase_lag_index": phase_lag.get("phase_lag_index", 0.0),
    }

    return {
        "schema_version": SEMANTIC_TDA_TILE_SCHEMA_VERSION,
        "timeline": timeline,
        "trends": trends,
        "phase_lag": phase_lag,
        "pattern_classifier_input": pattern_classifier_input,
    }


__all__ = [
    "SEMANTIC_TDA_TILE_SCHEMA_VERSION",
    "attach_semantic_tda_to_evidence",
    "build_semantic_tda_correlation_summary",
    "build_semantic_tda_tile_for_global_health",
    "extract_correlation_for_pattern_classifier",
]

