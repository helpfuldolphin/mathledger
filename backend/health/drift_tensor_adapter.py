"""
Drift Tensor Governance Tile Adapter.

Provides drift governance tile for global health integration.

SHADOW MODE: Observation-only. No control paths.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"


def build_drift_governance_tile(
    drift_tensor: Dict[str, Any],
    poly_cause_view: Dict[str, Any],
    director_tile_v2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build governance tile for multi-axis drift.

    SHADOW MODE: Observation-only. No control paths.
    This tile is purely observational and does NOT influence other tiles.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_tensor: Dict from build_drift_tensor.
            Expected keys: "global_tensor_norm", "tensor", "ranked_slices".
        poly_cause_view: Dict from build_drift_poly_cause_analyzer.
            Expected keys: "poly_cause_detected", "cause_vectors", "risk_band", "notes".
        director_tile_v2: Dict from build_drift_director_tile_v2.
            Expected keys: "status_light", "tensor_norm", "poly_cause_status", "risk_band".

    Returns:
        Dict with governance tile:
        - status_light: "GREEN" | "YELLOW" | "RED" (from director_tile_v2)
        - global_tensor_norm: float (from drift_tensor)
        - risk_band: "LOW" | "MEDIUM" | "HIGH" (from poly_cause_view)
        - slices_with_poly_cause_drift: List[str] (slices with multi-axis drift)
        - headline: str (from director_tile_v2)
        - schema_version: "1.0.0"
    """
    # Extract status light from director tile
    status_light = director_tile_v2.get("status_light", "GREEN")

    # Extract global tensor norm from drift tensor
    global_tensor_norm = drift_tensor.get("global_tensor_norm", 0.0)

    # Extract risk band from poly-cause view (prefer director_tile_v2 if available)
    risk_band = director_tile_v2.get("risk_band") or poly_cause_view.get("risk_band", "LOW")

    # Extract slices with poly-cause drift from cause vectors
    cause_vectors = poly_cause_view.get("cause_vectors", [])
    slices_with_poly_cause_drift = sorted(
        [vec["slice"] for vec in cause_vectors if len(vec.get("axes", [])) >= 2]
    )

    # Extract headline from director tile
    headline = director_tile_v2.get("headline", "No drift issues detected.")

    return {
        "status_light": status_light,
        "global_tensor_norm": round(global_tensor_norm, 6),
        "risk_band": risk_band,
        "slices_with_poly_cause_drift": slices_with_poly_cause_drift,
        "headline": headline,
        "schema_version": DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
    }


def extract_drift_signal_for_shadow(
    tensor: Dict[str, Any],
    poly_cause: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract drift signal for P4 shadow runs.

    SHADOW MODE: Observation-only. No control paths.
    Used by FirstLightShadowRunnerP4 to log drift status.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        tensor: Dict from build_drift_tensor.
            Expected keys: "global_tensor_norm", "tensor", "ranked_slices".
        poly_cause: Dict from build_drift_poly_cause_analyzer.
            Expected keys: "poly_cause_detected", "cause_vectors", "risk_band".

    Returns:
        Dict with drift signal for shadow logging:
        - global_tensor_norm: float
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - slices_with_poly_cause_drift: List[str]
    """
    global_tensor_norm = tensor.get("global_tensor_norm", 0.0)
    risk_band = poly_cause.get("risk_band", "LOW")

    # Extract slices with poly-cause drift
    cause_vectors = poly_cause.get("cause_vectors", [])
    slices_with_poly_cause_drift = sorted(
        [vec["slice"] for vec in cause_vectors if len(vec.get("axes", [])) >= 2]
    )

    poly_cause_detected = poly_cause.get("poly_cause_detected", False)

    return {
        "global_tensor_norm": round(global_tensor_norm, 6),
        "risk_band": risk_band,
        "poly_cause_detected": poly_cause_detected,
        "slices_with_poly_cause_drift": slices_with_poly_cause_drift,
    }


def attach_drift_governance_to_evidence(
    evidence: Dict[str, Any],
    governance_tile: Dict[str, Any],
    drift_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach drift governance tile and signal to an evidence pack (read-only, additive).

    SHADOW MODE: Observation-only. No control paths.
    This function is read-only (aside from dict construction).
    The attached data is purely observational.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        governance_tile: Drift governance tile from build_drift_governance_tile().
        drift_signal: Drift signal from extract_drift_signal_for_shadow().

    Returns:
        New dict with evidence contents plus drift governance data attached
        under evidence["governance"]["drift"].

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_drift_governance_tile(tensor, poly_cause, director_tile)
        >>> signal = extract_drift_signal_for_shadow(tensor, poly_cause)
        >>> enriched = attach_drift_governance_to_evidence(evidence, tile, signal)
        >>> "governance" in enriched
        True
        >>> "drift" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}

    # Attach drift governance data
    enriched["governance"] = enriched["governance"].copy()
    enriched["governance"]["drift"] = {
        "drift_band": governance_tile.get("risk_band", "LOW"),
        "tensor_norm": governance_tile.get("global_tensor_norm", 0.0),
        "poly_cause_detected": drift_signal.get("poly_cause_detected", False),
        "highlighted_cases": governance_tile.get("slices_with_poly_cause_drift", []),
        "first_light_scenario_map": build_first_light_scenario_drift_map(drift_signal),
    }

    return enriched


def attach_drift_cluster_view_to_evidence(
    evidence: Dict[str, Any],
    cluster_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach drift cluster view to an evidence pack (read-only, additive).

    SHADOW MODE: Observation-only. No control paths.
    This function is read-only (aside from dict construction).
    The attached data is purely observational.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        cluster_view: Drift cluster view from build_drift_cluster_view().

    Returns:
        New dict with evidence contents plus drift cluster view attached
        under evidence["governance"]["scenario_drift_cluster_view"].

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> maps = [map1, map2, map3]  # From CAL-EXP-1/2/3
        >>> cluster_view = build_drift_cluster_view(maps)
        >>> enriched = attach_drift_cluster_view_to_evidence(evidence, cluster_view)
        >>> "scenario_drift_cluster_view" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}

    # Attach drift cluster view
    enriched["governance"] = enriched["governance"].copy()
    enriched["governance"]["scenario_drift_cluster_view"] = cluster_view

    return enriched


def extract_scenario_drift_cluster_signal(
    cluster_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract scenario drift cluster signal for status hooks.

    SHADOW MODE: Observation-only. No control paths.
    This function extracts key metrics from the drift cluster view
    for integration into status signals and health surfaces.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        cluster_view: Drift cluster view from build_drift_cluster_view().

    Returns:
        Dict with scenario drift cluster signal:
        - schema_version: str (from cluster_view if present, else "1.0.0")
        - mode: "SHADOW"
        - experiments_analyzed: int
        - high_risk_slices: List[str]
        - persistence_score: float (rounded to 6 decimal places)
        - drivers: List[str] (reason-code drivers, max 2, deterministic ordering)
        - extraction_source: str ("MANIFEST" | "EVIDENCE_JSON" | "MISSING")
    """
    persistence_score = cluster_view.get("persistence_score", 0.0)
    high_risk_slices = cluster_view.get("high_risk_slices", [])
    
    # Build reason-code drivers (deterministic ordering: persistence first, slices second)
    # Only these two driver codes are valid:
    # - DRIVER_PERSISTENCE_SCORE_HIGH: when persistence_score >= 0.5
    # - DRIVER_HIGH_RISK_SLICES_PRESENT: when len(high_risk_slices) > 0
    drivers = []
    if persistence_score >= 0.5:
        drivers.append("DRIVER_PERSISTENCE_SCORE_HIGH")
    if len(high_risk_slices) > 0:
        drivers.append("DRIVER_HIGH_RISK_SLICES_PRESENT")
    
    # extraction_source is provided by caller (status integration tracks provenance)
    extraction_source = cluster_view.get("extraction_source", "MISSING")
    
    return {
        "schema_version": cluster_view.get("schema_version", "1.0.0"),
        "mode": "SHADOW",
        "experiments_analyzed": cluster_view.get("experiments_analyzed", 0),
        "high_risk_slices": high_risk_slices,
        "persistence_score": round(persistence_score, 6),
        "drivers": drivers,
        "extraction_source": extraction_source,
    }


def summarize_drift_for_uplift_council(tile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize drift governance tile for uplift council decision-making.

    SHADOW MODE: Observation-only. No control paths.
    This function provides structural monitoring information only.
    It does NOT make governance decisions.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        tile: Drift governance tile from build_drift_governance_tile().

    Returns:
        Dict with council summary:
        - status: "OK" | "WARN" | "BLOCK"
        - drift_band: "LOW" | "MEDIUM" | "HIGH"
        - poly_cause_detected: bool
        - implicated_slices: List[str] (slices with poly-cause drift)
        - implicated_axes: List[str] (axes showing drift, if available)
        - summary: str (neutral descriptive summary)
    """
    risk_band = tile.get("risk_band", "LOW")
    poly_cause_detected = len(tile.get("slices_with_poly_cause_drift", [])) > 0
    tensor_norm = tile.get("global_tensor_norm", 0.0)

    # Map drift_band and poly_cause_detected into status
    if risk_band == "HIGH" or (risk_band == "MEDIUM" and poly_cause_detected and tensor_norm > 0.5):
        status = "BLOCK"
    elif risk_band == "MEDIUM" or poly_cause_detected:
        status = "WARN"
    else:
        status = "OK"

    implicated_slices = tile.get("slices_with_poly_cause_drift", [])

    # Build summary
    summary_parts: List[str] = []
    summary_parts.append(f"Drift band: {risk_band}")
    if poly_cause_detected:
        summary_parts.append(f"Poly-cause patterns detected in {len(implicated_slices)} slice(s)")
    if tensor_norm > 0.0:
        summary_parts.append(f"Tensor norm: {tensor_norm:.3f}")

    summary = ". ".join(summary_parts) + "." if summary_parts else "No drift issues detected."

    return {
        "status": status,
        "drift_band": risk_band,
        "poly_cause_detected": poly_cause_detected,
        "implicated_slices": sorted(implicated_slices),
        "implicated_axes": [],  # Could be extracted from cause_vectors if needed
        "summary": summary,
    }


def build_first_light_scenario_drift_map(
    drift_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build scenario-level drift map for First Light review.

    SHADOW MODE: Observation-only. No control paths.
    This function provides a summary of which slices/scenarios contributed
    most to drift, enabling reviewers to see where drift clusters.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        drift_signal: Drift signal from extract_drift_signal_for_shadow().

    Returns:
        Dict with scenario drift map:
        - schema_version: "1.0.0"
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
        - tensor_norm: float
        - poly_cause_detected: bool
        - slices_with_poly_cause: List[str] (top 5 slices with poly-cause drift)
    """
    risk_band = drift_signal.get("risk_band", "LOW")
    tensor_norm = drift_signal.get("global_tensor_norm", 0.0)
    poly_cause_detected = drift_signal.get("poly_cause_detected", False)
    slices_with_poly_cause = drift_signal.get("slices_with_poly_cause_drift", [])[:5]

    return {
        "schema_version": DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
        "risk_band": risk_band,
        "tensor_norm": round(tensor_norm, 6),
        "poly_cause_detected": poly_cause_detected,
        "slices_with_poly_cause": sorted(slices_with_poly_cause),
    }


def emit_cal_exp_scenario_drift_map(
    cal_id: str,
    scenario_map: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Emit scenario drift map for a calibration experiment to disk.

    SHADOW MODE: Observation-only. No control paths.
    Saves the scenario drift map as JSON file for later aggregation.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2").
        scenario_map: Scenario drift map from build_first_light_scenario_drift_map().
        output_dir: Optional output directory. Defaults to artifacts/calibration/.

    Returns:
        Path to the saved JSON file.

    Example:
        >>> signal = extract_drift_signal_for_shadow(tensor, poly_cause)
        >>> map = build_first_light_scenario_drift_map(signal)
        >>> path = emit_cal_exp_scenario_drift_map("CAL-EXP-1", map)
        >>> path.exists()
        True
    """
    if output_dir is None:
        output_dir = Path("artifacts") / "calibration"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"scenario_drift_map_{cal_id}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scenario_map, f, indent=2, sort_keys=True)
    
    return output_file


def build_drift_cluster_view(
    maps: List[Dict[str, Any]],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Build drift cluster view across multiple calibration experiments.

    SHADOW MODE: Observation-only. No control paths.
    This function aggregates scenario drift maps from multiple experiments
    to identify persistent drift clusters across CAL-EXP-1/2/3.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        maps: List of scenario drift maps from build_first_light_scenario_drift_map().
        top_n: Number of top slices to include in high_risk_slices (default: 5).

    Returns:
        Dict with drift cluster view:
        - schema_version: "1.0.0"
        - slice_frequency: Dict[str, int] (slice_name -> count, sorted desc)
        - high_risk_slices: List[str] (top N slices by frequency)
        - experiments_analyzed: int (number of maps processed)
        - persistence_buckets: Dict with appears_in_1, appears_in_2, appears_in_3 (sorted lists)
        - persistence_score: float (mean persistence across all slices)
    """
    # Count frequency of each slice across experiments
    slice_frequency: Dict[str, int] = {}
    
    for map_data in maps:
        slices = map_data.get("slices_with_poly_cause", [])
        for slice_name in slices:
            slice_frequency[slice_name] = slice_frequency.get(slice_name, 0) + 1
    
    # Sort by frequency (descending), then by slice name (ascending) for determinism
    sorted_frequency = dict(
        sorted(
            slice_frequency.items(),
            key=lambda x: (-x[1], x[0])  # Negative count for descending order
        )
    )
    
    # Get top N slices
    high_risk_slices = list(sorted_frequency.keys())[:top_n]
    
    # Build persistence buckets
    experiments_analyzed = len(maps)
    appears_in_1: List[str] = []
    appears_in_2: List[str] = []
    appears_in_3: List[str] = []
    
    for slice_name, count in sorted_frequency.items():
        if count == 1:
            appears_in_1.append(slice_name)
        elif count == 2:
            appears_in_2.append(slice_name)
        elif count >= 3:
            appears_in_3.append(slice_name)
    
    # Sort buckets for determinism
    appears_in_1 = sorted(appears_in_1)
    appears_in_2 = sorted(appears_in_2)
    appears_in_3 = sorted(appears_in_3)
    
    # Compute persistence score: mean(experiment_count / experiments_analyzed) over all slices
    if experiments_analyzed > 0 and len(sorted_frequency) > 0:
        persistence_scores = [
            count / experiments_analyzed
            for count in sorted_frequency.values()
        ]
        persistence_score = sum(persistence_scores) / len(persistence_scores)
    else:
        persistence_score = 0.0
    
    return {
        "schema_version": DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
        "slice_frequency": sorted_frequency,
        "high_risk_slices": high_risk_slices,
        "experiments_analyzed": experiments_analyzed,
        "persistence_buckets": {
            "appears_in_1": appears_in_1,
            "appears_in_2": appears_in_2,
            "appears_in_3": appears_in_3,
        },
        "persistence_score": round(persistence_score, 6),
    }


def build_drift_governance_for_calibration_report(
    governance_tile: Dict[str, Any],
    drift_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build drift governance section for P4 calibration report.

    SHADOW MODE: Observation-only. No control paths.
    Used when generating p4_calibration_report.json.

    Statistics provided here are descriptive; uplift evaluation is handled
    solely by governance-gate logic.

    Args:
        governance_tile: Drift governance tile from build_drift_governance_tile().
        drift_signal: Drift signal from extract_drift_signal_for_shadow().

    Returns:
        Dict with calibration report drift governance section:
        - global_tensor_norm: float
        - drift_band: "LOW" | "MEDIUM" | "HIGH"
        - poly_cause_detected: bool
        - highlighted_cases: List[str] (slices with poly-cause drift)
    """
    return {
        "global_tensor_norm": governance_tile.get("global_tensor_norm", 0.0),
        "drift_band": governance_tile.get("risk_band", "LOW"),
        "poly_cause_detected": drift_signal.get("poly_cause_detected", False),
        "highlighted_cases": governance_tile.get("slices_with_poly_cause_drift", []),
    }


__all__ = [
    "DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION",
    "build_drift_governance_tile",
    "extract_drift_signal_for_shadow",
    "attach_drift_governance_to_evidence",
    "summarize_drift_for_uplift_council",
    "build_drift_governance_for_calibration_report",
    "build_first_light_scenario_drift_map",
    "emit_cal_exp_scenario_drift_map",
    "build_drift_cluster_view",
    "attach_drift_cluster_view_to_evidence",
    "extract_scenario_drift_cluster_signal",
]

