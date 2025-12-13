"""Semantic Drift Governance Tile Adapter.

STATUS: PHASE X — SEMANTIC DRIFT GOVERNANCE INTEGRATION

Provides semantic drift governance tile for global health integration.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The semantic drift tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SEMANTIC_DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"


def _validate_drift_tensor(drift_tensor: Dict[str, Any]) -> None:
    """Validate drift tensor structure.
    
    Args:
        drift_tensor: Drift tensor dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["drift_components", "semantic_hotspots", "tensor_norm"]
    missing = [key for key in required_keys if key not in drift_tensor]
    if missing:
        raise ValueError(
            f"drift_tensor missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(drift_tensor.keys()))}"
        )


def _validate_counterfactual(counterfactual: Dict[str, Any]) -> None:
    """Validate counterfactual analysis structure.
    
    Args:
        counterfactual: Counterfactual analysis dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["projected_unstable_slices", "stability_timeline", "neutral_notes"]
    missing = [key for key in required_keys if key not in counterfactual]
    if missing:
        raise ValueError(
            f"counterfactual missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(counterfactual.keys()))}"
        )


def _validate_drift_director_panel(drift_director_panel: Dict[str, Any]) -> None:
    """Validate drift director panel structure.
    
    Args:
        drift_director_panel: Drift director panel dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = [
        "status_light",
        "semantic_hotspots",
        "projected_instability_count",
        "gating_recommendation",
        "recommendation_reasons",
        "headline",
    ]
    missing = [key for key in required_keys if key not in drift_director_panel]
    if missing:
        raise ValueError(
            f"drift_director_panel missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(drift_director_panel.keys()))}"
        )


def build_semantic_drift_governance_tile(
    drift_tensor: Dict[str, Any],
    counterfactual: Dict[str, Any],
    drift_director_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Governance tile summarizing semantic drift and projected instability.

    STATUS: PHASE X — SEMANTIC DRIFT GOVERNANCE INTEGRATION
    
    SHADOW MODE: Observational only.
    This tile is purely observational and does NOT influence other tiles.

    Args:
        drift_tensor: Dict from build_semantic_drift_tensor().
            Must contain: drift_components, semantic_hotspots, tensor_norm
        counterfactual: Dict from analyze_semantic_drift_counterfactual().
            Must contain: projected_unstable_slices, stability_timeline, neutral_notes
        drift_director_panel: Dict from build_semantic_drift_director_panel_v3().
            Must contain: status_light, semantic_hotspots, projected_instability_count,
            gating_recommendation, recommendation_reasons, headline

    Returns:
        Semantic drift governance tile dictionary with:
        - schema_version: "1.0.0"
        - status_light: "GREEN" | "YELLOW" | "RED" (from drift_director_panel)
        - tensor_norm: float (from drift_tensor)
        - semantic_hotspots: List[str] (from drift_tensor, sorted)
        - projected_instability_count: int (from counterfactual)
        - gating_recommendation: "OK" | "WARN" | "BLOCK" (from drift_director_panel)
        - recommendation_reasons: List[str] (from drift_director_panel)
        - headline: str (from drift_director_panel)
    """
    # Validate inputs
    _validate_drift_tensor(drift_tensor)
    _validate_counterfactual(counterfactual)
    _validate_drift_director_panel(drift_director_panel)
    
    # Extract status light from director panel
    status_light = drift_director_panel.get("status_light", "GREEN")
    
    # Extract tensor norm from drift tensor
    tensor_norm = drift_tensor.get("tensor_norm", 0.0)
    
    # Extract semantic hotspots from drift tensor (already sorted)
    semantic_hotspots = drift_tensor.get("semantic_hotspots", [])
    # Ensure sorted for determinism
    if isinstance(semantic_hotspots, list):
        semantic_hotspots = sorted(semantic_hotspots)
    else:
        semantic_hotspots = []
    
    # Extract projected instability count from counterfactual
    projected_unstable = counterfactual.get("projected_unstable_slices", [])
    projected_instability_count = len(projected_unstable) if isinstance(projected_unstable, list) else 0
    
    # Extract gating recommendation from director panel
    gating_recommendation = drift_director_panel.get("gating_recommendation", "OK")
    
    # Extract recommendation reasons from director panel
    recommendation_reasons = drift_director_panel.get("recommendation_reasons", [])
    if not isinstance(recommendation_reasons, list):
        recommendation_reasons = []
    
    # Extract headline from director panel
    headline = drift_director_panel.get("headline", "Semantic drift analysis complete.")
    
    return {
        "schema_version": SEMANTIC_DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION,
        "status_light": status_light,
        "tensor_norm": float(tensor_norm),
        "semantic_hotspots": semantic_hotspots,
        "projected_instability_count": projected_instability_count,
        "gating_recommendation": gating_recommendation,
        "recommendation_reasons": recommendation_reasons,
        "headline": headline,
    }


def extract_drift_advisory_for_uplift(
    drift_director_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract semantic drift advisory signal for uplift gating systems.
    
    STATUS: PHASE X — SEMANTIC DRIFT UPLIFT ADVISORY
    
    SHADOW MODE: Observational only.
    Used by uplift gating systems in advisory mode.
    
    Args:
        drift_director_panel: Dict from build_semantic_drift_director_panel_v3().
            Must contain: status_light, gating_recommendation, semantic_hotspots
    
    Returns:
        Dict with uplift advisory signal:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - gating_recommendation: "OK" | "WARN" | "BLOCK"
        - semantic_hotspots: List[str] (sorted)
    """
    # Extract key fields
    status_light = drift_director_panel.get("status_light", "GREEN")
    gating_recommendation = drift_director_panel.get("gating_recommendation", "OK")
    semantic_hotspots = drift_director_panel.get("semantic_hotspots", [])
    
    # Ensure sorted for determinism
    if isinstance(semantic_hotspots, list):
        semantic_hotspots = sorted(semantic_hotspots)
    else:
        semantic_hotspots = []
    
    return {
        "status_light": status_light,
        "gating_recommendation": gating_recommendation,
        "semantic_hotspots": semantic_hotspots,
    }


def build_semantic_drift_summary_for_p3(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build semantic drift summary for First-Light P3 stability report.

    STATUS: PHASE X — SEMANTIC DRIFT FIRST-LIGHT INTEGRATION

    Extracts key semantic drift metrics for inclusion in First-Light P3 summary.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents

    Args:
        tile: Semantic drift governance tile from build_semantic_drift_governance_tile().

    Returns:
        P3 summary dictionary with:
        - tensor_norm: float
        - semantic_hotspots: List[str]
        - projected_instability_count: int
        - status_light: "GREEN" | "YELLOW" | "RED"
        - gating_recommendation: "OK" | "WARN" | "BLOCK"
    """
    return {
        "tensor_norm": tile.get("tensor_norm", 0.0),
        "semantic_hotspots": tile.get("semantic_hotspots", []),
        "projected_instability_count": tile.get("projected_instability_count", 0),
        "status_light": tile.get("status_light", "GREEN"),
        "gating_recommendation": tile.get("gating_recommendation", "OK"),
    }


def build_semantic_drift_calibration_for_p4(
    tile: Dict[str, Any],
    drift_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build semantic drift calibration for P4 calibration report.

    STATUS: PHASE X — SEMANTIC DRIFT P4 INTEGRATION

    Extracts semantic drift metrics for inclusion in P4 calibration reports.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned calibration is purely observational
    - No control flow depends on the calibration contents

    Args:
        tile: Semantic drift governance tile from build_semantic_drift_governance_tile().
        drift_signal: Optional drift signal from extract_drift_advisory_for_uplift().

    Returns:
        P4 calibration dictionary with:
        - tensor_norm: float
        - hotspots: List[str]
        - regression_status: str (derived from gating_recommendation: BLOCK → "REGRESSED", WARN → "ATTENTION", OK → "STABLE")
        - projected_instability_count: int
    """
    gating_recommendation = tile.get("gating_recommendation", "OK")
    
    # Map gating_recommendation to regression_status
    if gating_recommendation == "BLOCK":
        regression_status = "REGRESSED"
    elif gating_recommendation == "WARN":
        regression_status = "ATTENTION"
    else:  # OK
        regression_status = "STABLE"
    
    return {
        "tensor_norm": tile.get("tensor_norm", 0.0),
        "hotspots": tile.get("semantic_hotspots", []),
        "regression_status": regression_status,
        "projected_instability_count": tile.get("projected_instability_count", 0),
    }


def build_semantic_drift_failure_shelf_for_first_light(
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build failure shelf from semantic drift P3/P4 data for First-Light review.

    STATUS: PHASE X — SEMANTIC DRIFT FAILURE SHELF

    Produces a compact "failure shelf" identifying worst semantic drift episodes
    for reviewers. If you only inspect 5 places, look here.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned shelf is purely observational
    - No control flow depends on the shelf contents

    Args:
        p3_summary: P3 summary from build_semantic_drift_summary_for_p3().
        p4_calibration: P4 calibration from build_semantic_drift_calibration_for_p4().

    Returns:
        Failure shelf dictionary with:
        - schema_version: "1.0.0"
        - p3_tensor_norm: float
        - p4_tensor_norm: float
        - semantic_hotspots: List[str] (top 5)
        - regression_status: str
    """
    # Extract tensor norms
    p3_tensor_norm = p3_summary.get("tensor_norm", 0.0)
    p4_tensor_norm = p4_calibration.get("tensor_norm", 0.0)
    
    # Extract semantic hotspots (limit to top 5)
    p3_hotspots = p3_summary.get("semantic_hotspots", [])
    if isinstance(p3_hotspots, list):
        semantic_hotspots = p3_hotspots[:5]
    else:
        semantic_hotspots = []
    
    # Extract regression status
    regression_status = p4_calibration.get("regression_status", "STABLE")
    
    return {
        "schema_version": "1.0.0",
        "p3_tensor_norm": float(p3_tensor_norm),
        "p4_tensor_norm": float(p4_tensor_norm),
        "semantic_hotspots": semantic_hotspots,
        "regression_status": regression_status,
    }


def emit_cal_exp_semantic_failure_shelf(
    cal_id: str,
    shelf: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Emit semantic failure shelf for a calibration experiment.

    STATUS: PHASE X — CAL-EXP SEMANTIC DRIFT SHELF EMISSION

    Writes the semantic failure shelf to a JSON file for a specific CAL-EXP run.
    This enables cross-experiment aggregation into a global triage index.

    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions
    - Non-mutating: does not modify input shelf

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3").
        shelf: Semantic failure shelf from build_semantic_drift_failure_shelf_for_first_light().
        output_dir: Optional output directory. Defaults to Path("calibration/") if not provided.

    Returns:
        Path to written shelf file (e.g., calibration/semantic_failure_shelf_CAL-EXP-1.json)

    Example:
        >>> shelf = build_semantic_drift_failure_shelf_for_first_light(p3_summary, p4_calibration)
        >>> path = emit_cal_exp_semantic_failure_shelf("CAL-EXP-1", shelf)
        >>> path.name
        'semantic_failure_shelf_CAL-EXP-1.json'
    """
    if output_dir is None:
        output_dir = Path("calibration")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create enriched shelf with cal_id
    enriched_shelf = dict(shelf)
    enriched_shelf["cal_id"] = cal_id
    
    # Write to file
    filename = f"semantic_failure_shelf_{cal_id}.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_shelf, f, indent=2, sort_keys=True)
    
    return output_path


def attach_semantic_drift_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    drift_signal: Optional[Dict[str, Any]] = None,
    p3_summary: Optional[Dict[str, Any]] = None,
    p4_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach semantic drift governance tile to evidence pack.

    STATUS: PHASE X — SEMANTIC DRIFT EVIDENCE INTEGRATION

    Stores semantic drift governance information under evidence["governance"]["semantic_drift"]
    for inclusion in evidence packs.

    SHADOW MODE CONTRACT:
    - This function modifies the evidence dict in-place
    - The attachment is purely observational
    - No control flow depends on the attached data

    Args:
        evidence: Evidence pack dictionary (will be modified in-place)
        tile: Semantic drift governance tile from build_semantic_drift_governance_tile()
        drift_signal: Optional drift signal from extract_drift_advisory_for_uplift()
        p3_summary: Optional P3 summary from build_semantic_drift_summary_for_p3()
        p4_calibration: Optional P4 calibration from build_semantic_drift_calibration_for_p4()

    Returns:
        Modified evidence dictionary with semantic drift governance attached
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    
    # Build semantic drift evidence block
    semantic_drift_evidence = {
        "status_light": tile.get("status_light", "GREEN"),
        "tensor_norm": tile.get("tensor_norm", 0.0),
        "semantic_hotspots": tile.get("semantic_hotspots", []),
        "projected_instability_count": tile.get("projected_instability_count", 0),
        "gating_recommendation": tile.get("gating_recommendation", "OK"),
        "headline": tile.get("headline", ""),
    }
    
    # Add drift signal if provided
    if drift_signal is not None:
        semantic_drift_evidence["drift_signal"] = {
            "status_light": drift_signal.get("status_light", "GREEN"),
            "gating_recommendation": drift_signal.get("gating_recommendation", "OK"),
            "semantic_hotspots": drift_signal.get("semantic_hotspots", []),
        }
    
    # Add failure shelf if both P3 summary and P4 calibration are provided
    if p3_summary is not None and p4_calibration is not None:
        semantic_drift_evidence["first_light_failure_shelf"] = (
            build_semantic_drift_failure_shelf_for_first_light(p3_summary, p4_calibration)
        )
    
    # Attach semantic drift governance
    evidence["governance"]["semantic_drift"] = semantic_drift_evidence
    
    return evidence


def extract_semantic_failure_triage_index_signal(
    triage_index: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract compact semantic failure triage index signal for status integration.

    STATUS: PHASE X — SEMANTIC DRIFT TRIAGE INDEX STATUS SIGNAL

    Extracts a minimal signal from the triage index for inclusion in First Light status files.
    Provides a Top10 summary for auditors.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        triage_index: Triage index from build_semantic_failure_triage_index()

    Returns:
        Compact signal dictionary with:
        - total_items: int (number of items in triage index)
        - top5: List[Dict] with {cal_id, regression_status, combined_tensor_norm, hotspots_count}
        - advisory_warning: Optional[str] (warning if any regression_status == "REGRESSED")

    Example:
        >>> index = build_semantic_failure_triage_index(shelves)
        >>> signal = extract_semantic_failure_triage_index_signal(index)
        >>> signal["total_items"]
        2
    """
    items = triage_index.get("items", [])
    total_items = len(items)
    
    # Extract top 5 items (or all if fewer than 5)
    top5_items = items[:5]
    
    # Build top5 summary
    top5: List[Dict[str, Any]] = []
    has_regressed = False
    
    for item in top5_items:
        cal_id = item.get("cal_id", "unknown")
        regression_status = item.get("regression_status", "STABLE")
        combined_tensor_norm = item.get("combined_tensor_norm", 0.0)
        semantic_hotspots = item.get("semantic_hotspots", [])
        hotspots_count = len(semantic_hotspots) if isinstance(semantic_hotspots, list) else 0
        
        if regression_status == "REGRESSED":
            has_regressed = True
        
        top5.append({
            "cal_id": cal_id,
            "regression_status": regression_status,
            "combined_tensor_norm": float(combined_tensor_norm),
            "hotspots_count": hotspots_count,
        })
    
    # Build advisory warning if any regressed
    advisory_warning: Optional[str] = None
    if has_regressed:
        regressed_count = sum(
            1 for item in items if item.get("regression_status") == "REGRESSED"
        )
        advisory_warning = (
            f"Semantic drift triage index contains {regressed_count} "
            f"experiment(s) with REGRESSED status. Review recommended."
        )
    
    signal = {
        "total_items": total_items,
        "top5": top5,
    }
    
    if advisory_warning is not None:
        signal["advisory_warning"] = advisory_warning
    
    return signal


def attach_semantic_failure_triage_index_to_evidence(
    evidence: Dict[str, Any],
    triage_index: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach semantic failure triage index to evidence pack.

    STATUS: PHASE X — SEMANTIC DRIFT TRIAGE INDEX EVIDENCE INTEGRATION

    Stores the global triage index under evidence["governance"]["semantic_failure_triage_index"]
    and extracts a compact signal under evidence["signals"]["semantic_failure_triage_index"]
    for inclusion in evidence packs.

    SHADOW MODE CONTRACT:
    - This function modifies the evidence dict in-place
    - The attachment is purely observational
    - No control flow depends on the attached data

    Args:
        evidence: Evidence pack dictionary (will be modified in-place)
        triage_index: Triage index from build_semantic_failure_triage_index()

    Returns:
        Modified evidence dictionary with triage index attached
    """
    # Ensure governance section exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    
    # Attach triage index
    evidence["governance"]["semantic_failure_triage_index"] = triage_index
    
    # Extract and attach status signal
    signal = extract_semantic_failure_triage_index_signal(triage_index)
    
    # Ensure signals section exists
    if "signals" not in evidence:
        evidence["signals"] = {}
    
    # Attach signal
    evidence["signals"]["semantic_failure_triage_index"] = signal
    
    return evidence


def build_semantic_failure_triage_index(
    shelves: List[Dict[str, Any]],
    max_items: int = 10,
) -> Dict[str, Any]:
    """
    Build global semantic failure triage index from multiple CAL-EXP shelves.

    STATUS: PHASE X — SEMANTIC DRIFT TRIAGE INDEX

    Aggregates individual failure shelves from CAL-EXP-1/2/3 into a global triage index
    for auditors. Ranks items by severity and truncates to top N.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned index is purely observational
    - No control flow depends on the index contents
    - Non-mutating: does not modify input shelves

    Args:
        shelves: List of semantic failure shelves (each with cal_id, p3_tensor_norm,
            p4_tensor_norm, semantic_hotspots, regression_status).
        max_items: Maximum number of items to include in triage index (default: 10).

    Returns:
        Triage index dictionary with:
        - schema_version: "1.0.0"
        - items: List of ranked items, each with:
            - cal_id: str
            - semantic_hotspots: List[str]
            - p3_tensor_norm: float
            - p4_tensor_norm: float
            - regression_status: str
            - combined_tensor_norm: float (p3 + p4)
            - shelf_path_hint: str (path to shelf file, e.g., "calibration/semantic_failure_shelf_CAL-EXP-1.json")
        - total_shelves: int
        - neutral_notes: List[str]

    Ranking Logic:
        Items are ranked by:
        1. Regression status (REGRESSED > ATTENTION > STABLE)
        2. Combined tensor norm (p3 + p4, descending)
        3. Number of hotspots (descending)
        4. cal_id (alphabetical, for determinism)
    """
    if not shelves:
        return {
            "schema_version": "1.0.0",
            "items": [],
            "total_shelves": 0,
            "neutral_notes": ["No semantic failure shelves provided for triage index."],
        }
    
    # Build items with severity scores
    items: List[Dict[str, Any]] = []
    
    for shelf in shelves:
        cal_id = shelf.get("cal_id", "unknown")
        p3_tensor_norm = shelf.get("p3_tensor_norm", 0.0)
        p4_tensor_norm = shelf.get("p4_tensor_norm", 0.0)
        semantic_hotspots = shelf.get("semantic_hotspots", [])
        regression_status = shelf.get("regression_status", "STABLE")
        
        # Calculate combined tensor norm
        combined_tensor_norm = float(p3_tensor_norm) + float(p4_tensor_norm)
        
        # Calculate severity score for ranking
        # Regression status weight: REGRESSED=100, ATTENTION=50, STABLE=0
        status_weight = {
            "REGRESSED": 100.0,
            "ATTENTION": 50.0,
            "STABLE": 0.0,
        }.get(regression_status, 0.0)
        
        # Combined tensor norm weight (scaled to 0-50 range, assuming max ~5.0)
        tensor_weight = min(50.0, combined_tensor_norm * 10.0)
        
        # Hotspot count weight (max 5 hotspots, so max 5 points)
        hotspot_weight = min(5.0, len(semantic_hotspots))
        
        severity_score = status_weight + tensor_weight + hotspot_weight
        
        # Build shelf path hint for navigation
        shelf_path_hint = f"calibration/semantic_failure_shelf_{cal_id}.json"
        
        items.append({
            "cal_id": cal_id,
            "semantic_hotspots": semantic_hotspots if isinstance(semantic_hotspots, list) else [],
            "p3_tensor_norm": float(p3_tensor_norm),
            "p4_tensor_norm": float(p4_tensor_norm),
            "regression_status": regression_status,
            "combined_tensor_norm": combined_tensor_norm,
            "shelf_path_hint": shelf_path_hint,
            "severity_score": severity_score,
        })
    
    # Sort by severity score (descending), then by cal_id (ascending) for determinism
    items.sort(key=lambda x: (-x["severity_score"], x["cal_id"]))
    
    # Truncate to max_items
    items = items[:max_items]
    
    # Remove severity_score from final output (it's only for ranking)
    for item in items:
        item.pop("severity_score", None)
    
    # Build neutral notes
    neutral_notes: List[str] = []
    neutral_notes.append(
        f"Triage index compiled from {len(shelves)} calibration experiment shelf(s)."
    )
    neutral_notes.append(
        f"Top {len(items)} item(s) ranked by regression status and tensor norm magnitude."
    )
    
    regressed_count = sum(1 for item in items if item["regression_status"] == "REGRESSED")
    attention_count = sum(1 for item in items if item["regression_status"] == "ATTENTION")
    
    if regressed_count > 0:
        neutral_notes.append(f"{regressed_count} experiment(s) with REGRESSED status.")
    if attention_count > 0:
        neutral_notes.append(f"{attention_count} experiment(s) with ATTENTION status.")
    
    return {
        "schema_version": "1.0.0",
        "items": items,
        "total_shelves": len(shelves),
        "neutral_notes": neutral_notes,
    }


def extract_semantic_drift_triage_signal_for_ggfl(
    triage_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract semantic drift triage signal for GGFL integration (SIG-SDRIFT).

    STATUS: PHASE X — SEMANTIC DRIFT TRIAGE INDEX GGFL ADAPTER

    Converts semantic failure triage index signal to GGFL format for inclusion
    in build_global_alignment_view(). Provides LOW weight signal with conflict=false.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - Advisory only: warns if REGRESSED exists in top5

    Args:
        triage_signal: Semantic failure triage index signal from extract_semantic_failure_triage_index_signal()

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-SDRIFT" (constant)
        - status: "ok" | "warn" (lowercase, warn if any REGRESSED in top5)
        - conflict: False (invariant: semantic drift triage never conflicts)
        - weight_hint: "LOW" (constant, advisory weight)
        - drivers: List[str] (reason codes only, max 3, deterministic ordering)
            - DRIVER_REGRESSED_PRESENT (if regressed_count > 0)
            - DRIVER_TOP_CAL_IDS_PRESENT (if regressed cal_ids present)
        - summary: str (one neutral sentence)
        - shadow_mode_invariants: Dict with observational_only, no_control_flow, advisory_weight
        - total_items: int
        - regressed_count: int

    Example:
        >>> triage_signal = extract_semantic_failure_triage_index_signal(triage_index)
        >>> ggfl_signal = extract_semantic_drift_triage_signal_for_ggfl(triage_signal)
        >>> view = build_global_alignment_view(semantic_drift=ggfl_signal, ...)
    """
    top5 = triage_signal.get("top5", [])
    total_items = triage_signal.get("total_items", 0)
    
    # Count REGRESSED items in top5
    regressed_count = sum(
        1 for item in top5 if item.get("regression_status") == "REGRESSED"
    )
    
    # Determine status: warn if any REGRESSED, else ok (lowercase)
    status = "warn" if regressed_count > 0 else "ok"
    
    # Build reason code drivers (deterministic ordering: regressed → top cal_ids)
    drivers: List[str] = []
    
    # 1. DRIVER_REGRESSED_PRESENT (if regressed_count > 0)
    if regressed_count > 0:
        drivers.append("DRIVER_REGRESSED_PRESENT")
    
    # 2. DRIVER_TOP_CAL_IDS_PRESENT (if regressed cal_ids present)
    regressed_cal_ids = [
        item["cal_id"]
        for item in top5
        if item.get("regression_status") == "REGRESSED"
    ][:3]  # Top 3 cal_ids
    if regressed_cal_ids:
        drivers.append("DRIVER_TOP_CAL_IDS_PRESENT")
    
    # Limit to max 3 (shouldn't exceed, but defensive)
    drivers = drivers[:3]
    
    # Build neutral summary
    if total_items == 0:
        summary = "Semantic drift triage index: no items analyzed"
    elif regressed_count > 0:
        summary = (
            f"Semantic drift triage index: {regressed_count} of {total_items} "
            f"item(s) with REGRESSED status in top5"
        )
    else:
        summary = (
            f"Semantic drift triage index: {total_items} item(s) analyzed, "
            f"no REGRESSED status detected"
        )
    
    return {
        "signal_type": "SIG-SDRIFT",
        "status": status,  # Frozen enum: "ok" | "warn"
        "conflict": False,  # Invariant: semantic drift triage never conflicts
        "weight_hint": "LOW",  # Constant, advisory weight
        "drivers": drivers,  # Reason codes only, no prose
        "summary": summary,
        "shadow_mode_invariants": {
            "observational_only": True,
            "no_control_flow": True,
            "advisory_weight": "LOW",
        },
        "total_items": total_items,
        "regressed_count": regressed_count,
    }


def summarize_semantic_drift_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize semantic drift for uplift council decision-making.

    STATUS: PHASE X — SEMANTIC DRIFT COUNCIL ADAPTER

    Maps semantic drift governance tile to council decision signals:
    - BLOCK/BROKEN → BLOCK
    - ATTENTION/WARN → WARN
    - OK → OK

    SHADOW MODE CONTRACT:
    - This function is read-only
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - This is a drift observer, not a hard control path

    Args:
        tile: Semantic drift governance tile from build_semantic_drift_governance_tile()

    Returns:
        Council summary dictionary with:
        - status: "OK" | "WARN" | "BLOCK"
        - semantic_hotspots: List[str]
        - tensor_norm: float
        - gating_recommendation: str
        - advisory: str (neutral explanation)
    """
    gating_recommendation = tile.get("gating_recommendation", "OK")
    status_light = tile.get("status_light", "GREEN")
    
    # Map to council status: BLOCK/BROKEN → BLOCK, ATTENTION/WARN → WARN, OK → OK
    # Check both gating_recommendation and status_light for robust mapping
    if gating_recommendation == "BLOCK" or status_light == "RED":
        council_status = "BLOCK"
    elif gating_recommendation == "WARN" or status_light == "YELLOW":
        council_status = "WARN"
    else:  # OK or GREEN
        council_status = "OK"
    
    # Build advisory message
    semantic_hotspots = tile.get("semantic_hotspots", [])
    projected_count = tile.get("projected_instability_count", 0)
    
    if council_status == "BLOCK":
        advisory = (
            f"Semantic drift analysis indicates blocking condition: "
            f"{len(semantic_hotspots)} hotspot(s), {projected_count} projected unstable slice(s)."
        )
    elif council_status == "WARN":
        advisory = (
            f"Semantic drift analysis indicates attention required: "
            f"{len(semantic_hotspots)} hotspot(s), {projected_count} projected unstable slice(s)."
        )
    else:
        advisory = (
            f"Semantic drift analysis indicates stable condition: "
            f"{len(semantic_hotspots)} hotspot(s), {projected_count} projected unstable slice(s)."
        )
    
    return {
        "status": council_status,
        "semantic_hotspots": sorted(tile.get("semantic_hotspots", [])),  # Sorted for determinism
        "tensor_norm": tile.get("tensor_norm", 0.0),
        "gating_recommendation": gating_recommendation,
        "advisory": advisory,
    }


__all__ = [
    "SEMANTIC_DRIFT_GOVERNANCE_TILE_SCHEMA_VERSION",
    "build_semantic_drift_governance_tile",
    "extract_drift_advisory_for_uplift",
    "build_semantic_drift_summary_for_p3",
    "build_semantic_drift_calibration_for_p4",
    "build_semantic_drift_failure_shelf_for_first_light",
    "emit_cal_exp_semantic_failure_shelf",
    "build_semantic_failure_triage_index",
    "extract_semantic_failure_triage_index_signal",
    "extract_semantic_drift_triage_signal_for_ggfl",
    "attach_semantic_drift_to_evidence",
    "attach_semantic_failure_triage_index_to_evidence",
    "summarize_semantic_drift_for_uplift_council",
]

