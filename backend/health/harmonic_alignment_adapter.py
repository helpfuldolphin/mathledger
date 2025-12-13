"""Harmonic alignment integration adapter for global health.

STATUS: PHASE X — HARMONIC GOVERNANCE TILE

Provides integration between semantic-curriculum harmonic alignment signals
(harmonic map, evolution forecaster, director panel) and the global health
surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The semantic_curriculum_harmonic tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

from typing import Any, Dict, Optional

HARMONIC_ALIGNMENT_TILE_SCHEMA_VERSION = "1.0.0"

# Forbidden words for neutral language enforcement
FORBIDDEN_LANGUAGE = {
    "good", "bad", "better", "worse", "improve", "improvement",
    "should", "must", "need", "required", "fail", "success",
    "correct", "incorrect", "right", "wrong", "fix", "broken",
}


def _validate_harmonic_map(harmonic_map: Dict[str, Any]) -> None:
    """Validate harmonic map structure.
    
    Args:
        harmonic_map: Harmonic map dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["harmonic_scores", "convergence_band", "misaligned_concepts"]
    missing = [key for key in required_keys if key not in harmonic_map]
    if missing:
        raise ValueError(
            f"harmonic_map missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(harmonic_map.keys()))}"
        )


def _validate_evolution_forecaster(evolution_forecaster: Dict[str, Any]) -> None:
    """Validate evolution forecaster structure.
    
    Args:
        evolution_forecaster: Evolution forecaster dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["forecast_status", "forecasted_adjustments"]
    missing = [key for key in required_keys if key not in evolution_forecaster]
    if missing:
        raise ValueError(
            f"evolution_forecaster missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(evolution_forecaster.keys()))}"
        )


def _validate_harmonic_director_panel(harmonic_director_panel: Dict[str, Any]) -> None:
    """Validate harmonic director panel structure.
    
    Args:
        harmonic_director_panel: Harmonic director panel dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["status_light", "headline"]
    missing = [key for key in required_keys if key not in harmonic_director_panel]
    if missing:
        raise ValueError(
            f"harmonic_director_panel missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(harmonic_director_panel.keys()))}"
        )


def _check_neutral_language(text: str) -> bool:
    """Check if text contains neutral language only.
    
    Args:
        text: Text to check
    
    Returns:
        True if text is neutral, False if it contains forbidden words
    """
    text_lower = text.lower()
    words = text_lower.split()
    for word in words:
        # Remove punctuation for comparison
        word_clean = word.strip(".,!?;:()[]{}'\"")
        if word_clean in FORBIDDEN_LANGUAGE:
            return False
    return True


def build_harmonic_governance_tile(
    harmonic_map: Dict[str, Any],
    evolution_forecaster: Dict[str, Any],
    harmonic_director_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build harmonic governance tile for global health surface.
    
    STATUS: PHASE X — HARMONIC GOVERNANCE TILE
    
    Integrates semantic-curriculum harmonic alignment signals (harmonic map,
    evolution forecaster, director panel) into a unified governance tile for
    the global health dashboard.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    
    Args:
        harmonic_map: Harmonic map from build_semantic_curriculum_harmonic_map().
            Must contain: harmonic_scores, convergence_band, misaligned_concepts
            May contain: neutral_notes
        evolution_forecaster: Evolution forecaster from build_curriculum_evolution_forecaster().
            Must contain: forecast_status, forecasted_adjustments
            May contain: neutral_notes
        harmonic_director_panel: Director panel from build_harmonic_director_panel().
            Must contain: status_light, headline
            May contain: convergence_band, forecast_status, misaligned_count, integrated_risks
    
    Returns:
        Harmonic governance tile dictionary with:
        - schema_version: "1.0.0"
        - tile_type: "semantic_curriculum_harmonic"
        - status_light: "GREEN" | "YELLOW" | "RED" (from harmonic_director_panel)
        - harmonic_band: "COHERENT" | "PARTIAL" | "MISMATCHED" (from harmonic_map)
        - global_harmonic_score: float (mean of slice scores from harmonic_map)
        - misaligned_concepts: List[str] (from harmonic_map)
        - evolution_status: "STABLE" | "EVOLVING" | "DIVERGING" (from evolution_forecaster)
        - prioritized_adjustments: List[str] (top 3 adjustment slice names from forecaster)
        - headline: str (from harmonic_director_panel, neutral descriptive text)
    """
    # Validate inputs
    _validate_harmonic_map(harmonic_map)
    _validate_evolution_forecaster(evolution_forecaster)
    _validate_harmonic_director_panel(harmonic_director_panel)
    
    # Extract core fields
    status_light = harmonic_director_panel["status_light"]
    harmonic_band = harmonic_map["convergence_band"]
    misaligned_concepts = sorted(harmonic_map.get("misaligned_concepts", []))
    evolution_status = evolution_forecaster["forecast_status"]
    
    # Calculate global harmonic score (mean of all slice scores)
    harmonic_scores = harmonic_map.get("harmonic_scores", {})
    if harmonic_scores:
        global_harmonic_score = sum(harmonic_scores.values()) / len(harmonic_scores)
        global_harmonic_score = round(global_harmonic_score, 6)
    else:
        global_harmonic_score = 0.0
    
    # Extract top 3 prioritized adjustments (slice names only)
    forecasted_adjustments = evolution_forecaster.get("forecasted_adjustments", [])
    # Sort by priority (HIGH first), then take top 3
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_adjustments = sorted(
        forecasted_adjustments,
        key=lambda a: (priority_order.get(a.get("priority", "LOW"), 99), a.get("slice", ""))
    )
    prioritized_adjustments = [adj.get("slice") for adj in sorted_adjustments[:3] if adj.get("slice")]
    
    # Extract headline from director panel
    headline = harmonic_director_panel.get("headline", "")
    
    # Verify headline neutrality
    if headline and not _check_neutral_language(headline):
        # Use minimal fallback
        headline = (
            f"Semantic-curriculum harmonic alignment: {harmonic_band} "
            f"({len(misaligned_concepts)} misaligned concept(s)), "
            f"evolution {evolution_status}, global score {global_harmonic_score:.3f}"
        )
    
    # Build tile
    tile = {
        "schema_version": HARMONIC_ALIGNMENT_TILE_SCHEMA_VERSION,
        "tile_type": "semantic_curriculum_harmonic",
        "status_light": status_light,
        "harmonic_band": harmonic_band,
        "global_harmonic_score": global_harmonic_score,
        "misaligned_concepts": misaligned_concepts,
        "evolution_status": evolution_status,
        "prioritized_adjustments": prioritized_adjustments,
        "headline": headline,
    }
    
    return tile


def extract_harmonic_signal_for_curriculum(
    harmonic_map: Dict[str, Any],
    evolution_forecaster: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract compact harmonic signal for curriculum/atlas systems.
    
    This helper extracts the minimal set of harmonic signals needed for
    curriculum and atlas systems in advisory mode.
    
    Args:
        harmonic_map: Harmonic map from build_semantic_curriculum_harmonic_map().
            Must contain: convergence_band, misaligned_concepts
        evolution_forecaster: Evolution forecaster from build_curriculum_evolution_forecaster().
            Must contain: forecast_status
    
    Returns:
        Dictionary with:
        - harmonic_band: str (COHERENT|PARTIAL|MISMATCHED)
        - num_misaligned_concepts: int
        - evolution_status: str (STABLE|EVOLVING|DIVERGING)
    """
    _validate_harmonic_map(harmonic_map)
    _validate_evolution_forecaster(evolution_forecaster)
    
    return {
        "harmonic_band": harmonic_map["convergence_band"],
        "num_misaligned_concepts": len(harmonic_map.get("misaligned_concepts", [])),
        "evolution_status": evolution_forecaster["forecast_status"],
    }


def build_harmonic_alignment_summary_for_p3(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build harmonic alignment summary for P3 First-Light summary.json.

    STATUS: PHASE X — HARMONIC GOVERNANCE TILE

    Extracts key harmonic alignment metrics for inclusion in First-Light P3 summary.json.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents

    Args:
        tile: Harmonic governance tile from build_harmonic_governance_tile().

    Returns:
        Harmonic alignment summary dictionary with:
        - global_harmonic_score: float
        - harmonic_band: "COHERENT" | "PARTIAL" | "MISMATCHED"
        - misaligned_concepts: List[str]
        - priority_adjustments: List[str] (top 3 slice names)
        - status_light: "GREEN" | "YELLOW" | "RED"
    """
    return {
        "global_harmonic_score": tile.get("global_harmonic_score", 0.0),
        "harmonic_band": tile.get("harmonic_band", "PARTIAL"),
        "misaligned_concepts": sorted(tile.get("misaligned_concepts", [])),
        "priority_adjustments": tile.get("prioritized_adjustments", []),
        "status_light": tile.get("status_light", "GREEN"),
    }


def build_harmonic_alignment_calibration_for_p4(
    tile: Dict[str, Any],
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build harmonic alignment calibration for P4 calibration report.

    STATUS: PHASE X — HARMONIC GOVERNANCE TILE

    Extracts harmonic alignment data for inclusion in P4 calibration reports.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned calibration is purely observational
    - No control flow depends on the calibration contents

    Args:
        tile: Harmonic governance tile from build_harmonic_governance_tile().
        signal: Harmonic signal from extract_harmonic_signal_for_curriculum().

    Returns:
        Harmonic alignment calibration dictionary with:
        - harmonic_band: "COHERENT" | "PARTIAL" | "MISMATCHED"
        - misaligned_concepts: List[str]
        - evolution_status: "STABLE" | "EVOLVING" | "DIVERGING"
        - priority_adjustments: List[str] (top 3 slice names)
        - global_harmonic_score: float
    """
    return {
        "harmonic_band": signal.get("harmonic_band", tile.get("harmonic_band", "PARTIAL")),
        "misaligned_concepts": sorted(tile.get("misaligned_concepts", [])),
        "evolution_status": signal.get("evolution_status", "STABLE"),
        "priority_adjustments": tile.get("prioritized_adjustments", []),
        "global_harmonic_score": tile.get("global_harmonic_score", 0.0),
    }


def attach_harmonic_alignment_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    signal: Dict[str, Any],
    p3_summary: Optional[Dict[str, Any]] = None,
    p4_calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach harmonic alignment to evidence pack.

    STATUS: PHASE X — HARMONIC GOVERNANCE TILE

    Stores harmonic alignment information under evidence["governance"]["harmonic_alignment"]
    for inclusion in evidence packs. Optionally includes curriculum annex if P3/P4 data provided.

    SHADOW MODE CONTRACT:
    - This function modifies the evidence dict in-place
    - The attachment is purely observational
    - No control flow depends on the attached data
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Evidence pack dictionary (will be copied, not modified in-place)
        tile: Harmonic governance tile from build_harmonic_governance_tile()
        signal: Harmonic signal from extract_harmonic_signal_for_curriculum()
        p3_summary: Optional P3 First-Light summary with harmonic_alignment_summary field
        p4_calibration: Optional P4 calibration report with harmonic_alignment field

    Returns:
        New dict with evidence contents plus harmonic_alignment attached under governance key.
        The attached data includes:
        - harmonic_band: "COHERENT" | "PARTIAL" | "MISMATCHED"
        - score: float (global_harmonic_score)
        - misaligned_concepts: List[str]
        - curriculum_annex: Optional dict (if p3_summary and p4_calibration provided)
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Ensure governance structure exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    else:
        enriched["governance"] = dict(enriched["governance"])

    # Attach harmonic alignment (minimal fields as specified)
    enriched["governance"]["harmonic_alignment"] = {
        "harmonic_band": signal.get("harmonic_band", tile.get("harmonic_band", "PARTIAL")),
        "score": tile.get("global_harmonic_score", 0.0),
        "misaligned_concepts": sorted(tile.get("misaligned_concepts", [])),
    }

    # Attach curriculum annex if P3/P4 data provided
    if p3_summary is not None and p4_calibration is not None:
        from backend.health.harmonic_alignment_p3p4_integration import (
            build_curriculum_harmonic_annex,
        )
        # Extract harmonic_alignment_summary from p3_summary if nested
        p3_harmonic = p3_summary.get("harmonic_alignment_summary", p3_summary)
        # Extract harmonic_alignment from p4_calibration if nested
        p4_harmonic = p4_calibration.get("harmonic_alignment", p4_calibration)
        
        curriculum_annex = build_curriculum_harmonic_annex(p3_harmonic, p4_harmonic)
        enriched["governance"]["harmonic_alignment"]["curriculum_annex"] = curriculum_annex

    return enriched


def summarize_harmonic_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize harmonic alignment for uplift council decision-making.

    STATUS: PHASE X — HARMONIC GOVERNANCE TILE

    Maps harmonic alignment band to council decision signals:
    - MISMATCHED → BLOCK
    - PARTIAL → WARN
    - COHERENT → OK

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - This is an alignment observer, not a hard control path

    Args:
        tile: Harmonic governance tile from build_harmonic_governance_tile()

    Returns:
        Council summary dictionary with:
        - status: "OK" | "WARN" | "BLOCK"
        - misaligned_concepts: List[str]
        - priority_adjustments: List[str] (top 3 slice names)
        - num_misaligned_concepts: int
        - num_priority_adjustments: int
    """
    harmonic_band = tile.get("harmonic_band", "PARTIAL")
    misaligned_concepts = sorted(tile.get("misaligned_concepts", []))
    priority_adjustments = tile.get("prioritized_adjustments", [])

    # Map harmonic band to council status
    if harmonic_band == "MISMATCHED":
        status = "BLOCK"
    elif harmonic_band == "PARTIAL":
        status = "WARN"
    else:  # COHERENT
        status = "OK"

    return {
        "status": status,
        "misaligned_concepts": misaligned_concepts,
        "priority_adjustments": priority_adjustments,
        "num_misaligned_concepts": len(misaligned_concepts),
        "num_priority_adjustments": len(priority_adjustments),
    }


__all__ = [
    "HARMONIC_ALIGNMENT_TILE_SCHEMA_VERSION",
    "build_harmonic_governance_tile",
    "extract_harmonic_signal_for_curriculum",
    "build_harmonic_alignment_summary_for_p3",
    "build_harmonic_alignment_calibration_for_p4",
    "attach_harmonic_alignment_to_evidence",
    "summarize_harmonic_for_uplift_council",
]

