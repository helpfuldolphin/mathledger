"""Convergence pressure integration adapter for global health.

STATUS: PHASE X — CONVERGENCE GOVERNANCE INTEGRATION

Provides integration between Phase V convergence pressure analysis and the global
health surface builder. This adapter consumes the pressure tensor, early-warning
radar, and director tile to produce a unified tile for global health dashboards.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The convergence_pressure tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

from typing import Any, Dict, List, Optional

CONVERGENCE_PRESSURE_TILE_SCHEMA_VERSION = "1.0.0"

# Forbidden words for neutral language enforcement
FORBIDDEN_LANGUAGE = {
    "good", "bad", "better", "worse", "improve", "improvement",
    "should", "must", "need", "required", "fail", "success",
    "correct", "incorrect", "right", "wrong", "fix", "broken",
}


def _validate_pressure_tensor(tensor: Dict[str, Any]) -> None:
    """Validate pressure tensor structure.
    
    Args:
        tensor: Pressure tensor dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["global_pressure_norm"]
    missing = [key for key in required_keys if key not in tensor]
    if missing:
        raise ValueError(
            f"pressure_tensor missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(tensor.keys()))}"
        )


def _validate_early_warning(early_warning: Dict[str, Any]) -> None:
    """Validate early-warning radar structure.
    
    Args:
        early_warning: Early-warning radar dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["transition_likelihood_band"]
    missing = [key for key in required_keys if key not in early_warning]
    if missing:
        raise ValueError(
            f"early_warning missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(early_warning.keys()))}"
        )


def _validate_director_tile(director_tile: Dict[str, Any]) -> None:
    """Validate director tile structure.
    
    Args:
        director_tile: Director tile dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["status_light"]
    missing = [key for key in required_keys if key not in director_tile]
    if missing:
        raise ValueError(
            f"director_tile missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(director_tile.keys()))}"
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


def _compute_status_light(
    pressure_norm: float,
    transition_band: str,
) -> str:
    """
    Compute status light using director tile rules.
    
    Rules (from director tile):
    - GREEN: pressure < 1.0 AND transition_band == 'LOW'
    - RED: pressure > 2.0 OR transition_band == 'HIGH'
    - YELLOW: Otherwise
    
    Args:
        pressure_norm: Global pressure norm
        transition_band: Transition likelihood band (LOW/MEDIUM/HIGH)
    
    Returns:
        Status light: GREEN, YELLOW, or RED
    """
    if pressure_norm < 1.0 and transition_band == "LOW":
        return "GREEN"
    elif pressure_norm > 2.0 or transition_band == "HIGH":
        return "RED"
    else:
        return "YELLOW"


def _extract_slices_at_risk(
    pressure_tensor: Dict[str, Any],
    early_warning: Dict[str, Any],
) -> List[str]:
    """
    Extract slices at risk from pressure tensor and early warning.
    
    Uses first_slices_at_risk from early_warning if available,
    otherwise falls back to pressure_ranked_slices from tensor.
    Limits to top 3 slices.
    
    Args:
        pressure_tensor: Pressure tensor dictionary
        early_warning: Early-warning radar dictionary
    
    Returns:
        List of slice names at risk (max 3, sorted alphabetically)
    """
    # Prefer early_warning's first_slices_at_risk
    if "first_slices_at_risk" in early_warning:
        slices = early_warning["first_slices_at_risk"]
    elif "pressure_ranked_slices" in pressure_tensor:
        slices = pressure_tensor["pressure_ranked_slices"]
    else:
        slices = []
    
    # Limit to top 3 and sort for determinism
    slices_limited = sorted(slices[:3])
    return slices_limited


def _extract_pressure_drivers(
    early_warning: Dict[str, Any],
) -> List[str]:
    """
    Extract top 3 pressure drivers from early warning.
    
    Args:
        early_warning: Early-warning radar dictionary
    
    Returns:
        List of pressure driver strings (max 3)
    """
    drivers = early_warning.get("root_drivers", [])
    # Limit to top 3 for dashboard space constraints
    return drivers[:3]


def _build_neutral_headline(
    pressure_norm: float,
    transition_band: str,
    slices_at_risk: List[str],
) -> str:
    """
    Build neutral headline for convergence pressure tile.
    
    Format: "Convergence status: pressure norm: {norm}, transition likelihood: {band}, {N} slice(s) at elevated risk"
    
    Args:
        pressure_norm: Global pressure norm
        transition_band: Transition likelihood band
        slices_at_risk: List of slice names at risk
    
    Returns:
        Neutral headline string
    """
    num_slices = len(slices_at_risk)
    slice_text = f"{num_slices} slice(s) at elevated risk" if num_slices > 0 else "no slices at elevated risk"
    
    headline = (
        f"Convergence status: pressure norm: {pressure_norm:.2f}, "
        f"transition likelihood: {transition_band.lower()}, {slice_text}"
    )
    
    # Verify neutrality
    if not _check_neutral_language(headline):
        # Fallback to minimal neutral headline
        headline = (
            f"Convergence pressure norm: {pressure_norm:.2f}, "
            f"transition band: {transition_band}, {num_slices} slice(s) monitored"
        )
    
    return headline


def build_convergence_pressure_tile(
    pressure_tensor: Dict[str, Any],
    early_warning: Dict[str, Any],
    director_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build convergence pressure tile for global health surface.
    
    STATUS: PHASE X — CONVERGENCE GOVERNANCE INTEGRATION
    
    Integrates Phase V convergence pressure analysis (pressure tensor, early-warning
    radar, director tile) into a unified tile for the global health dashboard.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    
    Args:
        pressure_tensor: Pressure tensor from build_convergence_pressure_tensor.
            Must contain: global_pressure_norm
            May contain: pressure_ranked_slices, slice_pressure_vectors
        early_warning: Early-warning radar from build_phase_transition_early_warning_radar.
            Must contain: transition_likelihood_band
            May contain: root_drivers, first_slices_at_risk, time_to_inflection_estimate
        director_tile: Director tile from build_convergence_director_tile.
            Must contain: status_light
            May contain: headline, pressure_drivers
    
    Returns:
        Convergence pressure tile dictionary with:
        - schema_version
        - tile_type: "convergence_pressure"
        - status_light: "GREEN" | "YELLOW" | "RED" (computed using director tile rules)
        - global_pressure_norm: float
        - transition_likelihood_band: str (LOW/MEDIUM/HIGH)
        - slices_at_risk: List[str] (top 3 slices, sorted)
        - pressure_drivers: List[str] (top 3 drivers)
        - headline: str (neutral descriptive text)
    """
    # Validate inputs
    _validate_pressure_tensor(pressure_tensor)
    _validate_early_warning(early_warning)
    _validate_director_tile(director_tile)
    
    # Extract core fields
    pressure_norm = pressure_tensor["global_pressure_norm"]
    transition_band = early_warning["transition_likelihood_band"]
    director_status_light = director_tile["status_light"]
    
    # Compute status light using director tile rules
    status_light = _compute_status_light(pressure_norm, transition_band)
    
    # Extract slices at risk
    slices_at_risk = _extract_slices_at_risk(pressure_tensor, early_warning)
    
    # Extract pressure drivers
    pressure_drivers = _extract_pressure_drivers(early_warning)
    
    # Build neutral headline
    headline = _build_neutral_headline(pressure_norm, transition_band, slices_at_risk)
    
    # Verify headline neutrality
    if not _check_neutral_language(headline):
        # Use minimal fallback
        headline = f"Convergence pressure norm: {pressure_norm:.2f}, transition band: {transition_band}"
    
    # Build tile
    tile = {
        "schema_version": CONVERGENCE_PRESSURE_TILE_SCHEMA_VERSION,
        "tile_type": "convergence_pressure",
        "status_light": status_light,
        "global_pressure_norm": round(pressure_norm, 4),
        "transition_likelihood_band": transition_band,
        "slices_at_risk": sorted(slices_at_risk),  # Ensure sorted for determinism
        "pressure_drivers": pressure_drivers,
        "headline": headline,
    }
    
    return tile


def attach_convergence_pressure_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    early_warning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach convergence pressure tile to evidence pack (read-only, additive).

    STATUS: PHASE X — CONVERGENCE GOVERNANCE INTEGRATION

    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["convergence_pressure"].

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input
    - Deterministic: same inputs produce same outputs
    - JSON-safe: all values are JSON-serializable

    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        tile: Convergence pressure tile from build_convergence_pressure_tile().
        early_warning: Optional early-warning radar from
            build_phase_transition_early_warning_radar(). Used to extract
            horizon_estimate (time_to_inflection_estimate).

    Returns:
        New dict with evidence contents plus convergence_pressure tile attached
        under governance key.

    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_convergence_pressure_tile(tensor, early_warning, director_tile)
        >>> enriched = attach_convergence_pressure_to_evidence(evidence, tile, early_warning)
        >>> "governance" in enriched
        True
        >>> "convergence_pressure" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}

    # Extract horizon_estimate from early_warning if available
    horizon_estimate = None
    if early_warning:
        horizon_estimate = early_warning.get("time_to_inflection_estimate")

    # Attach convergence pressure tile
    enriched["governance"] = enriched["governance"].copy()
    enriched["governance"]["convergence_pressure"] = {
        "global_pressure_norm": tile.get("global_pressure_norm", 0.0),
        "transition_likelihood_band": tile.get("transition_likelihood_band", "LOW"),
        "slices_at_risk": sorted(tile.get("slices_at_risk", [])),  # Sorted for determinism
        "pressure_drivers": tile.get("pressure_drivers", []),
        "horizon_estimate": horizon_estimate,
    }

    return enriched


def summarize_convergence_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize convergence pressure for Uplift Council aggregation.

    STATUS: PHASE X — CONVERGENCE GOVERNANCE INTEGRATION

    Normalizes convergence pressure tile to council-compatible format:
    - status: "OK" | "WARN" | "BLOCK"
    - slices_at_risk: List of slice names
    - band: Transition likelihood band

    Council Rules:
    - BLOCK: band == "HIGH" OR global_pressure_norm > 2.0
    - WARN: band == "MEDIUM"
    - OK: Otherwise

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The returned summary is purely advisory
    - No control flow depends on this summary
    - Compatible with Uplift Council aggregator

    Args:
        tile: Convergence pressure tile from build_convergence_pressure_tile().

    Returns:
        Council-compatible summary dictionary with:
        - status: "OK" | "WARN" | "BLOCK"
        - slices_at_risk: List[str] (sorted alphabetically)
        - band: str (transition likelihood band)
    """
    global_pressure_norm = tile.get("global_pressure_norm", 0.0)
    transition_band = tile.get("transition_likelihood_band", "LOW")
    slices_at_risk = tile.get("slices_at_risk", [])

    # Determine council status
    if transition_band == "HIGH" or global_pressure_norm > 2.0:
        status = "BLOCK"
    elif transition_band == "MEDIUM":
        status = "WARN"
    else:
        status = "OK"

    return {
        "status": status,
        "slices_at_risk": sorted(slices_at_risk),  # Sorted for determinism
        "band": transition_band,
    }


__all__ = [
    "CONVERGENCE_PRESSURE_TILE_SCHEMA_VERSION",
    "attach_convergence_pressure_to_evidence",
    "build_convergence_pressure_tile",
    "summarize_convergence_for_uplift_council",
]

