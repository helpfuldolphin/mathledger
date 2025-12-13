"""Atlas governance integration adapter for global health.

STATUS: PHASE X — ATLAS GOVERNANCE INTEGRATION

Provides integration between Phase VI Atlas Convergence Lattice and the global
health surface builder. This adapter consumes the lattice, phase transition gate,
and director tile v2 to produce a unified governance tile for global health dashboards.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The atlas_governance tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ATLAS_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"
STRUCTURAL_COHESION_ANNEX_SCHEMA_VERSION = "1.0.0"
STRUCTURAL_COHESION_REGISTER_SCHEMA_VERSION = "1.0.0"

# Forbidden words for neutral language enforcement
FORBIDDEN_LANGUAGE = {
    "good", "bad", "better", "worse", "improve", "improvement",
    "should", "must", "need", "required", "fail", "success",
    "correct", "incorrect", "right", "wrong", "fix", "broken",
}


def _validate_lattice(lattice: Dict[str, Any]) -> None:
    """Validate lattice structure.
    
    Args:
        lattice: Lattice dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["convergence_band", "global_lattice_norm"]
    missing = [key for key in required_keys if key not in lattice]
    if missing:
        raise ValueError(
            f"lattice missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(lattice.keys()))}"
        )


def _validate_phase_gate(phase_gate: Dict[str, Any]) -> None:
    """Validate phase transition gate structure.
    
    Args:
        phase_gate: Phase transition gate dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["transition_status"]
    missing = [key for key in required_keys if key not in phase_gate]
    if missing:
        raise ValueError(
            f"phase_gate missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(phase_gate.keys()))}"
        )


def _validate_director_tile_v2(director_tile_v2: Dict[str, Any]) -> None:
    """Validate director tile v2 structure.
    
    Args:
        director_tile_v2: Director tile v2 dictionary
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["status_light"]
    missing = [key for key in required_keys if key not in director_tile_v2]
    if missing:
        raise ValueError(
            f"director_tile_v2 missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(director_tile_v2.keys()))}"
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


def build_atlas_governance_tile(
    lattice: Dict[str, Any],
    phase_gate: Dict[str, Any],
    director_tile_v2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build atlas governance tile for global health surface.
    
    STATUS: PHASE X — ATLAS GOVERNANCE INTEGRATION
    
    Integrates Phase VI Atlas Convergence Lattice (lattice, phase transition gate,
    director tile v2) into a unified governance tile for the global health dashboard.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned tile is purely observational
    - No control flow depends on the tile contents
    
    Args:
        lattice: Lattice from build_atlas_convergence_lattice.
            Must contain: convergence_band, global_lattice_norm
            May contain: lattice_vectors, neutral_notes
        phase_gate: Phase transition gate from derive_atlas_phase_transition_gate.
            Must contain: transition_status
            May contain: drivers, slices_ready, slices_needing_alignment, headline
        director_tile_v2: Director tile v2 from build_atlas_director_tile_v2.
            Must contain: status_light
            May contain: lattice_coherence, structural_status, transition_recommendation, atlas_ok, headline
    
    Returns:
        Atlas governance tile dictionary with:
        - schema_version: "1.0.0"
        - tile_type: "atlas_governance"
        - status_light: "GREEN" | "YELLOW" | "RED" (from director_tile_v2)
        - lattice_coherence_band: str (COHERENT|PARTIAL|MISALIGNED)
        - global_lattice_norm: float
        - transition_status: str (OK|ATTENTION|BLOCK)
        - slices_ready: List[str] (sorted)
        - slices_needing_alignment: List[str] (sorted)
        - headline: str (neutral descriptive text)
    """
    # Validate inputs
    _validate_lattice(lattice)
    _validate_phase_gate(phase_gate)
    _validate_director_tile_v2(director_tile_v2)
    
    # Extract core fields
    status_light = director_tile_v2["status_light"]
    lattice_coherence_band = lattice["convergence_band"]
    global_lattice_norm = lattice["global_lattice_norm"]
    transition_status = phase_gate["transition_status"]
    
    # Extract slices
    slices_ready = sorted(phase_gate.get("slices_ready", []))
    slices_needing_alignment = sorted(phase_gate.get("slices_needing_alignment", []))
    
    # Build neutral headline
    # Prefer director_tile_v2 headline, fallback to phase_gate headline, then construct
    headline = director_tile_v2.get("headline") or phase_gate.get("headline")
    if not headline:
        # Construct minimal neutral headline
        headline = (
            f"Atlas governance: {len(slices_ready)} slices ready, "
            f"{len(slices_needing_alignment)} need alignment, "
            f"lattice norm {global_lattice_norm:.3f} ({lattice_coherence_band}), "
            f"transition {transition_status}"
        )
    
    # Verify headline neutrality
    if not _check_neutral_language(headline):
        # Use minimal fallback
        headline = (
            f"Atlas governance: lattice norm {global_lattice_norm:.3f} "
            f"({lattice_coherence_band}), transition {transition_status}"
        )
    
    # Build tile
    tile = {
        "schema_version": ATLAS_GOVERNANCE_TILE_SCHEMA_VERSION,
        "tile_type": "atlas_governance",
        "status_light": status_light,
        "lattice_coherence_band": lattice_coherence_band,
        "global_lattice_norm": round(global_lattice_norm, 6),
        "transition_status": transition_status,
        "slices_ready": slices_ready,
        "slices_needing_alignment": slices_needing_alignment,
        "headline": headline,
    }
    
    return tile


def extract_atlas_signal_for_first_light(
    lattice: Dict[str, Any],
    phase_gate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract key atlas signals for First Light summary.
    
    This helper extracts the minimal set of atlas signals needed for First Light
    summary generation, focusing on global convergence metrics and transition status.
    
    Args:
        lattice: Lattice from build_atlas_convergence_lattice.
            Must contain: convergence_band, global_lattice_norm
        phase_gate: Phase transition gate from derive_atlas_phase_transition_gate.
            Must contain: transition_status
    
    Returns:
        Dictionary with:
        - global_lattice_norm: float
        - lattice_convergence_band: str (COHERENT|PARTIAL|MISALIGNED)
        - transition_status: str (OK|ATTENTION|BLOCK)
    """
    _validate_lattice(lattice)
    _validate_phase_gate(phase_gate)
    
    return {
        "global_lattice_norm": round(lattice["global_lattice_norm"], 6),
        "lattice_convergence_band": lattice["convergence_band"],
        "transition_status": phase_gate["transition_status"],
    }


def attach_atlas_governance_to_p3_stability_report(
    stability_report: Dict[str, Any],
    governance_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach atlas governance summary to P3 stability report.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Atlas governance is observational only
    
    Args:
        stability_report: P3 stability report dictionary (read-only, not modified)
        governance_tile: Atlas governance tile from build_atlas_governance_tile
        
    Returns:
        New dictionary with atlas_governance_summary field added
    """
    # Extract atlas governance data from tile
    atlas_summary = {
        "lattice_coherence_band": governance_tile.get("lattice_coherence_band", "PARTIAL"),
        "global_lattice_norm": governance_tile.get("global_lattice_norm", 0.0),
        "transition_status": governance_tile.get("transition_status", "ATTENTION"),
        "slices_ready": governance_tile.get("slices_ready", []),
        "slices_needing_alignment": governance_tile.get("slices_needing_alignment", []),
    }
    
    # Create new dict (non-mutating)
    updated_report = dict(stability_report)
    updated_report["atlas_governance_summary"] = atlas_summary
    
    return updated_report


def attach_atlas_governance_to_p4_calibration_report(
    calibration_report: Dict[str, Any],
    governance_tile: Dict[str, Any],
    atlas_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach atlas governance calibration to P4 calibration report.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Atlas governance is observational only
    
    Args:
        calibration_report: P4 calibration report dictionary (read-only, not modified)
        governance_tile: Atlas governance tile from build_atlas_governance_tile
        atlas_signal: Optional atlas signal from extract_atlas_signal_for_first_light
        
    Returns:
        New dictionary with atlas_governance_calibration field added
    """
    # Use signal if provided, otherwise extract from tile
    if atlas_signal is not None:
        lattice_norm = atlas_signal.get("global_lattice_norm", governance_tile.get("global_lattice_norm", 0.0))
        coherence_band = atlas_signal.get("lattice_convergence_band", governance_tile.get("lattice_coherence_band", "PARTIAL"))
        transition_status = atlas_signal.get("transition_status", governance_tile.get("transition_status", "ATTENTION"))
    else:
        lattice_norm = governance_tile.get("global_lattice_norm", 0.0)
        coherence_band = governance_tile.get("lattice_coherence_band", "PARTIAL")
        transition_status = governance_tile.get("transition_status", "ATTENTION")
    
    # Extract atlas governance data
    atlas_calibration = {
        "lattice_coherence_band": coherence_band,
        "global_lattice_norm": lattice_norm,
        "transition_status": transition_status,
        "slices_ready": governance_tile.get("slices_ready", []),
        "slices_needing_alignment": governance_tile.get("slices_needing_alignment", []),
        "calibration_notes": [
            f"Lattice coherence band: {coherence_band}.",
            f"Global lattice norm: {lattice_norm:.3f}.",
            f"Transition status: {transition_status}.",
            f"{len(governance_tile.get('slices_ready', []))} slice{'s' if len(governance_tile.get('slices_ready', [])) != 1 else ''} ready for phase transition.",
            f"{len(governance_tile.get('slices_needing_alignment', []))} slice{'s' if len(governance_tile.get('slices_needing_alignment', [])) != 1 else ''} need alignment.",
        ],
    }
    
    # Create new dict (non-mutating)
    updated_report = dict(calibration_report)
    updated_report["atlas_governance_calibration"] = atlas_calibration
    
    return updated_report


def attach_atlas_governance_to_evidence(
    evidence: Dict[str, Any],
    governance_tile: Dict[str, Any],
    atlas_signal: Optional[Dict[str, Any]] = None,
    structure_tile: Optional[Dict[str, Any]] = None,
    coherence_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach atlas governance tile to an evidence pack (read-only, additive).
    
    STATUS: PHASE X — ATLAS GOVERNANCE EVIDENCE INTEGRATION
    
    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the tile attached
    under evidence["governance"]["atlas"].
    
    Optionally includes Structural Cohesion Annex if structure_tile and/or
    coherence_tile are provided.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached tile is purely observational
    - No control flow depends on the tile contents
    - Non-mutating: returns new dict, does not modify input
    
    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        governance_tile: Atlas governance tile from build_atlas_governance_tile().
        atlas_signal: Optional atlas signal from extract_atlas_signal_for_first_light().
        structure_tile: Optional Lean shadow structure tile for structural cohesion annex.
        coherence_tile: Optional coherence governance tile for structural cohesion annex.
    
    Returns:
        New dict with evidence contents plus atlas governance tile attached under governance key.
        If structure_tile or coherence_tile provided, also includes first_light_structural_annex.
    
    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> tile = build_atlas_governance_tile(lattice, phase_gate, director_tile_v2)
        >>> signal = extract_atlas_signal_for_first_light(lattice, phase_gate)
        >>> enriched = attach_atlas_governance_to_evidence(evidence, tile, signal)
        >>> "governance" in enriched
        True
        >>> "atlas" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Build atlas governance data for evidence
    atlas_data = {
        "lattice_coherence_band": governance_tile.get("lattice_coherence_band", "PARTIAL"),
        "global_lattice_norm": governance_tile.get("global_lattice_norm", 0.0),
        "transition_status": governance_tile.get("transition_status", "ATTENTION"),
        "slices_ready": governance_tile.get("slices_ready", []),
        "slices_needing_alignment": governance_tile.get("slices_needing_alignment", []),
    }
    
    # If signal provided, prefer its values for core metrics
    if atlas_signal is not None:
        atlas_data["global_lattice_norm"] = atlas_signal.get("global_lattice_norm", atlas_data["global_lattice_norm"])
        atlas_data["lattice_convergence_band"] = atlas_signal.get("lattice_convergence_band", atlas_data["lattice_coherence_band"])
        atlas_data["transition_status"] = atlas_signal.get("transition_status", atlas_data["transition_status"])
    
    # Attach atlas governance tile
    enriched["governance"] = enriched["governance"].copy()
    enriched["governance"]["atlas"] = atlas_data
    
    # Optionally attach Structural Cohesion Annex if structure or coherence tiles provided
    if structure_tile is not None or coherence_tile is not None:
        annex = build_first_light_structural_cohesion_annex(
            atlas_tile=governance_tile,
            structure_tile=structure_tile,
            coherence_tile=coherence_tile,
        )
        enriched["governance"]["atlas"]["first_light_structural_annex"] = annex
    
    return enriched


def build_first_light_structural_cohesion_annex(
    atlas_tile: Dict[str, Any],
    structure_tile: Optional[Dict[str, Any]] = None,
    coherence_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build First Light Structural Cohesion Annex combining atlas, structure, and coherence.
    
    STATUS: PHASE X — STRUCTURAL COHESION ANNEX
    
    This annex provides a unified view of structural cohesion across:
    - Atlas governance (lattice convergence)
    - Lean shadow structure (structural integrity)
    - Coherence (topological alignment)
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned annex is purely observational
    - No control flow depends on the annex contents
    - No gate semantics or enforcement
    
    Args:
        atlas_tile: Atlas governance tile from build_atlas_governance_tile.
            Must contain: lattice_coherence_band, transition_status
        structure_tile: Optional Lean shadow structure tile.
            May contain: status (from build_lean_shadow_tile_for_global_health)
        coherence_tile: Optional coherence governance tile.
            May contain: coherence_band (from build_coherence_governance_tile)
    
    Returns:
        Structural cohesion annex dictionary with:
        - schema_version: "1.0.0"
        - lattice_band: str (COHERENT|PARTIAL|MISALIGNED)
        - transition_status: str (OK|ATTENTION|BLOCK)
        - lean_shadow_status: Optional[str] (OK|WARN|BLOCK)
        - coherence_band: Optional[str] (COHERENT|PARTIAL|MISALIGNED)
    """
    # Extract atlas fields (required)
    lattice_band = atlas_tile.get("lattice_coherence_band", "PARTIAL")
    transition_status = atlas_tile.get("transition_status", "ATTENTION")
    
    # Extract structure tile status if available
    lean_shadow_status = None
    if structure_tile is not None:
        # Lean shadow tile uses "status" field
        lean_shadow_status = structure_tile.get("status")
    
    # Extract coherence band if available
    coherence_band = None
    if coherence_tile is not None:
        coherence_band = coherence_tile.get("coherence_band")
    
    # Build annex
    annex = {
        "schema_version": "1.0.0",
        "lattice_band": lattice_band,
        "transition_status": transition_status,
        "lean_shadow_status": lean_shadow_status,
        "coherence_band": coherence_band,
    }
    
    return annex


def summarize_atlas_for_uplift_council(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize atlas governance for uplift council decision-making.
    
    Maps atlas governance status to council signals:
    - MISALIGNED or transition_status="BLOCK" → BLOCK
    - PARTIAL or transition_status="ATTENTION" → WARN
    - COHERENT + transition_status="OK" → OK
    
    SHADOW MODE CONTRACT:
    - This is advisory only
    - No hard gates or enforcement
    - Purely observational
    
    STATUS: PHASE X — UPLIFT COUNCIL HOOK
    
    Args:
        tile: Atlas governance tile from build_atlas_governance_tile
        
    Returns:
        Council summary with:
        - status: "OK" | "WARN" | "BLOCK"
        - lattice_coherence_band: "COHERENT" | "PARTIAL" | "MISALIGNED"
        - transition_status: "OK" | "ATTENTION" | "BLOCK"
        - slices_needing_alignment: List[str]
        - advisory_notes: List[str]
    """
    lattice_coherence_band = tile.get("lattice_coherence_band", "PARTIAL")
    transition_status = tile.get("transition_status", "ATTENTION")
    slices_needing_alignment = tile.get("slices_needing_alignment", [])
    
    # Map to council status
    # BLOCK: MISALIGNED or transition_status="BLOCK"
    # WARN: PARTIAL or transition_status="ATTENTION"
    # OK: COHERENT + transition_status="OK"
    if lattice_coherence_band == "MISALIGNED" or transition_status == "BLOCK":
        status = "BLOCK"
    elif lattice_coherence_band == "PARTIAL" or transition_status == "ATTENTION":
        status = "WARN"
    else:  # COHERENT + OK
        status = "OK"
    
    # Build advisory notes
    advisory_notes = [
        f"Lattice coherence band: {lattice_coherence_band}.",
        f"Transition status: {transition_status}.",
        f"Council status: {status}.",
    ]
    
    if len(slices_needing_alignment) > 0:
        advisory_notes.append(
            f"{len(slices_needing_alignment)} slice{'s' if len(slices_needing_alignment) != 1 else ''} "
            f"need alignment: {', '.join(slices_needing_alignment[:5])}"
            + ("..." if len(slices_needing_alignment) > 5 else "")
        )
    else:
        advisory_notes.append("No slices need alignment.")
    
    return {
        "status": status,
        "lattice_coherence_band": lattice_coherence_band,
        "transition_status": transition_status,
        "slices_needing_alignment": slices_needing_alignment,
        "advisory_notes": advisory_notes,
    }


def emit_cal_exp_structural_cohesion_annex(
    cal_id: str,
    annex: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Emit and persist a calibration experiment structural cohesion annex.
    
    STATUS: PHASE X — CAL-EXP STRUCTURAL COHESION REGISTER
    
    This function creates a per-experiment annex document and persists it to disk.
    The annex captures structural cohesion state for a single calibration experiment.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from file I/O)
    - The emitted annex is purely observational
    - No control flow depends on the annex contents
    - File I/O is for persistence only, not enforcement
    
    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2")
        annex: Structural cohesion annex from build_first_light_structural_cohesion_annex()
        output_dir: Optional output directory. Defaults to "calibration/" if not provided.
    
    Returns:
        Emitted annex document with:
        - schema_version: "1.0.0"
        - cal_id: str
        - lattice_band: str (COHERENT|PARTIAL|MISALIGNED)
        - transition_status: str (OK|ATTENTION|BLOCK)
        - lean_shadow_status: Optional[str] (OK|WARN|BLOCK)
        - coherence_band: Optional[str] (COHERENT|PARTIAL|MISALIGNED)
    
    Side effects:
        Writes annex to: {output_dir}/structural_cohesion_annex_{cal_id}.json
    """
    # Determine output directory
    if output_dir is None:
        output_dir = "calibration"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build emitted annex document
    emitted_annex = {
        "schema_version": STRUCTURAL_COHESION_ANNEX_SCHEMA_VERSION,
        "cal_id": cal_id,
        "lattice_band": annex.get("lattice_band", "PARTIAL"),
        "transition_status": annex.get("transition_status", "ATTENTION"),
        "lean_shadow_status": annex.get("lean_shadow_status"),
        "coherence_band": annex.get("coherence_band"),
    }
    
    # Persist to disk
    filename = f"structural_cohesion_annex_{cal_id}.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(emitted_annex, f, indent=2, sort_keys=True)
    
    return emitted_annex


def build_structural_cohesion_register(
    annexes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build structural cohesion register from multiple calibration experiment annexes.
    
    STATUS: PHASE X — CAL-EXP STRUCTURAL COHESION REGISTER
    
    This function aggregates structural cohesion data across multiple calibration
    experiments to identify patterns and misalignments.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned register is purely observational
    - No control flow depends on the register contents
    - No gate semantics or enforcement
    
    Args:
        annexes: List of emitted annex documents from emit_cal_exp_structural_cohesion_annex().
            Each annex must contain: cal_id, lattice_band, transition_status,
            lean_shadow_status (optional), coherence_band (optional)
    
    Returns:
        Structural cohesion register dictionary with:
        - schema_version: "1.0.0"
        - total_experiments: int
        - band_combinations: Dict[str, int] (counts of lattice_band × coherence_band combinations)
        - experiments_with_misaligned_structure: List[str] (cal_ids with any misalignment)
        - top_misaligned: List[str] (up to 5 cal_ids ranked by severity: lattice MISALIGNED, then lean shadow BLOCK, then coherence MISALIGNED, then cal_id)
        - lattice_band_distribution: Dict[str, int]
        - transition_status_distribution: Dict[str, int]
        - lean_shadow_status_distribution: Dict[str, int] (only for experiments with lean shadow)
        - coherence_band_distribution: Dict[str, int] (only for experiments with coherence)
    """
    if not annexes:
        return {
            "schema_version": STRUCTURAL_COHESION_REGISTER_SCHEMA_VERSION,
            "total_experiments": 0,
            "band_combinations": {},
            "experiments_with_misaligned_structure": [],
            "top_misaligned": [],
            "lattice_band_distribution": {},
            "transition_status_distribution": {},
            "lean_shadow_status_distribution": {},
            "coherence_band_distribution": {},
        }
    
    # Initialize counters
    band_combinations: Dict[str, int] = {}
    lattice_band_counts: Dict[str, int] = {}
    transition_status_counts: Dict[str, int] = {}
    lean_shadow_status_counts: Dict[str, int] = {}
    coherence_band_counts: Dict[str, int] = {}
    misaligned_experiments: List[str] = []
    
    # Process each annex
    for annex in annexes:
        cal_id = annex.get("cal_id", "unknown")
        lattice_band = annex.get("lattice_band", "PARTIAL")
        transition_status = annex.get("transition_status", "ATTENTION")
        lean_shadow_status = annex.get("lean_shadow_status")
        coherence_band = annex.get("coherence_band")
        
        # Count lattice bands
        lattice_band_counts[lattice_band] = lattice_band_counts.get(lattice_band, 0) + 1
        
        # Count transition status
        transition_status_counts[transition_status] = (
            transition_status_counts.get(transition_status, 0) + 1
        )
        
        # Count lean shadow status if available
        if lean_shadow_status is not None:
            lean_shadow_status_counts[lean_shadow_status] = (
                lean_shadow_status_counts.get(lean_shadow_status, 0) + 1
            )
        
        # Count coherence bands if available
        if coherence_band is not None:
            coherence_band_counts[coherence_band] = (
                coherence_band_counts.get(coherence_band, 0) + 1
            )
            
            # Count band combinations (lattice × coherence)
            combination_key = f"{lattice_band}×{coherence_band}"
            band_combinations[combination_key] = (
                band_combinations.get(combination_key, 0) + 1
            )
        
        # Identify misaligned experiments
        # MISALIGNED if any of:
        # - lattice_band == "MISALIGNED"
        # - lean_shadow_status == "BLOCK"
        # - coherence_band == "MISALIGNED"
        is_misaligned = (
            lattice_band == "MISALIGNED"
            or lean_shadow_status == "BLOCK"
            or coherence_band == "MISALIGNED"
        )
        
        if is_misaligned:
            misaligned_experiments.append(cal_id)
    
    # Sort misaligned experiments for determinism
    misaligned_experiments = sorted(misaligned_experiments)
    
    # Compute top misalignments (up to 5, ranked by severity)
    # Ranking priority:
    # 1. lattice_band == "MISALIGNED" (highest)
    # 2. lean_shadow_status == "BLOCK"
    # 3. coherence_band == "MISALIGNED"
    # 4. cal_id (for determinism)
    top_misaligned = []
    
    # Build list of misaligned experiments with ranking metadata
    misaligned_with_rank = []
    for annex in annexes:
        cal_id = annex.get("cal_id", "unknown")
        if cal_id not in misaligned_experiments:
            continue
        
        lattice_band = annex.get("lattice_band", "PARTIAL")
        lean_shadow_status = annex.get("lean_shadow_status")
        coherence_band = annex.get("coherence_band")
        
        # Compute rank (lower number = higher priority)
        rank = 0
        if lattice_band == "MISALIGNED":
            rank = 1  # Highest priority
        elif lean_shadow_status == "BLOCK":
            rank = 2
        elif coherence_band == "MISALIGNED":
            rank = 3
        else:
            rank = 4  # Should not happen if cal_id is in misaligned_experiments
        
        misaligned_with_rank.append((rank, cal_id, annex))
    
    # Sort by rank, then by cal_id for determinism
    misaligned_with_rank.sort(key=lambda x: (x[0], x[1]))
    
    # Take top 5
    top_misaligned = [cal_id for _, cal_id, _ in misaligned_with_rank[:5]]
    
    # Build register
    register = {
        "schema_version": STRUCTURAL_COHESION_REGISTER_SCHEMA_VERSION,
        "total_experiments": len(annexes),
        "band_combinations": dict(sorted(band_combinations.items())),
        "experiments_with_misaligned_structure": misaligned_experiments,
        "top_misaligned": top_misaligned,
        "lattice_band_distribution": dict(sorted(lattice_band_counts.items())),
        "transition_status_distribution": dict(sorted(transition_status_counts.items())),
        "lean_shadow_status_distribution": dict(sorted(lean_shadow_status_counts.items())),
        "coherence_band_distribution": dict(sorted(coherence_band_counts.items())),
    }
    
    return register


def attach_structural_cohesion_register_to_evidence(
    evidence: Dict[str, Any],
    register: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach structural cohesion register to an evidence pack (read-only, additive).
    
    STATUS: PHASE X — CAL-EXP STRUCTURAL COHESION REGISTER
    
    This is a lightweight helper for evidence chain builders. It does not
    modify the input evidence dict, but returns a new dict with the register attached
    under evidence["governance"]["structural_cohesion_register"].
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached register is purely observational
    - No control flow depends on the register contents
    - Non-mutating: returns new dict, does not modify input
    
    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
        register: Structural cohesion register from build_structural_cohesion_register().
    
    Returns:
        New dict with evidence contents plus structural cohesion register attached under governance key.
    
    Example:
        >>> evidence = {"timestamp": "2024-01-01", "data": {...}}
        >>> register = build_structural_cohesion_register(annexes)
        >>> enriched = attach_structural_cohesion_register_to_evidence(evidence, register)
        >>> "governance" in enriched
        True
        >>> "structural_cohesion_register" in enriched["governance"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Ensure governance key exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    
    # Attach structural cohesion register
    enriched["governance"] = enriched["governance"].copy()
    enriched["governance"]["structural_cohesion_register"] = register
    
    return enriched


def extract_structural_cohesion_register_signal(
    evidence: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract structural cohesion register signal from evidence pack.
    
    STATUS: PHASE X — CAL-EXP STRUCTURAL COHESION REGISTER STATUS HOOK
    
    This function extracts a compact status signal from the structural cohesion register
    if present in the evidence pack. The signal provides a quick view of structural
    cohesion status without requiring full register inspection.
    
    SHADOW MODE CONTRACT:
    - This function is read-only
    - The extracted signal is purely observational
    - No control flow depends on the signal contents
    - Returns None if register not present (graceful degradation)
    
    Args:
        evidence: Evidence pack dictionary. May contain:
            evidence["governance"]["structural_cohesion_register"]
    
    Returns:
        Status signal dictionary with:
        - total_experiments: int
        - experiments_with_misaligned_structure_count: int
        - top_misaligned: List[str] (up to 5 cal_ids)
        Or None if register not present in evidence.
    
    Example:
        >>> evidence = {
        ...     "governance": {
        ...         "structural_cohesion_register": {
        ...             "total_experiments": 3,
        ...             "experiments_with_misaligned_structure": ["CAL-EXP-2"],
        ...             "top_misaligned": ["CAL-EXP-2"],
        ...         }
        ...     }
        ... }
        >>> signal = extract_structural_cohesion_register_signal(evidence)
        >>> signal["total_experiments"]
        3
    """
    # Check if register exists in evidence
    if "governance" not in evidence:
        return None
    
    governance = evidence.get("governance", {})
    register = governance.get("structural_cohesion_register")
    
    if register is None:
        return None
    
    # Extract signal
    signal = {
        "total_experiments": register.get("total_experiments", 0),
        "experiments_with_misaligned_structure_count": len(
            register.get("experiments_with_misaligned_structure", [])
        ),
        "top_misaligned": register.get("top_misaligned", []),
    }
    
    return signal


def attach_structural_cohesion_register_signal_to_evidence(
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach structural cohesion register signal to evidence pack signals section.
    
    STATUS: PHASE X — CAL-EXP STRUCTURAL COHESION REGISTER STATUS HOOK
    
    This function extracts the structural cohesion register signal and attaches it
    to evidence["signals"]["structural_cohesion_register"] for quick status access.
    
    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attached signal is purely observational
    - No control flow depends on the signal contents
    - Non-mutating: returns new dict, does not modify input
    - Gracefully handles missing register (no signal attached)
    
    Args:
        evidence: Existing evidence pack dict (read-only, not modified).
            May contain evidence["governance"]["structural_cohesion_register"].
    
    Returns:
        New dict with evidence contents plus structural cohesion register signal
        attached under signals key (if register present in evidence).
    
    Example:
        >>> evidence = {
        ...     "governance": {
        ...         "structural_cohesion_register": {
        ...             "total_experiments": 3,
        ...             "top_misaligned": ["CAL-EXP-2"],
        ...         }
        ...     }
        ... }
        >>> enriched = attach_structural_cohesion_register_signal_to_evidence(evidence)
        >>> "signals" in enriched
        True
        >>> "structural_cohesion_register" in enriched["signals"]
        True
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()
    
    # Extract signal from register if present
    signal = extract_structural_cohesion_register_signal(evidence)
    
    if signal is not None:
        # Ensure signals key exists
        if "signals" not in enriched:
            enriched["signals"] = {}
        else:
            enriched["signals"] = dict(enriched["signals"])
        
        # Attach signal
        enriched["signals"]["structural_cohesion_register"] = signal
    
    return enriched


def structural_cohesion_register_for_alignment_view(
    signal_or_register: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert structural cohesion register signal/register to GGFL alignment view format.
    
    STATUS: PHASE X — GGFL ADAPTER FOR STRUCTURAL COHESION REGISTER
    
    Normalizes structural cohesion register into Global Governance Fusion Layer
    (GGFL) unified format for cross-subsystem alignment views.
    
    SHADOW MODE CONTRACT:
    - This function is read-only
    - The output is purely observational
    - No control flow depends on the output contents
    - Advisory only; no enforcement
    
    Args:
        signal_or_register: Either:
            - Status signal from first_light_status.json signals["structural_cohesion_register"]
            - Full register from build_structural_cohesion_register()
    
    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-STR" (constant)
        - status: "ok" | "warn" (warn if misaligned_count > 0)
        - conflict: False (constant, structural cohesion is advisory)
        - weight_hint: "LOW" (constant, structural cohesion is low-weight advisory)
        - drivers: List[str] (max 1: DRIVER_MISALIGNED_EXPERIMENTS_PRESENT if misalignments exist)
        - summary: str (single neutral sentence)
    
    Example:
        >>> signal = {
        ...     "total_experiments": 3,
        ...     "experiments_with_misaligned_structure_count": 2,
        ...     "top_misaligned": ["CAL-EXP-2", "CAL-EXP-3"],
        ... }
        >>> view = structural_cohesion_register_for_alignment_view(signal)
        >>> view["signal_type"]
        'SIG-STR'
        >>> view["status"]
        'warn'
        >>> view["conflict"]
        False
    """
    # Extract fields (handle both signal and register formats)
    total_experiments = signal_or_register.get("total_experiments", 0)
    misaligned_count = signal_or_register.get("experiments_with_misaligned_structure_count", 0)
    
    # If register format, compute count from list
    if misaligned_count == 0:
        misaligned_experiments = signal_or_register.get("experiments_with_misaligned_structure", [])
        if misaligned_experiments:
            misaligned_count = len(misaligned_experiments)
    
    # Determine status: warn if any misalignments, otherwise ok
    status = "warn" if misaligned_count > 0 else "ok"
    
    # Build drivers list (max 1: DRIVER_MISALIGNED_EXPERIMENTS_PRESENT)
    drivers: List[str] = []
    if misaligned_count > 0:
        drivers.append("DRIVER_MISALIGNED_EXPERIMENTS_PRESENT")
    
    # Build neutral summary sentence
    if misaligned_count > 0:
        summary = (
            f"Structural cohesion register: {misaligned_count} out of {total_experiments} "
            f"calibration experiment(s) show misaligned structure across lattice, "
            f"lean shadow, or coherence dimensions."
        )
    else:
        summary = (
            f"Structural cohesion register: {total_experiments} calibration experiment(s) "
            f"show aligned structure across lattice, lean shadow, and coherence dimensions."
        )
    
    return {
        "signal_type": "SIG-STR",
        "status": status,
        "conflict": False,  # Structural cohesion is advisory only
        "weight_hint": "LOW",  # Low-weight advisory signal
        "drivers": drivers,
        "summary": summary,
        "shadow_mode_invariants": {
            "advisory_only": True,  # Structural cohesion is purely advisory
            "no_enforcement": True,  # No enforcement or gating behavior
            "conflict_invariant": True,  # Conflict must always be False
        },
    }


def summarize_structural_cohesion_register_signal_consistency(
    status_signal: Dict[str, Any],
    ggfl_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Cross-check consistency between status signal and GGFL signal for structural cohesion register.
    
    STATUS: PHASE X — STRUCTURAL COHESION REGISTER CONSISTENCY CHECK
    
    This function validates that the status signal and GGFL signal are consistent
    with each other and that the conflict invariant is maintained (conflict must
    always be False).
    
    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Detects inconsistencies for advisory purposes only
    - No gating, just advisory notes
    
    Args:
        status_signal: Structural cohesion register signal from status JSON (from generate_first_light_status).
            Expected keys: total_experiments, experiments_with_misaligned_structure_count, top_misaligned
        ggfl_signal: Structural cohesion register signal from GGFL adapter (from structural_cohesion_register_for_alignment_view).
            Expected keys: signal_type, status, conflict, weight_hint, drivers, summary
    
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - mode: "SHADOW"
        - consistency: "CONSISTENT" | "PARTIAL" | "INCONSISTENT"
        - notes: List[str] (neutral descriptive notes about inconsistencies)
        - conflict_invariant_violated: bool (True if conflict is ever True)
        - top_mismatch_type: Optional[str] (Top mismatch type for INCONSISTENT cases)
    """
    notes: List[str] = []
    consistency = "CONSISTENT"
    conflict_invariant_violated = False
    top_mismatch_type: Optional[str] = None
    
    # Extract misaligned count from status signal
    status_misaligned_count = status_signal.get("experiments_with_misaligned_structure_count", 0)
    
    # Derive expected status from misaligned count
    # status should be "warn" if misaligned_count > 0, "ok" otherwise
    expected_status = "warn" if status_misaligned_count > 0 else "ok"
    ggfl_status = ggfl_signal.get("status", "").lower()  # "ok" | "warn"
    
    # Check status consistency
    status_mismatch = False
    if expected_status != ggfl_status:
        notes.append(
            f"Status mismatch: status signal indicates {status_misaligned_count} misaligned experiment(s) "
            f"(expected status='{expected_status}') but GGFL signal says '{ggfl_status}'"
        )
        status_mismatch = True
        consistency = "PARTIAL"
        if top_mismatch_type is None:
            top_mismatch_type = "status_mismatch"
    
    # Check conflict invariant (MUST always be False)
    # This is the only condition that causes INCONSISTENT
    ggfl_conflict = ggfl_signal.get("conflict", False)
    if ggfl_conflict is True:
        notes.append(
            "CRITICAL: Conflict invariant violated - GGFL signal has conflict=True. "
            "Structural cohesion register must never trigger conflict (conflict must always be False)."
        )
        conflict_invariant_violated = True
        consistency = "INCONSISTENT"
        top_mismatch_type = "conflict_invariant_violated"
    
    # Check signal_type
    ggfl_signal_type = ggfl_signal.get("signal_type", "")
    if ggfl_signal_type != "SIG-STR":
        notes.append(
            f"Signal type mismatch: GGFL signal_type is '{ggfl_signal_type}' but expected 'SIG-STR'"
        )
        if consistency == "CONSISTENT":
            consistency = "PARTIAL"
        if top_mismatch_type is None:
            top_mismatch_type = "signal_type_mismatch"
    
    # Check weight_hint (should always be "LOW")
    ggfl_weight_hint = ggfl_signal.get("weight_hint", "")
    if ggfl_weight_hint != "LOW":
        notes.append(
            f"Weight hint mismatch: GGFL weight_hint is '{ggfl_weight_hint}' but expected 'LOW'"
        )
        if consistency == "CONSISTENT":
            consistency = "PARTIAL"
        if top_mismatch_type is None:
            top_mismatch_type = "weight_hint_mismatch"
    
    # If no issues found, return consistent
    if not notes:
        notes.append("Status signal and GGFL signal are consistent")
    
    return {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "consistency": consistency,
        "notes": notes,
        "conflict_invariant_violated": conflict_invariant_violated,
        "top_mismatch_type": top_mismatch_type,
    }


__all__ = [
    "ATLAS_GOVERNANCE_TILE_SCHEMA_VERSION",
    "attach_atlas_governance_to_evidence",
    "attach_atlas_governance_to_p3_stability_report",
    "attach_atlas_governance_to_p4_calibration_report",
    "attach_structural_cohesion_register_to_evidence",
    "attach_structural_cohesion_register_signal_to_evidence",
    "build_atlas_governance_tile",
    "build_first_light_structural_cohesion_annex",
    "build_structural_cohesion_register",
    "emit_cal_exp_structural_cohesion_annex",
    "extract_atlas_signal_for_first_light",
    "extract_structural_cohesion_register_signal",
    "structural_cohesion_register_for_alignment_view",
    "summarize_atlas_for_uplift_council",
    "summarize_structural_cohesion_register_signal_consistency",
]

