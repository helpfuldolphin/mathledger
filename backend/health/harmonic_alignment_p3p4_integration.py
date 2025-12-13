"""Harmonic alignment P3/P4 integration for curriculum annex.

STATUS: PHASE X — HARMONIC CURRICULUM ANNEX

Provides curriculum-centric harmonic alignment annex that summarizes misaligned
concepts and evolution status from P3 and P4 data.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Harmonic alignment signals are purely observational
- They do NOT influence any other signals or system health classification
- No control flow depends on harmonic values
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRICULUM_HARMONIC_ANNEX_SCHEMA_VERSION = "1.0.0"
CURRICULUM_HARMONIC_GRID_SCHEMA_VERSION = "1.0.0"
CURRICULUM_HARMONIC_DELTA_SCHEMA_VERSION = "1.0.0"


def build_curriculum_harmonic_annex(
    p3_summary: Dict[str, Any],
    p4_calibration: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build curriculum harmonic annex from P3 summary and P4 calibration.

    STATUS: PHASE X — HARMONIC CURRICULUM ANNEX

    Combines harmonic alignment data from P3 First-Light summary and P4 calibration
    report to provide a curriculum-centric view of harmonic alignment.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned annex is purely observational
    - No control flow depends on the annex contents

    Args:
        p3_summary: P3 First-Light summary with harmonic_alignment_summary field.
            Expected to have: harmonic_band, misaligned_concepts, priority_adjustments
        p4_calibration: P4 calibration report with harmonic_alignment field.
            Expected to have: evolution_status, misaligned_concepts

    Returns:
        Curriculum harmonic annex dictionary with:
        - schema_version: "1.0.0"
        - harmonic_band: "COHERENT" | "PARTIAL" | "MISMATCHED" (from p3_summary)
        - evolution_status: "STABLE" | "EVOLVING" | "DIVERGING" (from p4_calibration)
        - misaligned_concepts: List[str] (sorted, deduplicated, max 10)
        - priority_adjustments: List[str] (from p3_summary, max 5)
    """
    # Extract harmonic band from P3 summary
    harmonic_band = p3_summary.get("harmonic_band", "PARTIAL")

    # Extract evolution status from P4 calibration
    evolution_status = p4_calibration.get("evolution_status", "STABLE")

    # Combine and deduplicate misaligned concepts (max 10)
    p3_misaligned = p3_summary.get("misaligned_concepts", [])
    p4_misaligned = p4_calibration.get("misaligned_concepts", [])
    all_misaligned = sorted(set(p3_misaligned + p4_misaligned))[:10]

    # Extract priority adjustments from P3 summary (max 5)
    priority_adjustments = p3_summary.get("priority_adjustments", [])[:5]

    return {
        "schema_version": CURRICULUM_HARMONIC_ANNEX_SCHEMA_VERSION,
        "harmonic_band": harmonic_band,
        "evolution_status": evolution_status,
        "misaligned_concepts": all_misaligned,
        "priority_adjustments": priority_adjustments,
    }


def emit_cal_exp_curriculum_harmonic_annex(
    cal_id: str,
    annex: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Emit curriculum harmonic annex snapshot for a calibration experiment.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Creates a per-experiment snapshot of the curriculum harmonic annex suitable
    for aggregation across CAL-EXP-1/2/3 runs.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned snapshot is purely observational
    - No control flow depends on the snapshot contents

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2")
        annex: Curriculum harmonic annex from build_curriculum_harmonic_annex()

    Returns:
        Snapshot dictionary with:
        - schema_version: "1.0.0"
        - cal_id: str (experiment identifier)
        - harmonic_band: "COHERENT" | "PARTIAL" | "MISMATCHED"
        - evolution_status: "STABLE" | "EVOLVING" | "DIVERGING"
        - misaligned_concepts: List[str]
        - priority_adjustments: List[str]
    """
    return {
        "schema_version": CURRICULUM_HARMONIC_ANNEX_SCHEMA_VERSION,
        "cal_id": cal_id,
        "harmonic_band": annex.get("harmonic_band", "PARTIAL"),
        "evolution_status": annex.get("evolution_status", "STABLE"),
        "misaligned_concepts": sorted(annex.get("misaligned_concepts", [])),
        "priority_adjustments": annex.get("priority_adjustments", []),
    }


def persist_curriculum_harmonic_annex(
    snapshot: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Persist curriculum harmonic annex snapshot to disk.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Writes a snapshot to calibration/curriculum_harmonic_annex_<cal_id>.json.
    Creates the output directory if it doesn't exist.

    SHADOW MODE CONTRACT:
    - This is a write-only operation for recording snapshots
    - It does not affect any governance state

    Args:
        snapshot: Snapshot dictionary from emit_cal_exp_curriculum_harmonic_annex()
        output_dir: Base directory for calibration artifacts (e.g., Path("calibration"))

    Returns:
        Path to the written snapshot file
    """
    cal_id = snapshot.get("cal_id", "UNKNOWN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = output_dir / f"curriculum_harmonic_annex_{cal_id}.json"

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)

    return snapshot_path


def build_curriculum_harmonic_grid(
    annexes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build curriculum harmonic grid from multiple calibration experiment annexes.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Aggregates curriculum harmonic annexes across multiple calibration experiments
    (CAL-EXP-1/2/3) into a single grid for cross-experiment alignment witness.

    SHADOW MODE CONTRACT:
    - This is evidence-only, not a gate
    - The grid is observational and does not influence calibration experiment behavior
    - Provides alignment witness over time, not a control signal

    Args:
        annexes: List of annex snapshots from emit_cal_exp_curriculum_harmonic_annex().
            Each annex must contain: cal_id, harmonic_band, misaligned_concepts

    Returns:
        Grid dictionary with:
        - schema_version: "1.0.0"
        - num_experiments: int (number of annexes)
        - harmonic_band_counts: Dict[str, int] (COHERENT/PARTIAL/MISMATCHED counts)
        - top_misaligned_concepts: List[Dict[str, Any]] (concepts with frequency, sorted by frequency desc, limited)
            Each entry: {"concept": str, "frequency": int, "experiments": List[str]}
    """
    if not annexes:
        return {
            "schema_version": CURRICULUM_HARMONIC_GRID_SCHEMA_VERSION,
            "num_experiments": 0,
            "harmonic_band_counts": {
                "COHERENT": 0,
                "PARTIAL": 0,
                "MISMATCHED": 0,
            },
            "top_misaligned_concepts": [],
            "top_driver_concepts": [],
            "top_driver_cal_ids": {},
        }

    # Count harmonic bands
    band_counter = Counter(annex.get("harmonic_band", "PARTIAL") for annex in annexes)
    harmonic_band_counts = {
        "COHERENT": band_counter.get("COHERENT", 0),
        "PARTIAL": band_counter.get("PARTIAL", 0),
        "MISMATCHED": band_counter.get("MISMATCHED", 0),
    }

    # Count concept frequencies across experiments
    concept_frequency: Dict[str, Dict[str, Any]] = {}
    for annex in annexes:
        cal_id = annex.get("cal_id", "UNKNOWN")
        misaligned_concepts = annex.get("misaligned_concepts", [])
        for concept in misaligned_concepts:
            if concept not in concept_frequency:
                concept_frequency[concept] = {
                    "concept": concept,
                    "frequency": 0,
                    "experiments": [],
                }
            concept_frequency[concept]["frequency"] += 1
            if cal_id not in concept_frequency[concept]["experiments"]:
                concept_frequency[concept]["experiments"].append(cal_id)

    # Sort experiments for determinism
    for concept_data in concept_frequency.values():
        concept_data["experiments"] = sorted(concept_data["experiments"])

    # Sort by frequency (descending), then by concept name (ascending) for determinism
    top_concepts = sorted(
        concept_frequency.values(),
        key=lambda x: (-x["frequency"], x["concept"]),
    )[:10]  # Limit to top 10

    # Extract top 5 driver concepts (same sorting logic)
    top_driver_concepts = sorted(
        concept_frequency.values(),
        key=lambda x: (-x["frequency"], x["concept"]),
    )[:5]  # Limit to top 5

    # Build top_driver_cal_ids mapping
    top_driver_cal_ids: Dict[str, List[str]] = {}
    for driver in top_driver_concepts:
        concept_name = driver["concept"]
        top_driver_cal_ids[concept_name] = sorted(driver["experiments"])

    return {
        "schema_version": CURRICULUM_HARMONIC_GRID_SCHEMA_VERSION,
        "num_experiments": len(annexes),
        "harmonic_band_counts": harmonic_band_counts,
        "top_misaligned_concepts": top_concepts,
        "top_driver_concepts": [d["concept"] for d in top_driver_concepts],
        "top_driver_cal_ids": top_driver_cal_ids,
    }


def attach_curriculum_harmonic_grid_to_evidence(
    evidence: Dict[str, Any],
    grid: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach curriculum harmonic grid to evidence pack.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Stores curriculum harmonic grid under evidence["governance"]["harmonic_curriculum_panel"]
    for inclusion in evidence packs.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attachment is purely observational
    - No control flow depends on the attached data
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Evidence pack dictionary (will be copied, not modified in-place)
        grid: Curriculum harmonic grid from build_curriculum_harmonic_grid()

    Returns:
        New dict with evidence contents plus harmonic_curriculum_panel attached under governance key
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Ensure governance structure exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    else:
        enriched["governance"] = dict(enriched["governance"])

    # Attach harmonic curriculum panel
    enriched["governance"]["harmonic_curriculum_panel"] = grid

    return enriched


def build_curriculum_harmonic_delta(
    mock_grid: Dict[str, Any],
    real_grid: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build curriculum harmonic delta comparing mock vs real grids.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Compares top_driver_concepts overlap and frequency shifts between mock and real
    calibration experiment grids. This provides observational data for calibration
    analysis but does not gate any operations.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned delta is purely observational
    - No control flow depends on the delta contents
    - Provides alignment witness for mock-vs-real comparison

    Args:
        mock_grid: Curriculum harmonic grid from mock calibration experiments
        real_grid: Curriculum harmonic grid from real calibration experiments

    Returns:
        Delta dictionary with:
        - schema_version: "1.0.0"
        - top_driver_overlap: List[str] (concepts appearing in both grids, sorted)
        - top_driver_only_mock: List[str] (concepts only in mock, sorted)
        - top_driver_only_real: List[str] (concepts only in real, sorted)
        - frequency_shifts: List[Dict[str, Any]] (concepts with frequency changes)
            Each entry: {"concept": str, "mock_frequency": int, "real_frequency": int, "delta": int}
            Sorted by absolute delta (descending), then concept name (ascending)
    """
    mock_drivers = set(mock_grid.get("top_driver_concepts", []))
    real_drivers = set(real_grid.get("top_driver_concepts", []))

    # Find overlap and differences
    top_driver_overlap = sorted(mock_drivers & real_drivers)
    top_driver_only_mock = sorted(mock_drivers - real_drivers)
    top_driver_only_real = sorted(real_drivers - mock_drivers)

    # Build frequency maps from top_misaligned_concepts
    mock_frequencies: Dict[str, int] = {}
    for concept_data in mock_grid.get("top_misaligned_concepts", []):
        concept = concept_data.get("concept", "")
        if concept:
            mock_frequencies[concept] = concept_data.get("frequency", 0)

    real_frequencies: Dict[str, int] = {}
    for concept_data in real_grid.get("top_misaligned_concepts", []):
        concept = concept_data.get("concept", "")
        if concept:
            real_frequencies[concept] = concept_data.get("frequency", 0)

    # Compute frequency shifts for all concepts that appear in either grid
    all_concepts = set(mock_frequencies.keys()) | set(real_frequencies.keys())
    frequency_shifts = []
    for concept in all_concepts:
        mock_freq = mock_frequencies.get(concept, 0)
        real_freq = real_frequencies.get(concept, 0)
        delta = real_freq - mock_freq
        # Only include concepts with non-zero frequency in at least one grid
        if mock_freq > 0 or real_freq > 0:
            frequency_shifts.append({
                "concept": concept,
                "mock_frequency": mock_freq,
                "real_frequency": real_freq,
                "delta": delta,
            })

    # Sort by absolute delta (descending), then concept name (ascending) for determinism
    frequency_shifts = sorted(
        frequency_shifts,
        key=lambda x: (-abs(x["delta"]), x["concept"]),
    )

    return {
        "schema_version": CURRICULUM_HARMONIC_DELTA_SCHEMA_VERSION,
        "top_driver_overlap": top_driver_overlap,
        "top_driver_only_mock": top_driver_only_mock,
        "top_driver_only_real": top_driver_only_real,
        "frequency_shifts": frequency_shifts,
    }


def attach_curriculum_harmonic_delta_to_evidence(
    evidence: Dict[str, Any],
    delta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach curriculum harmonic delta to evidence pack.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Stores curriculum harmonic delta under evidence["governance"]["harmonic_curriculum_panel"]["delta"]
    for inclusion in evidence packs when both mock and real grids are available.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The attachment is purely observational
    - No control flow depends on the attached data
    - Non-mutating: returns new dict, does not modify input

    Args:
        evidence: Evidence pack dictionary (will be copied, not modified in-place)
        delta: Curriculum harmonic delta from build_curriculum_harmonic_delta()

    Returns:
        New dict with evidence contents plus delta attached under harmonic_curriculum_panel
    """
    # Create a copy to avoid mutating the original
    enriched = evidence.copy()

    # Ensure governance structure exists
    if "governance" not in enriched:
        enriched["governance"] = {}
    else:
        enriched["governance"] = dict(enriched["governance"])

    # Ensure harmonic_curriculum_panel exists
    if "harmonic_curriculum_panel" not in enriched["governance"]:
        enriched["governance"]["harmonic_curriculum_panel"] = {}
    else:
        enriched["governance"]["harmonic_curriculum_panel"] = dict(
            enriched["governance"]["harmonic_curriculum_panel"]
        )

    # Attach delta
    enriched["governance"]["harmonic_curriculum_panel"]["delta"] = delta

    return enriched


def extract_harmonic_curriculum_signal_for_status(
    panel: Dict[str, Any],
    *,
    extraction_source: str = "MISSING",
    frequency_shift_threshold: int = 2,
) -> Dict[str, Any]:
    """
    Extract compact harmonic curriculum signal for First Light status.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Extracts a compact signal from the harmonic curriculum panel suitable for
    inclusion in first_light_status.json signals section.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents

    Args:
        panel: Harmonic curriculum panel from build_curriculum_harmonic_grid()
        extraction_source: Source of extraction ("MANIFEST", "EVIDENCE_JSON", or "MISSING")
        frequency_shift_threshold: Threshold for frequency shift warnings (default: 2)

    Returns:
        Compact signal dictionary with:
        - schema_version: str (from panel, or "1.0.0" if missing, always present)
        - mode: "SHADOW" (constant, always present)
        - extraction_source: str ("MANIFEST" | "EVIDENCE_JSON" | "MISSING")
        - band_counts: Dict[str, int] (COHERENT/PARTIAL/MISMATCHED counts)
        - top_driver_concepts_top5: List[str] (top 5 driver concepts)
        - delta: Optional[Dict[str, Any]] (if delta present in panel)
            - top_driver_overlap_top5: List[str]
            - frequency_shift_top3: List[Dict[str, Any]] (concept + delta)
    """
    schema_version = panel.get("schema_version", "1.0.0")
    band_counts = panel.get("harmonic_band_counts", {
        "COHERENT": 0,
        "PARTIAL": 0,
        "MISMATCHED": 0,
    })
    top_driver_concepts = panel.get("top_driver_concepts", [])[:5]

    signal: Dict[str, Any] = {
        "schema_version": schema_version,
        "mode": "SHADOW",
        "extraction_source": extraction_source,
        "band_counts": band_counts,
        "top_driver_concepts_top5": top_driver_concepts,
    }

    # Extract delta if present
    delta = panel.get("delta")
    if delta:
        overlap = delta.get("top_driver_overlap", [])[:5]
        frequency_shifts = delta.get("frequency_shifts", [])[:3]
        # Extract top 3 frequency shifts (concept + delta)
        shift_top3 = [
            {
                "concept": shift.get("concept", ""),
                "delta": shift.get("delta", 0),
            }
            for shift in frequency_shifts
        ]

        signal["delta"] = {
            "top_driver_overlap_top5": overlap,
            "frequency_shift_top3": shift_top3,
        }

    return signal


def check_harmonic_curriculum_warning(
    panel: Dict[str, Any],
    *,
    frequency_shift_threshold: int = 2,
) -> Optional[str]:
    """
    Generate a single warning for harmonic curriculum panel if needed.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Generates at most one warning if:
    - Any MISMATCHED count > 0, OR
    - Delta shows any concept with abs(delta) >= frequency_shift_threshold

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from string construction)
    - The returned warning is advisory only
    - No control flow depends on the warning

    Args:
        panel: Harmonic curriculum panel from build_curriculum_harmonic_grid()
        frequency_shift_threshold: Threshold for frequency shift warnings (default: 2)

    Returns:
        Warning string if conditions met, None otherwise
    """
    band_counts = panel.get("harmonic_band_counts", {})
    mismatched_count = band_counts.get("MISMATCHED", 0)

    if mismatched_count > 0:
        num_experiments = panel.get("num_experiments", 0)
        return (
            f"Harmonic curriculum panel: {mismatched_count} experiment(s) with MISMATCHED "
            f"harmonic band (out of {num_experiments} total)"
        )

    # Check delta for frequency shifts
    delta = panel.get("delta")
    if delta:
        frequency_shifts = delta.get("frequency_shifts", [])
        for shift in frequency_shifts:
            abs_delta = abs(shift.get("delta", 0))
            if abs_delta >= frequency_shift_threshold:
                concept = shift.get("concept", "unknown")
                return (
                    f"Harmonic curriculum delta: concept '{concept}' shows frequency shift "
                    f"of {shift.get('delta', 0)} (threshold: {frequency_shift_threshold})"
                )

    return None


def harmonic_grid_for_alignment_view(
    signal_or_panel: Dict[str, Any],
    *,
    frequency_shift_threshold: int = 2,
) -> Dict[str, Any]:
    """
    Convert harmonic curriculum signal/panel to GGFL alignment view format.

    STATUS: PHASE X — CAL-EXP CURRICULUM HARMONIC GRID

    Normalizes harmonic curriculum panel into Global Governance Fusion Layer
    (GGFL) unified format for cross-subsystem alignment views.

    SHADOW MODE: This is read-only, observational, and does not influence
    governance decisions. The output is advisory only.

    Args:
        signal_or_panel: Either a status signal from extract_harmonic_curriculum_signal_for_status()
            or a panel from build_curriculum_harmonic_grid()
        frequency_shift_threshold: Threshold for delta shift detection (default: 2)

    Returns:
        GGFL-normalized dict with stable output contract:
        - signal_type: "SIG-HAR" (constant)
        - status: "ok" | "warn" (lowercase)
        - conflict: False (invariant: harmonic curriculum never triggers conflict directly)
        - drivers: List[str] (reason codes, max 3, deterministic ordering)
            - DRIVER_MISMATCHED_PRESENT (if mismatched_count > 0)
            - DRIVER_DELTA_SHIFT_PRESENT (if delta present and threshold triggered)
            - DRIVER_TOP_CONCEPTS_PRESENT (if top concepts present)
        - top_reason_code: str (most important driver, or "NONE" if no drivers)
        - summary: str (single neutral sentence)
        - shadow_mode_invariants: Dict[str, Any] (SHADOW MODE contract indicators)
            - advisory_only: True (invariant)
            - no_enforcement: True (invariant)
            - conflict_invariant: True (invariant)
        - weight_hint: "LOW" (invariant: harmonic curriculum is low-weight advisory)
    """
    # Extract band counts (from signal or panel)
    band_counts = signal_or_panel.get("band_counts") or signal_or_panel.get("harmonic_band_counts", {})
    mismatched_count = band_counts.get("MISMATCHED", 0)

    # Determine status: warn if any MISMATCHED
    status = "warn" if mismatched_count > 0 else "ok"

    # Build reason code drivers (deterministic ordering: mismatched → delta shift → top concepts)
    drivers: List[str] = []

    # 1. DRIVER_MISMATCHED_PRESENT (if mismatched_count > 0)
    if mismatched_count > 0:
        drivers.append("DRIVER_MISMATCHED_PRESENT")

    # 2. DRIVER_DELTA_SHIFT_PRESENT (only if delta fields present and threshold triggered)
    delta = signal_or_panel.get("delta")
    if delta:
        frequency_shifts = delta.get("frequency_shifts", []) or delta.get("frequency_shift_top3", [])
        for shift in frequency_shifts:
            abs_delta = abs(shift.get("delta", 0))
            if abs_delta >= frequency_shift_threshold:
                drivers.append("DRIVER_DELTA_SHIFT_PRESENT")
                break  # Only add once

    # 3. DRIVER_TOP_CONCEPTS_PRESENT (if top concepts present)
    top_drivers = signal_or_panel.get("top_driver_concepts_top5") or signal_or_panel.get("top_driver_concepts", [])
    if top_drivers:
        drivers.append("DRIVER_TOP_CONCEPTS_PRESENT")

    # Limit to max 3 (shouldn't exceed, but defensive)
    drivers = drivers[:3]

    # Build neutral summary
    num_experiments = signal_or_panel.get("num_experiments", 0)
    if num_experiments == 0:
        summary = "Harmonic curriculum alignment: no experiments analyzed"
    elif mismatched_count > 0:
        summary = f"Harmonic curriculum alignment: {mismatched_count} of {num_experiments} experiment(s) show MISMATCHED harmonic band"
    else:
        coherent_count = band_counts.get("COHERENT", 0)
        summary = f"Harmonic curriculum alignment: {coherent_count} of {num_experiments} experiment(s) show COHERENT harmonic band"

    # Determine top_reason_code (most important driver, deterministic)
    top_reason_code = drivers[0] if drivers else "NONE"

    return {
        "signal_type": "SIG-HAR",
        "status": status,
        "conflict": False,  # Harmonic curriculum never triggers conflict directly (invariant)
        "drivers": drivers,
        "top_reason_code": top_reason_code,
        "summary": summary,
        "shadow_mode_invariants": {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        },
        "weight_hint": "LOW",  # Harmonic curriculum is low-weight advisory signal
    }


__all__ = [
    "CURRICULUM_HARMONIC_ANNEX_SCHEMA_VERSION",
    "CURRICULUM_HARMONIC_GRID_SCHEMA_VERSION",
    "CURRICULUM_HARMONIC_DELTA_SCHEMA_VERSION",
    "build_curriculum_harmonic_annex",
    "emit_cal_exp_curriculum_harmonic_annex",
    "persist_curriculum_harmonic_annex",
    "build_curriculum_harmonic_grid",
    "build_curriculum_harmonic_delta",
    "attach_curriculum_harmonic_delta_to_evidence",
    "extract_harmonic_curriculum_signal_for_status",
    "check_harmonic_curriculum_warning",
    "harmonic_grid_for_alignment_view",
]

