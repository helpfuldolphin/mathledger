"""Consensus Polygraph adapter for global health.

STATUS: PHASE X — CONSENSUS GOVERNANCE TILE

Provides integration between consensus polygraph signals (conflicts, agreement rate, predictive risks)
and the global health surface builder.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The consensus_governance tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CONSENSUS_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"


def _validate_polygraph_result(polygraph_result: Dict[str, Any]) -> None:
    """Validate polygraph result structure.
    
    Args:
        polygraph_result: Polygraph result dictionary
        
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["agreement_rate", "consensus_band", "system_conflicts"]
    missing = [key for key in required_keys if key not in polygraph_result]
    if missing:
        raise ValueError(
            f"polygraph_result missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(polygraph_result.keys()))}"
        )


def _validate_director_panel(director_panel: Dict[str, Any]) -> None:
    """Validate director panel structure.
    
    Args:
        director_panel: Director panel dictionary
        
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["status_light", "headline"]
    missing = [key for key in required_keys if key not in director_panel]
    if missing:
        raise ValueError(
            f"director_panel missing required keys: {', '.join(missing)}. "
            f"Available keys: {', '.join(sorted(director_panel.keys()))}"
        )


def _extract_predictive_risk_band(predictive_result: Optional[Dict[str, Any]]) -> str:
    """Extract predictive risk band from predictive result.
    
    Args:
        predictive_result: Optional predictive conflict detection result
        
    Returns:
        Risk band: "HIGH", "MEDIUM", "LOW", or "UNKNOWN"
    """
    if predictive_result is None:
        return "UNKNOWN"
    
    high_risk_count = predictive_result.get("high_risk_predictions", 0)
    total_predictions = predictive_result.get("total_predictions", 0)
    
    if high_risk_count > 0:
        return "HIGH"
    elif total_predictions > 0:
        return "MEDIUM"
    else:
        return "LOW"


def build_consensus_governance_tile(
    polygraph_result: Dict[str, Any],
    predictive_result: Optional[Dict[str, Any]],
    director_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build consensus governance tile for global health surface.
    
    Governance tile summarizing cross-system consensus and predictive conflict risk.
    
    SHADOW MODE: Observational only.
    
    Args:
        polygraph_result: Consensus polygraph result from build_consensus_polygraph()
        predictive_result: Optional predictive conflict detection result from detect_predictive_conflicts()
        director_panel: Director panel from build_consensus_director_panel()
        
    Returns:
        Consensus governance tile dictionary with fields:
        - status_light: Status light from director panel (GREEN/YELLOW/RED)
        - consensus_band: Consensus band from polygraph (HIGH/MEDIUM/LOW)
        - agreement_rate: Agreement rate from polygraph (0.0-1.0)
        - conflict_count: Number of system conflicts
        - predictive_risk_band: Risk band from predictive detector (HIGH/MEDIUM/LOW/UNKNOWN)
        - predictive_conflict_count: Number of predicted conflicts
        - headline: Headline from director panel
        - schema_version: Schema version string
    """
    # Validate inputs
    _validate_polygraph_result(polygraph_result)
    _validate_director_panel(director_panel)
    
    # Extract conflict count
    system_conflicts = polygraph_result.get("system_conflicts", [])
    conflict_count = len(system_conflicts) if isinstance(system_conflicts, list) else 0
    
    # Extract predictive conflict count
    predictive_conflict_count = 0
    if predictive_result is not None:
        predictive_conflicts = predictive_result.get("predictive_conflicts", [])
        if isinstance(predictive_conflicts, list):
            predictive_conflict_count = len(predictive_conflicts)
    
    # Extract predictive risk band
    predictive_risk_band = _extract_predictive_risk_band(predictive_result)
    
    # Build tile
    tile = {
        "schema_version": CONSENSUS_GOVERNANCE_TILE_SCHEMA_VERSION,
        "status_light": director_panel.get("status_light", "UNKNOWN"),
        "consensus_band": polygraph_result.get("consensus_band", "UNKNOWN"),
        "agreement_rate": polygraph_result.get("agreement_rate", 0.0),
        "conflict_count": conflict_count,
        "predictive_risk_band": predictive_risk_band,
        "predictive_conflict_count": predictive_conflict_count,
        "headline": director_panel.get("headline", ""),
    }
    
    return tile


def extract_consensus_signal_for_evidence(polygraph_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract consensus signal for evidence packs.
    
    Returns a compact signal suitable for inclusion in evidence packs for external reviewers.
    
    Args:
        polygraph_result: Consensus polygraph result from build_consensus_polygraph()
        
    Returns:
        Dictionary with fields:
        - consensus_band: Consensus band (HIGH/MEDIUM/LOW)
        - agreement_rate: Agreement rate (0.0-1.0)
        - conflict_count: Number of system conflicts
    """
    # Validate input
    _validate_polygraph_result(polygraph_result)
    
    # Extract conflict count
    system_conflicts = polygraph_result.get("system_conflicts", [])
    conflict_count = len(system_conflicts) if isinstance(system_conflicts, list) else 0
    
    return {
        "consensus_band": polygraph_result.get("consensus_band", "UNKNOWN"),
        "agreement_rate": polygraph_result.get("agreement_rate", 0.0),
        "conflict_count": conflict_count,
    }


def build_p3_consensus_summary(
    tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build consensus governance summary for P3 summary.json.
    
    Args:
        tile: Consensus governance tile from build_consensus_governance_tile()
        
    Returns:
        Dictionary with fields:
        - consensus_band: Consensus band (HIGH/MEDIUM/LOW)
        - agreement_rate: Agreement rate (0.0-1.0)
        - conflict_count: Number of conflicts
        - predictive_risk_band: Risk band (HIGH/MEDIUM/LOW/UNKNOWN)
        - status_light: Status light (GREEN/YELLOW/RED)
    """
    return {
        "consensus_band": tile.get("consensus_band", "UNKNOWN"),
        "agreement_rate": tile.get("agreement_rate", 0.0),
        "conflict_count": tile.get("conflict_count", 0),
        "predictive_risk_band": tile.get("predictive_risk_band", "UNKNOWN"),
        "status_light": tile.get("status_light", "UNKNOWN"),
    }


def build_p4_consensus_calibration(
    tile: Dict[str, Any],
    highlighted_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build consensus governance calibration data for P4.
    
    Args:
        tile: Consensus governance tile from build_consensus_governance_tile()
        highlighted_cases: Optional list of highlighted conflict cases
        
    Returns:
        Dictionary with fields:
        - agreement_rate: Agreement rate (0.0-1.0)
        - conflict_count: Number of conflicts
        - predictive_risk_band: Risk band (HIGH/MEDIUM/LOW/UNKNOWN)
        - highlighted_cases: List of highlighted cases (if provided)
    """
    result = {
        "agreement_rate": tile.get("agreement_rate", 0.0),
        "conflict_count": tile.get("conflict_count", 0),
        "predictive_risk_band": tile.get("predictive_risk_band", "UNKNOWN"),
    }
    
    if highlighted_cases is not None:
        result["highlighted_cases"] = highlighted_cases
    
    return result


def build_first_light_conflict_ledger(
    consensus_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build First Light conflict ledger annex.
    
    SHADOW MODE: Observational only. Enumerates cross-layer disagreements for First Light.
    
    Args:
        consensus_tile: Consensus governance tile from build_consensus_governance_tile()
        
    Returns:
        Dictionary with fields:
        - schema_version: Schema version string
        - consensus_band: Consensus band (HIGH/MEDIUM/LOW)
        - agreement_rate: Agreement rate (0.0-1.0)
        - conflict_count: Number of conflicts
        - predictive_risk_band: Risk band (HIGH/MEDIUM/LOW/UNKNOWN)
    """
    return {
        "schema_version": "1.0.0",
        "consensus_band": consensus_tile.get("consensus_band", "UNKNOWN"),
        "agreement_rate": consensus_tile.get("agreement_rate", 0.0),
        "conflict_count": consensus_tile.get("conflict_count", 0),
        "predictive_risk_band": consensus_tile.get("predictive_risk_band", "UNKNOWN"),
    }


def emit_cal_exp_conflict_ledger(
    cal_id: str,
    consensus_tile: Dict[str, Any],
    output_dir: str = "calibration",
) -> Path:
    """
    Emit conflict ledger JSON for a calibration experiment.
    
    STATUS: PHASE X — CAL-EXP CONSENSUS CONFLICT LEDGER EMISSION
    
    Writes the conflict ledger to a JSON file for a specific CAL-EXP run.
    The file is named `conflict_ledger_<cal_id>.json` in the output directory.
    
    SHADOW MODE: Observational only. No governance writes.
    
    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3")
        consensus_tile: Consensus governance tile from build_consensus_governance_tile()
        output_dir: Output directory for conflict ledger files (default: "calibration")
        
    Returns:
        Path to written conflict ledger file (e.g., calibration/conflict_ledger_CAL-EXP-1.json)
        
    Example:
        >>> tile = {"consensus_band": "MEDIUM", "agreement_rate": 0.75, ...}
        >>> path = emit_cal_exp_conflict_ledger("CAL-EXP-1", tile)
        >>> path.name
        'conflict_ledger_CAL-EXP-1.json'
    """
    ledger = build_first_light_conflict_ledger(consensus_tile)
    
    # Add cal_id to ledger for traceability
    ledger["cal_id"] = cal_id
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write ledger to file
    filename = f"conflict_ledger_{cal_id}.json"
    file_path = output_path / filename
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    
    return file_path


def build_consensus_conflict_register(
    ledgers: List[Dict[str, Any]],
    high_conflict_threshold: int = 5,
) -> Dict[str, Any]:
    """
    Build consensus conflict register aggregating multiple calibration experiment ledgers.
    
    STATUS: PHASE X — CAL-EXP CONSENSUS CONFLICT REGISTER
    
    Aggregates conflict ledgers across CAL-EXP-1/2/3 to provide a summary view of
    cross-layer disagreement patterns across experiments.
    
    SHADOW MODE: Observational only. Provides diagnostic summary.
    
    Args:
        ledgers: List of conflict ledger dictionaries from build_first_light_conflict_ledger()
                or emit_cal_exp_conflict_ledger(). Each ledger should have:
                - consensus_band
                - agreement_rate
                - conflict_count
                - predictive_risk_band
                - cal_id (optional, for traceability)
        high_conflict_threshold: Threshold for high conflict classification (default: 5)
        
    Returns:
        Dictionary with fields:
        - schema_version: Schema version string
        - total_experiments: Number of experiments analyzed
        - conflict_count_distribution: Distribution of conflict_count buckets
        - high_risk_experiments_count: Count of experiments with predictive_risk_band="HIGH"
        - experiments_high_conflict: List of cal_ids exceeding high_conflict_threshold
        - consensus_band_distribution: Distribution of consensus_band values
        - average_agreement_rate: Average agreement rate across experiments
    """
    if not ledgers:
        return {
            "schema_version": "1.0.0",
            "total_experiments": 0,
            "conflict_count_distribution": {
                "0-2": 0,
                "3-5": 0,
                ">5": 0,
            },
            "high_risk_experiments_count": 0,
            "experiments_high_conflict": [],
            "consensus_band_distribution": {
                "HIGH": 0,
                "MEDIUM": 0,
                "LOW": 0,
            },
            "average_agreement_rate": 0.0,
        }
    
    # Initialize distributions
    conflict_buckets = {"0-2": 0, "3-5": 0, ">5": 0}
    consensus_bands = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    high_risk_count = 0
    high_conflict_cal_ids: List[str] = []
    agreement_rates: List[float] = []
    
    # Process each ledger
    for ledger in ledgers:
        conflict_count = ledger.get("conflict_count", 0)
        consensus_band = ledger.get("consensus_band", "UNKNOWN")
        predictive_risk_band = ledger.get("predictive_risk_band", "UNKNOWN")
        agreement_rate = ledger.get("agreement_rate", 0.0)
        cal_id = ledger.get("cal_id", "unknown")
        
        # Bucket conflict count
        if conflict_count <= 2:
            conflict_buckets["0-2"] += 1
        elif conflict_count <= 5:
            conflict_buckets["3-5"] += 1
        else:
            conflict_buckets[">5"] += 1
        
        # Track high conflict experiments
        if conflict_count > high_conflict_threshold:
            high_conflict_cal_ids.append(cal_id)
        
        # Count consensus bands
        if consensus_band in consensus_bands:
            consensus_bands[consensus_band] += 1
        
        # Count high risk experiments
        if predictive_risk_band == "HIGH":
            high_risk_count += 1
        
        # Collect agreement rates
        if isinstance(agreement_rate, (int, float)):
            agreement_rates.append(agreement_rate)
    
    # Calculate average agreement rate
    average_agreement_rate = (
        sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0.0
    )
    
    # Sort high conflict cal_ids for determinism
    high_conflict_cal_ids.sort()
    
    return {
        "schema_version": "1.0.0",
        "total_experiments": len(ledgers),
        "conflict_count_distribution": conflict_buckets,
        "high_risk_experiments_count": high_risk_count,
        "experiments_high_conflict": high_conflict_cal_ids,
        "consensus_band_distribution": consensus_bands,
        "average_agreement_rate": round(average_agreement_rate, 4),
    }


def summarize_consensus_vs_fusion(
    register: Dict[str, Any],
    ggfl_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize consistency between consensus conflict register and GGFL fusion results.
    
    STATUS: PHASE X — CONSENSUS REGISTER × GGFL CONSISTENCY CROSS-CHECK
    
    Analyzes consistency between consensus conflict register (CAL-EXP aggregation) and
    GGFL fusion outcomes to detect potential misalignments.
    
    SHADOW MODE CONTRACT:
    - Pure advisory analyzer, no gating
    - Non-mutating (returns new dict)
    - Observational only
    
    Args:
        register: Consensus conflict register from build_consensus_conflict_register()
        ggfl_results: Optional GGFL fusion results with escalation and fusion_result
        
    Returns:
        Cross-check summary with:
        - schema_version: Schema version string
        - consistency_status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - examples: List of normalized example dictionaries (max 5, sorted deterministically)
        - advisory_notes: List of neutral advisory notes
    """
    if not register:
        return {
            "schema_version": "1.0.0",
            "consistency_status": "CONSISTENT",
            "examples": [],
            "advisory_notes": ["No consensus conflict register provided for analysis."],
        }
    
    # Extract register metrics
    experiments_high_conflict = register.get("experiments_high_conflict", [])
    high_risk_count = register.get("high_risk_experiments_count", 0)
    total_experiments = register.get("total_experiments", 0)
    
    # Extract GGFL signals if provided
    ggfl_escalation = None
    ggfl_decision = None
    if ggfl_results:
        escalation = ggfl_results.get("escalation", {})
        if isinstance(escalation, dict):
            ggfl_escalation = escalation.get("level_name")
        else:
            ggfl_escalation = ggfl_results.get("escalation_level")
        
        fusion_result = ggfl_results.get("fusion_result", {})
        if isinstance(fusion_result, dict):
            ggfl_decision = fusion_result.get("decision")
        else:
            ggfl_decision = ggfl_results.get("decision")
    
    # Determine consistency status
    consistency_status = "CONSISTENT"
    example_candidates = []
    advisory_notes = []
    
    # CONFLICT: GGFL is L0/ALLOW but register has experiments_high_conflict non-empty
    if ggfl_escalation == "L0_NOMINAL" or ggfl_decision == "ALLOW":
        if experiments_high_conflict:
            consistency_status = "CONFLICT"
            for cal_id in sorted(experiments_high_conflict)[:5]:  # Deterministic, max 5
                example_candidates.append({
                    "cal_id": cal_id,
                    "reason": f"GGFL reports {ggfl_escalation or 'ALLOW'} while experiment {cal_id} has high conflict count.",
                    "severity": "CONFLICT",
                })
            advisory_notes.append(
                f"GGFL reports nominal/allow while {len(experiments_high_conflict)} experiment(s) "
                f"exceed high conflict threshold."
            )
    
    # TENSION: GGFL WARNING and register has high_risk_band_count high
    if ggfl_escalation in ("L1_WARNING", "L2_DEGRADED"):
        if high_risk_count > 0:
            if consistency_status == "CONSISTENT":
                consistency_status = "TENSION"
            # Add examples for high-risk experiments
            conflict_dist = register.get("consensus_band_distribution", {})
            low_band_count = conflict_dist.get("LOW", 0)
            if low_band_count > 0:
                example_candidates.append({
                    "cal_id": "aggregate",
                    "reason": f"GGFL escalation {ggfl_escalation} with {high_risk_count} high-risk experiment(s) and {low_band_count} LOW consensus band(s).",
                    "severity": "TENSION",
                })
            advisory_notes.append(
                f"GGFL reports {ggfl_escalation} while {high_risk_count} experiment(s) have HIGH predictive risk band."
            )
    
    # If no conflicts or tensions detected, add a consistent note
    if consistency_status == "CONSISTENT" and total_experiments > 0:
        advisory_notes.append(
            f"Consensus conflict register and GGFL outcomes are consistent across {total_experiments} experiment(s)."
        )
    
    # Sort examples deterministically (by cal_id, then by severity)
    example_candidates.sort(key=lambda x: (x["severity"], x["cal_id"]))
    examples = example_candidates[:5]  # Max 5 examples
    
    return {
        "schema_version": "1.0.0",
        "consistency_status": consistency_status,
        "examples": examples,
        "advisory_notes": advisory_notes,
    }


def attach_consensus_conflict_register_to_evidence(
    evidence: Dict[str, Any],
    register: Dict[str, Any],
    fusion_crosscheck: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach consensus conflict register to evidence pack.
    
    Inserts consensus conflict register under evidence["governance"]["consensus_conflict_register"].
    If fusion_crosscheck is provided, attaches it under the register's "fusion_crosscheck" field.
    
    SHADOW MODE: Non-mutating (returns new dict structure, but modifies evidence in place).
    
    Args:
        evidence: Evidence pack dictionary (will be modified in place)
        register: Consensus conflict register from build_consensus_conflict_register()
        fusion_crosscheck: Optional fusion cross-check summary from summarize_consensus_vs_fusion()
        
    Returns:
        Modified evidence dictionary with consensus conflict register attached
    """
    # Ensure governance structure exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    
    # Attach consensus conflict register
    evidence["governance"]["consensus_conflict_register"] = register.copy()
    
    # Attach fusion crosscheck if provided
    if fusion_crosscheck is not None:
        evidence["governance"]["consensus_conflict_register"]["fusion_crosscheck"] = fusion_crosscheck
    
    return evidence


def attach_consensus_governance_to_evidence(
    evidence: Dict[str, Any],
    tile: Dict[str, Any],
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach consensus governance data to evidence pack.
    
    Inserts consensus governance information under evidence["governance"]["consensus"].
    Uses neutral language throughout.
    
    Args:
        evidence: Evidence pack dictionary (will be modified in place)
        tile: Consensus governance tile from build_consensus_governance_tile()
        signal: Consensus signal from extract_consensus_signal_for_evidence()
        
    Returns:
        Modified evidence dictionary with consensus governance attached
    """
    # Ensure governance structure exists
    if "governance" not in evidence:
        evidence["governance"] = {}
    
    # Attach consensus governance
    evidence["governance"]["consensus"] = {
        "consensus_band": signal.get("consensus_band", "UNKNOWN"),
        "agreement_rate": signal.get("agreement_rate", 0.0),
        "conflict_count": signal.get("conflict_count", 0),
        "status_light": tile.get("status_light", "UNKNOWN"),
        "predictive_risk_band": tile.get("predictive_risk_band", "UNKNOWN"),
        "predictive_conflict_count": tile.get("predictive_conflict_count", 0),
        "headline": tile.get("headline", ""),
    }
    
    # Attach First Light conflict ledger
    evidence["governance"]["consensus"]["first_light_conflict_ledger"] = (
        build_first_light_conflict_ledger(tile)
    )
    
    return evidence


def summarize_consensus_for_uplift_council(
    tile: Dict[str, Any],
    conflict_threshold: int = 5,
) -> Dict[str, Any]:
    """
    Summarize consensus governance for uplift council decision-making.
    
    Rules:
    - conflict_count > threshold OR predictive_risk_band="HIGH" → BLOCK
    - predictive_risk_band="MEDIUM" → WARN
    - Else → OK
    
    Args:
        tile: Consensus governance tile from build_consensus_governance_tile()
        conflict_threshold: Threshold for conflict count to trigger BLOCK (default: 5)
        
    Returns:
        Dictionary with:
        - verdict: "OK" | "WARN" | "BLOCK"
        - rationale: Short neutral description
        - conflicting_slices: List of slice IDs with conflicts (if available)
        - predictive_anomalies: List of predictive conflict descriptions (if available)
    """
    conflict_count = tile.get("conflict_count", 0)
    predictive_risk_band = tile.get("predictive_risk_band", "UNKNOWN")
    
    # Determine verdict
    if conflict_count > conflict_threshold or predictive_risk_band == "HIGH":
        verdict = "BLOCK"
        rationale_parts = []
        if conflict_count > conflict_threshold:
            rationale_parts.append(f"{conflict_count} conflicts exceed threshold")
        if predictive_risk_band == "HIGH":
            rationale_parts.append("high predictive risk detected")
        rationale = ". ".join(rationale_parts) + "."
    elif predictive_risk_band == "MEDIUM":
        verdict = "WARN"
        rationale = f"Medium predictive risk detected with {conflict_count} conflicts"
    else:
        verdict = "OK"
        rationale = f"Consensus governance status acceptable ({conflict_count} conflicts, {predictive_risk_band} risk)"
    
    # Extract conflicting slices (if available in tile)
    conflicting_slices: List[str] = []
    if "conflicts" in tile:
        # If tile has detailed conflict information
        for conflict in tile.get("conflicts", []):
            if isinstance(conflict, dict) and "slice_id" in conflict:
                conflicting_slices.append(conflict["slice_id"])
    
    # Extract predictive anomalies (if available)
    predictive_anomalies: List[str] = []
    if "predictive_conflicts" in tile:
        for pred in tile.get("predictive_conflicts", []):
            if isinstance(pred, dict) and "reason" in pred:
                predictive_anomalies.append(pred["reason"])
    
    return {
        "verdict": verdict,
        "rationale": rationale,
        "conflicting_slices": conflicting_slices,
        "predictive_anomalies": predictive_anomalies,
        "conflict_count": conflict_count,
        "predictive_risk_band": predictive_risk_band,
    }


def attach_consensus_conflicts_signal(
    signals: Dict[str, Any],
    fusion_crosscheck: Dict[str, Any],
    register: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach consensus conflicts signal to signals dictionary.
    
    STATUS: PHASE X — CONSENSUS CONFLICTS STATUS HOOK
    
    If fusion_crosscheck exists, adds consensus_conflicts signal to the signals dictionary
    with experiments_high_conflict_count, high_risk_band_count, and fusion_consistency_status.
    
    SHADOW MODE: Observational only. Does not gate any decisions.
    
    Args:
        signals: Signals dictionary (will be modified in place)
        fusion_crosscheck: Fusion cross-check summary from summarize_consensus_vs_fusion()
        register: Consensus conflict register from build_consensus_conflict_register()
        
    Returns:
        Modified signals dictionary with consensus_conflicts signal attached
    """
    if not fusion_crosscheck:
        return signals
    
    experiments_high_conflict = register.get("experiments_high_conflict", [])
    high_risk_count = register.get("high_risk_experiments_count", 0)
    consistency_status = fusion_crosscheck.get("consistency_status", "UNKNOWN")
    
    signals["consensus_conflicts"] = {
        "experiments_high_conflict_count": len(experiments_high_conflict),
        "high_risk_band_count": high_risk_count,
        "fusion_consistency_status": consistency_status,
    }
    
    return signals


def consensus_conflicts_for_alignment_view(
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert consensus conflicts signal to GGFL alignment view format.
    
    STATUS: PHASE X — GGFL ADAPTER FOR CONSENSUS CONFLICTS
    
    Normalizes the consensus conflicts signal into the Global Governance Fusion Layer
    (GGFL) unified format for cross-subsystem alignment views.
    
    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It does not gate or block any operations
    - Never claims "good/bad", only descriptive
    - Consensus conflicts never trigger conflict directly (conflict: false always, invariant)
    - Deterministic output for identical inputs
    - Advisory only; no enforcement
    
    SHADOW MODE INVARIANTS:
    - conflict: False (always, consensus conflicts are advisory only)
    - weight_hint: "LOW" (always, consensus conflicts are low-weight advisory signal)
    - signal_type: "SIG-CON" (constant identifier)
    
    Args:
        signal: Consensus conflicts signal from first_light_status.json signals["consensus_conflicts"]
            Must contain: experiments_high_conflict_count, high_risk_band_count,
            fusion_consistency_status, extraction_source, top_reason_code (optional)
    
    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-CON" (identifies this as a consensus signal)
        - status: "ok" | "warn" (warn if TENSION/CONFLICT or high_conflict_count>0 or high_risk_band_count>0)
        - conflict: False (invariant: consensus conflicts never trigger conflict directly)
        - weight_hint: "LOW" (invariant: consensus conflicts are low-weight advisory)
        - drivers: List[str] (max 3, deterministic: reason codes only)
        - summary: str (one neutral sentence)
        - extraction_source: str (MANIFEST | EVIDENCE_JSON | MISSING)
        - shadow_mode_invariants: Dict with advisory_only, no_enforcement, conflict_invariant
    """
    experiments_high_conflict_count = signal.get("experiments_high_conflict_count", 0)
    high_risk_band_count = signal.get("high_risk_band_count", 0)
    fusion_consistency_status = signal.get("fusion_consistency_status", "UNKNOWN")
    extraction_source = signal.get("extraction_source", "MISSING")
    
    # Determine status: warn if TENSION/CONFLICT or high_conflict_count>0 or high_risk_band_count>0
    status = "ok"
    if (
        fusion_consistency_status in ("TENSION", "CONFLICT")
        or experiments_high_conflict_count > 0
        or high_risk_band_count > 0
    ):
        status = "warn"
    
    # Build drivers using reason codes only (max 3, deterministic ordering)
    drivers: List[str] = []
    
    # DRIVER_FUSION_TENSION_OR_CONFLICT: if fusion consistency is TENSION or CONFLICT
    if fusion_consistency_status in ("TENSION", "CONFLICT"):
        drivers.append("DRIVER_FUSION_TENSION_OR_CONFLICT")
    
    # DRIVER_HIGH_CONFLICT_EXPERIMENTS_PRESENT: if any high conflict experiments
    if experiments_high_conflict_count > 0:
        drivers.append("DRIVER_HIGH_CONFLICT_EXPERIMENTS_PRESENT")
    
    # DRIVER_HIGH_RISK_BAND_PRESENT: if any high risk band experiments
    if high_risk_band_count > 0:
        drivers.append("DRIVER_HIGH_RISK_BAND_PRESENT")
    
    # Limit to 3 drivers (already deterministic ordering)
    drivers = drivers[:3]
    
    # Build neutral summary sentence
    if experiments_high_conflict_count == 0 and high_risk_band_count == 0:
        summary = (
            "Consensus conflict register: no high-conflict or high-risk experiments detected "
            "across calibration runs."
        )
    elif fusion_consistency_status in ("TENSION", "CONFLICT"):
        summary = (
            f"Consensus conflict register: fusion consistency status is {fusion_consistency_status} "
            f"({experiments_high_conflict_count} high-conflict, {high_risk_band_count} high-risk experiments)."
        )
    else:
        summary = (
            f"Consensus conflict register: {experiments_high_conflict_count} high-conflict "
            f"and {high_risk_band_count} high-risk experiment(s) detected across calibration runs."
        )
    
    return {
        "signal_type": "SIG-CON",
        "status": status,
        "conflict": False,  # Invariant: consensus conflicts never trigger conflict directly
        "weight_hint": "LOW",  # Invariant: consensus conflicts are low-weight advisory
        "drivers": drivers,
        "summary": summary,
        "extraction_source": extraction_source,
        "shadow_mode_invariants": {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        },
    }


__all__ = [
    "CONSENSUS_GOVERNANCE_TILE_SCHEMA_VERSION",
    "build_consensus_governance_tile",
    "extract_consensus_signal_for_evidence",
    "build_p3_consensus_summary",
    "build_p4_consensus_calibration",
    "build_first_light_conflict_ledger",
    "emit_cal_exp_conflict_ledger",
    "build_consensus_conflict_register",
    "summarize_consensus_vs_fusion",
    "attach_consensus_conflict_register_to_evidence",
    "attach_consensus_conflicts_signal",
    "consensus_conflicts_for_alignment_view",
    "attach_consensus_governance_to_evidence",
    "summarize_consensus_for_uplift_council",
]

