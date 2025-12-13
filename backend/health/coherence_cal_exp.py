"""Coherence CAL-EXP cross-check integration.

STATUS: PHASE X — COHERENCE GOVERNANCE LAYER

Provides coherence snapshot generation for calibration experiments (CAL-EXP-1/2/3)
and cross-check analysis with GGFL fusion results.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Coherence snapshots are observational only
- Cross-check analysis is advisory, not a hard gate
- No control flow depends on coherence values
- No governance writes

SEVERITY SCORING DOCUMENTATION:
--------------------------------
The severity score formula is: score = (coherence_severity × 10) + ggfl_escalation_severity

Coherence is weighted 10× more than GGFL escalation because:
1. Coherence represents structural consistency between decoy confusability geometry
   and topology health - a fundamental architectural property
2. This ensures MISALIGNED coherence (score 20+) always outranks PARTIAL coherence
   (score 10+) regardless of GGFL escalation level
3. Coherence band changes are more significant than transient GGFL escalation changes
4. Examples are primarily sorted by structural consistency, with GGFL escalation
   providing secondary ordering within the same coherence band

See _score_example() function for implementation details.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

COHERENCE_SNAPSHOT_SCHEMA_VERSION = "1.0.0"
COHERENCE_CROSSCHECK_SCHEMA_VERSION = "1.0.0"
COHERENCE_CROSSCHECK_MODE = "SHADOW"

# Frozen allowed values (CROSSCHECK CONTRACT v1 HARD FREEZE)
ALLOWED_CONSISTENCY_STATUS = {"CONSISTENT", "TENSION", "CONFLICT", "UNKNOWN"}
ALLOWED_GGFL_ESCALATION = {
    "L0_NOMINAL", "L1_WARNING", "L2_DEGRADED",
    "L3_CRITICAL", "L4_CONFLICT", "L5_EMERGENCY"
}

# Selection contract (frozen formula and tie-breakers)
SELECTION_CONTRACT_SEVERITY_FORMULA = "score = (coherence_severity × 10) + ggfl_escalation_severity"
SELECTION_CONTRACT_TIE_BREAKERS = [
    "severity_score (descending)",
    "cal_id (ascending)"
]


def _score_example(coherence_band: str, ggfl_escalation: Optional[str]) -> int:
    """
    Compute deterministic severity score for an example.
    
    Severity Score Formula:
    -----------------------
    score = (coherence_severity × 10) + ggfl_escalation_severity
    
    Where:
    - coherence_severity: COHERENT=0, PARTIAL=1, MISALIGNED=2
    - ggfl_escalation_severity: L0_NOMINAL=0, L1_WARNING=1, L2_DEGRADED=2,
                                L3_CRITICAL=3, L4_CONFLICT=4, L5_EMERGENCY=5
    
    Rationale:
    - Coherence band is weighted 10x more than GGFL escalation
    - This ensures MISALIGNED coherence always ranks higher than PARTIAL
    - GGFL escalation provides fine-grained ordering within same coherence band
    
    Why Coherence is Weighted 10×:
    ------------------------------
    Coherence represents structural consistency between decoy confusability geometry
    and topology health. This is a fundamental architectural property that should
    take precedence over transient GGFL escalation levels. The 10× weighting ensures:
    
    1. MISALIGNED coherence (score 20+) always outranks PARTIAL coherence (score 10+)
       regardless of GGFL escalation level
    2. Coherence band changes are more significant than GGFL escalation changes
    3. Examples are primarily sorted by structural consistency, with GGFL escalation
       providing secondary ordering within the same coherence band
    
    Examples:
    - MISALIGNED + L3_CRITICAL = (2 × 10) + 3 = 23
    - PARTIAL + L1_WARNING = (1 × 10) + 1 = 11
    - COHERENT + L0_NOMINAL = (0 × 10) + 0 = 0
    
    Args:
        coherence_band: "COHERENT" | "PARTIAL" | "MISALIGNED"
        ggfl_escalation: Optional GGFL escalation level
        
    Returns:
        Integer severity score (higher = more severe)
    """
    # Map coherence bands to severity scores (higher = more severe)
    coherence_severity = {
        "COHERENT": 0,
        "PARTIAL": 1,
        "MISALIGNED": 2,
    }
    
    # Map escalation levels to severity scores (higher = more severe)
    escalation_severity = {
        "L0_NOMINAL": 0,
        "L1_WARNING": 1,
        "L2_DEGRADED": 2,
        "L3_CRITICAL": 3,
        "L4_CONFLICT": 4,
        "L5_EMERGENCY": 5,
    }
    
    coherence_sev = coherence_severity.get(coherence_band, 1)
    ggfl_sev = escalation_severity.get(ggfl_escalation, 0) if ggfl_escalation else 0
    
    return coherence_sev * 10 + ggfl_sev


def _coerce_consistency_status(status: Any) -> str:
    """
    Coerce consistency_status to allowed value.
    
    CROSSCHECK CONTRACT v1: Frozen allowed values.
    Unknown values → "UNKNOWN" with advisory note.
    
    Args:
        status: Consistency status value (any type)
        
    Returns:
        Coerced status: "CONSISTENT" | "TENSION" | "CONFLICT" | "UNKNOWN"
    """
    if isinstance(status, str) and status in ALLOWED_CONSISTENCY_STATUS:
        return status
    # Unknown value → UNKNOWN
    return "UNKNOWN"


def _coerce_ggfl_escalation(escalation: Any) -> str:
    """
    Coerce GGFL escalation to allowed value.
    
    CROSSCHECK CONTRACT v1: Frozen allowed values.
    Unknown values → "L0_NOMINAL" with advisory note.
    
    Args:
        escalation: GGFL escalation value (any type)
        
    Returns:
        Coerced escalation: "L0_NOMINAL" | "L1_WARNING" | ... | "L5_EMERGENCY"
    """
    if isinstance(escalation, str) and escalation in ALLOWED_GGFL_ESCALATION:
        return escalation
    # Unknown value → L0_NOMINAL (safest default)
    return "L0_NOMINAL"


def build_cal_exp_coherence_snapshot(
    cal_id: str,
    first_light_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build coherence snapshot for a calibration experiment.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Coherence is observational only
    
    Args:
        cal_id: Calibration experiment identifier (e.g., "cal_exp1", "CAL-EXP-1")
        first_light_summary: Coherence first-light summary from attach_coherence_to_evidence
        
    Returns:
        Coherence snapshot with:
        - schema_version: "1.0.0"
        - cal_id: str
        - coherence_band: "COHERENT" | "PARTIAL" | "MISALIGNED"
        - global_index: float
        - num_slices_at_risk: int
    """
    coherence_band = first_light_summary.get("coherence_band", "PARTIAL")
    global_index = first_light_summary.get("global_index", 0.5)
    slices_at_risk = first_light_summary.get("slices_at_risk", [])
    
    snapshot = {
        "schema_version": COHERENCE_SNAPSHOT_SCHEMA_VERSION,
        "cal_id": cal_id,
        "coherence_band": coherence_band,
        "global_index": round(global_index, 6),
        "num_slices_at_risk": len(slices_at_risk),
    }
    
    return snapshot


def persist_coherence_snapshot(
    snapshot: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Persist coherence snapshot to disk.
    
    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions
    
    Args:
        snapshot: Coherence snapshot from build_cal_exp_coherence_snapshot
        output_dir: Directory to write snapshot (e.g., calibration/)
        
    Returns:
        Path to written snapshot file
    """
    cal_id = snapshot.get("cal_id", "unknown")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"coherence_{cal_id}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    
    return output_path


def summarize_coherence_vs_fusion(
    coherence_snapshots: List[Dict[str, Any]],
    ggfl_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize coherence vs GGFL fusion consistency.
    
    Analyzes consistency between coherence signals and GGFL fusion results
    to identify tension or conflict patterns.
    
    SHADOW MODE CONTRACT:
    - Pure advisory analyzer, no gating
    - Non-mutating (returns new dict)
    - Observational only
    
    Args:
        coherence_snapshots: List of coherence snapshots from build_cal_exp_coherence_snapshot
        ggfl_results: Optional GGFL fusion results with escalation_level, decision, and recommendations
        
    Returns:
        Cross-check summary with:
        - consistency_status: "CONSISTENT" | "TENSION" | "CONFLICT"
        - examples: List of normalized example dictionaries (max 5, sorted by severity)
        - advisory_notes: List of neutral advisory notes
    """
    if not coherence_snapshots:
        return {
            "consistency_status": "CONSISTENT",
            "examples": [],
            "advisory_notes": ["No coherence snapshots provided for analysis."],
        }
    
    # Analyze coherence patterns
    misaligned_count = sum(
        1 for snap in coherence_snapshots
        if snap.get("coherence_band") == "MISALIGNED"
    )
    partial_count = sum(
        1 for snap in coherence_snapshots
        if snap.get("coherence_band") == "PARTIAL"
    )
    coherent_count = sum(
        1 for snap in coherence_snapshots
        if snap.get("coherence_band") == "COHERENT"
    )
    
    # Extract GGFL signals if provided
    ggfl_escalation = None
    ggfl_decision = None
    ggfl_warnings = []
    ggfl_primary_signal = None
    coercion_notes = []  # Track coercion events for advisory notes
    if ggfl_results:
        raw_escalation = ggfl_results.get("escalation_level")
        # Coerce GGFL escalation to allowed value (CROSSCHECK CONTRACT v1)
        ggfl_escalation = _coerce_ggfl_escalation(raw_escalation)
        if raw_escalation != ggfl_escalation and raw_escalation is not None:
            coercion_notes.append(f"GGFL escalation coerced from '{raw_escalation}' to '{ggfl_escalation}'.")
        ggfl_decision = ggfl_results.get("decision")
        recommendations = ggfl_results.get("recommendations", [])
        ggfl_warnings = [
            r for r in recommendations
            if r.get("action") == "WARNING"
        ]
        # Extract primary signal ID from highest priority recommendation
        if recommendations:
            primary_rec = max(recommendations, key=lambda r: r.get("priority", 0))
            ggfl_primary_signal = primary_rec.get("signal_id")
    
    # Determine consistency status and build normalized examples
    consistency_status = "CONSISTENT"
    example_candidates = []
    
    # Build example candidates from each snapshot
    for snap in coherence_snapshots:
        cal_id = snap.get("cal_id", "unknown")
        coherence_band = snap.get("coherence_band", "PARTIAL")
        coherence_index = snap.get("global_index")  # Optional: may be None
        
        # Determine example severity using helper function
        combined_severity = _score_example(coherence_band, ggfl_escalation)
        
        # Build reason string
        if ggfl_escalation and coherence_band == "MISALIGNED":
            if ggfl_escalation in ("L1_WARNING", "L2_DEGRADED"):
                reason = f"GGFL escalation {ggfl_escalation} with MISALIGNED coherence."
                if consistency_status == "CONSISTENT":
                    consistency_status = "TENSION"
            elif ggfl_escalation == "L0_NOMINAL":
                reason = "GGFL reports L0_NOMINAL while coherence is MISALIGNED."
                consistency_status = "CONFLICT"
            else:
                reason = f"GGFL escalation {ggfl_escalation} with MISALIGNED coherence."
                if consistency_status == "CONSISTENT":
                    consistency_status = "TENSION"
        elif coherence_band == "PARTIAL" and len(ggfl_warnings) >= 2:
            reason = f"Multiple GGFL warnings ({len(ggfl_warnings)}) with PARTIAL coherence."
            if consistency_status == "CONSISTENT":
                consistency_status = "TENSION"
        elif coherence_band == "MISALIGNED":
            reason = "MISALIGNED coherence detected."
            if consistency_status == "CONSISTENT":
                consistency_status = "TENSION"
        else:
            # Skip examples for consistent cases
            continue
        
        example_dict = {
            "cal_id": cal_id,
            "coherence_band": coherence_band,
            "ggfl_escalation": ggfl_escalation or "UNKNOWN",
            "ggfl_decision": ggfl_decision or "UNKNOWN",
            "reason": reason,
            "_severity": combined_severity,  # Internal field for sorting
        }
        
        # Add optional fields if available (never error if missing)
        if coherence_index is not None:
            example_dict["coherence_index"] = round(coherence_index, 6)
        if ggfl_primary_signal:
            example_dict["ggfl_primary_signal"] = ggfl_primary_signal
        
        example_candidates.append(example_dict)
    
    # Sort by severity (descending) then cal_id (ascending), limit to top 5
    example_candidates.sort(key=lambda x: (-x["_severity"], x["cal_id"]))
    examples = [
        {k: v for k, v in ex.items() if k != "_severity"}
        for ex in example_candidates[:5]
    ]
    
    # Coerce consistency_status to allowed value (CROSSCHECK CONTRACT v1)
    raw_consistency_status = consistency_status
    consistency_status = _coerce_consistency_status(consistency_status)
    if raw_consistency_status != consistency_status:
        coercion_notes.append(f"Consistency status coerced from '{raw_consistency_status}' to '{consistency_status}'.")
    
    # Build advisory notes
    advisory_notes = []
    advisory_notes.append(
        f"Coherence analysis: {coherent_count} COHERENT, "
        f"{partial_count} PARTIAL, {misaligned_count} MISALIGNED."
    )
    
    # Add coercion notes if any
    advisory_notes.extend(coercion_notes)
    
    if consistency_status == "CONSISTENT":
        advisory_notes.append("Coherence signals align with GGFL fusion results.")
    elif consistency_status == "TENSION":
        advisory_notes.append(
            "Tension detected between coherence and GGFL signals; "
            "review recommended but not blocking."
        )
    elif consistency_status == "CONFLICT":
        advisory_notes.append(
            "Conflict detected between coherence and GGFL signals; "
            "investigation recommended."
        )
    else:  # UNKNOWN
        advisory_notes.append(
            "Consistency status unknown; manual review recommended."
        )
    
    return {
        "schema_version": COHERENCE_CROSSCHECK_SCHEMA_VERSION,
        "mode": COHERENCE_CROSSCHECK_MODE,
        "consistency_status": consistency_status,
        "examples": examples,
        "advisory_notes": advisory_notes,
    }


def attach_coherence_fusion_crosscheck_to_evidence(
    evidence: Dict[str, Any],
    crosscheck_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach coherence-fusion cross-check to evidence pack.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Cross-check is observational only
    
    Args:
        evidence: Evidence dictionary
        crosscheck_summary: Cross-check summary from summarize_coherence_vs_fusion
        
    Returns:
        Updated evidence with coherence_fusion_crosscheck under evidence["governance"]
    """
    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)
    
    # Ensure governance structure exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])
    
    # Attach cross-check
    updated_evidence["governance"]["coherence_fusion_crosscheck"] = crosscheck_summary
    
    return updated_evidence


def extract_coherence_fusion_status(
    crosscheck_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract status summary for signals.coherence_fusion_crosscheck.
    
    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Observational only
    
    Args:
        crosscheck_summary: Cross-check summary from summarize_coherence_vs_fusion
        
    Returns:
        Status summary with:
        - consistency_status: "CONSISTENT" | "TENSION" | "CONFLICT" | "UNKNOWN"
        - example_count: int
        - top_example_cal_id: str | None
        - top_example_reason: str | None (one-line neutral explanation)
        - selection_contract: Dict with severity_formula and tie_breakers
    """
    raw_consistency_status = crosscheck_summary.get("consistency_status", "CONSISTENT")
    # Coerce consistency_status to allowed value
    consistency_status = _coerce_consistency_status(raw_consistency_status)
    examples = crosscheck_summary.get("examples", [])
    
    top_example_cal_id = None
    top_example_reason = None
    
    if examples:
        top_example = examples[0]
        top_example_cal_id = top_example.get("cal_id")
        
        # Build deterministic top-example reason
        coherence_band = top_example.get("coherence_band", "UNKNOWN")
        ggfl_escalation = top_example.get("ggfl_escalation", "UNKNOWN")
        
        # Format: "highest severity score (COHERENCE_BAND + GGFL_ESCALATION)"
        # Deterministic formatting; no emojis
        top_example_reason = f"highest severity score ({coherence_band} + {ggfl_escalation})"
    
    # Selection contract (frozen for audit-grade stability)
    selection_contract = {
        "severity_formula": SELECTION_CONTRACT_SEVERITY_FORMULA,
        "tie_breakers": SELECTION_CONTRACT_TIE_BREAKERS,
    }
    
    return {
        "consistency_status": consistency_status,
        "example_count": len(examples),
        "top_example_cal_id": top_example_cal_id,
        "top_example_reason": top_example_reason,
        "selection_contract": selection_contract,
    }


__all__ = [
    "COHERENCE_SNAPSHOT_SCHEMA_VERSION",
    "build_cal_exp_coherence_snapshot",
    "persist_coherence_snapshot",
    "summarize_coherence_vs_fusion",
    "attach_coherence_fusion_crosscheck_to_evidence",
    "extract_coherence_fusion_status",
]



