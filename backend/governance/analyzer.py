# PHASE IV — SUBSTRATE GOVERNANCE
#
# This module implements the Substrate Identity Governance Analyzer.
# It provides pure functions for analyzing an Identity Ledger to produce
# governance statuses and summaries for director-level review.

from typing import Any, Dict, List

def analyze_substrate_identity_ledger(ledger: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes a complete Identity Ledger to assess stability and conformance.

    Args:
        ledger: A list of SubstrateIdentityEnvelope dictionaries.

    Returns:
        A dictionary containing detailed analysis.
    """
    if not ledger:
        return {
            "schema_version": None,
            "identity_stability_index": 0.0,
            "repeated_behavioral_flags": [],
            "substrate_version_drift": {"detected": True, "hashes": []},
            "governance_status": "BLOCK",
            "error": "Ledger is empty."
        }

    # --- Schema Version ---
    schema_versions = {e.get("spec_version") for e in ledger}
    schema_version = schema_versions.pop() if len(schema_versions) == 1 else "INCONSISTENT"

    # --- Version Drift & Stability ---
    version_hashes = {e.get("version_hash") for e in ledger}
    has_drift = len(version_hashes) > 1
    # Stability is 1.0 only if there is exactly one version hash.
    stability_index = 1.0 / len(version_hashes) if version_hashes else 0.0

    # --- Behavioral Flags ---
    flag_counts: Dict[str, int] = {}
    for envelope in ledger:
        audit = envelope.get("forbidden_behavior_audit", {})
        for check, result in audit.items():
            if result == "FAILED":
                flag_counts[check] = flag_counts.get(check, 0) + 1
    
    has_flags = bool(flag_counts)

    # --- Governance Status ---
    status = "OK"
    if has_drift or has_flags:
        status = "BLOCK"
    elif stability_index < 1.0: # Redundant with has_drift but good for clarity
        status = "BLOCK"

    return {
        "schema_version": schema_version,
        "identity_stability_index": stability_index,
        "repeated_behavioral_flags": [{"check": k, "count": v} for k, v in flag_counts.items()],
        "substrate_version_drift": {"detected": has_drift, "hashes": list(version_hashes)},
        "governance_status": status,
    }

def evaluate_substrate_for_promotion(
    identity_analysis: Dict[str, Any],
    governance_summary: Dict[str, Any] # For future use, per prompt
) -> Dict[str, Any]:
    """
    Applies formal promotion gate rules to a substrate's analysis report.

    Args:
        identity_analysis: The output from analyze_substrate_identity_ledger.
        governance_summary: The output from the runner's build_substrate_summary.

    Returns:
        A dictionary with a clear promotion verdict.
    """
    blocking_reasons = []
    status = "OK"

    # Rule: BLOCK on identity hash drift
    if identity_analysis["substrate_version_drift"]["detected"]:
        status = "BLOCK"
        blocking_reasons.append(
            f"Substrate version drift detected. Found {len(identity_analysis['substrate_version_drift']['hashes'])} unique version hashes."
        )

    # Rule: BLOCK on repeated behavioral flags
    if identity_analysis["repeated_behavioral_flags"]:
        status = "BLOCK"
        for flag in identity_analysis["repeated_behavioral_flags"]:
            blocking_reasons.append(
                f"Forbidden behavior detected: '{flag['check']}' failed {flag['count']} time(s)."
            )

    # Rule: WARN on low stability index (currently equivalent to BLOCK on drift)
    # This can be expanded later with more nuanced metrics.
    # For now, any value less than 1.0 is a drift.
    if identity_analysis["identity_stability_index"] < 1.0 and status != "BLOCK":
        status = "WARN" # This case is unlikely given current rules but is kept for spec alignment
        blocking_reasons.append(f"Identity Stability Index is {identity_analysis['identity_stability_index']:.2f}, below the required threshold of 1.0.")

    return {
        "substrate_ok_for_promotion": status == "OK",
        "status": status,
        "blocking_reasons": blocking_reasons,
    }

def build_substrate_director_panel(
    identity_analysis: Dict[str, Any],
    promotion_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a high-level summary panel for director-level review.

    Args:
        identity_analysis: The output from analyze_substrate_identity_ledger.
        promotion_eval: The output from evaluate_substrate_for_promotion.

    Returns:
        A dictionary with a simple, high-level status display.
    """
    status_map = {
        "OK": "GREEN",
        "WARN": "AMBER",
        "BLOCK": "RED",
    }
    
    status = promotion_eval["status"]
    
    headline = "Substrate is stable, conformant, and approved for promotion."
    if status == "BLOCK":
        headline = f"BLOCK: {promotion_eval['blocking_reasons'][0]}"
    elif status == "WARN":
        headline = f"WARN: {promotion_eval['blocking_reasons'][0]}"

    hashes = identity_analysis["substrate_version_drift"]["hashes"]
    substrate_hash = hashes[0] if len(hashes) == 1 else "INCONSISTENT"

    return {
        "status_light": status_map.get(status, "RED"),
        "identity_stability_index": identity_analysis["identity_stability_index"],
        "substrate_hash": substrate_hash,
        "headline": headline,
    }
