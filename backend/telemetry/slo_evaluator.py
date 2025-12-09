# backend/telemetry/slo_evaluator.py
#
# MODULE: Telemetry SLO Evaluator & Auto-Action Engine
# JURISDICTION: Telemetry Integrity, SLO Enforcement, Automated Governance
# IDENTITY: GEMINI H, Telemetry Sentinel

from typing import Dict, Any, List, Tuple

# --- Canonical SLO Evaluation & Decision Logic ---

def evaluate_telemetry_slo(snapshot: Dict[str, Any], slo_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a telemetry conformance snapshot against a defined SLO configuration.

    This function serves as the core of the SLO enforcement engine. It is deterministic
    and produces a precise verdict based on immutable thresholds.

    Args:
        snapshot: A dictionary summarizing a batch of telemetry. Expected keys:
            - 'total_records': int
            - 'quarantined_records': int
            - 'l2_record_count': int
            - 'critical_violations': List[Dict]
        slo_config: A dictionary defining the SLO thresholds. Expected keys:
            - 'max_quarantine_ratio': float (e.g., 0.05 for 5%)
            - 'min_l2_percentage': float (e.g., 0.70 for 70%)
            - 'max_critical_violations': int (e.g., 0)
            - 'warn_quarantine_ratio': float (optional, for WARN state)

    Returns:
        A dictionary containing the SLO evaluation result.
    """
    reasons = []
    status = "OK"

    # --- Invariant 1: Quarantine Ratio ---
    max_quarantine_ratio = slo_config.get("max_quarantine_ratio", 1.0)
    current_quarantine_ratio = snapshot.get("quarantined_records", 0) / snapshot.get("total_records", 1)
    if current_quarantine_ratio > max_quarantine_ratio:
        reasons.append(
            f"BREACH: Quarantine ratio {current_quarantine_ratio:.2%} exceeds SLO of {max_quarantine_ratio:.2%}"
        )

    # --- Invariant 2: L2 Percentage ---
    min_l2_percentage = slo_config.get("min_l2_percentage", 0.0)
    current_l2_percentage = snapshot.get("l2_record_count", 0) / snapshot.get("total_records", 1)
    if current_l2_percentage < min_l2_percentage:
        reasons.append(
            f"BREACH: L2 record percentage {current_l2_percentage:.2%} is below SLO of {min_l2_percentage:.2%}"
        )

    # --- Invariant 3: Critical Violations ---
    max_critical_violations = slo_config.get("max_critical_violations", 0)
    num_critical_violations = len(snapshot.get("critical_violations", []))
    if num_critical_violations > max_critical_violations:
        reasons.append(
            f"BREACH: Found {num_critical_violations} critical violations, exceeding SLO of {max_critical_violations}"
        )

    # --- Determine Final Status ---
    if reasons:
        status = "BREACH"
    else:
        # Check for optional WARN condition if no breach occurred
        warn_quarantine_ratio = slo_config.get("warn_quarantine_ratio")
        if warn_quarantine_ratio is not None and current_quarantine_ratio > warn_quarantine_ratio:
            status = "WARN"
            reasons.append(
                f"WARN: Quarantine ratio {current_quarantine_ratio:.2%} exceeds warning threshold of {warn_quarantine_ratio:.2%}"
            )

    return {
        "slo_status": status,
        "reasons": reasons
    }


def decide_telemetry_publication(snapshot: Dict[str, Any], slo_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes a deterministic decision on whether to publish a telemetry snapshot.

    This function translates an SLO verdict into a concrete, automated action,
    forming the core of the auto-quarantine policy.

    Args:
        snapshot: The original telemetry conformance snapshot.
        slo_result: The result from evaluate_telemetry_slo.

    Returns:
        A dictionary containing the publication decision.
    """
    status = slo_result.get("slo_status")

    if status == "BREACH":
        return {
            "publish_allowed": False,
            "require_manual_review": True,
            "recommended_action": "QUARANTINE"
        }
    elif status == "WARN":
        return {
            "publish_allowed": True,
            "require_manual_review": True,
            "recommended_action": "PUBLISH_WITH_WARNING"
        }
    # Default case: status == "OK"
    else:
        return {
            "publish_allowed": True,
            "require_manual_review": False,
            "recommended_action": "PUBLISH"
        }


def summarize_telemetry_slo_for_global_health(slo_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a simplified SLO summary suitable for high-level monitoring.

    This function creates a high-signal, low-noise artifact for ingestion by
    a global health dashboard or Monitoring-as-a-Service (MAAS).

    Args:
        slo_result: The result from evaluate_telemetry_slo.

    Returns:
        A simplified dictionary for global health reporting.
    """
    status = slo_result.get("slo_status", "UNKNOWN")
    reasons = slo_result.get("reasons", [])
    
    key_reason = "All SLOs met."
    if reasons:
        key_reason = reasons[0]

    return {
        "slo_status": status,
        "any_breach": status == "BREACH",
        "key_reason": key_reason
    }
