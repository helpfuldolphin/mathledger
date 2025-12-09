# PHASE IV — CROSS-RUN GOVERNANCE RADAR & DETERMINISM CONTRACT ENFORCER
"""
Implements the Replay Governance Radar, the Determinism Promotion Coupler,
and the Director Replay Panel. These components provide advanced, cross-run
analysis of the replay evidence spine for governance and decision-making.
"""
from typing import Dict, Any, List, Sequence

# --- Foundational Data Structures (from Phase III) ---
# For clarity, these are redefined here. In a real system, they'd be imported.

class ReplayHistoryLedger(dict): pass
class ReplayIncident(dict): pass
class GlobalHealthSummary(dict): pass

# --- TASK 1: Cross-Run Determinism Governance Radar ---

import yaml

def _load_criticality_rules(config_path: str) -> List[str]:
    """Loads critical RECON codes from a YAML config file."""
    default_critical_codes = ["RECON-001", "RECON-005"]
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get("critical_recon_codes", default_critical_codes)
    except (FileNotFoundError, yaml.YAMLError):
        return default_critical_codes

def build_replay_governance_radar(
    history_ledger: ReplayHistoryLedger,
    incidents: List[ReplayIncident],
    criticality_config_path: str = "config/replay_criticality_rules.yaml"
) -> Dict[str, Any]:
    """Builds a deterministic governance radar from the history and incidents."""

    # This is a simulation. A real implementation would process a rich time-series.
    # Here, we derive rates from the ledger's totals.
    total = history_ledger.get("total_receipts", 0)
    verified = history_ledger.get("number_verified", 0)
    failed = history_ledger.get("number_failed", 0)
    
    determinism_rate = verified / total if total > 0 else 1.0
    incident_rate = failed / total if total > 0 else 0.0

    # Simulate a time-series for demonstration
    determinism_rate_series = [
        {"timestamp": "2025-01-01T00:00:00Z", "rate": 0.98},
        {"timestamp": "2025-01-02T00:00:00Z", "rate": 0.97},
        {"timestamp": "2025-01-03T00:00:00Z", "rate": determinism_rate},
    ]
    incident_rate_series = [
        {"timestamp": "2025-01-01T00:00:00Z", "rate": 0.01},
        {"timestamp": "2025-01-02T00:00:00Z", "rate": 0.02},
        {"timestamp": "2025-01-03T00:00:00Z", "rate": incident_rate},
    ]

    # Identify recurring critical incidents
    critical_codes = _load_criticality_rules(criticality_config_path)
    incident_fingerprints = {}
    for inc in incidents:
        codes = tuple(sorted(inc.get("recon_codes_seen", [])))
        if any(c in critical_codes for c in codes):
            fingerprint = f"{inc['status']}:{':'.join(codes)}"
            incident_fingerprints[fingerprint] = incident_fingerprints.get(fingerprint, 0) + 1
    
    recurring_critical_fingerprints = {k: v for k, v in incident_fingerprints.items() if v > 1}

    # Determine radar status
    status = "STABLE"
    if determinism_rate < 0.95:
        status = "DRIFTING"
    if determinism_rate < 0.90 or recurring_critical_fingerprints:
        status = "UNSTABLE"

    return {
        "schema_version": "1.0.0",
        "determinism_rate_series": determinism_rate_series,
        "incident_rate_series": incident_rate_series,
        "recurring_critical_incident_fingerprints": recurring_critical_fingerprints,
        "radar_status": status,
    }

# --- TASK 2: Determinism Promotion Coupler ---

def evaluate_replay_for_promotion(radar: Dict[str, Any], global_summary: GlobalHealthSummary) -> Dict[str, Any]:
    """Evaluates the replay posture for promotion gating."""
    reasons = []
    blocking_incidents = []
    
    determinism_rate = radar["determinism_rate_series"][-1]["rate"]
    
    # Rule 1: Hard block for unstable radar
    if radar["radar_status"] == "UNSTABLE":
        status = "BLOCK"
        if determinism_rate < 0.90:
            reasons.append(f"Determinism rate ({determinism_rate:.2%}) is below the 90% threshold.")
        if radar["recurring_critical_incident_fingerprints"]:
            reasons.append("Recurring critical incidents detected.")
            blocking_incidents = sorted(list(radar["recurring_critical_incident_fingerprints"].keys()))
    # Rule 2: Warning for drifting radar or global health warnings
    elif radar["radar_status"] == "DRIFTING" or global_summary.get("status") == "WARN":
        status = "WARN"
        if determinism_rate < 0.95:
            reasons.append(f"Determinism rate ({determinism_rate:.2%}) is below the 95% warning threshold.")
        if global_summary.get("status") == "WARN":
             reasons.append("Global health status is WARN.")
    # Rule 3: OK
    else:
        status = "OK"
        reasons.append("Determinism rate is stable and no critical incidents are recurring.")

    return {
        "replay_promotion_ok": status == "OK",
        "status": status,
        "blocking_incidents": blocking_incidents,
        "reasons": reasons,
    }

# --- TASK 3: Director Replay Panel ---

def build_replay_director_panel(radar: Dict[str, Any], promotion_eval: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a high-level summary panel for director-level review."""
    
    promo_status = promotion_eval["status"]
    if promo_status == "OK":
        status_light = "GREEN"
        headline = "Replay determinism is stable and within acceptable limits."
    elif promo_status == "WARN":
        status_light = "YELLOW"
        headline = "Replay determinism shows signs of drift; caution advised."
    else: # BLOCK
        status_light = "RED"
        headline = "Replay determinism is unstable; promotion is blocked."

    return {
        "status_light": status_light,
        "determinism_rate": f'{radar["determinism_rate_series"][-1]["rate"]:.2%}',
        "critical_incident_count": len(radar.get("recurring_critical_incident_fingerprints", {})),
        "headline": headline,
    }
