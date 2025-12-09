# backend/security/posture.py
"""
GEMINI-K: SIGMA-III - Security Posture Spine

This module defines the canonical data structures and governance logic for
evaluating the determinism and security posture of the U2 compute substrate.
"""

from typing import Dict, Any, Literal, TypedDict

# --- Canonical Data Structures ---

ReplayStatus = Literal["OK", "FAIL", "NOT_RUN"]
SeedClassification = Literal["PURE", "DRIFT", "UNKNOWN"]
LastMileValidation = Literal["PASS", "FAIL", "NOT_RUN"]
SecurityLevel = Literal["GREEN", "AMBER", "RED", "UNKNOWN"]

class SecurityPosture(TypedDict):
    """
    A canonical representation of the system's security posture at a point in time.
    """
    replay_status: ReplayStatus
    seed_classification: SeedClassification
    last_mile_validation: LastMileValidation

# --- Core Posture Construction ---

def build_security_posture(
    is_replay_ok: bool,
    did_seed_drift: bool,
    last_mile_passed: bool
) -> SecurityPosture:
    """
    Constructs a SecurityPosture dictionary from raw boolean checks.
    This function serves as the single source of truth for posture creation.
    """
    return {
        "replay_status": "OK" if is_replay_ok else "FAIL",
        "seed_classification": "PURE" if not did_seed_drift else "DRIFT",
        "last_mile_validation": "PASS" if last_mile_passed else "FAIL",
    }

# --- Governance Logic ---

def is_security_ok(posture: SecurityPosture) -> bool:
    """
    Determines if the overall security posture is acceptable.
    Returns True only if all checks are in a perfect state.
    """
    return (
        posture["replay_status"] == "OK" and
        posture["seed_classification"] == "PURE" and
        posture["last_mile_validation"] == "PASS"
    )

def get_security_level(posture: SecurityPosture) -> SecurityLevel:
    """
    Assigns a traffic-light security level based on the posture.
    This logic defines the risk tolerance of the system.
    """
    if is_security_ok(posture):
        return "GREEN"

    # RED conditions are critical failures that invalidate results.
    if posture["replay_status"] == "FAIL":
        return "RED"
    if posture["seed_classification"] == "DRIFT":
        return "RED"

    # AMBER conditions are warnings that indicate a problem but may not
    # have invalidated the core result.
    if posture["last_mile_validation"] == "FAIL":
        return "AMBER"

    return "UNKNOWN" # Should not be reached with valid inputs

def summarize_security_for_governance(posture: SecurityPosture) -> Dict[str, Any]:
    """
    Creates a human-readable summary suitable for dashboards and reports.
    """
    level = get_security_level(posture)
    summary = {
        "security_level": level,
        "is_ok": is_security_ok(posture),
        "components": posture,
        "narrative": "",
    }

    narratives = {
        "GREEN": "All determinism checks passed. Replay is consistent, seed is pure, and validation holds.",
        "AMBER": "WARNING: Core determinism holds, but last-mile validation failed. Results may be incomplete or fail downstream checks.",
        "RED": "CRITICAL: A fundamental determinism invariant was breached (replay failure or seed drift). Results are invalid.",
        "UNKNOWN": "Posture could not be determined due to an internal error."
    }
    summary["narrative"] = narratives.get(level, "Unknown state.")
    return summary

def merge_into_global_health(
    global_health: Dict[str, Any],
    posture_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merges the security posture summary into a higher-level global health dict.
    """
    # Defensive copy
    updated_health = global_health.copy()
    updated_health.setdefault("subsystems", {})
    updated_health["subsystems"]["security_determinism"] = posture_summary
    
    # Update the global traffic light based on security level precedence
    security_level = posture_summary.get("security_level")
    if security_level == "RED":
        updated_health["traffic_light"] = "RED"
    elif security_level == "AMBER" and updated_health.get("traffic_light") != "RED":
        updated_health["traffic_light"] = "AMBER"
        
    return updated_health

# --- Scenario Analysis ---

SECURITY_SCENARIOS: Dict[str, Dict[str, str]] = {
    "HEALTHY_GREEN": {
        "replay_status": "OK",
        "seed_classification": "PURE",
        "last_mile_validation": "PASS",
    },
    "REPLAY_FAILURE_RED": {
        "replay_status": "FAIL",
    },
    "SEED_DRIFT_RED": {
        "seed_classification": "DRIFT",
    },
    "VALIDATION_FAIL_AMBER": {
        "replay_status": "OK",
        "seed_classification": "PURE",
        "last_mile_validation": "FAIL",
    },
}

def classify_security_scenario(posture: SecurityPosture) -> str:
    """
    Classifies a given posture against the known scenario catalog.

    Args:
        posture: The security posture to classify.

    Returns:
        The matching scenario_id string, or "unknown".
    """
    for scenario_id, pattern in SECURITY_SCENARIOS.items():
        is_match = all(
            posture.get(key) == value for key, value in pattern.items()
        )
        if is_match:
            return scenario_id
    return "unknown"

# --- MAAS Adapter ---

def summarize_security_for_maas_tile(posture: SecurityPosture) -> Dict[str, Any]:
    """
    Emits a small, neutral JSON summary for a MAAS (Monitoring-As-A-Service) tile.
    
    Note: "counts per severity" is an aggregation concern. This function provides
    the severity for a SINGLE posture. The MAAS backend is responsible for
    aggregating these counts over time.
    """
    security_level = get_security_level(posture)
    scenario_id = classify_security_scenario(posture)
    
    is_binding = security_level in ["AMBER", "RED"]
    
    return {
        "dominant_scenario_id": scenario_id,
        "security_level": security_level,
        "is_binding_constraint": is_binding,
    }