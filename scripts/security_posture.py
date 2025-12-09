#!/usr/bin/env python3
"""
security_posture.py - Unified Security Posture Spine

PHASE II -- NOT RUN IN PHASE I

Provides a single consolidated security posture for each U2 run by fusing:
- Replay incident analysis (FULL_MATCH / PARTIAL_MATCH / NO_MATCH)
- Seed drift analysis (SEED_DRIFT / SUBSTRATE_NONDETERMINISM / UNKNOWN)
- Last-mile readiness (READY / CONDITIONAL / NO-GO)

This module defines the security posture contract and governance integration.
All functions are deterministic given identical inputs.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


# =============================================================================
# Schema Version
# =============================================================================

SCHEMA_VERSION = "1.0.0"


# =============================================================================
# Enums for Security Posture
# =============================================================================

class ReplayStatus(Enum):
    FULL_MATCH = "FULL_MATCH"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    NO_MATCH = "NO_MATCH"
    NOT_RUN = "NOT_RUN"


class ReplaySeverity(Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SeedClassification(Enum):
    NO_ISSUE = "NO_ISSUE"
    SEED_DRIFT = "SEED_DRIFT"
    SUBSTRATE_NONDETERMINISM = "SUBSTRATE_NONDETERMINISM"
    UNKNOWN = "UNKNOWN"
    NOT_RUN = "NOT_RUN"


class LastMileStatus(Enum):
    READY = "READY"
    CONDITIONAL = "CONDITIONAL"
    NO_GO = "NO_GO"
    NOT_RUN = "NOT_RUN"


class SecurityLevel(Enum):
    """Governance-level security classification."""
    OK = "OK"
    WARN = "WARN"
    NO_GO = "NO_GO"


# =============================================================================
# Security Posture Data Structure
# =============================================================================

@dataclass
class SecurityPosture:
    """
    Unified security posture for a U2 run.

    This is the single JSON object summarizing security status,
    combining replay, seed, and last-mile analysis.
    """
    schema_version: str
    run_id: str
    generated_at: str

    # Replay Analysis
    replay_status: str
    replay_severity: str
    replay_match_percentage: Optional[float]
    replay_divergence_point: Optional[int]

    # Seed Analysis
    seed_classification: str
    seed_confidence: Optional[str]
    has_seed_drift: bool
    has_substrate_nondeterminism: bool

    # Last-Mile Readiness
    lastmile_status: str
    lastmile_passed: int
    lastmile_total: int
    lastmile_blocking_items: list

    # Consolidated Status
    is_security_ok: bool
    security_level: str
    blocking_reasons: list

    # Metadata
    components_available: dict


def _normalize_replay_status(status: Optional[str]) -> ReplayStatus:
    """Normalize replay status string to enum."""
    if status is None:
        return ReplayStatus.NOT_RUN
    status_upper = status.upper().replace("-", "_").replace(" ", "_")
    try:
        return ReplayStatus(status_upper)
    except ValueError:
        if "FULL" in status_upper or "PASS" in status_upper:
            return ReplayStatus.FULL_MATCH
        elif "PARTIAL" in status_upper:
            return ReplayStatus.PARTIAL_MATCH
        elif "NO_MATCH" in status_upper or "NOMATCH" in status_upper or "FAIL" in status_upper:
            return ReplayStatus.NO_MATCH
        return ReplayStatus.NOT_RUN


def _normalize_severity(severity: Optional[str]) -> ReplaySeverity:
    """Normalize severity string to enum."""
    if severity is None:
        return ReplaySeverity.NONE
    try:
        return ReplaySeverity(severity.upper())
    except ValueError:
        return ReplaySeverity.NONE


def _normalize_seed_classification(classification: Optional[str]) -> SeedClassification:
    """Normalize seed classification string to enum."""
    if classification is None:
        return SeedClassification.NOT_RUN
    classification_upper = classification.upper().replace("-", "_").replace(" ", "_")
    try:
        return SeedClassification(classification_upper)
    except ValueError:
        if "DRIFT" in classification_upper:
            return SeedClassification.SEED_DRIFT
        elif "SUBSTRATE" in classification_upper or "NONDET" in classification_upper:
            return SeedClassification.SUBSTRATE_NONDETERMINISM
        elif "NO_DISAGREEMENT" in classification_upper or "NO_ISSUE" in classification_upper:
            return SeedClassification.NO_ISSUE
        return SeedClassification.UNKNOWN


def _normalize_lastmile_status(status: Optional[str]) -> LastMileStatus:
    """Normalize last-mile status string to enum."""
    if status is None:
        return LastMileStatus.NOT_RUN
    status_upper = status.upper().replace("-", "_").replace(" ", "_")
    try:
        return LastMileStatus(status_upper)
    except ValueError:
        if "READY" in status_upper:
            return LastMileStatus.READY
        elif "CONDITIONAL" in status_upper:
            return LastMileStatus.CONDITIONAL
        elif "NO" in status_upper or "NOT" in status_upper:
            return LastMileStatus.NO_GO
        return LastMileStatus.NOT_RUN


def _determine_security_ok(
    replay_status: ReplayStatus,
    replay_severity: ReplaySeverity,
    seed_classification: SeedClassification,
    lastmile_status: LastMileStatus
) -> tuple[bool, SecurityLevel, list]:
    """
    Determine if security is OK based on all components.

    Returns (is_ok, security_level, blocking_reasons)

    Security is OK when:
    - Replay is FULL_MATCH (or NOT_RUN with no prior expectation)
    - Seed classification is NO_ISSUE or NOT_RUN
    - Last-mile is READY

    Security is WARN when:
    - Replay is PARTIAL_MATCH with acceptable coverage
    - Seed classification is SUBSTRATE_NONDETERMINISM (recoverable)
    - Last-mile is CONDITIONAL

    Security is NO_GO when:
    - Replay is NO_MATCH
    - Seed classification is SEED_DRIFT
    - Last-mile is NO_GO
    - Any CRITICAL severity
    """
    blocking_reasons = []

    # Check replay
    replay_blocking = False
    if replay_status == ReplayStatus.NO_MATCH:
        blocking_reasons.append("Replay failed: NO_MATCH")
        replay_blocking = True
    elif replay_severity == ReplaySeverity.CRITICAL:
        blocking_reasons.append(f"Replay severity: CRITICAL")
        replay_blocking = True

    # Check seed classification
    seed_blocking = False
    if seed_classification == SeedClassification.SEED_DRIFT:
        blocking_reasons.append("Seed drift detected")
        seed_blocking = True

    # Check last-mile
    lastmile_blocking = False
    if lastmile_status == LastMileStatus.NO_GO:
        blocking_reasons.append("Last-mile check: NO_GO")
        lastmile_blocking = True

    # Determine overall security level
    if replay_blocking or seed_blocking or lastmile_blocking:
        return False, SecurityLevel.NO_GO, blocking_reasons

    # Check for warnings
    warning_reasons = []
    if replay_status == ReplayStatus.PARTIAL_MATCH:
        warning_reasons.append("Replay partial match")
    if seed_classification == SeedClassification.SUBSTRATE_NONDETERMINISM:
        warning_reasons.append("Substrate nondeterminism detected")
    if lastmile_status == LastMileStatus.CONDITIONAL:
        warning_reasons.append("Last-mile conditional")
    if seed_classification == SeedClassification.UNKNOWN:
        warning_reasons.append("Seed classification unknown")

    if warning_reasons:
        return True, SecurityLevel.WARN, warning_reasons

    return True, SecurityLevel.OK, []


def build_security_posture(
    replay_incident: Optional[Dict[str, Any]] = None,
    seed_analysis: Optional[Dict[str, Any]] = None,
    lastmile_report: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a unified security posture from component reports.

    This is the main function that fuses replay incident analysis,
    seed drift analysis, and last-mile readiness into a single
    JSON object summarizing security posture.

    Args:
        replay_incident: Output from security_replay_incident.py
        seed_analysis: Output from security_seed_drift_analysis.py
        lastmile_report: Output from lastmile_readiness_check.py
        run_id: Optional run ID override

    Returns:
        Dictionary containing the unified security posture
    """
    # Track which components are available
    components_available = {
        "replay_incident": replay_incident is not None,
        "seed_analysis": seed_analysis is not None,
        "lastmile_report": lastmile_report is not None,
    }

    # Extract run_id
    if run_id is None:
        run_id = (
            (replay_incident or {}).get("run_id") or
            (seed_analysis or {}).get("run_id") or
            (lastmile_report or {}).get("run_id") or
            "unknown"
        )

    # Extract and normalize replay data
    replay_incident = replay_incident or {}
    replay_status = _normalize_replay_status(replay_incident.get("replay_status"))
    replay_severity = _normalize_severity(replay_incident.get("severity"))
    replay_match_percentage = replay_incident.get("match_percentage")
    replay_divergence_point = replay_incident.get("divergence_point")

    # Extract and normalize seed data
    seed_analysis = seed_analysis or {}
    seed_classification_raw = seed_analysis.get("classification")
    seed_classification = _normalize_seed_classification(seed_classification_raw)
    seed_confidence = seed_analysis.get("confidence")

    # Determine seed drift flags
    has_seed_drift = seed_classification == SeedClassification.SEED_DRIFT
    has_substrate_nondeterminism = seed_classification == SeedClassification.SUBSTRATE_NONDETERMINISM

    # Handle NO_DISAGREEMENT as NO_ISSUE
    if seed_classification_raw and "NO_DISAGREEMENT" in seed_classification_raw.upper():
        seed_classification = SeedClassification.NO_ISSUE

    # Extract and normalize last-mile data
    lastmile_report = lastmile_report or {}
    lastmile_status = _normalize_lastmile_status(lastmile_report.get("overall_status"))
    lastmile_passed = lastmile_report.get("total_passed", 0)
    lastmile_total = lastmile_report.get("total_checks", 20)
    lastmile_blocking_items = lastmile_report.get("blocking_items", [])

    # Determine consolidated security status
    is_security_ok, security_level, blocking_reasons = _determine_security_ok(
        replay_status,
        replay_severity,
        seed_classification,
        lastmile_status
    )

    # Build posture object
    posture = SecurityPosture(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),

        replay_status=replay_status.value,
        replay_severity=replay_severity.value,
        replay_match_percentage=replay_match_percentage,
        replay_divergence_point=replay_divergence_point,

        seed_classification=seed_classification.value,
        seed_confidence=seed_confidence,
        has_seed_drift=has_seed_drift,
        has_substrate_nondeterminism=has_substrate_nondeterminism,

        lastmile_status=lastmile_status.value,
        lastmile_passed=lastmile_passed,
        lastmile_total=lastmile_total,
        lastmile_blocking_items=lastmile_blocking_items,

        is_security_ok=is_security_ok,
        security_level=security_level.value,
        blocking_reasons=blocking_reasons,

        components_available=components_available,
    )

    return asdict(posture)


def summarize_security_for_governance(posture: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a minimal signal for global health & MAAS integration.

    This function extracts the essential security signals that can be
    trivially embedded into global_health.json or similar governance
    structures.

    Args:
        posture: Output from build_security_posture()

    Returns:
        Dictionary with minimal governance-ready security summary:
        - security_level: OK/WARN/NO_GO
        - has_seed_drift: bool
        - has_substrate_nondeterminism: bool
        - is_security_ok: bool
        - lastmile_ready: bool
    """
    return {
        "security_level": posture.get("security_level", "NO_GO"),
        "has_seed_drift": posture.get("has_seed_drift", False),
        "has_substrate_nondeterminism": posture.get("has_substrate_nondeterminism", False),
        "is_security_ok": posture.get("is_security_ok", False),
        "lastmile_ready": posture.get("lastmile_status") == "READY",
        "replay_ok": posture.get("replay_status") in ("FULL_MATCH", "NOT_RUN"),
        "blocking_count": len(posture.get("blocking_reasons", [])),
    }


def merge_into_global_health(
    global_health: Dict[str, Any],
    posture: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge security posture summary into a global health object.

    This is a convenience function for MAAS integration that adds
    the governance summary to an existing global_health structure.

    Args:
        global_health: Existing global health dictionary
        posture: Output from build_security_posture()

    Returns:
        Updated global_health with security_posture field
    """
    summary = summarize_security_for_governance(posture)
    global_health["security_posture"] = summary
    return global_health


# =============================================================================
# Phase III: Security Scenario Intelligence & Simulation
# =============================================================================

class SecurityScenario(Enum):
    """
    Canonical security scenario identifiers.

    Each scenario represents a distinct security state pattern
    with specific operational implications.
    """
    # Green scenarios (OK)
    NOMINAL = "NOMINAL"                          # All systems green
    PRISTINE = "PRISTINE"                        # No analysis run yet, assumed clean

    # Yellow scenarios (WARN)
    PARTIAL_REPLAY = "PARTIAL_REPLAY"            # Replay partial match
    SUBSTRATE_VARIANCE = "SUBSTRATE_VARIANCE"    # Substrate nondeterminism detected
    CONDITIONAL_READY = "CONDITIONAL_READY"      # Last-mile conditional
    UNKNOWN_SEED_STATE = "UNKNOWN_SEED_STATE"    # Seed classification unknown
    DEGRADED_COVERAGE = "DEGRADED_COVERAGE"      # Multiple minor issues

    # Red scenarios (NO_GO)
    REPLAY_FAILURE = "REPLAY_FAILURE"            # Replay NO_MATCH
    SEED_DRIFT_DETECTED = "SEED_DRIFT_DETECTED"  # Seed drift confirmed
    LASTMILE_BLOCKED = "LASTMILE_BLOCKED"        # Last-mile NO_GO
    CRITICAL_SEVERITY = "CRITICAL_SEVERITY"      # Critical severity event
    MULTI_FAILURE = "MULTI_FAILURE"              # Multiple blocking issues
    INTEGRITY_COMPROMISED = "INTEGRITY_COMPROMISED"  # Seed drift + replay failure


class DirectorStatus(Enum):
    """Director console status signals."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def classify_security_scenario(posture: Dict[str, Any]) -> str:
    """
    Classify a security posture into a canonical scenario identifier.

    This function applies deterministic rules to map posture patterns
    to scenario labels for operational decision-making.

    Args:
        posture: Output from build_security_posture()

    Returns:
        Scenario identifier string (SecurityScenario value)

    Classification Rules (evaluated in priority order):
    1. INTEGRITY_COMPROMISED: seed_drift AND replay NO_MATCH
    2. MULTI_FAILURE: 2+ blocking reasons
    3. CRITICAL_SEVERITY: replay severity CRITICAL
    4. SEED_DRIFT_DETECTED: has_seed_drift
    5. REPLAY_FAILURE: replay NO_MATCH
    6. LASTMILE_BLOCKED: lastmile NO_GO
    7. DEGRADED_COVERAGE: 2+ warning conditions
    8. PARTIAL_REPLAY: replay PARTIAL_MATCH
    9. SUBSTRATE_VARIANCE: has_substrate_nondeterminism
    10. CONDITIONAL_READY: lastmile CONDITIONAL
    11. UNKNOWN_SEED_STATE: seed UNKNOWN
    12. PRISTINE: all NOT_RUN
    13. NOMINAL: default OK
    """
    # Extract key fields
    replay_status = posture.get("replay_status", "NOT_RUN")
    replay_severity = posture.get("replay_severity", "NONE")
    seed_classification = posture.get("seed_classification", "NOT_RUN")
    lastmile_status = posture.get("lastmile_status", "NOT_RUN")
    has_seed_drift = posture.get("has_seed_drift", False)
    has_substrate_nondet = posture.get("has_substrate_nondeterminism", False)
    blocking_reasons = posture.get("blocking_reasons", [])
    security_level = posture.get("security_level", "NO_GO")

    # Priority 1: INTEGRITY_COMPROMISED (most severe)
    if has_seed_drift and replay_status == "NO_MATCH":
        return SecurityScenario.INTEGRITY_COMPROMISED.value

    # Priority 2: MULTI_FAILURE (only for actual NO_GO blockers)
    if security_level == "NO_GO" and len(blocking_reasons) >= 2:
        return SecurityScenario.MULTI_FAILURE.value

    # Priority 3: CRITICAL_SEVERITY
    if replay_severity == "CRITICAL":
        return SecurityScenario.CRITICAL_SEVERITY.value

    # Priority 4: SEED_DRIFT_DETECTED
    if has_seed_drift:
        return SecurityScenario.SEED_DRIFT_DETECTED.value

    # Priority 5: REPLAY_FAILURE
    if replay_status == "NO_MATCH":
        return SecurityScenario.REPLAY_FAILURE.value

    # Priority 6: LASTMILE_BLOCKED
    if lastmile_status == "NO_GO":
        return SecurityScenario.LASTMILE_BLOCKED.value

    # Count warning conditions for DEGRADED_COVERAGE
    warning_count = 0
    if replay_status == "PARTIAL_MATCH":
        warning_count += 1
    if has_substrate_nondet:
        warning_count += 1
    if lastmile_status == "CONDITIONAL":
        warning_count += 1
    if seed_classification == "UNKNOWN":
        warning_count += 1

    # Priority 7: DEGRADED_COVERAGE
    if warning_count >= 2:
        return SecurityScenario.DEGRADED_COVERAGE.value

    # Priority 8: PARTIAL_REPLAY
    if replay_status == "PARTIAL_MATCH":
        return SecurityScenario.PARTIAL_REPLAY.value

    # Priority 9: SUBSTRATE_VARIANCE
    if has_substrate_nondet:
        return SecurityScenario.SUBSTRATE_VARIANCE.value

    # Priority 10: CONDITIONAL_READY
    if lastmile_status == "CONDITIONAL":
        return SecurityScenario.CONDITIONAL_READY.value

    # Priority 11: UNKNOWN_SEED_STATE
    if seed_classification == "UNKNOWN":
        return SecurityScenario.UNKNOWN_SEED_STATE.value

    # Priority 12: PRISTINE (nothing run yet)
    components = posture.get("components_available", {})
    if not any(components.values()):
        return SecurityScenario.PRISTINE.value

    # Priority 13: NOMINAL (all OK)
    return SecurityScenario.NOMINAL.value


def simulate_security_variation(
    posture: Dict[str, Any],
    variation_spec: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulate a what-if security variation on a posture.

    This function creates a new posture by applying hypothetical
    changes to explore security implications without modifying
    the original posture.

    Args:
        posture: Original posture from build_security_posture()
        variation_spec: Dictionary specifying changes to apply
            Supported keys:
            - replay_status: Override replay status
            - replay_severity: Override replay severity
            - replay_match_percentage: Override match percentage
            - seed_classification: Override seed classification
            - lastmile_status: Override last-mile status
            - lastmile_passed: Override passed count
            - add_blocking_reason: Add a blocking reason
            - clear_blocking_reasons: Clear all blocking reasons

    Returns:
        New posture dictionary with variations applied and
        security status recalculated

    Note:
        This function is deterministic: same inputs always
        produce the same outputs.
    """
    # Deep copy the posture
    import copy
    new_posture = copy.deepcopy(posture)

    # Apply direct field overrides
    direct_overrides = [
        "replay_status",
        "replay_severity",
        "replay_match_percentage",
        "replay_divergence_point",
        "seed_classification",
        "seed_confidence",
        "lastmile_status",
        "lastmile_passed",
        "lastmile_total",
    ]

    for field in direct_overrides:
        if field in variation_spec:
            new_posture[field] = variation_spec[field]

    # Handle special seed classification effects
    if "seed_classification" in variation_spec:
        seed_class = variation_spec["seed_classification"]
        new_posture["has_seed_drift"] = seed_class == "SEED_DRIFT"
        new_posture["has_substrate_nondeterminism"] = seed_class == "SUBSTRATE_NONDETERMINISM"

    # Handle blocking reason modifications
    if variation_spec.get("clear_blocking_reasons"):
        new_posture["blocking_reasons"] = []

    if "add_blocking_reason" in variation_spec:
        if "blocking_reasons" not in new_posture:
            new_posture["blocking_reasons"] = []
        new_posture["blocking_reasons"].append(variation_spec["add_blocking_reason"])

    # Recalculate security status
    replay_status = _normalize_replay_status(new_posture.get("replay_status"))
    replay_severity = _normalize_severity(new_posture.get("replay_severity"))
    seed_classification = _normalize_seed_classification(new_posture.get("seed_classification"))
    lastmile_status = _normalize_lastmile_status(new_posture.get("lastmile_status"))

    is_security_ok, security_level, blocking_reasons = _determine_security_ok(
        replay_status,
        replay_severity,
        seed_classification,
        lastmile_status
    )

    # Merge new blocking reasons with existing (if not cleared)
    if not variation_spec.get("clear_blocking_reasons"):
        # Combine original and new blocking reasons, dedup
        existing = set(new_posture.get("blocking_reasons", []))
        new = set(blocking_reasons)
        blocking_reasons = list(existing | new)

    new_posture["is_security_ok"] = is_security_ok
    new_posture["security_level"] = security_level.value
    new_posture["blocking_reasons"] = blocking_reasons

    # Update scenario classification
    new_posture["scenario"] = classify_security_scenario(new_posture)

    # Mark as simulated
    new_posture["_simulated"] = True
    new_posture["_variation_spec"] = variation_spec

    return new_posture


def map_security_to_director_status(posture: Dict[str, Any]) -> str:
    """
    Map security posture to Director Console status signal.

    This function produces a simple traffic-light signal for
    integration with the Director Console.

    Args:
        posture: Output from build_security_posture()

    Returns:
        Director status: "GREEN", "YELLOW", or "RED"

    Mapping Rules:
    - GREEN: security_level == "OK"
    - YELLOW: security_level == "WARN"
    - RED: security_level == "NO_GO"

    The mapping also considers scenario-specific overrides:
    - INTEGRITY_COMPROMISED always maps to RED
    - PRISTINE maps to GREEN (benefit of the doubt)
    """
    security_level = posture.get("security_level", "NO_GO")

    # Check for scenario-specific overrides
    scenario = classify_security_scenario(posture)

    # INTEGRITY_COMPROMISED is always RED
    if scenario == SecurityScenario.INTEGRITY_COMPROMISED.value:
        return DirectorStatus.RED.value

    # PRISTINE (no data yet) gets benefit of the doubt
    if scenario == SecurityScenario.PRISTINE.value:
        return DirectorStatus.GREEN.value

    # Standard mapping
    if security_level == "OK":
        return DirectorStatus.GREEN.value
    elif security_level == "WARN":
        return DirectorStatus.YELLOW.value
    else:
        return DirectorStatus.RED.value


def get_scenario_description(scenario: str) -> Dict[str, Any]:
    """
    Get detailed description and operational guidance for a scenario.

    Args:
        scenario: Scenario identifier from classify_security_scenario()

    Returns:
        Dictionary with:
        - description: Human-readable description
        - severity: HIGH/MEDIUM/LOW
        - director_status: GREEN/YELLOW/RED
        - recommended_actions: List of suggested actions
    """
    descriptions = {
        SecurityScenario.NOMINAL.value: {
            "description": "All security checks pass. System operating normally.",
            "severity": "LOW",
            "director_status": DirectorStatus.GREEN.value,
            "recommended_actions": ["Continue normal operations"],
        },
        SecurityScenario.PRISTINE.value: {
            "description": "No security analysis run yet. Assumed clean.",
            "severity": "LOW",
            "director_status": DirectorStatus.GREEN.value,
            "recommended_actions": ["Run security analysis before U2 execution"],
        },
        SecurityScenario.PARTIAL_REPLAY.value: {
            "description": "Replay matched partially. Some cycles diverged.",
            "severity": "MEDIUM",
            "director_status": DirectorStatus.YELLOW.value,
            "recommended_actions": [
                "Review divergence point",
                "Run seed drift analysis",
                "Consider re-running affected cycles"
            ],
        },
        SecurityScenario.SUBSTRATE_VARIANCE.value: {
            "description": "Platform nondeterminism detected. Seeds matched but outputs differed.",
            "severity": "MEDIUM",
            "director_status": DirectorStatus.YELLOW.value,
            "recommended_actions": [
                "Check PYTHONHASHSEED setting",
                "Review floating-point operations",
                "Consider platform standardization"
            ],
        },
        SecurityScenario.CONDITIONAL_READY.value: {
            "description": "Last-mile checks conditionally passed. Some items need attention.",
            "severity": "MEDIUM",
            "director_status": DirectorStatus.YELLOW.value,
            "recommended_actions": [
                "Review blocking items",
                "Address conditional failures before production U2"
            ],
        },
        SecurityScenario.UNKNOWN_SEED_STATE.value: {
            "description": "Seed classification could not be determined.",
            "severity": "MEDIUM",
            "director_status": DirectorStatus.YELLOW.value,
            "recommended_actions": [
                "Run full seed drift analysis",
                "Verify manifest integrity"
            ],
        },
        SecurityScenario.DEGRADED_COVERAGE.value: {
            "description": "Multiple minor issues detected. Cumulative risk elevated.",
            "severity": "MEDIUM",
            "director_status": DirectorStatus.YELLOW.value,
            "recommended_actions": [
                "Address individual issues",
                "Consider delaying U2 until resolved"
            ],
        },
        SecurityScenario.REPLAY_FAILURE.value: {
            "description": "Replay failed completely. Run is not reproducible.",
            "severity": "HIGH",
            "director_status": DirectorStatus.RED.value,
            "recommended_actions": [
                "Investigate divergence at cycle 0",
                "Check initialization state",
                "Run may need full invalidation"
            ],
        },
        SecurityScenario.SEED_DRIFT_DETECTED.value: {
            "description": "PRNG seed changed between runs. Determinism violated.",
            "severity": "HIGH",
            "director_status": DirectorStatus.RED.value,
            "recommended_actions": [
                "Identify drift source",
                "Restore correct seed",
                "Re-run experiment"
            ],
        },
        SecurityScenario.LASTMILE_BLOCKED.value: {
            "description": "Last-mile security checks failed. U2 cannot proceed.",
            "severity": "HIGH",
            "director_status": DirectorStatus.RED.value,
            "recommended_actions": [
                "Review all blocking items",
                "Address each failure",
                "Re-run last-mile check"
            ],
        },
        SecurityScenario.CRITICAL_SEVERITY.value: {
            "description": "Critical severity event detected. Immediate attention required.",
            "severity": "HIGH",
            "director_status": DirectorStatus.RED.value,
            "recommended_actions": [
                "Halt any running operations",
                "Preserve artifacts",
                "Investigate root cause"
            ],
        },
        SecurityScenario.MULTI_FAILURE.value: {
            "description": "Multiple blocking failures detected. System integrity at risk.",
            "severity": "HIGH",
            "director_status": DirectorStatus.RED.value,
            "recommended_actions": [
                "Address each blocking reason",
                "Consider full security audit",
                "Do not proceed until all resolved"
            ],
        },
        SecurityScenario.INTEGRITY_COMPROMISED.value: {
            "description": "Seed drift AND replay failure. Run integrity cannot be verified.",
            "severity": "HIGH",
            "director_status": DirectorStatus.RED.value,
            "recommended_actions": [
                "Full run invalidation required",
                "Forensic investigation recommended",
                "Complete re-run needed"
            ],
        },
    }

    return descriptions.get(scenario, {
        "description": f"Unknown scenario: {scenario}",
        "severity": "HIGH",
        "director_status": DirectorStatus.RED.value,
        "recommended_actions": ["Investigate unknown scenario"],
    })


# =============================================================================
# Phase IV: Security Governance Tile & Release Scenario Gate
# =============================================================================

class ReleaseStatus(Enum):
    """Release gate status values."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


class HealthStatus(Enum):
    """Health snapshot status values."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    CRITICAL = "CRITICAL"


# Scenarios that BLOCK release
BLOCKING_SCENARIOS = frozenset([
    SecurityScenario.INTEGRITY_COMPROMISED.value,
    SecurityScenario.MULTI_FAILURE.value,
    SecurityScenario.CRITICAL_SEVERITY.value,
    SecurityScenario.SEED_DRIFT_DETECTED.value,
    SecurityScenario.REPLAY_FAILURE.value,
    SecurityScenario.LASTMILE_BLOCKED.value,
])

# Scenarios that WARN but don't block
WARNING_SCENARIOS = frozenset([
    SecurityScenario.PARTIAL_REPLAY.value,
    SecurityScenario.SUBSTRATE_VARIANCE.value,
    SecurityScenario.CONDITIONAL_READY.value,
    SecurityScenario.UNKNOWN_SEED_STATE.value,
    SecurityScenario.DEGRADED_COVERAGE.value,
])

# Scenarios that are OK for release
OK_SCENARIOS = frozenset([
    SecurityScenario.NOMINAL.value,
    SecurityScenario.PRISTINE.value,
])


def evaluate_security_for_release(
    scenario_label: str,
    posture: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate whether security posture allows release.

    This is a hard gating function that determines if a release
    can proceed based on the security scenario and posture.

    Args:
        scenario_label: Scenario from classify_security_scenario()
        posture: Output from build_security_posture()

    Returns:
        Dictionary with:
        - release_ok: bool - True if release can proceed
        - status: "OK" | "WARN" | "BLOCK"
        - blocking_reasons: list[str] - Reasons for blocking/warning
        - scenario: The evaluated scenario
        - recommendation: Human-readable recommendation

    Release Rules:
    - BLOCK: INTEGRITY_COMPROMISED, MULTI_FAILURE, CRITICAL_SEVERITY,
             SEED_DRIFT_DETECTED, REPLAY_FAILURE, LASTMILE_BLOCKED
    - WARN: PARTIAL_REPLAY, SUBSTRATE_VARIANCE, CONDITIONAL_READY,
            UNKNOWN_SEED_STATE, DEGRADED_COVERAGE
    - OK: NOMINAL, PRISTINE
    """
    blocking_reasons = []

    # Determine status based on scenario
    if scenario_label in BLOCKING_SCENARIOS:
        status = ReleaseStatus.BLOCK
        release_ok = False

        # Build blocking reasons from posture
        posture_reasons = posture.get("blocking_reasons", [])
        if posture_reasons:
            blocking_reasons.extend(posture_reasons)

        # Add scenario-specific reasons
        scenario_desc = get_scenario_description(scenario_label)
        blocking_reasons.append(f"Scenario {scenario_label}: {scenario_desc['description']}")

    elif scenario_label in WARNING_SCENARIOS:
        status = ReleaseStatus.WARN
        release_ok = True  # Can proceed with caution

        # Add warning reasons
        scenario_desc = get_scenario_description(scenario_label)
        blocking_reasons.append(f"Warning: {scenario_desc['description']}")

        # Include any posture warnings
        posture_reasons = posture.get("blocking_reasons", [])
        if posture_reasons:
            blocking_reasons.extend(posture_reasons)

    else:
        # OK scenarios (NOMINAL, PRISTINE) or unknown
        status = ReleaseStatus.OK
        release_ok = True
        # No blocking reasons for OK status

    # Generate recommendation
    if status == ReleaseStatus.BLOCK:
        recommendation = "Release blocked. Address all blocking issues before proceeding."
    elif status == ReleaseStatus.WARN:
        recommendation = "Release permitted with caution. Review warnings before production."
    else:
        recommendation = "Release approved. All security checks pass."

    return {
        "release_ok": release_ok,
        "status": status.value,
        "blocking_reasons": blocking_reasons,
        "scenario": scenario_label,
        "recommendation": recommendation,
    }


def _get_scenario_severity_tier(scenario: str) -> str:
    """
    Get the severity tier for a scenario.

    Returns: "critical", "warning", or "ok"
    """
    if scenario in BLOCKING_SCENARIOS:
        return "critical"
    elif scenario in WARNING_SCENARIOS:
        return "warning"
    else:
        return "ok"


def build_security_scenario_health_snapshot(
    scenarios: list
) -> Dict[str, Any]:
    """
    Build a health snapshot from a collection of security scenarios.

    This function aggregates multiple scenario observations to produce
    an overall health assessment for monitoring dashboards.

    Args:
        scenarios: List of scenario labels from classify_security_scenario()

    Returns:
        Dictionary with:
        - scenario_counts_by_severity: {"critical": N, "warning": N, "ok": N}
        - dominant_scenario: Most common or most severe scenario
        - status: "OK" | "ATTENTION" | "CRITICAL"
        - total_scenarios: Total count analyzed
        - breakdown: List of (scenario, count) tuples

    Status Rules:
    - CRITICAL: Any critical scenario present
    - ATTENTION: Any warning scenario present (no critical)
    - OK: All scenarios are OK
    """
    if not scenarios:
        return {
            "scenario_counts_by_severity": {"critical": 0, "warning": 0, "ok": 0},
            "dominant_scenario": None,
            "status": HealthStatus.OK.value,
            "total_scenarios": 0,
            "breakdown": [],
        }

    # Count scenarios by severity tier
    severity_counts = {"critical": 0, "warning": 0, "ok": 0}
    scenario_counts: Dict[str, int] = {}

    for scenario in scenarios:
        tier = _get_scenario_severity_tier(scenario)
        severity_counts[tier] += 1

        if scenario not in scenario_counts:
            scenario_counts[scenario] = 0
        scenario_counts[scenario] += 1

    # Determine dominant scenario
    # Priority: most frequent, with tie-breaker by severity
    def scenario_sort_key(item):
        scenario, count = item
        # Higher count = more dominant
        # Higher severity = more dominant (critical > warning > ok)
        severity_order = {"critical": 0, "warning": 1, "ok": 2}
        tier = _get_scenario_severity_tier(scenario)
        return (-count, severity_order.get(tier, 3), scenario)

    sorted_scenarios = sorted(scenario_counts.items(), key=scenario_sort_key)
    dominant_scenario = sorted_scenarios[0][0] if sorted_scenarios else None

    # Determine overall status
    if severity_counts["critical"] > 0:
        status = HealthStatus.CRITICAL
    elif severity_counts["warning"] > 0:
        status = HealthStatus.ATTENTION
    else:
        status = HealthStatus.OK

    # Build breakdown
    breakdown = [(s, c) for s, c in sorted_scenarios]

    return {
        "scenario_counts_by_severity": severity_counts,
        "dominant_scenario": dominant_scenario,
        "status": status.value,
        "total_scenarios": len(scenarios),
        "breakdown": breakdown,
    }


def build_security_director_panel(
    health_snapshot: Dict[str, Any],
    release_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a Director Console security panel.

    This function combines health snapshot and release evaluation
    into a unified panel suitable for display in the Director Console.

    Args:
        health_snapshot: Output from build_security_scenario_health_snapshot()
        release_eval: Output from evaluate_security_for_release()

    Returns:
        Dictionary with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - dominant_scenario: Primary scenario driving status
        - headline: Neutral sentence summarizing security posture
        - release_status: The release gate status
        - metrics: Key metrics for display

    Status Light Rules:
    - RED: release blocked OR health CRITICAL
    - YELLOW: release warned OR health ATTENTION
    - GREEN: all clear
    """
    release_status = release_eval.get("status", "BLOCK")
    health_status = health_snapshot.get("status", "CRITICAL")
    dominant_scenario = health_snapshot.get("dominant_scenario")
    release_scenario = release_eval.get("scenario")

    # Determine status light
    if release_status == "BLOCK" or health_status == "CRITICAL":
        status_light = DirectorStatus.RED.value
    elif release_status == "WARN" or health_status == "ATTENTION":
        status_light = DirectorStatus.YELLOW.value
    else:
        status_light = DirectorStatus.GREEN.value

    # Choose the most relevant scenario for display
    # Prefer release scenario if blocking, otherwise dominant
    display_scenario = release_scenario if release_status == "BLOCK" else dominant_scenario

    # Generate neutral headline
    headline = _generate_security_headline(
        status_light,
        display_scenario,
        health_snapshot,
        release_eval
    )

    # Build metrics
    severity_counts = health_snapshot.get("scenario_counts_by_severity", {})
    metrics = {
        "critical_count": severity_counts.get("critical", 0),
        "warning_count": severity_counts.get("warning", 0),
        "ok_count": severity_counts.get("ok", 0),
        "total_scenarios": health_snapshot.get("total_scenarios", 0),
        "release_ok": release_eval.get("release_ok", False),
        "blocking_reason_count": len(release_eval.get("blocking_reasons", [])),
    }

    return {
        "status_light": status_light,
        "dominant_scenario": display_scenario,
        "headline": headline,
        "release_status": release_status,
        "metrics": metrics,
    }


def _generate_security_headline(
    status_light: str,
    scenario: Optional[str],
    health_snapshot: Dict[str, Any],
    release_eval: Dict[str, Any]
) -> str:
    """
    Generate a neutral headline for the security panel.

    The headline is factual and non-alarmist, suitable for
    executive dashboards.
    """
    release_status = release_eval.get("status", "BLOCK")
    total = health_snapshot.get("total_scenarios", 0)
    severity_counts = health_snapshot.get("scenario_counts_by_severity", {})
    critical = severity_counts.get("critical", 0)
    warning = severity_counts.get("warning", 0)

    if status_light == DirectorStatus.RED.value:
        if release_status == "BLOCK":
            return f"Security gate blocked. {scenario or 'Issue'} requires resolution."
        else:
            return f"{critical} critical security scenario(s) detected."
    elif status_light == DirectorStatus.YELLOW.value:
        if warning > 0:
            return f"{warning} security warning(s) under review."
        else:
            return "Security posture requires attention."
    else:
        if total == 0:
            return "No security scenarios analyzed."
        elif scenario == SecurityScenario.PRISTINE.value:
            return "Security analysis pending. System ready for checks."
        else:
            return "All security checks passed."


# =============================================================================
# Phase V: Security as Cross-Cutting Constraint
# =============================================================================

class CompositeStatus(Enum):
    """Composite status for cross-cutting security view."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


def build_security_replay_ht_view(
    security_scenario: str,
    replay_status: str,
    ht_status: str
) -> Dict[str, Any]:
    """
    Build a composite view coupling security, replay, and HT status.

    This function creates a unified constraint view that considers
    security scenarios alongside replay and hash-chain (HT) status.
    Security acts as a cross-cutting constraint that can BLOCK even
    when replay and HT are OK.

    Args:
        security_scenario: Scenario from classify_security_scenario()
        replay_status: Replay status ("OK", "WARN", "FAIL", "NOT_RUN")
        ht_status: Hash-chain status ("OK", "WARN", "FAIL", "NOT_RUN")

    Returns:
        Dictionary with:
        - composite_status: "OK" | "WARN" | "BLOCK"
        - blocking_reasons: list[str] - All reasons for blocking/warning
        - scenarios_implicated: list[str] - Scenarios contributing to status
        - security_is_constraint: bool - True if security drove the status
        - replay_ok: bool
        - ht_ok: bool

    Constraint Rules:
    - Security BLOCK overrides replay/HT OK
    - Security WARN downgrades replay/HT OK to WARN
    - Any FAIL in replay/HT contributes to BLOCK
    - All three must be OK for composite OK
    """
    blocking_reasons = []
    scenarios_implicated = []

    # Normalize inputs
    replay_status = replay_status.upper() if replay_status else "NOT_RUN"
    ht_status = ht_status.upper() if ht_status else "NOT_RUN"

    # Determine security constraint level
    security_is_block = security_scenario in BLOCKING_SCENARIOS
    security_is_warn = security_scenario in WARNING_SCENARIOS
    security_is_ok = security_scenario in OK_SCENARIOS or security_scenario is None

    # Track if security drove the status
    security_is_constraint = False

    # Check replay status
    replay_ok = replay_status in ("OK", "PASS", "FULL_MATCH", "NOT_RUN")
    replay_fail = replay_status in ("FAIL", "NO_MATCH", "BLOCK")
    replay_warn = replay_status in ("WARN", "PARTIAL_MATCH", "PARTIAL")

    # Check HT status
    ht_ok = ht_status in ("OK", "PASS", "VERIFIED", "NOT_RUN")
    ht_fail = ht_status in ("FAIL", "INVALID", "BLOCK", "TAMPERED")
    ht_warn = ht_status in ("WARN", "PARTIAL", "UNVERIFIED")

    # Add implicated scenarios
    if security_scenario:
        scenarios_implicated.append(security_scenario)

    # Determine composite status with security as cross-cutting constraint

    # Priority 1: Any BLOCK-level issue
    if security_is_block:
        composite_status = CompositeStatus.BLOCK
        security_is_constraint = True
        scenario_desc = get_scenario_description(security_scenario)
        blocking_reasons.append(f"Security scenario {security_scenario}: {scenario_desc['description']}")

    if replay_fail:
        composite_status = CompositeStatus.BLOCK
        blocking_reasons.append(f"Replay status: {replay_status}")
        scenarios_implicated.append("REPLAY_FAILURE")

    if ht_fail:
        composite_status = CompositeStatus.BLOCK
        blocking_reasons.append(f"Hash-chain status: {ht_status}")
        scenarios_implicated.append("HT_INTEGRITY_FAILURE")

    # If we already have a BLOCK, return early
    if blocking_reasons:
        return {
            "composite_status": CompositeStatus.BLOCK.value,
            "blocking_reasons": blocking_reasons,
            "scenarios_implicated": list(set(scenarios_implicated)),
            "security_is_constraint": security_is_constraint,
            "replay_ok": replay_ok,
            "ht_ok": ht_ok,
        }

    # Priority 2: Any WARN-level issue
    warn_reasons = []

    if security_is_warn:
        security_is_constraint = True
        scenario_desc = get_scenario_description(security_scenario)
        warn_reasons.append(f"Security warning: {scenario_desc['description']}")

    if replay_warn:
        warn_reasons.append(f"Replay warning: {replay_status}")

    if ht_warn:
        warn_reasons.append(f"Hash-chain warning: {ht_status}")

    if warn_reasons:
        return {
            "composite_status": CompositeStatus.WARN.value,
            "blocking_reasons": warn_reasons,
            "scenarios_implicated": list(set(scenarios_implicated)),
            "security_is_constraint": security_is_constraint,
            "replay_ok": replay_ok,
            "ht_ok": ht_ok,
        }

    # Priority 3: All OK
    return {
        "composite_status": CompositeStatus.OK.value,
        "blocking_reasons": [],
        "scenarios_implicated": scenarios_implicated,
        "security_is_constraint": False,
        "replay_ok": replay_ok,
        "ht_ok": ht_ok,
    }


def summarize_security_for_global_console(
    health_snapshot: Dict[str, Any],
    release_eval: Dict[str, Any],
    composite_view: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize security for the global console integration.

    This function produces a unified security summary suitable for
    embedding in the global console's health view. It combines
    health snapshot, release evaluation, and composite replay/HT view.

    Args:
        health_snapshot: Output from build_security_scenario_health_snapshot()
        release_eval: Output from evaluate_security_for_release()
        composite_view: Output from build_security_replay_ht_view()

    Returns:
        Dictionary with:
        - security_ok: bool - Master security signal
        - status_light: "GREEN" | "YELLOW" | "RED"
        - dominant_scenario: Primary scenario driving status
        - headline: Neutral summary sentence
        - composite_status: From replay/HT coupling
        - is_hard_constraint: True if security is blocking

    This is the primary integration point for CLAUDE I's meta-view.
    """
    # Determine master security signal
    release_ok = release_eval.get("release_ok", False)
    composite_status = composite_view.get("composite_status", "BLOCK")
    health_status = health_snapshot.get("status", "CRITICAL")

    # Security is OK only if all three are OK
    security_ok = (
        release_ok and
        composite_status == "OK" and
        health_status == "OK"
    )

    # Determine status light (most restrictive wins)
    if composite_status == "BLOCK" or health_status == "CRITICAL" or not release_ok:
        status_light = DirectorStatus.RED.value
    elif composite_status == "WARN" or health_status == "ATTENTION":
        status_light = DirectorStatus.YELLOW.value
    else:
        status_light = DirectorStatus.GREEN.value

    # Get dominant scenario
    dominant_scenario = (
        release_eval.get("scenario") or
        health_snapshot.get("dominant_scenario") or
        (composite_view.get("scenarios_implicated", [None])[0] if composite_view.get("scenarios_implicated") else None)
    )

    # Generate headline
    headline = _generate_global_console_headline(
        security_ok,
        status_light,
        dominant_scenario,
        composite_view
    )

    # Is security acting as a hard constraint?
    is_hard_constraint = (
        composite_view.get("security_is_constraint", False) or
        release_eval.get("status") == "BLOCK"
    )

    return {
        "security_ok": security_ok,
        "status_light": status_light,
        "dominant_scenario": dominant_scenario,
        "headline": headline,
        "composite_status": composite_status,
        "is_hard_constraint": is_hard_constraint,
        "blocking_reasons": composite_view.get("blocking_reasons", []) + release_eval.get("blocking_reasons", []),
    }


def _generate_global_console_headline(
    security_ok: bool,
    status_light: str,
    scenario: Optional[str],
    composite_view: Dict[str, Any]
) -> str:
    """Generate headline for global console security summary."""
    composite_status = composite_view.get("composite_status", "BLOCK")
    security_is_constraint = composite_view.get("security_is_constraint", False)

    if status_light == DirectorStatus.RED.value:
        if security_is_constraint:
            return f"Security constraint active. {scenario or 'Issue'} blocking pipeline."
        else:
            return "Pipeline blocked. Security and/or integrity issues detected."
    elif status_light == DirectorStatus.YELLOW.value:
        if security_is_constraint:
            return f"Security warning. {scenario or 'Issue'} requires attention."
        else:
            return "Pipeline proceeding with caution. Review warnings."
    else:
        return "Security clear. Pipeline may proceed."


# =============================================================================
# Phase VI: Security as Meta-Governance Layer
# =============================================================================

# GovernanceSignal Schema Version
GOVERNANCE_SIGNAL_SCHEMA_VERSION = "1.0.0"


@dataclass
class GovernanceSignal:
    """
    Formal schema for security governance signals consumed by CLAUDE I.

    This is the canonical contract between CLAUDE K (security) and CLAUDE I
    (meta-governance). All fields are required and typed.

    SECURITY PRIORITY OVERRIDE RULE:
    ================================
    Any signal with status="RED" MUST force global_status=BLOCK in
    CLAUDE I's synthesizer. Security can override everything else.

    This rule is non-negotiable and enforced at the schema level via
    the `forces_global_block` field.
    """
    # Schema metadata
    schema_version: str
    signal_type: str  # Always "SECURITY_POSTURE"

    # Identity
    run_id: str
    timestamp: str  # ISO 8601 format

    # Core security signals
    security_ok: bool           # Master security signal (True = all clear)
    status: str                 # "GREEN" | "YELLOW" | "RED"
    dominant_scenario: Optional[str]  # Primary scenario driving status

    # Governance integration
    is_blocking: bool           # True if security is blocking pipeline
    forces_global_block: bool   # CRITICAL: True if CLAUDE I must set global_status=BLOCK

    # Human-readable
    summary: str                # One-line summary for dashboards

    # Extended context
    blocking_reasons: list      # List of blocking/warning reasons
    scenarios_implicated: list  # All scenarios contributing to status

    # Metadata for debugging/audit
    metadata: Dict[str, Any]


def to_governance_signal(
    global_console_summary: Dict[str, Any],
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert security summary to governance signal for CLAUDE I meta-view.

    This function produces a GovernanceSignal-compliant dictionary that
    CLAUDE I's synthesizer will consume to determine global pipeline status.

    SECURITY PRIORITY OVERRIDE RULE:
    ================================
    If status == "RED", then forces_global_block == True.
    CLAUDE I MUST honor this by setting global_status = "BLOCK".
    Security can override ALL other signals (replay, HT, health, etc.).

    Args:
        global_console_summary: Output from summarize_security_for_global_console()
        run_id: Optional run identifier
        timestamp: Optional timestamp (ISO format)

    Returns:
        GovernanceSignal-compliant dictionary with all required fields.

    Schema Contract:
        - schema_version: "1.0.0"
        - signal_type: "SECURITY_POSTURE"
        - run_id: string
        - timestamp: ISO 8601 string
        - security_ok: bool
        - status: "GREEN" | "YELLOW" | "RED"
        - dominant_scenario: string | null
        - is_blocking: bool
        - forces_global_block: bool (True iff status == "RED")
        - summary: string
        - blocking_reasons: list[string]
        - scenarios_implicated: list[string]
        - metadata: object
    """
    from datetime import datetime, timezone

    status = global_console_summary.get("status_light", "RED")

    # SECURITY PRIORITY OVERRIDE: RED status forces global block
    forces_global_block = (status == "RED")

    # Build scenarios implicated list
    scenarios_implicated = []
    dominant = global_console_summary.get("dominant_scenario")
    if dominant:
        scenarios_implicated.append(dominant)

    return {
        # Schema metadata
        "schema_version": GOVERNANCE_SIGNAL_SCHEMA_VERSION,
        "signal_type": "SECURITY_POSTURE",

        # Identity
        "run_id": run_id or "unknown",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),

        # Core security signals
        "security_ok": global_console_summary.get("security_ok", False),
        "status": status,
        "dominant_scenario": dominant,

        # Governance integration
        "is_blocking": global_console_summary.get("is_hard_constraint", False),
        "forces_global_block": forces_global_block,

        # Human-readable
        "summary": global_console_summary.get("headline", "Security status unknown"),

        # Extended context
        "blocking_reasons": global_console_summary.get("blocking_reasons", []),
        "scenarios_implicated": scenarios_implicated,

        # Metadata
        "metadata": {
            "composite_status": global_console_summary.get("composite_status"),
            "blocking_reason_count": len(global_console_summary.get("blocking_reasons", [])),
            "source": "CLAUDE_K_SECURITY_POSTURE",
        },
    }


def validate_governance_signal(signal: Dict[str, Any]) -> tuple:
    """
    Validate a governance signal against the schema.

    Args:
        signal: Dictionary to validate

    Returns:
        Tuple of (is_valid: bool, errors: list[str])
    """
    errors = []

    required_fields = [
        "schema_version", "signal_type", "run_id", "timestamp",
        "security_ok", "status", "is_blocking", "forces_global_block",
        "summary", "blocking_reasons", "scenarios_implicated", "metadata"
    ]

    for field in required_fields:
        if field not in signal:
            errors.append(f"Missing required field: {field}")

    if signal.get("signal_type") != "SECURITY_POSTURE":
        errors.append(f"Invalid signal_type: {signal.get('signal_type')}")

    if signal.get("status") not in ("GREEN", "YELLOW", "RED"):
        errors.append(f"Invalid status: {signal.get('status')}")

    # Enforce SECURITY PRIORITY OVERRIDE rule
    if signal.get("status") == "RED" and not signal.get("forces_global_block"):
        errors.append("VIOLATION: status=RED must have forces_global_block=True")

    if not isinstance(signal.get("security_ok"), bool):
        errors.append(f"security_ok must be bool, got {type(signal.get('security_ok'))}")

    if not isinstance(signal.get("blocking_reasons"), list):
        errors.append("blocking_reasons must be a list")

    return len(errors) == 0, errors


def apply_security_override_to_global_status(
    current_global_status: str,
    security_signal: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply security override rule to determine final global status.

    SECURITY PRIORITY OVERRIDE RULE:
    ================================
    If security_signal.forces_global_block == True:
        global_status = "BLOCK" (regardless of current_global_status)

    This function is intended for use by CLAUDE I's synthesizer.

    Args:
        current_global_status: Current global status from other signals
                               ("OK", "WARN", "BLOCK")
        security_signal: Output from to_governance_signal()

    Returns:
        Dictionary with:
        - global_status: Final status after security override
        - security_overrode: True if security forced the status
        - original_status: What status would have been without override
        - reason: Explanation of decision

    Example:
        # Even if all other signals are OK, security RED forces BLOCK
        result = apply_security_override_to_global_status("OK", security_signal)
        if result["security_overrode"]:
            print(f"Security forced BLOCK: {result['reason']}")
    """
    forces_block = security_signal.get("forces_global_block", False)
    security_status = security_signal.get("status", "RED")

    if forces_block:
        return {
            "global_status": "BLOCK",
            "security_overrode": True,
            "original_status": current_global_status,
            "reason": f"Security status {security_status} forces global BLOCK. "
                      f"Scenario: {security_signal.get('dominant_scenario', 'unknown')}. "
                      f"{security_signal.get('summary', '')}",
        }

    # Security YELLOW should at minimum produce WARN
    if security_status == "YELLOW" and current_global_status == "OK":
        return {
            "global_status": "WARN",
            "security_overrode": True,
            "original_status": current_global_status,
            "reason": f"Security status YELLOW downgrades OK to WARN. "
                      f"{security_signal.get('summary', '')}",
        }

    # No override needed
    return {
        "global_status": current_global_status,
        "security_overrode": False,
        "original_status": current_global_status,
        "reason": "Security status compatible with current global status.",
    }


# =============================================================================
# Documentation: Security Priority Override Rule
# =============================================================================

SECURITY_OVERRIDE_RULE_DOC = """
================================================================================
SECURITY PRIORITY OVERRIDE RULE
================================================================================

RULE: Any SECURITY RED must force global_status=BLOCK in CLAUDE I's synthesizer.

RATIONALE:
    Security is a meta-governance layer that can override all other signals.
    This ensures that security concerns (seed drift, replay failures, integrity
    compromises) cannot be bypassed by favorable readings from other components.

IMPLEMENTATION:
    1. CLAUDE K emits GovernanceSignal with `forces_global_block` field
    2. If status == "RED", forces_global_block is automatically True
    3. CLAUDE I's synthesizer MUST check forces_global_block FIRST
    4. If forces_global_block == True, set global_status = "BLOCK"

SCENARIOS THAT TRIGGER RED (and thus global BLOCK):
    - INTEGRITY_COMPROMISED: Seed drift AND replay failure
    - MULTI_FAILURE: Multiple blocking issues detected
    - CRITICAL_SEVERITY: Critical severity event
    - SEED_DRIFT_DETECTED: PRNG seed changed between runs
    - REPLAY_FAILURE: Replay failed completely
    - LASTMILE_BLOCKED: Last-mile security checks failed

INTEGRATION CODE FOR CLAUDE I:
    ```python
    def synthesize_global_status(signals: List[Dict]) -> str:
        # Find security signal
        security_signal = next(
            (s for s in signals if s.get("signal_type") == "SECURITY_POSTURE"),
            None
        )

        # SECURITY OVERRIDE CHECK - MUST BE FIRST
        if security_signal and security_signal.get("forces_global_block"):
            return "BLOCK"  # Security overrides everything

        # ... rest of synthesis logic
    ```

AUDIT TRAIL:
    All security overrides are logged with:
    - Original status that would have been set
    - Security scenario that triggered override
    - Full blocking reasons
    - Timestamp

NON-NEGOTIABLE:
    This rule cannot be disabled, bypassed, or weakened.
    Security RED = Global BLOCK. Always.

================================================================================
"""


def get_security_override_rule_doc() -> str:
    """Return the security override rule documentation."""
    return SECURITY_OVERRIDE_RULE_DOC
