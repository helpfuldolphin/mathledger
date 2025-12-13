"""Structural Drill Adapter for First-Light Status and GGFL Integration.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The structural_drill signal is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this signal
- Advisory only: no gating, no enforcement

See docs/system_law/P5_Structural_Drill_Package.md
"""

from typing import Any, Dict, List, Optional, Tuple

STRUCTURAL_DRILL_SCHEMA_VERSION = "1.1.0"  # Bumped for extraction_source + reason codes

# Extraction source constants
EXTRACTION_SOURCE_MANIFEST = "MANIFEST"
EXTRACTION_SOURCE_EVIDENCE_JSON = "EVIDENCE_JSON"
EXTRACTION_SOURCE_MISSING = "MISSING"

# GGFL Reason codes (deterministic, max 3)
REASON_WORST_SEVERITY_CRITICAL = "DRIVER_WORST_SEVERITY_CRITICAL"
REASON_MAX_STREAK_GE_2 = "DRIVER_MAX_STREAK_GE_2"
REASON_DRILL_SUCCESS_FALSE = "DRIVER_DRILL_SUCCESS_FALSE"


def extract_structural_drill_signal(
    drill_reference: Dict[str, Any],
    extraction_source: str = EXTRACTION_SOURCE_MISSING,
) -> Dict[str, Any]:
    """
    Extract compact structural drill signal for status output.

    SHADOW MODE CONTRACT:
    - Purely observational, no side effects
    - Advisory signal only, no gating

    Args:
        drill_reference: Structural drill reference dict from manifest
            Expected keys: drill_id, scenario_id, sha256, drill_success,
                          max_streak, break_events, pattern_counts, mode
        extraction_source: Source of extraction (MANIFEST|EVIDENCE_JSON|MISSING)

    Returns:
        Dict with: drill_success, max_streak, worst_severity, schema_version,
                   mode, extraction_source
        Returns empty dict if drill_reference is None/empty (explicit optional)
    """
    if not drill_reference:
        return {}

    # Extract core fields
    drill_success = drill_reference.get("drill_success", False)
    max_streak = drill_reference.get("max_streak", 0)
    pattern_counts = drill_reference.get("pattern_counts", {})

    # Determine worst_severity from pattern_counts
    # Priority: STRUCTURAL_BREAK (CRITICAL) > DRIFT (WARN) > NONE (INFO)
    worst_severity = "INFO"
    if pattern_counts.get("STRUCTURAL_BREAK", 0) > 0:
        worst_severity = "CRITICAL"
    elif pattern_counts.get("DRIFT", 0) > 0:
        worst_severity = "WARN"

    return {
        "drill_success": drill_success,
        "max_streak": max_streak,
        "worst_severity": worst_severity,
        "schema_version": STRUCTURAL_DRILL_SCHEMA_VERSION,
        "mode": "SHADOW",
        "extraction_source": extraction_source,
    }


def extract_drill_reason_codes(
    drill_reference: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Extract deterministic reason codes from structural drill reference.

    SHADOW MODE CONTRACT:
    - Pure function, no side effects
    - Advisory classification only, no gating

    Reason codes (deterministic ordering, max 3):
    1. DRIVER_WORST_SEVERITY_CRITICAL (if worst_severity == CRITICAL)
    2. DRIVER_MAX_STREAK_GE_2 (if max_streak >= 2)
    3. DRIVER_DRILL_SUCCESS_FALSE (if drill_success == False)

    Args:
        drill_reference: Structural drill reference dict from manifest, or None

    Returns:
        List of reason code strings, max 3 items. Empty list if drill absent.
    """
    if not drill_reference:
        return []

    reason_codes: List[str] = []

    # Extract fields
    drill_success = drill_reference.get("drill_success", False)
    max_streak = drill_reference.get("max_streak", 0)
    pattern_counts = drill_reference.get("pattern_counts", {})

    # Derive worst_severity
    worst_severity = derive_worst_severity_from_pattern_counts(pattern_counts)

    # Reason code 1: CRITICAL severity
    if worst_severity == "CRITICAL":
        reason_codes.append(REASON_WORST_SEVERITY_CRITICAL)

    # Reason code 2: max_streak >= 2
    if max_streak >= 2:
        reason_codes.append(REASON_MAX_STREAK_GE_2)

    # Reason code 3: drill_success == False
    if not drill_success:
        reason_codes.append(REASON_DRILL_SUCCESS_FALSE)

    # Cap at 3 reason codes
    return reason_codes[:3]


def structural_drill_for_alignment_view(
    drill_reference: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Map structural drill signal to GGFL alignment view.

    SHADOW MODE CONTRACT:
    - Purely observational, no side effects
    - Advisory classification only, no gating
    - This function NEVER influences control flow

    Mapping:
        drill absent -> None (explicit optional, no false health surface)
        worst_severity == "INFO" -> "healthy" (no structural issues detected)
        worst_severity == "WARN" -> "degraded" (drift patterns observed)
        worst_severity == "CRITICAL" -> "unhealthy" (STRUCTURAL_BREAK detected)

    Args:
        drill_reference: Structural drill reference dict from manifest, or None

    Returns:
        None if drill artifact is absent (explicit optional).
        Otherwise Dict with:
            - alignment: "healthy" | "degraded" | "unhealthy"
            - advisory: str describing the classification
            - reason_codes: List[str] deterministic reason codes (max 3)
            - mode: "SHADOW" (always)
    """
    # Handle missing artifact: return None (explicit optional, no false health)
    if not drill_reference:
        return None

    # Extract signal
    signal = extract_structural_drill_signal(drill_reference)
    worst_severity = signal.get("worst_severity", "INFO")
    max_streak = signal.get("max_streak", 0)
    drill_success = signal.get("drill_success", False)

    # Extract deterministic reason codes
    reason_codes = extract_drill_reason_codes(drill_reference)

    # Map worst_severity to alignment
    if worst_severity == "CRITICAL":
        alignment = "unhealthy"
        advisory = (
            f"Structural drill recorded STRUCTURAL_BREAK pattern (max_streak={max_streak}). "
            f"Informational: SI-001/SI-010 conditions logged during stress test."
        )
    elif worst_severity == "WARN":
        alignment = "degraded"
        advisory = (
            f"Structural drill recorded DRIFT pattern. "
            f"Informational: SI-006 conditions logged during stress test."
        )
    else:
        alignment = "healthy"
        if drill_success:
            advisory = "Structural drill completed with expected patterns."
        else:
            advisory = (
                "Structural drill completed without STRUCTURAL_BREAK events. "
                "Expected patterns may not have been triggered."
            )

    return {
        "alignment": alignment,
        "advisory": advisory,
        "reason_codes": reason_codes,
        "mode": "SHADOW",
    }


def extract_structural_drill_from_sources(
    pack_manifest: Optional[Dict[str, Any]],
    evidence_data: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract structural drill reference with manifest-first precedence.

    SHADOW MODE CONTRACT:
    - Pure function, no side effects
    - Manifest-first: try pack_manifest["governance"]["structure"]["drill"] first
    - Fallback: evidence_data["governance"]["structure"]["drill"]

    Args:
        pack_manifest: Evidence pack manifest dict, or None
        evidence_data: Evidence.json data dict, or None

    Returns:
        Tuple of (drill_reference, extraction_source):
        - drill_reference: Dict if found, None otherwise
        - extraction_source: MANIFEST | EVIDENCE_JSON | MISSING
    """
    # Manifest-first
    if pack_manifest:
        governance = pack_manifest.get("governance", {})
        structure = governance.get("structure", {})
        drill_ref = structure.get("drill")
        if drill_ref:
            return drill_ref, EXTRACTION_SOURCE_MANIFEST

    # Fallback to evidence.json
    if evidence_data:
        governance = evidence_data.get("governance", {})
        structure = governance.get("structure", {})
        drill_ref = structure.get("drill")
        if drill_ref:
            return drill_ref, EXTRACTION_SOURCE_EVIDENCE_JSON

    return None, EXTRACTION_SOURCE_MISSING


def generate_structural_drill_warning(
    drill_reference: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Generate single warning line for structural drill (warning hygiene).

    SHADOW MODE CONTRACT:
    - Pure function, no side effects
    - Advisory only, no gating

    Warning criteria (at most ONE warning):
    - worst_severity == "CRITICAL", OR
    - max_streak >= 2

    If criteria not met, returns None (no warning).

    Args:
        drill_reference: Structural drill reference dict, or None

    Returns:
        Single warning string if criteria met, None otherwise.
    """
    if not drill_reference:
        return None

    # Extract fields
    drill_id = drill_reference.get("drill_id")
    max_streak = drill_reference.get("max_streak", 0)
    pattern_counts = drill_reference.get("pattern_counts", {})
    worst_severity = derive_worst_severity_from_pattern_counts(pattern_counts)

    # Warning criteria: CRITICAL severity OR max_streak >= 2
    should_warn = worst_severity == "CRITICAL" or max_streak >= 2

    if not should_warn:
        return None

    # Build warning parts
    parts = ["Structural drill:"]

    if worst_severity == "CRITICAL":
        parts.append("STRUCTURAL_BREAK pattern recorded")
    elif max_streak >= 2:
        parts.append(f"streak={max_streak}")

    # Include drill_id if present
    if drill_id:
        parts.append(f"[{drill_id}]")

    # Add severity context
    parts.append(f"(severity={worst_severity}, informational)")

    return " ".join(parts)


def derive_worst_severity_from_pattern_counts(
    pattern_counts: Dict[str, int],
) -> str:
    """
    Derive worst severity level from pattern counts.

    SHADOW MODE CONTRACT:
    - Pure function, no side effects
    - Advisory classification only

    Args:
        pattern_counts: Dict mapping pattern names to counts
            Expected keys: "NONE", "DRIFT", "STRUCTURAL_BREAK"

    Returns:
        Severity level: "INFO" | "WARN" | "CRITICAL"
    """
    if pattern_counts.get("STRUCTURAL_BREAK", 0) > 0:
        return "CRITICAL"
    elif pattern_counts.get("DRIFT", 0) > 0:
        return "WARN"
    return "INFO"


# Keep old function name for backward compatibility (deprecated)
def extract_drill_drivers(
    drill_reference: Optional[Dict[str, Any]],
) -> List[str]:
    """
    DEPRECATED: Use extract_drill_reason_codes() instead.

    This function is kept for backward compatibility but now returns
    reason codes instead of verbose driver strings.
    """
    return extract_drill_reason_codes(drill_reference)
