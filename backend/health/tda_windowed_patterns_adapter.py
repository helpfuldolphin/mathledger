"""
TDA Windowed Patterns Adapter for First-Light Status

Extracts signals.tda_windowed_patterns from evidence pack for status integration.

SHADOW MODE CONTRACT:
- Signal extraction is purely observational
- Missing signal is not an error (signal is optional)
- Advisory warnings are neutral (no enforcement language)
- Provides pattern classification context for reviewers

Reference: docs/system_law/TDA_PhaseX_Binding.md Section 12
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "extract_tda_windowed_patterns_signal_for_status",
    "extract_tda_windowed_patterns_warnings",
    "check_single_shot_windowed_disagreement",
    "extract_pattern_disagreement_for_status",
    "tda_windowed_patterns_for_alignment_view",
]


def extract_tda_windowed_patterns_signal_for_status(
    manifest: Optional[Mapping[str, Any]],
    evidence_data: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Extract TDA windowed patterns signal for first_light_status.json.

    Extraction order: manifest["signals"]["tda_windowed_patterns"] first,
    then fallback to evidence.json signals.

    Returns compact signal with:
    - dominant_pattern: Most frequent non-NONE pattern
    - max_streak: {pattern, length}
    - high_confidence_count: Windows with confidence >= 0.75
    - top_events_count: Number of top events in digest

    Args:
        manifest: Evidence pack manifest.json content
        evidence_data: Evidence pack evidence.json content

    Returns:
        Compact signal dict or None if not found

    SHADOW MODE: Observational only.
    """
    windowed_patterns = None
    extraction_source = "MISSING"

    # Try manifest first (preferred source)
    if manifest:
        signals_section = manifest.get("signals", {})
        if isinstance(signals_section, Mapping):
            candidate = signals_section.get("tda_windowed_patterns")
            if isinstance(candidate, Mapping):
                windowed_patterns = candidate
                extraction_source = "MANIFEST"

        # Also check governance section (alternate location)
        if windowed_patterns is None:
            governance = manifest.get("governance", {})
            if isinstance(governance, Mapping):
                tda = governance.get("tda", {})
                if isinstance(tda, Mapping):
                    candidate = tda.get("windowed_patterns")
                    if isinstance(candidate, Mapping):
                        windowed_patterns = candidate
                        extraction_source = "MANIFEST_GOVERNANCE"

    # Fallback to evidence.json
    if windowed_patterns is None and evidence_data:
        signals_section = evidence_data.get("signals", {})
        if isinstance(signals_section, Mapping):
            candidate = signals_section.get("tda_windowed_patterns")
            if isinstance(candidate, Mapping):
                windowed_patterns = candidate
                extraction_source = "EVIDENCE_JSON"

        # Also check governance section in evidence
        if windowed_patterns is None:
            governance = evidence_data.get("governance", {})
            if isinstance(governance, Mapping):
                tda = governance.get("tda", {})
                if isinstance(tda, Mapping):
                    candidate = tda.get("windowed_patterns")
                    if isinstance(candidate, Mapping):
                        windowed_patterns = candidate
                        extraction_source = "EVIDENCE_GOVERNANCE"

    if windowed_patterns is None:
        return None

    # Extract status sub-object if present
    status = windowed_patterns.get("status", {})
    if not isinstance(status, Mapping):
        status = {}

    # Build compact signal
    dominant_pattern = status.get("dominant_pattern", windowed_patterns.get("dominant_pattern", "NONE"))

    # Handle max_streak as object or separate fields
    max_streak = status.get("max_streak", {})
    if not isinstance(max_streak, Mapping):
        max_streak = {}
    max_streak_pattern = max_streak.get("pattern", windowed_patterns.get("max_streak_pattern", "NONE"))
    max_streak_length = max_streak.get("length", windowed_patterns.get("max_streak_length", 0))

    high_confidence_count = status.get(
        "high_confidence_count",
        windowed_patterns.get("high_confidence_count", 0)
    )

    top_events_count = windowed_patterns.get("top_events_count", 0)
    top_events = windowed_patterns.get("top_events", [])
    if isinstance(top_events, list) and top_events_count == 0:
        top_events_count = len(top_events)

    # Get coverage info
    coverage = status.get("coverage", {})
    if not isinstance(coverage, Mapping):
        coverage = {}
    total_windows = coverage.get("total_windows", windowed_patterns.get("total_windows", 0))
    windows_with_patterns = coverage.get("windows_with_patterns", windowed_patterns.get("windows_with_patterns", 0))

    return {
        "schema_version": windowed_patterns.get("schema_version", "1.0.0"),
        "mode": "SHADOW",
        "dominant_pattern": dominant_pattern,
        "max_streak": {
            "pattern": max_streak_pattern,
            "length": max_streak_length,
        },
        "high_confidence_count": high_confidence_count,
        "top_events_count": top_events_count,
        "coverage": {
            "total_windows": total_windows,
            "windows_with_patterns": windows_with_patterns,
        },
        "extraction_source": extraction_source,
    }


def extract_tda_windowed_patterns_warnings(
    manifest: Optional[Mapping[str, Any]],
    evidence_data: Optional[Mapping[str, Any]],
) -> List[str]:
    """
    Extract advisory warnings for TDA windowed patterns.

    Warning cap: maximum one line, triggered if:
    - dominant_pattern != NONE, or
    - high_confidence_count > 0

    Args:
        manifest: Evidence pack manifest.json content
        evidence_data: Evidence pack evidence.json content

    Returns:
        List with at most one warning string

    SHADOW MODE: Observational only, neutral wording.
    """
    signal = extract_tda_windowed_patterns_signal_for_status(manifest, evidence_data)
    if signal is None:
        return []

    warnings: List[str] = []

    dominant_pattern = signal.get("dominant_pattern", "NONE")
    high_confidence_count = signal.get("high_confidence_count", 0)
    max_streak = signal.get("max_streak", {})
    max_streak_length = max_streak.get("length", 0)
    max_streak_pattern = max_streak.get("pattern", "NONE")

    # Generate warning if patterns detected (neutral wording)
    if dominant_pattern != "NONE" or high_confidence_count > 0:
        # Build single line: "TDA windowed patterns: dominant=X, streak=Y(N), high_conf=Z"
        warning_parts = []
        if dominant_pattern != "NONE":
            warning_parts.append(f"dominant={dominant_pattern}")
        if max_streak_length > 1 and max_streak_pattern != "NONE":
            warning_parts.append(f"streak={max_streak_pattern}({max_streak_length})")
        if high_confidence_count > 0:
            warning_parts.append(f"high_conf={high_confidence_count}")

        if warning_parts:
            warnings.append(f"TDA windowed patterns: {', '.join(warning_parts)}")

    return warnings[:1]  # Cap to 1 warning


def check_single_shot_windowed_disagreement(
    single_shot_pattern: Optional[str],
    windowed_dominant_pattern: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Check for disagreement between single-shot and windowed classification.

    Flags disagreement when:
    - Single-shot says NONE but windowed dominant != NONE
    - Single-shot says non-NONE but windowed dominant == NONE

    Args:
        single_shot_pattern: Pattern from TDAPatternClassifier.classify() (single point)
        windowed_dominant_pattern: Dominant pattern from aggregate_pattern_summary()

    Returns:
        Advisory dict with disagreement details, or None if no disagreement

    SHADOW MODE: Advisory only, no enforcement.
    """
    if single_shot_pattern is None or windowed_dominant_pattern is None:
        return None

    # Normalize to uppercase
    single_shot = single_shot_pattern.upper()
    windowed = windowed_dominant_pattern.upper()

    # Check for disagreement
    single_is_none = single_shot == "NONE"
    windowed_is_none = windowed == "NONE"

    if single_is_none == windowed_is_none:
        # No disagreement - both NONE or both non-NONE
        return None

    # Disagreement detected - use DRIVER_ reason codes (no prose)
    if single_is_none and not windowed_is_none:
        reason_code = "DRIVER_WINDOWED_DETECTED_PATTERN"
    else:
        reason_code = "DRIVER_SINGLE_SHOT_DETECTED_PATTERN"

    return {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "disagreement_detected": True,
        "reason_code": reason_code,
        "single_shot_pattern": single_shot,
        "windowed_dominant_pattern": windowed,
    }


def extract_pattern_disagreement_for_status(
    manifest: Optional[Mapping[str, Any]],
    evidence_data: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Extract pattern disagreement from evidence for status integration.

    Checks both single-shot pattern classification and windowed patterns
    to detect disagreements.

    Args:
        manifest: Evidence pack manifest.json content
        evidence_data: Evidence pack evidence.json content

    Returns:
        Disagreement advisory dict or None if no disagreement

    SHADOW MODE: Advisory only.
    """
    # Extract single-shot pattern
    single_shot_pattern = None

    # Try manifest governance.tda.patterns section
    if manifest:
        governance = manifest.get("governance", {})
        if isinstance(governance, Mapping):
            tda = governance.get("tda", {})
            if isinstance(tda, Mapping):
                patterns = tda.get("patterns", {})
                if isinstance(patterns, Mapping):
                    single_shot_pattern = patterns.get("pattern")

    # Fallback to evidence
    if single_shot_pattern is None and evidence_data:
        governance = evidence_data.get("governance", {})
        if isinstance(governance, Mapping):
            tda = governance.get("tda", {})
            if isinstance(tda, Mapping):
                patterns = tda.get("patterns", {})
                if isinstance(patterns, Mapping):
                    single_shot_pattern = patterns.get("pattern")

    # Extract windowed dominant pattern
    windowed_signal = extract_tda_windowed_patterns_signal_for_status(manifest, evidence_data)
    windowed_dominant = windowed_signal.get("dominant_pattern") if windowed_signal else None

    # Check for disagreement
    return check_single_shot_windowed_disagreement(single_shot_pattern, windowed_dominant)


def tda_windowed_patterns_for_alignment_view(
    windowed_signal: Optional[Dict[str, Any]],
    disagreement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    GGFL adapter: Transform TDA windowed patterns signal to alignment view format.

    Produces a unified cross-subsystem signal for the Global Governance Fusion Layer.

    Args:
        windowed_signal: Output from extract_tda_windowed_patterns_signal_for_status()
        disagreement: Optional output from check_single_shot_windowed_disagreement()

    Returns:
        GGFL-normalized dict with:
        - signal_type: "SIG-TDAW"
        - status: "ok" | "warn"
        - conflict: False (always)
        - weight_hint: "LOW" (advisory only)
        - extraction_source: "MANIFEST" | "EVIDENCE_JSON" | "MISSING"
        - drivers: List of up to 3 reason codes
        - summary: One-sentence summary

    SHADOW MODE: Advisory only, no enforcement.
    """
    # Default response for missing signal
    if windowed_signal is None:
        return {
            "signal_type": "SIG-TDAW",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "extraction_source": "MISSING",
            "drivers": [],
            "summary": "No TDA windowed patterns signal available.",
        }

    # Extract key fields
    dominant_pattern = windowed_signal.get("dominant_pattern", "NONE")
    max_streak = windowed_signal.get("max_streak", {})
    max_streak_pattern = max_streak.get("pattern", "NONE") if isinstance(max_streak, dict) else "NONE"
    max_streak_length = max_streak.get("length", 0) if isinstance(max_streak, dict) else 0
    high_confidence_count = windowed_signal.get("high_confidence_count", 0)

    # Extract extraction_source (normalize to canonical values)
    raw_source = windowed_signal.get("extraction_source", "MISSING")
    # Normalize: MANIFEST_GOVERNANCE -> MANIFEST, EVIDENCE_GOVERNANCE -> EVIDENCE_JSON
    if raw_source in ("MANIFEST", "MANIFEST_GOVERNANCE"):
        extraction_source = "MANIFEST"
    elif raw_source in ("EVIDENCE_JSON", "EVIDENCE_GOVERNANCE"):
        extraction_source = "EVIDENCE_JSON"
    else:
        extraction_source = "MISSING"

    # Determine status: warn if dominant_pattern != NONE or disagreement present
    has_dominant_pattern = dominant_pattern.upper() != "NONE"
    has_disagreement = (
        disagreement is not None
        and disagreement.get("disagreement_detected", False)
    )

    status = "warn" if (has_dominant_pattern or has_disagreement) else "ok"

    # Build drivers (max 3) - use DRIVER_ reason codes
    drivers: List[str] = []

    if has_dominant_pattern:
        drivers.append(f"DRIVER_DOMINANT_PATTERN:{dominant_pattern}")

    if max_streak_length > 1 and max_streak_pattern.upper() != "NONE":
        drivers.append(f"DRIVER_STREAK:{max_streak_pattern}({max_streak_length})")

    if has_disagreement and disagreement is not None:
        reason_code = disagreement.get("reason_code", "DRIVER_UNKNOWN")
        drivers.append(reason_code)

    # Cap to 3 drivers
    drivers = drivers[:3]

    # Build summary (1 sentence)
    if status == "ok":
        summary = "TDA windowed analysis shows no dominant patterns."
    else:
        summary_parts = []
        if has_dominant_pattern:
            summary_parts.append(f"dominant pattern {dominant_pattern}")
        if has_disagreement and disagreement is not None:
            reason_code = disagreement.get("reason_code", "DRIVER_UNKNOWN")
            summary_parts.append(f"disagreement ({reason_code})")
        summary = f"TDA windowed analysis detected: {', '.join(summary_parts)}."

    return {
        "signal_type": "SIG-TDAW",
        "status": status,
        "conflict": False,
        "weight_hint": "LOW",
        "extraction_source": extraction_source,
        "drivers": drivers,
        "summary": summary,
    }
