"""
RTTS Status Adapter for First-Light Status

Phase X P5.2: Surfaces RTTS validation status in first_light_status.json.

This module extracts a compact RTTS status signal from rtts_validation.json
for inclusion in the First-Light status summary.

SHADOW MODE CONTRACT:
- All operations are OBSERVATIONAL ONLY
- Status is advisory, not a gate
- Single warning emitted for WARN/CRITICAL status
- No enforcement

RTTS Gap Closure: P5.2 VALIDATE stage
See: docs/system_law/RTTS_Gap_Closure_Blueprint.md

Status: P5.2 VALIDATE (NO ENFORCEMENT)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "extract_rtts_status_signal",
    "extract_rtts_status_for_first_light",
    "load_rtts_validation_for_status",
    "load_rtts_validation_from_manifest_reference",
    "generate_rtts_warning",
    "compute_rtts_file_sha256",
    "ExtractionSource",
    "DriverCategory",
    "VALID_DRIVER_CATEGORIES",
]


# Schema version for RTTS status signal
RTTS_STATUS_SIGNAL_SCHEMA_VERSION = "1.2.0"  # Bumped for extraction_source + frozen enums


# =============================================================================
# Frozen Enums (PROVENANCE + WARNING NORMAL FORM)
# =============================================================================

# Extraction source enum - tracks where RTTS data came from
class ExtractionSource:
    """Frozen enum for RTTS data extraction source."""
    MANIFEST_REFERENCE = "MANIFEST_REFERENCE"
    DIRECT_DISCOVERY = "DIRECT_DISCOVERY"
    MISSING = "MISSING"


# Driver category enum - frozen categories for top_driver_category
class DriverCategory:
    """Frozen enum for RTTS driver categories."""
    STATISTICAL = "STATISTICAL"
    CORRELATION = "CORRELATION"
    CONTINUITY = "CONTINUITY"
    UNKNOWN = "UNKNOWN"


# Valid driver categories (for validation)
VALID_DRIVER_CATEGORIES = frozenset({
    DriverCategory.STATISTICAL,
    DriverCategory.CORRELATION,
    DriverCategory.CONTINUITY,
    DriverCategory.UNKNOWN,
})


# MOCK code to category mapping (uses frozen enum values)
MOCK_CODE_CATEGORIES: Dict[str, str] = {
    "MOCK-001": DriverCategory.STATISTICAL,   # Var(H) below threshold
    "MOCK-002": DriverCategory.STATISTICAL,   # Var(rho) below threshold
    "MOCK-003": DriverCategory.CORRELATION,   # Low correlation |Cor(H,rho)|
    "MOCK-004": DriverCategory.CORRELATION,   # High correlation |Cor(H,rho)|
    "MOCK-005": DriverCategory.STATISTICAL,   # ACF below threshold
    "MOCK-006": DriverCategory.STATISTICAL,   # ACF above threshold
    "MOCK-007": DriverCategory.STATISTICAL,   # Kurtosis below threshold
    "MOCK-008": DriverCategory.STATISTICAL,   # Kurtosis above threshold
    "MOCK-009": DriverCategory.CONTINUITY,    # Jump in H (max delta)
    "MOCK-010": DriverCategory.CONTINUITY,    # Discrete rho values
}


def load_rtts_validation_for_status(
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load rtts_validation.json from run directory for status extraction.

    SHADOW MODE CONTRACT:
    - Read-only, no side effects
    - Returns None on missing/invalid file

    Args:
        run_dir: Path to run directory

    Returns:
        RTTS validation dict, or None if not found
    """
    candidates = [
        run_dir / "rtts_validation.json",
        run_dir / "p4_shadow" / "rtts_validation.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _compute_top_driver(
    mock_flags: List[str],
) -> Tuple[str, List[str]]:
    """
    Compute top driver category and codes from MOCK flags.

    Determines which category (STATISTICAL|CORRELATION|CONTINUITY) has
    the most violations and returns the top 3 codes from that category.

    SHADOW MODE: Deterministic (sorted alphabetically).
    Uses frozen DriverCategory enum values.

    Args:
        mock_flags: List of MOCK-NNN codes

    Returns:
        Tuple of (top_driver_category, top_driver_codes_top3)
        Category is from DriverCategory enum (UNKNOWN if no flags)
    """
    if not mock_flags:
        return DriverCategory.UNKNOWN, []

    # Count violations per category (using frozen enum values)
    category_counts: Dict[str, int] = {
        DriverCategory.STATISTICAL: 0,
        DriverCategory.CORRELATION: 0,
        DriverCategory.CONTINUITY: 0,
    }
    category_codes: Dict[str, List[str]] = {
        DriverCategory.STATISTICAL: [],
        DriverCategory.CORRELATION: [],
        DriverCategory.CONTINUITY: [],
    }

    for code in mock_flags:
        category = MOCK_CODE_CATEGORIES.get(code)
        if category:
            category_counts[category] += 1
            category_codes[category].append(code)

    # Find top category (deterministic: by count desc, then alphabetically)
    sorted_categories = sorted(
        category_counts.items(),
        key=lambda x: (-x[1], x[0])  # Descending count, then alphabetic
    )

    top_category = DriverCategory.UNKNOWN
    top_codes: List[str] = []

    for category, count in sorted_categories:
        if count > 0:
            top_category = category
            # Sort codes alphabetically and take top 3
            top_codes = sorted(category_codes[category])[:3]
            break

    return top_category, top_codes


def extract_rtts_status_signal(
    rtts_validation: Optional[Dict[str, Any]] = None,
    extraction_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract compact RTTS status signal for first_light_status.json.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - All fields are observational only
    - No gating or enforcement
    - Deterministic ordering (sorted alphabetically)

    Output fields:
    - overall_status: "OK" | "ATTENTION" | "WARN" | "CRITICAL" | "UNKNOWN"
    - violation_count: Total number of violations detected
    - top3_mock_codes: Top 3 MOCK-NNN codes sorted alphabetically
    - top_driver_category: "STATISTICAL" | "CORRELATION" | "CONTINUITY" | "UNKNOWN"
    - top_driver_codes_top3: Top 3 codes from the driver category (sorted)
    - extraction_source: "MANIFEST_REFERENCE" | "DIRECT_DISCOVERY" | "MISSING"
    - mode: "SHADOW"
    - action: "LOGGED_ONLY"
    - available: True if RTTS validation data was available

    Args:
        rtts_validation: Optional dict from rtts_validation.json
        extraction_source: Optional source identifier (from ExtractionSource enum)

    Returns:
        Compact RTTS status signal dict
    """
    signal: Dict[str, Any] = {
        "schema_version": RTTS_STATUS_SIGNAL_SCHEMA_VERSION,
        "mode": "SHADOW",
        "action": "LOGGED_ONLY",
    }

    if rtts_validation is None:
        signal["available"] = False
        signal["overall_status"] = "UNKNOWN"
        signal["violation_count"] = 0
        signal["top3_mock_codes"] = []
        signal["top_driver_category"] = DriverCategory.UNKNOWN
        signal["top_driver_codes_top3"] = []
        signal["extraction_source"] = extraction_source or ExtractionSource.MISSING
        return signal

    signal["available"] = True

    # Extract overall status
    signal["overall_status"] = rtts_validation.get("overall_status", "UNKNOWN")

    # Extract violation count (from warning_count as proxy)
    signal["violation_count"] = rtts_validation.get("warning_count", 0)

    # Extract top 3 MOCK codes (sorted alphabetically)
    mock_flags = rtts_validation.get("mock_detection_flags", [])
    # Sort alphabetically and take top 3
    sorted_flags = sorted(mock_flags)[:3]
    signal["top3_mock_codes"] = sorted_flags

    # Compute top driver category and codes (uses frozen enum)
    top_driver_category, top_driver_codes = _compute_top_driver(mock_flags)
    signal["top_driver_category"] = top_driver_category
    signal["top_driver_codes_top3"] = top_driver_codes

    # Set extraction source (provenance tracking)
    signal["extraction_source"] = extraction_source or ExtractionSource.DIRECT_DISCOVERY

    return signal


def generate_rtts_warning(
    rtts_signal: Dict[str, Any],
) -> Optional[str]:
    """
    Generate warning message if RTTS status is WARN or CRITICAL.

    WARNING NORMAL FORM (deterministic):
    - Format: "RTTS {overall_status}: {violation_count} violations | driver={top_driver_category} | flags=[{top3_mock_codes}]"
    - All fields sorted alphabetically where applicable
    - Single line, deterministic output

    SHADOW MODE CONTRACT:
    - Advisory warning only
    - No gating or enforcement
    - Returns None for OK/ATTENTION/UNKNOWN status

    Args:
        rtts_signal: RTTS status signal from extract_rtts_status_signal()

    Returns:
        Warning string if status is WARN or CRITICAL, else None
    """
    if not rtts_signal.get("available", False):
        return None

    status = rtts_signal.get("overall_status", "UNKNOWN")

    if status not in ("WARN", "CRITICAL"):
        return None

    # Extract fields for normal form
    violation_count = rtts_signal.get("violation_count", 0)
    top_driver_category = rtts_signal.get("top_driver_category", DriverCategory.UNKNOWN)
    mock_codes = rtts_signal.get("top3_mock_codes", [])

    # Build deterministic warning string (NORMAL FORM)
    # Format: "RTTS {status}: {count} violations | driver={category} | flags=[{codes}]"
    codes_str = ", ".join(mock_codes) if mock_codes else "none"

    return (
        f"RTTS {status}: {violation_count} violations | "
        f"driver={top_driver_category} | "
        f"flags=[{codes_str}]"
    )


def compute_rtts_file_sha256(file_path: Path) -> str:
    """
    Compute SHA-256 hash of rtts_validation.json file.

    Args:
        file_path: Path to the file

    Returns:
        Hex digest of SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_rtts_validation_from_manifest_reference(
    manifest_reference: Dict[str, Any],
    evidence_pack_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load rtts_validation.json using manifest reference with integrity check.

    SHADOW MODE CONTRACT:
    - Validates sha256 before returning data
    - Returns None on integrity failure (logged only, no error)
    - Deterministic behavior

    Args:
        manifest_reference: Dict with 'path' and 'sha256' keys
        evidence_pack_dir: Root of evidence pack

    Returns:
        RTTS validation dict if valid, None otherwise
    """
    ref_path = manifest_reference.get("path")
    ref_sha256 = manifest_reference.get("sha256")

    if not ref_path:
        return None

    # Resolve path relative to evidence pack
    file_path = evidence_pack_dir / ref_path
    if not file_path.exists():
        return None

    # Verify integrity
    if ref_sha256:
        actual_sha256 = compute_rtts_file_sha256(file_path)
        if actual_sha256 != ref_sha256:
            # Integrity mismatch - SHADOW MODE: log only, no error
            return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def extract_rtts_status_for_first_light(
    run_dir: Path,
    manifest_reference: Optional[Dict[str, Any]] = None,
    evidence_pack_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full extraction pipeline: load rtts_validation.json and extract status signal.

    SHADOW MODE CONTRACT:
    - Prefers manifest reference when available (with integrity check)
    - Falls back to direct file discovery
    - Read-only, no side effects
    - Returns signal with available=False if file missing
    - Tracks extraction_source for provenance

    Args:
        run_dir: Path to run directory
        manifest_reference: Optional manifest reference with path+sha256
        evidence_pack_dir: Optional root of evidence pack for manifest reference

    Returns:
        RTTS status signal dict with extraction_source provenance
    """
    rtts_validation = None
    extraction_source = ExtractionSource.MISSING

    # Prefer manifest reference if available
    if manifest_reference and evidence_pack_dir:
        rtts_validation = load_rtts_validation_from_manifest_reference(
            manifest_reference, evidence_pack_dir
        )
        if rtts_validation is not None:
            extraction_source = ExtractionSource.MANIFEST_REFERENCE

    # Fallback to direct discovery
    if rtts_validation is None:
        rtts_validation = load_rtts_validation_for_status(run_dir)
        if rtts_validation is not None:
            extraction_source = ExtractionSource.DIRECT_DISCOVERY

    return extract_rtts_status_signal(rtts_validation, extraction_source)
