"""
Pilot External Ingest Adapter

Enables external parties to supply log artifacts (JSON/JSONL) that can be
wrapped into a First Light evidence pack WITHOUT changing any existing schema.

SHADOW MODE CONTRACT:
- All operations are OBSERVATIONAL ONLY
- No new metrics are created
- No existing schemas are modified
- No gating or enforcement
- Strictly logged, non-invasive

CAL-EXP-3 SCOPE:
- Pilot-only ingestion adapter
- Composes with existing evidence pack builder
- Reuses existing manifest + integrity logic

Status: PILOT (SHADOW MODE)

# =============================================================================
# FROZEN SURFACE (v1.0.0)
# =============================================================================
#
# Frozen Enums:
#   - PilotIngestSource: EXTERNAL_JSON | EXTERNAL_JSONL | EXTERNAL_BUNDLE | INVALID
#   - PilotIngestResult: SUCCESS | SCHEMA_INVALID | FILE_NOT_FOUND | PARSE_ERROR | INTEGRITY_MISMATCH
#
# Mode Invariants:
#   - SHADOW only (LOGGED_ONLY)
#   - No new metrics created
#   - No enforcement / no gating
#   - No network calls
#   - No writes to CAL-EXP paths (p3/, p4/, governance/)
#   - All outputs under evidence_pack/external/
#
# Non-Interference:
#   - Does not import CAL-EXP-2 frozen modules
#   - Does not modify existing manifest fields
#   - Only adds governance.external_pilot section
#
# =============================================================================
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "PilotIngestSource",
    "PilotIngestResult",
    "validate_external_log_schema",
    "ingest_external_log",
    "wrap_for_evidence_pack",
    "compute_file_sha256",
    "PILOT_INGEST_SCHEMA_VERSION",
]


# Schema version for pilot ingest adapter
PILOT_INGEST_SCHEMA_VERSION = "1.0.0"


# =============================================================================
# Frozen Enums (SHADOW MODE)
# =============================================================================

class PilotIngestSource:
    """Frozen enum for pilot ingest source tracking."""
    EXTERNAL_JSON = "EXTERNAL_JSON"       # Single JSON file
    EXTERNAL_JSONL = "EXTERNAL_JSONL"     # JSONL (line-delimited JSON)
    EXTERNAL_BUNDLE = "EXTERNAL_BUNDLE"   # Directory of JSON files
    INVALID = "INVALID"                   # Failed validation


class PilotIngestResult:
    """Frozen enum for ingest operation results."""
    SUCCESS = "SUCCESS"                   # Successfully ingested
    SCHEMA_INVALID = "SCHEMA_INVALID"     # Schema validation failed
    FILE_NOT_FOUND = "FILE_NOT_FOUND"     # Source file missing
    PARSE_ERROR = "PARSE_ERROR"           # JSON parse error
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"  # SHA256 mismatch


# =============================================================================
# Minimal Schema Requirements
# =============================================================================

# Required fields for external log artifacts (minimal contract)
MINIMAL_SCHEMA_FIELDS = frozenset({
    # At least one of these must be present to identify the log type
    "log_type",    # Primary identifier (e.g., "runtime", "metrics", "events")
})

# Optional but recognized fields
RECOGNIZED_FIELDS = frozenset({
    "log_type",
    "timestamp",
    "entries",
    "metadata",
    "source",
    "version",
})


# =============================================================================
# Core Functions
# =============================================================================

def compute_file_sha256(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

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


def validate_external_log_schema(
    data: Dict[str, Any],
) -> Tuple[bool, str, List[str]]:
    """
    Validate external log artifact against minimal schema requirements.

    SHADOW MODE CONTRACT:
    - Non-mutating validation only
    - Returns validation result, does not raise
    - Permissive: only requires log_type field

    Args:
        data: Parsed JSON data from external source

    Returns:
        Tuple of (is_valid, result_code, warnings)
        - is_valid: True if schema is valid
        - result_code: PilotIngestResult enum value
        - warnings: List of non-fatal warnings
    """
    warnings: List[str] = []

    # Check for required fields
    if "log_type" not in data:
        return False, PilotIngestResult.SCHEMA_INVALID, ["missing required field: log_type"]

    # Check for unrecognized fields (warning only, not failure)
    for key in data.keys():
        if key not in RECOGNIZED_FIELDS:
            warnings.append(f"unrecognized field: {key}")

    return True, PilotIngestResult.SUCCESS, warnings


def _detect_source_type(file_path: Path) -> str:
    """
    Detect the source type from file path.

    Args:
        file_path: Path to external log file

    Returns:
        PilotIngestSource enum value
    """
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return PilotIngestSource.EXTERNAL_JSONL
    elif suffix == ".json":
        return PilotIngestSource.EXTERNAL_JSON
    else:
        return PilotIngestSource.INVALID


def _parse_json_file(file_path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Tuple of (parsed_data, result_code)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, PilotIngestResult.SUCCESS
    except json.JSONDecodeError:
        return None, PilotIngestResult.PARSE_ERROR
    except FileNotFoundError:
        return None, PilotIngestResult.FILE_NOT_FOUND
    except OSError:
        return None, PilotIngestResult.FILE_NOT_FOUND


def _parse_jsonl_file(file_path: Path) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Parse a JSONL file (line-delimited JSON).

    Args:
        file_path: Path to JSONL file

    Returns:
        Tuple of (parsed_entries, result_code)
    """
    entries: List[Dict[str, Any]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    # Skip malformed lines (SHADOW MODE: permissive)
                    continue
        return entries, PilotIngestResult.SUCCESS
    except FileNotFoundError:
        return None, PilotIngestResult.FILE_NOT_FOUND
    except OSError:
        return None, PilotIngestResult.FILE_NOT_FOUND


def ingest_external_log(
    file_path: Path,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest an external log artifact with schema validation.

    SHADOW MODE CONTRACT:
    - Validates schema presence
    - Emits extraction_source metadata
    - Non-mutating, observational only
    - Returns structured result (never raises)

    Args:
        file_path: Path to external log file (JSON or JSONL)
        expected_sha256: Optional SHA256 for integrity verification

    Returns:
        Ingest result dict with:
        - result: PilotIngestResult enum value
        - source_type: PilotIngestSource enum value
        - data: Parsed data (if successful)
        - sha256: Computed file hash
        - warnings: List of non-fatal warnings
        - extraction_source: Always "EXTERNAL_PILOT"
        - schema_version: Adapter schema version
    """
    result: Dict[str, Any] = {
        "schema_version": PILOT_INGEST_SCHEMA_VERSION,
        "extraction_source": "EXTERNAL_PILOT",
        "mode": "SHADOW",
        "action": "LOGGED_ONLY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Check file exists
    if not file_path.exists():
        result["result"] = PilotIngestResult.FILE_NOT_FOUND
        result["source_type"] = PilotIngestSource.INVALID
        result["data"] = None
        result["sha256"] = None
        result["warnings"] = [f"file not found: {file_path}"]
        return result

    # Compute SHA256
    actual_sha256 = compute_file_sha256(file_path)
    result["sha256"] = actual_sha256

    # Verify integrity if expected hash provided
    if expected_sha256 and actual_sha256 != expected_sha256:
        result["result"] = PilotIngestResult.INTEGRITY_MISMATCH
        result["source_type"] = PilotIngestSource.INVALID
        result["data"] = None
        result["warnings"] = [
            f"integrity mismatch: expected {expected_sha256}, got {actual_sha256}"
        ]
        return result

    # Detect source type
    source_type = _detect_source_type(file_path)
    result["source_type"] = source_type

    if source_type == PilotIngestSource.INVALID:
        result["result"] = PilotIngestResult.SCHEMA_INVALID
        result["data"] = None
        result["warnings"] = [f"unsupported file type: {file_path.suffix}"]
        return result

    # Parse based on source type
    if source_type == PilotIngestSource.EXTERNAL_JSON:
        data, parse_result = _parse_json_file(file_path)
        if parse_result != PilotIngestResult.SUCCESS:
            result["result"] = parse_result
            result["data"] = None
            result["warnings"] = [f"JSON parse unsuccessful: {file_path}"]
            return result

        # Validate schema
        is_valid, validation_result, warnings = validate_external_log_schema(data)
        result["warnings"] = warnings

        if not is_valid:
            result["result"] = validation_result
            result["data"] = None
            return result

        result["result"] = PilotIngestResult.SUCCESS
        result["data"] = data

    elif source_type == PilotIngestSource.EXTERNAL_JSONL:
        entries, parse_result = _parse_jsonl_file(file_path)
        if parse_result != PilotIngestResult.SUCCESS:
            result["result"] = parse_result
            result["data"] = None
            result["warnings"] = [f"JSONL parse unsuccessful: {file_path}"]
            return result

        # For JSONL, wrap entries in a container with log_type
        data = {
            "log_type": "external_jsonl",
            "entries": entries,
            "entry_count": len(entries),
        }
        result["result"] = PilotIngestResult.SUCCESS
        result["data"] = data
        result["warnings"] = []

    return result


def wrap_for_evidence_pack(
    ingest_result: Dict[str, Any],
    source_file: Path,
    target_subdir: str = "external",
) -> Dict[str, Any]:
    """
    Wrap ingested external log for inclusion in evidence pack.

    SHADOW MODE CONTRACT:
    - Produces manifest-compatible entry
    - Reuses existing manifest + integrity logic
    - No schema changes to evidence pack
    - Adds to governance.external_pilot section only

    Args:
        ingest_result: Result from ingest_external_log()
        source_file: Original source file path
        target_subdir: Subdirectory within evidence pack (default: "external")

    Returns:
        Evidence pack entry dict with:
        - path: Manifest-friendly path
        - sha256: File hash
        - pilot_metadata: Pilot-specific metadata
    """
    if ingest_result.get("result") != PilotIngestResult.SUCCESS:
        return {
            "valid": False,
            "error": ingest_result.get("result", PilotIngestResult.SCHEMA_INVALID),
            "warnings": ingest_result.get("warnings", []),
        }

    # Build manifest entry
    target_name = source_file.name
    manifest_path = f"{target_subdir}/{target_name}"

    return {
        "valid": True,
        "path": manifest_path,
        "sha256": ingest_result.get("sha256"),
        "pilot_metadata": {
            "source_type": ingest_result.get("source_type"),
            "extraction_source": "EXTERNAL_PILOT",
            "schema_version": PILOT_INGEST_SCHEMA_VERSION,
            "mode": "SHADOW",
            "action": "LOGGED_ONLY",
            "ingested_at": ingest_result.get("timestamp"),
        },
        "warnings": ingest_result.get("warnings", []),
    }


def attach_to_manifest(
    manifest: Dict[str, Any],
    evidence_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Attach pilot external entries to evidence pack manifest.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Adds to governance.external_pilot section only
    - Does not modify any existing manifest fields
    - Reuses existing manifest structure

    Args:
        manifest: Existing evidence pack manifest
        evidence_entries: List of entries from wrap_for_evidence_pack()

    Returns:
        New manifest with external_pilot section added
    """
    import copy
    new_manifest = copy.deepcopy(manifest)

    # Ensure governance section exists
    if "governance" not in new_manifest:
        new_manifest["governance"] = {}

    # Add external_pilot section
    valid_entries = [e for e in evidence_entries if e.get("valid", False)]
    invalid_entries = [e for e in evidence_entries if not e.get("valid", False)]

    new_manifest["governance"]["external_pilot"] = {
        "schema_version": PILOT_INGEST_SCHEMA_VERSION,
        "mode": "SHADOW",
        "action": "LOGGED_ONLY",
        "entries": [
            {
                "path": e["path"],
                "sha256": e["sha256"],
                "pilot_metadata": e["pilot_metadata"],
            }
            for e in valid_entries
        ],
        "entry_count": len(valid_entries),
        "invalid_count": len(invalid_entries),
        "warnings": [
            w
            for e in evidence_entries
            for w in e.get("warnings", [])
        ],
    }

    return new_manifest


def copy_to_evidence_pack(
    source_file: Path,
    evidence_pack_dir: Path,
    target_subdir: str = "external",
) -> Optional[Path]:
    """
    Copy external log file to evidence pack directory.

    SHADOW MODE CONTRACT:
    - Creates target directory if needed
    - Does not overwrite existing files
    - Returns None on failure (no exception)

    Args:
        source_file: Source file path
        evidence_pack_dir: Root of evidence pack
        target_subdir: Subdirectory within evidence pack

    Returns:
        Path to copied file, or None on failure
    """
    import shutil

    target_dir = evidence_pack_dir / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / source_file.name

    # Don't overwrite existing files
    if target_file.exists():
        return None

    try:
        shutil.copy2(source_file, target_file)
        return target_file
    except (OSError, shutil.Error):
        return None
