"""
PHASE II — Telemetry Conformance Checker

Implements conformance level classification (L0/L1/L2), quarantine pipeline,
and batch auditing per TELEMETRY_CONFORMANCE_SPEC.md.

Author: CLAUDE H (Telemetry Conformance Enforcer)
Date: 2025-12-06
Status: PHASE II — NOT RUN IN PHASE I
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# CONFORMANCE LEVELS
# =============================================================================

class ConformanceLevel(Enum):
    """Telemetry conformance levels per TELEMETRY_CONFORMANCE_SPEC.md Section 2."""
    L0 = "L0"  # Raw: parseable JSON, no schema guarantees
    L1 = "L1"  # Schema-valid: passes JSON schema validation
    L2 = "L2"  # Canonical: full canonical form conformance
    QUARANTINE = "QUARANTINE"  # Failed validation, requires quarantine


class ViolationSeverity(Enum):
    """Severity levels for conformance violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Violation:
    """A single conformance violation."""
    check_name: str
    check_category: str  # "schema", "format", "semantic", "temporal"
    severity: ViolationSeverity
    field: Optional[str]
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class ConformanceResult:
    """Result of conformance classification."""
    level: ConformanceLevel
    violations: list[Violation] = field(default_factory=list)
    record_type: Optional[str] = None
    canonical_hash: Optional[str] = None
    raw_json: Optional[str] = None
    parsed_record: Optional[dict] = None

    @property
    def is_valid(self) -> bool:
        """Check if record achieved at least L1 conformance."""
        return self.level in (ConformanceLevel.L1, ConformanceLevel.L2)

    @property
    def is_canonical(self) -> bool:
        """Check if record achieved L2 conformance."""
        return self.level == ConformanceLevel.L2

    @property
    def requires_quarantine(self) -> bool:
        """Check if record must be quarantined."""
        return self.level == ConformanceLevel.QUARANTINE


# =============================================================================
# SCHEMA DEFINITIONS (from TELEMETRY_CANONICAL_FORM.md)
# =============================================================================

# Canonical field names per record type (alphabetically ordered)
CYCLE_METRIC_FIELDS = [
    "cycle", "ht", "metric_type", "metric_value", "mode",
    "r_t", "run_id", "slice", "success", "ts", "u_t"
]

EXPERIMENT_SUMMARY_FIELDS = [
    "ci_95", "mode", "n_cycles", "p_success",
    "phase", "run_id", "slice", "uplift_delta"
]

UPLIFT_RESULT_FIELDS = [
    "baseline_run_id", "ci_95", "n_base", "n_rfl",
    "p_base", "p_rfl", "p_value", "phase", "rfl_run_id",
    "significant", "slice", "ts", "uplift_delta"
]

# Required fields per record type
REQUIRED_FIELDS = {
    "cycle_metric": set(CYCLE_METRIC_FIELDS),
    "experiment_summary": set(EXPERIMENT_SUMMARY_FIELDS),
    "uplift_result": set(UPLIFT_RESULT_FIELDS),
}

# Prohibited field names (from TELEMETRY_CANONICAL_FORM.md Section 4)
PROHIBITED_FIELDS = {
    "_id", "id", "timestamp", "time", "datetime",
    "created_at", "updated_at", "version", "schema_version",
    "type", "status", "result", "data", "payload",
    "metadata", "extra", "custom", "tags", "labels"
}

# Valid enum values
VALID_MODES = {"baseline", "rfl", "comparison"}
VALID_SLICES = {
    "slice_uplift_goal", "slice_throughput",
    "slice_depth_advance", "slice_novelty",
    # Also accept legacy U2_env_X format
    "U2_env_A", "U2_env_B", "U2_env_C", "U2_env_D"
}
VALID_METRIC_TYPES = {"goal_hit", "throughput", "depth_reached", "novel_count"}
VALID_PHASES = {"I", "II", "III"}

# Regex patterns
HASH_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
RUN_ID_PATTERN = re.compile(r"^U2-[0-9a-f-]+$")
FIELD_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


# =============================================================================
# CONFORMANCE CHECKS
# =============================================================================

def _detect_record_type(record: dict) -> Optional[str]:
    """Detect record type from field signature.

    Uses a more flexible detection that can identify records even with
    some missing required fields.
    """
    keys = set(record.keys())

    # Check for cycle_metric signature (any 2 of the key fields)
    cycle_metric_markers = {"cycle", "metric_type", "ht", "metric_value", "mode"}
    if len(keys & cycle_metric_markers) >= 2:
        return "cycle_metric"

    # Check for uplift_result signature
    uplift_markers = {"baseline_run_id", "rfl_run_id", "uplift_delta", "p_base", "p_rfl"}
    if len(keys & uplift_markers) >= 2:
        return "uplift_result"

    # Check for experiment_summary signature
    summary_markers = {"n_cycles", "p_success", "phase", "ci_95"}
    if len(keys & summary_markers) >= 2:
        return "experiment_summary"

    return None


def _check_required_fields(record: dict, record_type: str) -> list[Violation]:
    """Check for missing required fields."""
    violations = []
    required = REQUIRED_FIELDS.get(record_type, set())
    present = set(record.keys())
    missing = required - present

    for field_name in sorted(missing):
        violations.append(Violation(
            check_name="check_required_fields",
            check_category="schema",
            severity=ViolationSeverity.CRITICAL,
            field=field_name,
            message=f"Missing required field: {field_name}",
            expected="present",
            actual="missing"
        ))

    return violations


def _check_prohibited_fields(record: dict) -> list[Violation]:
    """Check for prohibited fields."""
    violations = []
    present = set(record.keys())
    prohibited = present & PROHIBITED_FIELDS

    for field_name in sorted(prohibited):
        violations.append(Violation(
            check_name="check_prohibited_fields",
            check_category="schema",
            severity=ViolationSeverity.ERROR,
            field=field_name,
            message=f"Prohibited field present: {field_name}",
            expected="absent",
            actual="present"
        ))

    return violations


def _check_field_types(record: dict, record_type: str) -> list[Violation]:
    """Check field types are correct."""
    violations = []

    type_specs = {
        "cycle": int,
        "n_cycles": int,
        "n_base": int,
        "n_rfl": int,
        "success": bool,
        "significant": bool,
        "p_success": (int, float),
        "p_base": (int, float),
        "p_rfl": (int, float),
        "p_value": (int, float),
        "metric_value": (int, float),
        "uplift_delta": (int, float, type(None)),
        "ts": str,
        "run_id": str,
        "baseline_run_id": str,
        "rfl_run_id": str,
        "slice": str,
        "mode": str,
        "metric_type": str,
        "phase": str,
        "ht": str,
        "r_t": str,
        "u_t": str,
        "ci_95": list,
    }

    for field_name, expected_type in type_specs.items():
        if field_name not in record:
            continue

        value = record[field_name]

        # Handle None for nullable fields
        if value is None and type(None) in (expected_type if isinstance(expected_type, tuple) else (expected_type,)):
            continue

        if not isinstance(value, expected_type):
            violations.append(Violation(
                check_name="check_field_types",
                check_category="schema",
                severity=ViolationSeverity.ERROR,
                field=field_name,
                message=f"Field {field_name} has wrong type",
                expected=str(expected_type),
                actual=str(type(value).__name__)
            ))

    return violations


def _check_enum_values(record: dict) -> list[Violation]:
    """Check enum field values are valid."""
    violations = []

    enum_specs = {
        "mode": VALID_MODES,
        "slice": VALID_SLICES,
        "metric_type": VALID_METRIC_TYPES,
        "phase": VALID_PHASES,
    }

    for field_name, valid_values in enum_specs.items():
        if field_name not in record:
            continue

        value = record[field_name]
        if value not in valid_values:
            violations.append(Violation(
                check_name="check_enum_values",
                check_category="schema",
                severity=ViolationSeverity.ERROR,
                field=field_name,
                message=f"Invalid enum value for {field_name}",
                expected=str(valid_values),
                actual=str(value)
            ))

    return violations


def _check_value_ranges(record: dict) -> list[Violation]:
    """Check numeric values are in valid ranges."""
    violations = []

    range_specs = {
        "p_success": (0.0, 1.0),
        "p_base": (0.0, 1.0),
        "p_rfl": (0.0, 1.0),
        "p_value": (0.0, 1.0),
        "cycle": (0, float("inf")),
        "n_cycles": (1, float("inf")),
        "n_base": (1, float("inf")),
        "n_rfl": (1, float("inf")),
    }

    for field_name, (min_val, max_val) in range_specs.items():
        if field_name not in record:
            continue

        value = record[field_name]
        if value is None:
            continue

        if not isinstance(value, (int, float)):
            continue

        if value < min_val or value > max_val:
            violations.append(Violation(
                check_name="check_value_ranges",
                check_category="schema",
                severity=ViolationSeverity.ERROR,
                field=field_name,
                message=f"Value out of range for {field_name}",
                expected=f"[{min_val}, {max_val}]",
                actual=str(value)
            ))

    # Check ci_95 ordering
    if "ci_95" in record and isinstance(record["ci_95"], list) and len(record["ci_95"]) == 2:
        ci = record["ci_95"]
        if isinstance(ci[0], (int, float)) and isinstance(ci[1], (int, float)):
            if ci[0] > ci[1]:
                violations.append(Violation(
                    check_name="check_value_ranges",
                    check_category="schema",
                    severity=ViolationSeverity.ERROR,
                    field="ci_95",
                    message="CI lower bound must be <= upper bound",
                    expected="ci_95[0] <= ci_95[1]",
                    actual=f"[{ci[0]}, {ci[1]}]"
                ))

    return violations


def _check_field_ordering(record: dict, record_type: str, raw_json: str) -> list[Violation]:
    """Check if fields are in canonical (lexicographic) order."""
    violations = []

    # Re-serialize with sorted keys and compare
    try:
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if raw_json.strip() != canonical:
            # Check if it's just ordering or also formatting
            actual_keys = list(record.keys())
            expected_keys = sorted(actual_keys)

            if actual_keys != expected_keys:
                violations.append(Violation(
                    check_name="check_field_ordering",
                    check_category="format",
                    severity=ViolationSeverity.WARNING,
                    field=None,
                    message="Fields not in lexicographic order",
                    expected=str(expected_keys),
                    actual=str(actual_keys)
                ))
    except Exception:
        pass

    return violations


def _check_serialization_format(raw_json: str) -> list[Violation]:
    """Check JSON serialization format (no extraneous whitespace)."""
    violations = []

    # Check for spaces after colons or commas
    if ": " in raw_json or ", " in raw_json:
        violations.append(Violation(
            check_name="check_serialization_format",
            check_category="format",
            severity=ViolationSeverity.WARNING,
            field=None,
            message="Non-compact JSON serialization (contains whitespace)",
            expected="No spaces after : or ,",
            actual="Whitespace detected"
        ))

    # Check for newlines (pretty-printed)
    if "\n" in raw_json.strip():
        violations.append(Violation(
            check_name="check_serialization_format",
            check_category="format",
            severity=ViolationSeverity.WARNING,
            field=None,
            message="Multi-line JSON not allowed in canonical form",
            expected="Single line",
            actual="Multi-line"
        ))

    return violations


def _check_timestamp_format(record: dict) -> list[Violation]:
    """Check timestamp format is canonical (ISO 8601 with microseconds and Z)."""
    violations = []

    for field_name in ["ts"]:
        if field_name not in record:
            continue

        value = record[field_name]
        if not isinstance(value, str):
            continue

        if not TIMESTAMP_PATTERN.match(value):
            violations.append(Violation(
                check_name="check_timestamp_format",
                check_category="format",
                severity=ViolationSeverity.CRITICAL,  # Must be CRITICAL to trigger quarantine
                field=field_name,
                message="Timestamp not in canonical format",
                expected="YYYY-MM-DDTHH:MM:SS.ffffffZ",
                actual=value
            ))
        else:
            # Validate it's a real timestamp
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                violations.append(Violation(
                    check_name="check_timestamp_format",
                    check_category="format",
                    severity=ViolationSeverity.CRITICAL,  # Must be CRITICAL to trigger quarantine
                    field=field_name,
                    message="Invalid timestamp value",
                    expected="Valid ISO 8601 datetime",
                    actual=value
                ))

    return violations


def _check_hash_format(record: dict) -> list[Violation]:
    """Check hash fields are 64 lowercase hex characters."""
    violations = []

    for field_name in ["ht", "r_t", "u_t"]:
        if field_name not in record:
            continue

        value = record[field_name]
        if not isinstance(value, str):
            continue

        if not HASH_64_PATTERN.match(value):
            violations.append(Violation(
                check_name="check_hash_format",
                check_category="format",
                severity=ViolationSeverity.CRITICAL,
                field=field_name,
                message="Hash not in canonical format",
                expected="64 lowercase hex characters",
                actual=f"'{value[:20]}...' (len={len(value)})"
            ))

    return violations


def _check_cross_field_consistency(record: dict, record_type: str) -> list[Violation]:
    """Check derived fields are consistent with source fields."""
    violations = []

    if record_type == "uplift_result":
        # Check uplift_delta = p_rfl - p_base
        if all(k in record for k in ["uplift_delta", "p_rfl", "p_base"]):
            p_rfl = record["p_rfl"]
            p_base = record["p_base"]
            delta = record["uplift_delta"]

            if all(isinstance(v, (int, float)) for v in [p_rfl, p_base, delta]):
                expected_delta = p_rfl - p_base
                if abs(expected_delta - delta) > 1e-6:
                    violations.append(Violation(
                        check_name="check_cross_field_consistency",
                        check_category="semantic",
                        severity=ViolationSeverity.CRITICAL,
                        field="uplift_delta",
                        message="uplift_delta does not equal p_rfl - p_base",
                        expected=str(expected_delta),
                        actual=str(delta)
                    ))

        # Check significant matches CI
        if "ci_95" in record and "significant" in record:
            ci = record["ci_95"]
            sig = record["significant"]

            if isinstance(ci, list) and len(ci) == 2 and isinstance(sig, bool):
                ci_lower = ci[0]
                if isinstance(ci_lower, (int, float)):
                    expected_sig = ci_lower > 0
                    if expected_sig != sig:
                        violations.append(Violation(
                            check_name="check_cross_field_consistency",
                            check_category="semantic",
                            severity=ViolationSeverity.CRITICAL,
                            field="significant",
                            message="significant flag inconsistent with CI",
                            expected=str(expected_sig),
                            actual=str(sig)
                        ))

    if record_type == "experiment_summary":
        # Check p_success is within ci_95
        if "ci_95" in record and "p_success" in record:
            ci = record["ci_95"]
            p = record["p_success"]

            if isinstance(ci, list) and len(ci) == 2 and isinstance(p, (int, float)):
                if all(isinstance(c, (int, float)) for c in ci):
                    if not (ci[0] <= p <= ci[1]):
                        violations.append(Violation(
                            check_name="check_cross_field_consistency",
                            check_category="semantic",
                            severity=ViolationSeverity.CRITICAL,
                            field="p_success",
                            message="p_success not within ci_95 bounds",
                            expected=f"[{ci[0]}, {ci[1]}]",
                            actual=str(p)
                        ))

    return violations


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_record_level(
    raw_json: str,
    record_type: Optional[str] = None
) -> ConformanceResult:
    """
    Classify a telemetry record's conformance level.

    Args:
        raw_json: Raw JSON string of the record
        record_type: Optional record type hint; auto-detected if not provided

    Returns:
        ConformanceResult with level and any violations
    """
    violations: list[Violation] = []

    # Step 1: Parse JSON (L0 check)
    try:
        record = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return ConformanceResult(
            level=ConformanceLevel.QUARANTINE,
            violations=[Violation(
                check_name="json_parse",
                check_category="schema",
                severity=ViolationSeverity.CRITICAL,
                field=None,
                message=f"JSON parse error: {e}",
                expected="Valid JSON",
                actual=raw_json[:100] + "..." if len(raw_json) > 100 else raw_json
            )],
            raw_json=raw_json
        )

    if not isinstance(record, dict):
        return ConformanceResult(
            level=ConformanceLevel.QUARANTINE,
            violations=[Violation(
                check_name="json_structure",
                check_category="schema",
                severity=ViolationSeverity.CRITICAL,
                field=None,
                message="Record must be a JSON object",
                expected="object",
                actual=str(type(record).__name__)
            )],
            raw_json=raw_json,
            parsed_record=record
        )

    # Step 2: Detect record type
    detected_type = record_type or _detect_record_type(record)
    if detected_type is None:
        return ConformanceResult(
            level=ConformanceLevel.L0,
            violations=[Violation(
                check_name="detect_record_type",
                check_category="schema",
                severity=ViolationSeverity.WARNING,
                field=None,
                message="Could not detect record type from field signature",
                expected="cycle_metric, experiment_summary, or uplift_result",
                actual="unknown"
            )],
            raw_json=raw_json,
            parsed_record=record
        )

    # Step 3: Schema validation (L1 checks)
    violations.extend(_check_required_fields(record, detected_type))
    violations.extend(_check_prohibited_fields(record))
    violations.extend(_check_field_types(record, detected_type))
    violations.extend(_check_enum_values(record))
    violations.extend(_check_value_ranges(record))

    # Check for CRITICAL violations that require quarantine
    critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
    error_violations = [v for v in violations if v.severity == ViolationSeverity.ERROR]

    if critical_violations:
        return ConformanceResult(
            level=ConformanceLevel.QUARANTINE,
            violations=violations,
            record_type=detected_type,
            raw_json=raw_json,
            parsed_record=record
        )

    if error_violations:
        return ConformanceResult(
            level=ConformanceLevel.QUARANTINE,
            violations=violations,
            record_type=detected_type,
            raw_json=raw_json,
            parsed_record=record
        )

    # At this point we have L1 conformance

    # Step 4: Canonical form checks (L2 checks)
    violations.extend(_check_field_ordering(record, detected_type, raw_json))
    violations.extend(_check_serialization_format(raw_json))
    violations.extend(_check_timestamp_format(record))
    violations.extend(_check_hash_format(record))
    violations.extend(_check_cross_field_consistency(record, detected_type))

    # Check for CRITICAL violations from L2 checks
    critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
    if critical_violations:
        return ConformanceResult(
            level=ConformanceLevel.QUARANTINE,
            violations=violations,
            record_type=detected_type,
            raw_json=raw_json,
            parsed_record=record
        )

    # Check for format violations that prevent L2
    format_warnings = [v for v in violations if v.check_category == "format"]
    if format_warnings:
        return ConformanceResult(
            level=ConformanceLevel.L1,
            violations=violations,
            record_type=detected_type,
            raw_json=raw_json,
            parsed_record=record
        )

    # Compute canonical hash for L2 records
    canonical_json = json.dumps(record, sort_keys=True, separators=(",", ":"))
    canonical_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return ConformanceResult(
        level=ConformanceLevel.L2,
        violations=violations,
        record_type=detected_type,
        canonical_hash=canonical_hash,
        raw_json=raw_json,
        parsed_record=record
    )


# =============================================================================
# QUARANTINE PIPELINE
# =============================================================================

@dataclass
class QuarantineEnvelope:
    """Quarantine envelope per TELEMETRY_CONFORMANCE_SPEC.md Section 4.3."""
    id: str
    quarantined_at: str
    quarantine_reason: str
    severity: str
    source_file: Optional[str]
    source_line: Optional[int]
    source_byte_offset: Optional[int]
    detector_version: str
    canonical_form_version: str
    violations: list[dict]
    raw_json: str
    parsed_record: Optional[dict]
    record_type: Optional[str]
    run_id: Optional[str]
    cycle: Optional[int]
    preceding_record_hash: Optional[str]
    following_record_hash: Optional[str]
    run_record_count: Optional[int]
    run_quarantine_count: Optional[int]
    status: str = "pending"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "quarantine": {
                "id": self.id,
                "quarantined_at": self.quarantined_at,
                "quarantine_reason": self.quarantine_reason,
                "severity": self.severity,
                "source_file": self.source_file,
                "source_line": self.source_line,
                "source_byte_offset": self.source_byte_offset,
                "detector_version": self.detector_version,
                "canonical_form_version": self.canonical_form_version,
            },
            "violations": self.violations,
            "record": {
                "raw_json": self.raw_json,
                "parsed": self.parsed_record,
                "record_type": self.record_type,
                "run_id": self.run_id,
                "cycle": self.cycle,
            },
            "context": {
                "preceding_record_hash": self.preceding_record_hash,
                "following_record_hash": self.following_record_hash,
                "run_record_count": self.run_record_count,
                "run_quarantine_count": self.run_quarantine_count,
            },
            "disposition": {
                "status": self.status,
                "diagnosed_at": None,
                "diagnosed_by": None,
                "diagnosis": None,
                "resolved_at": None,
                "resolution": None,
            },
        }


def _get_max_severity(violations: list[Violation]) -> str:
    """Get maximum severity from violations list."""
    severity_order = {
        ViolationSeverity.INFO: 0,
        ViolationSeverity.WARNING: 1,
        ViolationSeverity.ERROR: 2,
        ViolationSeverity.CRITICAL: 3,
    }
    max_sev = ViolationSeverity.INFO
    for v in violations:
        if severity_order[v.severity] > severity_order[max_sev]:
            max_sev = v.severity
    return max_sev.value


def create_quarantine_envelope(
    result: ConformanceResult,
    source_file: Optional[str] = None,
    source_line: Optional[int] = None,
    source_byte_offset: Optional[int] = None,
    preceding_hash: Optional[str] = None,
    following_hash: Optional[str] = None,
    run_record_count: Optional[int] = None,
    run_quarantine_count: Optional[int] = None,
) -> QuarantineEnvelope:
    """
    Create a quarantine envelope for a non-conforming record.

    Args:
        result: ConformanceResult from classify_record_level
        source_file: Path to source JSONL file
        source_line: Line number in source file
        source_byte_offset: Byte offset in source file
        preceding_hash: Hash of previous valid record
        following_hash: Hash of next valid record
        run_record_count: Total records in run
        run_quarantine_count: Quarantined records in run

    Returns:
        QuarantineEnvelope ready for serialization
    """
    # Extract run_id and cycle if available
    run_id = None
    cycle = None
    if result.parsed_record:
        run_id = result.parsed_record.get("run_id")
        cycle = result.parsed_record.get("cycle")

    # Build violations list
    violations_dicts = [
        {
            "check_name": v.check_name,
            "check_category": v.check_category,
            "severity": v.severity.value,
            "field": v.field,
            "message": v.message,
            "expected": v.expected,
            "actual": v.actual,
        }
        for v in result.violations
    ]

    # Create primary reason summary
    if result.violations:
        primary = result.violations[0]
        reason = f"{primary.check_name}: {primary.message}"
    else:
        reason = "Unknown conformance failure"

    return QuarantineEnvelope(
        id=str(uuid.uuid4()),
        quarantined_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        quarantine_reason=reason,
        severity=_get_max_severity(result.violations),
        source_file=source_file,
        source_line=source_line,
        source_byte_offset=source_byte_offset,
        detector_version="1.0.0",
        canonical_form_version="1.0",
        violations=violations_dicts,
        raw_json=result.raw_json or "",
        parsed_record=result.parsed_record,
        record_type=result.record_type,
        run_id=run_id,
        cycle=cycle,
        preceding_record_hash=preceding_hash,
        following_record_hash=following_hash,
        run_record_count=run_record_count,
        run_quarantine_count=run_quarantine_count,
    )


def write_quarantine_record(
    envelope: QuarantineEnvelope,
    quarantine_root: Path,
) -> Path:
    """
    Write a quarantine envelope to the quarantine directory structure.

    Args:
        envelope: QuarantineEnvelope to write
        quarantine_root: Root quarantine directory (e.g., results/quarantine)

    Returns:
        Path to written quarantine file
    """
    quarantine_root = Path(quarantine_root)

    # Ensure directories exist
    run_id = envelope.run_id or "unknown"
    run_dir = quarantine_root / "by_run" / run_id / "records"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write individual record
    cycle = envelope.cycle
    if cycle is not None:
        filename = f"{cycle:05d}.json"
    else:
        filename = f"{envelope.id}.json"

    record_path = run_dir / filename
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(envelope.to_dict(), f, indent=2)

    # Update index
    index_path = quarantine_root / "index.jsonl"
    index_entry = {
        "id": envelope.id,
        "quarantined_at": envelope.quarantined_at,
        "run_id": run_id,
        "cycle": envelope.cycle,
        "severity": envelope.severity,
        "primary_violation": envelope.violations[0]["check_name"] if envelope.violations else "unknown",
        "record_path": str(record_path.relative_to(quarantine_root)),
        "status": envelope.status,
    }

    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(index_entry) + "\n")

    # Update run manifest
    manifest_path = quarantine_root / "by_run" / run_id / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "run_id": run_id,
            "first_quarantine_at": envelope.quarantined_at,
            "last_quarantine_at": envelope.quarantined_at,
            "total_quarantined": 0,
            "by_severity": {"critical": 0, "error": 0, "warning": 0},
            "by_check": {},
            "affected_cycles": [],
            "run_invalidated": False,
            "invalidation_reason": None,
        }

    manifest["last_quarantine_at"] = envelope.quarantined_at
    manifest["total_quarantined"] += 1
    manifest["by_severity"][envelope.severity] = manifest["by_severity"].get(envelope.severity, 0) + 1

    for v in envelope.violations:
        check = v["check_name"]
        manifest["by_check"][check] = manifest["by_check"].get(check, 0) + 1

    if envelope.cycle is not None and envelope.cycle not in manifest["affected_cycles"]:
        manifest["affected_cycles"].append(envelope.cycle)
        manifest["affected_cycles"].sort()

    # Check if run should be invalidated (3+ critical or hash chain break)
    if manifest["by_severity"].get("critical", 0) >= 3:
        manifest["run_invalidated"] = True
        manifest["invalidation_reason"] = "Exceeded critical violation threshold"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return record_path


# =============================================================================
# BATCH AUDITOR
# =============================================================================

@dataclass
class ConformanceReport:
    """Conformance audit report for a telemetry file."""
    file_path: str
    audit_timestamp: str
    total_records: int
    by_level: dict[str, int]
    by_record_type: dict[str, int]
    violations_by_check: dict[str, int]
    violations_by_severity: dict[str, int]
    quarantined_count: int
    l2_percentage: float
    sample_violations: list[dict]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "audit_timestamp": self.audit_timestamp,
            "total_records": self.total_records,
            "by_level": self.by_level,
            "by_record_type": self.by_record_type,
            "violations_by_check": self.violations_by_check,
            "violations_by_severity": self.violations_by_severity,
            "quarantined_count": self.quarantined_count,
            "l2_percentage": self.l2_percentage,
            "sample_violations": self.sample_violations,
        }


def audit_telemetry_file(
    file_path: str | Path,
    quarantine_root: Optional[str | Path] = None,
    max_sample_violations: int = 10,
) -> ConformanceReport:
    """
    Audit a JSONL telemetry file for conformance.

    Args:
        file_path: Path to JSONL file to audit
        quarantine_root: Optional quarantine directory; if provided, non-conforming
                        records will be quarantined
        max_sample_violations: Maximum number of sample violations to include in report

    Returns:
        ConformanceReport with audit results
    """
    file_path = Path(file_path)

    by_level: dict[str, int] = {"L0": 0, "L1": 0, "L2": 0, "QUARANTINE": 0}
    by_record_type: dict[str, int] = {}
    violations_by_check: dict[str, int] = {}
    violations_by_severity: dict[str, int] = {}
    sample_violations: list[dict] = []
    quarantine_count = 0

    total_records = 0
    byte_offset = 0
    preceding_hash: Optional[str] = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if not line:
                byte_offset += len(line) + 1
                continue

            total_records += 1
            result = classify_record_level(line)

            # Update level counts
            by_level[result.level.value] += 1

            # Update record type counts
            if result.record_type:
                by_record_type[result.record_type] = by_record_type.get(result.record_type, 0) + 1

            # Update violation counts
            for v in result.violations:
                violations_by_check[v.check_name] = violations_by_check.get(v.check_name, 0) + 1
                violations_by_severity[v.severity.value] = violations_by_severity.get(v.severity.value, 0) + 1

                # Collect sample violations
                if len(sample_violations) < max_sample_violations:
                    sample_violations.append({
                        "line": line_num,
                        "check": v.check_name,
                        "severity": v.severity.value,
                        "message": v.message,
                    })

            # Handle quarantine
            if result.requires_quarantine:
                quarantine_count += 1
                if quarantine_root:
                    envelope = create_quarantine_envelope(
                        result,
                        source_file=str(file_path),
                        source_line=line_num,
                        source_byte_offset=byte_offset,
                        preceding_hash=preceding_hash,
                        run_record_count=total_records,
                        run_quarantine_count=quarantine_count,
                    )
                    write_quarantine_record(envelope, Path(quarantine_root))

            # Update preceding hash for L2 records
            if result.is_canonical and result.canonical_hash:
                preceding_hash = result.canonical_hash

            byte_offset += len(line) + 1

    # Calculate L2 percentage
    l2_pct = (by_level["L2"] / total_records * 100) if total_records > 0 else 0.0

    return ConformanceReport(
        file_path=str(file_path),
        audit_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        total_records=total_records,
        by_level=by_level,
        by_record_type=by_record_type,
        violations_by_check=violations_by_check,
        violations_by_severity=violations_by_severity,
        quarantined_count=quarantine_count,
        l2_percentage=l2_pct,
        sample_violations=sample_violations,
    )


def write_conformance_report(
    report: ConformanceReport,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Write conformance report to JSON file.

    Args:
        report: ConformanceReport to write
        output_path: Optional output path; defaults to same directory as audited file

    Returns:
        Path to written report file
    """
    if output_path is None:
        source = Path(report.file_path)
        output_path = source.parent / "telemetry_conformance_report.json"
    else:
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    return output_path


# =============================================================================
# TASK 1: TELEMETRY CONFORMANCE SNAPSHOT
# =============================================================================

# Schema version for conformance snapshots
CONFORMANCE_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


def build_telemetry_conformance_snapshot(report: ConformanceReport) -> dict[str, Any]:
    """
    Build a deterministic JSON-serializable snapshot summarizing telemetry conformance.

    This snapshot provides a stable, versioned summary suitable for:
    - Embedding in governance artifacts
    - Integration with last-mile checklists
    - Global health dashboards

    Args:
        report: ConformanceReport from audit_telemetry_file

    Returns:
        Dictionary with conformance snapshot data
    """
    total = report.total_records
    quarantine_ratio = (report.quarantined_count / total) if total > 0 else 0.0

    return {
        "schema_version": CONFORMANCE_SNAPSHOT_SCHEMA_VERSION,
        "total_records": total,
        "by_level": {
            "L0": report.by_level.get("L0", 0),
            "L1": report.by_level.get("L1", 0),
            "L2": report.by_level.get("L2", 0),
            "QUARANTINE": report.by_level.get("QUARANTINE", 0),
        },
        "quarantine_count": report.quarantined_count,
        "quarantine_ratio": round(quarantine_ratio, 6),
        "l2_percentage": round(report.l2_percentage, 2),
        "severity_mix": {
            "critical": report.violations_by_severity.get("critical", 0),
            "error": report.violations_by_severity.get("error", 0),
            "warning": report.violations_by_severity.get("warning", 0),
            "info": report.violations_by_severity.get("info", 0),
        },
    }


# =============================================================================
# TASK 2: LIGHTWEIGHT STREAMING HOOK API
# =============================================================================

def check_record_conformance(
    record_dict: dict[str, Any],
    record_type: Optional[str] = None
) -> ConformanceResult:
    """
    Check conformance of a single record (streaming-safe).

    This function is designed for use in streaming pipelines where records
    are processed one at a time. It is:
    - Idempotent: same input always produces same output
    - Deterministic: no random or time-dependent behavior in classification
    - Thread-safe: no shared mutable state

    Args:
        record_dict: The telemetry record as a dictionary
        record_type: Optional record type hint; auto-detected if not provided

    Returns:
        ConformanceResult with level and any violations

    Example:
        >>> record = {"cycle": 1, "mode": "rfl", ...}
        >>> result = check_record_conformance(record, "cycle_metric")
        >>> if should_quarantine(result):
        ...     handle_quarantine(result)
    """
    # Serialize to canonical JSON for classification
    raw_json = json.dumps(record_dict, sort_keys=True, separators=(",", ":"))
    return classify_record_level(raw_json, record_type)


def should_quarantine(result: ConformanceResult) -> bool:
    """
    Determine if a conformance result requires quarantine (streaming-safe).

    This is a simple predicate for use in streaming pipelines to decide
    whether a record should be quarantined.

    Args:
        result: ConformanceResult from check_record_conformance or classify_record_level

    Returns:
        True if the record should be quarantined, False otherwise

    Example:
        >>> result = check_record_conformance(record)
        >>> if should_quarantine(result):
        ...     quarantine_record(record)
    """
    return result.requires_quarantine


def is_canonical(result: ConformanceResult) -> bool:
    """
    Check if a conformance result achieved L2 (canonical) conformance (streaming-safe).

    Args:
        result: ConformanceResult from check_record_conformance or classify_record_level

    Returns:
        True if the record is L2-conformant, False otherwise
    """
    return result.is_canonical


def is_schema_valid(result: ConformanceResult) -> bool:
    """
    Check if a conformance result achieved at least L1 (schema-valid) conformance (streaming-safe).

    Args:
        result: ConformanceResult from check_record_conformance or classify_record_level

    Returns:
        True if the record is at least L1-conformant, False otherwise
    """
    return result.is_valid


# =============================================================================
# TASK 3: GOVERNANCE INTEGRATION SIGNAL
# =============================================================================

# Default threshold for healthy telemetry (max quarantine ratio)
DEFAULT_QUARANTINE_THRESHOLD = 0.01  # 1%


def summarize_telemetry_for_governance(
    snapshot: dict[str, Any],
    quarantine_threshold: float = DEFAULT_QUARANTINE_THRESHOLD
) -> dict[str, Any]:
    """
    Generate a minimal telemetry signal for governance tools.

    This summary is designed to be embedded in other JSON artifacts and consumed by:
    - CLAUDE K's last-mile checklist
    - CLAUDE I's governance verifier
    - CLAUDE O's MAAS engine
    - Global health dashboards

    The `is_telemetry_healthy` flag uses an explicit threshold to determine health.
    By default, telemetry is considered healthy if quarantine_ratio <= 1%.

    Args:
        snapshot: Conformance snapshot from build_telemetry_conformance_snapshot
        quarantine_threshold: Maximum acceptable quarantine ratio (default: 0.01 = 1%)

    Returns:
        Dictionary with governance-relevant telemetry signals

    Example:
        >>> report = audit_telemetry_file("telemetry.jsonl")
        >>> snapshot = build_telemetry_conformance_snapshot(report)
        >>> gov_signal = summarize_telemetry_for_governance(snapshot)
        >>> if not gov_signal["is_telemetry_healthy"]:
        ...     raise TelemetryHealthError("Telemetry failed health check")
    """
    quarantine_ratio = snapshot.get("quarantine_ratio", 0.0)
    quarantine_count = snapshot.get("quarantine_count", 0)
    has_quarantine = quarantine_count > 0
    is_healthy = quarantine_ratio <= quarantine_threshold

    return {
        "has_quarantine_records": has_quarantine,
        "quarantine_ratio": quarantine_ratio,
        "quarantine_count": quarantine_count,
        "is_telemetry_healthy": is_healthy,
        "health_threshold": quarantine_threshold,
        "total_records": snapshot.get("total_records", 0),
        "l2_percentage": snapshot.get("l2_percentage", 0.0),
        "critical_violations": snapshot.get("severity_mix", {}).get("critical", 0),
    }


# =============================================================================
# PHASE III: TELEMETRY SLO ENGINE
# =============================================================================

class SLOStatus(Enum):
    """SLO evaluation status."""
    OK = "OK"
    WARN = "WARN"
    BREACH = "BREACH"


@dataclass
class SLORule:
    """A single SLO rule definition."""
    name: str
    metric: str
    operator: str  # "<=", ">=", "<", ">", "=="
    warn_threshold: float
    breach_threshold: float
    description: str = ""

    def evaluate(self, value: float) -> SLOStatus:
        """Evaluate this rule against a value."""
        if self.operator == "<=":
            if value > self.breach_threshold:
                return SLOStatus.BREACH
            elif value > self.warn_threshold:
                return SLOStatus.WARN
            return SLOStatus.OK
        elif self.operator == ">=":
            if value < self.breach_threshold:
                return SLOStatus.BREACH
            elif value < self.warn_threshold:
                return SLOStatus.WARN
            return SLOStatus.OK
        elif self.operator == "<":
            if value >= self.breach_threshold:
                return SLOStatus.BREACH
            elif value >= self.warn_threshold:
                return SLOStatus.WARN
            return SLOStatus.OK
        elif self.operator == ">":
            if value <= self.breach_threshold:
                return SLOStatus.BREACH
            elif value <= self.warn_threshold:
                return SLOStatus.WARN
            return SLOStatus.OK
        elif self.operator == "==":
            if value != self.breach_threshold:
                return SLOStatus.BREACH
            return SLOStatus.OK
        return SLOStatus.OK


@dataclass
class SLOViolation:
    """A violation of an SLO rule."""
    rule_name: str
    metric: str
    status: SLOStatus
    actual_value: float
    warn_threshold: float
    breach_threshold: float
    description: str


@dataclass
class SLOResult:
    """Result of SLO evaluation."""
    slo_status: SLOStatus
    violated_rules: list[SLOViolation]
    evaluated_rules: int
    passed_rules: int
    warn_rules: int
    breach_rules: int
    snapshot_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slo_status": self.slo_status.value,
            "violated_rules": [
                {
                    "rule_name": v.rule_name,
                    "metric": v.metric,
                    "status": v.status.value,
                    "actual_value": v.actual_value,
                    "warn_threshold": v.warn_threshold,
                    "breach_threshold": v.breach_threshold,
                    "description": v.description,
                }
                for v in self.violated_rules
            ],
            "evaluated_rules": self.evaluated_rules,
            "passed_rules": self.passed_rules,
            "warn_rules": self.warn_rules,
            "breach_rules": self.breach_rules,
            "snapshot_summary": self.snapshot_summary,
        }


# Default SLO configuration
DEFAULT_SLO_CONFIG: list[dict[str, Any]] = [
    {
        "name": "quarantine_ratio",
        "metric": "quarantine_ratio",
        "operator": "<=",
        "warn_threshold": 0.005,   # 0.5% warn
        "breach_threshold": 0.01,  # 1% breach
        "description": "Quarantine rate must stay below 1%",
    },
    {
        "name": "l2_conformance",
        "metric": "l2_percentage",
        "operator": ">=",
        "warn_threshold": 95.0,    # 95% warn
        "breach_threshold": 90.0,  # 90% breach
        "description": "L2 (canonical) conformance should be at least 90%",
    },
    {
        "name": "critical_violations",
        "metric": "severity_mix.critical",
        "operator": "<=",
        "warn_threshold": 1,       # 1 critical = warn
        "breach_threshold": 3,     # 3+ critical = breach
        "description": "Critical violations must be minimized",
    },
    {
        "name": "minimum_records",
        "metric": "total_records",
        "operator": ">=",
        "warn_threshold": 10,      # 10 records = warn
        "breach_threshold": 1,     # 0 records = breach
        "description": "Must have at least 1 telemetry record",
    },
]


def _get_metric_value(snapshot: dict[str, Any], metric_path: str) -> Optional[float]:
    """Extract a metric value from snapshot using dot notation."""
    parts = metric_path.split(".")
    value = snapshot
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_slo_config(slo_cfg: list[dict[str, Any]]) -> list[SLORule]:
    """Parse SLO configuration into SLORule objects."""
    rules = []
    for cfg in slo_cfg:
        rules.append(SLORule(
            name=cfg["name"],
            metric=cfg["metric"],
            operator=cfg["operator"],
            warn_threshold=cfg["warn_threshold"],
            breach_threshold=cfg["breach_threshold"],
            description=cfg.get("description", ""),
        ))
    return rules


def evaluate_telemetry_slo(
    snapshot: dict[str, Any],
    slo_cfg: Optional[list[dict[str, Any]]] = None
) -> SLOResult:
    """
    Evaluate telemetry conformance snapshot against SLO rules.

    This function checks the telemetry snapshot against a set of Service Level
    Objective (SLO) rules and returns the overall status and any violations.

    Args:
        snapshot: Conformance snapshot from build_telemetry_conformance_snapshot
        slo_cfg: Optional list of SLO rule configurations. Uses DEFAULT_SLO_CONFIG if not provided.

    Returns:
        SLOResult with overall status and list of violations

    Example:
        >>> snapshot = build_telemetry_conformance_snapshot(report)
        >>> slo_result = evaluate_telemetry_slo(snapshot)
        >>> if slo_result.slo_status == SLOStatus.BREACH:
        ...     handle_slo_breach(slo_result)
    """
    if slo_cfg is None:
        slo_cfg = DEFAULT_SLO_CONFIG

    rules = _parse_slo_config(slo_cfg)
    violations: list[SLOViolation] = []
    passed = 0
    warn_count = 0
    breach_count = 0

    for rule in rules:
        value = _get_metric_value(snapshot, rule.metric)
        if value is None:
            # Skip rules for missing metrics
            continue

        status = rule.evaluate(value)

        if status == SLOStatus.OK:
            passed += 1
        elif status == SLOStatus.WARN:
            warn_count += 1
            violations.append(SLOViolation(
                rule_name=rule.name,
                metric=rule.metric,
                status=status,
                actual_value=value,
                warn_threshold=rule.warn_threshold,
                breach_threshold=rule.breach_threshold,
                description=rule.description,
            ))
        elif status == SLOStatus.BREACH:
            breach_count += 1
            violations.append(SLOViolation(
                rule_name=rule.name,
                metric=rule.metric,
                status=status,
                actual_value=value,
                warn_threshold=rule.warn_threshold,
                breach_threshold=rule.breach_threshold,
                description=rule.description,
            ))

    # Determine overall status
    if breach_count > 0:
        overall_status = SLOStatus.BREACH
    elif warn_count > 0:
        overall_status = SLOStatus.WARN
    else:
        overall_status = SLOStatus.OK

    return SLOResult(
        slo_status=overall_status,
        violated_rules=violations,
        evaluated_rules=len(rules),
        passed_rules=passed,
        warn_rules=warn_count,
        breach_rules=breach_count,
        snapshot_summary={
            "total_records": snapshot.get("total_records", 0),
            "quarantine_ratio": snapshot.get("quarantine_ratio", 0.0),
            "l2_percentage": snapshot.get("l2_percentage", 0.0),
        },
    )


# =============================================================================
# PHASE III: AUTO-QUARANTINE ENGINE
# =============================================================================

class QuarantineAction(Enum):
    """Recommended quarantine action."""
    ALLOW_PUBLISH = "allow_publish"
    QUARANTINE_RUN = "quarantine_run"
    QUARANTINE_AND_ALERT = "quarantine_and_alert"
    BLOCK_PIPELINE = "block_pipeline"


@dataclass
class QuarantineDecision:
    """Auto-quarantine decision result."""
    publish_allowed: bool
    quarantine_required: bool
    recommended_action: QuarantineAction
    reasons: list[str]
    slo_status: SLOStatus
    breach_count: int
    warn_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "publish_allowed": self.publish_allowed,
            "quarantine_required": self.quarantine_required,
            "recommended_action": self.recommended_action.value,
            "reasons": self.reasons,
            "slo_status": self.slo_status.value,
            "breach_count": self.breach_count,
            "warn_count": self.warn_count,
        }


def decide_telemetry_quarantine(
    snapshot: dict[str, Any],
    slo_result: SLOResult
) -> QuarantineDecision:
    """
    Decide whether telemetry run should be quarantined based on SLO evaluation.

    This function implements the auto-quarantine brain logic:
    - OK status: Allow publish, no quarantine
    - WARN status: Allow publish with warnings, soft quarantine
    - BREACH status: Block publish, hard quarantine required

    Args:
        snapshot: Conformance snapshot from build_telemetry_conformance_snapshot
        slo_result: SLO evaluation result from evaluate_telemetry_slo

    Returns:
        QuarantineDecision with publish/quarantine flags and recommended action

    Example:
        >>> snapshot = build_telemetry_conformance_snapshot(report)
        >>> slo_result = evaluate_telemetry_slo(snapshot)
        >>> decision = decide_telemetry_quarantine(snapshot, slo_result)
        >>> if not decision.publish_allowed:
        ...     quarantine_telemetry_run(run_id)
    """
    reasons: list[str] = []
    quarantine_count = snapshot.get("quarantine_count", 0)
    total_records = snapshot.get("total_records", 0)

    # Collect reasons from SLO violations
    for violation in slo_result.violated_rules:
        severity = "BREACH" if violation.status == SLOStatus.BREACH else "WARN"
        reasons.append(
            f"[{severity}] {violation.rule_name}: {violation.actual_value:.4f} "
            f"(threshold: {violation.breach_threshold})"
        )

    # Decision logic based on SLO status
    if slo_result.slo_status == SLOStatus.OK:
        return QuarantineDecision(
            publish_allowed=True,
            quarantine_required=False,
            recommended_action=QuarantineAction.ALLOW_PUBLISH,
            reasons=reasons if reasons else ["All SLO rules passed"],
            slo_status=slo_result.slo_status,
            breach_count=slo_result.breach_rules,
            warn_count=slo_result.warn_rules,
        )

    elif slo_result.slo_status == SLOStatus.WARN:
        # Warnings allow publish but recommend monitoring
        return QuarantineDecision(
            publish_allowed=True,
            quarantine_required=False,
            recommended_action=QuarantineAction.ALLOW_PUBLISH,
            reasons=reasons,
            slo_status=slo_result.slo_status,
            breach_count=slo_result.breach_rules,
            warn_count=slo_result.warn_rules,
        )

    else:  # BREACH
        # Determine severity of breach
        critical_breaches = sum(
            1 for v in slo_result.violated_rules
            if v.status == SLOStatus.BREACH and v.metric in ["quarantine_ratio", "severity_mix.critical"]
        )

        if critical_breaches >= 2 or quarantine_count > total_records * 0.1:
            # Severe breach - block pipeline
            action = QuarantineAction.BLOCK_PIPELINE
            reasons.append("Multiple critical SLO breaches detected - pipeline blocked")
        elif quarantine_count > 0:
            # Has quarantined records - alert
            action = QuarantineAction.QUARANTINE_AND_ALERT
            reasons.append(f"{quarantine_count} records require quarantine")
        else:
            # Standard quarantine
            action = QuarantineAction.QUARANTINE_RUN

        return QuarantineDecision(
            publish_allowed=False,
            quarantine_required=True,
            recommended_action=action,
            reasons=reasons,
            slo_status=slo_result.slo_status,
            breach_count=slo_result.breach_rules,
            warn_count=slo_result.warn_rules,
        )


# =============================================================================
# PHASE III: GLOBAL HEALTH SUMMARY
# =============================================================================

@dataclass
class GlobalHealthSummary:
    """Global health summary for telemetry."""
    telemetry_ok: bool
    breach_ratio: float
    key_reasons: list[str]
    slo_status: str
    total_rules: int
    passed_rules: int
    failed_rules: int
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "telemetry_ok": self.telemetry_ok,
            "breach_ratio": self.breach_ratio,
            "key_reasons": self.key_reasons,
            "slo_status": self.slo_status,
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "recommendation": self.recommendation,
        }


def summarize_telemetry_for_global_health(
    slo_result: SLOResult
) -> GlobalHealthSummary:
    """
    Generate a global health summary from SLO evaluation results.

    This summary is designed for:
    - Global health dashboards
    - Integration with CI/CD pipelines
    - Executive reporting
    - Cross-system health aggregation

    Args:
        slo_result: SLO evaluation result from evaluate_telemetry_slo

    Returns:
        GlobalHealthSummary with telemetry health status and key reasons

    Example:
        >>> slo_result = evaluate_telemetry_slo(snapshot)
        >>> health = summarize_telemetry_for_global_health(slo_result)
        >>> dashboard.update_telemetry_health(health.to_dict())
    """
    # Calculate breach ratio
    total_rules = slo_result.evaluated_rules
    failed_rules = slo_result.warn_rules + slo_result.breach_rules
    breach_ratio = (failed_rules / total_rules) if total_rules > 0 else 0.0

    # Determine if telemetry is OK (no breaches)
    telemetry_ok = slo_result.slo_status != SLOStatus.BREACH

    # Extract key reasons (top 3 violations)
    key_reasons: list[str] = []
    sorted_violations = sorted(
        slo_result.violated_rules,
        key=lambda v: (0 if v.status == SLOStatus.BREACH else 1, v.rule_name)
    )
    for violation in sorted_violations[:3]:
        status_prefix = "❌" if violation.status == SLOStatus.BREACH else "⚠️"
        key_reasons.append(
            f"{status_prefix} {violation.rule_name}: {violation.description}"
        )

    # Generate recommendation
    if slo_result.slo_status == SLOStatus.OK:
        recommendation = "Telemetry healthy - proceed with confidence"
    elif slo_result.slo_status == SLOStatus.WARN:
        recommendation = "Telemetry has warnings - review before proceeding"
    else:
        recommendation = "Telemetry breached SLOs - investigation required"

    return GlobalHealthSummary(
        telemetry_ok=telemetry_ok,
        breach_ratio=round(breach_ratio, 4),
        key_reasons=key_reasons,
        slo_status=slo_result.slo_status.value,
        total_rules=total_rules,
        passed_rules=slo_result.passed_rules,
        failed_rules=failed_rules,
        recommendation=recommendation,
    )


# =============================================================================
# PHASE IV: TELEMETRY RELEASE GATE
# =============================================================================

class ReleaseStatus(Enum):
    """Release gate status."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class ReleaseGateResult:
    """Result of release gate evaluation."""
    release_ok: bool
    status: ReleaseStatus
    blocking_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "release_ok": self.release_ok,
            "status": self.status.value,
            "blocking_reasons": self.blocking_reasons,
        }


def evaluate_telemetry_for_release(
    slo_result: dict[str, Any],
    decision: dict[str, Any]
) -> ReleaseGateResult:
    """
    Evaluate telemetry for release gating.

    This function determines if telemetry quality meets the bar for release.
    It combines SLO evaluation results with quarantine decisions to produce
    a definitive release gate verdict.

    Args:
        slo_result: SLO evaluation result dict (from SLOResult.to_dict())
        decision: Quarantine decision dict (from QuarantineDecision.to_dict())

    Returns:
        ReleaseGateResult with release_ok, status, and blocking_reasons

    Example:
        >>> slo_result = evaluate_telemetry_slo(snapshot).to_dict()
        >>> decision = decide_telemetry_quarantine(snapshot, slo_obj).to_dict()
        >>> gate = evaluate_telemetry_for_release(slo_result, decision)
        >>> if not gate.release_ok:
        ...     abort_release(gate.blocking_reasons)
    """
    blocking_reasons: list[str] = []
    slo_status = slo_result.get("slo_status", "OK")
    publish_allowed = decision.get("publish_allowed", True)
    quarantine_required = decision.get("quarantine_required", False)

    # Determine release status based on SLO and quarantine decision
    if slo_status == "BREACH" or quarantine_required:
        status = ReleaseStatus.BLOCK
        release_ok = False

        # Collect blocking reasons from violated rules
        for violation in slo_result.get("violated_rules", []):
            if violation.get("status") == "BREACH":
                blocking_reasons.append(
                    f"SLO breach: {violation.get('rule_name')} "
                    f"({violation.get('actual_value'):.4f} vs threshold {violation.get('breach_threshold')})"
                )

        # Add quarantine-related reasons
        if quarantine_required:
            for reason in decision.get("reasons", []):
                if reason not in blocking_reasons:
                    blocking_reasons.append(reason)

        if not blocking_reasons:
            blocking_reasons.append("Telemetry SLO breach detected")

    elif slo_status == "WARN":
        status = ReleaseStatus.WARN
        release_ok = True  # Warnings don't block release

        # Collect warning reasons
        for violation in slo_result.get("violated_rules", []):
            if violation.get("status") == "WARN":
                blocking_reasons.append(
                    f"SLO warning: {violation.get('rule_name')} "
                    f"({violation.get('actual_value'):.4f} approaching threshold)"
                )

    else:
        status = ReleaseStatus.OK
        release_ok = True

    return ReleaseGateResult(
        release_ok=release_ok,
        status=status,
        blocking_reasons=blocking_reasons,
    )


# =============================================================================
# PHASE IV: MAAS TELEMETRY ADAPTER
# =============================================================================

class MAASStatus(Enum):
    """MAAS telemetry status."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


@dataclass
class MAASAdapterResult:
    """Result of MAAS telemetry adapter."""
    telemetry_admissible: bool
    status: MAASStatus
    violation_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "telemetry_admissible": self.telemetry_admissible,
            "status": self.status.value,
            "violation_codes": self.violation_codes,
        }


def summarize_telemetry_for_maas(
    slo_result: dict[str, Any],
    decision: dict[str, Any]
) -> MAASAdapterResult:
    """
    Generate telemetry summary for MAAS (Metrics-as-a-Service) integration.

    This adapter translates SLO results into MAAS-compatible format for
    integration with automated metrics pipelines and alerting systems.

    Args:
        slo_result: SLO evaluation result dict (from SLOResult.to_dict())
        decision: Quarantine decision dict (from QuarantineDecision.to_dict())

    Returns:
        MAASAdapterResult with telemetry_admissible, status, and violation_codes

    Example:
        >>> slo_result = evaluate_telemetry_slo(snapshot).to_dict()
        >>> decision = decide_telemetry_quarantine(snapshot, slo_obj).to_dict()
        >>> maas = summarize_telemetry_for_maas(slo_result, decision)
        >>> maas_pipeline.ingest(maas.to_dict())
    """
    violation_codes: list[str] = []
    slo_status = slo_result.get("slo_status", "OK")
    quarantine_required = decision.get("quarantine_required", False)

    # Extract violation codes from violated rules
    for violation in slo_result.get("violated_rules", []):
        rule_name = violation.get("rule_name", "unknown")
        violation_status = violation.get("status", "UNKNOWN")
        code = f"{rule_name.upper()}_{violation_status}"
        violation_codes.append(code)

    # Add quarantine code if required
    if quarantine_required:
        action = decision.get("recommended_action", "quarantine_run")
        violation_codes.append(f"QUARANTINE_{action.upper()}")

    # Determine MAAS status
    if slo_status == "BREACH" or quarantine_required:
        status = MAASStatus.BLOCK
        telemetry_admissible = False
    elif slo_status == "WARN":
        status = MAASStatus.ATTENTION
        telemetry_admissible = True
    else:
        status = MAASStatus.OK
        telemetry_admissible = True

    return MAASAdapterResult(
        telemetry_admissible=telemetry_admissible,
        status=status,
        violation_codes=violation_codes,
    )


# =============================================================================
# PHASE IV: DIRECTOR TELEMETRY PANEL
# =============================================================================

class StatusLight(Enum):
    """Director panel status light."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class DirectorTelemetryPanel:
    """Director telemetry panel for dashboard display."""
    status_light: StatusLight
    telemetry_ok: bool
    breach_ratio: float
    headline: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status_light": self.status_light.value,
            "telemetry_ok": self.telemetry_ok,
            "breach_ratio": self.breach_ratio,
            "headline": self.headline,
        }


def build_telemetry_director_panel(
    slo_summary: dict[str, Any],
    release_eval: dict[str, Any]
) -> DirectorTelemetryPanel:
    """
    Build a telemetry panel for the Director dashboard.

    This function creates a high-level summary suitable for executive
    dashboards and system-wide health monitoring displays.

    Args:
        slo_summary: Global health summary dict (from GlobalHealthSummary.to_dict())
        release_eval: Release gate result dict (from ReleaseGateResult.to_dict())

    Returns:
        DirectorTelemetryPanel with status_light, telemetry_ok, breach_ratio, headline

    Example:
        >>> health = summarize_telemetry_for_global_health(slo_result)
        >>> gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        >>> panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())
        >>> director_dashboard.update_tile("telemetry", panel.to_dict())
    """
    # Extract values
    telemetry_ok = slo_summary.get("telemetry_ok", True)
    breach_ratio = slo_summary.get("breach_ratio", 0.0)
    slo_status = slo_summary.get("slo_status", "OK")
    release_status = release_eval.get("status", "OK")
    release_ok = release_eval.get("release_ok", True)
    total_rules = slo_summary.get("total_rules", 0)
    passed_rules = slo_summary.get("passed_rules", 0)

    # Determine status light
    if release_status == "BLOCK" or slo_status == "BREACH":
        status_light = StatusLight.RED
    elif release_status == "WARN" or slo_status == "WARN":
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Generate headline
    if status_light == StatusLight.GREEN:
        if total_rules > 0:
            headline = f"Telemetry conformance nominal: {passed_rules}/{total_rules} SLO rules passed."
        else:
            headline = "Telemetry conformance nominal: all checks passed."
    elif status_light == StatusLight.YELLOW:
        failed_rules = slo_summary.get("failed_rules", 0)
        headline = f"Telemetry conformance attention: {failed_rules} SLO warning(s) detected."
    else:  # RED
        blocking_count = len(release_eval.get("blocking_reasons", []))
        if blocking_count > 0:
            headline = f"Telemetry conformance breach: {blocking_count} blocking issue(s) require resolution."
        else:
            headline = "Telemetry conformance breach: SLO thresholds exceeded."

    return DirectorTelemetryPanel(
        status_light=status_light,
        telemetry_ok=telemetry_ok and release_ok,
        breach_ratio=breach_ratio,
        headline=headline,
    )


# =============================================================================
# PHASE V: ALERT VIOLATION CODES
# =============================================================================

# Canonical violation codes for global console alerts
# These are the ONLY codes that should appear on the global console
class AlertCode(Enum):
    """Canonical alert violation codes for global console."""
    # Telemetry-specific alerts
    TELEMETRY_QUARANTINE_SPIKE = "TELEMETRY_QUARANTINE_SPIKE"
    TELEMETRY_L2_DEGRADATION = "TELEMETRY_L2_DEGRADATION"
    TELEMETRY_CRITICAL_VIOLATIONS = "TELEMETRY_CRITICAL_VIOLATIONS"

    # TDA correlation alerts
    TELEMETRY_TDA_MISMATCH = "TELEMETRY_TDA_MISMATCH"
    TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE = "TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE"

    # Pipeline alerts
    TELEMETRY_PIPELINE_BLOCKED = "TELEMETRY_PIPELINE_BLOCKED"


# Thresholds for alert generation
ALERT_THRESHOLDS = {
    "quarantine_spike": 0.05,      # 5% quarantine rate triggers spike alert
    "l2_degradation": 80.0,        # Below 80% L2 triggers degradation alert
    "low_hss_threshold": 0.5,      # HSS below 0.5 is considered low
    "high_quarantine_threshold": 0.02,  # 2% quarantine with low HSS = correlation
}


def get_alert_codes_for_telemetry(
    slo_result: dict[str, Any],
    decision: dict[str, Any]
) -> list[str]:
    """
    Generate canonical alert codes based on telemetry state.

    These are the violation codes that should appear on the global console.
    Only a small, well-defined set of alerts are generated.

    Args:
        slo_result: SLO evaluation result dict
        decision: Quarantine decision dict

    Returns:
        List of AlertCode values that apply
    """
    alerts: list[str] = []

    # Check for quarantine spike
    snapshot_summary = slo_result.get("snapshot_summary", {})
    quarantine_ratio = snapshot_summary.get("quarantine_ratio", 0.0)
    l2_percentage = snapshot_summary.get("l2_percentage", 100.0)

    if quarantine_ratio >= ALERT_THRESHOLDS["quarantine_spike"]:
        alerts.append(AlertCode.TELEMETRY_QUARANTINE_SPIKE.value)

    if l2_percentage < ALERT_THRESHOLDS["l2_degradation"]:
        alerts.append(AlertCode.TELEMETRY_L2_DEGRADATION.value)

    # Check for critical violations in violated rules
    critical_count = sum(
        1 for v in slo_result.get("violated_rules", [])
        if v.get("metric") == "severity_mix.critical" and v.get("status") == "BREACH"
    )
    if critical_count > 0:
        alerts.append(AlertCode.TELEMETRY_CRITICAL_VIOLATIONS.value)

    # Check for pipeline blocked
    if decision.get("recommended_action") == "block_pipeline":
        alerts.append(AlertCode.TELEMETRY_PIPELINE_BLOCKED.value)

    return alerts


# =============================================================================
# PHASE V: TDA-TELEMETRY CORRELATION
# =============================================================================

@dataclass
class TDACorrelationResult:
    """Result of TDA-Telemetry correlation analysis."""
    low_hss_high_quarantine_pattern: bool
    correlated_failures: list[str]
    tda_telemetry_mismatch: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "low_hss_high_quarantine_pattern": self.low_hss_high_quarantine_pattern,
            "correlated_failures": self.correlated_failures,
            "tda_telemetry_mismatch": self.tda_telemetry_mismatch,
            "notes": self.notes,
        }


def summarize_telemetry_tda_correlation(
    slo_result: dict[str, Any],
    tda_stats: dict[str, Any]
) -> TDACorrelationResult:
    """
    Correlate telemetry conformance with TDA (CORTEX) statistics.

    This function identifies patterns where telemetry quality issues
    correlate with TDA health metrics, serving as a reality check.

    Args:
        slo_result: SLO evaluation result dict (from SLOResult.to_dict())
        tda_stats: TDA statistics dict containing:
            - hss: float (Health Stability Score, 0-1)
            - slice_health: dict[str, float] (per-slice health scores)
            - failing_slices: list[str] (slice IDs with issues)
            - tda_ok: bool (overall TDA health)

    Returns:
        TDACorrelationResult with correlation analysis

    Example:
        >>> slo_result = evaluate_telemetry_slo(snapshot).to_dict()
        >>> tda_stats = {"hss": 0.4, "failing_slices": ["slice_A"], "tda_ok": False}
        >>> corr = summarize_telemetry_tda_correlation(slo_result, tda_stats)
        >>> if corr.low_hss_high_quarantine_pattern:
        ...     investigate_correlated_issues()
    """
    notes: list[str] = []
    correlated_failures: list[str] = []

    # Extract telemetry metrics
    snapshot_summary = slo_result.get("snapshot_summary", {})
    quarantine_ratio = snapshot_summary.get("quarantine_ratio", 0.0)
    l2_percentage = snapshot_summary.get("l2_percentage", 100.0)
    slo_status = slo_result.get("slo_status", "OK")

    # Extract TDA metrics
    hss = tda_stats.get("hss", 1.0)
    tda_ok = tda_stats.get("tda_ok", True)
    failing_slices = tda_stats.get("failing_slices", [])
    slice_health = tda_stats.get("slice_health", {})

    # Pattern 1: Low HSS + High Quarantine
    low_hss = hss < ALERT_THRESHOLDS["low_hss_threshold"]
    high_quarantine = quarantine_ratio >= ALERT_THRESHOLDS["high_quarantine_threshold"]
    low_hss_high_quarantine = low_hss and high_quarantine

    if low_hss_high_quarantine:
        notes.append(
            f"Correlation detected: Low HSS ({hss:.2f}) with elevated quarantine rate "
            f"({quarantine_ratio:.2%}). Investigate shared root cause."
        )
        # Add failing slices as correlated failures
        correlated_failures.extend(failing_slices)

    # Pattern 2: TDA-Telemetry mismatch (TDA says OK but telemetry has issues, or vice versa)
    telemetry_has_issues = slo_status in ("WARN", "BREACH") or quarantine_ratio > 0.01
    tda_telemetry_mismatch = (tda_ok and telemetry_has_issues) or (not tda_ok and slo_status == "OK")

    if tda_telemetry_mismatch:
        if tda_ok and telemetry_has_issues:
            notes.append(
                "Mismatch: TDA reports healthy but telemetry shows conformance issues. "
                "Telemetry may be detecting problems TDA hasn't surfaced yet."
            )
        elif not tda_ok and slo_status == "OK":
            notes.append(
                "Mismatch: TDA reports issues but telemetry is clean. "
                "TDA issues may not be affecting telemetry conformance."
            )

    # Pattern 3: Per-slice correlation
    for slice_id, health in slice_health.items():
        if health < 0.5:  # Unhealthy slice
            notes.append(f"Slice '{slice_id}' has low health ({health:.2f})")
            if slice_id not in correlated_failures:
                correlated_failures.append(slice_id)

    # Add summary note
    if not notes:
        notes.append("No significant TDA-Telemetry correlations detected.")

    return TDACorrelationResult(
        low_hss_high_quarantine_pattern=low_hss_high_quarantine,
        correlated_failures=correlated_failures,
        tda_telemetry_mismatch=tda_telemetry_mismatch,
        notes=notes,
    )


# =============================================================================
# PHASE V: GLOBAL CONSOLE ADAPTER V2
# =============================================================================

@dataclass
class GlobalConsoleResult:
    """Global console telemetry summary with TDA correlation."""
    telemetry_ok: bool
    status_light: StatusLight
    breach_ratio: float
    headline: str
    alert_codes: list[str]
    tda_correlation_detected: bool
    correlated_slice_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "telemetry_ok": self.telemetry_ok,
            "status_light": self.status_light.value,
            "breach_ratio": self.breach_ratio,
            "headline": self.headline,
            "alert_codes": self.alert_codes,
            "tda_correlation_detected": self.tda_correlation_detected,
            "correlated_slice_count": self.correlated_slice_count,
        }


def summarize_telemetry_for_global_console(
    slo_summary: dict[str, Any],
    release_eval: dict[str, Any],
    tda_correlation: dict[str, Any]
) -> GlobalConsoleResult:
    """
    Generate comprehensive telemetry summary for global console display.

    This is the unified view combining:
    - Telemetry SLO status
    - Release gate evaluation
    - TDA correlation analysis

    Args:
        slo_summary: Global health summary dict (from GlobalHealthSummary.to_dict())
        release_eval: Release gate result dict (from ReleaseGateResult.to_dict())
        tda_correlation: TDA correlation result dict (from TDACorrelationResult.to_dict())

    Returns:
        GlobalConsoleResult for dashboard display

    Example:
        >>> health = summarize_telemetry_for_global_health(slo_result)
        >>> gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        >>> tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)
        >>> console = summarize_telemetry_for_global_console(
        ...     health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        ... )
    """
    # Extract values
    telemetry_ok = slo_summary.get("telemetry_ok", True)
    breach_ratio = slo_summary.get("breach_ratio", 0.0)
    slo_status = slo_summary.get("slo_status", "OK")
    release_status = release_eval.get("status", "OK")
    release_ok = release_eval.get("release_ok", True)

    # TDA correlation values
    low_hss_high_quarantine = tda_correlation.get("low_hss_high_quarantine_pattern", False)
    tda_mismatch = tda_correlation.get("tda_telemetry_mismatch", False)
    correlated_failures = tda_correlation.get("correlated_failures", [])
    tda_correlation_detected = low_hss_high_quarantine or tda_mismatch

    # Determine status light (TDA correlation can escalate to yellow)
    if release_status == "BLOCK" or slo_status == "BREACH":
        status_light = StatusLight.RED
    elif release_status == "WARN" or slo_status == "WARN" or tda_correlation_detected:
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # Generate alert codes
    alert_codes: list[str] = []

    if low_hss_high_quarantine:
        alert_codes.append(AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value)

    if tda_mismatch:
        alert_codes.append(AlertCode.TELEMETRY_TDA_MISMATCH.value)

    # Add release-blocking alerts
    if release_status == "BLOCK":
        alert_codes.append(AlertCode.TELEMETRY_PIPELINE_BLOCKED.value)

    # Generate headline
    total_rules = slo_summary.get("total_rules", 0)
    passed_rules = slo_summary.get("passed_rules", 0)

    if status_light == StatusLight.GREEN:
        if total_rules > 0:
            headline = f"Telemetry nominal: {passed_rules}/{total_rules} SLO rules passed, no TDA correlation issues."
        else:
            headline = "Telemetry nominal: all checks passed."
    elif status_light == StatusLight.YELLOW:
        if tda_correlation_detected:
            headline = f"Telemetry attention: TDA correlation detected across {len(correlated_failures)} slice(s)."
        else:
            failed_rules = slo_summary.get("failed_rules", 0)
            headline = f"Telemetry attention: {failed_rules} SLO warning(s) detected."
    else:  # RED
        blocking_count = len(release_eval.get("blocking_reasons", []))
        if blocking_count > 0:
            headline = f"Telemetry breach: {blocking_count} blocking issue(s), release gated."
        else:
            headline = "Telemetry breach: SLO thresholds exceeded, release gated."

    return GlobalConsoleResult(
        telemetry_ok=telemetry_ok and release_ok and not low_hss_high_quarantine,
        status_light=status_light,
        breach_ratio=breach_ratio,
        headline=headline,
        alert_codes=alert_codes,
        tda_correlation_detected=tda_correlation_detected,
        correlated_slice_count=len(correlated_failures),
    )


# =============================================================================
# PHASE V: ENHANCED MAAS ADAPTER
# =============================================================================

def summarize_telemetry_for_maas_v2(
    slo_result: dict[str, Any],
    decision: dict[str, Any],
    tda_correlation: Optional[dict[str, Any]] = None
) -> MAASAdapterResult:
    """
    Enhanced MAAS adapter that includes TDA correlation alert codes.

    This is the v2 adapter that surfaces canonical alert codes for
    global console integration.

    Args:
        slo_result: SLO evaluation result dict
        decision: Quarantine decision dict
        tda_correlation: Optional TDA correlation result dict

    Returns:
        MAASAdapterResult with canonical alert codes
    """
    # Start with base violation codes
    violation_codes: list[str] = []
    slo_status = slo_result.get("slo_status", "OK")
    quarantine_required = decision.get("quarantine_required", False)

    # Add canonical alert codes (not raw violation codes)
    alert_codes = get_alert_codes_for_telemetry(slo_result, decision)
    violation_codes.extend(alert_codes)

    # Add TDA correlation alerts if provided
    if tda_correlation:
        if tda_correlation.get("low_hss_high_quarantine_pattern"):
            violation_codes.append(AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value)
        if tda_correlation.get("tda_telemetry_mismatch"):
            violation_codes.append(AlertCode.TELEMETRY_TDA_MISMATCH.value)

    # Deduplicate
    violation_codes = list(dict.fromkeys(violation_codes))

    # Determine MAAS status
    if slo_status == "BREACH" or quarantine_required:
        status = MAASStatus.BLOCK
        telemetry_admissible = False
    elif slo_status == "WARN" or (tda_correlation and tda_correlation.get("tda_telemetry_mismatch")):
        status = MAASStatus.ATTENTION
        telemetry_admissible = True
    else:
        status = MAASStatus.OK
        telemetry_admissible = True

    return MAASAdapterResult(
        telemetry_admissible=telemetry_admissible,
        status=status,
        violation_codes=violation_codes,
    )


# =============================================================================
# PHASE V-B: TELEMETRY GOVERNANCE SIGNAL ADAPTER
# =============================================================================

class GovernanceSignal(Enum):
    """Governance signal status for telemetry integrity."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class TelemetryGovernanceResult:
    """Result of telemetry governance signal evaluation."""
    signal: GovernanceSignal
    telemetry_ok: bool
    blocking_rules: list[str]
    reasons: list[str]
    tda_coupled: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal": self.signal.value,
            "telemetry_ok": self.telemetry_ok,
            "blocking_rules": self.blocking_rules,
            "reasons": self.reasons,
            "tda_coupled": self.tda_coupled,
        }


# Blocking rules that trigger BLOCK signal
GOVERNANCE_BLOCKING_RULES: set[str] = {
    AlertCode.TELEMETRY_PIPELINE_BLOCKED.value,
    AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value,
}

# Warning rules that trigger WARN signal
GOVERNANCE_WARN_RULES: set[str] = {
    AlertCode.TELEMETRY_TDA_MISMATCH.value,
    AlertCode.TELEMETRY_QUARANTINE_SPIKE.value,
    AlertCode.TELEMETRY_L2_DEGRADATION.value,
    AlertCode.TELEMETRY_CRITICAL_VIOLATIONS.value,
}


def to_governance_signal_for_telemetry(
    console_summary: dict[str, Any]
) -> TelemetryGovernanceResult:
    """
    Generate a governance signal from telemetry console summary.

    This function makes telemetry a first-class part of the global epistemic
    integrity surface by emitting a structured governance signal.

    Signal Rules:
    - BLOCK if: TELEMETRY_PIPELINE_BLOCKED or TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE
    - WARN if: any TDA-related mismatch or other alert codes present
    - OK only when: both telemetry SLO and TDA agree on health (no alerts)

    Args:
        console_summary: Global console result dict (from GlobalConsoleResult.to_dict())

    Returns:
        TelemetryGovernanceResult with signal, blocking_rules, and reasons

    Example:
        >>> console = summarize_telemetry_for_global_console(
        ...     health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        ... )
        >>> gov_signal = to_governance_signal_for_telemetry(console.to_dict())
        >>> if gov_signal.signal == GovernanceSignal.BLOCK:
        ...     halt_pipeline(gov_signal.blocking_rules)
    """
    # Extract alert codes from console summary
    alert_codes = console_summary.get("alert_codes", [])
    telemetry_ok = console_summary.get("telemetry_ok", True)
    status_light = console_summary.get("status_light", "green")
    tda_correlation_detected = console_summary.get("tda_correlation_detected", False)
    headline = console_summary.get("headline", "")

    blocking_rules: list[str] = []
    reasons: list[str] = []

    # Check for BLOCK-level alerts
    for code in alert_codes:
        if code in GOVERNANCE_BLOCKING_RULES:
            blocking_rules.append(code)
            if code == AlertCode.TELEMETRY_PIPELINE_BLOCKED.value:
                reasons.append("Pipeline blocked due to critical telemetry failures")
            elif code == AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value:
                reasons.append(
                    "Correlated failure: Low TDA health (HSS) with elevated telemetry quarantine rate"
                )

    # If we have blocking rules, emit BLOCK
    if blocking_rules:
        return TelemetryGovernanceResult(
            signal=GovernanceSignal.BLOCK,
            telemetry_ok=False,
            blocking_rules=blocking_rules,
            reasons=reasons,
            tda_coupled=tda_correlation_detected,
        )

    # Check for WARN-level alerts
    warn_rules: list[str] = []
    for code in alert_codes:
        if code in GOVERNANCE_WARN_RULES:
            warn_rules.append(code)
            if code == AlertCode.TELEMETRY_TDA_MISMATCH.value:
                reasons.append("TDA-Telemetry mismatch: systems disagree on health status")
            elif code == AlertCode.TELEMETRY_QUARANTINE_SPIKE.value:
                reasons.append("Telemetry quarantine rate has spiked above threshold")
            elif code == AlertCode.TELEMETRY_L2_DEGRADATION.value:
                reasons.append("Telemetry L2 (canonical) conformance has degraded")
            elif code == AlertCode.TELEMETRY_CRITICAL_VIOLATIONS.value:
                reasons.append("Critical telemetry violations detected")

    # If we have warning rules, emit WARN
    if warn_rules:
        return TelemetryGovernanceResult(
            signal=GovernanceSignal.WARN,
            telemetry_ok=telemetry_ok,  # WARN allows publish with warnings
            blocking_rules=warn_rules,
            reasons=reasons,
            tda_coupled=tda_correlation_detected,
        )

    # Additional WARN conditions: status light is yellow but no specific alert codes
    if status_light == "yellow" or status_light == StatusLight.YELLOW.value:
        return TelemetryGovernanceResult(
            signal=GovernanceSignal.WARN,
            telemetry_ok=telemetry_ok,
            blocking_rules=[],
            reasons=[headline] if headline else ["Telemetry warnings detected"],
            tda_coupled=tda_correlation_detected,
        )

    # Additional WARN condition: TDA correlation detected but no alert codes
    if tda_correlation_detected:
        return TelemetryGovernanceResult(
            signal=GovernanceSignal.WARN,
            telemetry_ok=telemetry_ok,
            blocking_rules=[],
            reasons=["TDA-Telemetry correlation detected, review recommended"],
            tda_coupled=True,
        )

    # OK: No alerts, no warnings, telemetry and TDA agree
    return TelemetryGovernanceResult(
        signal=GovernanceSignal.OK,
        telemetry_ok=True,
        blocking_rules=[],
        reasons=["Telemetry integrity nominal, TDA-coupled health confirmed"],
        tda_coupled=False,  # No correlation issues means clean agreement
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for telemetry conformance auditing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Telemetry Conformance Checker - Phase V"
    )
    parser.add_argument(
        "file",
        help="JSONL file to audit"
    )
    parser.add_argument(
        "--quarantine-dir",
        help="Directory for quarantined records",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output path for conformance report",
        default=None
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1

    # Run audit
    print(f"Auditing: {args.file}")
    report = audit_telemetry_file(
        args.file,
        quarantine_root=args.quarantine_dir
    )

    # Write report
    report_path = write_conformance_report(report, args.output)
    print(f"Report written to: {report_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("CONFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total records:    {report.total_records}")
    print(f"L2 (Canonical):   {report.by_level['L2']} ({report.l2_percentage:.1f}%)")
    print(f"L1 (Schema-valid):{report.by_level['L1']}")
    print(f"L0 (Raw):         {report.by_level['L0']}")
    print(f"Quarantined:      {report.quarantined_count}")

    if args.verbose and report.sample_violations:
        print(f"\nSample Violations:")
        for v in report.sample_violations[:5]:
            print(f"  Line {v['line']}: [{v['severity'].upper()}] {v['check']} - {v['message']}")

    return 0 if report.quarantined_count == 0 else 1


if __name__ == "__main__":
    exit(main())
