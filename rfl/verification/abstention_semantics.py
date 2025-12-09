"""
Abstention Semantics: Semantic Tree, Categories, and JSON Schema Validation
============================================================================

This module provides semantic classification and validation for the abstention
taxonomy, enabling higher-level analysis and dashboard integration.

DESIGN PRINCIPLES:
    1. Semantic categories group related abstention types for analysis
    2. JSON Schema validates all AbstentionRecord data at boundaries
    3. Aggregation functions enable dashboard-ready metrics
    4. Deterministic behavior preserves audit trail integrity

SEMANTIC TREE:
    TIMEOUT_RELATED
        ├── ABSTAIN_TIMEOUT
        └── ABSTAIN_LEAN_TIMEOUT
    
    RESOURCE_RELATED
        └── ABSTAIN_BUDGET
    
    CRASH_RELATED
        ├── ABSTAIN_CRASH
        └── ABSTAIN_LEAN_ERROR
    
    ORACLE_RELATED
        └── ABSTAIN_ORACLE_UNAVAILABLE
    
    INVALID_RELATED
        └── ABSTAIN_INVALID

INVARIANTS:
    INV-ABS-1: All abstentions converge on a single taxonomy
    INV-ABS-2: Serialization never breaks backward compatibility
    INV-ABS-3: Histograms always use canonical order
    INV-ABS-4: No abstention signal is ever lost or miscategorized

PHASE II — VERIFICATION BUREAU
Agent B4 (verifier-ops-4)
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, TYPE_CHECKING
from dataclasses import dataclass

from .abstention_taxonomy import AbstentionType

if TYPE_CHECKING:
    from .abstention_record import AbstentionRecord


# ---------------------------------------------------------------------------
# Taxonomy Version
# ---------------------------------------------------------------------------
#
# VERSION POLICY:
#   - Bump MAJOR for breaking changes (removing types, renaming categories)
#   - Bump MINOR for additions (new types, new categories)
#   - Bump PATCH for documentation/description changes only
#
# MIGRATION STEPS (for developers):
#   1. Make your taxonomy changes in this file
#   2. Bump ABSTENTION_TAXONOMY_VERSION appropriately
#   3. Regenerate export: uv run python -c "from rfl.verification.abstention_semantics import export_semantics; export_semantics('artifacts/abstention_semantics.json')"
#   4. Run diff: uv run python scripts/diff_abstention_taxonomy.py --old artifacts/abstention_semantics_prev.json --new artifacts/abstention_semantics.json
#   5. Update tests if needed
#   6. Commit with message: "taxonomy: bump to vX.Y.Z - <description>"
#

ABSTENTION_TAXONOMY_VERSION: str = "1.0.0"
"""
Semantic versioning for the abstention taxonomy.

This version tracks changes to:
- AbstentionType enum members
- SemanticCategory enum members  
- Type-to-category mappings (ABSTENTION_TREE)
- Legacy key mappings (COMPLETE_MAPPING)
"""


def get_taxonomy_version() -> str:
    """
    Return the current taxonomy version string.
    
    Returns:
        The semantic version string (e.g., "1.0.0")
        
    Example:
        >>> get_taxonomy_version()
        '1.0.0'
    """
    return ABSTENTION_TAXONOMY_VERSION


# ---------------------------------------------------------------------------
# Semantic Categories
# ---------------------------------------------------------------------------


class SemanticCategory(str, Enum):
    """
    High-level semantic categories for abstention types.
    
    These categories enable dashboard-level aggregation and trend analysis
    across related abstention types.
    """
    TIMEOUT_RELATED = "timeout_related"
    RESOURCE_RELATED = "resource_related"
    CRASH_RELATED = "crash_related"
    ORACLE_RELATED = "oracle_related"
    INVALID_RELATED = "invalid_related"
    
    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Semantic Tree: Maps AbstentionType → SemanticCategory
# ---------------------------------------------------------------------------


ABSTENTION_TREE: Dict[AbstentionType, SemanticCategory] = {
    # Timeout-related: Processing exceeded time limit
    AbstentionType.ABSTAIN_TIMEOUT: SemanticCategory.TIMEOUT_RELATED,
    AbstentionType.ABSTAIN_LEAN_TIMEOUT: SemanticCategory.TIMEOUT_RELATED,
    
    # Resource-related: Budget/resource exhaustion
    AbstentionType.ABSTAIN_BUDGET: SemanticCategory.RESOURCE_RELATED,
    
    # Crash-related: Unexpected failures
    AbstentionType.ABSTAIN_CRASH: SemanticCategory.CRASH_RELATED,
    AbstentionType.ABSTAIN_LEAN_ERROR: SemanticCategory.CRASH_RELATED,
    
    # Oracle-related: External service unavailable
    AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE: SemanticCategory.ORACLE_RELATED,
    
    # Invalid-related: Bad input or state
    AbstentionType.ABSTAIN_INVALID: SemanticCategory.INVALID_RELATED,
}


# Reverse mapping: SemanticCategory → List[AbstentionType]
_CATEGORY_TO_TYPES: Dict[SemanticCategory, List[AbstentionType]] = {}
for abst_type, category in ABSTENTION_TREE.items():
    if category not in _CATEGORY_TO_TYPES:
        _CATEGORY_TO_TYPES[category] = []
    _CATEGORY_TO_TYPES[category].append(abst_type)


# ---------------------------------------------------------------------------
# Category Classification Functions
# ---------------------------------------------------------------------------


def categorize(abstention_type: AbstentionType) -> SemanticCategory:
    """
    Categorize an AbstentionType into its semantic category.
    
    Args:
        abstention_type: The AbstentionType to categorize
        
    Returns:
        The SemanticCategory for this abstention type
        
    Raises:
        KeyError: If abstention_type is not in the semantic tree
        
    Examples:
        >>> categorize(AbstentionType.ABSTAIN_TIMEOUT)
        SemanticCategory.TIMEOUT_RELATED
        
        >>> categorize(AbstentionType.ABSTAIN_LEAN_TIMEOUT)
        SemanticCategory.TIMEOUT_RELATED
    """
    return ABSTENTION_TREE[abstention_type]


def get_category(abstention_type: AbstentionType) -> Optional[SemanticCategory]:
    """
    Get the semantic category for an AbstentionType (returns None if not found).
    
    This is a safe version of categorize() that doesn't raise on unknown types.
    
    Args:
        abstention_type: The AbstentionType to categorize
        
    Returns:
        The SemanticCategory, or None if not in tree
    """
    return ABSTENTION_TREE.get(abstention_type)


def get_types_for_category(category: SemanticCategory) -> List[AbstentionType]:
    """
    Get all AbstentionTypes that belong to a semantic category.
    
    Args:
        category: The SemanticCategory to query
        
    Returns:
        List of AbstentionTypes in this category
        
    Examples:
        >>> get_types_for_category(SemanticCategory.TIMEOUT_RELATED)
        [AbstentionType.ABSTAIN_TIMEOUT, AbstentionType.ABSTAIN_LEAN_TIMEOUT]
    """
    return _CATEGORY_TO_TYPES.get(category, [])


def get_all_categories() -> List[SemanticCategory]:
    """
    Get all semantic categories in definition order.
    
    Returns:
        List of all SemanticCategory values
    """
    return list(SemanticCategory)


# ---------------------------------------------------------------------------
# Category Predicate Functions
# ---------------------------------------------------------------------------


def is_timeout_related(abstention_type: AbstentionType) -> bool:
    """Check if abstention type is timeout-related."""
    return get_category(abstention_type) == SemanticCategory.TIMEOUT_RELATED


def is_crash_related(abstention_type: AbstentionType) -> bool:
    """Check if abstention type is crash-related."""
    return get_category(abstention_type) == SemanticCategory.CRASH_RELATED


def is_resource_related(abstention_type: AbstentionType) -> bool:
    """Check if abstention type is resource/budget-related."""
    return get_category(abstention_type) == SemanticCategory.RESOURCE_RELATED


def is_oracle_related(abstention_type: AbstentionType) -> bool:
    """Check if abstention type is oracle-related."""
    return get_category(abstention_type) == SemanticCategory.ORACLE_RELATED


def is_invalid_related(abstention_type: AbstentionType) -> bool:
    """Check if abstention type is invalid-input-related."""
    return get_category(abstention_type) == SemanticCategory.INVALID_RELATED


# ---------------------------------------------------------------------------
# JSON Schema Definition
# ---------------------------------------------------------------------------


ABSTENTION_RECORD_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://mathledger.io/schemas/abstention-record.json",
    "title": "AbstentionRecord",
    "description": "Unified abstention record for RFL verification layer",
    "type": "object",
    "required": ["abstention_type"],
    "properties": {
        "abstention_type": {
            "type": "string",
            "enum": [t.value for t in AbstentionType],
            "description": "Canonical abstention type from AbstentionType enum"
        },
        "failure_state": {
            "type": ["string", "null"],
            "description": "FailureState value if abstention originated from execution failure"
        },
        "method": {
            "type": ["string", "null"],
            "description": "Verification method string if abstention originated from verification"
        },
        "details": {
            "type": ["string", "null"],
            "description": "Human-readable description of the abstention cause"
        },
        "source": {
            "type": "string",
            "description": "Origin of this record (experiment, pipeline, runner, etc.)"
        },
        "timestamp": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "ISO 8601 timestamp when this abstention was recorded"
        },
        "context": {
            "type": ["object", "null"],
            "description": "Additional context dictionary",
            "additionalProperties": True
        }
    },
    "additionalProperties": False
}


def get_schema() -> Dict[str, Any]:
    """
    Get the JSON Schema for AbstentionRecord validation.
    
    Returns:
        The JSON Schema as a dictionary
    """
    return ABSTENTION_RECORD_SCHEMA.copy()


def get_schema_path() -> Path:
    """
    Get the path where the schema file should be stored.
    
    Returns:
        Path to schemas/abstention-record.json
    """
    return Path("schemas") / "abstention-record.json"


def export_schema(path: Optional[Path] = None) -> Path:
    """
    Export the JSON Schema to a file.
    
    Args:
        path: Optional path override; defaults to get_schema_path()
        
    Returns:
        Path where schema was written
    """
    target = path or get_schema_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    
    with open(target, "w", encoding="utf-8") as f:
        json.dump(ABSTENTION_RECORD_SCHEMA, f, indent=2)
    
    return target


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class AbstentionValidationError(ValueError):
    """
    Raised when AbstentionRecord validation fails.
    
    Attributes:
        errors: List of validation error messages
        data: The data that failed validation
    """
    
    def __init__(self, errors: List[str], data: Any = None):
        self.errors = errors
        self.data = data
        message = f"AbstentionRecord validation failed: {'; '.join(errors)}"
        super().__init__(message)


def validate_abstention_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate a dictionary against the AbstentionRecord schema.
    
    This is a lightweight validator that doesn't require jsonschema library.
    For full JSON Schema validation, use validate_abstention_json().
    
    Args:
        data: Dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors: List[str] = []
    
    # Check required fields
    if "abstention_type" not in data:
        errors.append("Missing required field: abstention_type")
        return errors
    
    # Validate abstention_type value
    abstention_type = data["abstention_type"]
    valid_types = {t.value for t in AbstentionType}
    if abstention_type not in valid_types:
        errors.append(
            f"Invalid abstention_type: {abstention_type!r}. "
            f"Must be one of: {sorted(valid_types)}"
        )
    
    # Validate optional field types
    if "failure_state" in data and data["failure_state"] is not None:
        if not isinstance(data["failure_state"], str):
            errors.append("failure_state must be a string or null")
    
    if "method" in data and data["method"] is not None:
        if not isinstance(data["method"], str):
            errors.append("method must be a string or null")
    
    if "details" in data and data["details"] is not None:
        if not isinstance(data["details"], str):
            errors.append("details must be a string or null")
    
    if "source" in data:
        if not isinstance(data["source"], str):
            errors.append("source must be a string")
    
    if "timestamp" in data and data["timestamp"] is not None:
        if not isinstance(data["timestamp"], str):
            errors.append("timestamp must be a string or null")
    
    if "context" in data and data["context"] is not None:
        if not isinstance(data["context"], dict):
            errors.append("context must be an object or null")
    
    return errors


def validate_abstention_record(record: "AbstentionRecord") -> List[str]:
    """
    Validate an AbstentionRecord object.
    
    Args:
        record: The AbstentionRecord to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    return validate_abstention_data(record.to_dict())


def validate_abstention_json(json_str: str) -> List[str]:
    """
    Validate a JSON string representing an AbstentionRecord.
    
    Args:
        json_str: JSON string to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    
    if not isinstance(data, dict):
        return ["JSON must be an object"]
    
    return validate_abstention_data(data)


def validate_or_raise(data: Union[Dict[str, Any], "AbstentionRecord", str]) -> None:
    """
    Validate data and raise AbstentionValidationError if invalid.
    
    Args:
        data: Dictionary, AbstentionRecord, or JSON string to validate
        
    Raises:
        AbstentionValidationError: If validation fails
    """
    if isinstance(data, str):
        errors = validate_abstention_json(data)
    elif hasattr(data, "to_dict"):
        errors = validate_abstention_record(data)  # type: ignore
    else:
        errors = validate_abstention_data(data)  # type: ignore
    
    if errors:
        raise AbstentionValidationError(errors, data)


# ---------------------------------------------------------------------------
# Aggregation Functions
# ---------------------------------------------------------------------------


def aggregate_by_category(
    abstention_types: List[AbstentionType]
) -> Dict[SemanticCategory, int]:
    """
    Aggregate a list of abstention types by semantic category.
    
    Args:
        abstention_types: List of AbstentionType values
        
    Returns:
        Dictionary mapping SemanticCategory to count
        
    Examples:
        >>> types = [
        ...     AbstentionType.ABSTAIN_TIMEOUT,
        ...     AbstentionType.ABSTAIN_TIMEOUT,
        ...     AbstentionType.ABSTAIN_CRASH,
        ... ]
        >>> aggregate_by_category(types)
        {SemanticCategory.TIMEOUT_RELATED: 2, SemanticCategory.CRASH_RELATED: 1}
    """
    result: Dict[SemanticCategory, int] = {}
    
    for abst_type in abstention_types:
        category = get_category(abst_type)
        if category:
            result[category] = result.get(category, 0) + 1
    
    return result


def aggregate_histogram_by_category(
    histogram: Dict[str, int]
) -> Dict[str, int]:
    """
    Aggregate a histogram by semantic category.
    
    This converts a fine-grained abstention histogram into category-level
    aggregates suitable for dashboards.
    
    Args:
        histogram: Dictionary mapping abstention_type.value → count
        
    Returns:
        Dictionary mapping category.value → count
        
    Examples:
        >>> histogram = {
        ...     "abstain_timeout": 5,
        ...     "abstain_lean_timeout": 3,
        ...     "abstain_crash": 2,
        ... }
        >>> aggregate_histogram_by_category(histogram)
        {'timeout_related': 8, 'crash_related': 2}
    """
    result: Dict[str, int] = {}
    
    for key, count in histogram.items():
        # Try to parse as AbstentionType
        try:
            abst_type = AbstentionType(key)
            category = get_category(abst_type)
            if category:
                cat_key = category.value
                result[cat_key] = result.get(cat_key, 0) + count
        except ValueError:
            # Unknown key - skip (or could aggregate into "unknown" category)
            pass
    
    return result


def get_category_summary(histogram: Dict[str, int]) -> Dict[str, Any]:
    """
    Generate a summary report of abstentions by category.
    
    Args:
        histogram: Dictionary mapping abstention_type.value → count
        
    Returns:
        Summary dictionary with category totals and percentages
    """
    by_category = aggregate_histogram_by_category(histogram)
    total = sum(by_category.values())
    
    summary: Dict[str, Any] = {
        "total_abstentions": total,
        "by_category": {},
    }
    
    for category in SemanticCategory:
        cat_key = category.value
        count = by_category.get(cat_key, 0)
        percentage = (count / total * 100) if total > 0 else 0.0
        
        summary["by_category"][cat_key] = {
            "count": count,
            "percentage": round(percentage, 2),
            "types": [t.value for t in get_types_for_category(category)],
        }
    
    return summary


# ---------------------------------------------------------------------------
# Slice-Level Analytics
# ---------------------------------------------------------------------------


def summarize_abstentions(
    records: "List[AbstentionRecord]",
    top_n: int = 5,
    max_reason_length: int = 100,
) -> Dict[str, Any]:
    """
    Generate a compact analytics summary for a slice's abstention records.
    
    This function produces a dashboard-ready summary with:
    - Total count
    - Breakdown by canonical type (histogram)
    - Breakdown by semantic category
    - Top reason strings (safely truncated)
    
    Args:
        records: Sequence of AbstentionRecord objects
        top_n: Number of top reasons to include (default 5)
        max_reason_length: Maximum length for reason strings (default 100)
        
    Returns:
        Dictionary with deterministic, JSON-serializable structure:
        {
            "total": int,
            "by_type": {abstention_type.value: count, ...},  # canonical order
            "by_category": {category.value: count, ...},  # all categories
            "top_reasons": [
                {"reason": str, "count": int, "type": str},
                ...
            ]
        }
        
    Examples:
        >>> records = [
        ...     AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="timeout 1"),
        ...     AbstentionRecord.from_failure_state(FailureState.TIMEOUT_ABSTAIN, details="timeout 2"),
        ...     AbstentionRecord.from_failure_state(FailureState.CRASH_ABSTAIN, details="crash"),
        ... ]
        >>> summary = summarize_abstentions(records)
        >>> summary["total"]
        3
        >>> summary["by_type"]["abstain_timeout"]
        2
    """
    from .abstention_record import CANONICAL_ABSTENTION_ORDER
    
    if not records:
        # Empty result with full structure
        return {
            "total": 0,
            "by_type": {t.value: 0 for t in CANONICAL_ABSTENTION_ORDER},
            "by_category": {c.value: 0 for c in SemanticCategory},
            "top_reasons": [],
        }
    
    # Count by type
    type_counts: Dict[str, int] = {t.value: 0 for t in CANONICAL_ABSTENTION_ORDER}
    for record in records:
        type_counts[record.abstention_type.value] = type_counts.get(
            record.abstention_type.value, 0
        ) + 1
    
    # Count by category
    category_counts: Dict[str, int] = {c.value: 0 for c in SemanticCategory}
    for record in records:
        cat = get_category(record.abstention_type)
        if cat:
            category_counts[cat.value] += 1
    
    # Aggregate reasons (details field)
    reason_counts: Dict[str, Dict[str, Any]] = {}
    for record in records:
        reason = record.details or "(no details)"
        # Truncate safely
        if len(reason) > max_reason_length:
            reason = reason[:max_reason_length - 3] + "..."
        
        if reason not in reason_counts:
            reason_counts[reason] = {
                "count": 0,
                "type": record.abstention_type.value,
            }
        reason_counts[reason]["count"] += 1
    
    # Sort reasons by count (descending), then by reason string (ascending) for stability
    sorted_reasons = sorted(
        reason_counts.items(),
        key=lambda x: (-x[1]["count"], x[0]),
    )
    
    top_reasons = [
        {
            "reason": reason,
            "count": data["count"],
            "type": data["type"],
        }
        for reason, data in sorted_reasons[:top_n]
    ]
    
    return {
        "total": len(records),
        "by_type": type_counts,
        "by_category": category_counts,
        "top_reasons": top_reasons,
    }


# ---------------------------------------------------------------------------
# Red Flag Detection (Advisory)
# ---------------------------------------------------------------------------

# Default thresholds for red flag detection (configurable later)
_DEFAULT_RED_FLAG_THRESHOLDS = {
    # Category thresholds (percentage of total)
    "timeout_threshold_pct": 50.0,     # >50% timeouts is a red flag
    "crash_threshold_pct": 30.0,       # >30% crashes is a red flag
    "invalid_threshold_pct": 80.0,     # >80% invalid is a red flag (possibly bad input)
    "oracle_threshold_pct": 90.0,      # >90% oracle unavailable is a red flag
    
    # Absolute thresholds for rare types
    "lean_error_threshold_abs": 10,    # >10 lean errors is concerning
    
    # Minimum sample size for percentage-based flags
    "min_sample_size": 5,
}


def detect_abstention_red_flags(
    summary: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Detect red flags in an abstention summary (advisory only).
    
    This function performs read-only analysis on the output of
    `summarize_abstentions()` and returns human-readable strings
    describing any concerning patterns.
    
    Args:
        summary: Output from summarize_abstentions()
        thresholds: Optional custom thresholds. Keys:
            - timeout_threshold_pct: % of total that triggers timeout flag
            - crash_threshold_pct: % of total that triggers crash flag
            - invalid_threshold_pct: % of total that triggers invalid flag
            - oracle_threshold_pct: % of total that triggers oracle flag
            - lean_error_threshold_abs: absolute count for lean error flag
            - min_sample_size: minimum records for percentage-based flags
            
    Returns:
        List of human-readable warning strings. Empty if no red flags.
        
    Note:
        This function never raises exceptions. On any error, it returns
        an empty list to ensure CI safety.
        
    Examples:
        >>> summary = {"total": 100, "by_category": {"timeout_related": 60, ...}, ...}
        >>> flags = detect_abstention_red_flags(summary)
        >>> "timeout" in flags[0].lower()
        True
    """
    try:
        flags: List[str] = []
        
        # Merge custom thresholds with defaults
        cfg = {**_DEFAULT_RED_FLAG_THRESHOLDS, **(thresholds or {})}
        
        total = summary.get("total", 0)
        by_category = summary.get("by_category", {})
        by_type = summary.get("by_type", {})
        
        # Skip if insufficient sample size
        if total < cfg["min_sample_size"]:
            return []
        
        # Check category-level red flags
        category_checks = [
            (
                "timeout_related",
                cfg["timeout_threshold_pct"],
                "TIMEOUT",
                "High timeout rate may indicate resource constraints or deadlocks",
            ),
            (
                "crash_related", 
                cfg["crash_threshold_pct"],
                "CRASH",
                "High crash rate may indicate infrastructure instability",
            ),
            (
                "invalid_related",
                cfg["invalid_threshold_pct"],
                "INVALID",
                "Very high invalid rate may indicate bad input data or corpus issues",
            ),
            (
                "oracle_related",
                cfg["oracle_threshold_pct"],
                "ORACLE",
                "Oracle unavailability may indicate Lean service issues",
            ),
        ]
        
        for cat_key, threshold, label, description in category_checks:
            count = by_category.get(cat_key, 0)
            pct = (count / total) * 100 if total > 0 else 0
            
            if pct > threshold:
                flags.append(
                    f"[{label}] {count}/{total} ({pct:.1f}%) abstentions are {cat_key}. "
                    f"Threshold: {threshold:.1f}%. {description}."
                )
        
        # Check absolute thresholds for rare but concerning types
        lean_error_count = by_type.get("abstain_lean_error", 0)
        if lean_error_count > cfg["lean_error_threshold_abs"]:
            flags.append(
                f"[LEAN_ERROR] {lean_error_count} Lean verification errors detected. "
                f"Threshold: {cfg['lean_error_threshold_abs']}. "
                "May indicate Lean environment issues or malformed proofs."
            )
        
        # Check for suspiciously uniform distributions
        if total >= 10:
            non_zero_categories = sum(1 for c in by_category.values() if c > 0)
            if non_zero_categories == 1:
                dominant_cat = next(
                    (k for k, v in by_category.items() if v > 0), None
                )
                if dominant_cat and by_category[dominant_cat] == total:
                    flags.append(
                        f"[UNIFORM] All {total} abstentions are {dominant_cat}. "
                        "Single-category distribution may indicate systematic issue."
                    )
        
        return flags
        
    except Exception:
        # Never raise - this is advisory only
        return []


# ---------------------------------------------------------------------------
# Phase III: Red-Flag Feed & Global Health
# ---------------------------------------------------------------------------

# Health snapshot schema version (separate from taxonomy version)
HEALTH_SNAPSHOT_SCHEMA_VERSION: str = "1.0.0"

# Thresholds for radar status determination
_RADAR_STATUS_THRESHOLDS = {
    "attention_red_flag_pct": 20.0,   # >20% slices with red flags = ATTENTION
    "critical_red_flag_pct": 50.0,    # >50% slices with red flags = CRITICAL
    "timeout_dominated_threshold": 50.0,  # >50% timeout = timeout-dominated
    "crash_dominated_threshold": 30.0,    # >30% crash = crash-dominated
}

# Thresholds for uplift guard
_UPLIFT_GUARD_THRESHOLDS = {
    "max_blocking_slices": 0,          # Any blocking slice = not uplift safe
    "blocking_crash_pct": 40.0,        # >40% crash in a slice = blocking
    "blocking_timeout_pct": 60.0,      # >60% timeout in a slice = blocking
}


def build_abstention_health_snapshot(
    summary: Dict[str, Any],
    red_flags: List[str],
    slice_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a canonical abstention health snapshot for a single slice.
    
    This is the official health contract per slice, suitable for MAAS
    ingestion, dashboards, and governance reporting.
    
    Args:
        summary: Output from summarize_abstentions()
        red_flags: Output from detect_abstention_red_flags()
        slice_name: Optional name/ID of the slice
        
    Returns:
        Canonical health snapshot with:
        - schema_version: Version of this snapshot schema
        - slice_name: Name of the slice (or "unnamed")
        - total: Total abstention count
        - by_type: Percentage breakdown by abstention type
        - by_category: Percentage breakdown by semantic category
        - red_flag_count: Number of red flags
        - red_flags: List of red flag strings
        
    Examples:
        >>> summary = summarize_abstentions(records)
        >>> flags = detect_abstention_red_flags(summary)
        >>> snapshot = build_abstention_health_snapshot(summary, flags, "prop_logic_001")
        >>> snapshot["slice_name"]
        'prop_logic_001'
        >>> snapshot["schema_version"]
        '1.0.0'
    """
    total = summary.get("total", 0)
    by_type = summary.get("by_type", {})
    by_category = summary.get("by_category", {})
    
    # Convert counts to percentages for by_type
    by_type_pct: Dict[str, float] = {}
    for type_key, count in by_type.items():
        pct = (count / total * 100) if total > 0 else 0.0
        by_type_pct[type_key] = round(pct, 2)
    
    # Convert counts to percentages for by_category
    by_category_pct: Dict[str, float] = {}
    for cat_key, count in by_category.items():
        pct = (count / total * 100) if total > 0 else 0.0
        by_category_pct[cat_key] = round(pct, 2)
    
    return {
        "schema_version": HEALTH_SNAPSHOT_SCHEMA_VERSION,
        "slice_name": slice_name or "unnamed",
        "total": total,
        "by_type": by_type_pct,
        "by_category": by_category_pct,
        "red_flag_count": len(red_flags),
        "red_flags": list(red_flags),  # Copy for immutability
    }


def build_abstention_radar(
    snapshots: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build a multi-slice abstention radar for global health monitoring.
    
    This aggregates health snapshots from multiple slices to provide
    a system-wide view of abstention patterns.
    
    Args:
        snapshots: List of health snapshots from build_abstention_health_snapshot()
        thresholds: Optional custom thresholds for status determination
        
    Returns:
        Radar summary with:
        - total_slices: Number of slices analyzed
        - slices_with_red_flags: Count of slices with at least one red flag
        - timeout_dominated_slices: Count of slices where timeout_related > threshold
        - crash_dominated_slices: Count of slices where crash_related > threshold
        - status: "OK" | "ATTENTION" | "CRITICAL"
        
    Status determination:
        - CRITICAL: >50% of slices have red flags
        - ATTENTION: >20% of slices have red flags
        - OK: ≤20% of slices have red flags
        
    Examples:
        >>> snapshots = [build_abstention_health_snapshot(s, f, name) for ...]
        >>> radar = build_abstention_radar(snapshots)
        >>> radar["status"]
        'OK'
    """
    cfg = {**_RADAR_STATUS_THRESHOLDS, **(thresholds or {})}
    
    total_slices = len(snapshots)
    
    if total_slices == 0:
        return {
            "total_slices": 0,
            "slices_with_red_flags": 0,
            "timeout_dominated_slices": 0,
            "crash_dominated_slices": 0,
            "status": "OK",
            "slice_details": [],
        }
    
    slices_with_red_flags = 0
    timeout_dominated_slices = 0
    crash_dominated_slices = 0
    slice_details: List[Dict[str, Any]] = []
    
    for snapshot in snapshots:
        red_flag_count = snapshot.get("red_flag_count", 0)
        by_category = snapshot.get("by_category", {})
        slice_name = snapshot.get("slice_name", "unnamed")
        
        # Count red flag slices
        if red_flag_count > 0:
            slices_with_red_flags += 1
        
        # Count timeout-dominated slices
        timeout_pct = by_category.get("timeout_related", 0.0)
        if timeout_pct > cfg["timeout_dominated_threshold"]:
            timeout_dominated_slices += 1
        
        # Count crash-dominated slices
        crash_pct = by_category.get("crash_related", 0.0)
        if crash_pct > cfg["crash_dominated_threshold"]:
            crash_dominated_slices += 1
        
        # Collect slice detail
        slice_details.append({
            "slice_name": slice_name,
            "red_flag_count": red_flag_count,
            "timeout_pct": timeout_pct,
            "crash_pct": crash_pct,
        })
    
    # Determine status
    red_flag_pct = (slices_with_red_flags / total_slices) * 100
    
    if red_flag_pct > cfg["critical_red_flag_pct"]:
        status = "CRITICAL"
    elif red_flag_pct > cfg["attention_red_flag_pct"]:
        status = "ATTENTION"
    else:
        status = "OK"
    
    return {
        "total_slices": total_slices,
        "slices_with_red_flags": slices_with_red_flags,
        "timeout_dominated_slices": timeout_dominated_slices,
        "crash_dominated_slices": crash_dominated_slices,
        "status": status,
        "slice_details": slice_details,
    }


def summarize_abstentions_for_uplift(
    radar: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generate an uplift safety summary from the abstention radar.
    
    This function determines whether the current abstention patterns
    allow safe model uplift. It identifies blocking slices whose
    abstention patterns are too pathological for uplift.
    
    Args:
        radar: Output from build_abstention_radar()
        thresholds: Optional custom thresholds for blocking determination
        
    Returns:
        Uplift summary with:
        - uplift_safe: bool - True if no blocking slices
        - blocking_slices: List of slice names with pathological patterns
        - blocking_reasons: Dict mapping slice name to reason
        - status: "OK" | "WARN" | "BLOCK"
        
    Status determination:
        - BLOCK: Any blocking slice present (uplift_safe=False)
        - WARN: Radar status is ATTENTION but no blocking slices
        - OK: Radar status is OK and no blocking slices
        
    Examples:
        >>> radar = build_abstention_radar(snapshots)
        >>> uplift = summarize_abstentions_for_uplift(radar)
        >>> if uplift["uplift_safe"]:
        ...     proceed_with_uplift()
    """
    cfg = {**_UPLIFT_GUARD_THRESHOLDS, **(thresholds or {})}
    
    blocking_slices: List[str] = []
    blocking_reasons: Dict[str, str] = {}
    
    slice_details = radar.get("slice_details", [])
    
    for detail in slice_details:
        slice_name = detail.get("slice_name", "unnamed")
        crash_pct = detail.get("crash_pct", 0.0)
        timeout_pct = detail.get("timeout_pct", 0.0)
        
        # Check if slice is blocking
        reasons = []
        
        if crash_pct > cfg["blocking_crash_pct"]:
            reasons.append(f"crash_rate={crash_pct:.1f}%>{cfg['blocking_crash_pct']}%")
        
        if timeout_pct > cfg["blocking_timeout_pct"]:
            reasons.append(f"timeout_rate={timeout_pct:.1f}%>{cfg['blocking_timeout_pct']}%")
        
        if reasons:
            blocking_slices.append(slice_name)
            blocking_reasons[slice_name] = "; ".join(reasons)
    
    uplift_safe = len(blocking_slices) <= cfg["max_blocking_slices"]
    
    # Determine status
    radar_status = radar.get("status", "OK")
    
    if not uplift_safe:
        status = "BLOCK"
    elif radar_status == "ATTENTION":
        status = "WARN"
    else:
        status = "OK"
    
    return {
        "uplift_safe": uplift_safe,
        "blocking_slices": blocking_slices,
        "blocking_reasons": blocking_reasons,
        "status": status,
    }


def summarize_abstentions_for_global_health(
    radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a global health summary from the abstention radar.
    
    This provides a high-level health assessment suitable for
    monitoring dashboards and alerting systems.
    
    Args:
        radar: Output from build_abstention_radar()
        
    Returns:
        Global health summary with:
        - abstention_ok: bool - True if status is OK
        - red_flag_slice_count: Number of slices with red flags
        - total_slices: Total number of slices
        - status: "OK" | "WARN" | "CRITICAL"
        - summary_text: Human-readable summary
        
    Status mapping:
        - Radar CRITICAL → Global CRITICAL
        - Radar ATTENTION → Global WARN
        - Radar OK → Global OK
        
    Examples:
        >>> radar = build_abstention_radar(snapshots)
        >>> health = summarize_abstentions_for_global_health(radar)
        >>> print(health["summary_text"])
        'Abstention health OK: 0/5 slices have red flags'
    """
    radar_status = radar.get("status", "OK")
    red_flag_slice_count = radar.get("slices_with_red_flags", 0)
    total_slices = radar.get("total_slices", 0)
    timeout_dominated = radar.get("timeout_dominated_slices", 0)
    crash_dominated = radar.get("crash_dominated_slices", 0)
    
    # Map radar status to global status
    status_map = {
        "OK": "OK",
        "ATTENTION": "WARN",
        "CRITICAL": "CRITICAL",
    }
    status = status_map.get(radar_status, "OK")
    
    abstention_ok = status == "OK"
    
    # Generate summary text
    if total_slices == 0:
        summary_text = "No slices to analyze"
    elif abstention_ok:
        summary_text = f"Abstention health OK: {red_flag_slice_count}/{total_slices} slices have red flags"
    elif status == "WARN":
        summary_text = (
            f"Abstention health WARN: {red_flag_slice_count}/{total_slices} slices have red flags. "
            f"Timeout-dominated: {timeout_dominated}, Crash-dominated: {crash_dominated}"
        )
    else:  # CRITICAL
        summary_text = (
            f"Abstention health CRITICAL: {red_flag_slice_count}/{total_slices} slices have red flags. "
            f"Timeout-dominated: {timeout_dominated}, Crash-dominated: {crash_dominated}. "
            "Immediate attention required."
        )
    
    return {
        "abstention_ok": abstention_ok,
        "red_flag_slice_count": red_flag_slice_count,
        "total_slices": total_slices,
        "timeout_dominated_slices": timeout_dominated,
        "crash_dominated_slices": crash_dominated,
        "status": status,
        "summary_text": summary_text,
    }


# ---------------------------------------------------------------------------
# Phase IV: Epistemic Risk Decomposition & Cross-Signal Integration
# ---------------------------------------------------------------------------

# Epistemic profile schema version
EPISTEMIC_PROFILE_SCHEMA_VERSION: str = "1.0.0"

# Risk band thresholds
_EPISTEMIC_RISK_THRESHOLDS = {
    "low_timeout_max": 20.0,      # ≤20% timeout = LOW
    "low_crash_max": 10.0,         # ≤10% crash = LOW
    "low_invalid_max": 40.0,       # ≤40% invalid = LOW
    "medium_timeout_max": 50.0,    # ≤50% timeout = MEDIUM
    "medium_crash_max": 30.0,      # ≤30% crash = MEDIUM
    "medium_invalid_max": 70.0,    # ≤70% invalid = MEDIUM
    # Above medium thresholds = HIGH
}


def build_epistemic_abstention_profile(
    snapshot: Dict[str, Any],
    verifier_noise_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build an epistemic risk profile from an abstention health snapshot.
    
    This function treats abstention patterns as epistemic risk signals,
    categorizing slices into risk bands based on timeout, crash, and
    invalid rates. Optionally correlates with verifier noise statistics.
    
    Args:
        snapshot: Output from build_abstention_health_snapshot()
        verifier_noise_stats: Optional dict with verifier noise metrics:
            - noise_rate: float - Overall verifier noise rate
            - correlation_coefficient: float - Correlation with abstention
            
    Returns:
        Epistemic profile with:
        - schema_version: Version of this profile schema
        - slice_name: Name of the slice
        - timeout_rate: Percentage of timeout-related abstentions
        - crash_rate: Percentage of crash-related abstentions
        - invalid_rate: Percentage of invalid-related abstentions
        - verifier_noise_correlation: Optional correlation metric
        - epistemic_risk_band: "LOW" | "MEDIUM" | "HIGH"
        
    Risk Band Determination:
        - LOW: All rates below low thresholds
        - MEDIUM: Any rate above low but below medium thresholds
        - HIGH: Any rate above medium thresholds
        
    Examples:
        >>> snapshot = build_abstention_health_snapshot(summary, flags, "slice_001")
        >>> profile = build_epistemic_abstention_profile(snapshot)
        >>> profile["epistemic_risk_band"]
        'LOW'
    """
    by_category = snapshot.get("by_category", {})
    slice_name = snapshot.get("slice_name", "unnamed")
    
    # Extract rates from category percentages
    timeout_rate = by_category.get("timeout_related", 0.0)
    crash_rate = by_category.get("crash_related", 0.0)
    invalid_rate = by_category.get("invalid_related", 0.0)
    
    # Determine risk band
    cfg = _EPISTEMIC_RISK_THRESHOLDS
    
    # Check if any rate exceeds HIGH threshold
    is_high = (
        timeout_rate > cfg["medium_timeout_max"] or
        crash_rate > cfg["medium_crash_max"] or
        invalid_rate > cfg["medium_invalid_max"]
    )
    
    # Check if any rate exceeds LOW threshold (but not HIGH)
    is_medium = (
        (timeout_rate > cfg["low_timeout_max"] and timeout_rate <= cfg["medium_timeout_max"]) or
        (crash_rate > cfg["low_crash_max"] and crash_rate <= cfg["medium_crash_max"]) or
        (invalid_rate > cfg["low_invalid_max"] and invalid_rate <= cfg["medium_invalid_max"])
    )
    
    if is_high:
        risk_band = "HIGH"
    elif is_medium:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"
    
    # Optional verifier noise correlation
    verifier_noise_correlation = None
    if verifier_noise_stats:
        correlation = verifier_noise_stats.get("correlation_coefficient")
        if correlation is not None:
            verifier_noise_correlation = round(float(correlation), 3)
    
    return {
        "schema_version": EPISTEMIC_PROFILE_SCHEMA_VERSION,
        "slice_name": slice_name,
        "timeout_rate": round(timeout_rate, 2),
        "crash_rate": round(crash_rate, 2),
        "invalid_rate": round(invalid_rate, 2),
        "verifier_noise_correlation": verifier_noise_correlation,
        "epistemic_risk_band": risk_band,
    }


def compose_abstention_with_budget_and_perf(
    epistemic_profiles: List[Dict[str, Any]],
    budget_view: Optional[Dict[str, Any]] = None,
    perf_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compose abstention epistemic risk with budget and performance signals.
    
    This function identifies slices with compounded risk across multiple
    signal dimensions: abstention patterns, budget exhaustion, and
    performance degradation.
    
    Args:
        epistemic_profiles: List of epistemic profiles from build_epistemic_abstention_profile()
        budget_view: Optional budget summary dict with:
            - exhausted_slices: List[str] - Slices with budget exhaustion
            - high_budget_usage_slices: List[str] - Slices with high budget usage
        perf_view: Optional performance summary dict with:
            - degraded_slices: List[str] - Slices with performance degradation
            - low_throughput_slices: List[str] - Slices with low throughput
            
    Returns:
        Compound risk view with:
        - slices_with_compounded_risk: List of slice names with multiple risk signals
        - global_risk_band: "LOW" | "MEDIUM" | "HIGH" - Overall system risk
        - reasoning: List of neutral textual hints explaining risk composition
        
    Risk Band Determination:
        - HIGH: >30% of slices have HIGH epistemic risk OR >20% have compounded risk
        - MEDIUM: >20% of slices have MEDIUM+ epistemic risk OR >10% have compounded risk
        - LOW: Otherwise
        
    Examples:
        >>> profiles = [build_epistemic_abstention_profile(s) for s in snapshots]
        >>> compound = compose_abstention_with_budget_and_perf(
        ...     profiles,
        ...     budget_view={"exhausted_slices": ["slice_001"]},
        ...     perf_view={"degraded_slices": ["slice_002"]}
        ... )
        >>> compound["global_risk_band"]
        'MEDIUM'
    """
    if not epistemic_profiles:
        return {
            "slices_with_compounded_risk": [],
            "global_risk_band": "LOW",
            "reasoning": ["No epistemic profiles provided"],
        }
    
    total_slices = len(epistemic_profiles)
    
    # Extract slice names and risk bands
    slice_risk_map: Dict[str, str] = {}
    high_risk_slices: List[str] = []
    medium_plus_risk_slices: List[str] = []
    
    for profile in epistemic_profiles:
        slice_name = profile.get("slice_name", "unnamed")
        risk_band = profile.get("epistemic_risk_band", "LOW")
        slice_risk_map[slice_name] = risk_band
        
        if risk_band == "HIGH":
            high_risk_slices.append(slice_name)
        if risk_band in ("MEDIUM", "HIGH"):
            medium_plus_risk_slices.append(slice_name)
    
    # Store high risk slices for director panel
    all_high_risk_slices = list(high_risk_slices)
    
    # Identify slices with compounded risk
    compounded_risk_slices: List[str] = []
    reasoning: List[str] = []
    
    # Check budget signals
    budget_exhausted = set()
    budget_high_usage = set()
    if budget_view:
        budget_exhausted = set(budget_view.get("exhausted_slices", []))
        budget_high_usage = set(budget_view.get("high_budget_usage_slices", []))
    
    # Check performance signals
    perf_degraded = set()
    perf_low_throughput = set()
    if perf_view:
        perf_degraded = set(perf_view.get("degraded_slices", []))
        perf_low_throughput = set(perf_view.get("low_throughput_slices", []))
    
    # Find slices with multiple risk signals
    for slice_name in slice_risk_map:
        risk_signals = []
        
        # Epistemic risk
        if slice_risk_map[slice_name] in ("MEDIUM", "HIGH"):
            risk_signals.append("epistemic")
        
        # Budget signals
        if slice_name in budget_exhausted:
            risk_signals.append("budget_exhausted")
        elif slice_name in budget_high_usage:
            risk_signals.append("budget_high_usage")
        
        # Performance signals
        if slice_name in perf_degraded:
            risk_signals.append("perf_degraded")
        elif slice_name in perf_low_throughput:
            risk_signals.append("perf_low_throughput")
        
        # Compounded if 2+ signals
        if len(risk_signals) >= 2:
            compounded_risk_slices.append(slice_name)
            reasoning.append(
                f"{slice_name}: compounded risk from {', '.join(risk_signals)}"
            )
    
    # Determine global risk band
    high_risk_pct = (len(high_risk_slices) / total_slices) * 100 if total_slices > 0 else 0
    medium_plus_pct = (len(medium_plus_risk_slices) / total_slices) * 100 if total_slices > 0 else 0
    compounded_pct = (len(compounded_risk_slices) / total_slices) * 100 if total_slices > 0 else 0
    
    # Check if any compounded slice has HIGH epistemic risk (critical)
    compounded_high_risk = any(
        slice_name in high_risk_slices for slice_name in compounded_risk_slices
    )
    
    # Global risk band: prioritize actual HIGH risk slices over compounded percentage
    if high_risk_pct > 30.0 or (compounded_high_risk and compounded_pct > 20.0):
        global_risk_band = "HIGH"
    elif medium_plus_pct > 20.0 or compounded_pct > 10.0:
        global_risk_band = "MEDIUM"
    else:
        global_risk_band = "LOW"
    
    # Add summary reasoning
    if not reasoning:
        reasoning.append("No compounded risk signals detected across slices")
    else:
        reasoning.insert(0, f"{len(compounded_risk_slices)}/{total_slices} slices show compounded risk")
    
    return {
        "slices_with_compounded_risk": compounded_risk_slices,
        "high_risk_slices": all_high_risk_slices,
        "global_risk_band": global_risk_band,
        "reasoning": reasoning,
    }


def build_abstention_director_panel(
    compound_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a high-level director panel for abstention epistemic risk.
    
    This function provides an executive-level summary suitable for
    governance dashboards and decision-making panels.
    
    Args:
        compound_view: Output from compose_abstention_with_budget_and_perf()
        
    Returns:
        Director panel with:
        - status_light: "GREEN" | "YELLOW" | "RED" - Visual status indicator
        - high_risk_slices: List of slice names with HIGH epistemic risk
        - dominant_abstention_categories: List of most common abstention categories
        - headline: Neutral summary of epistemic/abstention risk patterns
        
    Status Light Mapping:
        - GREEN: global_risk_band = LOW
        - YELLOW: global_risk_band = MEDIUM
        - RED: global_risk_band = HIGH
        
    Examples:
        >>> compound = compose_abstention_with_budget_and_perf(profiles, ...)
        >>> panel = build_abstention_director_panel(compound)
        >>> panel["status_light"]
        'GREEN'
    """
    global_risk_band = compound_view.get("global_risk_band", "LOW")
    compounded_slices = compound_view.get("slices_with_compounded_risk", [])
    reasoning = compound_view.get("reasoning", [])
    
    # Map risk band to status light
    status_light_map = {
        "LOW": "GREEN",
        "MEDIUM": "YELLOW",
        "HIGH": "RED",
    }
    status_light = status_light_map.get(global_risk_band, "GREEN")
    
    # Extract high risk slices from compound view
    high_risk_slices = compound_view.get("high_risk_slices", [])
    
    # Determine dominant categories from reasoning hints
    # This is a simplified extraction - in practice would analyze actual abstention data
    dominant_categories = []
    if any("timeout" in r.lower() for r in reasoning):
        dominant_categories.append("timeout_related")
    if any("crash" in r.lower() for r in reasoning):
        dominant_categories.append("crash_related")
    if any("invalid" in r.lower() for r in reasoning):
        dominant_categories.append("invalid_related")
    if not dominant_categories:
        dominant_categories = ["none_dominant"]
    
    # Generate headline
    if global_risk_band == "LOW":
        headline = (
            f"Abstention epistemic risk: LOW. "
            f"{len(compounded_slices)} slice(s) with compounded risk signals."
        )
    elif global_risk_band == "MEDIUM":
        headline = (
            f"Abstention epistemic risk: MEDIUM. "
            f"{len(compounded_slices)} slice(s) show compounded risk across multiple signal dimensions."
        )
    else:  # HIGH
        headline = (
            f"Abstention epistemic risk: HIGH. "
            f"{len(compounded_slices)} slice(s) with compounded risk. "
            "Review recommended."
        )
    
    return {
        "status_light": status_light,
        "high_risk_slices": high_risk_slices,
        "dominant_abstention_categories": dominant_categories,
        "headline": headline,
    }


def build_abstention_storyline(
    profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a narrative storyline from epistemic profiles showing trends.
    
    This function analyzes a sequence of epistemic profiles to determine
    whether the overall epistemic risk trend is improving, stable, or degrading.
    
    Args:
        profiles: List of epistemic profiles from build_epistemic_abstention_profile()
                 Ordered chronologically (oldest to newest)
        
    Returns:
        Storyline with:
        - schema_version: "1.0.0"
        - slices: List of per-slice summaries with risk bands
        - global_epistemic_trend: "IMPROVING" | "STABLE" | "DEGRADING"
        - story: Neutral narrative sentence describing the trend
        
    Trend Determination:
        - IMPROVING: More slices moving from HIGH→MEDIUM or MEDIUM→LOW than reverse
        - DEGRADING: More slices moving from LOW→MEDIUM or MEDIUM→HIGH than reverse
        - STABLE: No net movement or equal improvements/degradations
        
    Examples:
        >>> profiles = [build_epistemic_abstention_profile(s) for s in snapshots]
        >>> storyline = build_abstention_storyline(profiles)
        >>> storyline["global_epistemic_trend"]
        'STABLE'
    """
    if not profiles:
        return {
            "schema_version": "1.0.0",
            "slices": [],
            "global_epistemic_trend": "STABLE",
            "story": "No epistemic profiles available for trend analysis.",
        }
    
    # Build per-slice summaries
    slice_summaries: List[Dict[str, Any]] = []
    risk_band_counts: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    
    for profile in profiles:
        slice_name = profile.get("slice_name", "unnamed")
        risk_band = profile.get("epistemic_risk_band", "LOW")
        
        slice_summaries.append({
            "slice_name": slice_name,
            "epistemic_risk_band": risk_band,
            "timeout_rate": profile.get("timeout_rate", 0.0),
            "crash_rate": profile.get("crash_rate", 0.0),
            "invalid_rate": profile.get("invalid_rate", 0.0),
        })
        
        risk_band_counts[risk_band] = risk_band_counts.get(risk_band, 0) + 1
    
    # Determine trend by comparing risk band distribution
    # For simplicity, we compare the proportion of HIGH/MEDIUM/LOW bands
    # In a real implementation, you'd compare profiles across time windows
    total = len(profiles)
    high_pct = (risk_band_counts["HIGH"] / total) * 100 if total > 0 else 0
    medium_pct = (risk_band_counts["MEDIUM"] / total) * 100 if total > 0 else 0
    low_pct = (risk_band_counts["LOW"] / total) * 100 if total > 0 else 0
    
    # Trend determination based on risk distribution
    # IMPROVING: Low risk dominates (>60% LOW)
    # DEGRADING: High risk dominates (>30% HIGH) or medium+ dominates (>50% MEDIUM+)
    # STABLE: Otherwise
    if low_pct > 60.0:
        trend = "IMPROVING"
    elif high_pct > 30.0 or (medium_pct + high_pct) > 50.0:
        trend = "DEGRADING"
    else:
        trend = "STABLE"
    
    # Generate neutral narrative
    if trend == "IMPROVING":
        story = (
            f"Epistemic risk trend: IMPROVING. "
            f"{risk_band_counts['LOW']}/{total} slices ({low_pct:.1f}%) show LOW risk. "
            f"Risk distribution: LOW={low_pct:.1f}%, MEDIUM={medium_pct:.1f}%, HIGH={high_pct:.1f}%."
        )
    elif trend == "DEGRADING":
        story = (
            f"Epistemic risk trend: DEGRADING. "
            f"{risk_band_counts['HIGH']}/{total} slices ({high_pct:.1f}%) show HIGH risk. "
            f"Risk distribution: LOW={low_pct:.1f}%, MEDIUM={medium_pct:.1f}%, HIGH={high_pct:.1f}%."
        )
    else:  # STABLE
        story = (
            f"Epistemic risk trend: STABLE. "
            f"Risk distribution: LOW={low_pct:.1f}%, MEDIUM={medium_pct:.1f}%, HIGH={high_pct:.1f}%. "
            f"No significant trend detected."
        )
    
    return {
        "schema_version": "1.0.0",
        "slices": slice_summaries,
        "global_epistemic_trend": trend,
        "story": story,
    }


def evaluate_abstention_for_uplift(
    compound_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate abstention patterns as an epistemic gate for model uplift.
    
    This function provides a stricter epistemic gate than the general uplift
    summary, focusing specifically on HIGH epistemic risk combined with
    compounded signals as blocking conditions.
    
    Args:
        compound_view: Output from compose_abstention_with_budget_and_perf()
        
    Returns:
        Uplift evaluation with:
        - uplift_ok: bool - True if uplift is epistemically safe
        - status: "OK" | "WARN" | "BLOCK" - Uplift gate status
        - blocking_slices: List of slice names blocking uplift
        - reasons: List of neutral reasons for the decision
        
    Status Determination:
        - BLOCK: Any slice with HIGH epistemic risk AND compounded risk signals
        - WARN: Global risk band is HIGH or MEDIUM, or any compounded risk exists
        - OK: Global risk band is LOW and no compounded risk
        
    Examples:
        >>> compound = compose_abstention_with_budget_and_perf(profiles, ...)
        >>> gate = evaluate_abstention_for_uplift(compound)
        >>> if not gate["uplift_ok"]:
        ...     print(f"BLOCKED: {gate['blocking_slices']}")
    """
    global_risk_band = compound_view.get("global_risk_band", "LOW")
    high_risk_slices = compound_view.get("high_risk_slices", [])
    compounded_slices = compound_view.get("slices_with_compounded_risk", [])
    
    # Identify critical blocking slices: HIGH epistemic risk + compounded signals
    blocking_slices: List[str] = []
    reasons: List[str] = []
    
    # Check for critical slices (HIGH risk + compounded)
    critical_slices = set(high_risk_slices) & set(compounded_slices)
    if critical_slices:
        blocking_slices.extend(sorted(critical_slices))
        reasons.append(
            f"{len(critical_slices)} slice(s) have HIGH epistemic risk combined with "
            f"compounded risk signals: {', '.join(sorted(critical_slices))}"
        )
    
    # Determine status
    if blocking_slices:
        uplift_ok = False
        status = "BLOCK"
        if not reasons:
            reasons.append(
                f"Uplift BLOCKED: {len(blocking_slices)} slice(s) with critical risk patterns."
            )
    elif global_risk_band == "HIGH":
        uplift_ok = False
        status = "BLOCK"
        reasons.append(
            f"Uplift BLOCKED: Global epistemic risk band is HIGH. "
            f"{len(high_risk_slices)} slice(s) with HIGH risk."
        )
    elif global_risk_band == "MEDIUM" or compounded_slices:
        uplift_ok = True  # Not blocked, but caution advised
        status = "WARN"
        if compounded_slices:
            reasons.append(
                f"Uplift WARN: {len(compounded_slices)} slice(s) show compounded risk signals."
            )
        if global_risk_band == "MEDIUM":
            reasons.append("Uplift WARN: Global epistemic risk band is MEDIUM.")
    else:
        uplift_ok = True
        status = "OK"
        reasons.append(
            "Uplift OK: Global epistemic risk band is LOW and no compounded risk detected."
        )
    
    return {
        "uplift_ok": uplift_ok,
        "status": status,
        "blocking_slices": blocking_slices,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Phase V: Double-Helix Drift Radar & Global Governance Linkage
# ---------------------------------------------------------------------------

# Drift timeline schema version
DRIFT_TIMELINE_SCHEMA_VERSION: str = "1.0.0"

# Drift detection thresholds
_DRIFT_THRESHOLDS = {
    "stable_max_drift": 0.2,      # drift_index ≤ 0.2 = STABLE
    "drifting_max_drift": 0.6,    # drift_index ≤ 0.6 = DRIFTING
    # drift_index > 0.6 = VOLATILE
    "change_point_threshold": 0.3,  # Minimum change to register as change point
}


def build_epistemic_drift_timeline(
    profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build an epistemic drift timeline from a sequence of profiles.
    
    This function analyzes temporal patterns in epistemic risk bands to detect
    drift, volatility, and significant change points in the abstention landscape.
    
    Args:
        profiles: List of epistemic profiles ordered chronologically
                 (oldest to newest). Each profile should have:
                 - slice_name: str
                 - epistemic_risk_band: "LOW" | "MEDIUM" | "HIGH"
                 - timeout_rate, crash_rate, invalid_rate: float
                 
    Returns:
        Drift timeline with:
        - schema_version: "1.0.0"
        - drift_index: float (0.0-1.0) - Quantitative drift measure
        - risk_band: "STABLE" | "DRIFTING" | "VOLATILE"
        - change_points: List of dicts with slice_name, transition, timestamp
        - summary_text: Neutral narrative describing drift patterns
        
    Drift Index Calculation:
        - Measures variance in risk bands across profiles
        - 0.0 = all profiles same risk band (stable)
        - 1.0 = maximum variance (volatile)
        
    Risk Band Determination:
        - STABLE: drift_index ≤ 0.2
        - DRIFTING: 0.2 < drift_index ≤ 0.6
        - VOLATILE: drift_index > 0.6
        
    Examples:
        >>> profiles = [build_epistemic_abstention_profile(s) for s in snapshots]
        >>> timeline = build_epistemic_drift_timeline(profiles)
        >>> timeline["risk_band"]
        'STABLE'
    """
    if not profiles:
        return {
            "schema_version": DRIFT_TIMELINE_SCHEMA_VERSION,
            "drift_index": 0.0,
            "risk_band": "STABLE",
            "change_points": [],
            "summary_text": "No profiles available for drift analysis.",
        }
    
    if len(profiles) == 1:
        return {
            "schema_version": DRIFT_TIMELINE_SCHEMA_VERSION,
            "drift_index": 0.0,
            "risk_band": "STABLE",
            "change_points": [],
            "summary_text": "Single profile: no drift detected (baseline established).",
        }
    
    # Map risk bands to numeric values for variance calculation
    risk_band_values = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    
    # Extract risk band values
    risk_values = []
    for profile in profiles:
        band = profile.get("epistemic_risk_band", "LOW")
        risk_values.append(risk_band_values.get(band, 0))
    
    # Calculate drift index (normalized variance)
    mean_risk = sum(risk_values) / len(risk_values) if risk_values else 0
    variance = sum((v - mean_risk) ** 2 for v in risk_values) / len(risk_values) if risk_values else 0
    # Normalize to 0-1 range (max variance for 3 bands = 1.33, so divide by 1.33)
    drift_index = min(variance / 1.33, 1.0)
    
    # Determine risk band
    cfg = _DRIFT_THRESHOLDS
    if drift_index <= cfg["stable_max_drift"]:
        risk_band = "STABLE"
    elif drift_index <= cfg["drifting_max_drift"]:
        risk_band = "DRIFTING"
    else:
        risk_band = "VOLATILE"
    
    # Detect change points (significant transitions)
    change_points: List[Dict[str, Any]] = []
    change_threshold = cfg["change_point_threshold"]
    
    for i in range(1, len(profiles)):
        prev_band = profiles[i-1].get("epistemic_risk_band", "LOW")
        curr_band = profiles[i].get("epistemic_risk_band", "LOW")
        
        if prev_band != curr_band:
            prev_value = risk_band_values.get(prev_band, 0)
            curr_value = risk_band_values.get(curr_band, 0)
            change_magnitude = abs(curr_value - prev_value) / 2.0  # Normalize to 0-1
            
            if change_magnitude >= change_threshold:
                change_points.append({
                    "slice_name": profiles[i].get("slice_name", "unnamed"),
                    "transition": f"{prev_band} → {curr_band}",
                    "change_magnitude": round(change_magnitude, 3),
                    "index": i,
                })
    
    # Generate summary text
    if risk_band == "STABLE":
        summary_text = (
            f"Epistemic drift: STABLE (drift_index={drift_index:.3f}). "
            f"Risk bands show minimal variation across {len(profiles)} profiles. "
            f"{len(change_points)} significant change point(s) detected."
        )
    elif risk_band == "DRIFTING":
        summary_text = (
            f"Epistemic drift: DRIFTING (drift_index={drift_index:.3f}). "
            f"Moderate variation in risk bands across {len(profiles)} profiles. "
            f"{len(change_points)} significant change point(s) detected."
        )
    else:  # VOLATILE
        summary_text = (
            f"Epistemic drift: VOLATILE (drift_index={drift_index:.3f}). "
            f"High variation in risk bands across {len(profiles)} profiles. "
            f"{len(change_points)} significant change point(s) detected. "
            f"Review recommended."
        )
    
    return {
        "schema_version": DRIFT_TIMELINE_SCHEMA_VERSION,
        "drift_index": round(drift_index, 3),
        "risk_band": risk_band,
        "change_points": change_points,
        "summary_text": summary_text,
    }


def summarize_abstention_for_global_console(
    profile: Dict[str, Any],
    storyline: Dict[str, Any],
    drift_timeline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a unified abstention summary for global governance console.
    
    This function bridges abstention signals to the global governance dashboard,
    providing a single consolidated view of epistemic risk, trends, and drift.
    
    Args:
        profile: Epistemic profile from build_epistemic_abstention_profile()
                 (typically the most recent/latest profile)
        storyline: Storyline from build_abstention_storyline()
        drift_timeline: Drift timeline from build_epistemic_drift_timeline()
        
    Returns:
        Global console summary with:
        - abstention_status_light: "GREEN" | "YELLOW" | "RED"
        - epistemic_risk: Current epistemic risk band
        - storyline_snapshot: Trend summary from storyline
        - drift_band: Drift risk band from timeline
        - headline: Unified headline for governance dashboard
        
    Status Light Mapping:
        - GREEN: LOW risk + STABLE drift + IMPROVING/STABLE trend
        - YELLOW: MEDIUM risk OR DRIFTING drift OR DEGRADING trend
        - RED: HIGH risk OR VOLATILE drift
        
    Examples:
        >>> profile = build_epistemic_abstention_profile(snapshot)
        >>> storyline = build_abstention_storyline(profiles)
        >>> timeline = build_epistemic_drift_timeline(profiles)
        >>> console = summarize_abstention_for_global_console(profile, storyline, timeline)
        >>> console["abstention_status_light"]
        'GREEN'
    """
    # Extract key signals
    epistemic_risk = profile.get("epistemic_risk_band", "LOW")
    trend = storyline.get("global_epistemic_trend", "STABLE")
    drift_band = drift_timeline.get("risk_band", "STABLE")
    drift_index = drift_timeline.get("drift_index", 0.0)
    
    # Determine status light
    if epistemic_risk == "HIGH" or drift_band == "VOLATILE":
        status_light = "RED"
    elif epistemic_risk == "MEDIUM" or drift_band == "DRIFTING" or trend == "DEGRADING":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Generate unified headline
    if status_light == "GREEN":
        headline = (
            f"Abstention health: GREEN. Epistemic risk {epistemic_risk}, "
            f"trend {trend}, drift {drift_band}."
        )
    elif status_light == "YELLOW":
        headline = (
            f"Abstention health: YELLOW. Epistemic risk {epistemic_risk}, "
            f"trend {trend}, drift {drift_band}. Monitor recommended."
        )
    else:  # RED
        headline = (
            f"Abstention health: RED. Epistemic risk {epistemic_risk}, "
            f"trend {trend}, drift {drift_band} (drift_index={drift_index:.3f}). "
            f"Review required."
        )
    
    return {
        "abstention_status_light": status_light,
        "epistemic_risk": epistemic_risk,
        "storyline_snapshot": {
            "trend": trend,
            "story": storyline.get("story", ""),
        },
        "drift_band": drift_band,
        "drift_index": drift_index,
        "headline": headline,
    }


def compose_abstention_with_uplift_decision(
    epistemic_eval: Dict[str, Any],
    uplift_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compose abstention epistemic evaluation with uplift decision logic.
    
    This function applies upgrade rules: epistemic BLOCK upgrades uplift WARN → BLOCK,
    and DRIFTING drift adds advisory fields to the uplift tile.
    
    Args:
        epistemic_eval: Output from evaluate_abstention_for_uplift()
        uplift_eval: Output from summarize_abstentions_for_uplift()
        
    Returns:
        Composed uplift decision with:
        - final_status: "OK" | "WARN" | "BLOCK" - Final uplift decision
        - final_uplift_ok: bool - Final uplift safety flag
        - epistemic_upgrade_applied: bool - Whether epistemic gate upgraded status
        - advisory_fields: List of advisory messages (for DRIFTING, etc.)
        - blocking_slices: Combined blocking slices from both evaluations
        - reasons: Combined reasons from both evaluations
        
    Upgrade Rules:
        - Epistemic BLOCK → Final BLOCK (regardless of uplift_eval status)
        - Epistemic WARN + Uplift WARN → Final WARN
        - Epistemic WARN + Uplift OK → Final WARN (epistemic takes precedence)
        - Epistemic OK + Uplift WARN → Final WARN
        - Epistemic OK + Uplift OK → Final OK
        
    Examples:
        >>> epistemic = evaluate_abstention_for_uplift(compound)
        >>> uplift = summarize_abstentions_for_uplift(radar)
        >>> final = compose_abstention_with_uplift_decision(epistemic, uplift)
        >>> if not final["final_uplift_ok"]:
        ...     print(f"BLOCKED: {final['blocking_slices']}")
    """
    epistemic_status = epistemic_eval.get("status", "OK")
    uplift_status = uplift_eval.get("status", "OK")
    
    # Apply upgrade rule: epistemic BLOCK → final BLOCK
    if epistemic_status == "BLOCK":
        final_status = "BLOCK"
        final_uplift_ok = False
        epistemic_upgrade_applied = uplift_status != "BLOCK"  # Upgrade if uplift was WARN/OK
    elif epistemic_status == "WARN":
        # Epistemic WARN takes precedence over uplift OK
        if uplift_status == "BLOCK":
            final_status = "BLOCK"
            final_uplift_ok = False
            epistemic_upgrade_applied = False
        else:
            final_status = "WARN"
            final_uplift_ok = True  # Not blocked, but warned
            epistemic_upgrade_applied = uplift_status == "OK"
    else:  # epistemic OK
        # Use uplift status directly
        final_status = uplift_status
        final_uplift_ok = uplift_eval.get("uplift_safe", True)
        epistemic_upgrade_applied = False
    
    # Combine blocking slices
    epistemic_blocking = set(epistemic_eval.get("blocking_slices", []))
    uplift_blocking = set(uplift_eval.get("blocking_slices", []))
    combined_blocking = sorted(epistemic_blocking | uplift_blocking)
    
    # Combine reasons
    epistemic_reasons = epistemic_eval.get("reasons", [])
    uplift_reasons = uplift_eval.get("blocking_reasons", {})
    combined_reasons = list(epistemic_reasons)
    
    # Add uplift blocking reasons
    for slice_name, reason in uplift_reasons.items():
        combined_reasons.append(f"Uplift blocking: {slice_name} - {reason}")
    
    # Add advisory fields for DRIFTING (if applicable)
    advisory_fields: List[str] = []
    if epistemic_status == "WARN":
        advisory_fields.append("Epistemic risk evaluation: WARN - monitor recommended")
    
    # Add upgrade notification if applied
    if epistemic_upgrade_applied:
        combined_reasons.append(
            f"Epistemic gate upgraded status from {uplift_status} to {final_status}"
        )
    
    return {
        "final_status": final_status,
        "final_uplift_ok": final_uplift_ok,
        "epistemic_upgrade_applied": epistemic_upgrade_applied,
        "advisory_fields": advisory_fields,
        "blocking_slices": combined_blocking,
        "reasons": combined_reasons,
    }


# ---------------------------------------------------------------------------
# Completeness Check
# ---------------------------------------------------------------------------


def verify_tree_completeness() -> List[str]:
    """
    Verify that all AbstentionTypes are covered by the semantic tree.
    
    Returns:
        List of uncovered AbstentionType values (empty if complete)
    """
    uncovered = []
    for abst_type in AbstentionType:
        if abst_type not in ABSTENTION_TREE:
            uncovered.append(abst_type.value)
    return uncovered


# ---------------------------------------------------------------------------
# Governance Export
# ---------------------------------------------------------------------------


def export_semantics(out_path: Union[str, Path]) -> Path:
    """
    Export the complete abstention semantics to a JSON document.
    
    This produces a governance-ready document containing:
        - All AbstentionType values with their categories
        - Semantic tree structure
        - Legacy key mappings
        - Schema reference
    
    This document serves as input to future governance/telemetry tools
    and provides an auditable vocabulary for safety analysis.
    
    Args:
        out_path: Path to write the JSON document
        
    Returns:
        Path where document was written
        
    Example output:
        {
            "version": "1.0.0",
            "abstention_types": {
                "abstain_timeout": {
                    "category": "timeout_related",
                    "lean_specific": false,
                    "legacy_keys": ["timeout", "derivation_timeout"]
                },
                ...
            },
            "categories": {
                "timeout_related": ["abstain_timeout", "abstain_lean_timeout"],
                ...
            },
            "legacy_mappings": {
                "lean-disabled": "abstain_oracle_unavailable",
                ...
            },
            "schema_version": "draft-07"
        }
    """
    from .abstention_taxonomy import (
        COMPLETE_MAPPING,
        ABSTENTION_METHOD_STRINGS,
    )
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build abstention types section
    abstention_types = {}
    for abst_type in AbstentionType:
        category = ABSTENTION_TREE.get(abst_type)
        
        # Find legacy keys that map to this type
        legacy_keys = [
            key for key, mapped_type in COMPLETE_MAPPING.items()
            if mapped_type == abst_type
        ]
        
        abstention_types[abst_type.value] = {
            "category": category.value if category else None,
            "lean_specific": abst_type.value.startswith("abstain_lean_"),
            "legacy_keys": sorted(legacy_keys),
            "description": _get_type_description(abst_type),
        }
    
    # Build categories section
    categories = {}
    for category in SemanticCategory:
        types_in_category = get_types_for_category(category)
        categories[category.value] = [t.value for t in types_in_category]
    
    # Build legacy mappings section
    legacy_mappings = {
        key: mapped_type.value 
        for key, mapped_type in COMPLETE_MAPPING.items()
    }
    
    # Build complete document
    document = {
        "version": "1.0.0",
        "taxonomy_version": ABSTENTION_TAXONOMY_VERSION,
        "generated_by": "rfl.verification.abstention_semantics.export_semantics",
        "abstention_types": abstention_types,
        "categories": categories,
        "category_descriptions": {
            SemanticCategory.TIMEOUT_RELATED.value: "Processing exceeded time limit",
            SemanticCategory.RESOURCE_RELATED.value: "Resource budget exhausted",
            SemanticCategory.CRASH_RELATED.value: "Unexpected crash or error",
            SemanticCategory.ORACLE_RELATED.value: "External oracle/service unavailable",
            SemanticCategory.INVALID_RELATED.value: "Invalid input, state, or candidate",
        },
        "legacy_mappings": legacy_mappings,
        "verification_methods": sorted(ABSTENTION_METHOD_STRINGS),
        "schema_version": "draft-07",
        "schema_path": str(get_schema_path()),
        "invariants": {
            "INV-TAX-1": "No drift across layers",
            "INV-TAX-2": "All vocabularies collapse to canonical system",
            "INV-TAX-3": "Serializers stable and forward-compatible",
        },
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    
    return out_path


def _get_type_description(abst_type: AbstentionType) -> str:
    """Get human-readable description for an AbstentionType."""
    descriptions = {
        AbstentionType.ABSTAIN_TIMEOUT: "General timeout - processing exceeded time limit",
        AbstentionType.ABSTAIN_BUDGET: "Resource budget exhausted (memory, candidates, API calls)",
        AbstentionType.ABSTAIN_CRASH: "Unexpected crash or exception",
        AbstentionType.ABSTAIN_INVALID: "Invalid input, state, or non-tautology candidate",
        AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE: "External oracle or service unavailable",
        AbstentionType.ABSTAIN_LEAN_TIMEOUT: "Lean verification timed out",
        AbstentionType.ABSTAIN_LEAN_ERROR: "Lean verification error or crash",
    }
    return descriptions.get(abst_type, "No description available")


# Runtime verification
_uncovered = verify_tree_completeness()
if _uncovered:
    import warnings
    warnings.warn(
        f"ABSTENTION_TREE is incomplete. Missing types: {_uncovered}",
        UserWarning
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Semantic tree
    "ABSTENTION_TREE",
    "SemanticCategory",
    "categorize",
    "get_category",
    "get_types_for_category",
    "get_all_categories",
    # Predicates
    "is_timeout_related",
    "is_crash_related",
    "is_resource_related",
    "is_oracle_related",
    "is_invalid_related",
    # Schema
    "ABSTENTION_RECORD_SCHEMA",
    "get_schema",
    "get_schema_path",
    "export_schema",
    # Validation
    "AbstentionValidationError",
    "validate_abstention_data",
    "validate_abstention_record",
    "validate_abstention_json",
    "validate_or_raise",
    # Aggregation
    "aggregate_by_category",
    "aggregate_histogram_by_category",
    "get_category_summary",
    # Analytics (Phase II v1.1)
    "summarize_abstentions",
    "detect_abstention_red_flags",
    # Phase III: Red-Flag Feed & Global Health
    "HEALTH_SNAPSHOT_SCHEMA_VERSION",
    "build_abstention_health_snapshot",
    "build_abstention_radar",
    "summarize_abstentions_for_uplift",
    "summarize_abstentions_for_global_health",
    # Phase IV: Epistemic Risk Decomposition & Cross-Signal Integration
    "EPISTEMIC_PROFILE_SCHEMA_VERSION",
    "build_epistemic_abstention_profile",
    "compose_abstention_with_budget_and_perf",
    "build_abstention_director_panel",
    "build_abstention_storyline",
    "evaluate_abstention_for_uplift",
    # Phase V: Double-Helix Drift Radar & Global Governance Linkage
    "DRIFT_TIMELINE_SCHEMA_VERSION",
    "build_epistemic_drift_timeline",
    "summarize_abstention_for_global_console",
    "compose_abstention_with_uplift_decision",
    # Completeness & Governance
    "verify_tree_completeness",
    "export_semantics",
    # Versioning
    "ABSTENTION_TAXONOMY_VERSION",
    "get_taxonomy_version",
]
