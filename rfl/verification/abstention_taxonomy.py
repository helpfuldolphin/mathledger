"""
Canonical Abstention Taxonomy for MathLedger RFL System

This module provides a unified vocabulary for abstention classification across all
verification and derivation components. It normalizes the various abstention strings
used throughout the codebase into a single, machine-readable enumeration.

DESIGN PRINCIPLES:
    1. Classification only — no semantic changes to when abstention occurs
    2. Backward compatible — serialization produces same strings as before
    3. Deterministic — same input always maps to same output
    4. Exhaustive — every known abstention string has a canonical mapping

USAGE:
    from rfl.verification import AbstentionType, classify_verification_method

    # Classify a verification method string
    method = "lean-disabled"
    abst_type = classify_verification_method(method)
    # Returns: AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE

    # Serialize for logging (backward compatible)
    serialized = serialize_abstention(abst_type)
    # Returns: "abstain_oracle_unavailable"

MAPPING TABLE:
    Verification Methods:
        "lean-disabled"           → ABSTAIN_ORACLE_UNAVAILABLE
        "lean-timeout"            → ABSTAIN_LEAN_TIMEOUT
        "lean-error"              → ABSTAIN_LEAN_ERROR
        "truth-table-error"       → ABSTAIN_INVALID
        "truth-table-non-tautology" → ABSTAIN_INVALID (not a tautology = invalid candidate)

    Breakdown Keys:
        "engine_failure"          → ABSTAIN_CRASH
        "timeout"                 → ABSTAIN_TIMEOUT
        "unexpected_error"        → ABSTAIN_CRASH
        "empty_run"               → ABSTAIN_INVALID
        "pending_validation"      → ABSTAIN_ORACLE_UNAVAILABLE
        "no_successful_proofs"    → ABSTAIN_INVALID
        "zero_throughput"         → ABSTAIN_INVALID
        "budget_exceeded"         → ABSTAIN_BUDGET
        "candidate_limit"         → ABSTAIN_BUDGET

Phase IIb Lean-Specific Types:
    ABSTAIN_LEAN_TIMEOUT and ABSTAIN_LEAN_ERROR are specialized subtypes that
    allow fine-grained analysis of Lean verification failures while still
    being classifiable under the broader ABSTAIN_TIMEOUT/ABSTAIN_CRASH categories.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Optional


class AbstentionType(str, Enum):
    """
    Canonical abstention type enumeration.

    Inherits from str to ensure JSON serialization produces string values,
    maintaining backward compatibility with existing JSONL logs.

    Categories:
        ABSTAIN_TIMEOUT: Processing exceeded time limit
        ABSTAIN_BUDGET: Resource budget exhausted (memory, candidates, API calls)
        ABSTAIN_CRASH: Unexpected crash or exception
        ABSTAIN_INVALID: Invalid input, state, or non-tautology candidate
        ABSTAIN_ORACLE_UNAVAILABLE: External oracle/service unavailable

    Phase IIb Lean-Specific:
        ABSTAIN_LEAN_TIMEOUT: Lean verification timed out
        ABSTAIN_LEAN_ERROR: Lean verification error/crash
    """

    # Core types (Phase I)
    ABSTAIN_TIMEOUT = "abstain_timeout"
    ABSTAIN_BUDGET = "abstain_budget"
    ABSTAIN_CRASH = "abstain_crash"
    ABSTAIN_INVALID = "abstain_invalid"
    ABSTAIN_ORACLE_UNAVAILABLE = "abstain_oracle_unavailable"

    # Lean-specific types (Phase IIb)
    ABSTAIN_LEAN_TIMEOUT = "abstain_lean_timeout"
    ABSTAIN_LEAN_ERROR = "abstain_lean_error"

    def __str__(self) -> str:
        """Return the string value for logging compatibility."""
        return self.value

    @property
    def is_lean_specific(self) -> bool:
        """Check if this is a Lean-specific abstention type."""
        return self in (AbstentionType.ABSTAIN_LEAN_TIMEOUT, AbstentionType.ABSTAIN_LEAN_ERROR)

    @property
    def general_category(self) -> "AbstentionType":
        """
        Map Lean-specific types to their general category.

        This allows analysis at both fine-grained (Lean-specific) and
        coarse-grained (general category) levels.
        """
        if self == AbstentionType.ABSTAIN_LEAN_TIMEOUT:
            return AbstentionType.ABSTAIN_TIMEOUT
        elif self == AbstentionType.ABSTAIN_LEAN_ERROR:
            return AbstentionType.ABSTAIN_CRASH
        return self


# ---------------------------------------------------------------------------
# Verification Method Classification
# ---------------------------------------------------------------------------

# Mapping from verification method strings to canonical abstention types
_VERIFICATION_METHOD_MAP: dict[str, AbstentionType] = {
    # Lean fallback states
    "lean-disabled": AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    "lean-timeout": AbstentionType.ABSTAIN_LEAN_TIMEOUT,
    "lean-error": AbstentionType.ABSTAIN_LEAN_ERROR,
    # Truth-table states
    "truth-table-error": AbstentionType.ABSTAIN_INVALID,
    "truth-table-non-tautology": AbstentionType.ABSTAIN_INVALID,
    # Legacy/alternative spellings (for robustness)
    "lean_disabled": AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    "lean_timeout": AbstentionType.ABSTAIN_LEAN_TIMEOUT,
    "lean_error": AbstentionType.ABSTAIN_LEAN_ERROR,
    "truth_table_error": AbstentionType.ABSTAIN_INVALID,
    # Bootstrap/statistical abstention
    "ABSTAIN": AbstentionType.ABSTAIN_INVALID,
}

# Set of all verification method strings that indicate abstention
# This replaces the ad-hoc ABSTENTION_METHODS frozenset in derivation/pipeline.py
ABSTENTION_METHOD_STRINGS: FrozenSet[str] = frozenset({
    "lean-disabled",
    "lean-timeout",
    "lean-error",
    "truth-table-error",
    "truth-table-non-tautology",
})


def classify_verification_method(method: str) -> Optional[AbstentionType]:
    """
    Classify a verification method string to its canonical abstention type.

    Args:
        method: The verification method string (e.g., "lean-disabled", "lean-timeout")

    Returns:
        The corresponding AbstentionType, or None if the method is not an abstention
        (e.g., "pattern", "truth-table", "lean" indicating success).

    Examples:
        >>> classify_verification_method("lean-disabled")
        AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE

        >>> classify_verification_method("lean-timeout")
        AbstentionType.ABSTAIN_LEAN_TIMEOUT

        >>> classify_verification_method("pattern")
        None  # "pattern" means successful verification

        >>> classify_verification_method("truth-table")
        None  # "truth-table" means successful verification
    """
    return _VERIFICATION_METHOD_MAP.get(method)


def is_abstention_method(method: str) -> bool:
    """
    Check if a verification method string indicates abstention.

    This is a convenience function equivalent to checking if
    classify_verification_method(method) is not None, but also includes
    the canonical ABSTENTION_METHOD_STRINGS set for backward compatibility.

    Args:
        method: The verification method string

    Returns:
        True if the method indicates abstention, False otherwise.

    Examples:
        >>> is_abstention_method("lean-disabled")
        True

        >>> is_abstention_method("pattern")
        False
    """
    return method in ABSTENTION_METHOD_STRINGS or method in _VERIFICATION_METHOD_MAP


# ---------------------------------------------------------------------------
# Breakdown Key Classification
# ---------------------------------------------------------------------------

# Mapping from experiment breakdown keys to canonical abstention types
_BREAKDOWN_KEY_MAP: dict[str, AbstentionType] = {
    # Crash/error states
    "engine_failure": AbstentionType.ABSTAIN_CRASH,
    "unexpected_error": AbstentionType.ABSTAIN_CRASH,
    "crash": AbstentionType.ABSTAIN_CRASH,
    # Timeout states
    "timeout": AbstentionType.ABSTAIN_TIMEOUT,
    "timeout_abstain": AbstentionType.ABSTAIN_TIMEOUT,
    "derivation_timeout": AbstentionType.ABSTAIN_TIMEOUT,
    # Invalid/empty states
    "empty_run": AbstentionType.ABSTAIN_INVALID,
    "no_successful_proofs": AbstentionType.ABSTAIN_INVALID,
    "zero_throughput": AbstentionType.ABSTAIN_INVALID,
    "invalid_input": AbstentionType.ABSTAIN_INVALID,
    # Pending/unavailable states
    "pending_validation": AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    "oracle_unavailable": AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    # Budget states
    "budget_exceeded": AbstentionType.ABSTAIN_BUDGET,
    "candidate_limit": AbstentionType.ABSTAIN_BUDGET,
    "resource_exhausted": AbstentionType.ABSTAIN_BUDGET,
    "memory_limit": AbstentionType.ABSTAIN_BUDGET,
    # Lean-specific (for histogram keys)
    "lean_timeout": AbstentionType.ABSTAIN_LEAN_TIMEOUT,
    "lean_error": AbstentionType.ABSTAIN_LEAN_ERROR,
    # Derivation abstention aggregate
    "derivation_abstain": AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    # Attestation tracking (meta-keys, map to general unavailable)
    "attestation_mass": AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    "attestation_events": AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
}


def classify_breakdown_key(key: str) -> Optional[AbstentionType]:
    """
    Classify an experiment breakdown key to its canonical abstention type.

    Breakdown keys are used in rfl/experiment.py to categorize abstention
    causes in the abstention_breakdown dictionary.

    Args:
        key: The breakdown key string (e.g., "engine_failure", "timeout")

    Returns:
        The corresponding AbstentionType, or None if the key is not recognized.

    Examples:
        >>> classify_breakdown_key("engine_failure")
        AbstentionType.ABSTAIN_CRASH

        >>> classify_breakdown_key("timeout")
        AbstentionType.ABSTAIN_TIMEOUT

        >>> classify_breakdown_key("budget_exceeded")
        AbstentionType.ABSTAIN_BUDGET
    """
    return _BREAKDOWN_KEY_MAP.get(key)


# ---------------------------------------------------------------------------
# Serialization / Deserialization
# ---------------------------------------------------------------------------


def serialize_abstention(abstention_type: AbstentionType) -> str:
    """
    Serialize an AbstentionType to a string for logging.

    This function returns the enum's string value, ensuring backward
    compatibility with existing JSONL logs that expect string values.

    Args:
        abstention_type: The AbstentionType to serialize

    Returns:
        The string value of the abstention type.

    Examples:
        >>> serialize_abstention(AbstentionType.ABSTAIN_TIMEOUT)
        'abstain_timeout'

        >>> serialize_abstention(AbstentionType.ABSTAIN_LEAN_ERROR)
        'abstain_lean_error'
    """
    return abstention_type.value


def deserialize_abstention(value: str) -> AbstentionType:
    """
    Deserialize a string to an AbstentionType.

    Args:
        value: The string value to deserialize (e.g., "abstain_timeout")

    Returns:
        The corresponding AbstentionType enum member.

    Raises:
        ValueError: If the value does not correspond to any AbstentionType.

    Examples:
        >>> deserialize_abstention("abstain_timeout")
        AbstentionType.ABSTAIN_TIMEOUT

        >>> deserialize_abstention("abstain_lean_error")
        AbstentionType.ABSTAIN_LEAN_ERROR

        >>> deserialize_abstention("unknown_value")
        Raises ValueError
    """
    try:
        return AbstentionType(value)
    except ValueError:
        raise ValueError(
            f"Unknown abstention type: {value!r}. "
            f"Valid values are: {[t.value for t in AbstentionType]}"
        )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def get_all_abstention_types() -> list[AbstentionType]:
    """
    Return all abstention types in definition order.

    Useful for iteration and completeness checks in tests.
    """
    return list(AbstentionType)


def get_core_abstention_types() -> list[AbstentionType]:
    """
    Return only the core (Phase I) abstention types.

    Excludes Lean-specific types for general analysis.
    """
    return [
        AbstentionType.ABSTAIN_TIMEOUT,
        AbstentionType.ABSTAIN_BUDGET,
        AbstentionType.ABSTAIN_CRASH,
        AbstentionType.ABSTAIN_INVALID,
        AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    ]


def get_lean_abstention_types() -> list[AbstentionType]:
    """
    Return only the Lean-specific (Phase IIb) abstention types.
    """
    return [
        AbstentionType.ABSTAIN_LEAN_TIMEOUT,
        AbstentionType.ABSTAIN_LEAN_ERROR,
    ]


def format_abstention_for_log(
    abstention_type: AbstentionType,
    context: Optional[dict] = None
) -> str:
    """
    Format an abstention type for structured logging.

    Args:
        abstention_type: The AbstentionType to format
        context: Optional context dictionary with additional details

    Returns:
        A formatted log string.

    Examples:
        >>> format_abstention_for_log(AbstentionType.ABSTAIN_LEAN_TIMEOUT)
        '[ABSTAIN:abstain_lean_timeout]'

        >>> format_abstention_for_log(
        ...     AbstentionType.ABSTAIN_CRASH,
        ...     {"error": "segfault"}
        ... )
        '[ABSTAIN:abstain_crash] error=segfault'
    """
    base = f"[ABSTAIN:{abstention_type.value}]"
    if context:
        ctx_str = " ".join(f"{k}={v}" for k, v in sorted(context.items()))
        return f"{base} {ctx_str}"
    return base


# ---------------------------------------------------------------------------
# Mapping Summary (for documentation and testing)
# ---------------------------------------------------------------------------

# Complete mapping from all known abstention strings to canonical types
# This is the authoritative reference for the taxonomy
COMPLETE_MAPPING: dict[str, AbstentionType] = {
    **_VERIFICATION_METHOD_MAP,
    **_BREAKDOWN_KEY_MAP,
}


def get_mapping_table() -> dict[str, str]:
    """
    Return the complete mapping table as a dict of string -> string.

    Useful for documentation and debugging.
    """
    return {k: v.value for k, v in COMPLETE_MAPPING.items()}


__all__ = [
    "AbstentionType",
    "classify_verification_method",
    "classify_breakdown_key",
    "serialize_abstention",
    "deserialize_abstention",
    "is_abstention_method",
    "ABSTENTION_METHOD_STRINGS",
    "COMPLETE_MAPPING",
    "get_all_abstention_types",
    "get_core_abstention_types",
    "get_lean_abstention_types",
    "format_abstention_for_log",
    "get_mapping_table",
]

