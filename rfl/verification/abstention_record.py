"""
Unified Abstention Record for RFL Pipeline
===========================================

This module provides the canonical AbstentionRecord dataclass that bridges
the verification layer (FailureState) and experiment layer (AbstentionType).

The AbstentionRecord is the single source of truth for abstention events
as they flow through the pipeline:

    Pipeline → Experiment → Runner → Telemetry

DESIGN PRINCIPLES:
    1. Immutable (frozen dataclass) - records are append-only audit trail
    2. Bidirectional mapping - can reconstruct either FailureState or AbstentionType
    3. Deterministic serialization - same input always produces same output
    4. Canonical ordering - histogram keys follow fixed order for consistency

CANONICAL ORDERING (for histogram keys and reports):
    1. abstain_timeout
    2. abstain_budget  
    3. abstain_invalid
    4. abstain_crash
    5. abstain_oracle_unavailable
    6. abstain_lean_timeout
    7. abstain_lean_error

PHASE II — VERIFICATION BUREAU
Agent B4 (failure-ops-4)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json

from .abstention_taxonomy import (
    AbstentionType,
    classify_verification_method,
    classify_breakdown_key,
    serialize_abstention,
)
from .failure_classifier import (
    FailureState,
    classify_exception,
    classify_from_status,
    failure_to_abstention,
    normalize_legacy_key,
)


# ---------------------------------------------------------------------------
# Global Validation Configuration
# ---------------------------------------------------------------------------

def _is_validation_enabled() -> bool:
    """
    Check if schema validation is enabled globally.
    
    Validation is opt-in for performance reasons. Enable via:
        - RFL_VALIDATE_ABSTENTIONS=1 environment variable
        - Explicit validate=True parameter on methods
    
    Returns:
        True if global validation is enabled
    """
    return os.environ.get("RFL_VALIDATE_ABSTENTIONS", "").lower() in ("1", "true", "yes")


# Module-level flag (can be set programmatically for tests)
GLOBAL_VALIDATION_ENABLED: bool = _is_validation_enabled()


# Canonical ordering for histogram keys and reports
# This ensures deterministic iteration order across all consumers
CANONICAL_ABSTENTION_ORDER: Tuple[AbstentionType, ...] = (
    AbstentionType.ABSTAIN_TIMEOUT,
    AbstentionType.ABSTAIN_BUDGET,
    AbstentionType.ABSTAIN_INVALID,
    AbstentionType.ABSTAIN_CRASH,
    AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE,
    AbstentionType.ABSTAIN_LEAN_TIMEOUT,
    AbstentionType.ABSTAIN_LEAN_ERROR,
)

# Set for O(1) lookup
CANONICAL_ABSTENTION_SET = frozenset(CANONICAL_ABSTENTION_ORDER)


@dataclass(frozen=True)
class AbstentionRecord:
    """
    Unified abstention record bridging verification and experiment layers.
    
    This immutable record captures a single abstention event with full
    provenance information. It can be created from either:
    - A FailureState (execution failure)
    - A verification method string (verification abstention)
    - A legacy breakdown key (backward compatibility)
    
    Attributes:
        abstention_type: Canonical AbstentionType enum value
        failure_state: Optional FailureState if originating from execution failure
        method: Optional verification method string that caused abstention
        details: Optional human-readable description of the abstention cause
        source: Origin of this record (e.g., "experiment", "pipeline", "runner")
        timestamp: When this abstention was recorded (ISO format)
        context: Optional additional context dictionary
        
    Examples:
        >>> # From execution failure
        >>> record = AbstentionRecord.from_failure_state(
        ...     FailureState.TIMEOUT_ABSTAIN,
        ...     details="Derivation timed out after 3600s"
        ... )
        >>> record.abstention_type
        AbstentionType.ABSTAIN_TIMEOUT
        
        >>> # From verification method
        >>> record = AbstentionRecord.from_verification_method("lean-disabled")
        >>> record.abstention_type
        AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE
        
        >>> # From legacy key (backward compatibility)
        >>> record = AbstentionRecord.from_legacy_key("engine_failure")
        >>> record.abstention_type
        AbstentionType.ABSTAIN_CRASH
    """
    
    abstention_type: AbstentionType
    failure_state: Optional[FailureState] = None
    method: Optional[str] = None
    details: Optional[str] = None
    source: str = "unknown"
    timestamp: Optional[str] = None
    context: Optional[Dict[str, Any]] = field(default=None, hash=False)
    
    def __post_init__(self) -> None:
        """Validate record consistency."""
        # Ensure abstention_type is valid
        if self.abstention_type not in CANONICAL_ABSTENTION_SET:
            raise ValueError(
                f"Invalid abstention_type: {self.abstention_type}. "
                f"Must be one of: {[t.value for t in CANONICAL_ABSTENTION_ORDER]}"
            )
    
    @classmethod
    def from_failure_state(
        cls,
        state: FailureState,
        details: Optional[str] = None,
        source: str = "experiment",
        context: Optional[Dict[str, Any]] = None,
    ) -> "AbstentionRecord":
        """
        Create an AbstentionRecord from a FailureState.
        
        Args:
            state: The FailureState that caused this abstention
            details: Optional description of what caused the failure
            source: Where this record originated
            context: Optional additional context
            
        Returns:
            AbstentionRecord with appropriate abstention_type
            
        Raises:
            ValueError: If state is SUCCESS (not an abstention)
        """
        if state == FailureState.SUCCESS:
            raise ValueError("Cannot create AbstentionRecord from SUCCESS state")
        
        abst_type = failure_to_abstention(state)
        if abst_type is None:
            # Defensive fallback
            abst_type = AbstentionType.ABSTAIN_CRASH
        
        return cls(
            abstention_type=abst_type,
            failure_state=state,
            method=None,
            details=details,
            source=source,
            timestamp=datetime.utcnow().isoformat() + "Z",
            context=context,
        )
    
    @classmethod
    def from_verification_method(
        cls,
        method: str,
        details: Optional[str] = None,
        source: str = "pipeline",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional["AbstentionRecord"]:
        """
        Create an AbstentionRecord from a verification method string.
        
        Args:
            method: The verification method string (e.g., "lean-disabled")
            details: Optional description
            source: Where this record originated
            context: Optional additional context
            
        Returns:
            AbstentionRecord if method indicates abstention, None otherwise
        """
        abst_type = classify_verification_method(method)
        if abst_type is None:
            return None
        
        return cls(
            abstention_type=abst_type,
            failure_state=None,
            method=method,
            details=details,
            source=source,
            timestamp=datetime.utcnow().isoformat() + "Z",
            context=context,
        )
    
    @classmethod
    def from_legacy_key(
        cls,
        key: str,
        details: Optional[str] = None,
        source: str = "legacy",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional["AbstentionRecord"]:
        """
        Create an AbstentionRecord from a legacy breakdown key.
        
        This provides backward compatibility with old abstention_breakdown
        dictionaries that use ad-hoc string keys.
        
        Args:
            key: The legacy breakdown key (e.g., "engine_failure", "timeout")
            details: Optional description
            source: Where this record originated
            context: Optional additional context
            
        Returns:
            AbstentionRecord if key is recognized, None otherwise
        """
        # First try canonical AbstentionType taxonomy
        abst_type = classify_breakdown_key(key)
        if abst_type is not None:
            return cls(
                abstention_type=abst_type,
                failure_state=None,
                method=None,
                details=details or f"Legacy key: {key}",
                source=source,
                timestamp=datetime.utcnow().isoformat() + "Z",
                context=context,
            )
        
        # Fall back to FailureState legacy mapping
        normalized = normalize_legacy_key(key)
        if normalized != key:
            # Key was normalized - find corresponding AbstentionType
            for fs in FailureState:
                if fs.value == normalized and fs != FailureState.SUCCESS:
                    abst_type = failure_to_abstention(fs)
                    if abst_type:
                        return cls(
                            abstention_type=abst_type,
                            failure_state=fs,
                            method=None,
                            details=details or f"Legacy key: {key} → {normalized}",
                            source=source,
                            timestamp=datetime.utcnow().isoformat() + "Z",
                            context=context,
                        )
        
        # Unrecognized key - return None (caller decides how to handle)
        return None
    
    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        context: Optional[Dict[str, Any]] = None,
        source: str = "experiment",
    ) -> "AbstentionRecord":
        """
        Create an AbstentionRecord from an exception.
        
        Args:
            exc: The exception that caused the abstention
            context: Optional context dict for classification
            source: Where this record originated
            
        Returns:
            AbstentionRecord with classified abstention type
        """
        failure_state = classify_exception(exc, context)
        return cls.from_failure_state(
            failure_state,
            details=f"{type(exc).__name__}: {exc}",
            source=source,
            context=context,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON/JSONL output.
        
        Returns:
            Dictionary with all fields, suitable for JSON serialization
        """
        return {
            "abstention_type": self.abstention_type.value,
            "failure_state": self.failure_state.value if self.failure_state else None,
            "method": self.method,
            "details": self.details,
            "source": self.source,
            "timestamp": self.timestamp,
            "context": self.context,
        }
    
    def to_json(self, validate: bool = False) -> str:
        """
        Serialize to JSON string.
        
        Args:
            validate: If True, validate against JSON schema before serializing.
                      Raises AbstentionValidationError on validation failure.
        
        Returns:
            JSON string representation
        """
        if validate:
            self.validate()
        return json.dumps(self.to_dict(), ensure_ascii=True)
    
    def validate(self) -> None:
        """
        Validate this record against the JSON schema.
        
        Raises:
            AbstentionValidationError: If validation fails
            
        This method is opt-in for performance-sensitive paths.
        Enable via RFL_VALIDATE_ABSTENTIONS=1 environment variable
        or call explicitly in tests/CI.
        """
        from .abstention_semantics import validate_abstention_record, AbstentionValidationError
        errors = validate_abstention_record(self)
        if errors:
            raise AbstentionValidationError(errors, self.to_dict())
    
    def is_valid(self) -> bool:
        """
        Check if this record is valid without raising.
        
        Returns:
            True if valid, False otherwise
        """
        from .abstention_semantics import validate_abstention_record
        return len(validate_abstention_record(self)) == 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], validate: bool = False) -> "AbstentionRecord":
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary with AbstentionRecord fields
            validate: If True, validate data before creating record
            
        Returns:
            AbstentionRecord instance
            
        Raises:
            AbstentionValidationError: If validate=True and data is invalid
        """
        if validate:
            from .abstention_semantics import validate_abstention_data, AbstentionValidationError
            errors = validate_abstention_data(data)
            if errors:
                raise AbstentionValidationError(errors, data)
        
        abst_type = AbstentionType(data["abstention_type"])
        failure_state = None
        if data.get("failure_state"):
            failure_state = FailureState(data["failure_state"])
        
        return cls(
            abstention_type=abst_type,
            failure_state=failure_state,
            method=data.get("method"),
            details=data.get("details"),
            source=data.get("source", "unknown"),
            timestamp=data.get("timestamp"),
            context=data.get("context"),
        )
    
    @classmethod
    def from_json(cls, json_str: str, validate: bool = False) -> "AbstentionRecord":
        """
        Deserialize from JSON string.
        
        Args:
            json_str: JSON string to parse
            validate: If True, validate data before creating record
            
        Returns:
            AbstentionRecord instance
            
        Raises:
            AbstentionValidationError: If validate=True and data is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        data = json.loads(json_str)
        return cls.from_dict(data, validate=validate)
    
    @property
    def canonical_key(self) -> str:
        """
        Return the canonical histogram key for this abstention.
        
        This is the value that should be used as a key in abstention
        histograms and breakdown dictionaries.
        """
        return self.abstention_type.value
    
    @property
    def general_category(self) -> AbstentionType:
        """
        Return the general category for this abstention.
        
        Maps Lean-specific types to their parent category.
        """
        return self.abstention_type.general_category


# ---------------------------------------------------------------------------
# Histogram Utilities
# ---------------------------------------------------------------------------


def create_canonical_histogram() -> Dict[str, int]:
    """
    Create an empty histogram with canonical ordering.
    
    Returns:
        Dictionary with all canonical keys initialized to 0,
        in canonical order.
    """
    return {t.value: 0 for t in CANONICAL_ABSTENTION_ORDER}


def normalize_histogram(
    histogram: Dict[str, int],
    include_zeros: bool = False,
) -> Dict[str, int]:
    """
    Normalize a histogram to use canonical keys in canonical order.
    
    This function:
    1. Maps legacy keys to canonical AbstentionType values
    2. Aggregates counts for keys that map to the same canonical type
    3. Returns keys in canonical order
    
    Args:
        histogram: Input histogram with potentially legacy keys
        include_zeros: If True, include canonical keys with zero count
        
    Returns:
        Normalized histogram with canonical keys in canonical order
    """
    # Start with canonical histogram if including zeros
    result = create_canonical_histogram() if include_zeros else {}
    
    for key, count in histogram.items():
        # Try to create an AbstentionRecord to get canonical type
        record = AbstentionRecord.from_legacy_key(key)
        if record:
            canonical_key = record.canonical_key
        else:
            # Check if already canonical
            try:
                AbstentionType(key)
                canonical_key = key
            except ValueError:
                # Unknown key - skip or warn
                continue
        
        result[canonical_key] = result.get(canonical_key, 0) + int(count)
    
    # Return in canonical order
    if include_zeros:
        return result
    
    ordered = {}
    for t in CANONICAL_ABSTENTION_ORDER:
        if t.value in result and result[t.value] > 0:
            ordered[t.value] = result[t.value]
    return ordered


def merge_histograms(
    *histograms: Dict[str, int],
    normalize: bool = True,
) -> Dict[str, int]:
    """
    Merge multiple histograms into one.
    
    Args:
        *histograms: Variable number of histograms to merge
        normalize: If True, normalize keys to canonical form
        
    Returns:
        Merged histogram
    """
    combined: Dict[str, int] = {}
    
    for hist in histograms:
        for key, count in hist.items():
            combined[key] = combined.get(key, 0) + int(count)
    
    if normalize:
        return normalize_histogram(combined)
    return combined


def histogram_to_records(
    histogram: Dict[str, int],
    source: str = "histogram",
) -> List[AbstentionRecord]:
    """
    Convert a histogram to a list of AbstentionRecords.
    
    Each histogram entry with count > 0 produces count records.
    
    Args:
        histogram: The histogram to convert
        source: Source label for the records
        
    Returns:
        List of AbstentionRecords
    """
    records = []
    for key, count in histogram.items():
        record = AbstentionRecord.from_legacy_key(key, source=source)
        if record:
            for _ in range(count):
                records.append(record)
    return records


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AbstentionRecord",
    "CANONICAL_ABSTENTION_ORDER",
    "CANONICAL_ABSTENTION_SET",
    "create_canonical_histogram",
    "normalize_histogram",
    "merge_histograms",
    "histogram_to_records",
]

