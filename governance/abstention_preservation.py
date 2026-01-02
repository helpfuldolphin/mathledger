"""
Abstention Preservation Gate: Tier A Enforcement

This module enforces the abstention preservation invariant from FM §4.1:
    "Abstention is a typed outcome... first-class ledger artifact"

INVARIANT: ABSTAINED outcomes MUST:
1. Be explicitly typed (never null, undefined, or missing)
2. Be included in R_t Merkle root (for MV/PA/FV trust classes)
3. Be preserved through all data transformations
4. Never be silently dropped or coerced

ENFORCEMENT: The gate validates reasoning artifacts before attestation:
1. Every reasoning artifact MUST have validation_outcome field
2. validation_outcome MUST be one of: VERIFIED, REFUTED, ABSTAINED
3. null/None/undefined validation_outcome is a violation
4. Missing validation_outcome field is a violation

FAIL-CLOSED: Any detected violation raises AbstentionPreservationViolation.
There is no fallback, no retry, no degraded mode.

JURISDICTION (v0.1):
- Reasoning artifacts entering R_t
- Evidence pack construction
- Verification outcome aggregation
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Valid Outcomes (typed, never null)
# ---------------------------------------------------------------------------

class ValidationOutcome(str, Enum):
    """Valid verification outcomes - ABSTAINED is first-class."""
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    ABSTAINED = "ABSTAINED"


VALID_OUTCOMES: Set[str] = {outcome.value for outcome in ValidationOutcome}


# ---------------------------------------------------------------------------
# Audit Log: Abstention Preservation Violations
# ---------------------------------------------------------------------------

_abstention_violations: List[Dict[str, Any]] = []


def get_abstention_violations() -> List[Dict[str, Any]]:
    """Return a copy of the abstention violations log (for testing/audit)."""
    return list(_abstention_violations)


def clear_abstention_violations() -> None:
    """Clear the abstention violations log (for testing)."""
    _abstention_violations.clear()


def _record_abstention_violation(
    artifact_index: int,
    claim_id: Optional[str],
    reason: str,
    violation_type: str,
    request_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record an abstention preservation violation in the audit log.

    This entry does NOT contaminate R_t or U_t - it is stored in a
    separate log stream for forensic analysis.

    Args:
        artifact_index: Index of the violating artifact in the batch
        claim_id: The claim ID involved (if known)
        reason: Human-readable reason for the violation
        violation_type: Type of violation (MISSING_FIELD, NULL_VALUE, INVALID_VALUE)
        request_payload: Optional payload to hash for audit trail

    Returns:
        The recorded audit entry
    """
    if request_payload is not None:
        payload_json = json.dumps(request_payload, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    else:
        payload_hash = None

    entry = {
        "artifact_kind": "ABSTENTION_PRESERVATION_VIOLATION",
        "artifact_index": artifact_index,
        "claim_id": claim_id,
        "reason": reason,
        "violation_type": violation_type,
        "request_payload_hash": f"sha256:{payload_hash}" if payload_hash else None,
        "timestamp_epoch": int(time.time()),
    }

    _abstention_violations.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class AbstentionPreservationViolation(Exception):
    """
    Raised when a reasoning artifact has missing or null validation_outcome.

    This exception indicates a governance violation: the system attempted to
    process a reasoning artifact without a proper validation outcome, which
    would silently drop abstention status.

    FAIL-CLOSED: This exception must never be caught and suppressed.
    Any code that catches this exception must re-raise or terminate.

    Attributes:
        message: Human-readable error message
        artifact_index: Index of the violating artifact
        claim_id: The claim ID that has the violation
        violation_type: Type of violation
        details: Structured details for diagnostics
    """

    ERROR_CODE = "ABSTENTION_PRESERVATION_VIOLATION"

    def __init__(
        self,
        message: str,
        artifact_index: Optional[int] = None,
        claim_id: Optional[str] = None,
        violation_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.artifact_index = artifact_index
        self.claim_id = claim_id
        self.violation_type = violation_type
        self.details = details or {}

    def to_error_response(self) -> Dict[str, Any]:
        """
        Convert exception to structured HTTP error response.

        Returns:
            Dict suitable for JSON response body
        """
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "artifact_index": self.artifact_index,
            "claim_id": self.claim_id,
            "violation_type": self.violation_type,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Gate Functions
# ---------------------------------------------------------------------------


def verify_outcome_present(
    artifact: Dict[str, Any],
    artifact_index: int,
    record_violation: bool = True,
) -> None:
    """
    Verify that a reasoning artifact has a valid validation_outcome.

    This gate ensures ABSTAINED is never silently dropped by requiring
    an explicit typed outcome on every reasoning artifact.

    Args:
        artifact: The reasoning artifact dict
        artifact_index: Index in the batch (for error reporting)
        record_violation: If True, record violations to audit log

    Raises:
        AbstentionPreservationViolation: If outcome is missing, null, or invalid
    """
    claim_id = artifact.get("claim_id", "unknown")

    # Check field exists
    if "validation_outcome" not in artifact:
        reason = (
            f"Reasoning artifact missing validation_outcome field. "
            f"artifact_index={artifact_index}, claim_id={claim_id}. "
            f"ABSTAINED must be explicit, not missing."
        )

        if record_violation:
            _record_abstention_violation(
                artifact_index=artifact_index,
                claim_id=claim_id,
                reason=reason,
                violation_type="MISSING_FIELD",
                request_payload=artifact,
            )

        raise AbstentionPreservationViolation(
            f"Abstention preservation FAILED: validation_outcome field is missing. "
            f"All reasoning artifacts must have explicit validation_outcome. "
            f"ABSTAINED must be typed, not implied by absence.",
            artifact_index=artifact_index,
            claim_id=claim_id,
            violation_type="MISSING_FIELD",
            details={"artifact_keys": list(artifact.keys())},
        )

    outcome = artifact["validation_outcome"]

    # Check not null/None
    if outcome is None:
        reason = (
            f"Reasoning artifact has null validation_outcome. "
            f"artifact_index={artifact_index}, claim_id={claim_id}. "
            f"ABSTAINED must be explicit string 'ABSTAINED', not null."
        )

        if record_violation:
            _record_abstention_violation(
                artifact_index=artifact_index,
                claim_id=claim_id,
                reason=reason,
                violation_type="NULL_VALUE",
                request_payload=artifact,
            )

        raise AbstentionPreservationViolation(
            f"Abstention preservation FAILED: validation_outcome is null. "
            f"Use 'ABSTAINED' explicitly, not null/None. "
            f"Null outcomes cannot be distinguished from missing data.",
            artifact_index=artifact_index,
            claim_id=claim_id,
            violation_type="NULL_VALUE",
            details={"received_value": None},
        )

    # Normalize to string and uppercase for comparison
    if isinstance(outcome, Enum):
        outcome_str = outcome.value
    else:
        outcome_str = str(outcome).upper()

    # Check valid value
    if outcome_str not in VALID_OUTCOMES:
        reason = (
            f"Reasoning artifact has invalid validation_outcome '{outcome}'. "
            f"artifact_index={artifact_index}, claim_id={claim_id}. "
            f"Valid values: {VALID_OUTCOMES}"
        )

        if record_violation:
            _record_abstention_violation(
                artifact_index=artifact_index,
                claim_id=claim_id,
                reason=reason,
                violation_type="INVALID_VALUE",
                request_payload=artifact,
            )

        raise AbstentionPreservationViolation(
            f"Abstention preservation FAILED: invalid validation_outcome '{outcome}'. "
            f"Valid outcomes are: VERIFIED, REFUTED, ABSTAINED. "
            f"Custom or undefined outcomes are not permitted.",
            artifact_index=artifact_index,
            claim_id=claim_id,
            violation_type="INVALID_VALUE",
            details={"received_value": str(outcome), "valid_values": list(VALID_OUTCOMES)},
        )


def require_abstention_preservation(
    reasoning_artifacts: List[Dict[str, Any]],
    record_violation: bool = True,
) -> None:
    """
    Verify abstention preservation for a batch of reasoning artifacts.

    This is the MANDATORY gate that must be called before computing R_t.
    It ensures every artifact has a valid typed outcome (including ABSTAINED).

    FAIL-CLOSED: First violation raises AbstentionPreservationViolation.

    Args:
        reasoning_artifacts: List of reasoning artifact dicts
        record_violation: If True, record violations to audit log

    Raises:
        AbstentionPreservationViolation: If any artifact has invalid outcome
    """
    for idx, artifact in enumerate(reasoning_artifacts):
        verify_outcome_present(
            artifact=artifact,
            artifact_index=idx,
            record_violation=record_violation,
        )


def validate_outcome_aggregation(
    outcomes: List[Optional[str]],
    record_violation: bool = True,
) -> str:
    """
    Validate that outcome aggregation does not drop ABSTAINED.

    This gate prevents aggregation functions from ignoring ABSTAINED.
    If any outcome is ABSTAINED, the aggregate cannot be VERIFIED.

    Rules:
    - If any outcome is REFUTED → aggregate is REFUTED
    - If any outcome is ABSTAINED (and none REFUTED) → aggregate is ABSTAINED
    - If all outcomes are VERIFIED → aggregate is VERIFIED
    - Empty list → ABSTAINED (fail-safe)
    - null/None in list → violation

    Args:
        outcomes: List of outcome strings
        record_violation: If True, record violations to audit log

    Returns:
        The aggregated outcome string

    Raises:
        AbstentionPreservationViolation: If any outcome is null
    """
    if not outcomes:
        # Empty list → ABSTAINED (cannot verify nothing)
        return ValidationOutcome.ABSTAINED.value

    for idx, outcome in enumerate(outcomes):
        if outcome is None:
            reason = f"Null outcome in aggregation list at index {idx}"

            if record_violation:
                _record_abstention_violation(
                    artifact_index=idx,
                    claim_id=None,
                    reason=reason,
                    violation_type="NULL_VALUE",
                    request_payload={"outcomes": [str(o) for o in outcomes]},
                )

            raise AbstentionPreservationViolation(
                f"Abstention preservation FAILED: null outcome at index {idx} in aggregation. "
                f"Cannot aggregate outcomes containing null values.",
                artifact_index=idx,
                claim_id=None,
                violation_type="NULL_VALUE",
                details={"outcome_index": idx},
            )

    # Normalize outcomes
    normalized = [str(o).upper() if not isinstance(o, Enum) else o.value for o in outcomes]

    # REFUTED takes precedence
    if ValidationOutcome.REFUTED.value in normalized:
        return ValidationOutcome.REFUTED.value

    # ABSTAINED takes precedence over VERIFIED
    if ValidationOutcome.ABSTAINED.value in normalized:
        return ValidationOutcome.ABSTAINED.value

    # All VERIFIED
    return ValidationOutcome.VERIFIED.value


def verify_not_coerced_to_null(
    data: Dict[str, Any],
    outcome_key: str = "validation_outcome",
    record_violation: bool = True,
) -> None:
    """
    Verify that a serialized data structure has not coerced ABSTAINED to null.

    This gate catches JSON serialization bugs that might convert ABSTAINED
    to null during transport or storage.

    Args:
        data: The data structure to check
        outcome_key: The key containing the outcome
        record_violation: If True, record violations to audit log

    Raises:
        AbstentionPreservationViolation: If outcome is null after serialization
    """
    if outcome_key not in data:
        # Missing key is handled by verify_outcome_present
        return

    value = data[outcome_key]
    if value is None:
        reason = (
            f"Outcome '{outcome_key}' was coerced to null during serialization. "
            f"This indicates a bug in data transformation."
        )

        if record_violation:
            _record_abstention_violation(
                artifact_index=-1,
                claim_id=data.get("claim_id"),
                reason=reason,
                violation_type="COERCED_NULL",
                request_payload=data,
            )

        raise AbstentionPreservationViolation(
            f"Abstention preservation FAILED: '{outcome_key}' was coerced to null. "
            f"This indicates a serialization bug that would silently drop ABSTAINED.",
            artifact_index=None,
            claim_id=data.get("claim_id"),
            violation_type="COERCED_NULL",
            details={"key": outcome_key, "found_value": None},
        )
