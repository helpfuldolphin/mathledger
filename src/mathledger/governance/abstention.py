"""Typed ABSTAINED preservation rules for Wave A2."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence, Set, Union

from mathledger.governance.trust_class import Outcome


class ValidationOutcome(str, Enum):
    """Valid typed outcomes for authority artifacts."""

    VERIFIED = Outcome.VERIFIED.value
    REFUTED = Outcome.REFUTED.value
    ABSTAINED = Outcome.ABSTAINED.value


VALID_OUTCOMES: Set[str] = {value.value for value in ValidationOutcome}


class AbstentionPreservationViolation(ValueError):
    """Raised when typed outcome invariants are violated."""

    ERROR_CODE = "ABSTENTION_PRESERVATION_VIOLATION"

    def __init__(
        self,
        message: str,
        *,
        artifact_index: int | None = None,
        claim_id: str | None = None,
        violation_type: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.artifact_index = artifact_index
        self.claim_id = claim_id
        self.violation_type = violation_type
        self.details = dict(details or {})

    def to_error_response(self) -> dict[str, Any]:
        """Structured error object for API or audit callers."""
        return {
            "error_code": self.ERROR_CODE,
            "message": str(self),
            "artifact_index": self.artifact_index,
            "claim_id": self.claim_id,
            "violation_type": self.violation_type,
            "details": self.details,
        }


def _normalize_outcome_value(value: Union[str, Enum]) -> str:
    if isinstance(value, Enum):
        return str(value.value).upper()
    return str(value).upper()


def verify_outcome_present(artifact: Mapping[str, Any], artifact_index: int = 0) -> str:
    """
    Enforce typed outcome presence on a single artifact.

    Returns normalized outcome string when valid.
    """
    claim_id = str(artifact.get("claim_id", "unknown"))

    if "validation_outcome" not in artifact:
        raise AbstentionPreservationViolation(
            "validation_outcome field is missing; ABSTAINED must be explicit.",
            artifact_index=artifact_index,
            claim_id=claim_id,
            violation_type="MISSING_FIELD",
            details={"artifact_keys": sorted(artifact.keys())},
        )

    outcome = artifact["validation_outcome"]
    if outcome is None:
        raise AbstentionPreservationViolation(
            "validation_outcome is null; use explicit 'ABSTAINED'.",
            artifact_index=artifact_index,
            claim_id=claim_id,
            violation_type="NULL_VALUE",
        )

    outcome_str = _normalize_outcome_value(outcome)
    if outcome_str not in VALID_OUTCOMES:
        raise AbstentionPreservationViolation(
            f"Invalid validation_outcome '{outcome}'. Expected VERIFIED, REFUTED, or ABSTAINED.",
            artifact_index=artifact_index,
            claim_id=claim_id,
            violation_type="INVALID_VALUE",
            details={"received_value": str(outcome), "valid_values": sorted(VALID_OUTCOMES)},
        )

    return outcome_str


def require_abstention_preservation(reasoning_artifacts: Sequence[Mapping[str, Any]]) -> None:
    """Batch gate: every artifact must carry a valid typed outcome."""
    for idx, artifact in enumerate(reasoning_artifacts):
        verify_outcome_present(artifact, artifact_index=idx)


def validate_outcome_aggregation(outcomes: Sequence[Union[str, Enum, None]]) -> str:
    """
    Aggregate outcomes without dropping ABSTAINED.

    Rules:
    - any REFUTED -> REFUTED
    - else any ABSTAINED -> ABSTAINED
    - else all VERIFIED -> VERIFIED
    - empty list -> ABSTAINED
    """
    if not outcomes:
        return ValidationOutcome.ABSTAINED.value

    normalized: list[str] = []
    for idx, outcome in enumerate(outcomes):
        if outcome is None:
            raise AbstentionPreservationViolation(
                "Null outcome in aggregation input.",
                artifact_index=idx,
                violation_type="NULL_VALUE",
            )
        outcome_str = _normalize_outcome_value(outcome)
        if outcome_str not in VALID_OUTCOMES:
            raise AbstentionPreservationViolation(
                f"Invalid aggregated outcome '{outcome}'.",
                artifact_index=idx,
                violation_type="INVALID_VALUE",
            )
        normalized.append(outcome_str)

    if ValidationOutcome.REFUTED.value in normalized:
        return ValidationOutcome.REFUTED.value
    if ValidationOutcome.ABSTAINED.value in normalized:
        return ValidationOutcome.ABSTAINED.value
    return ValidationOutcome.VERIFIED.value


def verify_not_coerced_to_null(
    artifact_before: Mapping[str, Any],
    artifact_after: Mapping[str, Any],
    artifact_index: int = 0,
) -> None:
    """
    Detect silent ABSTAINED->null coercion across transformations.
    """
    before_outcome = artifact_before.get("validation_outcome")
    if _normalize_outcome_value(before_outcome) == ValidationOutcome.ABSTAINED.value:
        if "validation_outcome" not in artifact_after or artifact_after.get("validation_outcome") is None:
            claim_id = str(artifact_before.get("claim_id", "unknown"))
            raise AbstentionPreservationViolation(
                "ABSTAINED outcome was silently dropped or coerced to null.",
                artifact_index=artifact_index,
                claim_id=claim_id,
                violation_type="COERCED_TO_NULL",
            )

