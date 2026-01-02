"""
Test suite for Abstention Preservation Gate.

These tests verify that ABSTAINED outcomes are never silently dropped,
converted to null, or treated as missing values.

FM Reference: §4.1 - "abstention is a typed outcome... first-class ledger artifact"

Enforcement: Tier A (Structurally Enforced)

Key tests:
1. test_missing_outcome_rejected - Missing validation_outcome field
2. test_null_outcome_rejected - null/None validation_outcome
3. test_invalid_outcome_rejected - Invalid outcome values
4. test_valid_outcomes_accepted - VERIFIED, REFUTED, ABSTAINED all pass
5. test_aggregation_preserves_abstained - ABSTAINED propagates in aggregation
6. test_coerced_null_detected - Serialization bugs caught
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from governance.abstention_preservation import (
    AbstentionPreservationViolation,
    ValidationOutcome,
    VALID_OUTCOMES,
    clear_abstention_violations,
    get_abstention_violations,
    require_abstention_preservation,
    validate_outcome_aggregation,
    verify_not_coerced_to_null,
    verify_outcome_present,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_violations():
    """Clear the violations log before and after each test."""
    clear_abstention_violations()
    yield
    clear_abstention_violations()


@pytest.fixture
def valid_artifact() -> Dict[str, Any]:
    """Create a valid reasoning artifact with ABSTAINED outcome."""
    return {
        "claim_id": "sha256:test_claim_001",
        "claim_text": "Test claim",
        "trust_class": "MV",
        "validation_outcome": "ABSTAINED",
        "proof_payload": {},
    }


@pytest.fixture
def verified_artifact() -> Dict[str, Any]:
    """Create a valid reasoning artifact with VERIFIED outcome."""
    return {
        "claim_id": "sha256:test_claim_002",
        "claim_text": "2 + 2 = 4",
        "trust_class": "MV",
        "validation_outcome": "VERIFIED",
        "proof_payload": {"result": True},
    }


@pytest.fixture
def refuted_artifact() -> Dict[str, Any]:
    """Create a valid reasoning artifact with REFUTED outcome."""
    return {
        "claim_id": "sha256:test_claim_003",
        "claim_text": "2 + 2 = 5",
        "trust_class": "MV",
        "validation_outcome": "REFUTED",
        "proof_payload": {"result": False},
    }


# ---------------------------------------------------------------------------
# Test: Missing Outcome Field
# ---------------------------------------------------------------------------


class TestMissingOutcome:
    """Test that missing validation_outcome field is rejected."""

    def test_missing_outcome_raises_violation(self):
        """Verify missing field raises AbstentionPreservationViolation."""
        artifact = {
            "claim_id": "sha256:no_outcome",
            "claim_text": "Test claim",
            "trust_class": "MV",
            # validation_outcome intentionally missing
        }

        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            verify_outcome_present(artifact, artifact_index=0)

        assert exc_info.value.violation_type == "MISSING_FIELD"
        assert exc_info.value.claim_id == "sha256:no_outcome"
        assert "missing" in str(exc_info.value).lower()

    def test_missing_outcome_records_violation(self):
        """Verify violation is recorded to audit log."""
        artifact = {"claim_id": "sha256:no_outcome_log"}

        try:
            verify_outcome_present(artifact, artifact_index=5)
        except AbstentionPreservationViolation:
            pass

        violations = get_abstention_violations()
        assert len(violations) == 1
        assert violations[0]["violation_type"] == "MISSING_FIELD"
        assert violations[0]["artifact_index"] == 5
        assert violations[0]["claim_id"] == "sha256:no_outcome_log"

    def test_batch_with_missing_outcome_fails(self):
        """Verify batch validation fails on first missing outcome."""
        artifacts = [
            {"claim_id": "id1", "validation_outcome": "ABSTAINED"},
            {"claim_id": "id2"},  # Missing outcome
            {"claim_id": "id3", "validation_outcome": "VERIFIED"},
        ]

        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            require_abstention_preservation(artifacts)

        assert exc_info.value.artifact_index == 1


# ---------------------------------------------------------------------------
# Test: Null Outcome Value
# ---------------------------------------------------------------------------


class TestNullOutcome:
    """Test that null/None validation_outcome is rejected."""

    def test_null_outcome_raises_violation(self):
        """Verify null value raises AbstentionPreservationViolation."""
        artifact = {
            "claim_id": "sha256:null_outcome",
            "claim_text": "Test claim",
            "validation_outcome": None,
        }

        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            verify_outcome_present(artifact, artifact_index=0)

        assert exc_info.value.violation_type == "NULL_VALUE"
        assert "null" in str(exc_info.value).lower()

    def test_null_outcome_in_batch(self):
        """Verify batch fails on null outcome."""
        artifacts = [
            {"claim_id": "id1", "validation_outcome": "ABSTAINED"},
            {"claim_id": "id2", "validation_outcome": None},
        ]

        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            require_abstention_preservation(artifacts)

        assert exc_info.value.violation_type == "NULL_VALUE"
        assert exc_info.value.artifact_index == 1


# ---------------------------------------------------------------------------
# Test: Invalid Outcome Value
# ---------------------------------------------------------------------------


class TestInvalidOutcome:
    """Test that invalid outcome values are rejected."""

    @pytest.mark.parametrize("invalid_value", [
        "UNKNOWN",
        "PENDING",
        "SKIPPED",
        "N/A",
        "",
        "true",
        "false",
        0,
        1,
        True,
        False,
    ])
    def test_invalid_outcome_raises_violation(self, invalid_value):
        """Verify invalid values raise AbstentionPreservationViolation."""
        artifact = {
            "claim_id": "sha256:invalid",
            "validation_outcome": invalid_value,
        }

        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            verify_outcome_present(artifact, artifact_index=0)

        assert exc_info.value.violation_type == "INVALID_VALUE"

    def test_invalid_outcome_error_includes_valid_values(self):
        """Verify error message includes valid values."""
        artifact = {"claim_id": "id", "validation_outcome": "UNKNOWN"}

        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            verify_outcome_present(artifact, artifact_index=0)

        error_msg = str(exc_info.value)
        assert "VERIFIED" in error_msg
        assert "REFUTED" in error_msg
        assert "ABSTAINED" in error_msg


# ---------------------------------------------------------------------------
# Test: Valid Outcomes Accepted
# ---------------------------------------------------------------------------


class TestValidOutcomes:
    """Test that all valid outcomes are accepted."""

    @pytest.mark.parametrize("outcome", ["VERIFIED", "REFUTED", "ABSTAINED"])
    def test_valid_outcome_string_accepted(self, outcome):
        """Verify valid string outcomes pass validation."""
        artifact = {
            "claim_id": f"sha256:valid_{outcome}",
            "validation_outcome": outcome,
        }

        # Should not raise
        verify_outcome_present(artifact, artifact_index=0)

    @pytest.mark.parametrize("outcome", [
        ValidationOutcome.VERIFIED,
        ValidationOutcome.REFUTED,
        ValidationOutcome.ABSTAINED,
    ])
    def test_valid_outcome_enum_accepted(self, outcome):
        """Verify Enum outcomes pass validation."""
        artifact = {
            "claim_id": f"sha256:valid_enum",
            "validation_outcome": outcome,
        }

        verify_outcome_present(artifact, artifact_index=0)

    def test_mixed_valid_batch_accepted(
        self,
        valid_artifact,
        verified_artifact,
        refuted_artifact,
    ):
        """Verify batch with all valid outcomes passes."""
        artifacts = [valid_artifact, verified_artifact, refuted_artifact]

        # Should not raise
        require_abstention_preservation(artifacts)

    def test_all_abstained_batch_accepted(self):
        """Verify batch with all ABSTAINED passes."""
        artifacts = [
            {"claim_id": f"id{i}", "validation_outcome": "ABSTAINED"}
            for i in range(5)
        ]

        # Should not raise
        require_abstention_preservation(artifacts)


# ---------------------------------------------------------------------------
# Test: Aggregation Preserves ABSTAINED
# ---------------------------------------------------------------------------


class TestAggregationPreservation:
    """Test that outcome aggregation does not drop ABSTAINED."""

    def test_single_abstained_stays_abstained(self):
        """Verify single ABSTAINED aggregates to ABSTAINED."""
        result = validate_outcome_aggregation(["ABSTAINED"])
        assert result == "ABSTAINED"

    def test_abstained_with_verified_is_abstained(self):
        """Verify ABSTAINED + VERIFIED aggregates to ABSTAINED."""
        result = validate_outcome_aggregation(["VERIFIED", "ABSTAINED"])
        assert result == "ABSTAINED"

    def test_abstained_with_verified_is_abstained_order_independent(self):
        """Verify order doesn't affect aggregation."""
        result1 = validate_outcome_aggregation(["ABSTAINED", "VERIFIED"])
        result2 = validate_outcome_aggregation(["VERIFIED", "ABSTAINED"])
        assert result1 == result2 == "ABSTAINED"

    def test_refuted_takes_precedence(self):
        """Verify REFUTED overrides ABSTAINED and VERIFIED."""
        result = validate_outcome_aggregation([
            "VERIFIED", "ABSTAINED", "REFUTED"
        ])
        assert result == "REFUTED"

    def test_all_verified_is_verified(self):
        """Verify all VERIFIED aggregates to VERIFIED."""
        result = validate_outcome_aggregation([
            "VERIFIED", "VERIFIED", "VERIFIED"
        ])
        assert result == "VERIFIED"

    def test_empty_list_is_abstained(self):
        """Verify empty list aggregates to ABSTAINED (cannot verify nothing)."""
        result = validate_outcome_aggregation([])
        assert result == "ABSTAINED"

    def test_null_in_aggregation_raises(self):
        """Verify null in aggregation list raises violation."""
        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            validate_outcome_aggregation(["VERIFIED", None, "ABSTAINED"])

        assert exc_info.value.violation_type == "NULL_VALUE"
        assert exc_info.value.artifact_index == 1


# ---------------------------------------------------------------------------
# Test: Coerced Null Detection
# ---------------------------------------------------------------------------


class TestCoercedNullDetection:
    """Test that serialization bugs are caught."""

    def test_coerced_null_raises(self):
        """Verify null after serialization is detected."""
        data = {
            "claim_id": "sha256:coerced",
            "validation_outcome": None,  # Simulates JSON parse of null
        }

        with pytest.raises(AbstentionPreservationViolation) as exc_info:
            verify_not_coerced_to_null(data)

        assert exc_info.value.violation_type == "COERCED_NULL"

    def test_valid_outcome_passes_coercion_check(self):
        """Verify valid outcomes pass coercion check."""
        data = {
            "claim_id": "sha256:valid",
            "validation_outcome": "ABSTAINED",
        }

        # Should not raise
        verify_not_coerced_to_null(data)

    def test_missing_key_not_flagged_as_coercion(self):
        """Verify missing key is not flagged as coercion (different check)."""
        data = {"claim_id": "sha256:no_key"}

        # Should not raise (missing key is handled by verify_outcome_present)
        verify_not_coerced_to_null(data)


# ---------------------------------------------------------------------------
# Test: Error Response Format
# ---------------------------------------------------------------------------


class TestErrorResponseFormat:
    """Test that error responses are properly structured."""

    def test_error_response_has_required_fields(self):
        """Verify error response includes all required fields."""
        artifact = {"claim_id": "sha256:error_test"}

        try:
            verify_outcome_present(artifact, artifact_index=3)
        except AbstentionPreservationViolation as e:
            response = e.to_error_response()

            assert "error_code" in response
            assert response["error_code"] == "ABSTENTION_PRESERVATION_VIOLATION"
            assert "message" in response
            assert "artifact_index" in response
            assert response["artifact_index"] == 3
            assert "claim_id" in response
            assert "violation_type" in response
            assert "details" in response

    def test_error_code_is_constant(self):
        """Verify error code is consistent."""
        artifact = {"claim_id": "id", "validation_outcome": None}

        try:
            verify_outcome_present(artifact, artifact_index=0)
        except AbstentionPreservationViolation as e:
            assert e.ERROR_CODE == "ABSTENTION_PRESERVATION_VIOLATION"


# ---------------------------------------------------------------------------
# Test: No R_t/U_t Contamination
# ---------------------------------------------------------------------------


class TestNoContamination:
    """Test that violations do not contaminate attestation roots."""

    def test_violation_log_does_not_affect_attestation(self):
        """Verify violation log is separate from attestation."""
        from attestation.dual_root import (
            compute_composite_root,
            compute_reasoning_root,
            compute_ui_root,
        )

        # Test data
        uvil_events = [{"event_id": "evt_001", "action": "test"}]
        reasoning_artifacts = [{
            "claim_id": "sha256:001",
            "claim_text": "Test",
            "validation_outcome": "ABSTAINED",
        }]

        # Compute roots before violation
        u_t_before = compute_ui_root(uvil_events)
        r_t_before = compute_reasoning_root(reasoning_artifacts)
        h_t_before = compute_composite_root(r_t_before, u_t_before)

        # Create a violation
        try:
            verify_outcome_present({"claim_id": "bad"}, artifact_index=0)
        except AbstentionPreservationViolation:
            pass

        # Verify violation was logged
        violations = get_abstention_violations()
        assert len(violations) == 1

        # Compute roots after violation
        u_t_after = compute_ui_root(uvil_events)
        r_t_after = compute_reasoning_root(reasoning_artifacts)
        h_t_after = compute_composite_root(r_t_after, u_t_after)

        # Roots must be identical
        assert u_t_before == u_t_after, "Violation affected U_t!"
        assert r_t_before == r_t_after, "Violation affected R_t!"
        assert h_t_before == h_t_after, "Violation affected H_t!"


# ---------------------------------------------------------------------------
# Test: Integration with Reasoning Artifacts
# ---------------------------------------------------------------------------


class TestReasoningArtifactIntegration:
    """Test integration with actual reasoning artifact structure."""

    def test_full_reasoning_artifact_passes(self):
        """Verify complete reasoning artifact passes validation."""
        artifact = {
            "claim_id": "sha256:abc123def456",
            "claim_text": "forall x: nat, x + 0 = x",
            "trust_class": "FV",
            "validation_outcome": "ABSTAINED",
            "proof_payload": {},
            "rationale": "Identity property of addition",
            "v0_mock": True,
        }

        # Should not raise
        verify_outcome_present(artifact, artifact_index=0)

    def test_minimal_reasoning_artifact_passes(self):
        """Verify minimal required fields pass validation."""
        artifact = {
            "claim_id": "sha256:minimal",
            "validation_outcome": "ABSTAINED",
        }

        # Should not raise
        verify_outcome_present(artifact, artifact_index=0)
