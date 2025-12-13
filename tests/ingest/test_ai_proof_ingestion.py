"""
Tests for AI Proof Ingestion Adapter (Phase 1)

Tests the internal ingestion pipeline, provenance recording, and shadow mode enforcement.
"""

import pytest
from datetime import datetime, timezone

from backend.ingest.pipeline import (
    AIProofSubmission,
    ProvenanceMetadata,
    IngestResult,
    compute_raw_output_hash,
    compute_prompt_hash,
)
from backend.health.source_type_telemetry_adapter import (
    SOURCE_TYPE_INTERNAL,
    SOURCE_TYPE_EXTERNAL_AI,
    SourceTypeMetrics,
    build_source_type_metrics,
    build_source_type_telemetry_tile,
    compute_verification_rate,
    tag_telemetry_event,
)
from curriculum.shadow_mode import (
    ShadowModeFilter,
    DEFAULT_SHADOW_FILTER,
    require_shadow_mode_for_source,
    validate_shadow_mode_constraint,
    filter_metrics_for_curriculum,
    guard_slice_progression,
    build_shadow_mode_report,
)


class TestProvenanceMetadata:
    """Tests for ProvenanceMetadata validation."""

    def test_valid_metadata(self):
        """Valid metadata is accepted."""
        metadata = ProvenanceMetadata(
            source_type="external_ai",
            source_id="gpt-4-turbo-2025-01",
            raw_output_hash="a" * 64,
        )
        assert metadata.source_type == "external_ai"
        assert metadata.source_id == "gpt-4-turbo-2025-01"

    def test_invalid_source_type(self):
        """Invalid source type is rejected."""
        with pytest.raises(ValueError, match="source_type must be 'external_ai'"):
            ProvenanceMetadata(
                source_type="internal",
                source_id="test",
                raw_output_hash="a" * 64,
            )

    def test_empty_source_id(self):
        """Empty source_id is rejected."""
        with pytest.raises(ValueError, match="source_id is required"):
            ProvenanceMetadata(
                source_type="external_ai",
                source_id="",
                raw_output_hash="a" * 64,
            )

    def test_invalid_hash_length(self):
        """Invalid hash length is rejected."""
        with pytest.raises(ValueError, match="64-char SHA-256"):
            ProvenanceMetadata(
                source_type="external_ai",
                source_id="test",
                raw_output_hash="abc123",  # Too short
            )


class TestAIProofSubmission:
    """Tests for AIProofSubmission."""

    def test_creates_submission_id(self):
        """Submission ID is auto-generated."""
        sub = AIProofSubmission(
            statement="p -> p",
            proof_term="fun hp => hp",
            provenance=ProvenanceMetadata(
                source_type="external_ai",
                source_id="test-model",
                raw_output_hash="b" * 64,
            ),
        )
        assert sub.submission_id is not None
        assert len(sub.submission_id) == 36  # UUID format

    def test_attestation_hash_deterministic(self):
        """Same submission produces same attestation hash."""
        provenance = ProvenanceMetadata(
            source_type="external_ai",
            source_id="test-model",
            raw_output_hash="c" * 64,
        )
        sub1 = AIProofSubmission(
            statement="p -> p",
            proof_term="fun hp => hp",
            provenance=provenance,
            submission_id="fixed-id",
        )
        sub2 = AIProofSubmission(
            statement="p -> p",
            proof_term="fun hp => hp",
            provenance=provenance,
            submission_id="fixed-id",
        )
        assert sub1.compute_attestation_hash() == sub2.compute_attestation_hash()

    def test_attestation_hash_changes_with_content(self):
        """Different content produces different attestation hash."""
        provenance = ProvenanceMetadata(
            source_type="external_ai",
            source_id="test-model",
            raw_output_hash="d" * 64,
        )
        sub1 = AIProofSubmission(
            statement="p -> p",
            proof_term="fun hp => hp",
            provenance=provenance,
            submission_id="fixed-id",
        )
        sub2 = AIProofSubmission(
            statement="p -> q -> p",  # Different statement
            proof_term="fun hp => hp",
            provenance=provenance,
            submission_id="fixed-id",
        )
        assert sub1.compute_attestation_hash() != sub2.compute_attestation_hash()


class TestHashUtilities:
    """Tests for hash utility functions."""

    def test_raw_output_hash(self):
        """Raw output hash is 64-char hex."""
        result = compute_raw_output_hash("test output")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_raw_output_hash_deterministic(self):
        """Same input produces same hash."""
        h1 = compute_raw_output_hash("hello world")
        h2 = compute_raw_output_hash("hello world")
        assert h1 == h2

    def test_prompt_hash(self):
        """Prompt hash is 64-char hex."""
        result = compute_prompt_hash("Generate a proof for p -> p")
        assert len(result) == 64


class TestSourceTypeTelemetry:
    """Tests for source type telemetry adapter."""

    def test_verification_rate_no_attempts(self):
        """Zero attempts returns 100% rate."""
        assert compute_verification_rate(0, 0) == 1.0

    def test_verification_rate_all_verified(self):
        """All verified returns 100%."""
        assert compute_verification_rate(10, 0) == 1.0

    def test_verification_rate_mixed(self):
        """Mixed results compute correctly."""
        assert compute_verification_rate(8, 2) == 0.8

    def test_build_source_type_metrics(self):
        """Metrics are computed correctly by source type."""
        proofs = {
            "internal": [
                {"status": "success", "shadow_mode": False},
                {"status": "success", "shadow_mode": False},
                {"status": "failure", "shadow_mode": False},
            ],
            "external_ai": [
                {"status": "success", "shadow_mode": True},
                {"status": "queued", "shadow_mode": True},
            ],
        }
        metrics = build_source_type_metrics(proofs)

        assert "internal" in metrics
        assert metrics["internal"].total_proofs == 3
        assert metrics["internal"].verified_count == 2
        assert metrics["internal"].failed_count == 1

        assert "external_ai" in metrics
        assert metrics["external_ai"].total_proofs == 2
        assert metrics["external_ai"].shadow_mode_count == 2

    def test_build_telemetry_tile(self):
        """Telemetry tile has correct structure."""
        metrics = {
            SOURCE_TYPE_INTERNAL: SourceTypeMetrics(
                source_type=SOURCE_TYPE_INTERNAL,
                total_proofs=100,
                verified_count=95,
                failed_count=5,
                queued_count=0,
                verification_rate=0.95,
                shadow_mode_count=0,
            ),
        }
        tile = build_source_type_telemetry_tile(metrics)

        assert tile["schema_version"] == "1.0.0"
        assert tile["mode"] == "SHADOW"
        assert tile["internal"]["total_proofs"] == 100

    def test_tag_telemetry_event(self):
        """Events are tagged with source type."""
        event = {"proof_id": "123", "status": "success"}
        tagged = tag_telemetry_event(event, SOURCE_TYPE_EXTERNAL_AI, shadow_mode=True)

        assert tagged["source_type"] == SOURCE_TYPE_EXTERNAL_AI
        assert tagged["shadow_mode"] is True
        assert tagged["proof_id"] == "123"  # Original preserved

    def test_tag_invalid_source_type(self):
        """Invalid source type raises error."""
        with pytest.raises(ValueError, match="Invalid source_type"):
            tag_telemetry_event({}, "invalid_type")


class TestShadowModeFilter:
    """Tests for shadow mode filtering."""

    def test_filter_excludes_shadow_mode(self):
        """Shadow mode proofs are excluded by default."""
        proofs = [
            {"id": "1", "shadow_mode": False, "source_type": "internal"},
            {"id": "2", "shadow_mode": True, "source_type": "external_ai"},
            {"id": "3", "shadow_mode": False, "source_type": "internal"},
        ]
        filtered = DEFAULT_SHADOW_FILTER.filter_proofs(proofs)

        assert len(filtered) == 2
        assert all(p["id"] in ["1", "3"] for p in filtered)

    def test_filter_excludes_external_ai(self):
        """External AI proofs are excluded even without shadow_mode flag."""
        proofs = [
            {"id": "1", "shadow_mode": False, "source_type": "internal"},
            {"id": "2", "shadow_mode": False, "source_type": "external_ai"},
        ]
        filtered = DEFAULT_SHADOW_FILTER.filter_proofs(proofs)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "1"


class TestShadowModeEnforcement:
    """Tests for shadow mode constraint enforcement."""

    def test_external_ai_requires_shadow(self):
        """External AI source requires shadow mode."""
        assert require_shadow_mode_for_source("external_ai") is True

    def test_internal_no_shadow_required(self):
        """Internal source does not require shadow mode."""
        assert require_shadow_mode_for_source("internal") is False

    def test_validate_constraint_violation(self):
        """Constraint violation is detected."""
        error = validate_shadow_mode_constraint("external_ai", shadow_mode=False)
        assert error is not None
        assert "Shadow mode is required" in error

    def test_validate_constraint_satisfied(self):
        """Valid constraint returns None."""
        error = validate_shadow_mode_constraint("external_ai", shadow_mode=True)
        assert error is None


class TestSliceProgressionGuard:
    """Tests for slice progression guard."""

    def test_shadow_mode_blocked(self):
        """Shadow mode proofs are blocked from progression."""
        proof = {"id": "1", "shadow_mode": True, "source_type": "internal"}
        reason = guard_slice_progression(proof)

        assert reason is not None
        assert "Shadow mode" in reason

    def test_external_ai_blocked(self):
        """External AI proofs are blocked from progression."""
        proof = {"id": "1", "shadow_mode": False, "source_type": "external_ai"}
        reason = guard_slice_progression(proof)

        assert reason is not None
        assert "external_ai" in reason

    def test_internal_allowed(self):
        """Internal non-shadow proofs are allowed."""
        proof = {"id": "1", "shadow_mode": False, "source_type": "internal"}
        reason = guard_slice_progression(proof)

        assert reason is None


class TestFilterMetricsForCurriculum:
    """Tests for curriculum metric filtering."""

    def test_excludes_shadow_by_default(self):
        """Shadow mode proofs excluded from curriculum metrics."""
        proofs = [
            {"status": "success", "shadow_mode": False, "source_type": "internal"},
            {"status": "success", "shadow_mode": True, "source_type": "external_ai"},
            {"status": "failure", "shadow_mode": False, "source_type": "internal"},
        ]
        result = filter_metrics_for_curriculum(proofs)

        assert result["total_proofs"] == 2  # Only internal proofs
        assert result["verified_count"] == 1
        assert result["excluded_count"] == 1
        assert result["mode"] == "FILTERED"

    def test_include_shadow_for_reporting(self):
        """Shadow mode flag can be ignored for reporting, but source type still filtered."""
        proofs = [
            {"status": "success", "shadow_mode": False, "source_type": "internal"},
            {"status": "success", "shadow_mode": True, "source_type": "internal"},  # Internal with shadow
        ]
        result = filter_metrics_for_curriculum(proofs, include_shadow=True)

        # include_shadow=True allows internal shadow proofs through
        assert result["total_proofs"] == 2
        assert result["mode"] == "UNFILTERED"


class TestShadowModeReport:
    """Tests for shadow mode status report."""

    def test_report_structure(self):
        """Report has correct structure."""
        proofs = [
            {"status": "success", "shadow_mode": False, "source_type": "internal"},
            {"status": "success", "shadow_mode": True, "source_type": "external_ai"},
        ]
        report = build_shadow_mode_report(proofs)

        assert report["schema_version"] == "1.0.0"
        assert report["mode"] == "SHADOW"
        assert report["summary"]["total_proofs"] == 2
        assert report["summary"]["shadow_mode_count"] == 1
        assert report["shadow_breakdown"]["external_ai"] == 1

    def test_empty_report(self):
        """Empty proofs list produces valid report."""
        report = build_shadow_mode_report([])

        assert report["summary"]["total_proofs"] == 0
        assert "No proofs currently in shadow mode" in report["observations"]
