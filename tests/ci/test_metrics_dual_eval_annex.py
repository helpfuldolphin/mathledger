"""
Integration Tests for Metrics Dual Evaluation Annex.

SHADOW MODE: These tests verify advisory-only dual evaluation.
No governance decisions are enforced.

REAL-READY: Tests cover HYBRID mode annex generation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest import mock

import pytest


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def clean_env():
    """Ensure clean environment for threshold mode tests."""
    original = os.environ.get("METRIC_THRESHOLDS_MODE")
    yield
    if original is None:
        os.environ.pop("METRIC_THRESHOLDS_MODE", None)
    else:
        os.environ["METRIC_THRESHOLDS_MODE"] = original


@pytest.fixture
def hybrid_mode(clean_env):
    """Set HYBRID mode for tests."""
    os.environ["METRIC_THRESHOLDS_MODE"] = "HYBRID"
    yield


@pytest.fixture
def healthy_metrics() -> Dict[str, float]:
    """Metrics that should be GREEN in all modes (no divergence)."""
    return {
        "drift_magnitude": 0.10,
        "success_rate": 95.0,
        "budget_utilization": 50.0,
        "abstention_rate": 2.0,
        "block_rate": 0.03,
    }


@pytest.fixture
def divergent_metrics() -> Dict[str, float]:
    """Metrics at MOCK warn boundary (YELLOW in MOCK, GREEN in REAL)."""
    return {
        "drift_magnitude": 0.32,  # >= 0.30 (MOCK), < 0.35 (REAL)
        "success_rate": 78.0,     # < 80 (MOCK), >= 75 (REAL)
        "budget_utilization": 82.0,  # >= 80 (MOCK), < 85 (REAL)
        "abstention_rate": 6.0,   # >= 5 (MOCK), < 8 (REAL)
        "block_rate": 0.09,       # >= 0.08 (MOCK), < 0.12 (REAL)
    }


@pytest.fixture
def sample_evidence() -> Dict[str, Any]:
    """Sample evidence pack structure."""
    return {
        "schema_version": "1.0.0",
        "timestamp": "2025-12-11T00:00:00Z",
        "mode": "SHADOW",
        "governance": {
            "metrics": {
                "status_light": "GREEN",
            },
        },
    }


# ==============================================================================
# Test Class: Annex Building
# ==============================================================================


class TestBuildMetricsDualEvalAnnex:
    """Tests for build_metrics_dual_eval_annex function."""

    def test_annex_has_required_fields(self, healthy_metrics):
        """Annex should have all required fields."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(healthy_metrics)

        required_fields = [
            "schema_version",
            "mode",
            "advisory_only",
            "per_metric",
            "summary",
            "dual_verdict",
            "timestamp_note",
        ]

        for field in required_fields:
            assert field in annex, f"Missing required field: {field}"

    def test_annex_is_advisory_only(self, healthy_metrics):
        """Annex should always be advisory_only=True."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(healthy_metrics)
        assert annex["advisory_only"] is True

    def test_annex_schema_version(self, healthy_metrics):
        """Annex should have correct schema version."""
        from backend.health.metrics_dual_eval_annex import (
            DUAL_EVAL_ANNEX_SCHEMA_VERSION,
            build_metrics_dual_eval_annex,
        )

        annex = build_metrics_dual_eval_annex(healthy_metrics)
        assert annex["schema_version"] == DUAL_EVAL_ANNEX_SCHEMA_VERSION

    def test_annex_per_metric_analysis(self, divergent_metrics):
        """Annex should include per-metric analysis."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        per_metric = annex["per_metric"]
        assert "drift_magnitude" in per_metric
        assert "success_rate" in per_metric

        # Check per-metric structure
        drift = per_metric["drift_magnitude"]
        assert "value" in drift
        assert "mock_band" in drift
        assert "real_band" in drift
        assert "bands_match" in drift

    def test_annex_summary_counts(self, divergent_metrics):
        """Annex summary should have correct counts."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        summary = annex["summary"]
        assert "total_metrics" in summary
        assert "bands_match_count" in summary
        assert "bands_differ_count" in summary
        assert "differing_metrics" in summary
        assert "all_bands_match" in summary

        # With divergent metrics, bands should differ
        assert summary["bands_differ_count"] > 0
        assert not summary["all_bands_match"]

    def test_annex_dual_verdict_in_hybrid_mode(self, divergent_metrics):
        """HYBRID mode should include dual_verdict."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        dual_verdict = annex["dual_verdict"]
        assert dual_verdict is not None
        assert "mock_status" in dual_verdict
        assert "real_status" in dual_verdict
        assert "diverges" in dual_verdict

    def test_annex_no_dual_verdict_in_mock_mode(self, healthy_metrics):
        """MOCK mode should not include dual_verdict."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(healthy_metrics, mode="MOCK")

        assert annex["dual_verdict"] is None


# ==============================================================================
# Test Class: Determinism
# ==============================================================================


class TestAnnexDeterminism:
    """Tests for annex determinism (critical for advisory logging)."""

    def test_annex_is_deterministic(self, divergent_metrics):
        """Same inputs should produce identical annex."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex1 = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")
        annex2 = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        # Remove timestamp_note for comparison (may vary)
        annex1_copy = {k: v for k, v in annex1.items() if k != "timestamp_note"}
        annex2_copy = {k: v for k, v in annex2.items() if k != "timestamp_note"}

        assert annex1_copy == annex2_copy

    def test_annex_json_roundtrip_deterministic(self, divergent_metrics):
        """JSON round-trip should be deterministic."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        json1 = json.dumps(annex, sort_keys=True)
        json2 = json.dumps(annex, sort_keys=True)

        assert json1 == json2

    def test_annex_json_serializable(self, divergent_metrics):
        """Annex should be fully JSON-serializable."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        # Should not raise
        json_str = json.dumps(annex)
        assert isinstance(json_str, str)

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["advisory_only"] is True


# ==============================================================================
# Test Class: Evidence Pack Hook
# ==============================================================================


class TestEvidencePackHook:
    """Tests for attach_dual_eval_to_evidence function."""

    def test_attach_creates_dual_eval_key(self, sample_evidence, divergent_metrics):
        """Attachment should create dual_eval key under governance.metrics."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(sample_evidence, divergent_metrics)

        assert "governance" in enriched
        assert "metrics" in enriched["governance"]
        assert "dual_eval" in enriched["governance"]["metrics"]

    def test_attach_does_not_mutate_original(self, sample_evidence, divergent_metrics):
        """Attachment should not mutate original evidence."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        original_str = json.dumps(sample_evidence, sort_keys=True)
        enriched = attach_dual_eval_to_evidence(sample_evidence, divergent_metrics)

        # Original should be unchanged
        assert json.dumps(sample_evidence, sort_keys=True) == original_str

        # Enriched should be different
        assert "dual_eval" in enriched["governance"]["metrics"]
        assert "dual_eval" not in sample_evidence["governance"]["metrics"]

    def test_attach_with_no_metrics_skips(self, sample_evidence):
        """Attachment with no metrics should skip."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(sample_evidence, None)

        # Should not add dual_eval
        assert "dual_eval" not in enriched.get("governance", {}).get("metrics", {})

    def test_attach_with_empty_metrics_skips(self, sample_evidence):
        """Attachment with empty metrics should skip."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(sample_evidence, {})

        # Should not add dual_eval
        assert "dual_eval" not in enriched.get("governance", {}).get("metrics", {})

    def test_attach_preserves_existing_governance(self, sample_evidence, divergent_metrics):
        """Attachment should preserve existing governance data."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(sample_evidence, divergent_metrics)

        # Original metrics data should be preserved
        assert enriched["governance"]["metrics"]["status_light"] == "GREEN"
        # New dual_eval should be added
        assert enriched["governance"]["metrics"]["dual_eval"]["advisory_only"] is True


# ==============================================================================
# Test Class: Status Summary for CLI
# ==============================================================================


class TestStatusSummary:
    """Tests for build_dual_eval_status_summary function."""

    def test_summary_has_required_fields(self, divergent_metrics):
        """Summary should have all required fields (standardized shape)."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")

        # Standardized fields per spec
        required_fields = [
            "advisory_only",
            "mode",
            "diverges",  # Renamed from dual_verdict_diverges
            "bands_differ_count",
            "differing_metrics",  # Renamed from out_of_band_metrics
            "mock_status",
            "real_status",
        ]

        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"

    def test_summary_is_advisory_only(self, divergent_metrics):
        """Summary should always be advisory_only=True."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")
        assert summary["advisory_only"] is True

    def test_summary_detects_divergence(self, divergent_metrics):
        """Summary should detect MOCK/REAL divergence."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")

        assert summary["diverges"] is True  # Standardized field name
        assert summary["mock_status"] == "YELLOW"
        assert summary["real_status"] == "GREEN"

    def test_summary_no_divergence_healthy(self, healthy_metrics):
        """Summary should show no divergence for healthy metrics."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(healthy_metrics, mode="HYBRID")

        assert summary["diverges"] is False  # Standardized field name
        assert summary["mock_status"] == "GREEN"
        assert summary["real_status"] == "GREEN"


# ==============================================================================
# Test Class: HYBRID Mode Integration
# ==============================================================================


class TestHybridModeIntegration:
    """Integration tests for HYBRID mode dual evaluation."""

    def test_hybrid_mode_full_pipeline(self, hybrid_mode, divergent_metrics, sample_evidence):
        """Test full HYBRID pipeline: annex -> attach -> summary."""
        from backend.health.metrics_dual_eval_annex import (
            attach_dual_eval_to_evidence,
            build_dual_eval_status_summary,
            build_metrics_dual_eval_annex,
        )

        # Build annex
        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")
        assert annex["advisory_only"] is True
        assert annex["dual_verdict"]["diverges"] is True

        # Attach to evidence
        enriched = attach_dual_eval_to_evidence(sample_evidence, divergent_metrics)
        attached_annex = enriched["governance"]["metrics"]["dual_eval"]
        assert attached_annex["advisory_only"] is True

        # Build CLI summary
        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")
        assert summary["advisory_only"] is True
        assert summary["diverges"] is True  # Standardized field name

    def test_hybrid_mode_never_changes_governance_status(self, hybrid_mode, divergent_metrics):
        """HYBRID mode should NEVER change governance status (advisory only)."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        # Advisory only flag must always be True
        assert annex["advisory_only"] is True

        # Dual verdict exists but is informational only
        assert annex["dual_verdict"] is not None

        # No "governance_decision" or "enforce" fields should exist
        assert "governance_decision" not in annex
        assert "enforce" not in annex
        assert "block" not in annex
        assert "safe_for_promotion" not in annex

    def test_hybrid_mode_divergence_is_log_only(self, hybrid_mode, divergent_metrics):
        """Divergence detection should be log-only, not enforcement."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(divergent_metrics, mode="HYBRID")

        # Divergence is detected
        assert annex["dual_verdict"]["diverges"] is True

        # But it's purely informational
        assert annex["advisory_only"] is True
        assert "action" not in annex
        assert "enforcement" not in annex

    def test_annex_with_all_metrics(self):
        """Test annex with all five tracked metrics."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        metrics = {
            "drift_magnitude": 0.32,
            "success_rate": 78.0,
            "budget_utilization": 82.0,
            "abstention_rate": 6.0,
            "block_rate": 0.09,
        }

        annex = build_metrics_dual_eval_annex(metrics, mode="HYBRID")

        # All 5 metrics should be analyzed
        assert annex["summary"]["total_metrics"] == 5

        # All 5 should have divergent bands (boundary values)
        per_metric = annex["per_metric"]
        for metric_name in metrics:
            assert metric_name in per_metric
            assert "mock_band" in per_metric[metric_name]
            assert "real_band" in per_metric[metric_name]


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_metrics_payload(self):
        """Empty metrics should produce minimal annex."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex({}, mode="HYBRID")

        assert annex["advisory_only"] is True
        assert annex["summary"]["total_metrics"] == 0
        assert annex["summary"]["all_bands_match"] is True

    def test_partial_metrics_payload(self):
        """Partial metrics should only analyze provided metrics."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(
            {"drift_magnitude": 0.32}, mode="HYBRID"
        )

        assert annex["summary"]["total_metrics"] == 1
        assert "drift_magnitude" in annex["per_metric"]
        assert "success_rate" not in annex["per_metric"]

    def test_unknown_metrics_ignored(self):
        """Unknown metrics should be ignored gracefully."""
        from backend.health.metrics_dual_eval_annex import build_metrics_dual_eval_annex

        annex = build_metrics_dual_eval_annex(
            {"unknown_metric": 123.0, "drift_magnitude": 0.10}, mode="HYBRID"
        )

        # Only known metric should be analyzed
        assert annex["summary"]["total_metrics"] == 1
        assert "drift_magnitude" in annex["per_metric"]
        assert "unknown_metric" not in annex["per_metric"]


# ==============================================================================
# Test Class: Canonical Artifact Detection
# ==============================================================================


class TestCanonicalArtifactDetection:
    """Tests for discover_metrics_json and load_metrics_from_run_dir."""

    def test_discover_direct_path(self, tmp_path):
        """Discovery should find metrics_windows.json at direct path."""
        from backend.health.metrics_dual_eval_annex import discover_metrics_json

        # Create direct path file
        metrics_path = tmp_path / "metrics_windows.json"
        metrics_path.write_text('{"drift_magnitude": 0.25}')

        path, reason = discover_metrics_json(tmp_path)

        assert path == metrics_path
        assert reason == "direct:<run_dir>/metrics_windows.json"

    def test_discover_p3_subdir_path(self, tmp_path):
        """Discovery should find metrics in p3 subdirectory."""
        from backend.health.metrics_dual_eval_annex import discover_metrics_json

        # Create p3 subdirectory
        p3_dir = tmp_path / "p3"
        p3_dir.mkdir()
        metrics_path = p3_dir / "metrics_windows.json"
        metrics_path.write_text('{"drift_magnitude": 0.30}')

        path, reason = discover_metrics_json(tmp_path)

        assert path == metrics_path
        assert reason == "p3_subdir:<run_dir>/p3/metrics_windows.json"

    def test_discover_fl_subdir_path(self, tmp_path):
        """Discovery should find metrics in fl_* subdirectory."""
        from backend.health.metrics_dual_eval_annex import discover_metrics_json

        # Create fl_ subdirectory
        fl_dir = tmp_path / "fl_20251212_120000"
        fl_dir.mkdir()
        metrics_path = fl_dir / "metrics_windows.json"
        metrics_path.write_text('{"drift_magnitude": 0.35}')

        path, reason = discover_metrics_json(tmp_path)

        assert path == metrics_path
        assert "fl_subdir:" in reason
        assert "fl_20251212_120000" in reason

    def test_discover_direct_takes_priority(self, tmp_path):
        """Direct path should take priority over p3 subdirectory."""
        from backend.health.metrics_dual_eval_annex import discover_metrics_json

        # Create both direct and p3 paths
        direct_path = tmp_path / "metrics_windows.json"
        direct_path.write_text('{"drift_magnitude": 0.10}')

        p3_dir = tmp_path / "p3"
        p3_dir.mkdir()
        p3_path = p3_dir / "metrics_windows.json"
        p3_path.write_text('{"drift_magnitude": 0.20}')

        path, reason = discover_metrics_json(tmp_path)

        assert path == direct_path
        assert reason == "direct:<run_dir>/metrics_windows.json"

    def test_discover_nonexistent_returns_none(self, tmp_path):
        """Discovery should return None for nonexistent directory."""
        from backend.health.metrics_dual_eval_annex import discover_metrics_json

        nonexistent = tmp_path / "does_not_exist"
        path, reason = discover_metrics_json(nonexistent)

        assert path is None
        assert reason is None

    def test_discover_empty_dir_returns_none(self, tmp_path):
        """Discovery should return None for empty directory."""
        from backend.health.metrics_dual_eval_annex import discover_metrics_json

        path, reason = discover_metrics_json(tmp_path)

        assert path is None
        assert reason is None


# ==============================================================================
# Test Class: Deterministic Selection
# ==============================================================================


class TestDeterministicSelection:
    """Tests for deterministic artifact selection."""

    def test_selection_is_deterministic(self, tmp_path):
        """Same directory should always return same result."""
        from backend.health.metrics_dual_eval_annex import discover_metrics_json

        # Create p3 subdirectory with metrics
        p3_dir = tmp_path / "p3"
        p3_dir.mkdir()
        metrics_path = p3_dir / "metrics_windows.json"
        metrics_path.write_text('{"drift_magnitude": 0.32}')

        # Call multiple times
        results = [discover_metrics_json(tmp_path) for _ in range(5)]

        # All should be identical
        first = results[0]
        for result in results[1:]:
            assert result == first

    def test_load_returns_sha256(self, tmp_path):
        """Load function should return SHA256 of source file."""
        from backend.health.metrics_dual_eval_annex import load_metrics_from_run_dir
        import hashlib

        # Create metrics file
        metrics_path = tmp_path / "metrics_windows.json"
        content = '{"drift_magnitude": 0.25, "success_rate": 85.0}'
        metrics_path.write_text(content)

        # Calculate expected hash
        expected_sha256 = hashlib.sha256(content.encode()).hexdigest()

        metrics, selection_path, sha256 = load_metrics_from_run_dir(tmp_path)

        assert metrics is not None
        assert sha256 == expected_sha256
        assert selection_path is not None


# ==============================================================================
# Test Class: Standardized Status Signal Shape
# ==============================================================================


class TestStandardizedStatusSignalShape:
    """Tests for standardized signals.metrics_dual_eval shape."""

    def test_status_signal_has_diverges_field(self, divergent_metrics):
        """Status signal should have 'diverges' bool field."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")

        assert "diverges" in summary
        assert isinstance(summary["diverges"], bool)

    def test_status_signal_has_bands_differ_count(self, divergent_metrics):
        """Status signal should have 'bands_differ_count' int field."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")

        assert "bands_differ_count" in summary
        assert isinstance(summary["bands_differ_count"], int)

    def test_status_signal_differing_metrics_is_sorted(self):
        """Status signal 'differing_metrics' should always be sorted."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        # Use metrics at MOCK warn boundary
        metrics = {
            "drift_magnitude": 0.32,
            "success_rate": 78.0,
            "budget_utilization": 82.0,
            "abstention_rate": 6.0,
            "block_rate": 0.09,
        }

        summary = build_dual_eval_status_summary(metrics, mode="HYBRID")

        differing = summary["differing_metrics"]
        assert differing == sorted(differing)

    def test_status_signal_includes_source_provenance(self, healthy_metrics):
        """Status signal should include source_path and source_sha256 when provided."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(
            healthy_metrics,
            mode="HYBRID",
            source_path="p3_subdir:metrics_windows.json",
            source_sha256="abc123def456",
        )

        assert summary["source_path"] == "p3_subdir:metrics_windows.json"
        assert summary["source_sha256"] == "abc123def456"


# ==============================================================================
# Test Class: Evidence Pack Reference Block
# ==============================================================================


class TestEvidencePackReferenceBlock:
    """Tests for evidence pack manifest reference with sha256."""

    def test_evidence_includes_dual_eval_reference(self, sample_evidence, divergent_metrics):
        """Evidence should include dual_eval_reference when sha256 provided."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_path="p3_subdir:metrics_windows.json",
            source_sha256="abc123def456789",
        )

        assert "dual_eval_reference" in enriched["governance"]["metrics"]
        ref = enriched["governance"]["metrics"]["dual_eval_reference"]
        assert ref["sha256"] == "abc123def456789"
        assert ref["source_path"] == "p3_subdir:metrics_windows.json"

    def test_evidence_reference_has_schema_version(self, sample_evidence, divergent_metrics):
        """Evidence reference should include schema_version."""
        from backend.health.metrics_dual_eval_annex import (
            DUAL_EVAL_ANNEX_SCHEMA_VERSION,
            attach_dual_eval_to_evidence,
        )

        enriched = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_sha256="abc123",
        )

        ref = enriched["governance"]["metrics"]["dual_eval_reference"]
        assert ref["schema_version"] == DUAL_EVAL_ANNEX_SCHEMA_VERSION

    def test_evidence_no_reference_without_sha256(self, sample_evidence, divergent_metrics):
        """Evidence should not include reference when sha256 not provided."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(sample_evidence, divergent_metrics)

        # dual_eval should be present
        assert "dual_eval" in enriched["governance"]["metrics"]
        # But reference should not be present
        assert "dual_eval_reference" not in enriched["governance"]["metrics"]


# ==============================================================================
# Test Class: Schema Version and Mode Provenance
# ==============================================================================


class TestSchemaVersionAndModeProvenance:
    """Tests for schema_version passthrough and mode=SHADOW marker."""

    def test_status_signal_has_schema_version(self, healthy_metrics):
        """Status signal should include schema_version passthrough."""
        from backend.health.metrics_dual_eval_annex import (
            DUAL_EVAL_ANNEX_SCHEMA_VERSION,
            build_dual_eval_status_summary,
        )

        summary = build_dual_eval_status_summary(healthy_metrics, mode="HYBRID")

        assert "schema_version" in summary
        assert summary["schema_version"] == DUAL_EVAL_ANNEX_SCHEMA_VERSION

    def test_status_signal_has_shadow_mode_marker(self, healthy_metrics):
        """Status signal should have mode='SHADOW' constant marker."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(healthy_metrics, mode="HYBRID")

        assert summary["mode"] == "SHADOW"

    def test_source_path_requires_sha256(self, healthy_metrics):
        """source_path should only be included if source_sha256 is provided."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        # Only source_path, no sha256 - source_path should NOT be included
        summary = build_dual_eval_status_summary(
            healthy_metrics,
            mode="HYBRID",
            source_path="p3_subdir:metrics_windows.json",
        )

        assert "source_path" not in summary
        assert "source_sha256" not in summary

    def test_source_path_included_with_sha256(self, healthy_metrics):
        """source_path should be included when source_sha256 is provided."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        summary = build_dual_eval_status_summary(
            healthy_metrics,
            mode="HYBRID",
            source_path="p3_subdir:metrics_windows.json",
            source_sha256="abc123def456",
        )

        assert summary["source_path"] == "p3_subdir:metrics_windows.json"
        assert summary["source_sha256"] == "abc123def456"

    def test_sha256_present_reference_required(self, sample_evidence, divergent_metrics):
        """When sha256 provided, dual_eval_reference must be present."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_sha256="abc123def456",
        )

        # Reference should be present
        assert "dual_eval_reference" in enriched["governance"]["metrics"]
        ref = enriched["governance"]["metrics"]["dual_eval_reference"]
        assert ref["sha256"] == "abc123def456"

    def test_sha256_absent_no_reference(self, sample_evidence, divergent_metrics):
        """When sha256 not provided, dual_eval_reference must NOT be present."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_path="p3_subdir:metrics_windows.json",  # No sha256
        )

        # Reference should NOT be present
        assert "dual_eval_reference" not in enriched["governance"]["metrics"]


# ==============================================================================
# Test Class: Warning Hygiene
# ==============================================================================


class TestWarningHygiene:
    """Tests for warning generation hygiene - cap enforcement and top3 selection."""

    def test_warning_cap_to_one_line(self):
        """Warning should emit at most one line regardless of metrics count."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        # Use metrics that will produce 5 differing bands
        metrics = {
            "drift_magnitude": 0.32,
            "success_rate": 78.0,
            "budget_utilization": 82.0,
            "abstention_rate": 6.0,
            "block_rate": 0.09,
        }

        summary = build_dual_eval_status_summary(metrics, mode="HYBRID")

        # Verify divergence detected
        assert summary["diverges"] is True
        assert summary["bands_differ_count"] == 5

        # The warning generation logic caps to 1 line
        # (tested via generate_first_light_status integration)
        # Here we verify the signal has the data needed for the capped warning
        assert "bands_differ_count" in summary
        assert "differing_metrics" in summary
        assert len(summary["differing_metrics"]) == 5

    def test_top3_deterministic_selection(self):
        """Top 3 differing metrics should be deterministic (sorted)."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_status_summary

        # Use metrics that will produce 5 differing bands
        metrics = {
            "drift_magnitude": 0.32,
            "success_rate": 78.0,
            "budget_utilization": 82.0,
            "abstention_rate": 6.0,
            "block_rate": 0.09,
        }

        summary = build_dual_eval_status_summary(metrics, mode="HYBRID")

        # differing_metrics should be sorted
        differing = summary["differing_metrics"]
        assert differing == sorted(differing)

        # Top 3 should be deterministic (first 3 of sorted list)
        top3 = sorted(differing)[:3]

        # Verify determinism by calling multiple times
        for _ in range(5):
            summary2 = build_dual_eval_status_summary(metrics, mode="HYBRID")
            assert sorted(summary2["differing_metrics"])[:3] == top3


# ==============================================================================
# Test Class: GGFL Alignment View Adapter
# ==============================================================================


class TestGGFLAlignmentViewAdapter:
    """Tests for metrics_dual_eval_for_alignment_view() GGFL adapter."""

    def test_adapter_returns_none_for_none_metrics(self):
        """Adapter should return None when metrics_payload is None."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(None)
        assert signal is None

    def test_adapter_returns_none_for_empty_metrics(self):
        """Adapter should return None when metrics_payload is empty."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view({})
        assert signal is None

    def test_adapter_has_fixed_ggfl_shape(self, healthy_metrics):
        """Adapter should return fixed GGFL shape."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(healthy_metrics)

        # Fixed shape fields
        assert signal["signal_type"] == "SIG-MET"
        assert signal["status"] in ("ok", "warn")
        assert signal["conflict"] is False
        assert isinstance(signal["drivers"], list)
        assert isinstance(signal["summary"], str)

    def test_adapter_status_ok_when_no_divergence(self, healthy_metrics):
        """Adapter status should be 'ok' when no divergence."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(healthy_metrics)
        assert signal["status"] == "ok"

    def test_adapter_status_warn_when_diverges(self, divergent_metrics):
        """Adapter status should be 'warn' when diverges is True."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(divergent_metrics)
        assert signal["status"] == "warn"

    def test_adapter_drivers_max_3(self, divergent_metrics):
        """Adapter drivers should have max 3 items."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(divergent_metrics)
        assert len(signal["drivers"]) <= 3

    def test_adapter_drivers_ordering_deterministic(self):
        """Adapter drivers ordering should be deterministic."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        # Metrics that produce multiple differing bands
        metrics = {
            "drift_magnitude": 0.32,
            "success_rate": 78.0,
            "budget_utilization": 82.0,
            "abstention_rate": 6.0,
            "block_rate": 0.09,
        }

        signal1 = metrics_dual_eval_for_alignment_view(metrics)
        drivers1 = signal1["drivers"]

        # Call multiple times - should be identical
        for _ in range(5):
            signal2 = metrics_dual_eval_for_alignment_view(metrics)
            assert signal2["drivers"] == drivers1

    def test_adapter_is_json_serializable(self, divergent_metrics):
        """Adapter output should be JSON serializable."""
        import json
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(
            divergent_metrics,
            source_path="p3_subdir:metrics_windows.json",
            source_sha256="abc123def456",
        )

        # Should not raise
        json_str = json.dumps(signal)
        assert isinstance(json_str, str)

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "SIG-MET"

    def test_adapter_summary_is_one_sentence(self, divergent_metrics):
        """Adapter summary should be 1 sentence (ends with period)."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(divergent_metrics)
        summary = signal["summary"]

        # Should end with period and be a single sentence
        assert summary.endswith(".")
        # No multiple sentences (count periods)
        assert summary.count(".") == 1

    def test_adapter_includes_artifact_ref_when_sha256_provided(self, healthy_metrics):
        """Adapter should include artifact_ref when sha256 provided."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(
            healthy_metrics,
            source_path="p3_subdir:metrics_windows.json",
            source_sha256="abc123def456",
        )

        assert "artifact_ref" in signal
        assert signal["artifact_ref"]["sha256"] == "abc123def456"
        assert signal["artifact_ref"]["path"] == "p3_subdir:metrics_windows.json"

    def test_adapter_no_artifact_ref_without_sha256(self, healthy_metrics):
        """Adapter should not include artifact_ref when sha256 not provided."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(
            healthy_metrics,
            source_path="p3_subdir:metrics_windows.json",  # No sha256
        )

        assert "artifact_ref" not in signal


# ==============================================================================
# Test Class: Artifact Reference
# ==============================================================================


class TestArtifactReference:
    """Tests for build_dual_eval_artifact_reference()."""

    def test_artifact_reference_returns_none_without_sha256(self):
        """Artifact reference should return None without sha256."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_artifact_reference
        from pathlib import Path

        ref = build_dual_eval_artifact_reference(
            annex_path=Path("p3/dual_eval_annex.json"),
        )
        assert ref is None

    def test_artifact_reference_has_required_fields(self):
        """Artifact reference should have required fields."""
        from backend.health.metrics_dual_eval_annex import (
            DUAL_EVAL_ANNEX_SCHEMA_VERSION,
            build_dual_eval_artifact_reference,
        )
        from pathlib import Path

        ref = build_dual_eval_artifact_reference(
            annex_path=Path("p3/dual_eval_annex.json"),
            annex_sha256="abc123def456",
        )

        assert ref["artifact_type"] == "dual_eval_annex"
        assert ref["schema_version"] == DUAL_EVAL_ANNEX_SCHEMA_VERSION
        assert ref["sha256"] == "abc123def456"
        # Path string may use OS-specific separators
        assert "dual_eval_annex.json" in ref["path"]

    def test_artifact_reference_path_optional(self):
        """Artifact reference path should be optional."""
        from backend.health.metrics_dual_eval_annex import build_dual_eval_artifact_reference

        ref = build_dual_eval_artifact_reference(
            annex_sha256="abc123def456",
        )

        assert ref is not None
        assert "sha256" in ref
        assert "path" not in ref


# ==============================================================================
# Test Class: Warning Line Formatter
# ==============================================================================


class TestWarningLineFormatter:
    """Tests for format_dual_eval_warning_line()."""

    def test_warning_line_returns_none_when_no_divergence(self, healthy_metrics):
        """Warning line should return None when no divergence."""
        from backend.health.metrics_dual_eval_annex import (
            build_dual_eval_status_summary,
            format_dual_eval_warning_line,
        )

        summary = build_dual_eval_status_summary(healthy_metrics, mode="HYBRID")
        warning = format_dual_eval_warning_line(summary)

        assert warning is None

    def test_warning_line_single_line(self, divergent_metrics):
        """Warning line should be a single line."""
        from backend.health.metrics_dual_eval_annex import (
            build_dual_eval_status_summary,
            format_dual_eval_warning_line,
        )

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")
        warning = format_dual_eval_warning_line(summary)

        # Should be single line (no newlines)
        assert warning is not None
        assert "\n" not in warning

    def test_warning_line_includes_bands_count(self, divergent_metrics):
        """Warning line should include bands differ count."""
        from backend.health.metrics_dual_eval_annex import (
            build_dual_eval_status_summary,
            format_dual_eval_warning_line,
        )

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")
        warning = format_dual_eval_warning_line(summary)

        assert "band(s) differ" in warning
        assert "5" in warning  # 5 metrics at boundary

    def test_warning_line_includes_top3_metrics(self, divergent_metrics):
        """Warning line should include top 3 metrics."""
        from backend.health.metrics_dual_eval_annex import (
            build_dual_eval_status_summary,
            format_dual_eval_warning_line,
        )

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")
        warning = format_dual_eval_warning_line(summary)

        assert "top3:" in warning

    def test_warning_line_deterministic(self, divergent_metrics):
        """Warning line should be deterministic."""
        from backend.health.metrics_dual_eval_annex import (
            build_dual_eval_status_summary,
            format_dual_eval_warning_line,
        )

        summary = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")
        warning1 = format_dual_eval_warning_line(summary)

        # Call multiple times - should be identical
        for _ in range(5):
            summary2 = build_dual_eval_status_summary(divergent_metrics, mode="HYBRID")
            warning2 = format_dual_eval_warning_line(summary2)
            assert warning2 == warning1


# ==============================================================================
# Test Class: Manifest Binding and Extraction Source
# ==============================================================================


class TestManifestBindingAndExtractionSource:
    """Tests for manifest reference binding and extraction_source provenance."""

    def test_extraction_source_manifest_when_sha256_provided(self, healthy_metrics):
        """extraction_source should be MANIFEST when sha256 provided."""
        from backend.health.metrics_dual_eval_annex import (
            EXTRACTION_SOURCE_MANIFEST,
            metrics_dual_eval_for_alignment_view,
        )

        signal = metrics_dual_eval_for_alignment_view(
            healthy_metrics,
            source_sha256="abc123def456",
        )

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MANIFEST

    def test_extraction_source_missing_when_no_sha256(self, healthy_metrics):
        """extraction_source should be MISSING when no sha256 provided."""
        from backend.health.metrics_dual_eval_annex import (
            EXTRACTION_SOURCE_MISSING,
            metrics_dual_eval_for_alignment_view,
        )

        signal = metrics_dual_eval_for_alignment_view(healthy_metrics)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MISSING

    def test_extraction_source_explicit_override(self, healthy_metrics):
        """extraction_source should use explicit value when provided."""
        from backend.health.metrics_dual_eval_annex import (
            EXTRACTION_SOURCE_EVIDENCE_JSON,
            metrics_dual_eval_for_alignment_view,
        )

        signal = metrics_dual_eval_for_alignment_view(
            healthy_metrics,
            extraction_source=EXTRACTION_SOURCE_EVIDENCE_JSON,
        )

        assert signal["extraction_source"] == EXTRACTION_SOURCE_EVIDENCE_JSON

    def test_manifest_reference_present_only_with_sha256(self, sample_evidence, divergent_metrics):
        """Manifest reference should only be present when sha256 provided."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        # Without sha256 - no reference
        enriched_no_sha = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_path="p3_subdir:metrics_windows.json",
        )
        assert "dual_eval_reference" not in enriched_no_sha["governance"]["metrics"]

        # With sha256 - reference present
        enriched_with_sha = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_sha256="abc123def456",
        )
        assert "dual_eval_reference" in enriched_with_sha["governance"]["metrics"]

    def test_manifest_reference_deterministic(self, sample_evidence, divergent_metrics):
        """Manifest reference should be deterministic across calls."""
        from backend.health.metrics_dual_eval_annex import attach_dual_eval_to_evidence

        enriched1 = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_path="p3_subdir:metrics_windows.json",
            source_sha256="abc123def456",
        )
        ref1 = enriched1["governance"]["metrics"]["dual_eval_reference"]

        # Call multiple times - should be identical
        for _ in range(5):
            enriched2 = attach_dual_eval_to_evidence(
                sample_evidence,
                divergent_metrics,
                source_path="p3_subdir:metrics_windows.json",
                source_sha256="abc123def456",
            )
            ref2 = enriched2["governance"]["metrics"]["dual_eval_reference"]
            assert ref2 == ref1

    def test_manifest_reference_uses_artifact_builder(self, sample_evidence, divergent_metrics):
        """Manifest reference should be built using artifact reference builder."""
        from backend.health.metrics_dual_eval_annex import (
            DUAL_EVAL_ANNEX_SCHEMA_VERSION,
            attach_dual_eval_to_evidence,
        )

        enriched = attach_dual_eval_to_evidence(
            sample_evidence,
            divergent_metrics,
            source_sha256="abc123def456",
            annex_artifact_path="p3/dual_eval_annex.json",
            annex_artifact_sha256="def789ghi012",
        )

        ref = enriched["governance"]["metrics"]["dual_eval_reference"]
        # Should use annex_artifact_sha256 when provided
        assert ref["sha256"] == "def789ghi012"
        assert ref["artifact_type"] == "dual_eval_annex"
        assert ref["schema_version"] == DUAL_EVAL_ANNEX_SCHEMA_VERSION


# ==============================================================================
# Test Class: Driver Reason Codes
# ==============================================================================


class TestDriverReasonCodes:
    """Tests for GGFL driver reason codes."""

    def test_driver_codes_present_when_diverges(self, divergent_metrics):
        """driver_codes should be present when bands differ."""
        from backend.health.metrics_dual_eval_annex import (
            DRIVER_BANDS_DIFFER_PRESENT,
            DRIVER_TOP_METRICS_PRESENT,
            metrics_dual_eval_for_alignment_view,
        )

        signal = metrics_dual_eval_for_alignment_view(divergent_metrics)

        assert "driver_codes" in signal
        assert DRIVER_BANDS_DIFFER_PRESENT in signal["driver_codes"]
        assert DRIVER_TOP_METRICS_PRESENT in signal["driver_codes"]

    def test_driver_codes_empty_when_no_divergence(self, healthy_metrics):
        """driver_codes should be empty when no bands differ."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal = metrics_dual_eval_for_alignment_view(healthy_metrics)

        assert "driver_codes" in signal
        assert len(signal["driver_codes"]) == 0

    def test_driver_codes_deterministic(self, divergent_metrics):
        """driver_codes should be deterministic across calls."""
        from backend.health.metrics_dual_eval_annex import metrics_dual_eval_for_alignment_view

        signal1 = metrics_dual_eval_for_alignment_view(divergent_metrics)
        codes1 = signal1["driver_codes"]

        # Call multiple times - should be identical
        for _ in range(5):
            signal2 = metrics_dual_eval_for_alignment_view(divergent_metrics)
            assert signal2["driver_codes"] == codes1

    def test_drivers_and_codes_consistent(self, divergent_metrics):
        """drivers and driver_codes should be consistent."""
        from backend.health.metrics_dual_eval_annex import (
            DRIVER_BANDS_DIFFER_PRESENT,
            DRIVER_TOP_METRICS_PRESENT,
            metrics_dual_eval_for_alignment_view,
        )

        signal = metrics_dual_eval_for_alignment_view(divergent_metrics)

        drivers = signal["drivers"]
        codes = signal["driver_codes"]

        # If bands differ present, should have corresponding driver
        if DRIVER_BANDS_DIFFER_PRESENT in codes:
            assert any("band(s) differ" in d for d in drivers)

        # If top metrics present, should have metric names in drivers
        if DRIVER_TOP_METRICS_PRESENT in codes:
            # At least one driver should not contain "band(s) differ"
            non_band_drivers = [d for d in drivers if "band(s) differ" not in d]
            assert len(non_band_drivers) > 0
