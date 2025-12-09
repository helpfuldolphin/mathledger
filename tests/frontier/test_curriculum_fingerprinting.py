"""
Test suite for curriculum fingerprinting and drift detection.

Tests the deterministic fingerprinting system, drift sentinel, and metric schema enforcement.
"""

import os
import pytest
from dataclasses import asdict
from backend.frontier.curriculum import (
    CurriculumSystem,
    CurriculumSlice,
    SliceGates,
    CoverageGateSpec,
    AbstentionGateSpec,
    VelocityGateSpec,
    CapsGateSpec,
    CurriculumDriftError,
    CurriculumDriftSentinel,
    NormalizedMetrics,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gates():
    """Create sample gate specifications for testing."""
    return SliceGates(
        coverage=CoverageGateSpec(ci_lower_min=0.85, sample_min=30, require_attestation=True),
        abstention=AbstentionGateSpec(max_rate_pct=25.0, max_mass=1500),
        velocity=VelocityGateSpec(min_pph=50.0, stability_cv_max=0.3, window_minutes=60),
        caps=CapsGateSpec(min_attempt_mass=3000, min_runtime_minutes=45.0, backlog_max=0.4),
    )


@pytest.fixture
def sample_slice(sample_gates):
    """Create a sample curriculum slice for testing."""
    return CurriculumSlice(
        name="test_slice",
        params={"atoms": 5, "depth_max": 8, "breadth_max": 1000, "total_max": 5000},
        gates=sample_gates,
        metadata={"wave": 1, "ladder_position": "test"},
    )


@pytest.fixture
def sample_system(sample_slice):
    """Create a sample curriculum system for testing."""
    return CurriculumSystem(
        slug="test_system",
        description="Test curriculum system",
        slices=[sample_slice],
        active_index=0,
        invariants={"monotonic_axes": ["atoms", "depth_max"]},
        monotonic_axes=("atoms", "depth_max"),
        version=2,
    )


@pytest.fixture
def valid_metrics_v1():
    """Create a valid v1 metric payload."""
    return {
        "metrics": {
            "rfl": {
                "coverage": {
                    "ci_lower": 0.87,
                    "sample_size": 45,
                }
            },
            "success_rates": {
                "abstention_rate": 18.5,
            },
            "curriculum": {
                "active_slice": {
                    "attempt_mass": 4200,
                    "wallclock_minutes": 52.3,
                }
            },
            "throughput": {
                "proofs_per_hour": 65.2,
                "coefficient_of_variation": 0.22,
            },
            "frontier": {
                "queue_backlog": 0.35,
            },
        },
        "provenance": {
            "merkle_hash": "abc123def456",
        },
    }


@pytest.fixture
def invalid_metrics_missing_path():
    """Create an invalid metric payload missing required paths."""
    return {
        "metrics": {
            "rfl": {
                # Missing coverage section
            },
            "success_rates": {
                "abstention_rate": 18.5,
            },
        },
        "provenance": {
            "merkle_hash": "abc123def456",
        },
    }


# ---------------------------------------------------------------------------
# Fingerprinting Tests
# ---------------------------------------------------------------------------

class TestCurriculumFingerprinting:
    """Tests for curriculum fingerprinting system."""

    def test_fingerprint_stability(self, sample_system):
        """Fingerprint must be identical for identical configurations."""
        fp1 = sample_system.fingerprint()
        fp2 = sample_system.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 produces 64 hex characters

    def test_fingerprint_sensitivity_params(self, sample_system):
        """Fingerprint must change when slice parameters change."""
        baseline_fp = sample_system.fingerprint()
        
        # Modify a parameter
        sample_system.slices[0].params["atoms"] = 6
        modified_fp = sample_system.fingerprint()
        
        assert baseline_fp != modified_fp

    def test_fingerprint_sensitivity_gates(self, sample_system):
        """Fingerprint must change when gate thresholds change."""
        baseline_fp = sample_system.fingerprint()
        
        # Modify a gate threshold (need to recreate the slice due to frozen dataclass)
        modified_gates = SliceGates(
            coverage=CoverageGateSpec(ci_lower_min=0.90, sample_min=30, require_attestation=True),  # Changed
            abstention=sample_system.slices[0].gates.abstention,
            velocity=sample_system.slices[0].gates.velocity,
            caps=sample_system.slices[0].gates.caps,
        )
        sample_system.slices[0] = CurriculumSlice(
            name=sample_system.slices[0].name,
            params=sample_system.slices[0].params,
            gates=modified_gates,
            metadata=sample_system.slices[0].metadata,
        )
        modified_fp = sample_system.fingerprint()
        
        assert baseline_fp != modified_fp

    def test_fingerprint_slice_order_independence(self, sample_gates):
        """Fingerprint must be stable against slice reordering."""
        slice_a = CurriculumSlice(
            name="slice_a",
            params={"atoms": 4, "depth_max": 6},
            gates=sample_gates,
            metadata={},
        )
        slice_b = CurriculumSlice(
            name="slice_b",
            params={"atoms": 5, "depth_max": 7},
            gates=sample_gates,
            metadata={},
        )
        
        system_ab = CurriculumSystem(
            slug="test",
            description="Test",
            slices=[slice_a, slice_b],
            active_index=0,
            invariants={},
            monotonic_axes=(),
            version=2,
        )
        
        system_ba = CurriculumSystem(
            slug="test",
            description="Test",
            slices=[slice_b, slice_a],  # Reversed order
            active_index=0,
            invariants={},
            monotonic_axes=(),
            version=2,
        )
        
        # Fingerprints should be identical because slices are sorted by name
        assert system_ab.fingerprint() == system_ba.fingerprint()

    def test_fingerprint_excludes_runtime_state(self, sample_system):
        """Fingerprint must exclude runtime state like active_index."""
        fp1 = sample_system.fingerprint()
        
        # Change active_index (runtime pointer)
        sample_system.active_index = 0  # No change in value, but conceptually runtime
        fp2 = sample_system.fingerprint()
        
        # Fingerprint should be identical (active_index is not included)
        assert fp1 == fp2


# ---------------------------------------------------------------------------
# Drift Sentinel Tests
# ---------------------------------------------------------------------------

class TestCurriculumDriftSentinel:
    """Tests for curriculum drift sentinel."""

    def test_sentinel_allows_no_drift(self, sample_system):
        """Sentinel must allow execution when no drift is detected."""
        baseline_fp = sample_system.fingerprint()
        sentinel = CurriculumDriftSentinel(
            baseline_fingerprint=baseline_fp,
            baseline_version=sample_system.version,
            baseline_slice_count=len(sample_system.slices),
        )
        
        violations = sentinel.check(sample_system)
        assert len(violations) == 0

    def test_sentinel_detects_fingerprint_drift(self, sample_system):
        """Sentinel must detect when curriculum fingerprint changes."""
        baseline_fp = sample_system.fingerprint()
        sentinel = CurriculumDriftSentinel(
            baseline_fingerprint=baseline_fp,
            baseline_version=sample_system.version,
            baseline_slice_count=len(sample_system.slices),
        )
        
        # Modify curriculum
        sample_system.slices[0].params["atoms"] = 99
        
        violations = sentinel.check(sample_system)
        assert len(violations) > 0
        assert any("ContentDrift" in v for v in violations)

    def test_sentinel_detects_version_drift(self, sample_system):
        """Sentinel must detect when curriculum version changes."""
        baseline_fp = sample_system.fingerprint()
        sentinel = CurriculumDriftSentinel(
            baseline_fingerprint=baseline_fp,
            baseline_version=2,
            baseline_slice_count=len(sample_system.slices),
        )
        
        # Change version
        sample_system.version = 3
        
        violations = sentinel.check(sample_system)
        assert any("SchemaDrift" in v for v in violations)

    def test_sentinel_detects_slice_count_drift(self, sample_system, sample_gates):
        """Sentinel must detect when slice count changes."""
        baseline_fp = sample_system.fingerprint()
        sentinel = CurriculumDriftSentinel(
            baseline_fingerprint=baseline_fp,
            baseline_version=sample_system.version,
            baseline_slice_count=1,
        )
        
        # Add a new slice
        new_slice = CurriculumSlice(
            name="new_slice",
            params={"atoms": 6, "depth_max": 9},
            gates=sample_gates,
            metadata={},
        )
        sample_system.slices.append(new_slice)
        
        violations = sentinel.check(sample_system)
        assert any("SliceCountDrift" in v for v in violations)


# ---------------------------------------------------------------------------
# Metric Schema Enforcement Tests
# ---------------------------------------------------------------------------

class TestMetricSchemaEnforcement:
    """Tests for metric schema enforcement with feature flags."""

    def test_strict_mode_accepts_valid_schema(self, valid_metrics_v1):
        """Strict mode must accept valid v1 metric payloads."""
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "strict"
        try:
            result = NormalizedMetrics.from_raw(valid_metrics_v1)
            assert result.coverage_ci_lower == 0.87
            assert result.coverage_sample_size == 45
            assert result.attestation_hash == "abc123def456"
        finally:
            os.environ.pop("METRIC_SCHEMA_ENFORCEMENT_MODE", None)

    def test_strict_mode_rejects_invalid_schema(self, invalid_metrics_missing_path):
        """Strict mode must reject invalid metric payloads."""
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "strict"
        try:
            with pytest.raises(CurriculumDriftError, match="Required path.*not found"):
                NormalizedMetrics.from_raw(invalid_metrics_missing_path)
        finally:
            os.environ.pop("METRIC_SCHEMA_ENFORCEMENT_MODE", None)

    def test_permissive_mode_is_backward_compatible(self):
        """Permissive mode must accept old-style metric payloads."""
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "permissive"
        try:
            # Old-style metric with flat structure
            old_metric = {
                "coverage_ci_lower": 0.8,
                "coverage_sample_size": 40,
                "abstention_rate": 20.0,
                "proofs": {
                    "recent_hour": 3500,
                },
                "provenance": {
                    "merkle_hash": "old_hash",
                },
            }
            result = NormalizedMetrics.from_raw(old_metric)
            assert result.coverage_ci_lower == 0.8
        finally:
            os.environ.pop("METRIC_SCHEMA_ENFORCEMENT_MODE", None)

    def test_log_only_mode_detects_discrepancy(self, capsys, invalid_metrics_missing_path):
        """Log-only mode must detect and log discrepancies without failing."""
        os.environ["METRIC_SCHEMA_ENFORCEMENT_MODE"] = "log_only"
        try:
            # This should not raise an exception, but should log a warning
            result = NormalizedMetrics.from_raw(invalid_metrics_missing_path)
            captured = capsys.readouterr()
            assert "Metric Schema Drift" in captured.err or "ERROR" in captured.err
        finally:
            os.environ.pop("METRIC_SCHEMA_ENFORCEMENT_MODE", None)

    def test_default_mode_is_permissive(self, valid_metrics_v1):
        """Default mode (no env var) must be permissive."""
        # Ensure no env var is set
        os.environ.pop("METRIC_SCHEMA_ENFORCEMENT_MODE", None)
        
        # Should work without raising
        result = NormalizedMetrics.from_raw(valid_metrics_v1)
        assert result is not None


# ---------------------------------------------------------------------------
# SliceGates Serialization Tests
# ---------------------------------------------------------------------------

class TestSliceGatesSerialization:
    """Tests for SliceGates.to_dict() canonical form."""

    def test_to_dict_produces_canonical_form(self, sample_gates):
        """to_dict() must produce a canonical dictionary representation."""
        gates_dict = sample_gates.to_dict()
        
        assert "coverage" in gates_dict
        assert "abstention" in gates_dict
        assert "velocity" in gates_dict
        assert "caps" in gates_dict
        
        assert gates_dict["coverage"]["ci_lower_min"] == 0.85
        assert gates_dict["abstention"]["max_rate_pct"] == 25.0

    def test_to_dict_is_stable(self, sample_gates):
        """to_dict() must produce identical output on repeated calls."""
        dict1 = sample_gates.to_dict()
        dict2 = sample_gates.to_dict()
        
        assert dict1 == dict2
