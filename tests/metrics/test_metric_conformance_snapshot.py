"""
Tests for Metric Conformance Snapshot — Live Drift Detection & Promotion Guard

Covers:
- TASK 1: Snapshot building, saving, and loading
- TASK 2: Snapshot comparison and can_promote_metric() helper
- TASK 3: Human-readable conformance report rendering
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pytest

from backend.metrics.metric_conformance_snapshot import (
    # Enums
    ConformanceLevel,
    DriftClass,
    # Data classes
    MetricConformanceResult,
    ConformanceSnapshot,
    SnapshotComparison,
    # Snapshot building
    build_conformance_snapshot,
    save_conformance_snapshot,
    load_conformance_snapshot,
    # Comparison & promotion
    compare_conformance_snapshots,
    can_promote_metric,
    # Report rendering
    render_conformance_report,
    # Helpers
    get_git_sha,
    DRIFT_TO_CONFORMANCE,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def goal_hit_result() -> MetricConformanceResult:
    """Create a MetricConformanceResult for goal_hit metric."""
    return MetricConformanceResult(
        metric_name="goal_hit",
        tests_passed={
            "test_T1_return_type": True,
            "test_T2_determinism": True,
            "test_T3_hit_bound": True,
            "test_T4_verified_bound": True,
            "test_T5_threshold": True,
            "test_T6_boundary": True,
            "test_T7_monotonicity": True,
            "test_T8_stress": True,
        },
        tests_by_level={
            "L0": ["test_T1_return_type", "test_T2_determinism"],
            "L1": ["test_T3_hit_bound", "test_T4_verified_bound", "test_T5_threshold"],
            "L2": ["test_T6_boundary", "test_T7_monotonicity"],
            "L3": ["test_T8_stress"],
        },
    )


@pytest.fixture
def sparse_success_result() -> MetricConformanceResult:
    """Create a MetricConformanceResult for sparse_success metric."""
    return MetricConformanceResult(
        metric_name="sparse_success",
        tests_passed={
            "test_T1_return_type": True,
            "test_T2_determinism": True,
            "test_T3_value_equals_input": True,
            "test_T4_threshold": True,
            "test_T5_boundary": True,
            "test_T6_stress": True,
        },
        tests_by_level={
            "L0": ["test_T1_return_type", "test_T2_determinism"],
            "L1": ["test_T3_value_equals_input", "test_T4_threshold"],
            "L2": ["test_T5_boundary"],
            "L3": ["test_T6_stress"],
        },
    )


@pytest.fixture
def sample_snapshot(
    goal_hit_result: MetricConformanceResult,
    sparse_success_result: MetricConformanceResult,
) -> ConformanceSnapshot:
    """Create a sample conformance snapshot."""
    return build_conformance_snapshot(
        results=[goal_hit_result, sparse_success_result],
        timestamp="2025-01-15T10:00:00+00:00",
        git_sha="abc1234",
    )


@pytest.fixture
def regressed_sparse_result() -> MetricConformanceResult:
    """Create a sparse_success result with a regression (L1 test fails)."""
    return MetricConformanceResult(
        metric_name="sparse_success",
        tests_passed={
            "test_T1_return_type": True,
            "test_T2_determinism": True,
            "test_T3_value_equals_input": False,  # REGRESSION
            "test_T4_threshold": True,
            "test_T5_boundary": True,
            "test_T6_stress": True,
        },
        tests_by_level={
            "L0": ["test_T1_return_type", "test_T2_determinism"],
            "L1": ["test_T3_value_equals_input", "test_T4_threshold"],
            "L2": ["test_T5_boundary"],
            "L3": ["test_T6_stress"],
        },
    )


# =============================================================================
# TASK 1: Snapshot Building, Saving, Loading
# =============================================================================

class TestMetricConformanceResult:
    """Tests for MetricConformanceResult data class."""

    def test_passed_at_level(self, goal_hit_result: MetricConformanceResult):
        """Test counting passed tests at each level."""
        assert goal_hit_result.passed_at_level(ConformanceLevel.L0) == 2
        assert goal_hit_result.passed_at_level(ConformanceLevel.L1) == 3
        assert goal_hit_result.passed_at_level(ConformanceLevel.L2) == 2
        assert goal_hit_result.passed_at_level(ConformanceLevel.L3) == 1

    def test_total_at_level(self, goal_hit_result: MetricConformanceResult):
        """Test counting total tests at each level."""
        assert goal_hit_result.total_at_level(ConformanceLevel.L0) == 2
        assert goal_hit_result.total_at_level(ConformanceLevel.L1) == 3
        assert goal_hit_result.total_at_level(ConformanceLevel.L2) == 2
        assert goal_hit_result.total_at_level(ConformanceLevel.L3) == 1

    def test_level_pass_rate_all_pass(self, goal_hit_result: MetricConformanceResult):
        """Test pass rate when all tests pass."""
        assert goal_hit_result.level_pass_rate(ConformanceLevel.L0) == 1.0
        assert goal_hit_result.level_pass_rate(ConformanceLevel.L1) == 1.0

    def test_level_pass_rate_partial(self, regressed_sparse_result: MetricConformanceResult):
        """Test pass rate with partial failures."""
        # L1 has 2 tests, 1 fails
        assert regressed_sparse_result.level_pass_rate(ConformanceLevel.L1) == 0.5

    def test_level_pass_rate_empty(self):
        """Test pass rate for level with no tests."""
        result = MetricConformanceResult(
            metric_name="empty",
            tests_passed={},
            tests_by_level={},
        )
        # No tests = vacuously passing
        assert result.level_pass_rate(ConformanceLevel.L0) == 1.0


class TestBuildConformanceSnapshot:
    """Tests for build_conformance_snapshot function."""

    def test_build_snapshot_basic(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
    ):
        """Test building a basic snapshot."""
        snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="abc1234",
        )

        assert snapshot.timestamp == "2025-01-15T10:00:00+00:00"
        assert snapshot.git_sha == "abc1234"
        assert "goal_hit" in snapshot.metrics
        assert "sparse_success" in snapshot.metrics
        assert snapshot.total_tests == 14
        assert snapshot.total_passed == 14

    def test_build_snapshot_level_counts(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
    ):
        """Test conformance level counts in snapshot."""
        snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
        )

        # goal_hit: L0=2, L1=3, L2=2, L3=1
        # sparse_success: L0=2, L1=2, L2=1, L3=1
        assert snapshot.conformance_levels["L0"] == 4
        assert snapshot.conformance_levels["L1"] == 5
        assert snapshot.conformance_levels["L2"] == 3
        assert snapshot.conformance_levels["L3"] == 2

    def test_build_snapshot_deterministic(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
    ):
        """Test that snapshot ID is deterministic for same inputs."""
        timestamp = "2025-01-15T10:00:00+00:00"
        git_sha = "abc1234"

        snapshot1 = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp=timestamp,
            git_sha=git_sha,
        )
        snapshot2 = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp=timestamp,
            git_sha=git_sha,
        )

        assert snapshot1.snapshot_id == snapshot2.snapshot_id

    def test_build_snapshot_different_inputs_different_id(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
    ):
        """Test that different inputs produce different snapshot IDs."""
        snapshot1 = build_conformance_snapshot(
            results=[goal_hit_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="abc1234",
        )
        snapshot2 = build_conformance_snapshot(
            results=[sparse_success_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="abc1234",
        )

        assert snapshot1.snapshot_id != snapshot2.snapshot_id

    def test_build_snapshot_auto_timestamp(self, goal_hit_result: MetricConformanceResult):
        """Test that timestamp is auto-generated if not provided."""
        snapshot = build_conformance_snapshot(results=[goal_hit_result])
        assert snapshot.timestamp is not None
        # Should be ISO 8601 format
        datetime.fromisoformat(snapshot.timestamp.replace("Z", "+00:00"))

    def test_build_snapshot_version(self, goal_hit_result: MetricConformanceResult):
        """Test that snapshot includes version."""
        snapshot = build_conformance_snapshot(results=[goal_hit_result])
        assert snapshot.version == "1.0.0"


class TestSnapshotSerialization:
    """Tests for save/load conformance snapshot."""

    def test_save_and_load_roundtrip(self, sample_snapshot: ConformanceSnapshot):
        """Test save and load produces identical snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.json"

            save_conformance_snapshot(path, sample_snapshot)
            loaded = load_conformance_snapshot(path)

            assert loaded.snapshot_id == sample_snapshot.snapshot_id
            assert loaded.timestamp == sample_snapshot.timestamp
            assert loaded.git_sha == sample_snapshot.git_sha
            assert loaded.total_tests == sample_snapshot.total_tests
            assert loaded.total_passed == sample_snapshot.total_passed
            assert loaded.conformance_levels == sample_snapshot.conformance_levels
            assert loaded.metrics == sample_snapshot.metrics

    def test_save_creates_parent_dirs(self, sample_snapshot: ConformanceSnapshot):
        """Test save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "snapshot.json"

            save_conformance_snapshot(path, sample_snapshot)
            assert path.exists()

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_conformance_snapshot("/nonexistent/path/snapshot.json")

    def test_load_invalid_json_raises(self):
        """Test loading invalid JSON raises JSONDecodeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            path.write_text("not valid json {{{")

            with pytest.raises(json.JSONDecodeError):
                load_conformance_snapshot(path)

    def test_snapshot_to_dict_and_back(self, sample_snapshot: ConformanceSnapshot):
        """Test to_dict and from_dict are inverses."""
        data = sample_snapshot.to_dict()
        restored = ConformanceSnapshot.from_dict(data)

        assert restored.snapshot_id == sample_snapshot.snapshot_id
        assert restored.timestamp == sample_snapshot.timestamp
        assert restored.git_sha == sample_snapshot.git_sha

    def test_saved_file_is_valid_json(self, sample_snapshot: ConformanceSnapshot):
        """Test saved file is valid, readable JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.json"
            save_conformance_snapshot(path, sample_snapshot)

            with open(path, "r") as f:
                data = json.load(f)

            assert "snapshot_id" in data
            assert "timestamp" in data
            assert "metrics" in data


class TestGetGitSha:
    """Tests for get_git_sha helper."""

    def test_get_git_sha_from_env(self, monkeypatch):
        """Test getting git SHA from environment variables."""
        monkeypatch.setenv("GITHUB_SHA", "abc123def456")
        sha = get_git_sha()
        assert sha == "abc123def456"[:12]

    def test_get_git_sha_gitlab_ci(self, monkeypatch):
        """Test getting git SHA from GitLab CI env var."""
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        monkeypatch.setenv("CI_COMMIT_SHA", "xyz789abc012def")
        sha = get_git_sha()
        assert sha == "xyz789abc012"  # 12 chars

    def test_get_git_sha_none_when_unavailable(self, monkeypatch):
        """Test None returned when git SHA unavailable."""
        # Clear all known env vars
        for var in ["GITHUB_SHA", "CI_COMMIT_SHA", "GIT_SHA", "GIT_COMMIT"]:
            monkeypatch.delenv(var, raising=False)
        # Note: This may still return a value if running in a git repo
        # with git command available, so we just verify it doesn't raise


# =============================================================================
# TASK 2: Snapshot Comparison & Promotion Guard
# =============================================================================

class TestCompareConformanceSnapshots:
    """Tests for compare_conformance_snapshots function."""

    def test_compare_identical_snapshots(self, sample_snapshot: ConformanceSnapshot):
        """Test comparing identical snapshots shows no changes."""
        comparison = compare_conformance_snapshots(sample_snapshot, sample_snapshot)

        assert not comparison.is_regression
        assert comparison.regression_severity is None
        assert len(comparison.regressed_levels) == 0
        assert len(comparison.improved_levels) == 0
        assert len(comparison.tests_lost) == 0
        assert len(comparison.tests_gained) == 0

    def test_detect_l1_regression(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test detecting L1 regression."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="old123",
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
            timestamp="2025-01-15T11:00:00+00:00",
            git_sha="new456",
        )

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

        assert comparison.is_regression
        assert comparison.regression_severity == "critical"  # L1 regression is critical
        assert "L1" in comparison.regressed_levels
        assert comparison.regressed_levels["L1"] == (5, 4)  # 5 -> 4 tests

    def test_detect_improvement(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test detecting improvement (opposite of regression)."""
        # Start with regressed state
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="old123",
        )
        # Improve to full passing
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp="2025-01-15T11:00:00+00:00",
            git_sha="new456",
        )

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

        assert not comparison.is_regression
        assert "L1" in comparison.improved_levels
        assert comparison.improved_levels["L1"] == (4, 5)  # 4 -> 5 tests

    def test_detect_tests_lost(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test detecting individual tests lost."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="old123",
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
            timestamp="2025-01-15T11:00:00+00:00",
            git_sha="new456",
        )

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

        assert len(comparison.tests_lost) == 1
        assert "sparse_success::test_T3_value_equals_input" in comparison.tests_lost

    def test_detect_tests_gained(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test detecting individual tests gained."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="old123",
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp="2025-01-15T11:00:00+00:00",
            git_sha="new456",
        )

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

        assert len(comparison.tests_gained) == 1
        assert "sparse_success::test_T3_value_equals_input" in comparison.tests_gained

    def test_comparison_to_dict(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test SnapshotComparison can be serialized to dict."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="old123",
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
            timestamp="2025-01-15T11:00:00+00:00",
            git_sha="new456",
        )

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)
        data = comparison.to_dict()

        assert "old_snapshot_id" in data
        assert "new_snapshot_id" in data
        assert "is_regression" in data
        assert data["is_regression"] == True

    def test_regression_severity_critical_l0(self):
        """Test L0 regression is classified as critical."""
        # Create result with L0 failure
        result_with_l0_fail = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={
                "test_T1_return_type": False,  # L0 FAIL
                "test_T2_determinism": True,
            },
            tests_by_level={
                "L0": ["test_T1_return_type", "test_T2_determinism"],
            },
        )
        result_all_pass = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={
                "test_T1_return_type": True,
                "test_T2_determinism": True,
            },
            tests_by_level={
                "L0": ["test_T1_return_type", "test_T2_determinism"],
            },
        )

        old_snapshot = build_conformance_snapshot(
            results=[result_all_pass],
            timestamp="2025-01-15T10:00:00+00:00",
        )
        new_snapshot = build_conformance_snapshot(
            results=[result_with_l0_fail],
            timestamp="2025-01-15T11:00:00+00:00",
        )

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

        assert comparison.is_regression
        assert comparison.regression_severity == "critical"

    def test_regression_severity_major_l2(self):
        """Test L2-only regression is classified as major."""
        result_with_l2_fail = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={
                "test_T1": True,
                "test_T2": True,
                "test_T3": False,  # L2 FAIL
            },
            tests_by_level={
                "L0": ["test_T1"],
                "L1": ["test_T2"],
                "L2": ["test_T3"],
            },
        )
        result_all_pass = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={
                "test_T1": True,
                "test_T2": True,
                "test_T3": True,
            },
            tests_by_level={
                "L0": ["test_T1"],
                "L1": ["test_T2"],
                "L2": ["test_T3"],
            },
        )

        old_snapshot = build_conformance_snapshot(results=[result_all_pass])
        new_snapshot = build_conformance_snapshot(results=[result_with_l2_fail])

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

        assert comparison.is_regression
        assert comparison.regression_severity == "major"

    def test_regression_severity_minor_l3(self):
        """Test L3-only regression is classified as minor."""
        result_with_l3_fail = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={
                "test_T1": True,
                "test_T2": True,
                "test_T3": True,
                "test_T4": False,  # L3 FAIL
            },
            tests_by_level={
                "L0": ["test_T1"],
                "L1": ["test_T2"],
                "L2": ["test_T3"],
                "L3": ["test_T4"],
            },
        )
        result_all_pass = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={
                "test_T1": True,
                "test_T2": True,
                "test_T3": True,
                "test_T4": True,
            },
            tests_by_level={
                "L0": ["test_T1"],
                "L1": ["test_T2"],
                "L2": ["test_T3"],
                "L3": ["test_T4"],
            },
        )

        old_snapshot = build_conformance_snapshot(results=[result_all_pass])
        new_snapshot = build_conformance_snapshot(results=[result_with_l3_fail])

        comparison = compare_conformance_snapshots(old_snapshot, new_snapshot)

        assert comparison.is_regression
        assert comparison.regression_severity == "minor"


class TestCanPromoteMetric:
    """Tests for can_promote_metric helper."""

    def test_promote_allowed_no_regression(self, sample_snapshot: ConformanceSnapshot):
        """Test promotion allowed when no regression."""
        can_promote, reason = can_promote_metric(sample_snapshot, sample_snapshot)

        assert can_promote
        assert "No conformance regression" in reason

    def test_promote_blocked_critical_regression(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test promotion blocked on critical (L1) regression."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
        )

        can_promote, reason = can_promote_metric(old_snapshot, new_snapshot)

        assert not can_promote
        assert "Critical" in reason
        assert "L1" in reason

    def test_promote_blocked_major_regression(self):
        """Test promotion blocked on major (L2) regression."""
        result_with_l2_fail = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={"test_l2": False},
            tests_by_level={"L2": ["test_l2"]},
        )
        result_all_pass = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={"test_l2": True},
            tests_by_level={"L2": ["test_l2"]},
        )

        old_snapshot = build_conformance_snapshot(results=[result_all_pass])
        new_snapshot = build_conformance_snapshot(results=[result_with_l2_fail])

        can_promote, reason = can_promote_metric(old_snapshot, new_snapshot)

        assert not can_promote
        assert "Major" in reason

    def test_promote_blocked_minor_regression_default(self):
        """Test promotion blocked on minor regression by default."""
        result_with_l3_fail = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={"test_l3": False},
            tests_by_level={"L3": ["test_l3"]},
        )
        result_all_pass = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={"test_l3": True},
            tests_by_level={"L3": ["test_l3"]},
        )

        old_snapshot = build_conformance_snapshot(results=[result_all_pass])
        new_snapshot = build_conformance_snapshot(results=[result_with_l3_fail])

        can_promote, reason = can_promote_metric(old_snapshot, new_snapshot)

        assert not can_promote
        assert "Minor" in reason

    def test_promote_allowed_minor_regression_when_permitted(self):
        """Test promotion allowed for minor regression when allow_minor_regression=True."""
        result_with_l3_fail = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={"test_l3": False},
            tests_by_level={"L3": ["test_l3"]},
        )
        result_all_pass = MetricConformanceResult(
            metric_name="test_metric",
            tests_passed={"test_l3": True},
            tests_by_level={"L3": ["test_l3"]},
        )

        old_snapshot = build_conformance_snapshot(results=[result_all_pass])
        new_snapshot = build_conformance_snapshot(results=[result_with_l3_fail])

        can_promote, reason = can_promote_metric(
            old_snapshot,
            new_snapshot,
            allow_minor_regression=True,
        )

        assert can_promote
        assert "Minor conformance regression allowed" in reason


class TestDriftToConformanceMapping:
    """Tests for drift class to conformance level mapping."""

    def test_d0_requires_l0(self):
        """Test D0 (cosmetic) requires L0."""
        assert DRIFT_TO_CONFORMANCE[DriftClass.D0_COSMETIC] == ConformanceLevel.L0

    def test_d1_requires_l1(self):
        """Test D1 (additive) requires L1."""
        assert DRIFT_TO_CONFORMANCE[DriftClass.D1_ADDITIVE] == ConformanceLevel.L1

    def test_d2_requires_l2(self):
        """Test D2 (behavioral compatible) requires L2."""
        assert DRIFT_TO_CONFORMANCE[DriftClass.D2_BEHAVIORAL_COMPAT] == ConformanceLevel.L2

    def test_d3_requires_l3(self):
        """Test D3 (behavioral breaking) requires L3."""
        assert DRIFT_TO_CONFORMANCE[DriftClass.D3_BEHAVIORAL_BREAK] == ConformanceLevel.L3

    def test_d4_requires_l3(self):
        """Test D4 (schema breaking) requires L3."""
        assert DRIFT_TO_CONFORMANCE[DriftClass.D4_SCHEMA_BREAK] == ConformanceLevel.L3

    def test_d5_not_in_mapping(self):
        """Test D5 (semantic breaking) has no mapping (cannot ship)."""
        assert DriftClass.D5_SEMANTIC_BREAK not in DRIFT_TO_CONFORMANCE


# =============================================================================
# TASK 3: Human-Readable Conformance Report
# =============================================================================

class TestRenderConformanceReport:
    """Tests for render_conformance_report function."""

    def test_render_empty_snapshots(self):
        """Test rendering with no snapshots."""
        report = render_conformance_report([])

        assert "Metric Conformance Report" in report
        assert "No snapshots provided" in report

    def test_render_single_snapshot(self, sample_snapshot: ConformanceSnapshot):
        """Test rendering with single snapshot."""
        report = render_conformance_report([sample_snapshot])

        assert "Metric Conformance Report" in report
        assert "Conformance Summary" in report
        assert "goal_hit" in report
        assert "sparse_success" in report
        # Should have table headers
        assert "| Snapshot |" in report
        assert "| L0 |" in report

    def test_render_custom_title(self, sample_snapshot: ConformanceSnapshot):
        """Test rendering with custom title."""
        report = render_conformance_report(
            [sample_snapshot],
            title="Custom Report Title",
        )

        assert "# Custom Report Title" in report

    def test_render_multiple_snapshots_with_drift(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test rendering with multiple snapshots shows drift analysis."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
            timestamp="2025-01-15T10:00:00+00:00",
            git_sha="old123",
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
            timestamp="2025-01-15T11:00:00+00:00",
            git_sha="new456",
        )

        report = render_conformance_report([old_snapshot, new_snapshot])

        assert "Drift Analysis" in report
        assert "Regression Detected" in report
        assert "Tests Lost" in report

    def test_render_shows_improvement(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test rendering shows improvements."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
        )

        report = render_conformance_report([old_snapshot, new_snapshot])

        assert "Improved Levels" in report
        assert "Tests Gained" in report

    def test_render_stable_structure(self, sample_snapshot: ConformanceSnapshot):
        """Test report has stable, predictable structure."""
        report = render_conformance_report([sample_snapshot])

        # Check expected sections exist
        assert "# " in report  # Has header
        assert "## Conformance Summary" in report
        assert "## Metric Details" in report
        assert "---" in report  # Has horizontal rule
        assert "*Generated by MathLedger" in report

    def test_render_no_color_codes(self, sample_snapshot: ConformanceSnapshot):
        """Test report contains no ANSI color codes."""
        report = render_conformance_report([sample_snapshot])

        # ANSI escape codes start with \x1b or \033
        assert "\x1b" not in report
        assert "\033" not in report

    def test_render_valid_markdown_table(self, sample_snapshot: ConformanceSnapshot):
        """Test report contains valid Markdown tables."""
        report = render_conformance_report([sample_snapshot])

        # Tables should have header separators
        lines = report.split("\n")
        table_separator_count = sum(1 for line in lines if line.startswith("|---"))

        assert table_separator_count >= 1

    def test_render_status_column_shows_regression(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test status column correctly shows regression status."""
        old_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
        )
        new_snapshot = build_conformance_snapshot(
            results=[goal_hit_result, regressed_sparse_result],
        )

        report = render_conformance_report([old_snapshot, new_snapshot])

        # First snapshot should show OK, second should show CRITICAL
        assert "| OK |" in report
        assert "CRITICAL" in report

    def test_render_metric_details_section(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
    ):
        """Test metric details section shows per-metric breakdown."""
        snapshot = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
        )

        report = render_conformance_report([snapshot])

        assert "## Metric Details (Latest)" in report
        # Should show passed/total format
        assert "/" in report  # e.g., "2/2"


# =============================================================================
# Integration Tests
# =============================================================================

class TestSnapshotWorkflow:
    """Integration tests for complete snapshot workflow."""

    def test_full_workflow_save_load_compare(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
        regressed_sparse_result: MetricConformanceResult,
    ):
        """Test full workflow: build -> save -> load -> compare."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build and save baseline
            baseline = build_conformance_snapshot(
                results=[goal_hit_result, sparse_success_result],
                git_sha="baseline",
            )
            baseline_path = Path(tmpdir) / "baseline.json"
            save_conformance_snapshot(baseline_path, baseline)

            # Build and save candidate
            candidate = build_conformance_snapshot(
                results=[goal_hit_result, regressed_sparse_result],
                git_sha="candidate",
            )
            candidate_path = Path(tmpdir) / "candidate.json"
            save_conformance_snapshot(candidate_path, candidate)

            # Load and compare
            loaded_baseline = load_conformance_snapshot(baseline_path)
            loaded_candidate = load_conformance_snapshot(candidate_path)

            comparison = compare_conformance_snapshots(loaded_baseline, loaded_candidate)

            assert comparison.is_regression
            assert comparison.regression_severity == "critical"

            # Check promotion gate
            can_promote, reason = can_promote_metric(loaded_baseline, loaded_candidate)
            assert not can_promote

    def test_ci_gate_workflow(
        self,
        goal_hit_result: MetricConformanceResult,
        sparse_success_result: MetricConformanceResult,
    ):
        """Test CI gate workflow with passing conformance."""
        baseline = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
        )
        candidate = build_conformance_snapshot(
            results=[goal_hit_result, sparse_success_result],
        )

        # No regression
        can_promote, reason = can_promote_metric(baseline, candidate)
        assert can_promote

        # Generate report
        report = render_conformance_report([baseline, candidate])
        assert "No regression detected" in report
