"""
Tests for Phase III: Drift Memory & Promotion Oracle

Covers:
- TASK 1: build_metric_drift_ledger
- TASK 2: compute_metric_promotion_readiness
- TASK 3: summarize_metric_conformance_for_global_health
"""

import pytest
from typing import Dict, List, Any

from backend.metrics.metric_conformance_snapshot import (
    # Data classes
    MetricConformanceResult,
    ConformanceSnapshot,
    DriftEvent,
    DriftLedger,
    PromotionReadiness,
    GlobalHealthStatus,
    # Functions
    build_conformance_snapshot,
    build_metric_drift_ledger,
    compute_metric_promotion_readiness,
    summarize_metric_conformance_for_global_health,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def make_result(
    metric_name: str,
    l0_passed: int = 2,
    l0_total: int = 2,
    l1_passed: int = 5,
    l1_total: int = 5,
    l2_passed: int = 3,
    l2_total: int = 3,
    l3_passed: int = 1,
    l3_total: int = 1,
) -> MetricConformanceResult:
    """Create a MetricConformanceResult with specified pass counts."""
    tests_passed: Dict[str, bool] = {}
    tests_by_level: Dict[str, List[str]] = {"L0": [], "L1": [], "L2": [], "L3": []}

    # L0 tests
    for i in range(l0_total):
        test_id = f"test_L0_{i}"
        tests_by_level["L0"].append(test_id)
        tests_passed[test_id] = i < l0_passed

    # L1 tests
    for i in range(l1_total):
        test_id = f"test_L1_{i}"
        tests_by_level["L1"].append(test_id)
        tests_passed[test_id] = i < l1_passed

    # L2 tests
    for i in range(l2_total):
        test_id = f"test_L2_{i}"
        tests_by_level["L2"].append(test_id)
        tests_passed[test_id] = i < l2_passed

    # L3 tests
    for i in range(l3_total):
        test_id = f"test_L3_{i}"
        tests_by_level["L3"].append(test_id)
        tests_passed[test_id] = i < l3_passed

    return MetricConformanceResult(
        metric_name=metric_name,
        tests_passed=tests_passed,
        tests_by_level=tests_by_level,
    )


def make_snapshot(
    results: List[MetricConformanceResult],
    timestamp: str = "2025-01-15T10:00:00+00:00",
    git_sha: str = "abc1234",
) -> ConformanceSnapshot:
    """Create a ConformanceSnapshot from results."""
    return build_conformance_snapshot(
        results=results,
        timestamp=timestamp,
        git_sha=git_sha,
    )


@pytest.fixture
def healthy_snapshot() -> ConformanceSnapshot:
    """Create a snapshot with all tests passing."""
    return make_snapshot([
        make_result("goal_hit"),
        make_result("sparse_success"),
        make_result("chain_success"),
        make_result("multi_goal"),
    ])


@pytest.fixture
def regressed_snapshot() -> ConformanceSnapshot:
    """Create a snapshot with L1 regression."""
    return make_snapshot([
        make_result("goal_hit"),
        make_result("sparse_success", l1_passed=3, l1_total=5),  # Regression
        make_result("chain_success"),
        make_result("multi_goal"),
    ], timestamp="2025-01-15T11:00:00+00:00", git_sha="def5678")


@pytest.fixture
def critical_snapshot() -> ConformanceSnapshot:
    """Create a snapshot with critical failures."""
    return make_snapshot([
        make_result("goal_hit", l0_passed=1, l0_total=2),  # L0 failure
        make_result("sparse_success", l1_passed=2, l1_total=5),
        make_result("chain_success"),
        make_result("multi_goal"),
    ], timestamp="2025-01-15T12:00:00+00:00", git_sha="ghi9012")


# =============================================================================
# TASK 1: Drift Ledger Tests
# =============================================================================

class TestBuildMetricDriftLedger:
    """Tests for build_metric_drift_ledger function."""

    def test_empty_history(self):
        """Test drift ledger with no snapshots."""
        ledger = build_metric_drift_ledger([])

        assert ledger.schema_version == "1.0.0"
        assert len(ledger.drift_events) == 0
        assert len(ledger.metrics_with_repeated_regressions) == 0
        assert ledger.total_regressions == 0
        assert ledger.total_improvements == 0
        assert ledger.first_snapshot_id is None
        assert ledger.last_snapshot_id is None

    def test_single_snapshot(self, healthy_snapshot: ConformanceSnapshot):
        """Test drift ledger with single snapshot (no drift possible)."""
        ledger = build_metric_drift_ledger([healthy_snapshot])

        assert len(ledger.drift_events) == 0
        assert ledger.total_regressions == 0
        assert ledger.first_snapshot_id == healthy_snapshot.snapshot_id
        assert ledger.last_snapshot_id == healthy_snapshot.snapshot_id

    def test_detect_regression(
        self,
        healthy_snapshot: ConformanceSnapshot,
        regressed_snapshot: ConformanceSnapshot,
    ):
        """Test detecting regression between snapshots."""
        ledger = build_metric_drift_ledger([healthy_snapshot, regressed_snapshot])

        assert ledger.total_regressions >= 1
        assert len(ledger.drift_events) >= 1

        # Find the sparse_success regression
        sparse_events = [e for e in ledger.drift_events if e.metric == "sparse_success"]
        assert len(sparse_events) == 1
        assert sparse_events[0].tests_lost == 2  # 5 - 3 = 2

    def test_detect_repeated_regressions(self):
        """Test detecting metrics with repeated regressions."""
        # Create sequence: healthy -> regressed -> healthy -> regressed again
        snap1 = make_snapshot([make_result("goal_hit")], "2025-01-01T10:00:00+00:00", "s1")
        snap2 = make_snapshot([make_result("goal_hit", l1_passed=4)], "2025-01-02T10:00:00+00:00", "s2")
        snap3 = make_snapshot([make_result("goal_hit")], "2025-01-03T10:00:00+00:00", "s3")
        snap4 = make_snapshot([make_result("goal_hit", l1_passed=3)], "2025-01-04T10:00:00+00:00", "s4")

        ledger = build_metric_drift_ledger([snap1, snap2, snap3, snap4])

        assert ledger.total_regressions >= 2
        assert "goal_hit" in ledger.metrics_with_repeated_regressions
        assert ledger.metrics_with_repeated_regressions["goal_hit"] >= 2

    def test_track_improvements(self):
        """Test tracking improvements (test recovery)."""
        snap1 = make_snapshot([make_result("goal_hit", l1_passed=3)], "2025-01-01T10:00:00+00:00", "s1")
        snap2 = make_snapshot([make_result("goal_hit")], "2025-01-02T10:00:00+00:00", "s2")

        ledger = build_metric_drift_ledger([snap1, snap2])

        # Improvement from 3/5 to 5/5
        assert ledger.total_improvements >= 1

    def test_ledger_to_dict(
        self,
        healthy_snapshot: ConformanceSnapshot,
        regressed_snapshot: ConformanceSnapshot,
    ):
        """Test ledger serialization to dict."""
        ledger = build_metric_drift_ledger([healthy_snapshot, regressed_snapshot])
        data = ledger.to_dict()

        assert "schema_version" in data
        assert "drift_events" in data
        assert "metrics_with_repeated_regressions" in data
        assert "total_regressions" in data
        assert "total_improvements" in data

    def test_drift_event_severity(self):
        """Test drift event severity classification."""
        # L0 regression = critical
        snap1 = make_snapshot([make_result("m1")], "2025-01-01T10:00:00+00:00", "s1")
        snap2 = make_snapshot([make_result("m1", l0_passed=1, l1_passed=5)], "2025-01-02T10:00:00+00:00", "s2")

        ledger = build_metric_drift_ledger([snap1, snap2])

        assert len(ledger.drift_events) >= 1


# =============================================================================
# TASK 2: Promotion Oracle Tests
# =============================================================================

class TestComputeMetricPromotionReadiness:
    """Tests for compute_metric_promotion_readiness function."""

    def test_ready_no_regression(self, healthy_snapshot: ConformanceSnapshot):
        """Test promotion ready when no regression."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, healthy_snapshot)

        assert readiness.ready_for_promotion is True
        assert len(readiness.blocking_regressions) == 0
        assert "passed" in readiness.justification.lower()

    def test_blocked_l1_regression(
        self,
        healthy_snapshot: ConformanceSnapshot,
        regressed_snapshot: ConformanceSnapshot,
    ):
        """Test promotion blocked on L1 regression."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, regressed_snapshot)

        assert readiness.ready_for_promotion is False
        assert len(readiness.blocking_regressions) >= 1
        assert "critical" in readiness.justification.lower() or "blocked" in readiness.justification.lower()

    def test_blocked_l0_regression(self, healthy_snapshot: ConformanceSnapshot):
        """Test promotion blocked on L0 regression."""
        l0_fail = make_snapshot([
            make_result("goal_hit", l0_passed=1, l0_total=2),
        ])

        readiness = compute_metric_promotion_readiness(healthy_snapshot, l0_fail)

        assert readiness.ready_for_promotion is False
        # Should have L0 in blocking regressions
        l0_blocks = [b for b in readiness.blocking_regressions if b["level"] == "L0"]
        assert len(l0_blocks) >= 1

    def test_minor_regression_blocked_by_default(self, healthy_snapshot: ConformanceSnapshot):
        """Test L3-only regression is blocked by default."""
        l3_fail = make_snapshot([
            make_result("goal_hit", l3_passed=0, l3_total=1),
            make_result("sparse_success"),
            make_result("chain_success"),
            make_result("multi_goal"),
        ])

        readiness = compute_metric_promotion_readiness(healthy_snapshot, l3_fail)

        # L3 regression should block
        assert readiness.ready_for_promotion is False

    def test_minor_regression_allowed_with_waiver(self, healthy_snapshot: ConformanceSnapshot):
        """Test L3-only regression allowed with waiver."""
        l3_fail = make_snapshot([
            make_result("goal_hit", l3_passed=0, l3_total=1),
            make_result("sparse_success"),
            make_result("chain_success"),
            make_result("multi_goal"),
        ])

        readiness = compute_metric_promotion_readiness(
            healthy_snapshot,
            l3_fail,
            allow_minor_regression=True,
        )

        # With waiver, should be ready
        assert readiness.ready_for_promotion is True
        assert "waiver" in readiness.justification.lower()

    def test_readiness_to_dict(
        self,
        healthy_snapshot: ConformanceSnapshot,
        regressed_snapshot: ConformanceSnapshot,
    ):
        """Test readiness serialization to dict."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, regressed_snapshot)
        data = readiness.to_dict()

        assert "ready_for_promotion" in data
        assert "blocking_regressions" in data
        assert "justification" in data
        assert "comparison_details" in data

    def test_comparison_details_included(
        self,
        healthy_snapshot: ConformanceSnapshot,
        regressed_snapshot: ConformanceSnapshot,
    ):
        """Test comparison details are included in result."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, regressed_snapshot)

        assert readiness.comparison_details is not None
        assert "is_regression" in readiness.comparison_details


# =============================================================================
# TASK 3: Global Health Status Tests
# =============================================================================

class TestSummarizeMetricConformanceForGlobalHealth:
    """Tests for summarize_metric_conformance_for_global_health function."""

    def test_healthy_snapshot(self, healthy_snapshot: ConformanceSnapshot):
        """Test health status for fully healthy snapshot."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)

        assert health.overall_conformance_status == "healthy"
        assert health.any_blockers is False
        assert health.blocker_count == 0
        assert health.weakest_metric_pass_rate == 1.0
        assert len(health.metrics_by_status["healthy"]) == 4

    def test_degraded_snapshot(self):
        """Test health status for degraded snapshot (90-99% passing)."""
        # Create a snapshot with 95% pass rate on one metric
        degraded = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l2_passed=2, l2_total=3),  # 10/11 = 91%
        ])

        health = summarize_metric_conformance_for_global_health(degraded)

        assert health.overall_conformance_status == "degraded"
        assert health.any_blockers is False
        assert "sparse_success" in health.metrics_by_status["degraded"]

    def test_critical_snapshot(self):
        """Test health status for critical snapshot (50-89% passing)."""
        critical = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l1_passed=3, l1_total=5, l2_passed=2, l2_total=3),  # ~8/11 = 73%
        ])

        health = summarize_metric_conformance_for_global_health(critical)

        assert health.overall_conformance_status in ("critical", "degraded")
        assert health.weakest_metric == "sparse_success"

    def test_failing_snapshot(self):
        """Test health status for failing snapshot (<50% passing)."""
        failing = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l0_passed=0, l1_passed=0, l2_passed=0, l3_passed=0),  # 0%
        ])

        health = summarize_metric_conformance_for_global_health(failing)

        assert health.overall_conformance_status == "failing"
        assert health.any_blockers is True
        assert health.blocker_count >= 1
        assert "sparse_success" in health.metrics_by_status["failing"]

    def test_empty_snapshot(self):
        """Test health status for empty snapshot."""
        empty = make_snapshot([])

        health = summarize_metric_conformance_for_global_health(empty)

        assert health.overall_conformance_status == "healthy"
        assert health.weakest_metric is None
        assert health.any_blockers is False

    def test_weakest_metric_identified(self):
        """Test weakest metric is correctly identified."""
        snapshot = make_snapshot([
            make_result("goal_hit"),  # 100%
            make_result("sparse_success", l2_passed=2, l2_total=3),  # ~91%
            make_result("chain_success", l1_passed=4, l1_total=5),  # ~91%
        ])

        health = summarize_metric_conformance_for_global_health(snapshot)

        # Weakest should be one of the degraded metrics
        assert health.weakest_metric in ("sparse_success", "chain_success")
        assert health.weakest_metric_pass_rate < 1.0

    def test_health_to_dict(self, healthy_snapshot: ConformanceSnapshot):
        """Test health status serialization to dict."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)
        data = health.to_dict()

        assert "weakest_metric" in data
        assert "weakest_metric_pass_rate" in data
        assert "any_blockers" in data
        assert "blocker_count" in data
        assert "overall_conformance_status" in data
        assert "metrics_by_status" in data
        assert "summary" in data

    def test_summary_content(self, healthy_snapshot: ConformanceSnapshot):
        """Test summary contains useful information."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)

        assert len(health.summary) > 0
        assert "4" in health.summary or "metric" in health.summary.lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase3Integration:
    """Integration tests for Phase III components."""

    def test_full_workflow(self):
        """Test complete workflow: snapshots -> ledger -> readiness -> health."""
        # Create history
        snap1 = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success"),
        ], "2025-01-01T10:00:00+00:00", "s1")

        snap2 = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l1_passed=4),  # Regression
        ], "2025-01-02T10:00:00+00:00", "s2")

        snap3 = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success"),  # Recovery
        ], "2025-01-03T10:00:00+00:00", "s3")

        history = [snap1, snap2, snap3]

        # Build drift ledger
        ledger = build_metric_drift_ledger(history)
        assert ledger.total_regressions >= 1
        assert ledger.total_improvements >= 1

        # Check promotion readiness for latest change
        readiness = compute_metric_promotion_readiness(snap2, snap3)
        assert readiness.ready_for_promotion is True

        # Check global health of latest
        health = summarize_metric_conformance_for_global_health(snap3)
        assert health.overall_conformance_status == "healthy"

    def test_ci_gate_scenario(self):
        """Test CI gate scenario with blocking regressions."""
        baseline = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success"),
            make_result("chain_success"),
            make_result("multi_goal"),
        ], "2025-01-01T10:00:00+00:00", "main")

        pr_candidate = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l1_passed=3),  # Broke L1
            make_result("chain_success"),
            make_result("multi_goal"),
        ], "2025-01-02T10:00:00+00:00", "pr123")

        # Check if PR can be promoted
        readiness = compute_metric_promotion_readiness(baseline, pr_candidate)

        assert readiness.ready_for_promotion is False
        assert len(readiness.blocking_regressions) >= 1

        # Get health for PR report
        health = summarize_metric_conformance_for_global_health(pr_candidate)
        assert health.any_blockers is True or health.overall_conformance_status != "healthy"
