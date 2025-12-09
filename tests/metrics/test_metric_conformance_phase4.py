"""
Tests for Phase IV: Conformance Drift Compass & Promotion Dashboard

Covers:
- TASK 1: build_metric_drift_compass
- TASK 2: summarize_conformance_for_promotion_dashboard
- TASK 3: build_conformance_director_panel
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
    # Phase III Functions
    build_conformance_snapshot,
    build_metric_drift_ledger,
    compute_metric_promotion_readiness,
    summarize_metric_conformance_for_global_health,
    # Phase IV Functions
    build_metric_drift_compass,
    summarize_conformance_for_promotion_dashboard,
    build_conformance_director_panel,
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
def empty_ledger() -> DriftLedger:
    """Create an empty drift ledger."""
    return build_metric_drift_ledger([])


@pytest.fixture
def stable_ledger(healthy_snapshot: ConformanceSnapshot) -> DriftLedger:
    """Create a drift ledger with no regressions."""
    snap2 = make_snapshot([
        make_result("goal_hit"),
        make_result("sparse_success"),
        make_result("chain_success"),
        make_result("multi_goal"),
    ], timestamp="2025-01-15T12:00:00+00:00", git_sha="xyz9999")
    return build_metric_drift_ledger([healthy_snapshot, snap2])


@pytest.fixture
def caution_ledger() -> DriftLedger:
    """Create a drift ledger with a single regression."""
    snap1 = make_snapshot([make_result("goal_hit")], "2025-01-01T10:00:00+00:00", "s1")
    snap2 = make_snapshot([make_result("goal_hit", l1_passed=4)], "2025-01-02T10:00:00+00:00", "s2")
    return build_metric_drift_ledger([snap1, snap2])


@pytest.fixture
def critical_ledger() -> DriftLedger:
    """Create a drift ledger with chronic regressions."""
    snap1 = make_snapshot([make_result("goal_hit")], "2025-01-01T10:00:00+00:00", "s1")
    snap2 = make_snapshot([make_result("goal_hit", l1_passed=4)], "2025-01-02T10:00:00+00:00", "s2")
    snap3 = make_snapshot([make_result("goal_hit")], "2025-01-03T10:00:00+00:00", "s3")
    snap4 = make_snapshot([make_result("goal_hit", l1_passed=3)], "2025-01-04T10:00:00+00:00", "s4")
    return build_metric_drift_ledger([snap1, snap2, snap3, snap4])


# =============================================================================
# TASK 1: Drift Compass Tests
# =============================================================================

class TestBuildMetricDriftCompass:
    """Tests for build_metric_drift_compass function."""

    def test_empty_ledger(self, empty_ledger: DriftLedger):
        """Test drift compass with empty ledger."""
        compass = build_metric_drift_compass(empty_ledger)

        assert compass["schema_version"] == "1.0.0"
        assert compass["compass_status"] == "STABLE"
        assert len(compass["metrics_with_chronic_regressions"]) == 0
        assert len(compass["metrics_consistently_improving"]) == 0

    def test_stable_compass(self, stable_ledger: DriftLedger):
        """Test drift compass with no regressions."""
        compass = build_metric_drift_compass(stable_ledger)

        assert compass["compass_status"] == "STABLE"
        assert compass["total_regressions"] == 0
        assert len(compass["metrics_with_chronic_regressions"]) == 0

    def test_caution_compass(self, caution_ledger: DriftLedger):
        """Test drift compass with single regression (CAUTION)."""
        compass = build_metric_drift_compass(caution_ledger)

        assert compass["compass_status"] == "CAUTION"
        assert compass["total_regressions"] >= 1
        # No chronic regressions (only 1 regression, need 2+)
        assert len(compass["metrics_with_chronic_regressions"]) == 0

    def test_critical_compass(self, critical_ledger: DriftLedger):
        """Test drift compass with chronic regressions (CRITICAL)."""
        compass = build_metric_drift_compass(critical_ledger)

        assert compass["compass_status"] == "CRITICAL"
        assert compass["total_regressions"] >= 2
        assert "goal_hit" in compass["metrics_with_chronic_regressions"]

    def test_compass_schema_fields(self, stable_ledger: DriftLedger):
        """Test compass contains all required fields."""
        compass = build_metric_drift_compass(stable_ledger)

        assert "schema_version" in compass
        assert "metrics_with_chronic_regressions" in compass
        assert "metrics_consistently_improving" in compass
        assert "compass_status" in compass
        assert "total_regressions" in compass
        assert "total_improvements" in compass
        assert "drift_event_count" in compass

    def test_compass_critical_severity_event(self):
        """Test compass with critical severity drift event.

        Critical severity requires:
        1. A regression that drops the effective conformance level
        2. The new level is L0 or L1

        When tests fail at L0 but the level doesn't drop (because L0 is lowest),
        the severity is "minor" not "critical".

        To trigger CRITICAL status, we need either:
        - Chronic regressions (2+ on same metric), OR
        - A regression with critical severity event
        """
        # Create two regressions on the same metric to trigger CRITICAL status
        snap1 = make_snapshot([make_result("m1")], "2025-01-01T10:00:00+00:00", "s1")
        snap2 = make_snapshot([make_result("m1", l0_passed=1)], "2025-01-02T10:00:00+00:00", "s2")
        snap3 = make_snapshot([make_result("m1")], "2025-01-03T10:00:00+00:00", "s3")
        snap4 = make_snapshot([make_result("m1", l1_passed=4)], "2025-01-04T10:00:00+00:00", "s4")

        ledger = build_metric_drift_ledger([snap1, snap2, snap3, snap4])
        compass = build_metric_drift_compass(ledger)

        # Two regressions on same metric = chronic = CRITICAL
        assert compass["compass_status"] == "CRITICAL"


# =============================================================================
# TASK 2: Promotion Dashboard Adapter Tests
# =============================================================================

class TestSummarizeConformanceForPromotionDashboard:
    """Tests for summarize_conformance_for_promotion_dashboard function."""

    def test_ok_status(
        self,
        healthy_snapshot: ConformanceSnapshot,
        stable_ledger: DriftLedger,
    ):
        """Test OK status when promotion ready and compass stable."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, healthy_snapshot)
        compass = build_metric_drift_compass(stable_ledger)

        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )

        assert dashboard["promotion_ok"] is True
        assert dashboard["status"] == "OK"
        assert len(dashboard["blocking_metrics"]) == 0
        assert "passed" in dashboard["headline"].lower()

    def test_block_status(
        self,
        healthy_snapshot: ConformanceSnapshot,
        regressed_snapshot: ConformanceSnapshot,
        stable_ledger: DriftLedger,
    ):
        """Test BLOCK status when promotion blocked."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, regressed_snapshot)
        compass = build_metric_drift_compass(stable_ledger)

        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )

        assert dashboard["promotion_ok"] is False
        assert dashboard["status"] == "BLOCK"
        assert len(dashboard["blocking_metrics"]) >= 1
        assert "blocked" in dashboard["headline"].lower()

    def test_warn_status_caution_compass(
        self,
        healthy_snapshot: ConformanceSnapshot,
        caution_ledger: DriftLedger,
    ):
        """Test WARN status when promotion ok but compass caution."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, healthy_snapshot)
        compass = build_metric_drift_compass(caution_ledger)

        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )

        assert dashboard["promotion_ok"] is True
        assert dashboard["status"] == "WARN"
        assert "regression" in dashboard["headline"].lower()

    def test_warn_status_critical_compass(
        self,
        healthy_snapshot: ConformanceSnapshot,
        critical_ledger: DriftLedger,
    ):
        """Test WARN status when promotion ok but compass critical."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, healthy_snapshot)
        compass = build_metric_drift_compass(critical_ledger)

        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )

        assert dashboard["promotion_ok"] is True
        assert dashboard["status"] == "WARN"
        assert "repeated" in dashboard["headline"].lower() or "regression" in dashboard["headline"].lower()

    def test_dashboard_schema_fields(
        self,
        healthy_snapshot: ConformanceSnapshot,
        stable_ledger: DriftLedger,
    ):
        """Test dashboard contains all required fields."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, healthy_snapshot)
        compass = build_metric_drift_compass(stable_ledger)

        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )

        assert "promotion_ok" in dashboard
        assert "blocking_metrics" in dashboard
        assert "status" in dashboard
        assert "headline" in dashboard

    def test_blocking_metrics_populated(
        self,
        healthy_snapshot: ConformanceSnapshot,
        regressed_snapshot: ConformanceSnapshot,
        stable_ledger: DriftLedger,
    ):
        """Test blocking_metrics contains level identifiers."""
        readiness = compute_metric_promotion_readiness(healthy_snapshot, regressed_snapshot)
        compass = build_metric_drift_compass(stable_ledger)

        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )

        # Should have blocking metrics with level info
        assert len(dashboard["blocking_metrics"]) >= 1
        assert any("L1" in m for m in dashboard["blocking_metrics"])


# =============================================================================
# TASK 3: Director Conformance Panel Tests
# =============================================================================

class TestBuildConformanceDirectorPanel:
    """Tests for build_conformance_director_panel function."""

    def test_green_light(
        self,
        healthy_snapshot: ConformanceSnapshot,
        stable_ledger: DriftLedger,
    ):
        """Test GREEN light when healthy and stable."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)
        compass = build_metric_drift_compass(stable_ledger)

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        assert panel["status_light"] == "GREEN"
        assert panel["overall_conformance_status"] == "healthy"
        assert "stable" in panel["headline"].lower()

    def test_yellow_light_degraded(
        self,
        stable_ledger: DriftLedger,
    ):
        """Test YELLOW light when metrics degraded."""
        degraded = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l2_passed=2, l2_total=3),  # ~91%
        ])

        health = summarize_metric_conformance_for_global_health(degraded)
        compass = build_metric_drift_compass(stable_ledger)

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        assert panel["status_light"] == "YELLOW"
        assert "attention" in panel["headline"].lower() or "elevated" in panel["headline"].lower()

    def test_yellow_light_caution_compass(
        self,
        healthy_snapshot: ConformanceSnapshot,
        caution_ledger: DriftLedger,
    ):
        """Test YELLOW light when compass in CAUTION."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)
        compass = build_metric_drift_compass(caution_ledger)

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        # Healthy but caution compass = YELLOW
        assert panel["status_light"] == "YELLOW"

    def test_red_light_failing(self):
        """Test RED light when metrics failing."""
        failing = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l0_passed=0, l1_passed=0, l2_passed=0, l3_passed=0),
        ])

        health = summarize_metric_conformance_for_global_health(failing)
        compass = build_metric_drift_compass(build_metric_drift_ledger([]))

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        assert panel["status_light"] == "RED"
        assert "action" in panel["headline"].lower() or "required" in panel["headline"].lower()

    def test_red_light_critical_compass(
        self,
        healthy_snapshot: ConformanceSnapshot,
        critical_ledger: DriftLedger,
    ):
        """Test RED light when compass CRITICAL."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)
        compass = build_metric_drift_compass(critical_ledger)

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        assert panel["status_light"] == "RED"

    def test_panel_schema_fields(
        self,
        healthy_snapshot: ConformanceSnapshot,
        stable_ledger: DriftLedger,
    ):
        """Test panel contains all required fields."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)
        compass = build_metric_drift_compass(stable_ledger)

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        assert "status_light" in panel
        assert "overall_conformance_status" in panel
        assert "weakest_metric" in panel
        assert "headline" in panel
        assert "compass_status" in panel

    def test_weakest_metric_shown(self):
        """Test weakest metric is included in panel."""
        snapshot = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l2_passed=2, l2_total=3),
        ])

        health = summarize_metric_conformance_for_global_health(snapshot)
        compass = build_metric_drift_compass(build_metric_drift_ledger([]))

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        assert panel["weakest_metric"] == "sparse_success"

    def test_headline_neutral_tone(
        self,
        healthy_snapshot: ConformanceSnapshot,
        stable_ledger: DriftLedger,
    ):
        """Test headline uses neutral professional tone."""
        health = summarize_metric_conformance_for_global_health(healthy_snapshot)
        compass = build_metric_drift_compass(stable_ledger)

        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        headline = panel["headline"].lower()
        # No overly positive or negative language
        assert "great" not in headline
        assert "terrible" not in headline
        assert "amazing" not in headline


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase4Integration:
    """Integration tests for Phase IV components."""

    def test_full_dashboard_workflow(self):
        """Test complete workflow: ledger -> compass -> dashboard -> panel."""
        # Create history with regression and recovery
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

        # Build drift compass
        compass = build_metric_drift_compass(ledger)
        assert compass["compass_status"] in ("STABLE", "CAUTION")

        # Compute promotion readiness
        readiness = compute_metric_promotion_readiness(snap2, snap3)

        # Build promotion dashboard
        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )
        assert dashboard["promotion_ok"] is True

        # Build director panel
        health = summarize_metric_conformance_for_global_health(snap3)
        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )
        assert panel["overall_conformance_status"] == "healthy"

    def test_ci_blocking_scenario(self):
        """Test CI blocking scenario with full dashboard data."""
        baseline = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success"),
        ], "2025-01-01T10:00:00+00:00", "main")

        pr_candidate = make_snapshot([
            make_result("goal_hit"),
            make_result("sparse_success", l1_passed=3),  # Broke L1
        ], "2025-01-02T10:00:00+00:00", "pr123")

        # Build ledger from history
        ledger = build_metric_drift_ledger([baseline, pr_candidate])
        compass = build_metric_drift_compass(ledger)

        # Check promotion
        readiness = compute_metric_promotion_readiness(baseline, pr_candidate)
        dashboard = summarize_conformance_for_promotion_dashboard(
            readiness.to_dict(),
            compass,
        )

        assert dashboard["promotion_ok"] is False
        assert dashboard["status"] == "BLOCK"

        # Check director panel
        health = summarize_metric_conformance_for_global_health(pr_candidate)
        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        # Should show elevated attention due to regression
        assert panel["status_light"] in ("YELLOW", "RED")

    def test_chronic_regression_escalation(self):
        """Test chronic regression escalates compass to CRITICAL."""
        # Create 4 snapshots with repeated regressions on same metric
        snap1 = make_snapshot([make_result("goal_hit")], "2025-01-01T10:00:00+00:00", "s1")
        snap2 = make_snapshot([make_result("goal_hit", l1_passed=4)], "2025-01-02T10:00:00+00:00", "s2")
        snap3 = make_snapshot([make_result("goal_hit")], "2025-01-03T10:00:00+00:00", "s3")
        snap4 = make_snapshot([make_result("goal_hit", l1_passed=3)], "2025-01-04T10:00:00+00:00", "s4")

        history = [snap1, snap2, snap3, snap4]

        ledger = build_metric_drift_ledger(history)
        compass = build_metric_drift_compass(ledger)

        assert compass["compass_status"] == "CRITICAL"
        assert "goal_hit" in compass["metrics_with_chronic_regressions"]

        # Director panel should show RED
        health = summarize_metric_conformance_for_global_health(snap4)
        panel = build_conformance_director_panel(
            health.to_dict(),
            compass,
        )

        assert panel["status_light"] == "RED"
