"""
Tests for Phase V: Metrics × Budget × Policy Triangulation

Covers:
- TASK 1: build_metric_budget_joint_view
- TASK 2: summarize_conformance_for_global_console
- TASK 3: attach_policy_telemetry_hint
"""

import pytest
from typing import Dict, List, Any

from backend.metrics.metric_conformance_snapshot import (
    # Data classes
    MetricConformanceResult,
    ConformanceSnapshot,
    DriftLedger,
    # Phase III Functions
    build_conformance_snapshot,
    build_metric_drift_ledger,
    summarize_metric_conformance_for_global_health,
    # Phase IV Functions
    build_metric_drift_compass,
    build_conformance_director_panel,
    # Phase V Functions
    build_metric_budget_joint_view,
    summarize_conformance_for_global_console,
    attach_policy_telemetry_hint,
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

    for i in range(l0_total):
        test_id = f"test_L0_{i}"
        tests_by_level["L0"].append(test_id)
        tests_passed[test_id] = i < l0_passed

    for i in range(l1_total):
        test_id = f"test_L1_{i}"
        tests_by_level["L1"].append(test_id)
        tests_passed[test_id] = i < l1_passed

    for i in range(l2_total):
        test_id = f"test_L2_{i}"
        tests_by_level["L2"].append(test_id)
        tests_passed[test_id] = i < l2_passed

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
def stable_compass() -> Dict[str, Any]:
    """Create a stable drift compass (no regressions)."""
    return {
        "schema_version": "1.0.0",
        "metrics_with_chronic_regressions": [],
        "metrics_consistently_improving": [],
        "compass_status": "STABLE",
        "total_regressions": 0,
        "total_improvements": 0,
        "drift_event_count": 0,
    }


@pytest.fixture
def caution_compass() -> Dict[str, Any]:
    """Create a caution drift compass (single regression)."""
    return {
        "schema_version": "1.0.0",
        "metrics_with_chronic_regressions": [],
        "metrics_consistently_improving": [],
        "compass_status": "CAUTION",
        "total_regressions": 1,
        "total_improvements": 0,
        "drift_event_count": 1,
    }


@pytest.fixture
def critical_compass() -> Dict[str, Any]:
    """Create a critical drift compass (chronic regressions)."""
    return {
        "schema_version": "1.0.0",
        "metrics_with_chronic_regressions": ["goal_hit"],
        "metrics_consistently_improving": [],
        "compass_status": "CRITICAL",
        "total_regressions": 2,
        "total_improvements": 0,
        "drift_event_count": 2,
    }


@pytest.fixture
def ok_budget_view() -> Dict[str, Any]:
    """Create an OK budget uplift view."""
    return {
        "schema_version": "1.0.0",
        "uplift_ready": True,
        "blocking_slices": [],
        "status": "OK",
        "notes": ["Budget and metrics are within acceptable bounds"],
    }


@pytest.fixture
def warn_budget_view() -> Dict[str, Any]:
    """Create a WARN budget uplift view."""
    return {
        "schema_version": "1.0.0",
        "uplift_ready": True,
        "blocking_slices": ["slice_a"],
        "status": "WARN",
        "notes": ["Budget risk band is MEDIUM"],
    }


@pytest.fixture
def block_budget_view() -> Dict[str, Any]:
    """Create a BLOCK budget uplift view."""
    return {
        "schema_version": "1.0.0",
        "uplift_ready": False,
        "blocking_slices": ["slice_a", "slice_b"],
        "status": "BLOCK",
        "notes": ["Budget risk band is HIGH"],
    }


@pytest.fixture
def healthy_health_status() -> Dict[str, Any]:
    """Create a healthy global health status."""
    return {
        "weakest_metric": None,
        "weakest_metric_pass_rate": 1.0,
        "any_blockers": False,
        "blocker_count": 0,
        "overall_conformance_status": "healthy",
        "metrics_by_status": {"healthy": ["goal_hit", "sparse_success"], "degraded": [], "critical": [], "failing": []},
        "summary": "All metrics healthy",
    }


@pytest.fixture
def degraded_health_status() -> Dict[str, Any]:
    """Create a degraded global health status."""
    return {
        "weakest_metric": "sparse_success",
        "weakest_metric_pass_rate": 0.91,
        "any_blockers": False,
        "blocker_count": 0,
        "overall_conformance_status": "degraded",
        "metrics_by_status": {"healthy": ["goal_hit"], "degraded": ["sparse_success"], "critical": [], "failing": []},
        "summary": "1 metric degraded",
    }


# =============================================================================
# TASK 1: Joint Metric-Budget View Tests
# =============================================================================

class TestBuildMetricBudgetJointView:
    """Tests for build_metric_budget_joint_view function."""

    def test_ok_when_both_ok(
        self,
        stable_compass: Dict[str, Any],
        ok_budget_view: Dict[str, Any],
    ):
        """Test OK status when metrics stable and budget OK."""
        joint = build_metric_budget_joint_view(stable_compass, ok_budget_view)

        assert joint["joint_status"] == "OK"
        assert joint["blocking_metrics"] == []
        assert joint["blocking_slices"] == []
        assert "acceptable bounds" in joint["notes"][0].lower()

    def test_block_when_budget_blocks(
        self,
        stable_compass: Dict[str, Any],
        block_budget_view: Dict[str, Any],
    ):
        """Test BLOCK status when budget blocks."""
        joint = build_metric_budget_joint_view(stable_compass, block_budget_view)

        assert joint["joint_status"] == "BLOCK"
        assert joint["blocking_slices"] == ["slice_a", "slice_b"]
        assert any("budget" in note.lower() for note in joint["notes"])

    def test_block_when_metrics_critical(
        self,
        critical_compass: Dict[str, Any],
        ok_budget_view: Dict[str, Any],
    ):
        """Test BLOCK status when metrics critical (chronic regressions)."""
        joint = build_metric_budget_joint_view(critical_compass, ok_budget_view)

        assert joint["joint_status"] == "BLOCK"
        assert "goal_hit" in joint["blocking_metrics"]
        assert any("chronic" in note.lower() for note in joint["notes"])

    def test_block_when_both_block(
        self,
        critical_compass: Dict[str, Any],
        block_budget_view: Dict[str, Any],
    ):
        """Test BLOCK status when both metrics and budget block."""
        joint = build_metric_budget_joint_view(critical_compass, block_budget_view)

        assert joint["joint_status"] == "BLOCK"
        # Budget blocks first in priority
        assert any("budget" in note.lower() for note in joint["notes"])

    def test_warn_when_budget_warns_metrics_ok(
        self,
        stable_compass: Dict[str, Any],
        warn_budget_view: Dict[str, Any],
    ):
        """Test WARN status when budget warns but metrics stable."""
        joint = build_metric_budget_joint_view(stable_compass, warn_budget_view)

        assert joint["joint_status"] == "WARN"
        assert joint["blocking_slices"] == ["slice_a"]
        assert any("elevated" in note.lower() or "medium" in note.lower() for note in joint["notes"])

    def test_warn_when_metrics_caution_budget_ok(
        self,
        caution_compass: Dict[str, Any],
        ok_budget_view: Dict[str, Any],
    ):
        """Test WARN status when metrics caution but budget OK."""
        joint = build_metric_budget_joint_view(caution_compass, ok_budget_view)

        assert joint["joint_status"] == "WARN"
        assert any("regression" in note.lower() or "drift" in note.lower() for note in joint["notes"])

    def test_schema_fields(
        self,
        stable_compass: Dict[str, Any],
        ok_budget_view: Dict[str, Any],
    ):
        """Test joint view contains all required schema fields."""
        joint = build_metric_budget_joint_view(stable_compass, ok_budget_view)

        assert "schema_version" in joint
        assert "joint_status" in joint
        assert "blocking_metrics" in joint
        assert "blocking_slices" in joint
        assert "notes" in joint

    def test_block_when_uplift_not_ready(
        self,
        stable_compass: Dict[str, Any],
    ):
        """Test BLOCK when budget uplift_ready is False even if status OK."""
        budget_view = {
            "schema_version": "1.0.0",
            "uplift_ready": False,
            "blocking_slices": [],
            "status": "OK",  # Status can be OK but uplift_ready False
            "notes": ["Metrics not ready"],
        }

        joint = build_metric_budget_joint_view(stable_compass, budget_view)

        assert joint["joint_status"] == "BLOCK"
        assert any("uplift" in note.lower() or "readiness" in note.lower() for note in joint["notes"])


# =============================================================================
# TASK 2: Global Console Adapter Tests
# =============================================================================

class TestSummarizeConformanceForGlobalConsole:
    """Tests for summarize_conformance_for_global_console function."""

    def test_green_when_all_ok(
        self,
        healthy_health_status: Dict[str, Any],
        stable_compass: Dict[str, Any],
    ):
        """Test GREEN light when all systems healthy."""
        joint_view = {
            "schema_version": "1.0.0",
            "joint_status": "OK",
            "blocking_metrics": [],
            "blocking_slices": [],
            "notes": [],
        }

        console = summarize_conformance_for_global_console(
            healthy_health_status,
            stable_compass,
            joint_view,
        )

        assert console["metrics_ok"] is True
        assert console["status_light"] == "GREEN"
        assert "healthy" in console["headline"].lower()

    def test_red_when_joint_blocks(
        self,
        healthy_health_status: Dict[str, Any],
        stable_compass: Dict[str, Any],
    ):
        """Test RED light when joint view blocks."""
        joint_view = {
            "schema_version": "1.0.0",
            "joint_status": "BLOCK",
            "blocking_metrics": ["goal_hit"],
            "blocking_slices": ["slice_a"],
            "notes": [],
        }

        console = summarize_conformance_for_global_console(
            healthy_health_status,
            stable_compass,
            joint_view,
        )

        assert console["metrics_ok"] is False
        assert console["status_light"] == "RED"
        assert "action" in console["headline"].lower()

    def test_yellow_when_caution(
        self,
        healthy_health_status: Dict[str, Any],
        caution_compass: Dict[str, Any],
    ):
        """Test YELLOW light when compass shows caution."""
        joint_view = {
            "schema_version": "1.0.0",
            "joint_status": "WARN",
            "blocking_metrics": [],
            "blocking_slices": [],
            "notes": [],
        }

        console = summarize_conformance_for_global_console(
            healthy_health_status,
            caution_compass,
            joint_view,
        )

        assert console["metrics_ok"] is True
        assert console["status_light"] == "YELLOW"
        assert "attention" in console["headline"].lower() or "elevated" in console["headline"].lower()

    def test_yellow_when_degraded(
        self,
        degraded_health_status: Dict[str, Any],
        stable_compass: Dict[str, Any],
    ):
        """Test YELLOW light when health degraded."""
        joint_view = {
            "schema_version": "1.0.0",
            "joint_status": "OK",
            "blocking_metrics": [],
            "blocking_slices": [],
            "notes": [],
        }

        console = summarize_conformance_for_global_console(
            degraded_health_status,
            stable_compass,
            joint_view,
        )

        assert console["status_light"] == "YELLOW"
        assert console["weakest_metric"] == "sparse_success"

    def test_schema_fields(
        self,
        healthy_health_status: Dict[str, Any],
        stable_compass: Dict[str, Any],
    ):
        """Test console contains all required schema fields."""
        joint_view = {
            "schema_version": "1.0.0",
            "joint_status": "OK",
            "blocking_metrics": [],
            "blocking_slices": [],
            "notes": [],
        }

        console = summarize_conformance_for_global_console(
            healthy_health_status,
            stable_compass,
            joint_view,
        )

        assert "schema_version" in console
        assert "metrics_ok" in console
        assert "status_light" in console
        assert "weakest_metric" in console
        assert "headline" in console

    def test_red_when_critical_compass(
        self,
        healthy_health_status: Dict[str, Any],
        critical_compass: Dict[str, Any],
    ):
        """Test RED light when compass is CRITICAL."""
        joint_view = {
            "schema_version": "1.0.0",
            "joint_status": "BLOCK",
            "blocking_metrics": ["goal_hit"],
            "blocking_slices": [],
            "notes": [],
        }

        console = summarize_conformance_for_global_console(
            healthy_health_status,
            critical_compass,
            joint_view,
        )

        assert console["metrics_ok"] is False
        assert console["status_light"] == "RED"


# =============================================================================
# TASK 3: Policy Telemetry Hint Tests
# =============================================================================

class TestAttachPolicyTelemetryHint:
    """Tests for attach_policy_telemetry_hint function."""

    def test_default_values_when_no_radar(self):
        """Test default values when policy radar is None."""
        panel = {"status_light": "GREEN", "headline": "All good"}

        enriched = attach_policy_telemetry_hint(panel, None)

        assert enriched["policy_volatility_status"] == "UNKNOWN"
        assert enriched["max_policy_l2_drift"] == 0.0
        assert "policy_hint_schema_version" in enriched
        # Original fields preserved
        assert enriched["status_light"] == "GREEN"
        assert enriched["headline"] == "All good"

    def test_extracts_radar_values(self):
        """Test extraction of values from policy drift radar."""
        panel = {"status_light": "YELLOW"}
        radar = {
            "volatility_status": "VOLATILE",
            "max_l2_drift": 0.35,
        }

        enriched = attach_policy_telemetry_hint(panel, radar)

        assert enriched["policy_volatility_status"] == "VOLATILE"
        assert enriched["max_policy_l2_drift"] == 0.35

    def test_preserves_all_original_fields(self):
        """Test all original panel fields are preserved."""
        panel = {
            "status_light": "RED",
            "headline": "Issues detected",
            "weakest_metric": "goal_hit",
            "custom_field": "custom_value",
        }

        enriched = attach_policy_telemetry_hint(panel, None)

        assert enriched["status_light"] == "RED"
        assert enriched["headline"] == "Issues detected"
        assert enriched["weakest_metric"] == "goal_hit"
        assert enriched["custom_field"] == "custom_value"

    def test_handles_missing_radar_fields(self):
        """Test handles partial radar dict gracefully."""
        panel = {"status_light": "GREEN"}
        radar = {"volatility_status": "STABLE"}  # missing max_l2_drift

        enriched = attach_policy_telemetry_hint(panel, radar)

        assert enriched["policy_volatility_status"] == "STABLE"
        assert enriched["max_policy_l2_drift"] == 0.0  # Default

    def test_critical_volatility_status(self):
        """Test CRITICAL volatility status is preserved."""
        panel = {"status_light": "RED"}
        radar = {
            "volatility_status": "CRITICAL",
            "max_l2_drift": 0.85,
        }

        enriched = attach_policy_telemetry_hint(panel, radar)

        assert enriched["policy_volatility_status"] == "CRITICAL"
        assert enriched["max_policy_l2_drift"] == 0.85


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase5Integration:
    """Integration tests for Phase V components."""

    def test_full_triangulation_workflow(self):
        """Test complete workflow: compass + budget -> joint -> console."""
        # Build snapshots and compass
        snap1 = make_snapshot([make_result("goal_hit")], "2025-01-01T10:00:00+00:00", "s1")
        snap2 = make_snapshot([make_result("goal_hit")], "2025-01-02T10:00:00+00:00", "s2")

        ledger = build_metric_drift_ledger([snap1, snap2])
        compass = build_metric_drift_compass(ledger)

        # Budget view
        budget_view = {
            "schema_version": "1.0.0",
            "uplift_ready": True,
            "blocking_slices": [],
            "status": "OK",
            "notes": [],
        }

        # Build joint view
        joint = build_metric_budget_joint_view(compass, budget_view)
        assert joint["joint_status"] == "OK"

        # Get health status
        health = summarize_metric_conformance_for_global_health(snap2)

        # Build console summary
        console = summarize_conformance_for_global_console(
            health.to_dict(),
            compass,
            joint,
        )

        assert console["metrics_ok"] is True
        assert console["status_light"] == "GREEN"

    def test_budget_block_cascades_to_console(self):
        """Test that budget BLOCK cascades to RED console light."""
        compass = {
            "schema_version": "1.0.0",
            "metrics_with_chronic_regressions": [],
            "compass_status": "STABLE",
            "total_regressions": 0,
            "total_improvements": 0,
            "drift_event_count": 0,
        }

        budget_view = {
            "schema_version": "1.0.0",
            "uplift_ready": False,
            "blocking_slices": ["critical_slice"],
            "status": "BLOCK",
            "notes": ["Budget risk HIGH"],
        }

        joint = build_metric_budget_joint_view(compass, budget_view)
        assert joint["joint_status"] == "BLOCK"

        health = {
            "weakest_metric": None,
            "weakest_metric_pass_rate": 1.0,
            "any_blockers": False,
            "blocker_count": 0,
            "overall_conformance_status": "healthy",
            "metrics_by_status": {"healthy": [], "degraded": [], "critical": [], "failing": []},
            "summary": "",
        }

        console = summarize_conformance_for_global_console(health, compass, joint)
        assert console["status_light"] == "RED"
        assert console["metrics_ok"] is False

    def test_metric_critical_cascades_to_console(self):
        """Test that CRITICAL compass cascades to RED console light."""
        compass = {
            "schema_version": "1.0.0",
            "metrics_with_chronic_regressions": ["goal_hit"],
            "compass_status": "CRITICAL",
            "total_regressions": 2,
            "total_improvements": 0,
            "drift_event_count": 2,
        }

        budget_view = {
            "schema_version": "1.0.0",
            "uplift_ready": True,
            "blocking_slices": [],
            "status": "OK",
            "notes": [],
        }

        joint = build_metric_budget_joint_view(compass, budget_view)
        assert joint["joint_status"] == "BLOCK"

        health = {
            "weakest_metric": None,
            "weakest_metric_pass_rate": 1.0,
            "any_blockers": False,
            "blocker_count": 0,
            "overall_conformance_status": "healthy",
            "metrics_by_status": {"healthy": [], "degraded": [], "critical": [], "failing": []},
            "summary": "",
        }

        console = summarize_conformance_for_global_console(health, compass, joint)
        assert console["status_light"] == "RED"

    def test_enriched_director_panel(self):
        """Test director panel enriched with policy hints."""
        # Build director panel
        health = {
            "overall_conformance_status": "healthy",
            "weakest_metric": None,
            "weakest_metric_pass_rate": 1.0,
            "any_blockers": False,
            "blocker_count": 0,
        }
        compass = {
            "compass_status": "STABLE",
            "metrics_with_chronic_regressions": [],
        }

        panel = build_conformance_director_panel(health, compass)

        # Enrich with policy hints
        radar = {
            "volatility_status": "STABLE",
            "max_l2_drift": 0.05,
        }

        enriched = attach_policy_telemetry_hint(panel, radar)

        # All original fields present
        assert enriched["status_light"] == "GREEN"
        assert enriched["headline"] is not None

        # Policy hints added
        assert enriched["policy_volatility_status"] == "STABLE"
        assert enriched["max_policy_l2_drift"] == 0.05
