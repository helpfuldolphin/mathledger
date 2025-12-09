"""
Tests for Phase VI: Metrics + Budget as One Governance Surface

Covers:
- TASK 1: summarize_conformance_and_budget_for_global_console (Global Console Tile)
- TASK 2: to_governance_signal_for_metrics (GovernanceSignal Adapter)
"""

import pytest
from typing import Dict, List, Any

from backend.metrics.metric_conformance_snapshot import (
    # Phase VI Functions
    summarize_conformance_and_budget_for_global_console,
    to_governance_signal_for_metrics,
    # Phase V Functions (for building joint views)
    build_metric_budget_joint_view,
)


# =============================================================================
# Test Fixtures
# =============================================================================

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


@pytest.fixture
def failing_health_status() -> Dict[str, Any]:
    """Create a failing global health status."""
    return {
        "weakest_metric": "goal_hit",
        "weakest_metric_pass_rate": 0.50,
        "any_blockers": True,
        "blocker_count": 2,
        "overall_conformance_status": "failing",
        "metrics_by_status": {"healthy": [], "degraded": [], "critical": [], "failing": ["goal_hit", "sparse_success"]},
        "summary": "2 metrics failing",
    }


@pytest.fixture
def ok_joint_view() -> Dict[str, Any]:
    """Create an OK joint view."""
    return {
        "schema_version": "1.0.0",
        "joint_status": "OK",
        "blocking_metrics": [],
        "blocking_slices": [],
        "notes": ["Metrics and budget are within acceptable bounds"],
    }


@pytest.fixture
def warn_joint_view() -> Dict[str, Any]:
    """Create a WARN joint view."""
    return {
        "schema_version": "1.0.0",
        "joint_status": "WARN",
        "blocking_metrics": [],
        "blocking_slices": ["slice_a"],
        "notes": ["Budget risk is elevated"],
    }


@pytest.fixture
def block_joint_view_metrics() -> Dict[str, Any]:
    """Create a BLOCK joint view due to metrics."""
    return {
        "schema_version": "1.0.0",
        "joint_status": "BLOCK",
        "blocking_metrics": ["goal_hit"],
        "blocking_slices": [],
        "notes": ["Chronic metric regressions detected: goal_hit"],
    }


@pytest.fixture
def block_joint_view_budget() -> Dict[str, Any]:
    """Create a BLOCK joint view due to budget."""
    return {
        "schema_version": "1.0.0",
        "joint_status": "BLOCK",
        "blocking_metrics": [],
        "blocking_slices": ["slice_a", "slice_b"],
        "notes": ["Budget risk blocks promotion"],
    }


@pytest.fixture
def block_joint_view_both() -> Dict[str, Any]:
    """Create a BLOCK joint view due to both metrics and budget."""
    return {
        "schema_version": "1.0.0",
        "joint_status": "BLOCK",
        "blocking_metrics": ["goal_hit"],
        "blocking_slices": ["slice_a"],
        "notes": ["Budget risk blocks promotion", "Chronic regressions detected"],
    }


# =============================================================================
# TASK 1: Global Console Tile Tests
# =============================================================================

class TestSummarizeConformanceAndBudgetForGlobalConsole:
    """Tests for summarize_conformance_and_budget_for_global_console function."""

    def test_green_when_all_healthy(
        self,
        healthy_health_status: Dict[str, Any],
        ok_joint_view: Dict[str, Any],
    ):
        """Test GREEN light when both conformance and budget healthy."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            ok_joint_view,
        )

        assert tile["metrics_ok"] is True
        assert tile["status_light"] == "GREEN"
        assert "healthy" in tile["headline"].lower()

    def test_yellow_when_degraded(
        self,
        degraded_health_status: Dict[str, Any],
        ok_joint_view: Dict[str, Any],
    ):
        """Test YELLOW light when conformance degraded."""
        tile = summarize_conformance_and_budget_for_global_console(
            degraded_health_status,
            ok_joint_view,
        )

        assert tile["metrics_ok"] is True
        assert tile["status_light"] == "YELLOW"
        assert tile["weakest_metric"] == "sparse_success"

    def test_yellow_when_budget_warns(
        self,
        healthy_health_status: Dict[str, Any],
        warn_joint_view: Dict[str, Any],
    ):
        """Test YELLOW light when budget warns."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            warn_joint_view,
        )

        assert tile["metrics_ok"] is True
        assert tile["status_light"] == "YELLOW"
        assert "elevated" in tile["headline"].lower()

    def test_red_when_metrics_block(
        self,
        healthy_health_status: Dict[str, Any],
        block_joint_view_metrics: Dict[str, Any],
    ):
        """Test RED light when metrics block."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            block_joint_view_metrics,
        )

        assert tile["metrics_ok"] is False
        assert tile["status_light"] == "RED"
        assert "blocked" in tile["headline"].lower()

    def test_red_when_budget_blocks(
        self,
        healthy_health_status: Dict[str, Any],
        block_joint_view_budget: Dict[str, Any],
    ):
        """Test RED light when budget blocks."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            block_joint_view_budget,
        )

        assert tile["metrics_ok"] is False
        assert tile["status_light"] == "RED"
        assert "slice" in tile["headline"].lower()

    def test_red_when_conformance_failing(
        self,
        failing_health_status: Dict[str, Any],
        ok_joint_view: Dict[str, Any],
    ):
        """Test RED light when conformance failing."""
        tile = summarize_conformance_and_budget_for_global_console(
            failing_health_status,
            ok_joint_view,
        )

        assert tile["metrics_ok"] is False
        assert tile["status_light"] == "RED"

    def test_schema_fields(
        self,
        healthy_health_status: Dict[str, Any],
        ok_joint_view: Dict[str, Any],
    ):
        """Test tile contains all required schema fields."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            ok_joint_view,
        )

        assert "schema_version" in tile
        assert "metrics_ok" in tile
        assert "status_light" in tile
        assert "headline" in tile
        assert "weakest_metric" in tile


# =============================================================================
# TASK 2: GovernanceSignal Adapter Tests
# =============================================================================

class TestToGovernanceSignalForMetrics:
    """Tests for to_governance_signal_for_metrics function."""

    def test_proceed_when_all_ok(
        self,
        healthy_health_status: Dict[str, Any],
        ok_joint_view: Dict[str, Any],
    ):
        """Test PROCEED status when all systems healthy."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            ok_joint_view,
        )

        signal = to_governance_signal_for_metrics(tile, ok_joint_view)

        assert signal["status"] == "PROCEED"
        assert signal["metrics_ok"] is True
        assert signal["blocking_rules"] == []
        assert signal["blocking_rate"] == 0.0
        assert signal["signal_type"] == "METRICS_LAYER_GATE"

    def test_proceed_with_caution_when_warning(
        self,
        healthy_health_status: Dict[str, Any],
        warn_joint_view: Dict[str, Any],
    ):
        """Test PROCEED_WITH_CAUTION status when warnings present."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            warn_joint_view,
        )

        signal = to_governance_signal_for_metrics(tile, warn_joint_view)

        assert signal["status"] == "PROCEED_WITH_CAUTION"
        assert signal["metrics_ok"] is True
        assert signal["blocking_rate"] > 0.0
        assert signal["blocking_rate"] < 0.5

    def test_block_when_metrics_block(
        self,
        healthy_health_status: Dict[str, Any],
        block_joint_view_metrics: Dict[str, Any],
    ):
        """Test BLOCK status when metrics block."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            block_joint_view_metrics,
        )

        signal = to_governance_signal_for_metrics(tile, block_joint_view_metrics)

        assert signal["status"] == "BLOCK"
        assert signal["metrics_ok"] is False
        assert "METRIC_DRIFT_CRITICAL" in signal["blocking_rules"]
        assert signal["blocking_rate"] >= 0.5

    def test_block_when_budget_blocks(
        self,
        healthy_health_status: Dict[str, Any],
        block_joint_view_budget: Dict[str, Any],
    ):
        """Test BLOCK status when budget blocks."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            block_joint_view_budget,
        )

        signal = to_governance_signal_for_metrics(tile, block_joint_view_budget)

        assert signal["status"] == "BLOCK"
        assert signal["metrics_ok"] is False
        assert "BUDGET_HIGH_RISK" in signal["blocking_rules"]

    def test_block_when_both_block(
        self,
        healthy_health_status: Dict[str, Any],
        block_joint_view_both: Dict[str, Any],
    ):
        """Test BLOCK status with both blocking rules present."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            block_joint_view_both,
        )

        signal = to_governance_signal_for_metrics(tile, block_joint_view_both)

        assert signal["status"] == "BLOCK"
        assert "METRIC_DRIFT_CRITICAL" in signal["blocking_rules"]
        assert "BUDGET_HIGH_RISK" in signal["blocking_rules"]
        # Higher blocking rate with multiple blockers
        assert signal["blocking_rate"] >= 0.75

    def test_schema_fields(
        self,
        healthy_health_status: Dict[str, Any],
        ok_joint_view: Dict[str, Any],
    ):
        """Test signal contains all required schema fields."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            ok_joint_view,
        )

        signal = to_governance_signal_for_metrics(tile, ok_joint_view)

        assert "schema_version" in signal
        assert "signal_type" in signal
        assert "status" in signal
        assert "blocking_rules" in signal
        assert "blocking_rate" in signal
        assert "metrics_ok" in signal
        assert "notes" in signal

    def test_notes_include_joint_notes(
        self,
        healthy_health_status: Dict[str, Any],
        ok_joint_view: Dict[str, Any],
    ):
        """Test notes include joint view notes."""
        tile = summarize_conformance_and_budget_for_global_console(
            healthy_health_status,
            ok_joint_view,
        )

        signal = to_governance_signal_for_metrics(tile, ok_joint_view)

        # Should include both signal notes and joint notes
        assert len(signal["notes"]) >= 1
        assert any("acceptable bounds" in note for note in signal["notes"])


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase6Integration:
    """Integration tests for Phase VI components."""

    def test_full_governance_workflow(self):
        """Test complete workflow from compass to governance signal."""
        # Build compass
        compass = {
            "schema_version": "1.0.0",
            "metrics_with_chronic_regressions": [],
            "compass_status": "STABLE",
            "total_regressions": 0,
            "total_improvements": 0,
            "drift_event_count": 0,
        }

        # Build budget view
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

        # Build health status
        health = {
            "weakest_metric": None,
            "weakest_metric_pass_rate": 1.0,
            "any_blockers": False,
            "blocker_count": 0,
            "overall_conformance_status": "healthy",
            "metrics_by_status": {},
            "summary": "",
        }

        # Build console tile
        tile = summarize_conformance_and_budget_for_global_console(health, joint)
        assert tile["status_light"] == "GREEN"

        # Generate governance signal
        signal = to_governance_signal_for_metrics(tile, joint)
        assert signal["status"] == "PROCEED"
        assert signal["metrics_ok"] is True

    def test_budget_block_cascades_to_governance_block(self):
        """Test that budget BLOCK cascades to governance BLOCK."""
        compass = {
            "schema_version": "1.0.0",
            "metrics_with_chronic_regressions": [],
            "compass_status": "STABLE",
            "total_regressions": 0,
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
            "overall_conformance_status": "healthy",
        }

        tile = summarize_conformance_and_budget_for_global_console(health, joint)
        signal = to_governance_signal_for_metrics(tile, joint)

        assert signal["status"] == "BLOCK"
        assert "BUDGET_HIGH_RISK" in signal["blocking_rules"]

    def test_metric_critical_cascades_to_governance_block(self):
        """Test that CRITICAL compass cascades to governance BLOCK."""
        compass = {
            "schema_version": "1.0.0",
            "metrics_with_chronic_regressions": ["goal_hit"],
            "compass_status": "CRITICAL",
            "total_regressions": 2,
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
            "overall_conformance_status": "healthy",
        }

        tile = summarize_conformance_and_budget_for_global_console(health, joint)
        signal = to_governance_signal_for_metrics(tile, joint)

        assert signal["status"] == "BLOCK"
        assert "METRIC_DRIFT_CRITICAL" in signal["blocking_rules"]

    def test_warn_produces_caution_signal(self):
        """Test that WARN status produces PROCEED_WITH_CAUTION signal."""
        compass = {
            "schema_version": "1.0.0",
            "metrics_with_chronic_regressions": [],
            "compass_status": "CAUTION",
            "total_regressions": 1,
        }

        budget_view = {
            "schema_version": "1.0.0",
            "uplift_ready": True,
            "blocking_slices": [],
            "status": "OK",
            "notes": [],
        }

        joint = build_metric_budget_joint_view(compass, budget_view)
        assert joint["joint_status"] == "WARN"

        health = {
            "weakest_metric": None,
            "weakest_metric_pass_rate": 1.0,
            "any_blockers": False,
            "overall_conformance_status": "healthy",
        }

        tile = summarize_conformance_and_budget_for_global_console(health, joint)
        signal = to_governance_signal_for_metrics(tile, joint)

        assert signal["status"] == "PROCEED_WITH_CAUTION"
        assert signal["metrics_ok"] is True
