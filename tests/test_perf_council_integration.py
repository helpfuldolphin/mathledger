"""Tests for performance governance integration with Uplift Council.

Tests cover:
- Perf joint view adaptation for council
- Council status determination with perf data
- Perf BLOCK on critical slice → council BLOCK
- Perf MEDIUM on critical slice → council WARN
"""

import pytest
from experiments.uplift_council import (
    build_uplift_council_view,
    CouncilStatus,
    CRITICAL_UPLIFT_SLICES,
)
from experiments.verify_perf_equivalence import (
    build_perf_joint_governance_view,
    adapt_perf_joint_view_for_council,
)


class TestPerfCouncilIntegration:
    """Test performance governance integration with Uplift Council."""

    def test_perf_block_on_critical_slice_council_block(self):
        """Perf BLOCK on a critical slice should result in council_status = BLOCK."""
        # Create perf joint view with BLOCK on critical slice
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [
                {
                    "run_id": "run1",
                    "status": "FAIL",
                    "slice_name": "slice_uplift_goal",  # Critical slice
                },
            ],
            "release_risk_level": "HIGH",
        }
        budget_trend = {"budget_risk": "HIGH"}
        metric_conformance = {"status": "OK"}

        perf_joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )

        # Build council view with perf_joint_view
        council_view = build_uplift_council_view(
            budget_cross_view=None,
            perf_trend=None,
            metric_conformance=None,
            perf_joint_view=perf_joint_view,
        )

        assert council_view["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in council_view["slices_blocked_by_perf"]

    def test_perf_medium_on_critical_slice_council_warn(self):
        """Perf MEDIUM on critical slice with others OK should result in council_status = WARN."""
        # Create perf joint view with MEDIUM risk on critical slice
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [
                {
                    "run_id": "run1",
                    "status": "WARN",
                    "slice_name": "slice_uplift_goal",  # Critical slice
                },
            ],
            "release_risk_level": "MEDIUM",
        }
        budget_trend = {"budget_risk": "LOW"}  # Budget OK
        metric_conformance = {"status": "OK"}  # Metrics OK

        perf_joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )

        # Build council view
        council_view = build_uplift_council_view(
            budget_cross_view=None,
            perf_trend=None,
            metric_conformance=None,
            perf_joint_view=perf_joint_view,
        )

        # Should be WARN, not BLOCK, since budget and metrics are OK
        assert council_view["council_status"] == "WARN"
        # Critical slice should show up in per_slice_status with WARN
        assert (
            council_view["per_slice_status"]["slice_uplift_goal"]["perf"] == "WARN"
        )

    def test_perf_ok_all_slices_council_ok(self):
        """Perf OK on all slices should result in council_status = OK."""
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [
                {"run_id": "run1", "status": "PASS", "slice_name": "slice_uplift_goal"},
            ],
            "release_risk_level": "LOW",
        }
        budget_trend = {"budget_risk": "LOW"}
        metric_conformance = {"status": "OK"}

        perf_joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )

        council_view = build_uplift_council_view(
            budget_cross_view=None,
            perf_trend=None,
            metric_conformance=None,
            perf_joint_view=perf_joint_view,
        )

        assert council_view["council_status"] == "OK"

    def test_adapt_perf_joint_view_for_council(self):
        """Test adapter function converts perf_joint_view correctly."""
        perf_joint_view = {
            "perf_risk": "HIGH",
            "budget_risk": "LOW",
            "slices_with_regressions": ["slice_a", "all_slices"],
            "slices_blocking_uplift": ["slice_b"],
            "summary_note": "test",
        }

        adapted = adapt_perf_joint_view_for_council(perf_joint_view)

        assert adapted["perf_risk"] == "HIGH"
        assert "slice_a" in adapted["slices_with_regressions"]
        assert "all_slices" not in adapted["slices_with_regressions"]
        assert "slice_b" in adapted["slices_blocking_uplift"]

    def test_council_with_perf_and_budget(self):
        """Test council with both perf and budget dimensions."""
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [
                {"run_id": "run1", "status": "WARN", "slice_name": "slice_a"},
            ],
            "release_risk_level": "MEDIUM",
        }
        budget_trend = {"budget_risk": "LOW"}
        metric_conformance = {"status": "OK"}

        perf_joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )

        budget_cross_view = {
            "slices": [
                {"slice_name": "slice_a", "health_status": "SAFE"},
            ]
        }

        council_view = build_uplift_council_view(
            budget_cross_view=budget_cross_view,
            perf_trend=None,
            metric_conformance=None,
            perf_joint_view=perf_joint_view,
        )

        # Should aggregate worst-case: WARN from perf
        assert council_view["council_status"] == "WARN"
        assert "slice_a" in council_view["per_slice_status"]
        assert council_view["per_slice_status"]["slice_a"]["perf"] == "WARN"
        assert council_view["per_slice_status"]["slice_a"]["budget"] == "OK"

    def test_council_perf_joint_view_precedence(self):
        """Test that perf_joint_view takes precedence over perf_trend."""
        perf_joint_view = {
            "perf_risk": "HIGH",
            "budget_risk": "LOW",
            "slices_with_regressions": ["slice_a"],
            "slices_blocking_uplift": [],
            "summary_note": "test",
        }

        # Legacy perf_trend format (should be ignored)
        perf_trend = {
            "slices": [
                {"slice_name": "slice_a", "status": "OK"},
            ]
        }

        council_view = build_uplift_council_view(
            budget_cross_view=None,
            perf_trend=perf_trend,
            metric_conformance=None,
            perf_joint_view=perf_joint_view,
        )

        # Should use perf_joint_view (WARN from slices_with_regressions)
        assert council_view["council_status"] == "WARN"
        assert "slice_a" in council_view["per_slice_status"]
        assert council_view["per_slice_status"]["slice_a"]["perf"] == "WARN"

