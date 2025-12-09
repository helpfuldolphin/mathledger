"""
Tests for Uplift Council Multi-Dimensional Decision Making (A5)

PHASE II — NOT USED IN PHASE I

Tests verify:
    1. Council aggregates budget, performance, and metrics correctly
    2. BLOCK status when critical slices blocked by any dimension
    3. WARN status when non-critical slices at risk
    4. OK status when all dimensions healthy
    5. Director panel provides unified view
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.budget_integration import (
    UpliftCouncilStatus,
    CRITICAL_UPLIFT_SLICES,
    build_uplift_council_view,
    build_uplift_director_panel,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def healthy_budget_cross_view() -> Dict[str, Any]:
    """Budget cross-view with all slices healthy."""
    return {
        "experiments_count": 10,
        "slices_frequently_starved": [],
        "per_slice": {
            "slice_uplift_goal": {
                "classification_distribution": {"SAFE": 9, "TIGHT": 1, "STARVED": 0},
                "total_runs": 10,
            },
            "slice_uplift_sparse": {
                "classification_distribution": {"SAFE": 8, "TIGHT": 2, "STARVED": 0},
                "total_runs": 10,
            },
        },
        "global_block_count": 0,
    }


@pytest.fixture
def starved_budget_cross_view() -> Dict[str, Any]:
    """Budget cross-view with critical slice frequently starved."""
    return {
        "experiments_count": 10,
        "slices_frequently_starved": ["slice_uplift_goal"],
        "per_slice": {
            "slice_uplift_goal": {
                "classification_distribution": {"SAFE": 2, "TIGHT": 2, "STARVED": 6},
                "total_runs": 10,
            },
        },
        "global_block_count": 5,
    }


@pytest.fixture
def healthy_perf_trend() -> Dict[str, Any]:
    """Performance trend with all slices OK."""
    return {
        "slices": [
            {"slice_name": "slice_uplift_goal", "status": "OK"},
            {"slice_name": "slice_uplift_sparse", "status": "OK"},
        ],
    }


@pytest.fixture
def blocked_perf_trend() -> Dict[str, Any]:
    """Performance trend with critical slice blocked."""
    return {
        "slices": [
            {"slice_name": "slice_uplift_goal", "status": "BLOCK"},
            {"slice_name": "slice_uplift_sparse", "status": "OK"},
        ],
    }


@pytest.fixture
def ready_metric_conformance() -> Dict[str, Any]:
    """Metric conformance with all slices ready."""
    return {
        "slices": [
            {"slice_name": "slice_uplift_goal", "ready": True},
            {"slice_name": "slice_uplift_sparse", "ready": True},
        ],
    }


@pytest.fixture
def not_ready_metric_conformance() -> Dict[str, Any]:
    """Metric conformance with critical slice not ready."""
    return {
        "slices": [
            {"slice_name": "slice_uplift_goal", "ready": False},
            {"slice_name": "slice_uplift_sparse", "ready": True},
        ],
    }


# =============================================================================
# Test: Uplift Council View
# =============================================================================


class TestUpliftCouncilView:
    """Tests for build_uplift_council_view()."""

    def test_council_all_dimensions_ok(
        self,
        healthy_budget_cross_view: Dict[str, Any],
        healthy_perf_trend: Dict[str, Any],
        ready_metric_conformance: Dict[str, Any],
    ):
        """Test OK status when all dimensions are healthy."""
        result = build_uplift_council_view(
            healthy_budget_cross_view,
            healthy_perf_trend,
            ready_metric_conformance,
        )
        
        assert result["council_status"] == "OK"
        assert len(result["slices_ready_for_uplift"]) > 0

    def test_council_blocked_by_budget(
        self,
        starved_budget_cross_view: Dict[str, Any],
        healthy_perf_trend: Dict[str, Any],
        ready_metric_conformance: Dict[str, Any],
    ):
        """Test BLOCK status when critical slice blocked by budget."""
        result = build_uplift_council_view(
            starved_budget_cross_view,
            healthy_perf_trend,
            ready_metric_conformance,
        )
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_budget"]
        assert "slice_uplift_goal" in result["critical_slices_blocked"]

    def test_council_blocked_by_perf(
        self,
        healthy_budget_cross_view: Dict[str, Any],
        blocked_perf_trend: Dict[str, Any],
        ready_metric_conformance: Dict[str, Any],
    ):
        """Test BLOCK status when critical slice blocked by performance."""
        result = build_uplift_council_view(
            healthy_budget_cross_view,
            blocked_perf_trend,
            ready_metric_conformance,
        )
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_perf"]
        assert "slice_uplift_goal" in result["critical_slices_blocked"]

    def test_council_blocked_by_metrics(
        self,
        healthy_budget_cross_view: Dict[str, Any],
        healthy_perf_trend: Dict[str, Any],
        not_ready_metric_conformance: Dict[str, Any],
    ):
        """Test BLOCK status when critical slice blocked by metrics."""
        result = build_uplift_council_view(
            healthy_budget_cross_view,
            healthy_perf_trend,
            not_ready_metric_conformance,
        )
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_metrics"]
        assert "slice_uplift_goal" in result["critical_slices_blocked"]

    def test_council_warn_non_critical_blocked(self):
        """Test WARN status when non-critical slice blocked."""
        budget_view = {
            "experiments_count": 10,
            "slices_frequently_starved": ["slice_non_critical"],
            "per_slice": {
                "slice_non_critical": {
                    "classification_distribution": {"SAFE": 2, "TIGHT": 2, "STARVED": 6},
                    "total_runs": 10,
                },
            },
            "global_block_count": 2,
        }
        
        result = build_uplift_council_view(budget_view, None, None)
        
        assert result["council_status"] == "WARN"
        assert "slice_non_critical" in result["slices_blocked_by_budget"]
        assert len(result["critical_slices_blocked"]) == 0

    def test_council_budget_ok_perf_critical_blocked(
        self,
        healthy_budget_cross_view: Dict[str, Any],
        blocked_perf_trend: Dict[str, Any],
        ready_metric_conformance: Dict[str, Any],
    ):
        """Test: Budget OK but perf critical → blocked_by_perf."""
        result = build_uplift_council_view(
            healthy_budget_cross_view,
            blocked_perf_trend,
            ready_metric_conformance,
        )
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_perf"]
        assert len(result["slices_blocked_by_budget"]) == 0

    def test_council_budget_high_risk_metrics_not_ready_block(
        self,
        starved_budget_cross_view: Dict[str, Any],
        healthy_perf_trend: Dict[str, Any],
        not_ready_metric_conformance: Dict[str, Any],
    ):
        """Test: Budget HIGH risk and metrics NOT ready → BLOCK council."""
        result = build_uplift_council_view(
            starved_budget_cross_view,
            healthy_perf_trend,
            not_ready_metric_conformance,
        )
        
        assert result["council_status"] == "BLOCK"
        assert "slice_uplift_goal" in result["slices_blocked_by_budget"]
        assert "slice_uplift_goal" in result["slices_blocked_by_metrics"]
        assert "slice_uplift_goal" in result["critical_slices_blocked"]

    def test_council_all_three_dimensions_ok(
        self,
        healthy_budget_cross_view: Dict[str, Any],
        healthy_perf_trend: Dict[str, Any],
        ready_metric_conformance: Dict[str, Any],
    ):
        """Test: All three dimensions OK → OK council."""
        result = build_uplift_council_view(
            healthy_budget_cross_view,
            healthy_perf_trend,
            ready_metric_conformance,
        )
        
        assert result["council_status"] == "OK"
        assert len(result["slices_ready_for_uplift"]) >= 2
        assert len(result["slices_blocked_by_budget"]) == 0
        assert len(result["slices_blocked_by_perf"]) == 0
        assert len(result["slices_blocked_by_metrics"]) == 0

    def test_council_per_slice_status(self, healthy_budget_cross_view: Dict[str, Any]):
        """Test per-slice status breakdown."""
        result = build_uplift_council_view(healthy_budget_cross_view, None, None)
        
        assert "per_slice" in result
        assert "slice_uplift_goal" in result["per_slice"]
        slice_status = result["per_slice"]["slice_uplift_goal"]
        assert "budget" in slice_status
        assert "perf" in slice_status
        assert "metrics" in slice_status
        assert "overall" in slice_status

    def test_council_handles_missing_dimensions(self, healthy_budget_cross_view: Dict[str, Any]):
        """Test council handles missing perf/metrics data."""
        result = build_uplift_council_view(healthy_budget_cross_view, None, None)
        
        # Should default to OK for missing dimensions
        assert result["council_status"] in ["OK", "WARN", "BLOCK"]
        assert "per_slice" in result


# =============================================================================
# Test: Uplift Director Panel
# =============================================================================


class TestUpliftDirectorPanel:
    """Tests for build_uplift_director_panel()."""

    def test_panel_green_status(self):
        """Test GREEN status light for OK council."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal", "slice_uplift_sparse"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
            "critical_slices_blocked": [],
        }
        
        result = build_uplift_director_panel(council_view)
        
        assert result["status_light"] == "GREEN"
        assert "ready" in result["headline"].lower()

    def test_panel_yellow_status(self):
        """Test YELLOW status light for WARN council."""
        council_view = {
            "council_status": "WARN",
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_blocked_by_budget": ["slice_non_critical"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
            "critical_slices_blocked": [],
        }
        
        result = build_uplift_director_panel(council_view)
        
        assert result["status_light"] == "YELLOW"
        assert "caution" in result["headline"].lower()

    def test_panel_red_status(self):
        """Test RED status light for BLOCK council."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": [],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
            "critical_slices_blocked": ["slice_uplift_goal"],
        }
        
        result = build_uplift_director_panel(council_view)
        
        assert result["status_light"] == "RED"
        assert "blocked" in result["headline"].lower()

    def test_panel_ready_slices(self):
        """Test ready slices are included."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice_uplift_goal", "slice_uplift_sparse"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
            "critical_slices_blocked": [],
        }
        
        result = build_uplift_director_panel(council_view)
        
        assert len(result["ready_slices"]) == 2
        assert "slice_uplift_goal" in result["ready_slices"]

    def test_panel_blocked_slices(self):
        """Test blocked slices are included."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": [],
            "slices_blocked_by_budget": ["slice_uplift_goal"],
            "slices_blocked_by_perf": ["slice_uplift_sparse"],
            "slices_blocked_by_metrics": [],
            "critical_slices_blocked": ["slice_uplift_goal"],
        }
        
        result = build_uplift_director_panel(council_view)
        
        assert len(result["blocked_slices"]) == 2
        assert "slice_uplift_goal" in result["blocked_slices"]
        assert "slice_uplift_sparse" in result["blocked_slices"]

    def test_panel_blocked_breakdown(self):
        """Test blocked breakdown shows dimension counts."""
        council_view = {
            "council_status": "BLOCK",
            "slices_ready_for_uplift": [],
            "slices_blocked_by_budget": ["slice1"],
            "slices_blocked_by_perf": ["slice2"],
            "slices_blocked_by_metrics": ["slice3"],
            "critical_slices_blocked": ["slice1"],
        }
        
        result = build_uplift_director_panel(council_view)
        
        assert "blocked_breakdown" in result
        assert len(result["blocked_breakdown"]) == 3
        assert any("budget" in b for b in result["blocked_breakdown"])
        assert any("performance" in b for b in result["blocked_breakdown"])
        assert any("metrics" in b for b in result["blocked_breakdown"])

    def test_panel_summary_included(self):
        """Test summary statistics are included."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice1"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
            "critical_slices_blocked": [],
            "summary": {"total_slices": 2, "ready_count": 1},
        }
        
        result = build_uplift_director_panel(council_view)
        
        assert "summary" in result
        assert "council_status" in result["summary"]
        assert "ready_count" in result["summary"]
        assert "blocked_count" in result["summary"]


# =============================================================================
# Test: Read-Only Invariant
# =============================================================================


class TestUpliftCouncilReadOnlyInvariant:
    """Tests verifying council functions are read-only."""

    def test_council_view_does_not_modify_inputs(
        self,
        healthy_budget_cross_view: Dict[str, Any],
    ):
        """Test council view does not modify input dictionaries."""
        original = json.dumps(healthy_budget_cross_view, sort_keys=True)
        
        _ = build_uplift_council_view(healthy_budget_cross_view, None, None)
        
        assert json.dumps(healthy_budget_cross_view, sort_keys=True) == original

    def test_director_panel_does_not_modify_council_view(self):
        """Test director panel does not modify council view."""
        council_view = {
            "council_status": "OK",
            "slices_ready_for_uplift": ["slice1"],
            "slices_blocked_by_budget": [],
            "slices_blocked_by_perf": [],
            "slices_blocked_by_metrics": [],
            "critical_slices_blocked": [],
        }
        original = json.dumps(council_view, sort_keys=True)
        
        _ = build_uplift_director_panel(council_view)
        
        assert json.dumps(council_view, sort_keys=True) == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

