"""
Tests for Budget ↔ Metric Integration & Uplift Readiness (A5 Phase III)

PHASE II — NOT USED IN PHASE I

Tests verify:
    1. Joint budget-metric view identifies problematic combinations
    2. Uplift readiness signal correctly classifies OK/WARN/BLOCK
    3. Global health summary aggregates trends correctly
    4. Read-only invariant maintained
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.budget_integration import (
    UpliftReadinessStatus,
    MetricLevel,
    SliceJointStatus,
    CRITICAL_UPLIFT_SLICES,
    build_budget_metric_joint_view,
    summarize_budget_for_uplift,
    summarize_budget_for_global_health,
    build_cross_experiment_budget_view,
    plan_uplift_runs,
    build_budget_director_panel,
    _parse_metric_status,
    _identify_flags,
)
from experiments.summarize_budget_usage import BudgetSummary
from experiments.budget_trends import (
    TrendDirection,
    TrendReport,
    SliceTrend,
    RunHealth,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def safe_budget_summary() -> BudgetSummary:
    """Budget summary with SAFE status."""
    return BudgetSummary(
        path="test/safe.jsonl",
        slice_name="slice_uplift_goal",
        mode="rfl",
        total_cycles=1000,
        budget_exhausted_count=5,  # 0.5% < 1%
        max_candidates_hit_count=950,
        timeout_abstentions_total=50,  # 0.05 avg < 0.1
        cycles_with_budget_field=1000,
    )


@pytest.fixture
def tight_budget_summary() -> BudgetSummary:
    """Budget summary with TIGHT status."""
    return BudgetSummary(
        path="test/tight.jsonl",
        slice_name="slice_uplift_sparse",
        mode="rfl",
        total_cycles=500,
        budget_exhausted_count=15,  # 3%
        max_candidates_hit_count=400,
        timeout_abstentions_total=200,  # 0.4 avg
        cycles_with_budget_field=500,
    )


@pytest.fixture
def starved_budget_summary() -> BudgetSummary:
    """Budget summary with STARVED status."""
    return BudgetSummary(
        path="test/starved.jsonl",
        slice_name="slice_uplift_tree",
        mode="rfl",
        total_cycles=200,
        budget_exhausted_count=20,  # 10%
        max_candidates_hit_count=100,
        timeout_abstentions_total=300,  # 1.5 avg
        cycles_with_budget_field=200,
    )


@pytest.fixture
def metric_snapshots_high() -> Dict[str, Dict[str, Any]]:
    """Metric snapshots with high conformance."""
    return {
        "slice_uplift_goal": {"level": "L3"},
        "slice_uplift_sparse": {"level": "L2"},
        "slice_uplift_tree": {"level": "L2"},
    }


@pytest.fixture
def metric_snapshots_low() -> Dict[str, Dict[str, Any]]:
    """Metric snapshots with low conformance."""
    return {
        "slice_uplift_goal": {"level": "L1"},
        "slice_uplift_sparse": {"level": "L0"},
        "slice_uplift_tree": {"status": "FAIL"},
    }


@pytest.fixture
def improving_trend_report() -> TrendReport:
    """Trend report showing improvement."""
    return TrendReport(
        inputs=["run1.json", "run2.json", "run3.json"],
        slices=[
            SliceTrend(
                slice_name="slice_uplift_goal",
                runs=[],
                trend=TrendDirection.IMPROVING,
                status_sequence=["STARVED", "TIGHT", "SAFE"],
            ),
            SliceTrend(
                slice_name="slice_uplift_sparse",
                runs=[],
                trend=TrendDirection.STABLE,
                status_sequence=["SAFE", "SAFE", "SAFE"],
            ),
        ],
        summary={"improving": 1, "stable": 1, "degrading": 0},
    )


@pytest.fixture
def degrading_trend_report() -> TrendReport:
    """Trend report showing degradation."""
    return TrendReport(
        inputs=["run1.json", "run2.json", "run3.json"],
        slices=[
            SliceTrend(
                slice_name="slice_uplift_goal",
                runs=[],
                trend=TrendDirection.DEGRADING,
                status_sequence=["SAFE", "TIGHT", "STARVED"],
            ),
            SliceTrend(
                slice_name="slice_uplift_sparse",
                runs=[],
                trend=TrendDirection.DEGRADING,
                status_sequence=["TIGHT", "STARVED", "STARVED"],
            ),
        ],
        summary={"improving": 0, "stable": 0, "degrading": 2},
    )


# =============================================================================
# Test: Metric Status Parsing
# =============================================================================


class TestMetricStatusParsing:
    """Tests for _parse_metric_status()."""

    def test_parse_level_format(self):
        """Test parsing {"level": "L2"} format."""
        assert _parse_metric_status({"level": "L2"}) == "L2"
        assert _parse_metric_status({"level": "L0"}) == "L0"
        assert _parse_metric_status({"level": "L3"}) == "L3"

    def test_parse_status_format(self):
        """Test parsing {"status": "PASS"} format."""
        assert _parse_metric_status({"status": "PASS"}) == "PASS"
        assert _parse_metric_status({"status": "FAIL"}) == "FAIL"

    def test_parse_nested_format(self):
        """Test parsing {"conformance": {"level": "L1"}} format."""
        assert _parse_metric_status({"conformance": {"level": "L1"}}) == "L1"

    def test_parse_none_returns_unknown(self):
        """Test None returns UNKNOWN."""
        assert _parse_metric_status(None) == "UNKNOWN"

    def test_parse_empty_returns_unknown(self):
        """Test empty dict returns UNKNOWN."""
        assert _parse_metric_status({}) == "UNKNOWN"


# =============================================================================
# Test: Flag Identification
# =============================================================================


class TestFlagIdentification:
    """Tests for _identify_flags()."""

    def test_starved_with_high_metrics(self):
        """Test STARVED + high metrics is flagged."""
        flags = _identify_flags("STARVED", "L3")
        assert "STARVED_WITH_HIGH_METRICS" in flags

    def test_starved_with_pass(self):
        """Test STARVED + PASS is flagged."""
        flags = _identify_flags("STARVED", "PASS")
        assert "STARVED_WITH_HIGH_METRICS" in flags

    def test_safe_with_weak_metrics(self):
        """Test SAFE + weak metrics is flagged."""
        flags = _identify_flags("SAFE", "L0")
        assert "SAFE_WITH_WEAK_METRICS" in flags
        
        flags = _identify_flags("SAFE", "FAIL")
        assert "SAFE_WITH_WEAK_METRICS" in flags

    def test_tight_with_high_metrics(self):
        """Test TIGHT + high metrics is flagged."""
        flags = _identify_flags("TIGHT", "L3")
        assert "TIGHT_WITH_HIGH_METRICS" in flags

    def test_safe_with_high_metrics_no_flag(self):
        """Test SAFE + high metrics has no flag."""
        flags = _identify_flags("SAFE", "L3")
        assert len(flags) == 0

    def test_starved_with_low_metrics_no_flag(self):
        """Test STARVED + low metrics has no special flag."""
        flags = _identify_flags("STARVED", "L0")
        assert "STARVED_WITH_HIGH_METRICS" not in flags


# =============================================================================
# Test: Joint View
# =============================================================================


class TestBudgetMetricJointView:
    """Tests for build_budget_metric_joint_view()."""

    def test_joint_view_basic(
        self,
        safe_budget_summary: BudgetSummary,
        metric_snapshots_high: Dict[str, Dict[str, Any]],
    ):
        """Test basic joint view construction."""
        result = build_budget_metric_joint_view([safe_budget_summary], metric_snapshots_high)
        
        assert "slices" in result
        assert "summary" in result
        assert "concerns" in result
        assert len(result["slices"]) == 1

    def test_joint_view_identifies_starved_with_high(
        self,
        starved_budget_summary: BudgetSummary,
        metric_snapshots_high: Dict[str, Dict[str, Any]],
    ):
        """Test joint view identifies STARVED + high metrics."""
        result = build_budget_metric_joint_view([starved_budget_summary], metric_snapshots_high)
        
        assert "slice_uplift_tree" in result["concerns"]["starved_with_high_metrics"]

    def test_joint_view_identifies_safe_with_weak(
        self,
        safe_budget_summary: BudgetSummary,
        metric_snapshots_low: Dict[str, Dict[str, Any]],
    ):
        """Test joint view identifies SAFE + weak metrics."""
        result = build_budget_metric_joint_view([safe_budget_summary], metric_snapshots_low)
        
        assert "slice_uplift_goal" in result["concerns"]["safe_with_weak_metrics"]

    def test_joint_view_budget_counts(
        self,
        safe_budget_summary: BudgetSummary,
        tight_budget_summary: BudgetSummary,
        starved_budget_summary: BudgetSummary,
    ):
        """Test summary budget counts."""
        summaries = [safe_budget_summary, tight_budget_summary, starved_budget_summary]
        result = build_budget_metric_joint_view(summaries, None)
        
        counts = result["summary"]["budget_counts"]
        assert counts["SAFE"] == 1
        assert counts["TIGHT"] == 1
        assert counts["STARVED"] == 1

    def test_joint_view_no_metrics(
        self,
        safe_budget_summary: BudgetSummary,
    ):
        """Test joint view works without metric snapshots."""
        result = build_budget_metric_joint_view([safe_budget_summary], None)
        
        assert result["slices"][0]["metric_status"] == "UNKNOWN"


# =============================================================================
# Test: Uplift Readiness
# =============================================================================


class TestUpliftReadiness:
    """Tests for summarize_budget_for_uplift()."""

    def test_uplift_ok_with_healthy_slices(
        self,
        improving_trend_report: TrendReport,
    ):
        """Test OK status when all critical slices are healthy."""
        # Build joint view with all SAFE slices
        joint = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "SAFE", "flags": []},
                {"slice_name": "slice_uplift_sparse", "budget_status": "SAFE", "flags": []},
            ],
        }
        
        result = summarize_budget_for_uplift(improving_trend_report, joint)
        
        assert result["status"] == "OK"
        assert result["budget_ready_for_uplift"] is True

    def test_uplift_warn_with_tight_slice(self):
        """Test WARN status when critical slice is TIGHT."""
        joint = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "TIGHT", "flags": []},
            ],
        }
        
        result = summarize_budget_for_uplift(None, joint)
        
        assert result["status"] == "WARN"
        assert result["budget_ready_for_uplift"] is True  # Still ready, just warning
        assert "slice_uplift_goal" in result["at_risk_slices"]

    def test_uplift_warn_with_starved_slice(self):
        """Test WARN status when critical slice is STARVED (no degrading trend)."""
        joint = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "STARVED", "flags": []},
            ],
        }
        
        result = summarize_budget_for_uplift(None, joint)
        
        assert result["status"] in ["WARN", "BLOCK"]
        assert "slice_uplift_goal" in result["at_risk_slices"]

    def test_uplift_block_with_starved_degrading(
        self,
        degrading_trend_report: TrendReport,
    ):
        """Test BLOCK status when critical slice is STARVED + DEGRADING."""
        joint = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "STARVED", "flags": []},
            ],
        }
        
        result = summarize_budget_for_uplift(degrading_trend_report, joint)
        
        assert result["status"] == "BLOCK"
        assert result["budget_ready_for_uplift"] is False

    def test_uplift_includes_critical_slices(self):
        """Test that critical slices are listed."""
        result = summarize_budget_for_uplift(None, None)
        
        assert "critical_slices_checked" in result
        for slice_name in CRITICAL_UPLIFT_SLICES:
            assert slice_name in result["critical_slices_checked"]

    def test_uplift_reasons_provided(self):
        """Test that reasons are provided."""
        joint = {
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "STARVED", "flags": []},
            ],
        }
        
        result = summarize_budget_for_uplift(None, joint)
        
        assert len(result["reasons"]) > 0


# =============================================================================
# Test: Global Health Summary
# =============================================================================


class TestGlobalHealthSummary:
    """Tests for summarize_budget_for_global_health()."""

    def test_global_health_improving(self, improving_trend_report: TrendReport):
        """Test global health with improving trend."""
        result = summarize_budget_for_global_health(improving_trend_report)
        
        # Should detect some improvement
        assert result["trend_counts"]["IMPROVING"] == 1
        assert result["total_slices"] == 2

    def test_global_health_degrading(self, degrading_trend_report: TrendReport):
        """Test global health with degrading trend."""
        result = summarize_budget_for_global_health(degrading_trend_report)
        
        assert result["trend_status"] == "DEGRADING"
        assert result["trend_counts"]["DEGRADING"] == 2

    def test_global_health_none_report(self):
        """Test global health with None report."""
        result = summarize_budget_for_global_health(None)
        
        assert result["trend_status"] == "UNKNOWN"
        assert result["health_score"] == 0

    def test_global_health_score_range(self, improving_trend_report: TrendReport):
        """Test health score is in valid range."""
        result = summarize_budget_for_global_health(improving_trend_report)
        
        assert 0 <= result["health_score"] <= 100

    def test_global_health_slice_counts(self, improving_trend_report: TrendReport):
        """Test slice counts are computed."""
        result = summarize_budget_for_global_health(improving_trend_report)
        
        assert "slice_counts" in result
        assert "SAFE" in result["slice_counts"]
        assert "TIGHT" in result["slice_counts"]
        assert "STARVED" in result["slice_counts"]

    def test_global_health_total_runs(self, improving_trend_report: TrendReport):
        """Test total runs is computed."""
        result = summarize_budget_for_global_health(improving_trend_report)
        
        assert result["total_runs"] == 3  # 3 inputs in fixture


# =============================================================================
# Test: SliceJointStatus
# =============================================================================


class TestSliceJointStatus:
    """Tests for SliceJointStatus dataclass."""

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all expected fields."""
        status = SliceJointStatus(
            slice_name="test_slice",
            budget_status="TIGHT",
            metric_status="L2",
            budget_metrics={"budget_exhausted_pct": 3.0},
            metric_details={"level": "L2"},
            flags=["TIGHT_WITH_HIGH_METRICS"],
        )
        d = status.to_dict()
        
        assert d["slice_name"] == "test_slice"
        assert d["budget_status"] == "TIGHT"
        assert d["metric_status"] == "L2"
        assert d["flags"] == ["TIGHT_WITH_HIGH_METRICS"]


# =============================================================================
# Test: Read-Only Invariant
# =============================================================================


class TestReadOnlyInvariant:
    """Tests verifying integration functions are read-only."""

    def test_joint_view_does_not_modify_summaries(
        self,
        safe_budget_summary: BudgetSummary,
    ):
        """Test joint view does not modify input summaries."""
        original_dict = safe_budget_summary.to_dict()
        
        _ = build_budget_metric_joint_view([safe_budget_summary], None)
        
        assert safe_budget_summary.to_dict() == original_dict

    def test_uplift_readiness_does_not_modify_trend(
        self,
        improving_trend_report: TrendReport,
    ):
        """Test uplift readiness does not modify trend report."""
        original_slices = [s.slice_name for s in improving_trend_report.slices]
        
        _ = summarize_budget_for_uplift(improving_trend_report, None)
        
        assert [s.slice_name for s in improving_trend_report.slices] == original_slices

    def test_global_health_does_not_modify_trend(
        self,
        improving_trend_report: TrendReport,
    ):
        """Test global health does not modify trend report."""
        original_inputs = improving_trend_report.inputs.copy()
        
        _ = summarize_budget_for_global_health(improving_trend_report)
        
        assert improving_trend_report.inputs == original_inputs


# =============================================================================
# PHASE IV: CROSS-EXPERIMENT BUDGET GOVERNANCE & UPLIFT SCHEDULING
# =============================================================================


@pytest.fixture
def sample_run_summaries() -> List[Dict[str, Any]]:
    """Sample run summaries for cross-experiment analysis."""
    return [
        {
            "run_id": "run1",
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "SAFE"},
                {"slice_name": "slice_uplift_sparse", "budget_status": "TIGHT"},
            ],
            "uplift_readiness": {"status": "OK", "at_risk_slices": []},
        },
        {
            "run_id": "run2",
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "SAFE"},
                {"slice_name": "slice_uplift_sparse", "budget_status": "STARVED"},
            ],
            "uplift_readiness": {"status": "WARN", "at_risk_slices": ["slice_uplift_sparse"]},
        },
        {
            "run_id": "run3",
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "SAFE"},
                {"slice_name": "slice_uplift_sparse", "budget_status": "STARVED"},
            ],
            "uplift_readiness": {"status": "BLOCK", "at_risk_slices": ["slice_uplift_sparse"]},
        },
    ]


@pytest.fixture
def frequently_starved_runs() -> List[Dict[str, Any]]:
    """Run summaries with a slice frequently starved."""
    return [
        {
            "run_id": f"run{i}",
            "slices": [
                {"slice_name": "slice_uplift_goal", "budget_status": "STARVED"},
            ],
            "uplift_readiness": {"status": "BLOCK", "at_risk_slices": ["slice_uplift_goal"]},
        }
        for i in range(6)  # 6 runs, all STARVED = >50%
    ]


class TestCrossExperimentBudgetView:
    """Tests for build_cross_experiment_budget_view()."""

    def test_cross_view_basic(self, sample_run_summaries: List[Dict[str, Any]]):
        """Test basic cross-experiment view construction."""
        result = build_cross_experiment_budget_view(sample_run_summaries)
        
        assert result["experiments_count"] == 3
        assert "per_slice" in result
        assert "slices_frequently_starved" in result

    def test_cross_view_classification_distribution(
        self,
        sample_run_summaries: List[Dict[str, Any]],
    ):
        """Test classification distribution is computed correctly."""
        result = build_cross_experiment_budget_view(sample_run_summaries)
        
        goal_dist = result["per_slice"]["slice_uplift_goal"]["classification_distribution"]
        assert goal_dist["SAFE"] == 3
        assert goal_dist["TIGHT"] == 0
        assert goal_dist["STARVED"] == 0

    def test_cross_view_identifies_frequently_starved(
        self,
        frequently_starved_runs: List[Dict[str, Any]],
    ):
        """Test identifies slices frequently starved."""
        result = build_cross_experiment_budget_view(frequently_starved_runs)
        
        assert "slice_uplift_goal" in result["slices_frequently_starved"]

    def test_cross_view_block_count(self, sample_run_summaries: List[Dict[str, Any]]):
        """Test block count is computed."""
        result = build_cross_experiment_budget_view(sample_run_summaries)
        
        assert result["global_block_count"] == 1  # run3 has BLOCK

    def test_cross_view_per_slice_block_count(
        self,
        sample_run_summaries: List[Dict[str, Any]],
    ):
        """Test per-slice block count."""
        result = build_cross_experiment_budget_view(sample_run_summaries)
        
        sparse_block = result["per_slice"]["slice_uplift_sparse"]["block_count"]
        # slice_uplift_sparse is in at_risk_slices for run3 (BLOCK)
        assert sparse_block >= 1

    def test_cross_view_starved_percentage(
        self,
        sample_run_summaries: List[Dict[str, Any]],
    ):
        """Test starved percentage is computed."""
        result = build_cross_experiment_budget_view(sample_run_summaries)
        
        sparse_starved_pct = result["per_slice"]["slice_uplift_sparse"]["starved_percentage"]
        # 2 out of 3 runs = 66.7%
        assert sparse_starved_pct == pytest.approx(66.7, abs=0.1)


class TestUpliftPlanning:
    """Tests for plan_uplift_runs()."""

    def test_plan_identifies_ready_slices(self):
        """Test identifies slices ready for uplift."""
        cross_view = {
            "experiments_count": 10,
            "per_slice": {
                "slice_uplift_goal": {
                    "classification_distribution": {"SAFE": 8, "TIGHT": 2, "STARVED": 0},
                    "total_runs": 10,
                },
            },
            "global_block_count": 0,
        }
        
        result = plan_uplift_runs(cross_view)
        
        assert "slice_uplift_goal" in result["slices_ready_for_uplift"]

    def test_plan_identifies_tuning_needed(self):
        """Test identifies slices needing budget tuning."""
        cross_view = {
            "experiments_count": 10,
            "per_slice": {
                "slice_uplift_sparse": {
                    "classification_distribution": {"SAFE": 2, "TIGHT": 2, "STARVED": 6},
                    "total_runs": 10,
                },
            },
            "global_block_count": 5,
        }
        
        result = plan_uplift_runs(cross_view)
        
        assert "slice_uplift_sparse" in result["slices_needing_budget_tuning"]

    def test_plan_generates_recommendations(self):
        """Test generates scheduling recommendations."""
        cross_view = {
            "experiments_count": 10,
            "per_slice": {
                "slice_uplift_goal": {
                    "classification_distribution": {"SAFE": 8, "TIGHT": 2, "STARVED": 0},
                    "total_runs": 10,
                },
            },
            "global_block_count": 0,
        }
        
        result = plan_uplift_runs(cross_view)
        
        assert len(result["schedule_recommendations"]) > 0
        assert any("slice_uplift_goal" in rec for rec in result["schedule_recommendations"])

    def test_plan_handles_high_block_rate(self):
        """Test handles high block rate with global recommendation."""
        cross_view = {
            "experiments_count": 10,
            "per_slice": {},
            "global_block_count": 5,  # 50% blocked
        }
        
        result = plan_uplift_runs(cross_view)
        
        assert any("BLOCK" in rec or "block" in rec.lower() for rec in result["schedule_recommendations"])

    def test_plan_marginal_slices(self):
        """Test handles marginal slices (mostly TIGHT)."""
        cross_view = {
            "experiments_count": 10,
            "per_slice": {
                "slice_test": {
                    "classification_distribution": {"SAFE": 2, "TIGHT": 8, "STARVED": 0},
                    "total_runs": 10,
                },
            },
            "global_block_count": 0,
        }
        
        result = plan_uplift_runs(cross_view)
        
        # Should have recommendation for marginal slice
        assert any("slice_test" in rec for rec in result["schedule_recommendations"])


class TestDirectorPanel:
    """Tests for build_budget_director_panel()."""

    def test_panel_green_status(self):
        """Test GREEN status when all healthy."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": [],
            "global_block_count": 1,  # <20%
        }
        uplift_plan = {
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_needing_budget_tuning": [],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert result["budget_status_light"] == "GREEN"

    def test_panel_yellow_status(self):
        """Test YELLOW status with some risk."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": ["slice_non_critical"],  # Not in CRITICAL_UPLIFT_SLICES
            "global_block_count": 3,  # 30% = YELLOW range
        }
        uplift_plan = {
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_needing_budget_tuning": ["slice_non_critical"],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert result["budget_status_light"] == "YELLOW"

    def test_panel_red_status_critical_starved(self):
        """Test RED status when critical slice frequently starved."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": ["slice_uplift_goal"],  # Critical slice
            "global_block_count": 2,
        }
        uplift_plan = {
            "slices_ready_for_uplift": [],
            "slices_needing_budget_tuning": ["slice_uplift_goal"],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert result["budget_status_light"] == "RED"

    def test_panel_red_status_high_block_rate(self):
        """Test RED status with high block rate."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": [],
            "global_block_count": 6,  # 60% > 50%
        }
        uplift_plan = {
            "slices_ready_for_uplift": [],
            "slices_needing_budget_tuning": [],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert result["budget_status_light"] == "RED"

    def test_panel_critical_slices_at_risk(self):
        """Test identifies critical slices at risk."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": ["slice_uplift_goal"],
            "global_block_count": 2,
        }
        uplift_plan = {
            "slices_ready_for_uplift": [],
            "slices_needing_budget_tuning": ["slice_uplift_goal"],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert "slice_uplift_goal" in result["critical_slices_at_risk"]

    def test_panel_ready_slices(self):
        """Test includes ready slices."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": [],
            "global_block_count": 0,
        }
        uplift_plan = {
            "slices_ready_for_uplift": ["slice_uplift_goal", "slice_uplift_sparse"],
            "slices_needing_budget_tuning": [],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert len(result["ready_slices"]) == 2
        assert "slice_uplift_goal" in result["ready_slices"]

    def test_panel_message_provided(self):
        """Test message is provided."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": [],
            "global_block_count": 1,
        }
        uplift_plan = {
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_needing_budget_tuning": [],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert "message" in result
        assert len(result["message"]) > 0

    def test_panel_summary_included(self):
        """Test summary statistics are included."""
        cross_view = {
            "experiments_count": 10,
            "slices_frequently_starved": ["slice_test"],
            "global_block_count": 2,
        }
        uplift_plan = {
            "slices_ready_for_uplift": ["slice_uplift_goal"],
            "slices_needing_budget_tuning": ["slice_test"],
        }
        
        result = build_budget_director_panel(cross_view, uplift_plan)
        
        assert "summary" in result
        assert "experiments_analyzed" in result["summary"]
        assert "block_percentage" in result["summary"]


class TestPhaseIVReadOnlyInvariant:
    """Tests verifying Phase IV functions are read-only."""

    def test_cross_view_does_not_modify_inputs(self, sample_run_summaries: List[Dict[str, Any]]):
        """Test cross view does not modify input run summaries."""
        original = [r.copy() for r in sample_run_summaries]
        
        _ = build_cross_experiment_budget_view(sample_run_summaries)
        
        assert sample_run_summaries == original

    def test_plan_does_not_modify_cross_view(self):
        """Test plan does not modify cross view."""
        cross_view = {
            "experiments_count": 10,
            "per_slice": {"test": {"classification_distribution": {}, "total_runs": 10}},
            "global_block_count": 0,
        }
        original = cross_view.copy()
        
        _ = plan_uplift_runs(cross_view)
        
        assert cross_view == original

    def test_panel_does_not_modify_inputs(self):
        """Test panel does not modify inputs."""
        cross_view = {"experiments_count": 10, "slices_frequently_starved": [], "global_block_count": 0}
        uplift_plan = {"slices_ready_for_uplift": [], "slices_needing_budget_tuning": []}
        original_cross = cross_view.copy()
        original_plan = uplift_plan.copy()
        
        _ = build_budget_director_panel(cross_view, uplift_plan)
        
        assert cross_view == original_cross
        assert uplift_plan == original_plan


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

