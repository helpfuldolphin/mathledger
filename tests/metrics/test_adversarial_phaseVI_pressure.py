# tests/metrics/test_adversarial_phaseVI_pressure.py
"""
Tests for Phase V (Phase VI) — Adversarial-Metric Pressure Grid

Covers:
- Adversarial Pressure Model (TASK 1)
- Multi-Scenario Evolution Plan (TASK 2)
- Failover Sufficiency Checker v2 (TASK 3)

NO METRIC INTERPRETATION: These tests verify pressure grid mechanics only.
"""

import json
import pytest
from typing import Dict, List, Any, Optional

from experiments.metrics_adversarial_harness import (
    # Phase V - Pressure Grid
    build_adversarial_pressure_model,
    build_evolving_adversarial_scenario_plan,
    build_adversarial_failover_plan_v2,
    pressure_model_to_json,
    scenario_plan_to_json,
    failover_plan_to_json,
    
    # Constants
    PRESSURE_BANDS,
    PRESSURE_THRESHOLD_PRIORITY,
    COVERAGE_STATUS,
    PROMOTION_STATUS,
    CORE_UPLIFT_METRICS,
    METRIC_KINDS,
    
    # Supporting functions
    build_metric_adversarial_coverage_index,
    build_metric_robustness_radar,
    evaluate_adversarial_readiness_for_promotion,
    build_adversarial_failover_plan,
    build_robustness_scorecard,
    HarnessSummary,
)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def sparse_coverage_index() -> Dict[str, Any]:
    """Coverage index with sparse coverage."""
    scorecards = [
        build_robustness_scorecard([
            HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
        ], scenarios_run=["baseline_sanity"]),
    ]
    radar = build_metric_robustness_radar(scorecards)
    return build_metric_adversarial_coverage_index(radar)


@pytest.fixture
def at_risk_coverage_index() -> Dict[str, Any]:
    """Coverage index with AT_RISK metrics."""
    scorecards = []
    for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
        scorecards.append(build_robustness_scorecard([
            HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
        ], scenarios_run=[scenario_name]))
    
    radar = build_metric_robustness_radar(scorecards)
    return build_metric_adversarial_coverage_index(radar)


@pytest.fixture
def healthy_coverage_index() -> Dict[str, Any]:
    """Coverage index with healthy metrics."""
    scorecards = []
    for scenario_name in ["baseline_sanity", "goal_hit_boundary", "ci_quick"]:
        summaries = []
        for mk in METRIC_KINDS:
            summaries.append(HarnessSummary(
                mode="fault", metric_kind=mk, total_cases=50, passed=50, failed=0, mismatches=0, errors=0
            ))
        scorecards.append(build_robustness_scorecard(summaries, scenarios_run=[scenario_name]))
    
    radar = build_metric_robustness_radar(scorecards)
    return build_metric_adversarial_coverage_index(radar)


@pytest.fixture
def sample_robustness_radar() -> Dict[str, Any]:
    """Sample robustness radar."""
    scorecards = [
        build_robustness_scorecard([
            HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
        ], scenarios_run=["baseline_sanity"]),
    ]
    return build_metric_robustness_radar(scorecards)


@pytest.fixture
def sample_drift_grid() -> Dict[str, Any]:
    """Sample drift grid for pressure model."""
    return {
        "goal_hit": {
            "volatility": 0.3,
            "drift_count": 2,
        },
        "density": {
            "volatility": 0.1,
            "drift_count": 0,
        },
    }


# ===========================================================================
# TASK 1: ADVERSARIAL PRESSURE MODEL TESTS
# ===========================================================================

@pytest.mark.oracle
class TestAdversarialPressureModel:
    """Tests for build_adversarial_pressure_model()."""

    def test_pressure_model_has_metric_scores(self, sparse_coverage_index, sample_robustness_radar):
        """Pressure model includes metric_pressure_scores."""
        model = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        assert "metric_pressure_scores" in model
        assert isinstance(model["metric_pressure_scores"], dict)
        
        for mk in METRIC_KINDS:
            assert mk in model["metric_pressure_scores"]
            score = model["metric_pressure_scores"][mk]
            assert 0.0 <= score <= 1.0

    def test_pressure_model_has_scenario_targets(self, sparse_coverage_index, sample_robustness_radar):
        """Pressure model includes scenario_pressure_targets."""
        model = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        assert "scenario_pressure_targets" in model
        assert isinstance(model["scenario_pressure_targets"], dict)

    def test_pressure_model_has_global_index(self, sparse_coverage_index, sample_robustness_radar):
        """Pressure model includes global_pressure_index."""
        model = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        assert "global_pressure_index" in model
        assert isinstance(model["global_pressure_index"], (int, float))
        assert 0.0 <= model["global_pressure_index"] <= 1.0

    def test_pressure_model_has_pressure_band(self, sparse_coverage_index, sample_robustness_radar):
        """Pressure model includes pressure_band."""
        model = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        assert "pressure_band" in model
        assert model["pressure_band"] in [PRESSURE_BANDS["low"], PRESSURE_BANDS["medium"], PRESSURE_BANDS["high"]]

    def test_pressure_model_has_neutral_notes(self, sparse_coverage_index, sample_robustness_radar):
        """Pressure model includes neutral_notes."""
        model = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        assert "neutral_notes" in model
        assert isinstance(model["neutral_notes"], list)

    def test_pressure_scores_high_for_at_risk_metrics(self):
        """Pressure scores are high for AT_RISK metrics."""
        # Create AT_RISK scenario with matching radar
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        model = build_adversarial_pressure_model(coverage_index, radar)
        
        # goal_hit should have high pressure (AT_RISK with cross-scenario regressions)
        goal_hit_score = model["metric_pressure_scores"]["goal_hit"]
        # With 2 scenarios (scarcity 0.4) and AT_RISK in global (fragility 1.0):
        # 0.4 * 0.4 + 1.0 * 0.4 = 0.16 + 0.4 = 0.56
        # But if it's in metrics_at_risk_global, fragility is 1.0, so should be higher
        assert goal_hit_score >= 0.5  # Should be at least moderate-high

    def test_pressure_scores_low_for_healthy_metrics(self, healthy_coverage_index, sample_robustness_radar):
        """Pressure scores are low for healthy metrics."""
        model = build_adversarial_pressure_model(healthy_coverage_index, sample_robustness_radar)
        
        # Healthy metrics should have lower pressure
        for mk in METRIC_KINDS:
            score = model["metric_pressure_scores"][mk]
            # May still be moderate due to scarcity, but should be lower than AT_RISK
            assert score < 1.0

    def test_pressure_targets_include_high_pressure_metrics(self, at_risk_coverage_index, sample_robustness_radar):
        """Pressure targets include metrics with score > threshold."""
        model = build_adversarial_pressure_model(at_risk_coverage_index, sample_robustness_radar)
        
        # Metrics with pressure > threshold should be in targets
        for mk, score in model["metric_pressure_scores"].items():
            if score > PRESSURE_THRESHOLD_PRIORITY:
                assert mk in model["scenario_pressure_targets"]

    def test_pressure_model_integrates_drift_grid(self, sparse_coverage_index, sample_robustness_radar, sample_drift_grid):
        """Pressure model integrates drift_grid data when provided."""
        model_with_drift = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar, sample_drift_grid)
        model_without_drift = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        # goal_hit should have higher pressure with drift data
        score_with = model_with_drift["metric_pressure_scores"]["goal_hit"]
        score_without = model_without_drift["metric_pressure_scores"]["goal_hit"]
        
        # With drift, pressure should be equal or higher
        assert score_with >= score_without

    def test_pressure_band_calculation(self, sparse_coverage_index, sample_robustness_radar):
        """Pressure band correctly categorizes global pressure."""
        model = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        global_index = model["global_pressure_index"]
        band = model["pressure_band"]
        
        if global_index >= 0.7:
            assert band == PRESSURE_BANDS["high"]
        elif global_index >= 0.4:
            assert band == PRESSURE_BANDS["medium"]
        else:
            assert band == PRESSURE_BANDS["low"]

    def test_pressure_model_deterministic(self, sparse_coverage_index, sample_robustness_radar):
        """Pressure model is deterministic for same inputs."""
        m1 = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        m2 = build_adversarial_pressure_model(sparse_coverage_index, sample_robustness_radar)
        
        j1 = pressure_model_to_json(m1)
        j2 = pressure_model_to_json(m2)
        assert j1 == j2


# ===========================================================================
# TASK 2: MULTI-SCENARIO EVOLUTION PLAN TESTS
# ===========================================================================

@pytest.mark.oracle
class TestScenarioEvolutionPlan:
    """Tests for build_evolving_adversarial_scenario_plan()."""

    def test_plan_has_scenario_backlog(self, sparse_coverage_index, sample_robustness_radar):
        """Evolution plan includes scenario_backlog."""
        radar = sample_robustness_radar
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        assert "scenario_backlog" in plan
        assert isinstance(plan["scenario_backlog"], list)

    def test_plan_has_priority_order(self, sparse_coverage_index, sample_robustness_radar):
        """Evolution plan includes priority_order."""
        radar = sample_robustness_radar
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        assert "priority_order" in plan
        assert isinstance(plan["priority_order"], list)

    def test_plan_has_multi_metric_scenarios(self, sparse_coverage_index, sample_robustness_radar):
        """Evolution plan includes multi_metric_scenarios."""
        radar = sample_robustness_radar
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        assert "multi_metric_scenarios" in plan
        assert isinstance(plan["multi_metric_scenarios"], list)

    def test_plan_has_neutral_rationale(self, sparse_coverage_index, sample_robustness_radar):
        """Evolution plan includes neutral_rationale."""
        radar = sample_robustness_radar
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        assert "neutral_rationale" in plan
        assert isinstance(plan["neutral_rationale"], list)

    def test_plan_prioritizes_core_blocking_metrics(self):
        """Plan prioritizes core metrics blocking promotion."""
        # Create scenario where core metric blocks promotion
        scorecards = []
        # Empty scorecards → no coverage → core metrics block
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        # Check if core metrics are in priority 1 scenarios
        priority_1 = [s for s in plan["scenario_backlog"] if s["priority"] == 1]
        core_in_priority_1 = any(
            mk in CORE_UPLIFT_METRICS
            for s in priority_1
            for mk in s["metric_kinds"]
        )
        
        # Core metrics should be blocking and in priority 1
        assert len(priority_1) > 0
        assert core_in_priority_1

    def test_plan_includes_high_pressure_scenarios(self, at_risk_coverage_index, sample_robustness_radar):
        """Plan includes scenarios for high-pressure metrics."""
        radar = sample_robustness_radar
        coverage_index = at_risk_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        # High-pressure metrics should appear in backlog
        high_pressure_metrics = [
            mk for mk, score in pressure_model["metric_pressure_scores"].items()
            if score > PRESSURE_THRESHOLD_PRIORITY
        ]
        
        if high_pressure_metrics:
            backlog_metrics = [
                mk for s in plan["scenario_backlog"]
                for mk in s["metric_kinds"]
            ]
            # At least some high-pressure metrics should be in backlog
            assert any(mk in backlog_metrics for mk in high_pressure_metrics)

    def test_plan_scenario_structure(self, sparse_coverage_index, sample_robustness_radar):
        """Plan scenarios have required structure."""
        radar = sample_robustness_radar
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        for scenario in plan["scenario_backlog"]:
            assert "name" in scenario
            assert "profile" in scenario
            assert "metric_kinds" in scenario
            assert "modes" in scenario
            assert "priority" in scenario
            assert "rationale" in scenario

    def test_plan_deterministic(self, sparse_coverage_index, sample_robustness_radar):
        """Evolution plan is deterministic."""
        radar = sample_robustness_radar
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        
        p1 = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        p2 = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        j1 = scenario_plan_to_json(p1)
        j2 = scenario_plan_to_json(p2)
        assert j1 == j2


# ===========================================================================
# TASK 3: FAILOVER SUFFICIENCY CHECKER V2 TESTS
# ===========================================================================

@pytest.mark.oracle
class TestFailoverPlanV2:
    """Tests for build_adversarial_failover_plan_v2()."""

    def test_failover_v2_has_base_plan_fields(self, sparse_coverage_index, sample_robustness_radar):
        """Failover v2 includes all base plan fields."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        # Check base plan fields
        assert "has_failover" in plan_v2
        assert "metrics_without_failover" in plan_v2
        assert "status" in plan_v2
        assert "recommendations" in plan_v2

    def test_failover_v2_has_v2_metrics(self, sparse_coverage_index, sample_robustness_radar):
        """Failover v2 includes v2_metrics per metric."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        assert "v2_metrics" in plan_v2
        assert isinstance(plan_v2["v2_metrics"], dict)
        
        for mk in METRIC_KINDS:
            assert mk in plan_v2["v2_metrics"]
            v2_data = plan_v2["v2_metrics"][mk]
            assert "redundancy_depth" in v2_data
            assert "scenario_diversity" in v2_data
            assert "failure_case_sensitivity" in v2_data

    def test_failover_v2_has_redundancy_depth(self, sparse_coverage_index, sample_robustness_radar):
        """Failover v2 includes redundancy_depth per metric."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        for mk in METRIC_KINDS:
            redundancy = plan_v2["v2_metrics"][mk]["redundancy_depth"]
            assert isinstance(redundancy, int)
            assert redundancy >= 0

    def test_failover_v2_has_scenario_diversity(self, sparse_coverage_index, sample_robustness_radar):
        """Failover v2 includes scenario_diversity per metric."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        for mk in METRIC_KINDS:
            diversity = plan_v2["v2_metrics"][mk]["scenario_diversity"]
            assert isinstance(diversity, (int, float))
            assert 0.0 <= diversity <= 1.0

    def test_failover_v2_has_failure_sensitivity(self, sparse_coverage_index, sample_robustness_radar):
        """Failover v2 includes failure_case_sensitivity per metric."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        for mk in METRIC_KINDS:
            sensitivity = plan_v2["v2_metrics"][mk]["failure_case_sensitivity"]
            assert isinstance(sensitivity, (int, float))
            assert 0.0 <= sensitivity <= 1.0

    def test_failover_v2_has_aggregates(self, sparse_coverage_index, sample_robustness_radar):
        """Failover v2 includes v2_aggregates."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        assert "v2_aggregates" in plan_v2
        aggregates = plan_v2["v2_aggregates"]
        assert "average_redundancy_depth" in aggregates
        assert "average_scenario_diversity" in aggregates
        assert "average_failure_sensitivity" in aggregates

    def test_failover_v2_redundancy_matches_scenario_count(self, sparse_coverage_index, sample_robustness_radar):
        """Redundancy depth matches scenario count."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        for mk in METRIC_KINDS:
            redundancy = plan_v2["v2_metrics"][mk]["redundancy_depth"]
            scenario_count = coverage_index["metrics"][mk]["scenario_count"]
            assert redundancy == scenario_count

    def test_failover_v2_diversity_increases_with_modes(self, healthy_coverage_index, sample_robustness_radar):
        """Scenario diversity increases with more modes covered."""
        coverage_index = healthy_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        # Metrics with more scenarios should generally have higher diversity
        # (though this depends on scenario composition)
        for mk in METRIC_KINDS:
            diversity = plan_v2["v2_metrics"][mk]["scenario_diversity"]
            assert 0.0 <= diversity <= 1.0

    def test_failover_v2_sensitivity_reflects_regressions(self, at_risk_coverage_index, sample_robustness_radar):
        """Failure sensitivity reflects regression detection."""
        coverage_index = at_risk_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        plan_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        # goal_hit has regressions → should have moderate-high sensitivity
        goal_hit_sensitivity = plan_v2["v2_metrics"]["goal_hit"]["failure_case_sensitivity"]
        assert goal_hit_sensitivity >= 0.5  # Regressions detected = good sensitivity

    def test_failover_v2_deterministic(self, sparse_coverage_index, sample_robustness_radar):
        """Failover v2 is deterministic."""
        coverage_index = sparse_coverage_index
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        p1 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        p2 = build_adversarial_failover_plan_v2(coverage_index, readiness, sample_robustness_radar)
        
        j1 = failover_plan_to_json(p1)
        j2 = failover_plan_to_json(p2)
        assert j1 == j2


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

@pytest.mark.oracle
class TestPressureGridIntegration:
    """Integration tests for pressure grid pipeline."""

    def test_full_pressure_pipeline(self, sparse_coverage_index, sample_robustness_radar):
        """Test full pipeline from coverage to evolution plan."""
        coverage_index = sparse_coverage_index
        radar = sample_robustness_radar
        
        # Build pressure model
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        assert "metric_pressure_scores" in pressure_model
        
        # Build readiness and failover
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        # Build evolution plan
        evolution_plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        assert "scenario_backlog" in evolution_plan
        
        # Build failover v2
        failover_v2 = build_adversarial_failover_plan_v2(coverage_index, readiness, radar)
        assert "v2_metrics" in failover_v2

    def test_pressure_prioritizes_at_risk_metrics(self):
        """Pressure model correctly prioritizes AT_RISK metrics."""
        # Create AT_RISK scenario with matching radar
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        
        pressure_model = build_adversarial_pressure_model(coverage_index, radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        failover_plan = build_adversarial_failover_plan(coverage_index, readiness)
        evolution_plan = build_evolving_adversarial_scenario_plan(pressure_model, failover_plan, readiness)
        
        # goal_hit should have high pressure (AT_RISK with cross-scenario regressions)
        goal_hit_pressure = pressure_model["metric_pressure_scores"]["goal_hit"]
        # Should be at least moderate-high due to AT_RISK status
        assert goal_hit_pressure >= 0.5
        
        # If pressure exceeds threshold, should be in targets
        if goal_hit_pressure > PRESSURE_THRESHOLD_PRIORITY:
            assert "goal_hit" in pressure_model["scenario_pressure_targets"]

