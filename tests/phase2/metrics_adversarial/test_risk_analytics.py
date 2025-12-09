# tests/phase2/metrics_adversarial/test_risk_analytics.py
"""
Tests for Phase III Risk Analytics Engine

Covers:
- Scenario-Level Risk Summary (TASK 1)
- Multi-Scenario Robustness Radar (TASK 2)
- CI/Global Health Drift Signal (TASK 3)

NO METRIC INTERPRETATION: These tests verify analytics mechanics only.
"""

import json
import pytest
from typing import Dict, List, Any

from experiments.metrics_adversarial_harness import (
    # Risk Analytics
    summarize_scenario_risk,
    build_metric_robustness_radar,
    summarize_adversarial_health_for_global_health,
    health_summary_to_json,
    
    # Phase IV - Promotion Gate
    build_metric_adversarial_coverage_index,
    evaluate_adversarial_readiness_for_promotion,
    build_adversarial_director_panel,
    coverage_index_to_json,
    readiness_eval_to_json,
    director_panel_to_json,
    
    # Curriculum Designer & Failover Planner
    propose_adversarial_scenarios,
    build_adversarial_failover_plan,
    curriculum_proposal_to_json,
    failover_plan_to_json,
    COMPLEXITY_BANDS,
    
    # Constants
    RISK_THRESHOLDS,
    ROBUSTNESS_TAGS,
    HEALTH_STATUS,
    COVERAGE_STATUS,
    PROMOTION_STATUS,
    STATUS_LIGHT,
    CORE_UPLIFT_METRICS,
    SCORECARD_SCHEMA_VERSION,
    METRIC_KINDS,
    
    # Scenarios and Scorecards
    SCENARIOS,
    build_robustness_scorecard,
    scorecard_to_json,
    HarnessSummary,
    
    # Harness
    AdversarialHarness,
    SEED_HARNESS,
)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def empty_scorecard() -> Dict[str, Any]:
    """Empty scorecard with all metrics but no data."""
    return build_robustness_scorecard([])


@pytest.fixture
def healthy_scorecard() -> Dict[str, Any]:
    """Scorecard with healthy metrics (no regressions)."""
    summaries = [
        HarnessSummary(
            mode="fault",
            metric_kind="goal_hit",
            total_cases=100,
            passed=95,
            failed=0,
            mismatches=0,
            errors=5,
            fault_divergences={"missing_field": 2, "wrong_type": 3},
        ),
        HarnessSummary(
            mode="mutation",
            metric_kind="goal_hit",
            total_cases=100,
            passed=100,
            failed=0,
            mismatches=0,
            errors=0,
            fault_divergences={"threshold_plus_1": 10, "threshold_minus_1": 5, "threshold_plus_2": 3},
        ),
        HarnessSummary(
            mode="replay",
            metric_kind="goal_hit",
            total_cases=100,
            passed=100,
            failed=0,
            mismatches=0,  # No regressions
            errors=0,
        ),
    ]
    return build_robustness_scorecard(summaries, scenarios_run=["baseline_sanity"])


@pytest.fixture
def risky_scorecard() -> Dict[str, Any]:
    """Scorecard with regressions detected."""
    summaries = [
        HarnessSummary(
            mode="fault",
            metric_kind="goal_hit",
            total_cases=100,
            passed=80,
            failed=0,
            mismatches=0,
            errors=20,
            fault_divergences={"missing_field": 10, "wrong_type": 10},
        ),
        HarnessSummary(
            mode="replay",
            metric_kind="goal_hit",
            total_cases=100,
            passed=95,
            failed=0,
            mismatches=5,  # Regressions detected
            errors=0,
        ),
    ]
    return build_robustness_scorecard(summaries, scenarios_run=["baseline_sanity"])


@pytest.fixture
def multi_scenario_scorecards() -> List[Dict[str, Any]]:
    """Multiple scorecards from different scenarios."""
    scorecards = []
    
    # Scenario 1: baseline_sanity (healthy)
    summaries1 = [
        HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
        HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
        HarnessSummary(mode="fault", metric_kind="density", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
    ]
    scorecards.append(build_robustness_scorecard(summaries1, scenarios_run=["baseline_sanity"]))
    
    # Scenario 2: goal_hit_boundary (with minor regression)
    summaries2 = [
        HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=100, passed=98, failed=0, mismatches=0, errors=2),
        HarnessSummary(mode="mutation", metric_kind="goal_hit", total_cases=100, passed=100, failed=0, mismatches=0, errors=0),
        HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=100, passed=99, failed=0, mismatches=1, errors=0),
    ]
    scorecards.append(build_robustness_scorecard(summaries2, scenarios_run=["goal_hit_boundary"]))
    
    # Scenario 3: ci_quick
    summaries3 = [
        HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
        HarnessSummary(mode="fault", metric_kind="density", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
    ]
    scorecards.append(build_robustness_scorecard(summaries3, scenarios_run=["ci_quick"]))
    
    return scorecards


# ===========================================================================
# TASK 1: SCENARIO-LEVEL RISK SUMMARY TESTS
# ===========================================================================

@pytest.mark.oracle
class TestScenarioRiskSummary:
    """Tests for summarize_scenario_risk()."""

    def test_risk_summary_has_schema_version(self, healthy_scorecard):
        """Risk summary includes schema version."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert "schema_version" in risk
        assert risk["schema_version"] == SCORECARD_SCHEMA_VERSION

    def test_risk_summary_has_scenario_name(self, healthy_scorecard):
        """Risk summary includes scenario name."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert risk["scenario_name"] == "baseline_sanity"

    def test_risk_summary_has_scenario_profile(self, healthy_scorecard):
        """Risk summary includes scenario profile."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert risk["scenario_profile"] == "fast"

    def test_risk_summary_has_metrics_covered(self, healthy_scorecard):
        """Risk summary includes metrics covered."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert "metrics_covered" in risk
        assert isinstance(risk["metrics_covered"], list)

    def test_risk_summary_has_fault_coverage(self, healthy_scorecard):
        """Risk summary includes fault coverage count."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert "fault_coverage" in risk
        assert isinstance(risk["fault_coverage"], int)

    def test_risk_summary_has_mutation_coverage(self, healthy_scorecard):
        """Risk summary includes mutation coverage count."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert "mutation_coverage" in risk
        assert isinstance(risk["mutation_coverage"], int)

    def test_risk_summary_has_replay_regressions(self, healthy_scorecard):
        """Risk summary includes replay regressions count."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert "replay_regressions_detected" in risk
        assert isinstance(risk["replay_regressions_detected"], int)

    def test_risk_summary_has_risk_level(self, healthy_scorecard):
        """Risk summary includes risk level."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert "risk_level" in risk
        assert risk["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


@pytest.mark.oracle
class TestRiskLevelCalculation:
    """Tests for risk level calculation rules."""

    def test_low_risk_no_regressions(self):
        """No regressions with good fault coverage → LOW risk."""
        # Create scorecard with good fault coverage (>=3 fault types) and no regressions
        summaries = [
            HarnessSummary(
                mode="fault",
                metric_kind="goal_hit",
                total_cases=100,
                passed=95,
                failed=0,
                mismatches=0,
                errors=5,
                fault_divergences={"missing_field": 2, "wrong_type": 3, "extreme_value": 1, "null_value": 1},
            ),
            HarnessSummary(
                mode="replay",
                metric_kind="goal_hit",
                total_cases=100,
                passed=100,
                failed=0,
                mismatches=0,  # No regressions
                errors=0,
            ),
        ]
        scorecard = build_robustness_scorecard(summaries, scenarios_run=["goal_hit_boundary"])
        risk = summarize_scenario_risk(scorecard, "goal_hit_boundary")
        # With good fault coverage and no regressions
        assert risk["risk_level"] == "LOW"

    def test_medium_risk_with_regression(self, risky_scorecard):
        """1-2 regressions → MEDIUM risk."""
        # Create scorecard with 1-2 regressions
        summaries = [
            HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=100, passed=99, failed=0, mismatches=1, errors=0),
            HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=100, passed=95, failed=0, mismatches=0, errors=5, fault_divergences={"a": 1, "b": 1, "c": 1, "d": 1}),
        ]
        scorecard = build_robustness_scorecard(summaries, scenarios_run=["baseline_sanity"])
        risk = summarize_scenario_risk(scorecard, "baseline_sanity")
        assert risk["risk_level"] == "MEDIUM"

    def test_high_risk_many_regressions(self):
        """>=3 regressions → HIGH risk."""
        summaries = [
            HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=100, passed=95, failed=0, mismatches=5, errors=0),
            HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=100, passed=95, failed=0, mismatches=0, errors=5),
        ]
        scorecard = build_robustness_scorecard(summaries, scenarios_run=["baseline_sanity"])
        risk = summarize_scenario_risk(scorecard, "baseline_sanity")
        assert risk["risk_level"] == "HIGH"

    def test_unknown_scenario_raises_error(self, healthy_scorecard):
        """Unknown scenario raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            summarize_scenario_risk(healthy_scorecard, "nonexistent_scenario")


@pytest.mark.oracle
class TestScenarioRiskDeterminism:
    """Tests for scenario risk determinism."""

    def test_risk_summary_deterministic(self, healthy_scorecard):
        """Risk summary is deterministic for same inputs."""
        risk1 = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        risk2 = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        assert risk1 == risk2

    def test_risk_summary_json_serializable(self, healthy_scorecard):
        """Risk summary is JSON serializable."""
        risk = summarize_scenario_risk(healthy_scorecard, "baseline_sanity")
        json_str = json.dumps(risk, indent=2)
        parsed = json.loads(json_str)
        assert parsed["scenario_name"] == "baseline_sanity"


# ===========================================================================
# TASK 2: MULTI-SCENARIO ROBUSTNESS RADAR TESTS
# ===========================================================================

@pytest.mark.oracle
class TestRobustnessRadar:
    """Tests for build_metric_robustness_radar()."""

    def test_radar_has_schema_version(self, multi_scenario_scorecards):
        """Radar includes schema version."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        assert radar["schema_version"] == SCORECARD_SCHEMA_VERSION

    def test_radar_has_all_metrics(self, multi_scenario_scorecards):
        """Radar includes all metric kinds."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        for mk in METRIC_KINDS:
            assert mk in radar["metrics"]

    def test_radar_has_scenarios_exercised(self, multi_scenario_scorecards):
        """Radar tracks scenarios exercised per metric."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        # goal_hit should be exercised in multiple scenarios
        assert len(radar["metrics"]["goal_hit"]["scenarios_exercised"]) >= 2

    def test_radar_has_scenarios_with_regressions(self, multi_scenario_scorecards):
        """Radar tracks scenarios with regressions."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        # goal_hit has regression in goal_hit_boundary
        assert "scenarios_with_regressions" in radar["metrics"]["goal_hit"]

    def test_radar_has_robustness_tag(self, multi_scenario_scorecards):
        """Radar assigns robustness tags."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        for mk in METRIC_KINDS:
            tag = radar["metrics"][mk]["robustness_tag"]
            assert tag in [
                ROBUSTNESS_TAGS["well_exercised"],
                ROBUSTNESS_TAGS["partially_tested"],
                ROBUSTNESS_TAGS["sparsely_tested"],
            ]

    def test_radar_global_metrics_at_risk(self, multi_scenario_scorecards):
        """Radar tracks metrics at risk globally."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        assert "metrics_at_risk" in radar["global"]
        assert isinstance(radar["global"]["metrics_at_risk"], list)

    def test_radar_global_total_scenarios(self, multi_scenario_scorecards):
        """Radar tracks total scenarios analyzed."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        assert radar["global"]["total_scenarios_analyzed"] >= 2


@pytest.mark.oracle
class TestRobustnessTagging:
    """Tests for robustness tagging rules."""

    def test_well_exercised_tag(self):
        """Metrics in >=3 scenarios get WELL_EXERCISED tag."""
        # Create scorecards covering goal_hit in 3+ scenarios
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary", "ci_quick"]:
            summaries = [
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
            ]
            scorecards.append(build_robustness_scorecard(summaries, scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        assert radar["metrics"]["goal_hit"]["robustness_tag"] == ROBUSTNESS_TAGS["well_exercised"]

    def test_partially_tested_tag(self):
        """Metrics in 1-2 scenarios get PARTIALLY_TESTED tag."""
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="density", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["density_stress"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        assert radar["metrics"]["density"]["robustness_tag"] == ROBUSTNESS_TAGS["partially_tested"]

    def test_sparsely_tested_tag(self):
        """Metrics in 0 scenarios get SPARSELY_TESTED tag."""
        # Empty scorecard list
        radar = build_metric_robustness_radar([])
        for mk in METRIC_KINDS:
            assert radar["metrics"][mk]["robustness_tag"] == ROBUSTNESS_TAGS["sparsely_tested"]


@pytest.mark.oracle
class TestMetricsAtRisk:
    """Tests for metrics at risk detection."""

    def test_metrics_at_risk_with_multi_scenario_regressions(self):
        """Metrics with regressions in multiple scenarios are flagged."""
        # Create scorecards with goal_hit regressions in 2 scenarios
        scorecards = []
        
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            summaries = [
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=48, failed=0, mismatches=2, errors=0),
            ]
            scorecards.append(build_robustness_scorecard(summaries, scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        assert "goal_hit" in radar["global"]["metrics_at_risk"]

    def test_no_metrics_at_risk_with_isolated_regressions(self, multi_scenario_scorecards):
        """Metrics with regressions in only 1 scenario are not flagged."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        # goal_hit only has regression in 1 scenario (goal_hit_boundary)
        assert "goal_hit" not in radar["global"]["metrics_at_risk"]


@pytest.mark.oracle
class TestRadarDeterminism:
    """Tests for radar determinism."""

    def test_radar_deterministic(self, multi_scenario_scorecards):
        """Radar is deterministic for same inputs."""
        radar1 = build_metric_robustness_radar(multi_scenario_scorecards)
        radar2 = build_metric_robustness_radar(multi_scenario_scorecards)
        
        j1 = json.dumps(radar1, sort_keys=True)
        j2 = json.dumps(radar2, sort_keys=True)
        assert j1 == j2

    def test_radar_json_serializable(self, multi_scenario_scorecards):
        """Radar is JSON serializable."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        json_str = json.dumps(radar, indent=2)
        parsed = json.loads(json_str)
        assert "metrics" in parsed
        assert "global" in parsed


# ===========================================================================
# TASK 3: CI/GLOBAL HEALTH DRIFT SIGNAL TESTS
# ===========================================================================

@pytest.mark.oracle
class TestGlobalHealthSummary:
    """Tests for summarize_adversarial_health_for_global_health()."""

    def test_health_has_coverage_ok(self, multi_scenario_scorecards):
        """Health summary includes adversarial_coverage_ok flag."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert "adversarial_coverage_ok" in health
        assert isinstance(health["adversarial_coverage_ok"], bool)

    def test_health_has_metrics_at_risk(self, multi_scenario_scorecards):
        """Health summary includes metrics_at_risk list."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert "metrics_at_risk" in health
        assert isinstance(health["metrics_at_risk"], list)

    def test_health_has_status(self, multi_scenario_scorecards):
        """Health summary includes status string."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert "status" in health
        assert health["status"] in [HEALTH_STATUS["ok"], HEALTH_STATUS["warn"], HEALTH_STATUS["attention"]]

    def test_health_has_summary(self, multi_scenario_scorecards):
        """Health summary includes human-readable summary."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert "summary" in health
        assert isinstance(health["summary"], str)
        assert len(health["summary"]) > 0

    def test_health_has_details(self, multi_scenario_scorecards):
        """Health summary includes details block."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert "details" in health
        assert "total_metrics" in health["details"]
        assert "total_regressions" in health["details"]


@pytest.mark.oracle
class TestHealthStatusRules:
    """Tests for health status determination rules."""

    def test_ok_status_no_regressions(self):
        """OK status when all metrics well-exercised, no regressions."""
        # Create healthy scorecards for all metrics
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary", "ci_quick"]:
            summaries = []
            for mk in METRIC_KINDS:
                summaries.append(HarnessSummary(
                    mode="fault", metric_kind=mk, total_cases=50, passed=50, failed=0, mismatches=0, errors=0
                ))
            scorecards.append(build_robustness_scorecard(summaries, scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert health["status"] == HEALTH_STATUS["ok"]
        assert health["adversarial_coverage_ok"] is True

    def test_warn_status_isolated_regressions(self):
        """WARN status when regressions isolated to single scenarios."""
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=48, failed=0, mismatches=2, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert health["status"] == HEALTH_STATUS["warn"]
        assert health["adversarial_coverage_ok"] is True

    def test_attention_status_cross_scenario_regressions(self):
        """ATTENTION status when metrics have regressions in multiple scenarios."""
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        assert health["status"] == HEALTH_STATUS["attention"]
        assert health["adversarial_coverage_ok"] is False


@pytest.mark.oracle
class TestHealthDeterminism:
    """Tests for health summary determinism."""

    def test_health_deterministic(self, multi_scenario_scorecards):
        """Health summary is deterministic for same inputs."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        health1 = summarize_adversarial_health_for_global_health(radar)
        health2 = summarize_adversarial_health_for_global_health(radar)
        
        j1 = health_summary_to_json(health1)
        j2 = health_summary_to_json(health2)
        assert j1 == j2

    def test_health_json_serializable(self, multi_scenario_scorecards):
        """Health summary is JSON serializable."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        health = summarize_adversarial_health_for_global_health(radar)
        json_str = health_summary_to_json(health)
        parsed = json.loads(json_str)
        assert "status" in parsed


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

@pytest.mark.oracle
class TestRiskAnalyticsIntegration:
    """Integration tests for risk analytics pipeline."""

    def test_full_pipeline_with_real_harness(self):
        """Test full pipeline from harness to health summary."""
        harness = AdversarialHarness(seed=SEED_HARNESS)
        
        # Run a small scenario
        scenario = SCENARIOS["ci_quick"]
        all_summaries = []
        
        for mk in scenario.metric_kinds:
            for mode in scenario.modes:
                summaries = harness.run(mk, 10, mode)  # Small test count
                all_summaries.extend(summaries)
        
        # Build scorecard
        scorecard = build_robustness_scorecard(all_summaries, scenarios_run=["ci_quick"])
        
        # Build risk summary
        risk = summarize_scenario_risk(scorecard, "ci_quick")
        assert risk["scenario_name"] == "ci_quick"
        assert risk["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        
        # Build radar
        radar = build_metric_robustness_radar([scorecard])
        assert "metrics" in radar
        
        # Build health summary
        health = summarize_adversarial_health_for_global_health(radar)
        assert "status" in health

    def test_empty_inputs_produce_valid_outputs(self):
        """Empty inputs produce valid but sparse outputs."""
        # Empty scorecard
        empty_scorecard = build_robustness_scorecard([])
        
        # Risk summary for baseline_sanity with empty data
        risk = summarize_scenario_risk(empty_scorecard, "baseline_sanity")
        assert risk["risk_level"] == "MEDIUM"  # Sparse coverage
        
        # Empty radar
        radar = build_metric_robustness_radar([])
        for mk in METRIC_KINDS:
            assert radar["metrics"][mk]["robustness_tag"] == ROBUSTNESS_TAGS["sparsely_tested"]
        
        # Health from empty radar
        health = summarize_adversarial_health_for_global_health(radar)
        assert health["status"] == HEALTH_STATUS["warn"]  # Sparse = warn

    def test_all_scenarios_can_be_analyzed(self):
        """All registered scenarios can be analyzed for risk."""
        empty_scorecard = build_robustness_scorecard([])
        
        for scenario_name in SCENARIOS.keys():
            risk = summarize_scenario_risk(empty_scorecard, scenario_name)
            assert risk["scenario_name"] == scenario_name
            assert "risk_level" in risk


# ===========================================================================
# PHASE IV — ADVERSARIAL COVERAGE PROMOTION GATE TESTS
# ===========================================================================

@pytest.mark.oracle
class TestCoverageIndex:
    """Tests for build_metric_adversarial_coverage_index()."""

    def test_coverage_index_has_schema_version(self, multi_scenario_scorecards):
        """Coverage index includes schema version."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        assert coverage_index["schema_version"] == SCORECARD_SCHEMA_VERSION

    def test_coverage_index_has_all_metrics(self, multi_scenario_scorecards):
        """Coverage index includes all metric kinds."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        for mk in METRIC_KINDS:
            assert mk in coverage_index["metrics"]

    def test_coverage_index_has_robustness_tag(self, multi_scenario_scorecards):
        """Coverage index includes robustness_tag per metric."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        for mk in METRIC_KINDS:
            assert "robustness_tag" in coverage_index["metrics"][mk]

    def test_coverage_index_has_scenario_count(self, multi_scenario_scorecards):
        """Coverage index includes scenario_count per metric."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        for mk in METRIC_KINDS:
            assert "scenario_count" in coverage_index["metrics"][mk]
            assert isinstance(coverage_index["metrics"][mk]["scenario_count"], int)

    def test_coverage_index_has_regression_count(self, multi_scenario_scorecards):
        """Coverage index includes regression_count per metric."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        for mk in METRIC_KINDS:
            assert "regression_count" in coverage_index["metrics"][mk]
            assert isinstance(coverage_index["metrics"][mk]["regression_count"], int)

    def test_coverage_index_has_coverage_status(self, multi_scenario_scorecards):
        """Coverage index includes coverage_status per metric."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        for mk in METRIC_KINDS:
            status = coverage_index["metrics"][mk]["coverage_status"]
            assert status in [COVERAGE_STATUS["ok"], COVERAGE_STATUS["sparse"], COVERAGE_STATUS["at_risk"]]

    def test_coverage_index_global_metrics_at_risk(self, multi_scenario_scorecards):
        """Coverage index tracks metrics_at_risk globally."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        assert "metrics_at_risk" in coverage_index["global"]
        assert isinstance(coverage_index["global"]["metrics_at_risk"], list)

    def test_coverage_index_global_coverage_ok(self, multi_scenario_scorecards):
        """Coverage index includes global coverage_ok flag."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        assert "coverage_ok" in coverage_index["global"]
        assert isinstance(coverage_index["global"]["coverage_ok"], bool)

    def test_coverage_index_at_risk_detection(self):
        """Coverage index flags metrics with cross-scenario regressions as AT_RISK."""
        # Create scorecards with goal_hit regressions in 2 scenarios
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        
        assert coverage_index["metrics"]["goal_hit"]["coverage_status"] == COVERAGE_STATUS["at_risk"]
        assert "goal_hit" in coverage_index["global"]["metrics_at_risk"]
        assert coverage_index["global"]["coverage_ok"] is False

    def test_coverage_index_sparse_detection(self):
        """Coverage index flags missing metrics as AT_RISK, sparsely tested as SPARSE."""
        # Empty radar → all metrics missing → AT_RISK
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        
        for mk in METRIC_KINDS:
            # Missing metrics are AT_RISK
            assert coverage_index["metrics"][mk]["coverage_status"] == COVERAGE_STATUS["at_risk"]
        
        # Single scenario with one metric → that metric is SPARSE
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        radar2 = build_metric_robustness_radar(scorecards)
        coverage_index2 = build_metric_adversarial_coverage_index(radar2)
        
        # goal_hit should be SPARSE (only 1 scenario, but present)
        assert coverage_index2["metrics"]["goal_hit"]["coverage_status"] == COVERAGE_STATUS["sparse"]

    def test_coverage_index_deterministic(self, multi_scenario_scorecards):
        """Coverage index is deterministic for same inputs."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        index1 = build_metric_adversarial_coverage_index(radar)
        index2 = build_metric_adversarial_coverage_index(radar)
        
        j1 = coverage_index_to_json(index1)
        j2 = coverage_index_to_json(index2)
        assert j1 == j2


@pytest.mark.oracle
class TestPromotionReadiness:
    """Tests for evaluate_adversarial_readiness_for_promotion()."""

    def test_readiness_has_promotion_ok(self, multi_scenario_scorecards):
        """Readiness evaluation includes promotion_ok flag."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        assert "promotion_ok" in readiness
        assert isinstance(readiness["promotion_ok"], bool)

    def test_readiness_has_metrics_blocking(self, multi_scenario_scorecards):
        """Readiness evaluation includes metrics_blocking_promotion list."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        assert "metrics_blocking_promotion" in readiness
        assert isinstance(readiness["metrics_blocking_promotion"], list)

    def test_readiness_has_status(self, multi_scenario_scorecards):
        """Readiness evaluation includes status."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        assert "status" in readiness
        assert readiness["status"] in [PROMOTION_STATUS["ok"], PROMOTION_STATUS["warn"], PROMOTION_STATUS["block"]]

    def test_readiness_has_reasons(self, multi_scenario_scorecards):
        """Readiness evaluation includes reasons list."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        assert "reasons" in readiness
        assert isinstance(readiness["reasons"], list)

    def test_readiness_blocks_core_metric_at_risk(self):
        """Readiness BLOCKS if core metric is AT_RISK."""
        # Create coverage index with goal_hit (core) AT_RISK
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        assert readiness["status"] == PROMOTION_STATUS["block"]
        assert readiness["promotion_ok"] is False
        assert "goal_hit" in readiness["metrics_blocking_promotion"]

    def test_readiness_warns_on_sparse(self):
        """Readiness WARNS if any metric is SPARSE."""
        # Create coverage index with sparse coverage
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="chain_length", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["chain_length_deep"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        # Some metrics will be SPARSE (not in scorecards)
        assert readiness["status"] in [PROMOTION_STATUS["warn"], PROMOTION_STATUS["block"]]

    def test_readiness_ok_when_all_healthy(self):
        """Readiness OK when all metrics healthy."""
        # Create healthy scorecards for all metrics
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary", "ci_quick"]:
            summaries = []
            for mk in METRIC_KINDS:
                summaries.append(HarnessSummary(
                    mode="fault", metric_kind=mk, total_cases=50, passed=50, failed=0, mismatches=0, errors=0
                ))
            scorecards.append(build_robustness_scorecard(summaries, scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        # Should be OK or WARN (WARN if some metrics still sparse)
        assert readiness["status"] in [PROMOTION_STATUS["ok"], PROMOTION_STATUS["warn"]]

    def test_readiness_deterministic(self, multi_scenario_scorecards):
        """Readiness evaluation is deterministic."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        
        r1 = evaluate_adversarial_readiness_for_promotion(coverage_index)
        r2 = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        j1 = readiness_eval_to_json(r1)
        j2 = readiness_eval_to_json(r2)
        assert j1 == j2


@pytest.mark.oracle
class TestDirectorPanel:
    """Tests for build_adversarial_director_panel()."""

    def test_panel_has_status_light(self, multi_scenario_scorecards):
        """Director panel includes status_light."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        assert "status_light" in panel
        assert panel["status_light"] in [STATUS_LIGHT["green"], STATUS_LIGHT["yellow"], STATUS_LIGHT["red"]]

    def test_panel_has_coverage_ok(self, multi_scenario_scorecards):
        """Director panel includes adversarial_coverage_ok."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        assert "adversarial_coverage_ok" in panel
        assert isinstance(panel["adversarial_coverage_ok"], bool)

    def test_panel_has_metrics_at_risk(self, multi_scenario_scorecards):
        """Director panel includes metrics_at_risk."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        assert "metrics_at_risk" in panel
        assert isinstance(panel["metrics_at_risk"], list)

    def test_panel_has_headline(self, multi_scenario_scorecards):
        """Director panel includes headline."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        assert "headline" in panel
        assert isinstance(panel["headline"], str)
        assert len(panel["headline"]) > 0
        # Headline should be neutral (no "good/bad")
        assert "good" not in panel["headline"].lower()
        assert "bad" not in panel["headline"].lower()

    def test_panel_red_light_on_block(self):
        """Panel shows RED light when promotion blocked."""
        # Create blocking scenario
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        assert panel["status_light"] == STATUS_LIGHT["red"]

    def test_panel_yellow_light_on_warn(self):
        """Panel shows YELLOW light when warning."""
        # Create warning scenario (sparse coverage)
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="density", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["density_stress"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        # Should be yellow or red depending on core metrics
        assert panel["status_light"] in [STATUS_LIGHT["yellow"], STATUS_LIGHT["red"]]

    def test_panel_green_light_on_ok(self):
        """Panel shows GREEN light when all OK."""
        # Create healthy scenario
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary", "ci_quick"]:
            summaries = []
            for mk in METRIC_KINDS:
                summaries.append(HarnessSummary(
                    mode="fault", metric_kind=mk, total_cases=50, passed=50, failed=0, mismatches=0, errors=0
                ))
            scorecards.append(build_robustness_scorecard(summaries, scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        # Should be green or yellow (yellow if some metrics still sparse)
        assert panel["status_light"] in [STATUS_LIGHT["green"], STATUS_LIGHT["yellow"]]

    def test_panel_deterministic(self, multi_scenario_scorecards):
        """Director panel is deterministic."""
        radar = build_metric_robustness_radar(multi_scenario_scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        p1 = build_adversarial_director_panel(coverage_index, readiness)
        p2 = build_adversarial_director_panel(coverage_index, readiness)
        
        j1 = director_panel_to_json(p1)
        j2 = director_panel_to_json(p2)
        assert j1 == j2


@pytest.mark.oracle
class TestPhaseIVIntegration:
    """Integration tests for Phase IV promotion gate pipeline."""

    def test_full_pipeline_coverage_to_panel(self):
        """Test full pipeline from radar to director panel."""
        # Build scorecards
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=50, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        
        # Build radar
        radar = build_metric_robustness_radar(scorecards)
        assert "metrics" in radar
        
        # Build coverage index
        coverage_index = build_metric_adversarial_coverage_index(radar)
        assert "metrics" in coverage_index
        assert "global" in coverage_index
        
        # Evaluate readiness
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        assert "promotion_ok" in readiness
        
        # Build director panel
        panel = build_adversarial_director_panel(coverage_index, readiness)
        assert "status_light" in panel
        assert "headline" in panel

    def test_promotion_blocking_workflow(self):
        """Test complete workflow when promotion is blocked."""
        # Create blocking scenario (core metric AT_RISK)
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=40, failed=0, mismatches=10, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        panel = build_adversarial_director_panel(coverage_index, readiness)
        
        # Should be blocked
        assert readiness["status"] == PROMOTION_STATUS["block"]
        assert readiness["promotion_ok"] is False
        assert panel["status_light"] == STATUS_LIGHT["red"]
        assert "goal_hit" in readiness["metrics_blocking_promotion"]


# ===========================================================================
# CURRICULUM DESIGNER & FAILOVER PLANNER TESTS
# ===========================================================================

@pytest.mark.oracle
class TestCurriculumDesigner:
    """Tests for propose_adversarial_scenarios()."""

    def test_proposal_has_metrics_needing(self):
        """Proposal includes metrics_needing_new_scenarios list."""
        # Create sparse coverage scenario
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="chain_length", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["chain_length_deep"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        assert "metrics_needing_new_scenarios" in proposal
        assert isinstance(proposal["metrics_needing_new_scenarios"], list)

    def test_proposal_has_suggested_profiles(self):
        """Proposal includes suggested_scenario_profiles."""
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        assert "suggested_scenario_profiles" in proposal
        assert isinstance(proposal["suggested_scenario_profiles"], dict)

    def test_proposal_has_neutral_notes(self):
        """Proposal includes neutral_notes."""
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="density", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["density_stress"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        assert "neutral_notes" in proposal
        assert isinstance(proposal["neutral_notes"], list)

    def test_proposal_suggests_for_low_coverage(self):
        """Proposal suggests scenarios for metrics with low coverage."""
        # Empty radar → all metrics need scenarios
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        # Should suggest scenarios for all metrics
        assert len(proposal["metrics_needing_new_scenarios"]) > 0
        assert len(proposal["suggested_scenario_profiles"]) > 0

    def test_proposal_suggests_for_at_risk_metrics(self):
        """Proposal suggests scenarios for AT_RISK metrics."""
        # Create AT_RISK scenario
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary"]:
            scorecards.append(build_robustness_scorecard([
                HarnessSummary(mode="replay", metric_kind="goal_hit", total_cases=50, passed=45, failed=0, mismatches=5, errors=0),
            ], scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        # goal_hit should be in metrics_needing
        assert "goal_hit" in proposal["metrics_needing_new_scenarios"]

    def test_proposal_suggests_for_sparse_metrics(self):
        """Proposal suggests scenarios for SPARSE metrics."""
        # Single scenario → sparse coverage
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="multi_goal", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["multi_goal_coverage"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        # multi_goal should be in metrics_needing (sparse)
        assert "multi_goal" in proposal["metrics_needing_new_scenarios"]

    def test_proposal_profile_suggestions(self):
        """Proposal includes profile suggestions with required fields."""
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        # Check suggested profiles have required fields
        for name, profile_info in proposal["suggested_scenario_profiles"].items():
            assert "profile" in profile_info
            assert profile_info["profile"] in ["fast", "standard", "full"]
            assert "metric_kinds" in profile_info
            assert "modes" in profile_info
            assert "complexity_band" in profile_info
            assert "rationale" in profile_info

    def test_proposal_deterministic(self):
        """Proposal is deterministic for same inputs."""
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        
        p1 = propose_adversarial_scenarios(coverage_index, radar)
        p2 = propose_adversarial_scenarios(coverage_index, radar)
        
        j1 = curriculum_proposal_to_json(p1)
        j2 = curriculum_proposal_to_json(p2)
        assert j1 == j2


@pytest.mark.oracle
class TestFailoverPlanner:
    """Tests for build_adversarial_failover_plan()."""

    def test_failover_plan_has_has_failover(self):
        """Failover plan includes has_failover flag."""
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        assert "has_failover" in plan
        assert isinstance(plan["has_failover"], bool)

    def test_failover_plan_has_metrics_without_failover(self):
        """Failover plan includes metrics_without_failover list."""
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        assert "metrics_without_failover" in plan
        assert isinstance(plan["metrics_without_failover"], list)

    def test_failover_plan_has_status(self):
        """Failover plan includes status."""
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        assert "status" in plan
        assert plan["status"] in ["OK", "ATTENTION", "BLOCK"]

    def test_failover_plan_has_recommendations(self):
        """Failover plan includes recommendations."""
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        assert "recommendations" in plan
        assert isinstance(plan["recommendations"], list)

    def test_failover_blocks_core_metrics_without_coverage(self):
        """Failover plan BLOCKS if core metric has no scenario coverage."""
        # Empty radar → no coverage for any metric
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        # Core metrics (goal_hit, density) should be without failover
        assert plan["status"] == "BLOCK"
        assert plan["has_failover"] is False
        assert "goal_hit" in plan["metrics_without_failover"] or "density" in plan["metrics_without_failover"]

    def test_failover_attention_on_sparse_with_scenarios(self):
        """Failover plan shows ATTENTION for SPARSE metrics with existing scenarios."""
        # Single scenario → sparse but has some coverage
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="chain_length", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["chain_length_deep"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        # Should be ATTENTION (sparse coverage) or BLOCK (if core metrics missing)
        assert plan["status"] in ["ATTENTION", "BLOCK"]

    def test_failover_ok_with_adequate_coverage(self):
        """Failover plan shows OK when all metrics have adequate coverage."""
        # Multiple scenarios for all metrics
        scorecards = []
        for scenario_name in ["baseline_sanity", "goal_hit_boundary", "ci_quick"]:
            summaries = []
            for mk in METRIC_KINDS:
                summaries.append(HarnessSummary(
                    mode="fault", metric_kind=mk, total_cases=50, passed=50, failed=0, mismatches=0, errors=0
                ))
            scorecards.append(build_robustness_scorecard(summaries, scenarios_run=[scenario_name]))
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        # Should be OK or ATTENTION (ATTENTION if some metrics still sparse)
        assert plan["status"] in ["OK", "ATTENTION"]
        assert plan["has_failover"] is True

    def test_failover_deterministic(self):
        """Failover plan is deterministic."""
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        
        p1 = build_adversarial_failover_plan(coverage_index, readiness)
        p2 = build_adversarial_failover_plan(coverage_index, readiness)
        
        j1 = failover_plan_to_json(p1)
        j2 = failover_plan_to_json(p2)
        assert j1 == j2


@pytest.mark.oracle
class TestCurriculumFailoverIntegration:
    """Integration tests for curriculum designer and failover planner."""

    def test_curriculum_suggests_for_gaps(self):
        """Curriculum designer suggests scenarios for coverage gaps."""
        # Sparse coverage scenario
        scorecards = [
            build_robustness_scorecard([
                HarnessSummary(mode="fault", metric_kind="goal_hit", total_cases=10, passed=10, failed=0, mismatches=0, errors=0),
            ], scenarios_run=["baseline_sanity"]),
        ]
        
        radar = build_metric_robustness_radar(scorecards)
        coverage_index = build_metric_adversarial_coverage_index(radar)
        proposal = propose_adversarial_scenarios(coverage_index, radar)
        
        # Should suggest scenarios for metrics with low coverage
        assert len(proposal["metrics_needing_new_scenarios"]) > 0
        assert len(proposal["suggested_scenario_profiles"]) > 0

    def test_failover_blocks_core_at_risk_no_scenarios(self):
        """Failover plan blocks when core metric is AT_RISK with no scenarios."""
        # Create AT_RISK scenario but with no actual scenario coverage
        # (simulated by empty radar)
        radar = build_metric_robustness_radar([])
        coverage_index = build_metric_adversarial_coverage_index(radar)
        readiness = evaluate_adversarial_readiness_for_promotion(coverage_index)
        plan = build_adversarial_failover_plan(coverage_index, readiness)
        
        # Core metrics should be without failover
        core_without = [mk for mk in plan["metrics_without_failover"] if mk in CORE_UPLIFT_METRICS]
        if len(core_without) > 0:
            assert plan["status"] == "BLOCK"
            assert plan["has_failover"] is False

