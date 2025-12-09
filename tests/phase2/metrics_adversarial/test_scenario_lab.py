# tests/phase2/metrics_adversarial/test_scenario_lab.py
"""
Tests for Metric Adversary Lab v1.2

Covers:
- Scenario Registry (TASK 1)
- Metric Robustness Scorecard (TASK 2)
- Regression Radar Snapshot Guard (TASK 3)

NO METRIC INTERPRETATION: These tests verify lab mechanics only.
"""

import json
import os
import tempfile
import pytest
from typing import Dict, List, Any

from experiments.metrics_adversarial_harness import (
    # Scenario Registry
    Scenario,
    SCENARIOS,
    list_scenarios,
    get_scenario,
    validate_scenario,
    list_scenarios_json,
    
    # Profile contracts (for validation)
    PROFILE_CONTRACTS,
    METRIC_KINDS,
    
    # Scorecard
    SCORECARD_SCHEMA_VERSION,
    build_robustness_scorecard,
    scorecard_to_json,
    HarnessSummary,
    
    # Snapshot Guard
    create_radar_snapshot,
    snapshot_to_json,
    load_radar_snapshot,
    compare_radar_snapshots,
    print_snapshot_diff,
    run_regression_radar,
    RegressionResult,
    
    # Harness
    AdversarialHarness,
    SEED_HARNESS,
)


# ===========================================================================
# TASK 1: SCENARIO REGISTRY TESTS
# ===========================================================================

@pytest.mark.oracle
class TestScenarioRegistry:
    """Tests for the declarative scenario registry."""

    def test_scenario_registry_not_empty(self):
        """Registry contains at least one scenario."""
        assert len(SCENARIOS) > 0

    def test_scenario_registry_deterministic(self):
        """Scenario registry is deterministic across accesses."""
        keys1 = list(SCENARIOS.keys())
        keys2 = list(SCENARIOS.keys())
        assert keys1 == keys2

    def test_all_scenarios_have_required_fields(self):
        """All scenarios have name, profile, metric_kinds, modes, description."""
        for name, scenario in SCENARIOS.items():
            assert scenario.name == name
            assert scenario.profile in PROFILE_CONTRACTS
            assert len(scenario.metric_kinds) > 0
            assert len(scenario.modes) > 0
            assert len(scenario.description) > 0

    def test_baseline_sanity_scenario_exists(self):
        """baseline_sanity scenario exists and uses fast profile."""
        assert "baseline_sanity" in SCENARIOS
        scenario = SCENARIOS["baseline_sanity"]
        assert scenario.profile == "fast"
        assert len(scenario.metric_kinds) == 4  # All metrics

    def test_goal_hit_boundary_scenario_exists(self):
        """goal_hit_boundary scenario exists and targets goal_hit."""
        assert "goal_hit_boundary" in SCENARIOS
        scenario = SCENARIOS["goal_hit_boundary"]
        assert "goal_hit" in scenario.metric_kinds
        assert scenario.profile == "standard"

    def test_multi_metric_stress_scenario_exists(self):
        """multi_metric_stress scenario exists with full profile."""
        assert "multi_metric_stress" in SCENARIOS
        scenario = SCENARIOS["multi_metric_stress"]
        assert scenario.profile == "full"
        assert len(scenario.metric_kinds) == 4  # All metrics


@pytest.mark.oracle
class TestListScenarios:
    """Tests for list_scenarios helper."""

    def test_list_scenarios_returns_all(self):
        """list_scenarios() returns all scenarios."""
        scenarios = list_scenarios()
        assert len(scenarios) == len(SCENARIOS)

    def test_list_scenarios_respects_profile_filter_fast(self):
        """list_scenarios(filter_profile='fast') returns only fast scenarios."""
        scenarios = list_scenarios(filter_profile="fast")
        for s in scenarios:
            assert s.profile == "fast"

    def test_list_scenarios_respects_profile_filter_standard(self):
        """list_scenarios(filter_profile='standard') returns only standard scenarios."""
        scenarios = list_scenarios(filter_profile="standard")
        for s in scenarios:
            assert s.profile == "standard"

    def test_list_scenarios_respects_profile_filter_full(self):
        """list_scenarios(filter_profile='full') returns only full scenarios."""
        scenarios = list_scenarios(filter_profile="full")
        for s in scenarios:
            assert s.profile == "full"

    def test_list_scenarios_sorted_by_name(self):
        """list_scenarios returns scenarios sorted by name."""
        scenarios = list_scenarios()
        names = [s.name for s in scenarios]
        assert names == sorted(names)

    def test_list_scenarios_deterministic(self):
        """list_scenarios is deterministic across calls."""
        s1 = list_scenarios()
        s2 = list_scenarios()
        assert [s.name for s in s1] == [s.name for s in s2]


@pytest.mark.oracle
class TestScenarioValidation:
    """Tests for scenario validation."""

    def test_all_scenarios_validate(self):
        """Every registered scenario passes validation."""
        for name, scenario in SCENARIOS.items():
            valid, violations = validate_scenario(scenario)
            assert valid, f"Scenario '{name}' invalid: {violations}"

    def test_all_scenarios_reference_valid_profiles(self):
        """Every scenario references a valid ProfileContract."""
        for name, scenario in SCENARIOS.items():
            assert scenario.profile in PROFILE_CONTRACTS, (
                f"Scenario '{name}' has invalid profile: {scenario.profile}"
            )

    def test_all_scenarios_reference_valid_metrics(self):
        """Every scenario references valid metric kinds."""
        for name, scenario in SCENARIOS.items():
            for mk in scenario.metric_kinds:
                assert mk in METRIC_KINDS, (
                    f"Scenario '{name}' has invalid metric: {mk}"
                )

    def test_invalid_scenario_detected(self):
        """Validation catches invalid scenarios."""
        invalid = Scenario(
            name="bad",
            profile="nonexistent_profile",
            metric_kinds=("goal_hit",),
            modes=("fault",),
            description="Invalid scenario",
        )
        valid, violations = validate_scenario(invalid)
        assert not valid
        assert len(violations) > 0


@pytest.mark.oracle
class TestScenarioJSON:
    """Tests for scenario JSON export."""

    def test_list_scenarios_json_valid(self):
        """list_scenarios_json produces valid JSON."""
        json_str = list_scenarios_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_list_scenarios_json_contains_all_scenarios(self):
        """JSON export contains all scenarios."""
        json_str = list_scenarios_json()
        parsed = json.loads(json_str)
        assert set(parsed.keys()) == set(SCENARIOS.keys())

    def test_list_scenarios_json_deterministic(self):
        """JSON export is deterministic."""
        j1 = list_scenarios_json()
        j2 = list_scenarios_json()
        assert j1 == j2


# ===========================================================================
# TASK 2: METRIC ROBUSTNESS SCORECARD TESTS
# ===========================================================================

@pytest.mark.oracle
class TestRobustnessScorecard:
    """Tests for the metric robustness scorecard."""

    @pytest.fixture
    def sample_summaries(self) -> List[HarnessSummary]:
        """Create sample summaries for testing."""
        return [
            HarnessSummary(
                mode="fault",
                metric_kind="goal_hit",
                total_cases=50,
                passed=45,
                failed=0,
                mismatches=0,
                errors=5,
                fault_divergences={"missing_field": 2, "wrong_type": 3},
            ),
            HarnessSummary(
                mode="mutation",
                metric_kind="goal_hit",
                total_cases=100,
                passed=98,
                failed=0,
                mismatches=0,
                errors=2,
                fault_divergences={"threshold_plus_1": 10, "threshold_minus_1": 8},
            ),
            HarnessSummary(
                mode="replay",
                metric_kind="goal_hit",
                total_cases=50,
                passed=50,
                failed=0,
                mismatches=0,
                errors=0,
            ),
            HarnessSummary(
                mode="fault",
                metric_kind="density",
                total_cases=50,
                passed=48,
                failed=0,
                mismatches=0,
                errors=2,
                fault_divergences={"extreme_value": 2},
            ),
        ]

    def test_scorecard_has_schema_version(self, sample_summaries):
        """Scorecard contains schema_version."""
        scorecard = build_robustness_scorecard(sample_summaries)
        assert "schema_version" in scorecard
        assert scorecard["schema_version"] == SCORECARD_SCHEMA_VERSION

    def test_scorecard_contains_all_metric_kinds(self, sample_summaries):
        """Scorecard contains entries for all metric kinds."""
        scorecard = build_robustness_scorecard(sample_summaries)
        assert "metrics" in scorecard
        for mk in METRIC_KINDS:
            assert mk in scorecard["metrics"]

    def test_scorecard_counts_consistent(self, sample_summaries):
        """Scorecard counts match input summaries."""
        scorecard = build_robustness_scorecard(sample_summaries)
        
        # goal_hit should have 200 total tests (50+100+50)
        assert scorecard["metrics"]["goal_hit"]["total_tests"] == 200
        assert scorecard["metrics"]["goal_hit"]["total_passed"] == 45 + 98 + 50

    def test_scorecard_scenarios_tracked(self, sample_summaries):
        """Scorecard tracks scenarios when provided."""
        scorecard = build_robustness_scorecard(
            sample_summaries, 
            scenarios_run=["baseline_sanity"]
        )
        
        # baseline_sanity covers all metrics
        for mk in METRIC_KINDS:
            assert "baseline_sanity" in scorecard["metrics"][mk]["scenarios_covered"]

    def test_scorecard_deterministic(self, sample_summaries):
        """Scorecard is deterministic for same inputs."""
        s1 = build_robustness_scorecard(sample_summaries)
        s2 = build_robustness_scorecard(sample_summaries)
        
        j1 = scorecard_to_json(s1)
        j2 = scorecard_to_json(s2)
        
        assert j1 == j2

    def test_scorecard_to_json_valid(self, sample_summaries):
        """Scorecard serializes to valid JSON."""
        scorecard = build_robustness_scorecard(sample_summaries)
        json_str = scorecard_to_json(scorecard)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_empty_summaries_produce_valid_scorecard(self):
        """Empty summary list produces valid scorecard."""
        scorecard = build_robustness_scorecard([])
        assert "schema_version" in scorecard
        assert "metrics" in scorecard
        for mk in METRIC_KINDS:
            assert mk in scorecard["metrics"]
            assert scorecard["metrics"][mk]["total_tests"] == 0


@pytest.mark.oracle
class TestScorecardWithRealHarness:
    """Tests using real harness output."""

    def test_scorecard_from_fast_profile(self):
        """Scorecard from fast profile has expected structure."""
        harness = AdversarialHarness(seed=SEED_HARNESS)
        summaries = harness.run_with_profile("goal_hit", "fast")
        
        scorecard = build_robustness_scorecard(summaries)
        
        assert scorecard["metrics"]["goal_hit"]["total_tests"] > 0
        assert "fault" in scorecard["metrics"]["goal_hit"]["modes_run"]


# ===========================================================================
# TASK 3: REGRESSION RADAR SNAPSHOT GUARD TESTS
# ===========================================================================

@pytest.mark.oracle
class TestRadarSnapshotCreation:
    """Tests for radar snapshot creation."""

    def test_snapshot_has_schema_version(self):
        """Snapshot includes schema version."""
        results = run_regression_radar(seed=SEED_HARNESS)
        snapshot = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == "1.0.0"

    def test_snapshot_has_seed(self):
        """Snapshot includes seed for reproducibility."""
        results = run_regression_radar(seed=12345)
        snapshot = create_radar_snapshot(results, seed=12345)
        
        assert snapshot["seed"] == 12345

    def test_snapshot_has_timestamp(self):
        """Snapshot includes timestamp."""
        results = run_regression_radar(seed=SEED_HARNESS)
        snapshot = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        assert "timestamp" in snapshot
        assert snapshot["timestamp"].endswith("Z")

    def test_snapshot_has_summary_hash(self):
        """Snapshot includes summary hash."""
        results = run_regression_radar(seed=SEED_HARNESS)
        snapshot = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        assert "summary_hash" in snapshot
        assert len(snapshot["summary_hash"]) == 16

    def test_snapshot_results_sorted(self):
        """Snapshot results are sorted by metric_kind, test_name."""
        results = run_regression_radar(seed=SEED_HARNESS)
        snapshot = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        keys = [(r["metric_kind"], r["test_name"]) for r in snapshot["results"]]
        assert keys == sorted(keys)


@pytest.mark.oracle
class TestRadarSnapshotDeterminism:
    """Tests for snapshot determinism."""

    def test_snapshot_deterministic_for_fixed_seed(self):
        """Snapshot is deterministic for same seed."""
        results1 = run_regression_radar(seed=SEED_HARNESS)
        snapshot1 = create_radar_snapshot(results1, seed=SEED_HARNESS)
        
        results2 = run_regression_radar(seed=SEED_HARNESS)
        snapshot2 = create_radar_snapshot(results2, seed=SEED_HARNESS)
        
        # Compare summary hashes (timestamps will differ)
        assert snapshot1["summary_hash"] == snapshot2["summary_hash"]
        assert snapshot1["results"] == snapshot2["results"]

    def test_different_seeds_produce_same_boundary_tests(self):
        """
        Boundary tests are seed-independent (they use fixed inputs).
        """
        # Boundary tests don't depend on random generation
        results1 = run_regression_radar(seed=11111)
        results2 = run_regression_radar(seed=22222)
        
        # Extract boundary tests
        boundary1 = [r for r in results1 if "boundary" in r.test_name]
        boundary2 = [r for r in results2 if "boundary" in r.test_name]
        
        # These should match
        for r1, r2 in zip(
            sorted(boundary1, key=lambda x: x.test_name),
            sorted(boundary2, key=lambda x: x.test_name)
        ):
            assert r1.expected_hash == r2.expected_hash


@pytest.mark.oracle
class TestSnapshotRoundTrip:
    """Tests for snapshot save/load round-trip."""

    def test_snapshot_round_trip(self):
        """Snapshot survives JSON round-trip."""
        results = run_regression_radar(seed=SEED_HARNESS)
        snapshot = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        json_str = snapshot_to_json(snapshot)
        loaded = load_radar_snapshot(json_str)
        
        assert loaded["schema_version"] == snapshot["schema_version"]
        assert loaded["seed"] == snapshot["seed"]
        assert loaded["summary_hash"] == snapshot["summary_hash"]
        assert loaded["results"] == snapshot["results"]

    def test_snapshot_file_round_trip(self):
        """Snapshot survives file save/load."""
        results = run_regression_radar(seed=SEED_HARNESS)
        snapshot = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(snapshot_to_json(snapshot))
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as f:
                loaded = load_radar_snapshot(f.read())
            
            assert loaded["summary_hash"] == snapshot["summary_hash"]
        finally:
            os.unlink(temp_path)


@pytest.mark.oracle
class TestSnapshotComparison:
    """Tests for snapshot comparison."""

    def test_identical_snapshots_match(self):
        """Identical snapshots compare as matching."""
        results = run_regression_radar(seed=SEED_HARNESS)
        snapshot1 = create_radar_snapshot(results, seed=SEED_HARNESS)
        snapshot2 = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        match, differences = compare_radar_snapshots(snapshot1, snapshot2)
        
        assert match is True
        assert len(differences) == 0

    def test_hash_change_detected(self):
        """Changed hash is detected as mismatch."""
        results = run_regression_radar(seed=SEED_HARNESS)
        baseline = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        # Create altered snapshot
        current = create_radar_snapshot(results, seed=SEED_HARNESS)
        current["results"][0]["expected_hash"] = "altered_hash_abc"
        
        match, differences = compare_radar_snapshots(current, baseline)
        
        assert match is False
        assert len(differences) >= 1
        assert any(d["reason"] == "hash_changed" for d in differences)

    def test_new_test_detected(self):
        """New test in current is detected."""
        results = run_regression_radar(seed=SEED_HARNESS)
        baseline = create_radar_snapshot(results, seed=SEED_HARNESS)
        current = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        # Add new test to current
        current["results"].append({
            "metric_kind": "goal_hit",
            "test_name": "new_test",
            "expected_hash": "abc123",
            "actual_hash": "abc123",
            "match": True,
        })
        
        match, differences = compare_radar_snapshots(current, baseline)
        
        assert match is False
        assert any(d["reason"] == "new_test" for d in differences)

    def test_removed_test_detected(self):
        """Removed test is detected."""
        results = run_regression_radar(seed=SEED_HARNESS)
        baseline = create_radar_snapshot(results, seed=SEED_HARNESS)
        current = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        # Remove a test from current
        current["results"] = current["results"][1:]
        
        match, differences = compare_radar_snapshots(current, baseline)
        
        assert match is False
        assert any(d["reason"] == "test_removed" for d in differences)


@pytest.mark.oracle
class TestSnapshotDiffOutput:
    """Tests for snapshot diff output."""

    def test_diff_output_includes_metric_kind(self, capsys):
        """Diff output includes metric kind."""
        differences = [{
            "metric_kind": "goal_hit",
            "test_name": "test_boundary",
            "baseline_hash": "abc123",
            "current_hash": "xyz789",
            "reason": "hash_changed",
        }]
        
        print_snapshot_diff(differences)
        captured = capsys.readouterr()
        
        assert "goal_hit" in captured.out

    def test_diff_output_includes_test_name(self, capsys):
        """Diff output includes test name."""
        differences = [{
            "metric_kind": "density",
            "test_name": "boundary_at_threshold",
            "baseline_hash": "abc123",
            "current_hash": "xyz789",
            "reason": "hash_changed",
        }]
        
        print_snapshot_diff(differences)
        captured = capsys.readouterr()
        
        assert "boundary_at_threshold" in captured.out

    def test_diff_output_includes_both_hashes(self, capsys):
        """Diff output includes both baseline and current hashes."""
        differences = [{
            "metric_kind": "chain_length",
            "test_name": "linear_chain",
            "baseline_hash": "baseline_hash_123",
            "current_hash": "current_hash_456",
            "reason": "hash_changed",
        }]
        
        print_snapshot_diff(differences)
        captured = capsys.readouterr()
        
        assert "baseline_hash_123" in captured.out
        assert "current_hash_456" in captured.out

    def test_diff_output_no_stack_traces(self, capsys):
        """Diff output has no stack traces (clean error)."""
        differences = [{
            "metric_kind": "multi_goal",
            "test_name": "all_goals_met",
            "baseline_hash": "abc",
            "current_hash": "xyz",
            "reason": "hash_changed",
        }]
        
        print_snapshot_diff(differences)
        captured = capsys.readouterr()
        
        assert "Traceback" not in captured.out
        assert "Exception" not in captured.out


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

@pytest.mark.oracle
class TestScenarioLabIntegration:
    """Integration tests for the scenario lab."""

    def test_run_scenario_produces_summaries(self):
        """Running a scenario produces valid summaries."""
        scenario = get_scenario("ci_quick")
        harness = AdversarialHarness(seed=SEED_HARNESS)
        
        all_summaries = []
        n_tests = 20  # Small for test speed
        
        for mk in scenario.metric_kinds:
            for mode in scenario.modes:
                summaries = harness.run(mk, n_tests, mode)
                all_summaries.extend(summaries)
        
        assert len(all_summaries) > 0
        
        # Can build scorecard from summaries
        scorecard = build_robustness_scorecard(all_summaries, scenarios_run=["ci_quick"])
        assert "metrics" in scorecard

    def test_snapshot_guard_full_workflow(self):
        """Complete snapshot guard workflow."""
        # 1. Create baseline
        results = run_regression_radar(seed=SEED_HARNESS)
        baseline = create_radar_snapshot(results, seed=SEED_HARNESS)
        
        # 2. Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(snapshot_to_json(baseline))
            temp_path = f.name
        
        try:
            # 3. Load and check
            with open(temp_path, 'r') as f:
                loaded_baseline = load_radar_snapshot(f.read())
            
            # 4. Run current
            current_results = run_regression_radar(seed=SEED_HARNESS)
            current = create_radar_snapshot(current_results, seed=SEED_HARNESS)
            
            # 5. Compare
            match, differences = compare_radar_snapshots(current, loaded_baseline)
            
            assert match is True
            assert len(differences) == 0
        finally:
            os.unlink(temp_path)

