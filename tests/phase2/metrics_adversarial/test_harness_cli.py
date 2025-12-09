# tests/phase2/metrics_adversarial/test_harness_cli.py
"""
Tests for the Adversarial Harness CLI — Institutional Contracts

Covers:
- Profile contracts → test count minimums
- Coverage report CI contract → required keys
- Regression radar failure surface → exit codes and output format

NO METRIC INTERPRETATION: These tests verify CLI contract mechanics only.
"""

import json
import pytest
from typing import Dict, List, Tuple

from experiments.metrics_adversarial_harness import (
    PROFILE_CONFIG,
    PROFILE_CONTRACTS,
    METRIC_KINDS,
    COVERAGE_REPORT_REQUIRED_KEYS,
    AdversarialHarness,
    generate_coverage_report,
    coverage_report_to_json,
    validate_coverage_report,
    run_regression_radar,
    print_regression_radar_results,
    get_profile_test_count,
    get_profile_modes,
    get_profile_contract,
    get_profile_contract_json,
    validate_profile_meets_contract,
    InputGenerator,
    MetricExecutor,
    RegressionResult,
    _hash_result,
    SEED_HARNESS,
)


# ===========================================================================
# PROFILE CONTRACT TESTS
# ===========================================================================

@pytest.mark.mutation
class TestProfileContracts:
    """
    Verify profile → numeric test count mappings meet contracts.
    
    These tests ensure no profile silently regresses in coverage.
    """

    def test_fast_profile_meets_minimum_test_count(self):
        """fast profile yields >= 50 tests per metric (contract)."""
        contract = get_profile_contract("fast")
        config_tests = get_profile_test_count("fast")
        
        assert config_tests >= contract.min_tests_per_metric, (
            f"fast profile: {config_tests} tests < min {contract.min_tests_per_metric}"
        )

    def test_standard_profile_meets_minimum_test_count(self):
        """standard profile yields >= 250 tests per metric (contract)."""
        contract = get_profile_contract("standard")
        config_tests = get_profile_test_count("standard")
        
        assert config_tests >= contract.min_tests_per_metric, (
            f"standard profile: {config_tests} tests < min {contract.min_tests_per_metric}"
        )

    def test_full_profile_meets_minimum_test_count(self):
        """full profile yields >= 1000 tests per metric (contract)."""
        contract = get_profile_contract("full")
        config_tests = get_profile_test_count("full")
        
        assert config_tests >= contract.min_tests_per_metric, (
            f"full profile: {config_tests} tests < min {contract.min_tests_per_metric}"
        )

    def test_fast_profile_modes_match_contract(self):
        """fast profile runs exactly the contracted modes."""
        contract = get_profile_contract("fast")
        config_modes = set(get_profile_modes("fast"))
        contract_modes = set(contract.modes)
        
        assert contract_modes.issubset(config_modes), (
            f"fast missing modes: {contract_modes - config_modes}"
        )

    def test_standard_profile_modes_match_contract(self):
        """standard profile runs exactly the contracted modes."""
        contract = get_profile_contract("standard")
        config_modes = set(get_profile_modes("standard"))
        contract_modes = set(contract.modes)
        
        assert contract_modes.issubset(config_modes), (
            f"standard missing modes: {contract_modes - config_modes}"
        )

    def test_full_profile_modes_match_contract(self):
        """full profile runs exactly the contracted modes."""
        contract = get_profile_contract("full")
        config_modes = set(get_profile_modes("full"))
        contract_modes = set(contract.modes)
        
        assert contract_modes.issubset(config_modes), (
            f"full missing modes: {contract_modes - config_modes}"
        )

    def test_all_profiles_validate_against_contracts(self):
        """All profiles pass contract validation."""
        for profile in PROFILE_CONTRACTS.keys():
            passes, violations = validate_profile_meets_contract(profile)
            assert passes, f"Profile {profile} violates contract: {violations}"

    def test_profile_config_has_descriptions(self):
        """All profiles have descriptions."""
        for profile, config in PROFILE_CONFIG.items():
            assert "description" in config
            assert len(config["description"]) > 0

    def test_profile_contracts_json_export(self):
        """Profile contracts export as valid JSON."""
        json_str = get_profile_contract_json()
        parsed = json.loads(json_str)
        
        # All profiles present
        assert "fast" in parsed
        assert "standard" in parsed
        assert "full" in parsed
        
        # Each has required fields
        for profile, data in parsed.items():
            assert "min_tests_per_metric" in data
            assert "modes" in data
            assert "min_fault_types" in data
            assert "min_mutation_categories" in data

    def test_no_profile_regression_in_test_counts(self):
        """Profiles maintain or exceed their minimums (no regression)."""
        # This is a change-detection test
        expected_minimums = {
            "fast": 50,
            "standard": 250,
            "full": 1000,
        }
        
        for profile, expected_min in expected_minimums.items():
            actual = get_profile_test_count(profile)
            assert actual >= expected_min, (
                f"Profile {profile} regressed: {actual} < {expected_min}"
            )


# ===========================================================================
# COVERAGE REPORT CONTRACT TESTS
# ===========================================================================

@pytest.mark.oracle
class TestCoverageReportContract:
    """
    Tests for coverage report CI contract.
    
    Contract: Output contains all COVERAGE_REPORT_REQUIRED_KEYS.
    """

    def test_coverage_report_has_all_required_keys(self):
        """Coverage report JSON has all required keys."""
        report = generate_coverage_report()
        json_str = coverage_report_to_json(report)
        parsed = json.loads(json_str)
        
        for key in COVERAGE_REPORT_REQUIRED_KEYS:
            assert key in parsed, f"Missing required key: {key}"

    def test_coverage_report_includes_all_four_metrics(self):
        """Coverage report includes all four metric kinds."""
        report = generate_coverage_report()
        
        assert len(report.metric_kinds) == 4
        assert "goal_hit" in report.metric_kinds
        assert "density" in report.metric_kinds
        assert "chain_length" in report.metric_kinds
        assert "multi_goal" in report.metric_kinds

    def test_coverage_report_has_at_least_5_fault_types(self):
        """Coverage report has at least 5 fault types total."""
        report = generate_coverage_report()
        assert report.total_fault_types >= 5

    def test_no_empty_fault_lists(self):
        """Each metric has non-empty fault type list."""
        report = generate_coverage_report()
        
        for metric in METRIC_KINDS:
            assert metric in report.fault_types_per_metric
            assert len(report.fault_types_per_metric[metric]) > 0, (
                f"No fault types for {metric}"
            )

    def test_no_empty_mutation_lists(self):
        """Each metric has non-empty mutation category list."""
        report = generate_coverage_report()
        
        for metric in METRIC_KINDS:
            assert metric in report.mutation_categories_per_metric
            assert len(report.mutation_categories_per_metric[metric]) >= 2, (
                f"Insufficient mutation categories for {metric}"
            )

    def test_coverage_report_replay_sizes_positive(self):
        """Each metric has positive replay size."""
        report = generate_coverage_report()
        
        for metric in METRIC_KINDS:
            assert metric in report.replay_sizes_per_metric
            assert report.replay_sizes_per_metric[metric] > 0

    def test_coverage_report_to_json_valid(self):
        """Coverage report serializes to valid JSON."""
        report = generate_coverage_report()
        json_str = coverage_report_to_json(report)
        
        # Should parse without error
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_coverage_report_deterministic(self):
        """Coverage report is deterministic across runs."""
        report1 = generate_coverage_report()
        report2 = generate_coverage_report()
        
        json1 = coverage_report_to_json(report1)
        json2 = coverage_report_to_json(report2)
        
        assert json1 == json2, "Coverage report not deterministic"

    def test_coverage_report_validation_passes(self):
        """Coverage report passes validation."""
        report = generate_coverage_report()
        json_str = coverage_report_to_json(report)
        
        valid, violations = validate_coverage_report(json_str)
        assert valid, f"Validation failed: {violations}"

    def test_coverage_report_validation_catches_missing_keys(self):
        """Validation catches missing required keys."""
        incomplete = {"metric_kinds": ["goal_hit"]}
        json_str = json.dumps(incomplete)
        
        valid, violations = validate_coverage_report(json_str)
        assert not valid
        assert len(violations) > 0


# ===========================================================================
# REGRESSION RADAR FAILURE SURFACE TESTS
# ===========================================================================

@pytest.mark.oracle
class TestRegressionRadarFailureSurface:
    """
    Tests for regression radar failure surface.
    
    Contract:
    - On success: "REGRESSION RADAR: OK", exit code 0
    - On mismatch: Full diff, exit code 1
    """

    def test_regression_radar_success_exit_code_0(self):
        """Regression radar returns exit code 0 on success."""
        results = run_regression_radar()
        
        # All should match
        all_match = all(r.match for r in results)
        assert all_match, "Expected all tests to pass"
        
        # Simulate print_regression_radar_results logic
        exit_code = print_regression_radar_results(results, quiet=True)
        assert exit_code == 0

    def test_regression_radar_success_output_format(self, capsys):
        """Success output is single line 'REGRESSION RADAR: OK'."""
        results = run_regression_radar()
        exit_code = print_regression_radar_results(results, quiet=True)
        
        captured = capsys.readouterr()
        assert "REGRESSION RADAR: OK" in captured.out
        assert exit_code == 0

    def test_regression_radar_mismatch_exit_code_1(self):
        """Regression radar returns exit code 1 on mismatch."""
        # Create fake mismatch result
        fake_results = [
            RegressionResult(
                metric_kind="goal_hit",
                test_name="simulated_failure",
                expected_hash="abc123",
                actual_hash="xyz789",
                match=False,
            )
        ]
        
        exit_code = print_regression_radar_results(fake_results, quiet=True)
        assert exit_code == 1

    def test_regression_radar_mismatch_output_has_metric_name(self, capsys):
        """Mismatch output includes metric name."""
        fake_results = [
            RegressionResult(
                metric_kind="goal_hit",
                test_name="simulated_failure",
                expected_hash="abc123",
                actual_hash="xyz789",
                match=False,
            )
        ]
        
        print_regression_radar_results(fake_results, quiet=False)
        captured = capsys.readouterr()
        
        assert "goal_hit" in captured.out

    def test_regression_radar_mismatch_output_has_test_label(self, capsys):
        """Mismatch output includes test case label."""
        fake_results = [
            RegressionResult(
                metric_kind="density",
                test_name="boundary_failure_test",
                expected_hash="abc123",
                actual_hash="xyz789",
                match=False,
            )
        ]
        
        print_regression_radar_results(fake_results, quiet=False)
        captured = capsys.readouterr()
        
        assert "boundary_failure_test" in captured.out

    def test_regression_radar_mismatch_output_has_both_hashes(self, capsys):
        """Mismatch output includes expected and actual hashes."""
        fake_results = [
            RegressionResult(
                metric_kind="chain_length",
                test_name="hash_diff_test",
                expected_hash="expected_hash_abc",
                actual_hash="actual_hash_xyz",
                match=False,
            )
        ]
        
        print_regression_radar_results(fake_results, quiet=False)
        captured = capsys.readouterr()
        
        assert "expected_hash_abc" in captured.out
        assert "actual_hash_xyz" in captured.out

    def test_regression_radar_mismatch_says_mismatch(self, capsys):
        """Mismatch output explicitly says MISMATCH."""
        fake_results = [
            RegressionResult(
                metric_kind="multi_goal",
                test_name="failure",
                expected_hash="a",
                actual_hash="b",
                match=False,
            )
        ]
        
        print_regression_radar_results(fake_results, quiet=False)
        captured = capsys.readouterr()
        
        assert "REGRESSION RADAR: MISMATCH" in captured.out

    def test_regression_radar_deterministic(self):
        """Regression radar produces same results on repeated runs."""
        results1 = run_regression_radar(seed=SEED_HARNESS)
        results2 = run_regression_radar(seed=SEED_HARNESS)
        
        assert len(results1) == len(results2)
        
        for r1, r2 in zip(results1, results2):
            assert r1.metric_kind == r2.metric_kind
            assert r1.test_name == r2.test_name
            assert r1.expected_hash == r2.expected_hash
            assert r1.actual_hash == r2.actual_hash
            assert r1.match == r2.match

    def test_regression_radar_covers_all_metrics(self):
        """Regression radar covers all four metric kinds."""
        results = run_regression_radar()
        covered_metrics = {r.metric_kind for r in results}
        
        for metric in METRIC_KINDS:
            assert metric in covered_metrics, f"Missing metric: {metric}"

    def test_regression_radar_has_boundary_tests(self):
        """Regression radar includes boundary tests."""
        results = run_regression_radar()
        test_names = [r.test_name for r in results]
        
        boundary_tests = [t for t in test_names if "boundary" in t]
        assert len(boundary_tests) >= 2

    def test_regression_radar_has_high_volume_tests(self):
        """Regression radar includes high-volume determinism tests."""
        results = run_regression_radar()
        test_names = [r.test_name for r in results]
        
        high_volume_tests = [t for t in test_names if "high_volume" in t]
        assert len(high_volume_tests) >= len(METRIC_KINDS)


# ===========================================================================
# REGRESSION MISMATCH SIMULATION
# ===========================================================================

@pytest.mark.oracle
class TestRegressionMismatchSimulation:
    """Tests simulating regression detection."""

    def test_mismatch_detected_via_hash_difference(self):
        """
        Demonstrates that hash differences are detected.
        
        This test creates results with different expected/actual hashes
        and verifies the mismatch is properly flagged.
        """
        # Normal matching result
        matching_result = (True, 5.0)
        matching = RegressionResult(
            metric_kind="goal_hit",
            test_name="matching_test",
            expected_hash=_hash_result(matching_result),
            actual_hash=_hash_result(matching_result),
            match=True,
        )
        assert matching.match is True
        assert matching.expected_hash == matching.actual_hash
        
        # Mismatching result
        expected = (True, 5.0)
        actual = (False, 5.0)
        mismatching = RegressionResult(
            metric_kind="goal_hit",
            test_name="mismatching_test",
            expected_hash=_hash_result(expected),
            actual_hash=_hash_result(actual),
            match=_hash_result(expected) == _hash_result(actual),
        )
        assert mismatching.match is False
        assert mismatching.expected_hash != mismatching.actual_hash

    def test_hash_function_is_deterministic(self):
        """Hash function produces same output for same input."""
        result = (True, 42.0)
        
        hash1 = _hash_result(result)
        hash2 = _hash_result(result)
        
        assert hash1 == hash2

    def test_hash_function_distinguishes_bool_difference(self):
        """Hash function distinguishes (True, x) from (False, x)."""
        result_true = (True, 10.0)
        result_false = (False, 10.0)
        
        assert _hash_result(result_true) != _hash_result(result_false)

    def test_hash_function_distinguishes_value_difference(self):
        """Hash function distinguishes (b, x) from (b, y)."""
        result_10 = (True, 10.0)
        result_11 = (True, 11.0)
        
        assert _hash_result(result_10) != _hash_result(result_11)


# ===========================================================================
# HARNESS PROFILE EXECUTION TESTS
# ===========================================================================

@pytest.mark.mutation
class TestHarnessProfileExecution:
    """Tests for harness execution with profiles."""

    def test_harness_run_with_fast_profile(self):
        """Harness runs successfully with fast profile."""
        harness = AdversarialHarness(seed=SEED_HARNESS)
        summaries = harness.run_with_profile("goal_hit", "fast")
        
        assert len(summaries) > 0
        total_tests = sum(s.total_cases for s in summaries)
        assert total_tests >= 50

    def test_harness_run_with_standard_profile(self):
        """Harness runs successfully with standard profile."""
        harness = AdversarialHarness(seed=SEED_HARNESS)
        summaries = harness.run_with_profile("density", "standard")
        
        assert len(summaries) > 0
        total_tests = sum(s.total_cases for s in summaries)
        assert total_tests >= 250

    def test_harness_profile_modes_match_config(self):
        """Harness respects profile mode configuration."""
        harness = AdversarialHarness(seed=SEED_HARNESS)
        summaries = harness.run_with_profile("goal_hit", "fast")
        modes_run = {s.mode for s in summaries}
        
        expected_modes = set(PROFILE_CONFIG["fast"]["modes"])
        assert modes_run == expected_modes


# ===========================================================================
# INPUT GENERATOR DETERMINISM
# ===========================================================================

@pytest.mark.entropy
class TestInputGeneratorDeterminism:
    """Verify input generator determinism."""

    def test_input_generator_deterministic(self):
        """Input generator produces same sequence with same seed."""
        gen1 = InputGenerator(seed=12345)
        gen2 = InputGenerator(seed=12345)
        
        for metric in METRIC_KINDS:
            for _ in range(10):
                inputs1 = gen1.generate_inputs(metric)
                inputs2 = gen2.generate_inputs(metric)
                assert inputs1 == inputs2

    def test_input_generator_reset_restores_sequence(self):
        """Input generator reset restores initial sequence."""
        gen = InputGenerator(seed=12345)
        
        first_run = [gen.generate_inputs("goal_hit") for _ in range(5)]
        gen.reset()
        second_run = [gen.generate_inputs("goal_hit") for _ in range(5)]
        
        assert first_run == second_run

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds produce different input sequences."""
        gen1 = InputGenerator(seed=11111)
        gen2 = InputGenerator(seed=22222)
        
        inputs1 = gen1.generate_inputs("goal_hit")
        inputs2 = gen2.generate_inputs("goal_hit")
        
        # Should be different (with overwhelming probability)
        assert inputs1 != inputs2
