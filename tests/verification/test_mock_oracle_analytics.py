"""
Tests for Mock Oracle Scenario Analytics & Drift Detection.

Verifies:
1. summarize_scenario_results() produces correct analytics
2. detect_scenario_drift() correctly identifies drift states
3. CLI --drift-check works for CI integration
4. Drift detection thresholds behave correctly

SAFEGUARD: These tests verify test infrastructure — no production impact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import List

import pytest

from backend.verification.mock_config import (
    ANALYTICS_SCHEMA_VERSION,
    DRIFT_STATUS_BROKEN,
    DRIFT_STATUS_DRIFTED,
    DRIFT_STATUS_IN_CONTRACT,
    MOCK_ORACLE_CONTRACT_VERSION,
    MockOracleConfig,
    MockVerificationResult,
    PROFILE_CONTRACTS,
    SCENARIOS,
    detect_scenario_drift,
    export_mock_oracle_contract,
    get_scenario,
    summarize_scenario_results,
)
from backend.verification.mock_oracle import MockVerifiableOracle


class TestSummarizeScenarioResults:
    """Tests for summarize_scenario_results() function."""

    def test_summary_has_schema_version(self):
        """Summary includes schema_version."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(10)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert "schema_version" in summary
        assert summary["schema_version"] == ANALYTICS_SCHEMA_VERSION

    def test_summary_has_scenario_name(self):
        """Summary includes scenario_name."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(10)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert summary["scenario_name"] == "default_sanity"

    def test_summary_has_sample_count(self):
        """Summary includes sample_count."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(25)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert summary["sample_count"] == 25

    def test_summary_has_bucket_counts(self):
        """Summary includes raw bucket_counts."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(100)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert "bucket_counts" in summary
        counts = summary["bucket_counts"]
        
        # All buckets should be present
        for bucket in ["verified", "failed", "abstain", "timeout", "error", "crash", "negative_control"]:
            assert bucket in counts
        
        # Total should match sample count
        total = sum(counts.values())
        assert total == 100

    def test_summary_has_empirical_distribution(self):
        """Summary includes empirical_distribution as percentages."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(100)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert "empirical_distribution" in summary
        emp = summary["empirical_distribution"]
        
        # Percentages should sum to 100 (within rounding)
        total = sum(emp.values())
        assert abs(total - 100.0) < 0.1

    def test_summary_has_expected_distribution(self):
        """Summary includes expected_distribution from contract."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(100)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert "expected_distribution" in summary
        exp = summary["expected_distribution"]
        
        # Should match profile contract
        profile_dist = PROFILE_CONTRACTS["default"]
        assert exp["verified"] == profile_dist["verified"]
        assert exp["failed"] == profile_dist["failed"]

    def test_summary_has_deltas(self):
        """Summary includes deltas between empirical and expected."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(100)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert "deltas" in summary
        deltas = summary["deltas"]
        
        # Deltas should be empirical - expected
        for bucket in ["verified", "failed", "abstain"]:
            expected_delta = summary["empirical_distribution"][bucket] - summary["expected_distribution"][bucket]
            assert abs(deltas[bucket] - expected_delta) < 0.01

    def test_summary_has_max_delta(self):
        """Summary includes max_delta."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(100)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert "max_delta" in summary
        assert summary["max_delta"] >= 0

    def test_summary_has_contract_respected(self):
        """Summary includes contract_respected boolean."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify(f"formula_{i}") for i in range(100)]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert "contract_respected" in summary
        assert isinstance(summary["contract_respected"], bool)

    def test_summary_raises_on_unknown_scenario(self):
        """Raises KeyError for unknown scenario."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify("formula")]
        
        with pytest.raises(KeyError, match="not found"):
            summarize_scenario_results("nonexistent_scenario", outcomes)

    def test_summary_raises_on_empty_outcomes(self):
        """Raises ValueError for empty outcomes."""
        with pytest.raises(ValueError, match="empty outcomes"):
            summarize_scenario_results("default_sanity", [])

    def test_summary_negative_control_scenario(self):
        """Summary for NC scenario has correct expected distribution."""
        oracle = MockVerifiableOracle(MockOracleConfig(negative_control=True))
        outcomes = [oracle.verify(f"formula_{i}") for i in range(50)]
        
        summary = summarize_scenario_results("nc_baseline", outcomes)
        
        # NC scenario should expect 100% in negative_control bucket
        exp = summary["expected_distribution"]
        assert exp["negative_control"] == 100.0
        assert exp["verified"] == 0.0
        
        # Empirical should also be 100% NC
        emp = summary["empirical_distribution"]
        assert emp["negative_control"] == 100.0

    def test_summary_deterministic(self):
        """Same inputs produce identical summaries."""
        oracle = MockVerifiableOracle(MockOracleConfig(seed=42))
        outcomes = [oracle.verify(f"formula_{i}") for i in range(100)]
        
        summary1 = summarize_scenario_results("default_sanity", outcomes)
        summary2 = summarize_scenario_results("default_sanity", outcomes)
        
        assert summary1 == summary2

    def test_summary_drift_tolerance_scales_with_samples(self):
        """Drift tolerance adjusts based on sample size."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        
        # Small sample
        outcomes_small = [oracle.verify(f"small_{i}") for i in range(10)]
        summary_small = summarize_scenario_results("default_sanity", outcomes_small)
        
        # Large sample
        outcomes_large = [oracle.verify(f"large_{i}") for i in range(1000)]
        summary_large = summarize_scenario_results("default_sanity", outcomes_large)
        
        # Smaller samples should have higher tolerance
        assert summary_small["drift_tolerance"] > summary_large["drift_tolerance"]


class TestDetectScenarioDrift:
    """Tests for detect_scenario_drift() function."""

    def _make_summary(
        self,
        scenario_name: str = "default_sanity",
        empirical: dict = None,
        sample_count: int = 100,
    ) -> dict:
        """Helper to create a summary dict."""
        if empirical is None:
            empirical = PROFILE_CONTRACTS.get("default", {}).copy()
            empirical["negative_control"] = 0.0
        
        expected = PROFILE_CONTRACTS.get("default", {}).copy()
        expected["negative_control"] = 0.0
        
        deltas = {k: empirical.get(k, 0) - expected.get(k, 0) for k in empirical}
        
        return {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "scenario_name": scenario_name,
            "sample_count": sample_count,
            "empirical_distribution": empirical,
            "expected_distribution": expected,
            "deltas": deltas,
            "max_delta": max(abs(d) for d in deltas.values()),
            "contract_respected": True,
        }

    def test_drift_in_contract(self):
        """Detects IN_CONTRACT when within thresholds."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary()  # No drift
        
        report = detect_scenario_drift(contract, summary)
        
        assert report["status"] == DRIFT_STATUS_IN_CONTRACT

    def test_drift_drifted(self):
        """Detects DRIFTED when exceeding warning threshold."""
        contract = export_mock_oracle_contract()
        
        # Create summary with 5% drift on verified
        empirical = PROFILE_CONTRACTS.get("default", {}).copy()
        empirical["verified"] = 65.0  # +5% from expected 60%
        empirical["negative_control"] = 0.0
        
        summary = self._make_summary(empirical=empirical)
        
        report = detect_scenario_drift(
            contract, summary, warning_threshold=3.0, broken_threshold=10.0
        )
        
        assert report["status"] == DRIFT_STATUS_DRIFTED

    def test_drift_broken(self):
        """Detects BROKEN when exceeding broken threshold."""
        contract = export_mock_oracle_contract()
        
        # Create summary with 15% drift on verified
        empirical = PROFILE_CONTRACTS.get("default", {}).copy()
        empirical["verified"] = 75.0  # +15% from expected 60%
        empirical["negative_control"] = 0.0
        
        summary = self._make_summary(empirical=empirical)
        
        report = detect_scenario_drift(
            contract, summary, warning_threshold=3.0, broken_threshold=10.0
        )
        
        assert report["status"] == DRIFT_STATUS_BROKEN

    def test_drift_report_has_contract_version(self):
        """Report includes contract_version."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary()
        
        report = detect_scenario_drift(contract, summary)
        
        assert report["contract_version"] == MOCK_ORACLE_CONTRACT_VERSION

    def test_drift_report_has_scenario_name(self):
        """Report includes scenario_name."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary(scenario_name="test_scenario")
        
        report = detect_scenario_drift(contract, summary)
        
        assert report["scenario_name"] == "test_scenario"

    def test_drift_report_has_sample_count(self):
        """Report includes sample_count."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary(sample_count=500)
        
        report = detect_scenario_drift(contract, summary)
        
        assert report["sample_count"] == 500

    def test_drift_report_has_drift_signals(self):
        """Report includes drift_signals list."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary()
        
        report = detect_scenario_drift(contract, summary)
        
        assert "drift_signals" in report
        assert isinstance(report["drift_signals"], list)

    def test_drift_signals_contain_details(self):
        """Drift signals contain bucket, delta, expected, observed."""
        contract = export_mock_oracle_contract()
        
        # Create drift
        empirical = PROFILE_CONTRACTS.get("default", {}).copy()
        empirical["verified"] = 68.0  # +8% drift
        empirical["negative_control"] = 0.0
        summary = self._make_summary(empirical=empirical)
        
        report = detect_scenario_drift(contract, summary, warning_threshold=3.0)
        
        assert len(report["drift_signals"]) > 0
        signal = report["drift_signals"][0]
        
        assert "bucket" in signal
        assert "delta" in signal
        assert "expected" in signal
        assert "observed" in signal
        assert "severity" in signal

    def test_drift_report_has_max_drift(self):
        """Report includes max_drift."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary()
        
        report = detect_scenario_drift(contract, summary)
        
        assert "max_drift" in report
        assert report["max_drift"] >= 0

    def test_drift_report_has_thresholds(self):
        """Report includes thresholds used."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary()
        
        report = detect_scenario_drift(
            contract, summary, warning_threshold=5.0, broken_threshold=15.0
        )
        
        assert "thresholds" in report
        assert report["thresholds"]["warning"] == 5.0
        assert report["thresholds"]["broken"] == 15.0

    def test_drift_report_has_recommended_action(self):
        """Report includes recommended_action."""
        contract = export_mock_oracle_contract()
        summary = self._make_summary()
        
        report = detect_scenario_drift(contract, summary)
        
        assert "recommended_action" in report
        assert len(report["recommended_action"]) > 0

    def test_recommended_action_varies_by_status(self):
        """Recommended action differs based on drift status."""
        contract = export_mock_oracle_contract()
        
        # IN_CONTRACT
        summary_ok = self._make_summary()
        report_ok = detect_scenario_drift(contract, summary_ok)
        
        # DRIFTED
        emp_drifted = PROFILE_CONTRACTS.get("default", {}).copy()
        emp_drifted["verified"] = 65.0
        emp_drifted["negative_control"] = 0.0
        summary_drifted = self._make_summary(empirical=emp_drifted)
        report_drifted = detect_scenario_drift(
            contract, summary_drifted, warning_threshold=3.0
        )
        
        # BROKEN
        emp_broken = PROFILE_CONTRACTS.get("default", {}).copy()
        emp_broken["verified"] = 80.0
        emp_broken["negative_control"] = 0.0
        summary_broken = self._make_summary(empirical=emp_broken)
        report_broken = detect_scenario_drift(
            contract, summary_broken, warning_threshold=3.0, broken_threshold=10.0
        )
        
        # All should have different recommendations
        assert "No action" in report_ok["recommended_action"]
        # Check for key words in drifted recommendation (case insensitive)
        assert "drift" in report_drifted["recommended_action"].lower()
        assert "CRITICAL" in report_broken["recommended_action"]


class TestIntegrationAnalyticsWithOracle:
    """Integration tests: analytics with real oracle runs."""

    def test_high_sample_count_respects_contract(self):
        """With enough samples, distribution should match contract."""
        oracle = MockVerifiableOracle(MockOracleConfig(seed=42))
        outcomes = [oracle.verify(f"integration_test_{i}") for i in range(1000)]
        
        summary = summarize_scenario_results("default_stress", outcomes)
        
        # With 1000 samples, should be within reasonable tolerance
        assert summary["contract_respected"]
        assert summary["max_delta"] < 10.0  # Conservative check

    def test_nc_scenario_always_respects_contract(self):
        """NC scenarios always have zero drift."""
        oracle = MockVerifiableOracle(MockOracleConfig(negative_control=True))
        outcomes = [oracle.verify(f"nc_test_{i}") for i in range(200)]
        
        summary = summarize_scenario_results("nc_baseline", outcomes)
        
        # NC should have exactly 0 drift for all buckets
        assert summary["max_delta"] == 0.0
        assert summary["contract_respected"]

    def test_analytics_pipeline_end_to_end(self):
        """Full pipeline: oracle → summary → drift detection."""
        oracle = MockVerifiableOracle(MockOracleConfig(seed=123))
        outcomes = [oracle.verify(f"e2e_test_{i}") for i in range(500)]
        
        # Step 1: Summarize
        summary = summarize_scenario_results("default_stress", outcomes)
        assert summary["sample_count"] == 500
        
        # Step 2: Get contract
        contract = export_mock_oracle_contract()
        
        # Step 3: Detect drift (use wider thresholds for sampling variance)
        report = detect_scenario_drift(
            contract, summary, warning_threshold=6.0, broken_threshold=15.0
        )
        
        # Should be IN_CONTRACT with reasonable samples and wider thresholds
        assert report["status"] == DRIFT_STATUS_IN_CONTRACT


class TestCLIDriftCheck:
    """Tests for CLI --drift-check mode."""

    def test_cli_drift_check_passes(self):
        """--drift-check passes for valid scenario with appropriate thresholds."""
        # Use higher sample count and wider thresholds to account for sampling variance
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--drift-check", "default_sanity",
                "--drift-samples", "500",
                "--drift-seed", "42",
                "--warning-threshold", "6.0",
                "--broken-threshold", "15.0",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "Drift Check: default_sanity" in result.stdout
        assert "IN_CONTRACT" in result.stdout or "[PASS]" in result.stdout

    def test_cli_drift_check_json(self):
        """--drift-check --json outputs valid JSON."""
        # Use NC scenario for guaranteed pass, or wider thresholds
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--drift-check", "nc_baseline",
                "--drift-samples", "100",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        data = json.loads(result.stdout)
        assert "analytics" in data
        assert "drift_report" in data
        assert "config" in data
        assert "passed" in data

    def test_cli_drift_check_nc_scenario(self):
        """--drift-check works for NC scenarios."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--drift-check", "nc_baseline",
                "--drift-samples", "50",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        data = json.loads(result.stdout)
        assert data["passed"]
        assert data["drift_report"]["status"] == DRIFT_STATUS_IN_CONTRACT

    def test_cli_drift_check_invalid_scenario(self):
        """--drift-check with invalid scenario returns exit code 3."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--drift-check", "nonexistent_scenario",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Should fail with exit code 3 (contract violation/error)
        assert result.returncode == 3
        
        data = json.loads(result.stdout)
        assert "error" in data
        assert not data["passed"]

    def test_cli_drift_check_custom_thresholds(self):
        """--drift-check respects custom thresholds."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--drift-check", "default_sanity",
                "--drift-samples", "100",
                "--warning-threshold", "1.0",
                "--broken-threshold", "5.0",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # May pass or fail depending on sampling variance
        # Just verify it runs and uses the thresholds
        data = json.loads(result.stdout)
        assert data["config"]["warning_threshold"] == 1.0
        assert data["config"]["broken_threshold"] == 5.0

    def test_cli_drift_check_custom_seed(self):
        """--drift-check uses custom seed for reproducibility."""
        result1 = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--drift-check", "default_sanity",
                "--drift-samples", "100",
                "--drift-seed", "999",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        result2 = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--drift-check", "default_sanity",
                "--drift-samples", "100",
                "--drift-seed", "999",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result1.returncode == result2.returncode
        
        data1 = json.loads(result1.stdout)
        data2 = json.loads(result2.stdout)
        
        # Same seed should produce identical analytics
        assert data1["analytics"] == data2["analytics"]

    def test_cli_drift_check_all_scenarios(self):
        """--drift-check works for all predefined scenarios."""
        for scenario_name in SCENARIOS:
            scenario = get_scenario(scenario_name)
            # Use higher samples and wider thresholds for non-NC scenarios
            samples = "100" if scenario.negative_control else "500"
            warning = "1.0" if scenario.negative_control else "7.0"
            broken = "5.0" if scenario.negative_control else "15.0"
            
            result = subprocess.run(
                [
                    sys.executable, "-m", "backend.verification.mock_oracle_cli",
                    "--drift-check", scenario_name,
                    "--drift-samples", samples,
                    "--warning-threshold", warning,
                    "--broken-threshold", broken,
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            assert result.returncode == 0, f"Failed for scenario: {scenario_name}"
            
            data = json.loads(result.stdout)
            assert data["passed"], f"Drift detected for scenario: {scenario_name}"


class TestDriftStatusConstants:
    """Tests for drift status constants."""

    def test_status_values(self):
        """Drift status constants have expected values."""
        assert DRIFT_STATUS_IN_CONTRACT == "IN_CONTRACT"
        assert DRIFT_STATUS_DRIFTED == "DRIFTED"
        assert DRIFT_STATUS_BROKEN == "BROKEN"

    def test_status_distinct(self):
        """All status values are distinct."""
        statuses = [DRIFT_STATUS_IN_CONTRACT, DRIFT_STATUS_DRIFTED, DRIFT_STATUS_BROKEN]
        assert len(set(statuses)) == 3


class TestAnalyticsSchemaVersion:
    """Tests for analytics schema versioning."""

    def test_schema_version_exists(self):
        """ANALYTICS_SCHEMA_VERSION is defined."""
        assert ANALYTICS_SCHEMA_VERSION is not None
        assert len(ANALYTICS_SCHEMA_VERSION) > 0

    def test_schema_version_format(self):
        """Schema version follows semver format."""
        parts = ANALYTICS_SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_schema_version_in_summary(self):
        """Summary includes schema_version field."""
        oracle = MockVerifiableOracle(MockOracleConfig())
        outcomes = [oracle.verify("test")]
        
        summary = summarize_scenario_results("default_sanity", outcomes)
        
        assert summary["schema_version"] == ANALYTICS_SCHEMA_VERSION

