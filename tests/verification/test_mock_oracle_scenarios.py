"""
Tests for Mock Oracle Scenario Layer.

Verifies:
1. Scenario dataclass behavior
2. SCENARIOS registry determinism
3. list_scenarios() tag filtering
4. get_scenario() lookup
5. CLI --scenario and --list-scenarios
6. Contract export determinism

SAFEGUARD: These tests verify test infrastructure — no production impact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Set

import pytest

from backend.verification.mock_config import (
    MOCK_ORACLE_CONTRACT_VERSION,
    PROFILE_CONTRACTS,
    SCENARIOS,
    SLICE_PROFILES,
    Scenario,
    export_mock_oracle_contract,
    get_scenario,
    list_scenarios,
)


class TestScenarioDataclass:
    """Tests for Scenario dataclass validation and behavior."""

    def test_scenario_fields(self):
        """Scenario has all expected fields."""
        s = Scenario(
            name="test_scenario",
            profile="default",
            tags=frozenset({"sanity", "test"}),
            description="A test scenario",
        )
        
        assert s.name == "test_scenario"
        assert s.profile == "default"
        assert s.tags == frozenset({"sanity", "test"})
        assert s.description == "A test scenario"
        assert s.negative_control is False  # Default
        assert s.samples_default == 100  # Default

    def test_scenario_with_negative_control(self):
        """Scenario can be configured for negative control."""
        s = Scenario(
            name="nc_test",
            profile="default",
            tags=frozenset({"nc"}),
            description="NC test",
            negative_control=True,
            samples_default=50,
        )
        
        assert s.negative_control is True
        assert s.samples_default == 50

    def test_scenario_immutable(self):
        """Scenario is frozen (immutable)."""
        s = Scenario(
            name="test",
            profile="default",
            tags=frozenset({"test"}),
            description="Test",
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            s.name = "modified"  # type: ignore

    def test_scenario_invalid_profile(self):
        """Scenario rejects invalid profile names."""
        with pytest.raises(ValueError, match="Invalid profile 'nonexistent'"):
            Scenario(
                name="bad",
                profile="nonexistent",
                tags=frozenset({"test"}),
                description="Invalid",
            )

    def test_scenario_empty_name(self):
        """Scenario rejects empty name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Scenario(
                name="",
                profile="default",
                tags=frozenset({"test"}),
                description="Empty name",
            )

    def test_scenario_invalid_samples_default(self):
        """Scenario rejects samples_default < 1."""
        with pytest.raises(ValueError, match="samples_default must be at least 1"):
            Scenario(
                name="bad",
                profile="default",
                tags=frozenset({"test"}),
                description="Bad samples",
                samples_default=0,
            )

    def test_scenario_to_dict(self):
        """Scenario.to_dict() produces expected structure."""
        s = Scenario(
            name="test",
            profile="goal_hit",
            tags=frozenset({"a", "b", "c"}),
            description="Test description",
            negative_control=True,
            samples_default=200,
        )
        
        d = s.to_dict()
        
        assert d["name"] == "test"
        assert d["profile"] == "goal_hit"
        assert d["tags"] == ["a", "b", "c"]  # Sorted
        assert d["description"] == "Test description"
        assert d["negative_control"] is True
        assert d["samples_default"] == 200

    def test_scenario_to_dict_deterministic(self):
        """Scenario.to_dict() is deterministic."""
        s = Scenario(
            name="test",
            profile="default",
            tags=frozenset({"z", "a", "m"}),
            description="Test",
        )
        
        d1 = s.to_dict()
        d2 = s.to_dict()
        
        assert d1 == d2
        assert d1["tags"] == ["a", "m", "z"]  # Always sorted


class TestScenariosRegistry:
    """Tests for the SCENARIOS registry."""

    def test_scenarios_not_empty(self):
        """SCENARIOS registry is not empty."""
        assert len(SCENARIOS) > 0

    def test_all_scenarios_have_valid_profiles(self):
        """All scenarios reference valid profiles."""
        for name, scenario in SCENARIOS.items():
            assert scenario.profile in SLICE_PROFILES, (
                f"Scenario '{name}' references invalid profile '{scenario.profile}'"
            )

    def test_scenario_names_match_keys(self):
        """Scenario names match their registry keys."""
        for name, scenario in SCENARIOS.items():
            assert scenario.name == name, (
                f"Registry key '{name}' doesn't match scenario.name '{scenario.name}'"
            )

    def test_scenarios_registry_deterministic(self):
        """SCENARIOS registry is deterministic."""
        keys1 = list(SCENARIOS.keys())
        keys2 = list(SCENARIOS.keys())
        assert keys1 == keys2

    def test_expected_scenarios_exist(self):
        """Expected predefined scenarios exist."""
        expected = [
            "default_sanity",
            "default_stress",
            "goal_hit_stress",
            "goal_hit_sanity",
            "sparse_exploration",
            "tree_chain_building",
            "dependency_coordination",
            "nc_baseline",
            "nc_stress",
        ]
        
        for name in expected:
            assert name in SCENARIOS, f"Expected scenario '{name}' not found"

    def test_negative_control_scenarios_have_flag(self):
        """Scenarios with 'negative_control' tag have negative_control=True."""
        for name, scenario in SCENARIOS.items():
            if "negative_control" in scenario.tags:
                assert scenario.negative_control is True, (
                    f"Scenario '{name}' has 'negative_control' tag but flag is False"
                )


class TestListScenarios:
    """Tests for list_scenarios() function."""

    def test_list_all_scenarios(self):
        """list_scenarios() returns all scenarios when no filter."""
        scenarios = list_scenarios()
        assert len(scenarios) == len(SCENARIOS)

    def test_list_scenarios_sorted(self):
        """list_scenarios() returns scenarios sorted by name."""
        scenarios = list_scenarios()
        names = [s.name for s in scenarios]
        assert names == sorted(names)

    def test_filter_by_single_tag(self):
        """Filtering by a single tag works."""
        scenarios = list_scenarios({"sanity"})
        
        assert len(scenarios) > 0
        for s in scenarios:
            assert "sanity" in s.tags

    def test_filter_by_multiple_tags(self):
        """Filtering by multiple tags requires ALL tags."""
        scenarios = list_scenarios({"stress", "default"})
        
        assert len(scenarios) > 0
        for s in scenarios:
            assert "stress" in s.tags
            assert "default" in s.tags

    def test_filter_no_match(self):
        """Filtering with non-matching tags returns empty list."""
        scenarios = list_scenarios({"nonexistent_tag_xyz"})
        assert len(scenarios) == 0

    def test_filter_empty_set(self):
        """Empty filter set returns all scenarios."""
        scenarios = list_scenarios(set())
        assert len(scenarios) == len(SCENARIOS)

    def test_filter_negative_control(self):
        """Can filter for negative control scenarios."""
        scenarios = list_scenarios({"negative_control"})
        
        assert len(scenarios) >= 2  # nc_baseline and nc_stress
        for s in scenarios:
            assert s.negative_control is True


class TestGetScenario:
    """Tests for get_scenario() function."""

    def test_get_existing_scenario(self):
        """get_scenario() returns scenario by name."""
        scenario = get_scenario("default_sanity")
        
        assert scenario.name == "default_sanity"
        assert scenario.profile == "default"

    def test_get_nonexistent_scenario(self):
        """get_scenario() raises KeyError for unknown name."""
        with pytest.raises(KeyError, match="not found"):
            get_scenario("nonexistent_scenario")

    def test_get_scenario_error_lists_available(self):
        """KeyError message includes available scenario names."""
        with pytest.raises(KeyError) as exc_info:
            get_scenario("unknown")
        
        error_msg = str(exc_info.value)
        assert "Available" in error_msg


class TestExportMockOracleContract:
    """Tests for export_mock_oracle_contract() function."""

    def test_export_has_contract_version(self):
        """Export includes contract_version."""
        contract = export_mock_oracle_contract()
        assert "contract_version" in contract
        assert contract["contract_version"] == MOCK_ORACLE_CONTRACT_VERSION

    def test_export_has_all_profiles(self):
        """Export includes all profile definitions."""
        contract = export_mock_oracle_contract()
        
        assert "profiles" in contract
        for profile_name in PROFILE_CONTRACTS:
            assert profile_name in contract["profiles"]

    def test_export_profiles_have_boundaries_and_distribution(self):
        """Each profile has boundaries and distribution."""
        contract = export_mock_oracle_contract()
        
        for profile_name, profile_data in contract["profiles"].items():
            assert "boundaries" in profile_data
            assert "distribution" in profile_data
            
            # Verify boundaries match SLICE_PROFILES
            assert profile_data["boundaries"] == SLICE_PROFILES[profile_name]
            
            # Verify distribution matches PROFILE_CONTRACTS
            assert profile_data["distribution"] == PROFILE_CONTRACTS[profile_name]

    def test_export_has_negative_control(self):
        """Export includes negative_control semantics."""
        contract = export_mock_oracle_contract()
        
        assert "negative_control" in contract
        nc = contract["negative_control"]
        
        assert nc["verified"] is False
        assert nc["abstained"] is True
        assert nc["timed_out"] is False
        assert nc["crashed"] is False
        assert nc["reason"] == "negative_control"
        assert nc["bucket"] == "negative_control"
        assert nc["stats_suppressed"] is True

    def test_export_has_scenarios(self):
        """Export includes all scenarios."""
        contract = export_mock_oracle_contract()
        
        assert "scenarios" in contract
        for scenario_name in SCENARIOS:
            assert scenario_name in contract["scenarios"]

    def test_export_has_determinism_guarantees(self):
        """Export includes determinism guarantees."""
        contract = export_mock_oracle_contract()
        
        assert "determinism" in contract
        det = contract["determinism"]
        
        assert "hash_algorithm" in det
        assert det["hash_algorithm"] == "SHA-256"
        assert "bucket_selection" in det
        assert "guarantees" in det
        assert len(det["guarantees"]) >= 3

    def test_export_is_deterministic(self):
        """export_mock_oracle_contract() is deterministic."""
        c1 = export_mock_oracle_contract()
        c2 = export_mock_oracle_contract()
        
        assert c1 == c2

    def test_export_is_json_serializable(self):
        """Contract can be serialized to JSON."""
        contract = export_mock_oracle_contract()
        
        json_str = json.dumps(contract, indent=2)
        
        # Parse it back
        parsed = json.loads(json_str)
        assert parsed == contract

    def test_export_json_deterministic(self):
        """JSON serialization is deterministic."""
        c1 = export_mock_oracle_contract()
        c2 = export_mock_oracle_contract()
        
        json1 = json.dumps(c1, sort_keys=True)
        json2 = json.dumps(c2, sort_keys=True)
        
        assert json1 == json2


class TestCLIScenarios:
    """Tests for CLI scenario-related flags."""

    def test_cli_list_scenarios(self):
        """--list-scenarios outputs scenarios."""
        result = subprocess.run(
            [sys.executable, "-m", "backend.verification.mock_oracle_cli", "--list-scenarios"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "Available Scenarios" in result.stdout
        assert "default_sanity" in result.stdout

    def test_cli_list_scenarios_json(self):
        """--list-scenarios --json outputs valid JSON."""
        result = subprocess.run(
            [sys.executable, "-m", "backend.verification.mock_oracle_cli", "--list-scenarios", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        data = json.loads(result.stdout)
        assert "scenarios" in data
        assert "count" in data
        assert data["count"] > 0

    def test_cli_filter_tags(self):
        """--filter-tags filters scenarios."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--list-scenarios", "--filter-tags", "sanity", "--json"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        data = json.loads(result.stdout)
        for s in data["scenarios"]:
            assert "sanity" in s["tags"]

    def test_cli_run_scenario(self):
        """--scenario runs a scenario and shows distribution."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--scenario", "default_sanity", "--samples", "50"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "Scenario: default_sanity" in result.stdout
        assert "Bucket Distribution" in result.stdout
        assert "Expected" in result.stdout
        assert "Observed" in result.stdout

    def test_cli_run_scenario_json(self):
        """--scenario --json outputs valid JSON."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--scenario", "default_sanity", "--samples", "50", "--json"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        data = json.loads(result.stdout)
        assert "scenario" in data
        assert "samples" in data
        assert "observed" in data
        assert "expected" in data
        assert "differences" in data

    def test_cli_scenario_invalid_name(self):
        """--scenario with invalid name returns exit code 1."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--scenario", "nonexistent_scenario"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_cli_scenario_assert_contract_pass(self):
        """--assert-contract passes when distribution is within epsilon."""
        # Use a large sample size to reduce variance
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--scenario", "default_stress", "--samples", "1000", "--assert-contract"
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        # With 1000 samples, distribution should be close to expected
        assert result.returncode == 0
        assert "CONTRACT CHECK PASSED" in result.stdout or "[PASS]" in result.stdout

    def test_cli_scenario_negative_control(self):
        """--scenario with NC scenario works correctly."""
        result = subprocess.run(
            [
                sys.executable, "-m", "backend.verification.mock_oracle_cli",
                "--scenario", "nc_baseline", "--samples", "50", "--json"
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        data = json.loads(result.stdout)
        assert data["scenario"]["negative_control"] is True
        # In NC mode, all should be negative_control bucket
        assert data["observed"]["negative_control"] == 100.0


class TestCLIExportContract:
    """Tests for CLI --export-contract flag."""

    def test_cli_export_contract(self):
        """--export-contract outputs valid JSON."""
        result = subprocess.run(
            [sys.executable, "-m", "backend.verification.mock_oracle_cli", "--export-contract"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        data = json.loads(result.stdout)
        assert "contract_version" in data
        assert "profiles" in data
        assert "negative_control" in data
        assert "scenarios" in data
        assert "determinism" in data

    def test_cli_export_contract_matches_function(self):
        """CLI export matches export_mock_oracle_contract()."""
        result = subprocess.run(
            [sys.executable, "-m", "backend.verification.mock_oracle_cli", "--export-contract"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        
        cli_data = json.loads(result.stdout)
        func_data = export_mock_oracle_contract()
        
        assert cli_data == func_data

    def test_cli_export_contract_deterministic(self):
        """--export-contract produces identical output on multiple runs."""
        result1 = subprocess.run(
            [sys.executable, "-m", "backend.verification.mock_oracle_cli", "--export-contract"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        result2 = subprocess.run(
            [sys.executable, "-m", "backend.verification.mock_oracle_cli", "--export-contract"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert result1.stdout == result2.stdout


class TestContractSnapshotDrift:
    """Tests to detect unintended contract drift."""

    # This test acts as a snapshot guard. If the contract changes,
    # this test will fail, requiring explicit acknowledgment.

    EXPECTED_CONTRACT_VERSION = "1.0.0"

    EXPECTED_PROFILE_DISTRIBUTIONS = {
        "default": {"verified": 60.0, "failed": 15.0, "abstain": 10.0, "timeout": 8.0, "error": 4.0, "crash": 3.0},
        "goal_hit": {"verified": 15.0, "failed": 35.0, "abstain": 35.0, "timeout": 10.0, "error": 3.0, "crash": 2.0},
        "sparse": {"verified": 25.0, "failed": 30.0, "abstain": 30.0, "timeout": 10.0, "error": 3.0, "crash": 2.0},
        "tree": {"verified": 45.0, "failed": 20.0, "abstain": 20.0, "timeout": 10.0, "error": 3.0, "crash": 2.0},
        "dependency": {"verified": 35.0, "failed": 25.0, "abstain": 25.0, "timeout": 10.0, "error": 3.0, "crash": 2.0},
    }

    EXPECTED_NC_SEMANTICS = {
        "verified": False,
        "abstained": True,
        "timed_out": False,
        "crashed": False,
        "reason": "negative_control",
        "bucket": "negative_control",
        "stats_suppressed": True,
    }

    def test_contract_version_unchanged(self):
        """Contract version matches expected value."""
        assert MOCK_ORACLE_CONTRACT_VERSION == self.EXPECTED_CONTRACT_VERSION, (
            f"Contract version changed from {self.EXPECTED_CONTRACT_VERSION} to "
            f"{MOCK_ORACLE_CONTRACT_VERSION}. If intentional, update this test."
        )

    def test_profile_distributions_unchanged(self):
        """Profile distributions match expected values."""
        contract = export_mock_oracle_contract()
        
        for profile_name, expected_dist in self.EXPECTED_PROFILE_DISTRIBUTIONS.items():
            actual_dist = contract["profiles"][profile_name]["distribution"]
            
            for bucket, expected_pct in expected_dist.items():
                actual_pct = actual_dist[bucket]
                assert expected_pct == actual_pct, (
                    f"Profile '{profile_name}' bucket '{bucket}' changed from "
                    f"{expected_pct}% to {actual_pct}%. If intentional, update this test."
                )

    def test_nc_semantics_unchanged(self):
        """Negative control semantics match expected values."""
        contract = export_mock_oracle_contract()
        nc = contract["negative_control"]
        
        for key, expected_val in self.EXPECTED_NC_SEMANTICS.items():
            actual_val = nc[key]
            assert expected_val == actual_val, (
                f"NC semantics '{key}' changed from {expected_val} to {actual_val}. "
                f"If intentional, update this test."
            )

    def test_scenario_count_stable(self):
        """Number of predefined scenarios is stable."""
        # Current expected count
        EXPECTED_SCENARIO_COUNT = 9
        
        actual_count = len(SCENARIOS)
        assert actual_count == EXPECTED_SCENARIO_COUNT, (
            f"Scenario count changed from {EXPECTED_SCENARIO_COUNT} to {actual_count}. "
            f"If adding/removing scenarios, update this test."
        )

