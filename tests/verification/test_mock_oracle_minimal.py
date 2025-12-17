"""
Tests for minimal mock oracle system.

This test suite verifies:
- Determinism of mock_verify()
- Profile distributions
- Fleet summary classification
- CI evaluation exit codes
- Isolation and environment guards
"""

import os
import pytest
from typing import Dict, Any, List

# Set environment variable before importing
os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification.mock_oracle_minimal import (
    mock_verify,
    list_profiles,
    get_profile_info,
    build_mock_oracle_fleet_summary,
    evaluate_mock_oracle_fleet_for_ci,
    build_mock_oracle_drift_tile,
    build_first_light_mock_oracle_summary,
    build_control_arm_calibration_summary,
    build_control_vs_twin_panel,
    control_arm_for_alignment_view,
    summarize_control_arm_signal_consistency,
    attach_mock_oracle_to_evidence,
    MOCK_ORACLE_SCHEMA_VERSION,
    FLEET_SUMMARY_SCHEMA_VERSION,
    DRIFT_TILE_SCHEMA_VERSION,
    FIRST_LIGHT_SUMMARY_SCHEMA_VERSION,
    CONTROL_ARM_CALIBRATION_SCHEMA_VERSION,
    CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
    FLEET_STATUS_OK,
    FLEET_STATUS_DRIFTING,
    FLEET_STATUS_BROKEN,
    DRIFT_STATUS_OK,
    DRIFT_STATUS_DRIFTING,
    DRIFT_STATUS_INVALID_HEAVY,
    STATUS_LIGHT_GREEN,
    STATUS_LIGHT_YELLOW,
    STATUS_LIGHT_RED,
    VERDICT_SUCCESS,
    VERDICT_FAILURE,
    VERDICT_ABSTAIN,
    _check_mock_oracle_enabled,
)


class TestMockVerifyDeterminism:
    """Test that mock_verify() is fully deterministic."""
    
    def test_same_input_same_output(self):
        """Same formula and profile should produce identical results."""
        formula = "(p->q)"
        profile = "uniform"
        
        result1 = mock_verify(formula, profile)
        result2 = mock_verify(formula, profile)
        
        assert result1 == result2
        assert result1["trace_hash"] == result2["trace_hash"]
        assert result1["verdict"] == result2["verdict"]
    
    def test_determinism_across_runs(self):
        """Results should be deterministic across multiple calls."""
        formula = "forall x, P(x)"
        profile = "timeout_heavy"
        
        results = [mock_verify(formula, profile) for _ in range(10)]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
    
    def test_different_formulas_different_results(self):
        """Different formulas should produce different results (with high probability)."""
        formulas = ["(p->q)", "(q->p)", "p & q", "p | q", "~p"]
        results = [mock_verify(f, "uniform") for f in formulas]
        
        # Check that trace hashes are different
        trace_hashes = [r["trace_hash"] for r in results]
        assert len(set(trace_hashes)) == len(trace_hashes), "All formulas should produce unique trace hashes"
    
    def test_different_profiles_different_distributions(self):
        """Different profiles should produce different verdict distributions."""
        profiles = ["uniform", "timeout_heavy", "invalid_heavy", "success_heavy"]
        
        results = {}
        for profile in profiles:
            # Use different formulas for each profile to get distribution
            formulas = [f"formula_{profile}_{i}" for i in range(200)]
            verdicts = [mock_verify(f, profile)["verdict"] for f in formulas]
            results[profile] = {
                "success": verdicts.count(VERDICT_SUCCESS),
                "failure": verdicts.count(VERDICT_FAILURE),
                "abstain": verdicts.count(VERDICT_ABSTAIN),
            }
        
        # timeout_heavy should have more abstains (70% vs ~33%)
        assert results["timeout_heavy"]["abstain"] > results["uniform"]["abstain"]
        
        # invalid_heavy should have more failures (85% vs ~33%)
        assert results["invalid_heavy"]["failure"] > results["uniform"]["failure"]
        
        # success_heavy should have more successes (80% vs ~33%)
        assert results["success_heavy"]["success"] > results["uniform"]["success"]


class TestMockVerifySchema:
    """Test that mock_verify() returns correct schema."""
    
    def test_schema_version_present(self):
        """Result should include schema_version."""
        result = mock_verify("(p->q)", "uniform")
        assert result["schema_version"] == MOCK_ORACLE_SCHEMA_VERSION
    
    def test_required_fields_present(self):
        """Result should include all required fields."""
        result = mock_verify("(p->q)", "uniform")
        
        assert "schema_version" in result
        assert "profile" in result
        assert "verdict" in result
        assert "trace_hash" in result
        assert result["profile"] == "uniform"
        assert result["verdict"] in [VERDICT_SUCCESS, VERDICT_FAILURE, VERDICT_ABSTAIN]
        assert isinstance(result["trace_hash"], str)
        assert len(result["trace_hash"]) == 64  # SHA-256 hex length
    
    def test_abstention_reason_present_when_abstain(self):
        """abstention_reason should be present when verdict is 'abstain'."""
        # Try multiple formulas to find one that abstains
        for formula in ["(p->q)", "(q->p)", "p & q", "p | q", "~p", "forall x, P(x)"]:
            result = mock_verify(formula, "timeout_heavy")  # High abstention rate
            if result["verdict"] == VERDICT_ABSTAIN:
                assert "abstention_reason" in result
                assert isinstance(result["abstention_reason"], str)
                assert result["abstention_reason"] in [
                    "timeout",
                    "resource_exhausted",
                    "unknown_formula_structure",
                    "insufficient_context",
                ]
                break
        else:
            pytest.skip("Could not find a formula that produces abstain verdict")
    
    def test_abstention_reason_absent_when_not_abstain(self):
        """abstention_reason should not be present when verdict is not 'abstain'."""
        # Try multiple formulas to find one that succeeds or fails
        for formula in ["(p->q)", "(q->p)", "p & q", "p | q", "~p"]:
            result = mock_verify(formula, "success_heavy")  # High success rate
            if result["verdict"] != VERDICT_ABSTAIN:
                assert "abstention_reason" not in result
                break


class TestMockVerifyProfiles:
    """Test profile functionality."""
    
    def test_list_profiles(self):
        """list_profiles() should return all available profiles."""
        profiles = list_profiles()
        
        assert isinstance(profiles, dict)
        assert "uniform" in profiles
        assert "timeout_heavy" in profiles
        assert "invalid_heavy" in profiles
        assert "success_heavy" in profiles
    
    def test_get_profile_info(self):
        """get_profile_info() should return profile configuration."""
        info = get_profile_info("uniform")
        
        assert "success_pct" in info
        assert "failure_pct" in info
        assert "abstain_pct" in info
        assert "description" in info
        assert isinstance(info["success_pct"], (int, float))
        assert isinstance(info["failure_pct"], (int, float))
        assert isinstance(info["abstain_pct"], (int, float))
    
    def test_get_profile_info_unknown_profile(self):
        """get_profile_info() should raise ValueError for unknown profile."""
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile_info("nonexistent_profile")
    
    def test_mock_verify_unknown_profile(self):
        """mock_verify() should raise ValueError for unknown profile."""
        with pytest.raises(ValueError, match="Unknown profile"):
            mock_verify("(p->q)", "nonexistent_profile")
    
    def test_profile_percentages_sum_to_100(self):
        """All profiles should have percentages that sum to approximately 100."""
        profiles = list_profiles()
        
        for profile_name, config in profiles.items():
            total = config["success_pct"] + config["failure_pct"] + config["abstain_pct"]
            assert abs(total - 100.0) < 0.1, f"Profile '{profile_name}' percentages sum to {total}, not 100"


class TestMockVerifyInputValidation:
    """Test input validation."""
    
    def test_empty_formula_raises(self):
        """Empty formula should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            mock_verify("", "uniform")
    
    def test_whitespace_only_formula_raises(self):
        """Whitespace-only formula should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            mock_verify("   ", "uniform")
    
    def test_formula_normalization(self):
        """Formula should be normalized (case-insensitive, whitespace-stripped)."""
        formula1 = "(p->q)"
        formula2 = "  (P->Q)  "
        formula3 = "(p->q)"
        
        result1 = mock_verify(formula1, "uniform")
        result2 = mock_verify(formula2, "uniform")
        result3 = mock_verify(formula3, "uniform")
        
        # Normalized versions should produce same result
        assert result1 == result2
        assert result1 == result3


class TestFleetSummary:
    """Test fleet summary functionality."""
    
    def test_fleet_summary_empty_results(self):
        """Fleet summary with empty results should return OK status."""
        summary = build_mock_oracle_fleet_summary([])
        
        assert summary["schema_version"] == FLEET_SUMMARY_SCHEMA_VERSION
        assert summary["total_queries"] == 0
        assert summary["abstention_rate"] == 0.0
        assert summary["invalid_rate"] == 0.0
        assert summary["success_rate"] == 0.0
        assert summary["status"] == FLEET_STATUS_OK
    
    def test_fleet_summary_balanced(self):
        """Fleet summary with balanced results should return OK status."""
        # Generate balanced results (mix of all verdicts)
        results = []
        for i in range(100):
            formula = f"formula_{i}"
            # Use uniform profile to get balanced distribution
            result = mock_verify(formula, "uniform")
            results.append(result)
        
        summary = build_mock_oracle_fleet_summary(results)
        
        assert summary["total_queries"] == 100
        assert summary["status"] == FLEET_STATUS_OK
        assert 0.0 <= summary["abstention_rate"] <= 1.0
        assert 0.0 <= summary["invalid_rate"] <= 1.0
        assert 0.0 <= summary["success_rate"] <= 1.0
        # Rates should sum to approximately 1.0
        total_rate = summary["abstention_rate"] + summary["invalid_rate"] + summary["success_rate"]
        assert abs(total_rate - 1.0) < 0.01
    
    def test_fleet_summary_abstention_heavy(self):
        """Fleet summary with high abstention rate should return DRIFTING status."""
        # Generate results with high abstention rate using timeout_heavy profile
        results = []
        for i in range(100):
            formula = f"formula_{i}"
            result = mock_verify(formula, "timeout_heavy")
            results.append(result)
        
        summary = build_mock_oracle_fleet_summary(results)
        
        assert summary["total_queries"] == 100
        # timeout_heavy profile has 70% abstention, which exceeds 30% threshold
        assert summary["abstention_rate"] > 0.3
        assert summary["status"] == FLEET_STATUS_DRIFTING
    
    def test_fleet_summary_invalid_heavy(self):
        """Fleet summary with high invalid rate should return BROKEN status."""
        # Generate results with high failure rate using invalid_heavy profile
        results = []
        for i in range(100):
            formula = f"formula_{i}"
            result = mock_verify(formula, "invalid_heavy")
            results.append(result)
        
        summary = build_mock_oracle_fleet_summary(results)
        
        assert summary["total_queries"] == 100
        # invalid_heavy profile has 85% failure, which exceeds 50% threshold
        assert summary["invalid_rate"] > 0.5
        assert summary["status"] == FLEET_STATUS_BROKEN
    
    def test_fleet_summary_schema(self):
        """Fleet summary should include all required fields."""
        results = [mock_verify("(p->q)", "uniform") for _ in range(10)]
        summary = build_mock_oracle_fleet_summary(results)
        
        assert "schema_version" in summary
        assert "total_queries" in summary
        assert "abstention_rate" in summary
        assert "invalid_rate" in summary
        assert "success_rate" in summary
        assert "status" in summary
        assert "summary_text" in summary
        assert isinstance(summary["summary_text"], str)


class TestCIEvaluation:
    """Test CI evaluation functionality."""
    
    def test_ci_evaluation_ok(self):
        """CI evaluation with OK status should return exit code 0."""
        # Create OK fleet summary
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(50)]
        summary = build_mock_oracle_fleet_summary(results)
        
        # If status is OK, exit code should be 0
        if summary["status"] == FLEET_STATUS_OK:
            exit_code, message = evaluate_mock_oracle_fleet_for_ci(summary)
            assert exit_code == 0
            assert "OK" in message
            assert "Fleet status OK" in message
    
    def test_ci_evaluation_drifting(self):
        """CI evaluation with DRIFTING status should return exit code 1."""
        # Create DRIFTING fleet summary using timeout_heavy profile
        results = [mock_verify(f"formula_{i}", "timeout_heavy") for i in range(100)]
        summary = build_mock_oracle_fleet_summary(results)
        
        # timeout_heavy should produce DRIFTING status
        assert summary["status"] == FLEET_STATUS_DRIFTING
        
        exit_code, message = evaluate_mock_oracle_fleet_for_ci(summary)
        assert exit_code == 1
        assert "WARN" in message
        assert "DRIFTING" in message
    
    def test_ci_evaluation_broken(self):
        """CI evaluation with BROKEN status should return exit code 2."""
        # Create BROKEN fleet summary using invalid_heavy profile
        results = [mock_verify(f"formula_{i}", "invalid_heavy") for i in range(100)]
        summary = build_mock_oracle_fleet_summary(results)
        
        # invalid_heavy should produce BROKEN status
        assert summary["status"] == FLEET_STATUS_BROKEN
        
        exit_code, message = evaluate_mock_oracle_fleet_for_ci(summary)
        assert exit_code == 2
        assert "BLOCK" in message
        assert "BROKEN" in message
    
    def test_ci_evaluation_message_format(self):
        """CI evaluation message should be well-formatted."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(10)]
        summary = build_mock_oracle_fleet_summary(results)
        
        exit_code, message = evaluate_mock_oracle_fleet_for_ci(summary)
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert "Total queries" in message or "queries" in message.lower()


class TestIsolation:
    """Test isolation and environment guards."""
    
    def test_mock_oracle_enabled_with_env_var(self):
        """Mock oracle should work when MATHLEDGER_ALLOW_MOCK_ORACLE=1."""
        # Environment variable is set at module level
        result = mock_verify("(p->q)", "uniform")
        assert result is not None
        assert "verdict" in result
    
    def test_check_mock_oracle_enabled_raises_when_disabled(self):
        """_check_mock_oracle_enabled() should raise when disabled."""
        # Temporarily unset environment variable
        original_value = os.environ.get("MATHLEDGER_ALLOW_MOCK_ORACLE")
        try:
            if "MATHLEDGER_ALLOW_MOCK_ORACLE" in os.environ:
                del os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"]
            
            with pytest.raises(RuntimeError, match="Mock oracle is disabled"):
                _check_mock_oracle_enabled()
        finally:
            # Restore original value
            if original_value is not None:
                os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = original_value
            else:
                os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"


class TestProfileDistributions:
    """Test that profiles produce expected distributions."""
    
    def test_uniform_profile_distribution(self):
        """Uniform profile should produce roughly equal distribution."""
        formulas = [f"formula_{i}" for i in range(300)]
        results = [mock_verify(f, "uniform") for f in formulas]
        
        success_count = sum(1 for r in results if r["verdict"] == VERDICT_SUCCESS)
        failure_count = sum(1 for r in results if r["verdict"] == VERDICT_FAILURE)
        abstain_count = sum(1 for r in results if r["verdict"] == VERDICT_ABSTAIN)
        
        # Uniform profile should have roughly 33% each (allow 10% variance)
        total = len(results)
        assert abs(success_count / total - 0.33) < 0.10
        assert abs(failure_count / total - 0.33) < 0.10
        assert abs(abstain_count / total - 0.34) < 0.10
    
    def test_timeout_heavy_profile_distribution(self):
        """Timeout heavy profile should produce high abstention rate."""
        formulas = [f"formula_{i}" for i in range(200)]
        results = [mock_verify(f, "timeout_heavy") for f in formulas]
        
        abstain_count = sum(1 for r in results if r["verdict"] == VERDICT_ABSTAIN)
        abstain_rate = abstain_count / len(results)
        
        # Timeout heavy should have ~70% abstention (allow 15% variance)
        assert abstain_rate > 0.55, f"Expected high abstention rate, got {abstain_rate:.2%}"
    
    def test_invalid_heavy_profile_distribution(self):
        """Invalid heavy profile should produce high failure rate."""
        formulas = [f"formula_{i}" for i in range(200)]
        results = [mock_verify(f, "invalid_heavy") for f in formulas]
        
        failure_count = sum(1 for r in results if r["verdict"] == VERDICT_FAILURE)
        failure_rate = failure_count / len(results)
        
        # Invalid heavy should have ~85% failure (allow 15% variance)
        assert failure_rate > 0.70, f"Expected high failure rate, got {failure_rate:.2%}"
    
    def test_success_heavy_profile_distribution(self):
        """Success heavy profile should produce high success rate."""
        formulas = [f"formula_{i}" for i in range(200)]
        results = [mock_verify(f, "success_heavy") for f in formulas]
        
        success_count = sum(1 for r in results if r["verdict"] == VERDICT_SUCCESS)
        success_rate = success_count / len(results)
        
        # Success heavy should have ~80% success (allow 15% variance)
        assert success_rate > 0.65, f"Expected high success rate, got {success_rate:.2%}"


class TestDriftTile:
    """Test drift tile functionality."""
    
    def test_drift_tile_ok_status(self):
        """Drift tile with OK fleet status should return GREEN status light."""
        # Create OK fleet summary
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(50)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        
        # If status is OK, drift tile should be OK with GREEN light
        if fleet_summary["status"] == FLEET_STATUS_OK:
            tile = build_mock_oracle_drift_tile(fleet_summary)
            
            assert tile["schema_version"] == DRIFT_TILE_SCHEMA_VERSION
            assert tile["drift_status"] == DRIFT_STATUS_OK
            assert tile["status_light"] == STATUS_LIGHT_GREEN
            assert "abstention_rate" in tile
            assert "invalid_rate" in tile
            assert "headline" in tile
            assert "total_queries" in tile
            assert "OK" in tile["headline"]
    
    def test_drift_tile_drifting_status(self):
        """Drift tile with DRIFTING fleet status should return YELLOW status light."""
        # Create DRIFTING fleet summary using timeout_heavy profile
        results = [mock_verify(f"formula_{i}", "timeout_heavy") for i in range(100)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        
        # timeout_heavy should produce DRIFTING status
        assert fleet_summary["status"] == FLEET_STATUS_DRIFTING
        
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        assert tile["drift_status"] == DRIFT_STATUS_DRIFTING
        assert tile["status_light"] == STATUS_LIGHT_YELLOW
        assert "DRIFTING" in tile["headline"]
    
    def test_drift_tile_invalid_heavy_status(self):
        """Drift tile with BROKEN fleet status should return RED status light."""
        # Create BROKEN fleet summary using invalid_heavy profile
        results = [mock_verify(f"formula_{i}", "invalid_heavy") for i in range(100)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        
        # invalid_heavy should produce BROKEN status
        assert fleet_summary["status"] == FLEET_STATUS_BROKEN
        
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        assert tile["drift_status"] == DRIFT_STATUS_INVALID_HEAVY
        assert tile["status_light"] == STATUS_LIGHT_RED
        assert "INVALID-HEAVY" in tile["headline"]
    
    def test_drift_tile_schema(self):
        """Drift tile should include all required fields."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(10)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        assert "schema_version" in tile
        assert "drift_status" in tile
        assert "status_light" in tile
        assert "abstention_rate" in tile
        assert "invalid_rate" in tile
        assert "headline" in tile
        assert "total_queries" in tile
        assert isinstance(tile["abstention_rate"], float)
        assert isinstance(tile["invalid_rate"], float)
        assert isinstance(tile["total_queries"], int)
    
    def test_drift_tile_status_light_mapping(self):
        """Status light should correctly map to drift status."""
        # Test all three status mappings
        test_cases = [
            (FLEET_STATUS_OK, DRIFT_STATUS_OK, STATUS_LIGHT_GREEN),
            (FLEET_STATUS_DRIFTING, DRIFT_STATUS_DRIFTING, STATUS_LIGHT_YELLOW),
            (FLEET_STATUS_BROKEN, DRIFT_STATUS_INVALID_HEAVY, STATUS_LIGHT_RED),
        ]
        
        for fleet_status, expected_drift_status, expected_light in test_cases:
            # Create a mock fleet summary with the desired status
            fleet_summary = {
                "schema_version": FLEET_SUMMARY_SCHEMA_VERSION,
                "total_queries": 100,
                "abstention_rate": 0.1 if fleet_status == FLEET_STATUS_OK else 0.4,
                "invalid_rate": 0.1 if fleet_status != FLEET_STATUS_BROKEN else 0.6,
                "success_rate": 0.8 if fleet_status == FLEET_STATUS_OK else 0.2,
                "status": fleet_status,
                "summary_text": f"Test summary for {fleet_status}",
            }
            
            tile = build_mock_oracle_drift_tile(fleet_summary)
            
            assert tile["drift_status"] == expected_drift_status
            assert tile["status_light"] == expected_light
    
    def test_drift_tile_deterministic(self):
        """Drift tile should be deterministic for same fleet summary."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(50)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        
        tile1 = build_mock_oracle_drift_tile(fleet_summary)
        tile2 = build_mock_oracle_drift_tile(fleet_summary)
        
        assert tile1 == tile2
    
    def test_drift_tile_json_safe(self):
        """Drift tile should be JSON-serializable."""
        import json
        
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(10)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        # Should not raise
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed == tile
    
    def test_drift_tile_headline_format(self):
        """Drift tile headline should be informative and neutral."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(20)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        headline = tile["headline"]
        assert isinstance(headline, str)
        assert len(headline) > 0
        # Headline should mention the drift status
        assert tile["drift_status"] in headline or "Mock Oracle" in headline


class TestEvidencePackIntegration:
    """Test evidence pack integration."""
    
    def test_attach_mock_oracle_to_evidence(self):
        """attach_mock_oracle_to_evidence() should attach tile to evidence pack."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(10)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        evidence = {
            "evidence_type": "test",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        
        result = attach_mock_oracle_to_evidence(evidence, tile)
        
        assert result == evidence  # Should modify in-place
        assert "governance" in evidence
        assert "mock_oracle" in evidence["governance"]
        assert evidence["governance"]["mock_oracle"] == tile
    
    def test_attach_mock_oracle_to_evidence_existing_governance(self):
        """attach_mock_oracle_to_evidence() should preserve existing governance data."""
        evidence = {
            "evidence_type": "test",
            "governance": {
                "other_tile": {"status": "ok"},
            },
        }
        
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(10)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        result = attach_mock_oracle_to_evidence(evidence, tile)
        
        assert "other_tile" in evidence["governance"]
        assert "mock_oracle" in evidence["governance"]
        assert evidence["governance"]["mock_oracle"] == tile
    
    def test_attach_mock_oracle_to_evidence_empty_evidence(self):
        """attach_mock_oracle_to_evidence() should work with empty evidence dict."""
        evidence = {}
        
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(10)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        result = attach_mock_oracle_to_evidence(evidence, tile)
        
        assert "governance" in evidence
        assert "mock_oracle" in evidence["governance"]
        assert evidence["governance"]["mock_oracle"] == tile
    
    def test_attach_mock_oracle_to_evidence_with_first_light_summary(self):
        """
        attach_mock_oracle_to_evidence() should attach First Light summary when fleet_summary provided.
        
        NOTE: This summary is attached to evidence packs as a negative control arm for First Light
        verification. It provides a deliberately messy baseline to compare against real verifier behavior.
        The purpose is to prove the pipeline can detect expected stochasticity, not to improve performance.
        """
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(50)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        evidence = {
            "evidence_type": "first_light",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        
        result = attach_mock_oracle_to_evidence(evidence, tile, fleet_summary=fleet_summary)
        
        assert result == evidence
        assert "governance" in evidence
        assert "mock_oracle" in evidence["governance"]
        assert "first_light_summary" in evidence["governance"]["mock_oracle"]
        
        first_light_summary = evidence["governance"]["mock_oracle"]["first_light_summary"]
        assert first_light_summary["schema_version"] == FIRST_LIGHT_SUMMARY_SCHEMA_VERSION
        assert first_light_summary["status"] == fleet_summary["status"]
        assert first_light_summary["abstention_rate"] == fleet_summary["abstention_rate"]
        assert first_light_summary["invalid_rate"] == fleet_summary["invalid_rate"]
        assert first_light_summary["total_queries"] == fleet_summary["total_queries"]
    
    def test_attach_mock_oracle_to_evidence_without_first_light_summary(self):
        """attach_mock_oracle_to_evidence() should not attach First Light summary when fleet_summary not provided."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(10)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        evidence = {
            "evidence_type": "test",
        }
        
        result = attach_mock_oracle_to_evidence(evidence, tile)
        
        assert "governance" in evidence
        assert "mock_oracle" in evidence["governance"]
        assert "first_light_summary" not in evidence["governance"]["mock_oracle"]


class TestFirstLightSummary:
    """
    Test First Light summary functionality.
    
    NOTE: The First Light mock oracle summary is a negative control arm, not a performance
    improvement mechanism. These tests verify that the summary correctly captures the mock
    oracle's deliberately messy baseline, which serves to prove the pipeline can detect
    expected stochasticity. The summary should NOT match real verifier behavior - differences
    are intentional and validate that governance logic does not overfit to deterministic patterns.
    """
    
    def test_build_first_light_mock_oracle_summary_schema(self):
        """First Light summary should include all required fields."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(100)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        summary = build_first_light_mock_oracle_summary(fleet_summary)
        
        assert "schema_version" in summary
        assert "status" in summary
        assert "abstention_rate" in summary
        assert "invalid_rate" in summary
        assert "total_queries" in summary
        assert summary["schema_version"] == FIRST_LIGHT_SUMMARY_SCHEMA_VERSION
    
    def test_build_first_light_mock_oracle_summary_values(self):
        """First Light summary should extract correct values from fleet summary."""
        results = [mock_verify(f"formula_{i}", "timeout_heavy") for i in range(100)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        summary = build_first_light_mock_oracle_summary(fleet_summary)
        
        assert summary["status"] == fleet_summary["status"]
        assert summary["abstention_rate"] == fleet_summary["abstention_rate"]
        assert summary["invalid_rate"] == fleet_summary["invalid_rate"]
        assert summary["total_queries"] == fleet_summary["total_queries"]
    
    def test_build_first_light_mock_oracle_summary_deterministic(self):
        """First Light summary should be deterministic for same fleet summary."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(50)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        
        summary1 = build_first_light_mock_oracle_summary(fleet_summary)
        summary2 = build_first_light_mock_oracle_summary(fleet_summary)
        
        assert summary1 == summary2
    
    def test_build_first_light_mock_oracle_summary_json_safe(self):
        """First Light summary should be JSON-serializable."""
        import json
        
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(20)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        summary = build_first_light_mock_oracle_summary(fleet_summary)
        
        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed == summary
    
    def test_first_light_summary_all_statuses(self):
        """First Light summary should work for all fleet statuses."""
        test_cases = [
            ("uniform", FLEET_STATUS_OK),
            ("timeout_heavy", FLEET_STATUS_DRIFTING),
            ("invalid_heavy", FLEET_STATUS_BROKEN),
        ]
        
        for profile, expected_status in test_cases:
            results = [mock_verify(f"formula_{i}", profile) for i in range(100)]
            fleet_summary = build_mock_oracle_fleet_summary(results)
            
            # Verify fleet summary has expected status
            if fleet_summary["status"] == expected_status:
                summary = build_first_light_mock_oracle_summary(fleet_summary)
                assert summary["status"] == expected_status
                assert isinstance(summary["abstention_rate"], float)
                assert isinstance(summary["invalid_rate"], float)
                assert isinstance(summary["total_queries"], int)
    
    def test_first_light_summary_integration_with_evidence(self):
        """
        First Light summary should integrate correctly with evidence pack.
        
        NOTE: This test verifies the end-to-end integration of the mock oracle as a control arm
        in First Light evidence. The summary represents a negative control baseline that should
        differ from real verifier metrics. This difference validates that governance logic can
        distinguish between expected stochasticity (mock oracle) and actual verification behavior.
        """
        import json
        
        # Build synthetic fleet
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(100)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        # Attach to evidence
        evidence = {
            "evidence_type": "first_light",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        attach_mock_oracle_to_evidence(evidence, tile, fleet_summary=fleet_summary)
        
        # Verify structure
        assert "governance" in evidence
        assert "mock_oracle" in evidence["governance"]
        assert "first_light_summary" in evidence["governance"]["mock_oracle"]
        
        # Verify JSON safety
        json_str = json.dumps(evidence)
        parsed = json.loads(json_str)
        assert parsed["governance"]["mock_oracle"]["first_light_summary"]["status"] == fleet_summary["status"]
        
        # Verify determinism: same inputs should produce same output
        evidence2 = {
            "evidence_type": "first_light",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        attach_mock_oracle_to_evidence(evidence2, tile, fleet_summary=fleet_summary)
        
        assert (
            evidence["governance"]["mock_oracle"]["first_light_summary"]
            == evidence2["governance"]["mock_oracle"]["first_light_summary"]
        )


class TestControlArmCalibration:
    """Test control arm calibration summary functionality."""
    
    def test_build_control_arm_calibration_summary_schema(self):
        """Control arm calibration summary should include all required fields."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(100)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        summary = build_control_arm_calibration_summary(fleet_summary, profile="uniform")
        
        assert "schema_version" in summary
        assert "status" in summary
        assert "abstention_rate" in summary
        assert "invalid_rate" in summary
        assert "total_queries" in summary
        assert "profile" in summary
        assert summary["schema_version"] == CONTROL_ARM_CALIBRATION_SCHEMA_VERSION
        assert summary["profile"] == "uniform"
    
    def test_build_control_arm_calibration_summary_values(self):
        """Control arm calibration summary should extract correct values from fleet summary."""
        results = [mock_verify(f"formula_{i}", "timeout_heavy") for i in range(100)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        summary = build_control_arm_calibration_summary(fleet_summary, profile="timeout_heavy")
        
        assert summary["status"] == fleet_summary["status"]
        assert summary["abstention_rate"] == fleet_summary["abstention_rate"]
        assert summary["invalid_rate"] == fleet_summary["invalid_rate"]
        assert summary["total_queries"] == fleet_summary["total_queries"]
        assert summary["profile"] == "timeout_heavy"
    
    def test_build_control_arm_calibration_summary_deterministic(self):
        """Control arm calibration summary should be deterministic for same inputs."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(50)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        
        summary1 = build_control_arm_calibration_summary(fleet_summary, profile="uniform")
        summary2 = build_control_arm_calibration_summary(fleet_summary, profile="uniform")
        
        assert summary1 == summary2
    
    def test_build_control_arm_calibration_summary_json_safe(self):
        """Control arm calibration summary should be JSON-serializable."""
        import json
        
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(20)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        summary = build_control_arm_calibration_summary(fleet_summary, profile="uniform")
        
        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed == summary
    
    def test_build_control_arm_calibration_summary_all_profiles(self):
        """Control arm calibration summary should work with all profiles."""
        profiles = ["uniform", "timeout_heavy", "invalid_heavy", "success_heavy"]
        
        for profile in profiles:
            results = [mock_verify(f"formula_{i}", profile) for i in range(50)]
            fleet_summary = build_mock_oracle_fleet_summary(results)
            summary = build_control_arm_calibration_summary(fleet_summary, profile=profile)
            
            assert summary["profile"] == profile
            assert isinstance(summary["abstention_rate"], float)
            assert isinstance(summary["invalid_rate"], float)
            assert isinstance(summary["total_queries"], int)


class TestControlVsTwinPanel:
    """Test control vs twin panel functionality."""
    
    def test_build_control_vs_twin_panel_schema(self):
        """Control vs twin panel should include all required fields."""
        control_summaries = {
            "CAL-EXP-001": build_control_arm_calibration_summary(
                build_mock_oracle_fleet_summary([mock_verify(f"f_{i}", "uniform") for i in range(50)]),
                profile="uniform"
            ),
        }
        twin_summaries = {
            "CAL-EXP-001": {
                "status": "OK",
                "abstention_rate": 0.25,
                "invalid_rate": 0.20,
                "total_queries": 50,
            },
        }
        
        panel = build_control_vs_twin_panel(control_summaries, twin_summaries)
        
        assert "schema_version" in panel
        assert "experiments" in panel
        assert "control_vs_twin_delta" in panel
        assert "red_flags" in panel
        assert panel["schema_version"] == CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION
        assert isinstance(panel["experiments"], list)
        assert isinstance(panel["red_flags"], list)
    
    def test_build_control_vs_twin_panel_delta_computation(self):
        """Control vs twin panel should compute deltas correctly."""
        control_summaries = {
            "CAL-EXP-001": build_control_arm_calibration_summary(
                build_mock_oracle_fleet_summary([mock_verify(f"f_{i}", "uniform") for i in range(100)]),
                profile="uniform"
            ),
        }
        twin_summaries = {
            "CAL-EXP-001": {
                "status": "OK",
                "abstention_rate": 0.15,  # Different from control
                "invalid_rate": 0.25,  # Different from control
                "total_queries": 100,
            },
        }
        
        panel = build_control_vs_twin_panel(control_summaries, twin_summaries)
        
        assert "CAL-EXP-001" in panel["experiments"]
        assert "CAL-EXP-001" in panel["control_vs_twin_delta"]
        
        delta = panel["control_vs_twin_delta"]["CAL-EXP-001"]
        assert "abstention_rate_delta" in delta
        assert "invalid_rate_delta" in delta
        assert "status_match" in delta
        assert isinstance(delta["abstention_rate_delta"], float)
        assert isinstance(delta["invalid_rate_delta"], float)
        assert isinstance(delta["status_match"], bool)
    
    def test_build_control_vs_twin_panel_red_flag_too_similar(self):
        """Control vs twin panel should raise red flag when metrics are too similar."""
        control_summaries = {
            "CAL-EXP-001": {
                "schema_version": CONTROL_ARM_CALIBRATION_SCHEMA_VERSION,
                "status": "OK",
                "abstention_rate": 0.33,
                "invalid_rate": 0.33,
                "total_queries": 100,
                "profile": "uniform",
            },
        }
        # Twin matches control very closely (within 1%)
        twin_summaries = {
            "CAL-EXP-001": {
                "status": "OK",
                "abstention_rate": 0.332,  # Very close to control
                "invalid_rate": 0.331,  # Very close to control
                "total_queries": 100,
            },
        }
        
        panel = build_control_vs_twin_panel(control_summaries, twin_summaries)
        
        assert len(panel["red_flags"]) > 0
        assert any("too similar" in flag.lower() for flag in panel["red_flags"])
    
    def test_build_control_vs_twin_panel_no_red_flags_when_different(self):
        """Control vs twin panel should not raise red flags when metrics differ appropriately."""
        control_summaries = {
            "CAL-EXP-001": {
                "schema_version": CONTROL_ARM_CALIBRATION_SCHEMA_VERSION,
                "status": "OK",
                "abstention_rate": 0.33,
                "invalid_rate": 0.33,
                "total_queries": 100,
                "profile": "uniform",
            },
        }
        # Twin differs significantly from control (good - proves we can distinguish)
        twin_summaries = {
            "CAL-EXP-001": {
                "status": "DRIFTING",
                "abstention_rate": 0.60,  # Significantly different
                "invalid_rate": 0.10,  # Significantly different
                "total_queries": 100,
            },
        }
        
        panel = build_control_vs_twin_panel(control_summaries, twin_summaries)
        
        # Should have no red flags when control and twin differ appropriately
        assert len(panel["red_flags"]) == 0
    
    def test_build_control_vs_twin_panel_multiple_experiments(self):
        """Control vs twin panel should handle multiple experiments."""
        control_summaries = {
            "CAL-EXP-001": build_control_arm_calibration_summary(
                build_mock_oracle_fleet_summary([mock_verify(f"f1_{i}", "uniform") for i in range(50)]),
                profile="uniform"
            ),
            "CAL-EXP-002": build_control_arm_calibration_summary(
                build_mock_oracle_fleet_summary([mock_verify(f"f2_{i}", "timeout_heavy") for i in range(50)]),
                profile="timeout_heavy"
            ),
        }
        twin_summaries = {
            "CAL-EXP-001": {
                "status": "OK",
                "abstention_rate": 0.20,
                "invalid_rate": 0.25,
                "total_queries": 50,
            },
            "CAL-EXP-002": {
                "status": "DRIFTING",
                "abstention_rate": 0.50,
                "invalid_rate": 0.15,
                "total_queries": 50,
            },
        }
        
        panel = build_control_vs_twin_panel(control_summaries, twin_summaries)
        
        assert len(panel["experiments"]) == 2
        assert "CAL-EXP-001" in panel["experiments"]
        assert "CAL-EXP-002" in panel["experiments"]
        assert "CAL-EXP-001" in panel["control_vs_twin_delta"]
        assert "CAL-EXP-002" in panel["control_vs_twin_delta"]
    
    def test_build_control_vs_twin_panel_json_safe(self):
        """Control vs twin panel should be JSON-serializable."""
        import json
        
        control_summaries = {
            "CAL-EXP-001": build_control_arm_calibration_summary(
                build_mock_oracle_fleet_summary([mock_verify(f"f_{i}", "uniform") for i in range(50)]),
                profile="uniform"
            ),
        }
        twin_summaries = {
            "CAL-EXP-001": {
                "status": "OK",
                "abstention_rate": 0.25,
                "invalid_rate": 0.20,
                "total_queries": 50,
            },
        }
        
        panel = build_control_vs_twin_panel(control_summaries, twin_summaries)
        
        # Should not raise
        json_str = json.dumps(panel)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed == panel
    
    def test_attach_mock_oracle_to_evidence_with_panel(self):
        """attach_mock_oracle_to_evidence() should attach control vs twin panel when provided."""
        results = [mock_verify(f"formula_{i}", "uniform") for i in range(50)]
        fleet_summary = build_mock_oracle_fleet_summary(results)
        tile = build_mock_oracle_drift_tile(fleet_summary)
        
        control_summaries = {
            "CAL-EXP-001": build_control_arm_calibration_summary(fleet_summary, profile="uniform"),
        }
        twin_summaries = {
            "CAL-EXP-001": {
                "status": "OK",
                "abstention_rate": 0.25,
                "invalid_rate": 0.20,
                "total_queries": 50,
            },
        }
        panel = build_control_vs_twin_panel(control_summaries, twin_summaries)
        
        evidence = {
            "evidence_type": "calibration",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        
        result = attach_mock_oracle_to_evidence(evidence, tile, control_vs_twin_panel=panel)
        
        assert result == evidence
        assert "governance" in evidence
        assert "mock_oracle_panel" in evidence["governance"]
        assert evidence["governance"]["mock_oracle_panel"] == panel


class TestControlArmGGFLAdapter:
    """Test control arm GGFL alignment view adapter."""
    
    def test_control_arm_for_alignment_view_no_red_flags(self):
        """GGFL adapter should return 'ok' status when no red flags."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001", "CAL-EXP-002"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert result["status"] == "ok"
        assert "no red flags" in result["summary"].lower()
        assert result["conflict"] is False
    
    def test_control_arm_for_alignment_view_with_red_flags(self):
        """GGFL adapter should return 'warn' status when red flags present."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [
                "Experiment 'CAL-EXP-001': Control and twin metrics are too similar "
                "(abstention_delta=0.005, invalid_delta=0.003). "
                "This may indicate overfitting or lack of sensitivity."
            ],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert result["status"] == "warn"
        assert "too similar" in result["summary"].lower()
        assert result["conflict"] is False
    
    def test_control_arm_for_alignment_view_summary_from_top_red_flag(self):
        """GGFL adapter should use top red flag as summary."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [
                "First red flag",
                "Second red flag",
            ],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert result["status"] == "warn"
        assert result["summary"] == "First red flag"
        assert result["conflict"] is False
    
    def test_control_arm_for_alignment_view_never_conflicts(self):
        """GGFL adapter should always return conflict=False."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [
                "Experiment 'CAL-EXP-001': Control and twin have matching status with "
                "similar rates. This suggests the pipeline may not be distinguishing "
                "between expected stochasticity (control) and actual behavior (twin)."
            ],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert result["conflict"] is False
        assert result["status"] == "warn"
    
    def test_control_arm_for_alignment_view_deterministic(self):
        """GGFL adapter should be deterministic for same inputs."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": ["Test red flag"],
        }
        
        result1 = control_arm_for_alignment_view(panel)
        result2 = control_arm_for_alignment_view(panel)
        
        assert result1 == result2
    
    def test_control_arm_for_alignment_view_json_safe(self):
        """GGFL adapter output should be JSON-serializable."""
        import json
        
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed == result
    
    def test_control_arm_for_alignment_view_shape(self):
        """GGFL adapter should return correct shape with all required fields."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert "status" in result
        assert "summary" in result
        assert "conflict" in result
        assert isinstance(result["status"], str)
        assert isinstance(result["summary"], str)
        assert isinstance(result["conflict"], bool)
        assert result["status"] in ("ok", "warn", "block")


class TestControlArmSignalConsistency:
    """Test control arm signal consistency cross-check."""
    
    def test_summarize_control_arm_signal_consistency_consistent(self):
        """Consistency check should return CONSISTENT when signals match."""
        status_signal = {
            "status": "OK",
            "red_flag_count": 0,
            "weight_hint": "LOW",
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "recommendation": "NONE",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False
        assert result["schema_version"] == "1.0.0"
        assert result["mode"] == "SHADOW"
        assert any("consistent" in note.lower() for note in result["notes"])
    
    def test_summarize_control_arm_signal_consistency_status_mismatch(self):
        """Consistency check should detect status mismatch."""
        status_signal = {
            "status": "WARN",
            "red_flag_count": 1,
            "weight_hint": "LOW",
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",  # Mismatch: status says WARN but GGFL says ok
            "conflict": False,
            "weight_hint": "LOW",
            "recommendation": "WARNING",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        assert result["consistency"] == "PARTIAL"
        assert result["conflict_invariant_violated"] is False
        assert any("status mismatch" in note.lower() for note in result["notes"])
    
    def test_summarize_control_arm_signal_consistency_recommendation_mismatch(self):
        """Consistency check should detect recommendation mismatch."""
        status_signal = {
            "status": "OK",
            "red_flag_count": 0,
            "weight_hint": "LOW",
            "recommendation": "WARNING",  # Mismatch: should be NONE
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "recommendation": "NONE",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        assert result["consistency"] == "PARTIAL"
        assert result["conflict_invariant_violated"] is False
        assert any("recommendation mismatch" in note.lower() for note in result["notes"])
    
    def test_summarize_control_arm_signal_consistency_conflict_invariant_violated(self):
        """Consistency check should flag conflict invariant violation as INCONSISTENT."""
        status_signal = {
            "status": "OK",
            "red_flag_count": 0,
            "weight_hint": "LOW",
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",
            "conflict": True,  # VIOLATION: conflict must always be False
            "weight_hint": "LOW",
            "recommendation": "NONE",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        assert result["consistency"] == "INCONSISTENT"
        assert result["conflict_invariant_violated"] is True
        assert result["top_mismatch_type"] == "conflict_invariant_violated"
        assert any("conflict invariant violated" in note.lower() for note in result["notes"])
        assert any("conflict=true" in note.lower() for note in result["notes"])
    
    def test_summarize_control_arm_signal_consistency_warn_status_consistent(self):
        """Consistency check should handle WARN status correctly."""
        status_signal = {
            "status": "WARN",
            "red_flag_count": 1,
            "weight_hint": "LOW",
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "warn",
            "conflict": False,
            "weight_hint": "LOW",
            "recommendation": "WARNING",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False
    
    def test_summarize_control_arm_signal_consistency_missing_recommendation(self):
        """Consistency check should handle missing recommendation in status signal."""
        status_signal = {
            "status": "OK",
            "red_flag_count": 0,
            "weight_hint": "LOW",
            # recommendation not present
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "recommendation": "NONE",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        # Should be consistent if recommendation is missing from status (optional field)
        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False
    
    def test_summarize_control_arm_signal_consistency_missing_recommendation_regression(self):
        """REGRESSION: Missing recommendation should not produce false INCONSISTENT.
        
        If recommendation is missing but there's a status mismatch, it should be
        PARTIAL (not INCONSISTENT) unless conflict is True.
        """
        status_signal = {
            "status": "WARN",
            "red_flag_count": 1,
            "weight_hint": "LOW",
            # recommendation not present (missing)
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",  # Mismatch: status says WARN but GGFL says ok
            "conflict": False,  # Conflict is False (not violated)
            "weight_hint": "LOW",
            "recommendation": "WARNING",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        # Should be PARTIAL (not INCONSISTENT) because:
        # 1. Missing recommendation is optional, doesn't cause INCONSISTENT
        # 2. Status mismatch causes PARTIAL
        # 3. Conflict is False, so no conflict invariant violation
        assert result["consistency"] == "PARTIAL"
        assert result["conflict_invariant_violated"] is False
        assert result["top_mismatch_type"] == "status_mismatch"
        assert any("status mismatch" in note.lower() for note in result["notes"])
    
    def test_summarize_control_arm_signal_consistency_json_safe(self):
        """Consistency check result should be JSON-serializable."""
        import json
        
        status_signal = {
            "status": "OK",
            "red_flag_count": 0,
            "weight_hint": "LOW",
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "recommendation": "NONE",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed == result
    
    def test_summarize_control_arm_signal_consistency_shape(self):
        """Consistency check result should have correct shape."""
        status_signal = {
            "status": "OK",
            "red_flag_count": 0,
            "weight_hint": "LOW",
        }
        ggfl_signal = {
            "signal_type": "SIG-CTRL",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
            "recommendation": "NONE",
        }
        
        result = summarize_control_arm_signal_consistency(status_signal, ggfl_signal)
        
        assert "schema_version" in result
        assert "mode" in result
        assert "consistency" in result
        assert "notes" in result
        assert "conflict_invariant_violated" in result
        assert "top_mismatch_type" in result
        assert result["schema_version"] == "1.0.0"
        assert result["mode"] == "SHADOW"
        assert isinstance(result["consistency"], str)
        assert isinstance(result["notes"], list)
        assert isinstance(result["conflict_invariant_violated"], bool)
        assert result["consistency"] in ("CONSISTENT", "PARTIAL", "INCONSISTENT")
    
    def test_control_arm_for_alignment_view_signal_type(self):
        """GGFL adapter should include signal_type='SIG-CTRL'."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert "signal_type" in result
        assert result["signal_type"] == "SIG-CTRL"
    
    def test_control_arm_for_alignment_view_weight_hint(self):
        """GGFL adapter should include weight_hint='LOW'."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert "weight_hint" in result
        assert result["weight_hint"] == "LOW"
    
    def test_control_arm_for_alignment_view_recommendation_none(self):
        """GGFL adapter should return recommendation='NONE' when no red flags."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert "recommendation" in result
        assert result["recommendation"] == "NONE"
    
    def test_control_arm_for_alignment_view_recommendation_warning(self):
        """GGFL adapter should return recommendation='WARNING' when red flags present."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": ["Test red flag"],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        assert "recommendation" in result
        assert result["recommendation"] == "WARNING"
    
    def test_control_arm_for_alignment_view_conflict_always_false_regression(self):
        """REGRESSION: Conflict must always be False, even with red flags."""
        # Test with no red flags
        panel_no_flags = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        result_no_flags = control_arm_for_alignment_view(panel_no_flags)
        assert result_no_flags["conflict"] is False
        
        # Test with red flags
        panel_with_flags = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [
                "Experiment 'CAL-EXP-001': Control and twin metrics are too similar "
                "(abstention_delta=0.005, invalid_delta=0.003). "
                "This may indicate overfitting or lack of sensitivity."
            ],
        }
        result_with_flags = control_arm_for_alignment_view(panel_with_flags)
        assert result_with_flags["conflict"] is False
        
        # Test with multiple red flags
        panel_multiple_flags = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001", "CAL-EXP-002"],
            "control_vs_twin_delta": {},
            "red_flags": [
                "Flag 1",
                "Flag 2",
                "Flag 3",
            ],
        }
        result_multiple_flags = control_arm_for_alignment_view(panel_multiple_flags)
        assert result_multiple_flags["conflict"] is False
        
        # Verify all results have conflict=False
        assert all(r["conflict"] is False for r in [
            result_no_flags,
            result_with_flags,
            result_multiple_flags,
        ])
    
    def test_control_arm_for_alignment_view_backward_compatible(self):
        """GGFL adapter should maintain backward compatibility with existing fields."""
        panel = {
            "schema_version": CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            "experiments": ["CAL-EXP-001"],
            "control_vs_twin_delta": {},
            "red_flags": [],
        }
        
        result = control_arm_for_alignment_view(panel)
        
        # Existing fields must still be present
        assert "status" in result
        assert "summary" in result
        assert "conflict" in result
        
        # New fields are additive
        assert "signal_type" in result
        assert "weight_hint" in result
        assert "recommendation" in result

