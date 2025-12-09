"""
Tests for mock oracle negative control mode.

Verifies that negative control mode overrides all evaluations to return
verified=False with deterministic abstention semantics.

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import os

import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MockOracleConfig,
    MockVerifiableOracle,
    MockOracleExpectations,
)


@pytest.mark.unit
class TestNegativeControlBasics:
    """Basic tests for negative control mode."""
    
    def test_negative_control_config_option(self):
        """negative_control config option exists and defaults to False."""
        config = MockOracleConfig()
        assert config.negative_control is False
        
        config_nc = MockOracleConfig(negative_control=True)
        assert config_nc.negative_control is True
    
    def test_is_negative_control_property(self):
        """is_negative_control property reflects config."""
        oracle_normal = MockVerifiableOracle(MockOracleConfig(negative_control=False))
        assert oracle_normal.is_negative_control is False
        
        oracle_nc = MockVerifiableOracle(MockOracleConfig(negative_control=True))
        assert oracle_nc.is_negative_control is True
    
    def test_negative_control_overrides_all_results(self):
        """Negative control mode returns verified=False for all formulas."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        # These formulas would normally verify, fail, abstain, etc.
        formulas = [
            MockOracleExpectations.get_verified_formula("default"),
            MockOracleExpectations.get_failed_formula("default"),
            MockOracleExpectations.get_abstain_formula("default"),
            MockOracleExpectations.get_timeout_formula("default"),
        ]
        
        for formula in formulas:
            result = oracle.verify(formula)
            assert result.verified is False
            assert result.reason == "negative_control"
    
    def test_negative_control_reason_is_negative_control(self):
        """Negative control results have reason='negative_control'."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        result = oracle.verify("p -> p")
        assert result.reason == "negative_control"
    
    def test_negative_control_bucket_is_negative_control(self):
        """Negative control results have bucket='negative_control'."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        result = oracle.verify("p -> q")
        assert result.bucket == "negative_control"


@pytest.mark.unit
class TestNegativeControlSemantics:
    """Tests for negative control semantics."""
    
    def test_negative_control_sets_abstained_true(self):
        """Negative control mode sets abstained=True (as a form of non-result)."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        result = oracle.verify("p -> p")
        assert result.abstained is True
        assert result.verified is False
        assert result.timed_out is False
        assert result.crashed is False
    
    def test_negative_control_no_crashes(self):
        """Negative control mode never crashes, even with enable_crashes=True."""
        config = MockOracleConfig(
            negative_control=True,
            enable_crashes=True,
        )
        oracle = MockVerifiableOracle(config)
        
        # Get a formula that would normally crash
        crash_formula = MockOracleExpectations.get_crash_formula("default")
        
        # Should NOT raise, should return negative_control result
        result = oracle.verify(crash_formula)
        assert result.crashed is False
        assert result.reason == "negative_control"
    
    def test_negative_control_deterministic_latency(self):
        """Negative control mode has deterministic latency."""
        config = MockOracleConfig(negative_control=True, latency_jitter_pct=0.0)
        oracle = MockVerifiableOracle(config)
        
        formula = "p -> p"
        result1 = oracle.verify(formula)
        result2 = oracle.verify(formula)
        result3 = oracle.verify(formula)
        
        assert result1.latency_ms == result2.latency_ms == result3.latency_ms


@pytest.mark.unit
class TestNegativeControlStats:
    """Tests for negative control mode statistics."""
    
    def test_negative_control_suppresses_stats(self):
        """Negative control mode does NOT update stats."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        assert oracle.stats["total"] == 0
        
        # Verify multiple formulas
        for i in range(10):
            oracle.verify(f"formula_{i}")
        
        # Stats should still be zero (suppressed)
        assert oracle.stats["total"] == 0
        assert oracle.stats["verified"] == 0
        assert oracle.stats["abstain"] == 0
    
    def test_normal_mode_updates_stats(self):
        """Normal mode (negative_control=False) updates stats."""
        config = MockOracleConfig(negative_control=False)
        oracle = MockVerifiableOracle(config)
        
        for i in range(10):
            oracle.verify(f"formula_{i}")
        
        assert oracle.stats["total"] == 10


@pytest.mark.unit
class TestNegativeControlDeterminism:
    """Tests for negative control determinism."""
    
    def test_negative_control_same_input_same_output(self):
        """Same formula produces identical results in negative control mode."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        formula = "(p -> q) -> (~q -> ~p)"
        
        result1 = oracle.verify(formula)
        result2 = oracle.verify(formula)
        result3 = oracle.verify(formula)
        
        assert result1.verified == result2.verified == result3.verified
        assert result1.reason == result2.reason == result3.reason
        assert result1.latency_ms == result2.latency_ms == result3.latency_ms
        assert result1.hash_int == result2.hash_int == result3.hash_int
    
    def test_negative_control_consistent_across_instances(self):
        """Same formula produces identical results across oracle instances."""
        config = MockOracleConfig(negative_control=True, seed=42)
        
        oracle1 = MockVerifiableOracle(config)
        oracle2 = MockVerifiableOracle(config)
        
        formula = "p /\\ q -> p"
        
        result1 = oracle1.verify(formula)
        result2 = oracle2.verify(formula)
        
        assert result1.verified == result2.verified
        assert result1.latency_ms == result2.latency_ms
    
    def test_negative_control_hash_preserved(self):
        """Negative control mode still computes and returns hash."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        formula = "p -> p"
        result = oracle.verify(formula)
        
        # Hash should be computed (for audit trail)
        assert result.hash_int > 0


@pytest.mark.unit
class TestNegativeControlBatch:
    """Tests for negative control with batch verification."""
    
    def test_batch_all_negative_control(self):
        """Batch verification returns all negative_control results."""
        config = MockOracleConfig(negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        formulas = [
            MockOracleExpectations.get_verified_formula("default"),
            MockOracleExpectations.get_failed_formula("default"),
            MockOracleExpectations.get_crash_formula("default"),
        ]
        
        results = oracle.verify_batch(formulas)
        
        assert len(results) == 3
        for result in results:
            assert result.verified is False
            assert result.reason == "negative_control"


@pytest.mark.unit
class TestNegativeControlProfileInteraction:
    """Tests for negative control interaction with profiles."""
    
    def test_negative_control_ignores_profile(self):
        """Negative control overrides profile-specific behavior."""
        for profile in ["default", "goal_hit", "sparse", "tree", "dependency"]:
            config = MockOracleConfig(slice_profile=profile, negative_control=True)
            oracle = MockVerifiableOracle(config)
            
            result = oracle.verify("p -> p")
            assert result.verified is False
            assert result.reason == "negative_control"
    
    def test_profile_coverage_still_available(self):
        """profile_coverage() returns profile's theoretical coverage even in NC mode."""
        config = MockOracleConfig(slice_profile="goal_hit", negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        coverage = oracle.profile_coverage()
        
        # Theoretical coverage is for goal_hit
        assert coverage.profile_name == "goal_hit"
        assert coverage.verified_pct == 15.0
        
        # But actual results are negative_control
        result = oracle.verify("p -> p")
        assert result.verified is False
    
    def test_set_profile_preserves_negative_control(self):
        """set_profile() preserves negative_control setting."""
        config = MockOracleConfig(slice_profile="default", negative_control=True)
        oracle = MockVerifiableOracle(config)
        
        assert oracle.is_negative_control is True
        
        oracle.set_profile("goal_hit")
        
        # Still in negative control mode
        assert oracle.is_negative_control is True
        assert oracle.config.negative_control is True

