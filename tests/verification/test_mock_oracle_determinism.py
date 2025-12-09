"""
Tests for mock oracle determinism properties.

Verifies that the mock oracle produces identical results for:
- Same input across multiple calls
- Same input across multiple oracle instances
- Same input across different seeds (when seed affects only latency jitter)
- Batch vs single verification

ABSOLUTE SAFEGUARD: These tests exercise the mock oracle only — never production.
"""

from __future__ import annotations

import os
import pytest

os.environ["MATHLEDGER_ALLOW_MOCK_ORACLE"] = "1"

from backend.verification import (
    MockOracleConfig,
    MockVerifiableOracle,
    MockVerificationResult,
)


@pytest.mark.unit
class TestDeterministicHashing:
    """Tests for deterministic hash computation."""
    
    def test_same_formula_same_hash(self, default_oracle: MockVerifiableOracle):
        """Same formula always produces same hash."""
        formula = "p -> p"
        h1 = default_oracle._hash_formula(formula)
        h2 = default_oracle._hash_formula(formula)
        h3 = default_oracle._hash_formula(formula)
        
        assert h1 == h2 == h3
    
    def test_different_formulas_different_hashes(self, default_oracle: MockVerifiableOracle):
        """Different formulas produce different hashes (with high probability)."""
        formulas = [
            "p -> p",
            "p -> q",
            "q -> p",
            "p /\\ q",
            "p \\/ q",
        ]
        hashes = [default_oracle._hash_formula(f) for f in formulas]
        
        # All hashes should be unique
        assert len(set(hashes)) == len(hashes)
    
    def test_hash_is_integer(self, default_oracle: MockVerifiableOracle):
        """Hash function returns a positive integer."""
        h = default_oracle._hash_formula("p -> p")
        assert isinstance(h, int)
        assert h >= 0
    
    def test_hash_consistent_across_instances(self):
        """Same formula hashes identically across oracle instances."""
        formula = "((p -> q) /\\ (q -> r)) -> (p -> r)"
        
        oracle1 = MockVerifiableOracle(MockOracleConfig(seed=1))
        oracle2 = MockVerifiableOracle(MockOracleConfig(seed=2))
        oracle3 = MockVerifiableOracle(MockOracleConfig(seed=99))
        
        h1 = oracle1._hash_formula(formula)
        h2 = oracle2._hash_formula(formula)
        h3 = oracle3._hash_formula(formula)
        
        assert h1 == h2 == h3


@pytest.mark.unit
class TestDeterministicBucket:
    """Tests for deterministic bucket assignment."""
    
    def test_same_formula_same_bucket(self, default_oracle: MockVerifiableOracle):
        """Same formula always maps to same bucket."""
        formula = "p -> (q -> p)"
        
        bucket1 = default_oracle.get_expected_outcome(formula)
        bucket2 = default_oracle.get_expected_outcome(formula)
        bucket3 = default_oracle.get_expected_outcome(formula)
        
        assert bucket1 == bucket2 == bucket3
    
    def test_bucket_consistent_across_instances(self):
        """Same formula maps to same bucket across oracle instances."""
        formula = "~(p /\\ ~p)"
        
        config1 = MockOracleConfig(slice_profile="default", seed=1)
        config2 = MockOracleConfig(slice_profile="default", seed=999)
        
        oracle1 = MockVerifiableOracle(config1)
        oracle2 = MockVerifiableOracle(config2)
        
        assert oracle1.get_expected_outcome(formula) == oracle2.get_expected_outcome(formula)
    
    def test_bucket_depends_on_profile(self):
        """Same formula can map to different buckets under different profiles."""
        # Note: This tests that profiles CAN differ, not that they MUST
        # We use a formula that we've pre-computed to fall in different buckets
        formulas_tested = 0
        
        for i in range(100):
            formula = f"test_formula_{i}"
            buckets = {}
            
            for profile in ["default", "goal_hit", "sparse"]:
                config = MockOracleConfig(slice_profile=profile)
                oracle = MockVerifiableOracle(config)
                buckets[profile] = oracle.get_expected_outcome(formula)
            
            # Check if we found a formula with different buckets
            if len(set(buckets.values())) > 1:
                formulas_tested += 1
                break
        
        # Should find at least one formula that differs across profiles
        assert formulas_tested > 0, "Should find formula with different buckets across profiles"


@pytest.mark.unit
class TestDeterministicVerification:
    """Tests for deterministic verification results."""
    
    def test_same_formula_same_result(self, default_oracle: MockVerifiableOracle):
        """Same formula always produces identical result."""
        formula = "p \\/ ~p"
        
        result1 = default_oracle.verify(formula)
        result2 = default_oracle.verify(formula)
        result3 = default_oracle.verify(formula)
        
        assert result1.verified == result2.verified == result3.verified
        assert result1.abstained == result2.abstained == result3.abstained
        assert result1.timed_out == result2.timed_out == result3.timed_out
        assert result1.reason == result2.reason == result3.reason
        assert result1.bucket == result2.bucket == result3.bucket
        assert result1.latency_ms == result2.latency_ms == result3.latency_ms
    
    def test_batch_equals_sequential(self, default_oracle: MockVerifiableOracle, sample_formulas: list[str]):
        """Batch verification produces same results as sequential."""
        # Sequential verification
        sequential_results = [default_oracle.verify(f) for f in sample_formulas]
        
        # Reset stats for fair comparison
        default_oracle.reset_stats()
        
        # Batch verification
        batch_results = default_oracle.verify_batch(sample_formulas)
        
        assert len(sequential_results) == len(batch_results)
        
        for seq, batch in zip(sequential_results, batch_results):
            assert seq.verified == batch.verified
            assert seq.abstained == batch.abstained
            assert seq.timed_out == batch.timed_out
            assert seq.reason == batch.reason
            assert seq.bucket == batch.bucket
            assert seq.latency_ms == batch.latency_ms
    
    def test_result_consistent_across_instances(self):
        """Same formula produces identical results across oracle instances."""
        formula = "(p -> q) -> (~q -> ~p)"
        
        config = MockOracleConfig(slice_profile="default", seed=42)
        
        oracle1 = MockVerifiableOracle(config)
        oracle2 = MockVerifiableOracle(config)
        
        result1 = oracle1.verify(formula)
        result2 = oracle2.verify(formula)
        
        assert result1.verified == result2.verified
        assert result1.abstained == result2.abstained
        assert result1.timed_out == result2.timed_out
        assert result1.crashed == result2.crashed
        assert result1.reason == result2.reason
        assert result1.latency_ms == result2.latency_ms
        assert result1.bucket == result2.bucket


@pytest.mark.unit
class TestLatencyDeterminism:
    """Tests for deterministic latency computation."""
    
    def test_latency_is_deterministic(self, default_oracle: MockVerifiableOracle):
        """Same formula always produces same latency."""
        formula = "p /\\ q -> p"
        
        result1 = default_oracle.verify(formula)
        result2 = default_oracle.verify(formula)
        result3 = default_oracle.verify(formula)
        
        assert result1.latency_ms == result2.latency_ms == result3.latency_ms
    
    def test_latency_within_jitter_bounds(self, sample_formulas: list[str]):
        """Latency stays within configured jitter bounds."""
        config = MockOracleConfig(
            slice_profile="default",
            timeout_ms=100,
            latency_jitter_pct=0.20,  # ±20%
        )
        oracle = MockVerifiableOracle(config)
        
        for formula in sample_formulas:
            result = oracle.verify(formula)
            
            # Base latency depends on bucket
            if result.bucket == "timeout":
                base = config.timeout_ms
            else:
                from backend.verification.mock_config import BUCKET_BASE_LATENCY
                base = BUCKET_BASE_LATENCY[result.bucket]
            
            max_jitter = int(base * config.latency_jitter_pct)
            
            # Latency should be within base ± jitter (with floor at 0)
            assert result.latency_ms >= max(0, base - max_jitter)
            assert result.latency_ms <= base + max_jitter
    
    def test_zero_jitter_produces_exact_base(self):
        """With zero jitter, latency equals base latency exactly."""
        config = MockOracleConfig(
            slice_profile="default",
            timeout_ms=50,
            latency_jitter_pct=0.0,  # No jitter
        )
        oracle = MockVerifiableOracle(config)
        
        from backend.verification.mock_config import BUCKET_BASE_LATENCY
        
        # Test multiple formulas
        for i in range(20):
            formula = f"p -> p_{i}"
            result = oracle.verify(formula)
            
            if result.bucket == "timeout":
                expected = config.timeout_ms
            else:
                expected = BUCKET_BASE_LATENCY[result.bucket]
            
            assert result.latency_ms == expected


@pytest.mark.unit
class TestCrossRunDeterminism:
    """Tests simulating cross-run determinism."""
    
    def test_simulated_restart_same_results(self, sample_formulas: list[str]):
        """Simulated restart produces identical results."""
        config = MockOracleConfig(slice_profile="default", seed=42)
        
        # First "run"
        oracle1 = MockVerifiableOracle(config)
        results1 = [oracle1.verify(f) for f in sample_formulas]
        
        # Simulated restart - create fresh oracle
        oracle2 = MockVerifiableOracle(config)
        results2 = [oracle2.verify(f) for f in sample_formulas]
        
        for r1, r2 in zip(results1, results2):
            assert r1.verified == r2.verified
            assert r1.bucket == r2.bucket
            assert r1.latency_ms == r2.latency_ms
    
    def test_stats_dont_affect_results(self, sample_formulas: list[str]):
        """Stats tracking doesn't affect verification results."""
        config = MockOracleConfig(slice_profile="default", seed=42)
        oracle = MockVerifiableOracle(config)
        
        # First pass
        results1 = [oracle.verify(f) for f in sample_formulas]
        
        # Check stats were updated
        assert oracle.stats["total"] == len(sample_formulas)
        
        # Reset stats
        oracle.reset_stats()
        assert oracle.stats["total"] == 0
        
        # Second pass should give same results
        results2 = [oracle.verify(f) for f in sample_formulas]
        
        for r1, r2 in zip(results1, results2):
            assert r1.verified == r2.verified
            assert r1.bucket == r2.bucket

