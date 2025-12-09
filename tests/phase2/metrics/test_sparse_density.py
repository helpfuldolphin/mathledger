# tests/phase2/metrics/test_sparse_density.py
"""
Phase II Statistical Test Battery - Sparse/Density Metric Tests

Tests for compute_sparse_success from experiments.slice_success_metrics.

NO UPLIFT INTERPRETATION: These tests verify mechanical correctness only.
All tests are deterministic and self-contained.
"""

import pytest
from typing import List, Tuple

from experiments.slice_success_metrics import compute_sparse_success

from .conftest import (
    DeterministicGenerator,
    SEED_SPARSE_DENSITY,
    assert_tuple_bool_float,
    assert_bool_type,
    assert_float_type,
    SLICE_PARAMS,
)


# ===========================================================================
# BASIC FUNCTIONALITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestSparseDensityBasic:
    """Basic functionality tests for compute_sparse_success."""

    def test_verified_equals_min(self):
        """Verified count exactly equals minimum succeeds."""
        success, value = compute_sparse_success(verified_count=5, attempted_count=10, min_verified=5)
        assert success is True
        assert value == 5.0
        assert_tuple_bool_float((success, value))

    def test_verified_exceeds_min(self):
        """Verified count exceeds minimum succeeds."""
        success, value = compute_sparse_success(verified_count=10, attempted_count=20, min_verified=5)
        assert success is True
        assert value == 10.0

    def test_verified_below_min(self):
        """Verified count below minimum fails."""
        success, value = compute_sparse_success(verified_count=4, attempted_count=20, min_verified=5)
        assert success is False
        assert value == 4.0

    def test_zero_verified_nonzero_min(self):
        """Zero verified with nonzero minimum fails."""
        success, value = compute_sparse_success(verified_count=0, attempted_count=100, min_verified=1)
        assert success is False
        assert value == 0.0

    def test_nonzero_verified_zero_min(self):
        """Nonzero verified with zero minimum succeeds."""
        success, value = compute_sparse_success(verified_count=50, attempted_count=100, min_verified=0)
        assert success is True
        assert value == 50.0


# ===========================================================================
# BOUNDARY CONDITION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.boundary
class TestSparseDensityBoundary:
    """Boundary condition tests for compute_sparse_success."""

    def test_all_zeros(self):
        """All zeros is valid and succeeds (0 >= 0)."""
        success, value = compute_sparse_success(verified_count=0, attempted_count=0, min_verified=0)
        assert success is True
        assert value == 0.0

    def test_verified_zero_min_zero(self):
        """Zero verified with zero minimum succeeds."""
        success, value = compute_sparse_success(verified_count=0, attempted_count=100, min_verified=0)
        assert success is True
        assert value == 0.0

    def test_min_equals_one(self):
        """Minimum of 1 boundary test."""
        success, value = compute_sparse_success(verified_count=1, attempted_count=10, min_verified=1)
        assert success is True
        assert value == 1.0

    def test_min_one_verified_zero(self):
        """Minimum of 1 with zero verified fails."""
        success, value = compute_sparse_success(verified_count=0, attempted_count=10, min_verified=1)
        assert success is False
        assert value == 0.0

    def test_verified_equals_attempted(self):
        """100% verification rate."""
        success, value = compute_sparse_success(verified_count=100, attempted_count=100, min_verified=50)
        assert success is True
        assert value == 100.0

    def test_large_verified_count(self):
        """Very large verified count."""
        success, value = compute_sparse_success(verified_count=1000000, attempted_count=2000000, min_verified=999999)
        assert success is True
        assert value == 1000000.0

    def test_threshold_one_below(self):
        """Verified count one below threshold fails."""
        success, value = compute_sparse_success(verified_count=99, attempted_count=200, min_verified=100)
        assert success is False
        assert value == 99.0

    def test_threshold_one_above(self):
        """Verified count one above threshold succeeds."""
        success, value = compute_sparse_success(verified_count=101, attempted_count=200, min_verified=100)
        assert success is True
        assert value == 101.0


# ===========================================================================
# DEGENERATE CASE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.degenerate
class TestSparseDensityDegenerate:
    """Degenerate and edge case tests for compute_sparse_success."""

    def test_verified_exceeds_attempted(self):
        """Verified > attempted is technically valid (unusual but possible)."""
        success, value = compute_sparse_success(verified_count=150, attempted_count=100, min_verified=100)
        assert success is True
        assert value == 150.0

    def test_attempted_zero_verified_zero(self):
        """Zero attempted with zero verified."""
        success, value = compute_sparse_success(verified_count=0, attempted_count=0, min_verified=0)
        assert success is True
        assert value == 0.0

    def test_attempted_ignored(self):
        """Attempted count does not affect success (only verified and min matter)."""
        # Same verified and min, different attempted
        result1 = compute_sparse_success(verified_count=10, attempted_count=20, min_verified=5)
        result2 = compute_sparse_success(verified_count=10, attempted_count=100, min_verified=5)
        result3 = compute_sparse_success(verified_count=10, attempted_count=1000000, min_verified=5)
        
        assert result1 == result2 == result3

    def test_single_verified(self):
        """Single verified statement."""
        success, value = compute_sparse_success(verified_count=1, attempted_count=1, min_verified=1)
        assert success is True
        assert value == 1.0

    def test_very_sparse(self):
        """Very sparse verification (1 in million)."""
        success, value = compute_sparse_success(verified_count=1, attempted_count=1000000, min_verified=1)
        assert success is True
        assert value == 1.0

    def test_completely_failed(self):
        """Zero verified with high minimum."""
        success, value = compute_sparse_success(verified_count=0, attempted_count=1000, min_verified=500)
        assert success is False
        assert value == 0.0


# ===========================================================================
# TYPE STABILITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.type_stability
class TestSparseDensityTypeStability:
    """Type stability tests for compute_sparse_success."""

    def test_return_type_success(self):
        """Return type is (bool, float) on success."""
        result = compute_sparse_success(10, 20, 5)
        assert_tuple_bool_float(result, "success case")

    def test_return_type_failure(self):
        """Return type is (bool, float) on failure."""
        result = compute_sparse_success(2, 20, 5)
        assert_tuple_bool_float(result, "failure case")

    def test_return_type_zeros(self):
        """Return type is (bool, float) with zeros."""
        result = compute_sparse_success(0, 0, 0)
        assert_tuple_bool_float(result, "zero inputs")

    def test_value_is_float_not_int(self):
        """Value is float even for integer inputs."""
        _, value = compute_sparse_success(5, 10, 3)
        assert type(value) is float

    def test_bool_is_strict_bool(self):
        """Success is strictly bool type."""
        success, _ = compute_sparse_success(5, 10, 3)
        assert type(success) is bool


# ===========================================================================
# DETERMINISM TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.determinism
class TestSparseDensityDeterminism:
    """Determinism tests for compute_sparse_success."""

    def test_same_input_same_output_100_runs(self):
        """Same inputs produce same outputs over 100 runs."""
        results = [
            compute_sparse_success(verified_count=42, attempted_count=100, min_verified=30)
            for _ in range(100)
        ]
        assert all(r == results[0] for r in results)

    def test_determinism_across_parameter_space(self, gen_sparse_density: DeterministicGenerator):
        """Determinism across randomized parameter space."""
        gen = gen_sparse_density
        
        # Generate test cases
        gen.reset()
        test_cases = [
            (gen.int_value(0, 100), gen.int_value(0, 200), gen.int_value(0, 50))
            for _ in range(50)
        ]
        
        # Run twice and compare
        results1 = [compute_sparse_success(v, a, m) for v, a, m in test_cases]
        results2 = [compute_sparse_success(v, a, m) for v, a, m in test_cases]
        
        assert results1 == results2

    def test_argument_order_matters(self):
        """Argument order is significant."""
        r1 = compute_sparse_success(verified_count=10, attempted_count=5, min_verified=3)
        r2 = compute_sparse_success(verified_count=5, attempted_count=10, min_verified=3)
        # These should differ in value
        assert r1[1] != r2[1]


# ===========================================================================
# LARGE SCALE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.large_scale
class TestSparseDensityLargeScale:
    """Large scale tests for compute_sparse_success."""

    def test_million_verified(self):
        """Million verified statements."""
        success, value = compute_sparse_success(
            verified_count=1000000,
            attempted_count=2000000,
            min_verified=500000
        )
        assert success is True
        assert value == 1000000.0

    def test_billion_scale(self):
        """Billion scale (numerical stability)."""
        success, value = compute_sparse_success(
            verified_count=1000000000,
            attempted_count=2000000000,
            min_verified=999999999
        )
        assert success is True
        assert value == 1000000000.0

    @pytest.mark.parametrize("verified,attempted,min_ver,expected_success", [
        (0, 1000000, 1, False),
        (1, 1000000, 1, True),
        (500000, 1000000, 500000, True),
        (499999, 1000000, 500000, False),
        (1000000, 1000000, 1000000, True),
    ])
    def test_parametrized_large_scale(
        self,
        verified: int,
        attempted: int,
        min_ver: int,
        expected_success: bool
    ):
        """Parametrized large scale tests."""
        success, value = compute_sparse_success(verified, attempted, min_ver)
        assert success is expected_success
        assert value == float(verified)


# ===========================================================================
# CROSS-SLICE PARAMETER TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.cross_slice
class TestSparseDensityCrossSlice:
    """Cross-slice parameter smoke tests for compute_sparse_success."""

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_slice_min_samples(self, slice_id: str):
        """Test with min_samples from each slice config."""
        params = SLICE_PARAMS[slice_id]
        min_samples = params["min_samples"]
        
        success, value = compute_sparse_success(
            verified_count=min_samples,
            attempted_count=min_samples * 2,
            min_verified=min_samples
        )
        assert success is True
        assert value == float(min_samples)

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_slice_min_samples_minus_one(self, slice_id: str):
        """One below min_samples should fail."""
        params = SLICE_PARAMS[slice_id]
        min_samples = params["min_samples"]
        
        success, value = compute_sparse_success(
            verified_count=min_samples - 1,
            attempted_count=min_samples * 2,
            min_verified=min_samples
        )
        assert success is False
        assert value == float(min_samples - 1)


# ===========================================================================
# DENSITY CALCULATION TESTS (SEMANTIC)
# ===========================================================================

@pytest.mark.phase2_metrics
class TestSparseDensitySemantics:
    """Tests for density calculation semantics."""

    def test_50_percent_density(self):
        """50% density (verified/attempted)."""
        # Note: compute_sparse_success returns verified_count, not density ratio
        success, value = compute_sparse_success(
            verified_count=50,
            attempted_count=100,
            min_verified=50
        )
        assert success is True
        assert value == 50.0  # Returns count, not ratio

    def test_10_percent_density(self):
        """10% density."""
        success, value = compute_sparse_success(
            verified_count=10,
            attempted_count=100,
            min_verified=10
        )
        assert success is True
        assert value == 10.0

    def test_1_percent_density(self):
        """1% density."""
        success, value = compute_sparse_success(
            verified_count=1,
            attempted_count=100,
            min_verified=1
        )
        assert success is True
        assert value == 1.0

    def test_100_percent_density(self):
        """100% density."""
        success, value = compute_sparse_success(
            verified_count=100,
            attempted_count=100,
            min_verified=100
        )
        assert success is True
        assert value == 100.0


# ===========================================================================
# REPLAY EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestSparseDensityReplay:
    """Replay equivalence tests for compute_sparse_success."""

    def test_replay_with_seeded_generator(self, gen_sparse_density: DeterministicGenerator):
        """Results are identical when replayed with same generator seed."""
        gen = gen_sparse_density
        
        # First run
        gen.reset()
        v1, a1, m1 = gen.int_value(0, 100), gen.int_value(50, 200), gen.int_value(0, 50)
        result1 = compute_sparse_success(v1, a1, m1)
        
        # Replay
        gen.reset()
        v2, a2, m2 = gen.int_value(0, 100), gen.int_value(50, 200), gen.int_value(0, 50)
        result2 = compute_sparse_success(v2, a2, m2)
        
        assert (v1, a1, m1) == (v2, a2, m2)
        assert result1 == result2

    def test_batch_replay(self, gen_sparse_density: DeterministicGenerator):
        """Batch of computations replays identically."""
        gen = gen_sparse_density
        
        # First batch
        gen.reset()
        batch1 = [
            compute_sparse_success(
                gen.int_value(0, 100),
                gen.int_value(50, 200),
                gen.int_value(0, 50)
            )
            for _ in range(100)
        ]
        
        # Replay batch
        gen.reset()
        batch2 = [
            compute_sparse_success(
                gen.int_value(0, 100),
                gen.int_value(50, 200),
                gen.int_value(0, 50)
            )
            for _ in range(100)
        ]
        
        assert batch1 == batch2


# ===========================================================================
# NEGATIVE VALUE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestSparseDensityNegative:
    """Tests for negative value behavior."""

    def test_negative_verified(self):
        """Negative verified count behavior (edge case)."""
        # This tests robustness - negative values may not make semantic sense
        success, value = compute_sparse_success(
            verified_count=-1,
            attempted_count=100,
            min_verified=0
        )
        # Behavior: -1 < 0 is False if min_verified=0
        # But -1 >= -5 is True if min_verified=-5
        assert_tuple_bool_float((success, value))

    def test_negative_min_verified(self):
        """Negative minimum verified behavior."""
        success, value = compute_sparse_success(
            verified_count=0,
            attempted_count=100,
            min_verified=-1
        )
        # 0 >= -1 is True
        assert_tuple_bool_float((success, value))


# ===========================================================================
# MONOTONICITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestSparseDensityMonotonicity:
    """Tests for monotonicity properties."""

    def test_increasing_verified_increases_value(self):
        """Increasing verified count increases value."""
        values = []
        for v in range(0, 101, 10):
            _, value = compute_sparse_success(v, 200, 50)
            values.append(value)
        
        # Values should be monotonically increasing
        for i in range(1, len(values)):
            assert values[i] >= values[i-1]

    def test_increasing_min_decreases_success_rate(self):
        """Increasing min threshold decreases success likelihood."""
        success_count = 0
        for m in range(0, 101, 10):
            success, _ = compute_sparse_success(50, 100, m)
            if success:
                success_count += 1
        
        # First few should succeed, later ones fail
        # This tests that higher min makes success harder
        assert success_count > 0  # Some succeed
        assert success_count < 11  # Some fail

    def test_verified_equals_value(self):
        """Value always equals verified count."""
        for v in [0, 1, 10, 50, 100, 1000]:
            for a in [100, 200, 500]:
                for m in [0, 50]:
                    _, value = compute_sparse_success(v, a, m)
                    assert value == float(v)

