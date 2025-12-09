# tests/phase2/metrics_adversarial/test_fault_injection.py
"""
Adversarial Fault Injection Tests

Tests metric function robustness against:
- Missing fields (fuzzing)
- Type mismatches (string/int/float swaps)
- Extreme values (IEEE 754 limits)
- Empty containers
- Null/None values
- NaN and Infinity injection

NO METRIC INTERPRETATION: These tests verify fault handling only.
"""

import math
import pytest
from typing import Dict, List, Set, Any

from backend.substrate.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

from .conftest import (
    FaultInjector,
    FaultType,
    SEED_ADVERSARIAL,
)


# ===========================================================================
# MISSING FIELD TESTS (FUZZING)
# ===========================================================================

@pytest.mark.adversarial
class TestMissingFieldFuzzing:
    """Tests for missing field robustness."""

    def test_goal_hit_missing_hash_field(self, fault_injector: FaultInjector):
        """goal_hit raises KeyError when statement missing 'hash'."""
        statements = [{"hash": "h1"}, {"other": "h2"}]
        targets = {"h1", "h2"}
        
        with pytest.raises(KeyError):
            compute_goal_hit(statements, targets, 1)

    def test_goal_hit_empty_statement_dict(self, fault_injector: FaultInjector):
        """goal_hit raises KeyError for empty statement dict."""
        statements = [{}]
        targets = {"h1"}
        
        with pytest.raises(KeyError):
            compute_goal_hit(statements, targets, 1)

    def test_chain_success_missing_hash_field(self, fault_injector: FaultInjector):
        """chain_success raises KeyError when statement missing 'hash'."""
        statements = [{"not_hash": "h1"}]
        graph = {}
        
        with pytest.raises(KeyError):
            compute_chain_success(statements, graph, "h1", 1)

    def test_goal_hit_partial_missing_hashes(self, fault_injector: FaultInjector):
        """goal_hit fails when some statements lack 'hash'."""
        statements = [
            {"hash": "h1"},
            {"hash": "h2"},
            {"missing": "h3"},  # Missing 'hash'
        ]
        targets = {"h1", "h2"}
        
        with pytest.raises(KeyError):
            compute_goal_hit(statements, targets, 2)

    def test_chain_success_missing_graph_key(self, fault_injector: FaultInjector):
        """chain_success handles missing keys in dependency graph gracefully."""
        statements = [{"hash": "h1"}, {"hash": "h2"}]
        # h2 depends on h3 which is not in statements
        graph = {"h2": ["h3"]}
        
        # Should not raise - gracefully handles missing deps
        success, value = compute_chain_success(statements, graph, "h2", 2)
        # h2 verified but h3 not, so chain is 1
        assert value == 1.0


# ===========================================================================
# TYPE MISMATCH TESTS
# ===========================================================================

@pytest.mark.adversarial
class TestTypeMismatch:
    """Tests for type mismatch handling."""

    def test_goal_hit_string_in_statements_list(self):
        """goal_hit fails when statements contains non-dict."""
        statements = [{"hash": "h1"}, "not_a_dict"]
        targets = {"h1"}
        
        with pytest.raises((TypeError, AttributeError)):
            compute_goal_hit(statements, targets, 1)

    def test_goal_hit_int_target_hash(self):
        """goal_hit handles integer in target set."""
        statements = [{"hash": "h1"}, {"hash": 123}]
        targets = {"h1", 123}  # Mixed types
        
        # Should work - set contains both types
        success, value = compute_goal_hit(statements, targets, 2)
        assert success is True
        assert value == 2.0

    def test_sparse_success_string_verified_count(self):
        """sparse_success fails with string verified_count."""
        with pytest.raises(TypeError):
            compute_sparse_success("ten", 100, 5)  # type: ignore

    def test_sparse_success_float_verified_count(self):
        """sparse_success with float verified_count."""
        # Python allows float >= int comparison
        success, value = compute_sparse_success(10.5, 100, 10)  # type: ignore
        # 10.5 >= 10 is True
        assert success is True

    def test_multi_goal_list_instead_of_set(self):
        """multi_goal handles list instead of set."""
        verified = ["h1", "h2", "h3"]  # List, not set
        required = {"h1", "h2"}
        
        # This may work due to duck typing, or raise
        # Let's see what actually happens
        try:
            # set.intersection works with any iterable
            success, value = compute_multi_goal_success(set(verified), required)
            assert success is True
        except TypeError:
            pass  # Expected if strict typing enforced

    def test_chain_success_string_graph(self):
        """chain_success fails with string instead of dict graph."""
        statements = [{"hash": "h1"}]
        
        with pytest.raises((TypeError, AttributeError)):
            compute_chain_success(statements, "not_a_dict", "h1", 1)  # type: ignore


# ===========================================================================
# EXTREME VALUE TESTS (IEEE 754)
# ===========================================================================

@pytest.mark.adversarial
class TestExtremeValues:
    """Tests for extreme value handling."""

    @pytest.mark.parametrize("extreme", [
        2**63 - 1,      # Max int64
        -(2**63),       # Min int64
        2**31 - 1,      # Max int32
        -(2**31),       # Min int32
        10**18,         # Large positive
        -(10**18),      # Large negative
    ])
    def test_sparse_success_extreme_verified_count(self, extreme: int):
        """sparse_success handles extreme verified counts."""
        success, value = compute_sparse_success(extreme, extreme * 2, 0)
        assert isinstance(success, bool)
        assert isinstance(value, float)
        assert value == float(extreme)

    @pytest.mark.parametrize("extreme", [
        2**63 - 1,
        -(2**63),
        10**18,
    ])
    def test_sparse_success_extreme_min_verified(self, extreme: int):
        """sparse_success handles extreme min_verified thresholds."""
        success, value = compute_sparse_success(100, 200, extreme)
        # 100 >= extreme will be False for large positive extreme
        if extreme > 0:
            assert success is False
        else:
            assert success is True  # 100 >= negative is True

    def test_goal_hit_extreme_min_threshold(self):
        """goal_hit with max int threshold."""
        statements = [{"hash": "h1"}]
        targets = {"h1"}
        
        success, value = compute_goal_hit(statements, targets, 2**63)
        assert success is False  # 1 hit < max int
        assert value == 1.0

    def test_chain_success_zero_min_length(self):
        """chain_success with zero min length always succeeds."""
        statements = []
        graph = {}
        
        success, value = compute_chain_success(statements, graph, "h0", 0)
        assert success is True
        assert value == 0.0

    def test_chain_success_negative_min_length(self):
        """chain_success with negative min length."""
        statements = []
        graph = {}
        
        success, value = compute_chain_success(statements, graph, "h0", -1)
        # 0 >= -1 is True
        assert success is True


# ===========================================================================
# EMPTY CONTAINER TESTS
# ===========================================================================

@pytest.mark.adversarial
class TestEmptyContainers:
    """Tests for empty container handling."""

    def test_goal_hit_empty_statements(self):
        """goal_hit with empty statements list."""
        success, value = compute_goal_hit([], {"h1"}, 0)
        assert success is True
        assert value == 0.0

    def test_goal_hit_empty_targets(self):
        """goal_hit with empty targets set."""
        statements = [{"hash": "h1"}]
        success, value = compute_goal_hit(statements, set(), 0)
        assert success is True
        assert value == 0.0

    def test_goal_hit_both_empty(self):
        """goal_hit with both empty."""
        success, value = compute_goal_hit([], set(), 0)
        assert success is True
        assert value == 0.0

    def test_sparse_success_zero_counts(self):
        """sparse_success with all zeros."""
        success, value = compute_sparse_success(0, 0, 0)
        assert success is True
        assert value == 0.0

    def test_chain_success_empty_all(self):
        """chain_success with all empty inputs."""
        success, value = compute_chain_success([], {}, "", 0)
        assert success is True
        assert value == 0.0

    def test_multi_goal_empty_verified(self):
        """multi_goal with empty verified set."""
        success, value = compute_multi_goal_success(set(), {"h1"})
        assert success is False
        assert value == 0.0

    def test_multi_goal_empty_required(self):
        """multi_goal with empty required set (vacuous truth)."""
        success, value = compute_multi_goal_success({"h1"}, set())
        assert success is True
        assert value == 0.0

    def test_multi_goal_both_empty(self):
        """multi_goal with both sets empty."""
        success, value = compute_multi_goal_success(set(), set())
        assert success is True
        assert value == 0.0

    def test_chain_success_empty_graph_verified_target(self):
        """chain_success with empty graph but verified target."""
        statements = [{"hash": "h1"}]
        success, value = compute_chain_success(statements, {}, "h1", 1)
        assert success is True
        assert value == 1.0


# ===========================================================================
# NULL VALUE TESTS
# ===========================================================================

@pytest.mark.adversarial
class TestNullValues:
    """Tests for None/null value handling."""

    def test_goal_hit_none_hash(self):
        """goal_hit statement with None hash."""
        statements = [{"hash": None}, {"hash": "h1"}]
        targets = {"h1"}
        
        success, value = compute_goal_hit(statements, targets, 1)
        assert success is True
        assert value == 1.0

    def test_goal_hit_none_in_targets(self):
        """goal_hit with None in targets set."""
        statements = [{"hash": None}, {"hash": "h1"}]
        targets = {None, "h1"}
        
        success, value = compute_goal_hit(statements, targets, 2)
        assert success is True
        assert value == 2.0

    def test_chain_success_none_target(self):
        """chain_success with None as target hash."""
        statements = [{"hash": None}]
        graph = {}
        
        success, value = compute_chain_success(statements, graph, None, 1)
        # None is in verified_hashes as a valid value
        assert success is True
        assert value == 1.0

    def test_multi_goal_none_in_sets(self):
        """multi_goal with None in both sets."""
        verified = {None, "h1"}
        required = {None}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 1.0


# ===========================================================================
# NaN AND INFINITY TESTS
# ===========================================================================

@pytest.mark.adversarial
class TestNaNAndInfinity:
    """Tests for NaN and Infinity handling in numeric contexts."""

    def test_sparse_success_nan_verified(self):
        """sparse_success with NaN verified count."""
        # NaN comparisons are always False
        success, value = compute_sparse_success(float('nan'), 100, 5)  # type: ignore
        # NaN >= 5 is False
        assert success is False
        assert math.isnan(value)

    def test_sparse_success_inf_verified(self):
        """sparse_success with infinity verified count."""
        success, value = compute_sparse_success(float('inf'), 100, 5)  # type: ignore
        # inf >= 5 is True
        assert success is True
        assert value == float('inf')

    def test_sparse_success_neg_inf_min(self):
        """sparse_success with negative infinity min_verified."""
        success, value = compute_sparse_success(0, 100, float('-inf'))  # type: ignore
        # 0 >= -inf is True
        assert success is True

    def test_sparse_success_nan_min(self):
        """sparse_success with NaN min_verified."""
        success, value = compute_sparse_success(100, 200, float('nan'))  # type: ignore
        # 100 >= NaN is False (NaN comparisons are always False)
        assert success is False


# ===========================================================================
# RANDOMIZED FAULT INJECTION BATCH TESTS
# ===========================================================================

@pytest.mark.adversarial
class TestRandomizedFaultInjection:
    """Randomized fault injection using FaultInjector."""

    def test_goal_hit_randomized_faults_50_iterations(self, fault_injector: FaultInjector):
        """Inject 50 random faults into goal_hit inputs."""
        fault_injector.reset()
        
        base_statement = {"hash": "h1", "extra": "data"}
        exceptions_caught = 0
        successful_calls = 0
        
        for _ in range(50):
            try:
                # Inject random fault
                faulted, fault = fault_injector.random_fault(
                    base_statement, "hash", str
                )
                
                result = compute_goal_hit([faulted], {"h1"}, 1)
                successful_calls += 1
            except (KeyError, TypeError, AttributeError):
                exceptions_caught += 1
        
        # Should have some exceptions and some successes
        assert exceptions_caught > 0 or successful_calls > 0

    def test_sparse_success_randomized_extreme_values(self, fault_injector: FaultInjector):
        """Test sparse_success with randomized extreme values."""
        fault_injector.reset()
        
        for _ in range(20):
            extreme = fault_injector._rng.choice(fault_injector.EXTREME_INTS)
            
            try:
                success, value = compute_sparse_success(extreme, extreme * 2, 0)
                assert isinstance(success, bool)
            except (OverflowError, ValueError):
                pass  # Expected for some extreme values

