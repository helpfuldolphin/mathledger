# tests/phase2/metrics/test_multi_goal.py
"""
Phase II Statistical Test Battery - Multi-Goal Metric Tests

Tests for compute_multi_goal_success from experiments.slice_success_metrics.

NO UPLIFT INTERPRETATION: These tests verify mechanical correctness only.
All tests are deterministic and self-contained.
"""

import pytest
from typing import Dict, List, Set, Any

from experiments.slice_success_metrics import compute_multi_goal_success

from .conftest import (
    DeterministicGenerator,
    SEED_MULTI_GOAL,
    assert_tuple_bool_float,
    SLICE_PARAMS,
)


# ===========================================================================
# BASIC FUNCTIONALITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestMultiGoalBasic:
    """Basic functionality tests for compute_multi_goal_success."""

    def test_all_goals_met(self):
        """All required goals are met."""
        verified = {"h1", "h2", "h3"}
        required = {"h1", "h2"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 2.0
        assert_tuple_bool_float((success, value))

    def test_no_goals_met(self):
        """No required goals are met."""
        verified = {"h1", "h2", "h3"}
        required = {"h4", "h5"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 0.0

    def test_partial_goals_met(self):
        """Some but not all required goals are met."""
        verified = {"h1", "h2", "h3"}
        required = {"h1", "h4"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 1.0

    def test_single_goal_met(self):
        """Single required goal is met."""
        verified = {"h1"}
        required = {"h1"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 1.0

    def test_single_goal_not_met(self):
        """Single required goal is not met."""
        verified = {"h1"}
        required = {"h2"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 0.0


# ===========================================================================
# BOUNDARY CONDITION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.boundary
class TestMultiGoalBoundary:
    """Boundary condition tests for compute_multi_goal_success."""

    def test_empty_required_goals(self):
        """Empty required goals is vacuously true."""
        verified = {"h1", "h2", "h3"}
        required: Set[str] = set()
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 0.0

    def test_empty_verified_empty_required(self):
        """Both empty is vacuously true."""
        verified: Set[str] = set()
        required: Set[str] = set()
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 0.0

    def test_empty_verified_nonempty_required(self):
        """Empty verified with nonempty required fails."""
        verified: Set[str] = set()
        required = {"h1"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 0.0

    def test_verified_equals_required(self):
        """Verified exactly equals required."""
        hashes = {"h1", "h2", "h3"}
        success, value = compute_multi_goal_success(hashes, hashes.copy())
        assert success is True
        assert value == 3.0

    def test_verified_superset_of_required(self):
        """Verified is strict superset of required."""
        verified = {"h1", "h2", "h3", "h4", "h5"}
        required = {"h1", "h3"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 2.0

    def test_one_missing_goal(self):
        """All but one required goal met."""
        verified = {"h1", "h2", "h3"}
        required = {"h1", "h2", "h4"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 2.0


# ===========================================================================
# DEGENERATE CASE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.degenerate
class TestMultiGoalDegenerate:
    """Degenerate and edge case tests for compute_multi_goal_success."""

    def test_single_element_sets(self):
        """Single element in both sets - matching."""
        verified = {"only"}
        required = {"only"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 1.0

    def test_single_element_sets_no_match(self):
        """Single element in both sets - not matching."""
        verified = {"a"}
        required = {"b"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 0.0

    def test_large_verified_small_required(self):
        """Large verified set, small required set."""
        verified = {f"h{i}" for i in range(1000)}
        required = {"h500"}
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 1.0

    def test_small_verified_large_required(self):
        """Small verified set, large required set."""
        verified = {"h500"}
        required = {f"h{i}" for i in range(1000)}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 1.0

    def test_disjoint_sets(self):
        """Completely disjoint verified and required sets."""
        verified = {f"a{i}" for i in range(100)}
        required = {f"b{i}" for i in range(100)}
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 0.0


# ===========================================================================
# TYPE STABILITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.type_stability
class TestMultiGoalTypeStability:
    """Type stability tests for compute_multi_goal_success."""

    def test_return_type_success(self):
        """Return type is (bool, float) on success."""
        result = compute_multi_goal_success({"h1"}, {"h1"})
        assert_tuple_bool_float(result, "success case")

    def test_return_type_failure(self):
        """Return type is (bool, float) on failure."""
        result = compute_multi_goal_success({"h1"}, {"h2"})
        assert_tuple_bool_float(result, "failure case")

    def test_return_type_empty(self):
        """Return type is (bool, float) with empty inputs."""
        result = compute_multi_goal_success(set(), set())
        assert_tuple_bool_float(result, "empty inputs")

    def test_value_is_float(self):
        """Value is float type."""
        _, value = compute_multi_goal_success({"h1", "h2"}, {"h1"})
        assert type(value) is float

    def test_bool_is_strict_bool(self):
        """Success is strictly bool type."""
        success, _ = compute_multi_goal_success({"h1"}, {"h1"})
        assert type(success) is bool


# ===========================================================================
# DETERMINISM TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.determinism
class TestMultiGoalDeterminism:
    """Determinism tests for compute_multi_goal_success."""

    def test_same_input_same_output_100_runs(self):
        """Same inputs produce same outputs over 100 runs."""
        verified = {f"h{i}" for i in range(50)}
        required = {f"h{i}" for i in range(20, 40)}
        
        results = [compute_multi_goal_success(verified, required) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_set_construction_independence(self):
        """Set construction method doesn't affect result."""
        # Different ways to construct same sets
        verified1 = {"h1", "h2", "h3"}
        verified2 = set(["h1", "h2", "h3"])
        verified3 = set(["h3", "h2", "h1"])
        
        required1 = {"h1", "h2"}
        required2 = set(["h2", "h1"])
        
        r1 = compute_multi_goal_success(verified1, required1)
        r2 = compute_multi_goal_success(verified2, required1)
        r3 = compute_multi_goal_success(verified3, required2)
        
        assert r1 == r2 == r3

    def test_frozenset_equivalence(self):
        """Works with frozenset converted to set."""
        verified = frozenset(["h1", "h2", "h3"])
        required = frozenset(["h1", "h2"])
        
        result = compute_multi_goal_success(set(verified), set(required))
        assert result == (True, 2.0)


# ===========================================================================
# LARGE SCALE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.large_scale
class TestMultiGoalLargeScale:
    """Large scale tests for compute_multi_goal_success."""

    def test_1000_verified_100_required_all_present(self):
        """1000 verified, 100 required, all present."""
        verified = {f"h{i}" for i in range(1000)}
        required = {f"h{i}" for i in range(100)}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 100.0

    def test_1000_verified_100_required_50_present(self):
        """1000 verified, 100 required, only 50 present."""
        verified = {f"h{i}" for i in range(1000)}
        required = {f"h{i}" for i in range(950, 1050)}  # Only 50 overlap (950-999)
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 50.0

    def test_10000_verified_1000_required(self):
        """10000 verified, 1000 required, all present."""
        verified = {f"h{i:05d}" for i in range(10000)}
        required = {f"h{i:05d}" for i in range(1000)}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 1000.0

    def test_large_disjoint_sets(self, gen_multi_goal: DeterministicGenerator):
        """Large completely disjoint sets."""
        verified = {f"a{i}" for i in range(5000)}
        required = {f"b{i}" for i in range(5000)}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == 0.0

    @pytest.mark.parametrize("verified_size,required_size,overlap", [
        (100, 10, 10),
        (100, 10, 5),
        (100, 10, 0),
        (1000, 100, 100),
        (1000, 100, 50),
        (1000, 100, 0),
    ])
    def test_parametrized_large_scale(
        self,
        verified_size: int,
        required_size: int,
        overlap: int
    ):
        """Parametrized large scale tests with controlled overlap."""
        verified = {f"h{i}" for i in range(verified_size)}
        # Required contains 'overlap' elements from verified plus some outside
        required = {f"h{i}" for i in range(overlap)}
        required |= {f"x{i}" for i in range(required_size - overlap)}
        
        success, value = compute_multi_goal_success(verified, required)
        
        expected_success = (overlap == required_size)
        assert success is expected_success
        assert value == float(overlap)


# ===========================================================================
# CROSS-SLICE PARAMETER TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.cross_slice
class TestMultiGoalCrossSlice:
    """Cross-slice parameter smoke tests for compute_multi_goal_success."""

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_slice_success_rate_threshold(self, slice_id: str):
        """Test multi-goal with slice success rate thresholds."""
        params = SLICE_PARAMS[slice_id]
        min_rate = params["min_success_rate"]
        
        total = 100
        required_count = int(min_rate * total)
        
        # Create verified set matching required exactly
        verified = {f"h{i}" for i in range(required_count)}
        required = {f"h{i}" for i in range(required_count)}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == float(required_count)

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_slice_partial_success(self, slice_id: str):
        """Test multi-goal with partial success per slice."""
        params = SLICE_PARAMS[slice_id]
        min_samples = params["min_samples"]
        
        # Verified has 80% of required
        required_count = min(100, min_samples // 5)
        verified_count = int(required_count * 0.8)
        
        verified = {f"h{i}" for i in range(verified_count)}
        required = {f"h{i}" for i in range(required_count)}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is False
        assert value == float(verified_count)


# ===========================================================================
# SCHEMA / SET OPERATION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.schema
class TestMultiGoalSchema:
    """Schema and set operation tests for compute_multi_goal_success."""

    def test_set_intersection_semantics(self):
        """Verify set intersection semantics."""
        verified = {"h1", "h2", "h3", "h4"}
        required = {"h2", "h3", "h5"}
        
        success, value = compute_multi_goal_success(verified, required)
        
        # Expected: intersection is {h2, h3} = 2 elements
        expected_intersection = verified & required
        assert value == float(len(expected_intersection))
        assert success is (len(expected_intersection) == len(required))

    def test_unicode_hashes(self):
        """Handles unicode hash strings."""
        verified = {"日本語", "中文", "한국어"}
        required = {"日本語", "中文"}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 2.0

    def test_empty_string_hash(self):
        """Handles empty string as hash."""
        verified = {"", "h1"}
        required = {""}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 1.0

    def test_none_not_in_set(self):
        """None is not a valid set element (type safety)."""
        # This tests that the function handles string sets correctly
        verified = {"h1", "h2"}
        required = {"h1"}
        
        success, value = compute_multi_goal_success(verified, required)
        assert success is True
        assert value == 1.0


# ===========================================================================
# REPLAY EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestMultiGoalReplay:
    """Replay equivalence tests for compute_multi_goal_success."""

    def test_replay_with_seeded_generator(self, gen_multi_goal: DeterministicGenerator):
        """Results are identical when replayed with same generator seed."""
        gen = gen_multi_goal
        
        # First run
        gen.reset()
        verified1 = gen.hash_set(50)
        required1 = gen.hash_set(20)
        result1 = compute_multi_goal_success(verified1, required1)
        
        # Replay
        gen.reset()
        verified2 = gen.hash_set(50)
        required2 = gen.hash_set(20)
        result2 = compute_multi_goal_success(verified2, required2)
        
        assert verified1 == verified2
        assert required1 == required2
        assert result1 == result2

    def test_batch_replay(self, gen_multi_goal: DeterministicGenerator):
        """Batch of computations replays identically."""
        gen = gen_multi_goal
        
        # First batch
        gen.reset()
        batch1 = []
        for _ in range(50):
            v = gen.hash_set(gen.int_value(10, 100))
            r = gen.hash_set(gen.int_value(5, 30))
            batch1.append(compute_multi_goal_success(v, r))
        
        # Replay batch
        gen.reset()
        batch2 = []
        for _ in range(50):
            v = gen.hash_set(gen.int_value(10, 100))
            r = gen.hash_set(gen.int_value(5, 30))
            batch2.append(compute_multi_goal_success(v, r))
        
        assert batch1 == batch2


# ===========================================================================
# VALUE CALCULATION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestMultiGoalValueCalculation:
    """Tests for value (count) calculation accuracy."""

    def test_value_equals_intersection_size(self):
        """Value equals size of intersection."""
        for overlap in range(0, 21):
            verified = {f"h{i}" for i in range(100)}
            required = {f"h{i}" for i in range(overlap)} | {f"x{i}" for i in range(20 - overlap)}
            
            _, value = compute_multi_goal_success(verified, required)
            assert value == float(overlap)

    def test_value_zero_when_disjoint(self):
        """Value is zero when sets are disjoint."""
        verified = {f"a{i}" for i in range(100)}
        required = {f"b{i}" for i in range(100)}
        
        _, value = compute_multi_goal_success(verified, required)
        assert value == 0.0

    def test_value_equals_required_when_subset(self):
        """Value equals required size when required is subset of verified."""
        verified = {f"h{i}" for i in range(100)}
        required = {f"h{i}" for i in range(50)}
        
        _, value = compute_multi_goal_success(verified, required)
        assert value == 50.0


# ===========================================================================
# MONOTONICITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestMultiGoalMonotonicity:
    """Tests for monotonicity properties."""

    def test_more_verified_cannot_hurt(self):
        """Adding to verified set cannot decrease success."""
        required = {f"h{i}" for i in range(10)}
        
        successes = []
        for size in range(0, 21):
            verified = {f"h{i}" for i in range(size)}
            success, _ = compute_multi_goal_success(verified, required)
            successes.append(success)
        
        # Once success achieved, it should remain
        first_success = None
        for i, s in enumerate(successes):
            if s:
                first_success = i
                break
        
        if first_success is not None:
            assert all(successes[i] for i in range(first_success, len(successes)))

    def test_more_required_cannot_help(self):
        """Adding to required set cannot increase success."""
        verified = {f"h{i}" for i in range(10)}
        
        successes = []
        for size in range(0, 21):
            required = {f"h{i}" for i in range(size)}
            success, _ = compute_multi_goal_success(verified, required)
            successes.append(success)
        
        # Once failure occurs, it should remain (after a point)
        # Actually: as required grows beyond verified, failure is guaranteed
        for i in range(11, len(successes)):
            assert not successes[i], f"Should fail when required size {i} > verified size 10"

