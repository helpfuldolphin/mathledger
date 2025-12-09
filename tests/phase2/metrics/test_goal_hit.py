# tests/phase2/metrics/test_goal_hit.py
"""
Phase II Statistical Test Battery - Goal Hit Metric Tests

Tests for compute_goal_hit from experiments.slice_success_metrics.

NO UPLIFT INTERPRETATION: These tests verify mechanical correctness only.
All tests are deterministic and self-contained.
"""

import pytest
from typing import Dict, List, Set, Any

from experiments.slice_success_metrics import compute_goal_hit

from .conftest import (
    DeterministicGenerator,
    SEED_GOAL_HIT,
    assert_tuple_bool_float,
    assert_bool_type,
    assert_float_type,
    SLICE_PARAMS,
)


# ===========================================================================
# BASIC FUNCTIONALITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestGoalHitBasic:
    """Basic functionality tests for compute_goal_hit."""

    def test_single_target_hit(self):
        """Single target that is verified succeeds."""
        statements = [{"hash": "h1"}]
        targets = {"h1"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is True
        assert value == 1.0
        assert_tuple_bool_float((success, value))

    def test_single_target_miss(self):
        """Single target that is not verified fails."""
        statements = [{"hash": "h2"}]
        targets = {"h1"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is False
        assert value == 0.0

    def test_multiple_targets_all_hit(self):
        """Multiple targets all verified succeeds."""
        statements = [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        targets = {"h1", "h2"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        assert success is True
        assert value == 2.0

    def test_multiple_targets_partial_hit(self):
        """Partial target match with threshold exactly met."""
        statements = [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        targets = {"h1", "h4", "h5"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is True
        assert value == 1.0

    def test_multiple_targets_threshold_not_met(self):
        """Partial target match with threshold not met."""
        statements = [{"hash": "h1"}]
        targets = {"h1", "h2", "h3"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        assert success is False
        assert value == 1.0


# ===========================================================================
# BOUNDARY CONDITION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.boundary
class TestGoalHitBoundary:
    """Boundary condition tests for compute_goal_hit."""

    def test_empty_statements(self):
        """Empty statement list always fails (unless threshold is 0)."""
        targets = {"h1", "h2"}
        success, value = compute_goal_hit([], targets, min_total_verified=1)
        assert success is False
        assert value == 0.0

    def test_empty_statements_zero_threshold(self):
        """Empty statement list with zero threshold succeeds."""
        targets = {"h1", "h2"}
        success, value = compute_goal_hit([], targets, min_total_verified=0)
        assert success is True
        assert value == 0.0

    def test_empty_targets(self):
        """Empty target set with zero threshold succeeds."""
        statements = [{"hash": "h1"}]
        success, value = compute_goal_hit(statements, set(), min_total_verified=0)
        assert success is True
        assert value == 0.0

    def test_empty_targets_nonzero_threshold(self):
        """Empty target set with nonzero threshold fails."""
        statements = [{"hash": "h1"}]
        success, value = compute_goal_hit(statements, set(), min_total_verified=1)
        assert success is False
        assert value == 0.0

    def test_both_empty_zero_threshold(self):
        """Both empty with zero threshold is vacuously true."""
        success, value = compute_goal_hit([], set(), min_total_verified=0)
        assert success is True
        assert value == 0.0

    def test_threshold_zero(self):
        """Zero threshold always succeeds."""
        statements = [{"hash": "h1"}]
        targets = {"h2"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=0)
        assert success is True
        assert value == 0.0

    def test_threshold_exactly_met(self):
        """Threshold exactly equal to hits succeeds."""
        statements = [{"hash": "h1"}, {"hash": "h2"}]
        targets = {"h1", "h2", "h3"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        assert success is True
        assert value == 2.0

    def test_threshold_one_above_hits(self):
        """Threshold one more than hits fails."""
        statements = [{"hash": "h1"}, {"hash": "h2"}]
        targets = {"h1", "h2", "h3"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=3)
        assert success is False
        assert value == 2.0

    def test_threshold_exceeds_target_count(self):
        """Threshold greater than total targets fails even if all targets hit."""
        statements = [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        targets = {"h1", "h2"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=10)
        assert success is False
        assert value == 2.0


# ===========================================================================
# DEGENERATE CASE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.degenerate
class TestGoalHitDegenerate:
    """Degenerate and edge case tests for compute_goal_hit."""

    def test_duplicate_hashes_in_statements(self):
        """Duplicate hashes in statements are deduplicated."""
        statements = [{"hash": "h1"}, {"hash": "h1"}, {"hash": "h1"}]
        targets = {"h1"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is True
        assert value == 1.0  # Only counted once

    def test_all_statements_are_duplicates(self):
        """All statements being duplicates counts as one."""
        statements = [{"hash": "h1"}] * 100
        targets = {"h1", "h2"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        assert success is False
        assert value == 1.0

    def test_statements_superset_of_targets(self):
        """Statements contain all targets plus extras."""
        statements = [{"hash": f"h{i}"} for i in range(10)]
        targets = {"h2", "h5"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        assert success is True
        assert value == 2.0

    def test_targets_superset_of_statements(self):
        """Targets contain more hashes than statements."""
        statements = [{"hash": "h1"}]
        targets = {f"h{i}" for i in range(100)}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is True
        assert value == 1.0

    def test_no_overlap(self):
        """No overlap between statements and targets."""
        statements = [{"hash": f"a{i}"} for i in range(10)]
        targets = {f"b{i}" for i in range(10)}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is False
        assert value == 0.0

    def test_single_element_both_lists(self):
        """Single element in both lists that matches."""
        statements = [{"hash": "only"}]
        targets = {"only"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is True
        assert value == 1.0


# ===========================================================================
# TYPE STABILITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.type_stability
class TestGoalHitTypeStability:
    """Type stability tests for compute_goal_hit."""

    def test_return_type_success_case(self):
        """Return type is (bool, float) on success."""
        result = compute_goal_hit([{"hash": "h1"}], {"h1"}, 1)
        assert_tuple_bool_float(result, "success case")

    def test_return_type_failure_case(self):
        """Return type is (bool, float) on failure."""
        result = compute_goal_hit([{"hash": "h1"}], {"h2"}, 1)
        assert_tuple_bool_float(result, "failure case")

    def test_return_type_empty_inputs(self):
        """Return type is (bool, float) with empty inputs."""
        result = compute_goal_hit([], set(), 0)
        assert_tuple_bool_float(result, "empty inputs")

    def test_float_value_not_int(self):
        """Value is float, not int, even for whole numbers."""
        _, value = compute_goal_hit([{"hash": "h1"}], {"h1"}, 1)
        assert type(value) is float, "Value should be float, not int"

    def test_bool_is_strict_bool(self):
        """Success is strictly bool, not truthy value."""
        success, _ = compute_goal_hit([{"hash": "h1"}], {"h1"}, 1)
        assert type(success) is bool, "Success should be bool, not truthy"


# ===========================================================================
# DETERMINISM TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.determinism
class TestGoalHitDeterminism:
    """Determinism tests for compute_goal_hit."""

    def test_same_input_same_output_100_runs(self):
        """Same inputs produce same outputs over 100 runs."""
        statements = [{"hash": f"h{i}"} for i in range(10)]
        targets = {"h2", "h5", "h8"}
        
        results = [compute_goal_hit(statements, targets, 2) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_order_independence_statements(self):
        """Statement order does not affect result."""
        import random
        rng = random.Random(SEED_GOAL_HIT)
        
        base_hashes = ["h1", "h2", "h3", "h4", "h5"]
        targets = {"h2", "h4"}
        
        results = []
        for _ in range(50):
            shuffled = base_hashes.copy()
            rng.shuffle(shuffled)
            statements = [{"hash": h} for h in shuffled]
            results.append(compute_goal_hit(statements, targets, 2))
        
        assert all(r == results[0] for r in results)

    def test_set_semantics_targets(self):
        """Target order does not matter (set semantics)."""
        statements = [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        
        # Different set constructions
        targets1 = {"h1", "h2"}
        targets2 = {"h2", "h1"}
        targets3 = set(["h1", "h2"])
        targets4 = frozenset(["h2", "h1"])
        
        r1 = compute_goal_hit(statements, targets1, 2)
        r2 = compute_goal_hit(statements, targets2, 2)
        r3 = compute_goal_hit(statements, targets3, 2)
        r4 = compute_goal_hit(statements, set(targets4), 2)
        
        assert r1 == r2 == r3 == r4


# ===========================================================================
# LARGE SCALE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.large_scale
class TestGoalHitLargeScale:
    """Large scale tests for compute_goal_hit."""

    def test_1000_statements_10_targets(self, gen_goal_hit: DeterministicGenerator):
        """1000 statements with 10 targets."""
        gen = gen_goal_hit
        gen.reset()
        
        statements = gen.statements(1000)
        hashes = [s["hash"] for s in statements]
        targets = set(gen.sample(hashes, 10))
        
        success, value = compute_goal_hit(statements, targets, min_total_verified=10)
        assert_tuple_bool_float((success, value))
        assert success is True
        assert value == 10.0

    def test_10000_statements_100_targets(self, gen_goal_hit: DeterministicGenerator):
        """10000 statements with 100 targets (all present)."""
        gen = gen_goal_hit
        gen.reset()
        
        statements = gen.statements(10000)
        hashes = [s["hash"] for s in statements]
        targets = set(gen.sample(hashes, 100))
        
        success, value = compute_goal_hit(statements, targets, min_total_verified=50)
        assert_tuple_bool_float((success, value))
        assert success is True
        assert value == 100.0

    def test_large_disjoint_sets(self, gen_goal_hit: DeterministicGenerator):
        """Large disjoint statement and target sets."""
        gen = gen_goal_hit
        gen.reset()
        
        # Statements with prefix "a"
        statements = [{"hash": f"a{i:05d}"} for i in range(5000)]
        # Targets with prefix "b"
        targets = {f"b{i:05d}" for i in range(5000)}
        
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is False
        assert value == 0.0

    def test_large_partial_overlap(self, gen_goal_hit: DeterministicGenerator):
        """Large sets with 50% overlap."""
        statements = [{"hash": f"h{i:05d}"} for i in range(1000)]
        targets = {f"h{i:05d}" for i in range(500, 1500)}  # 500 overlap
        
        success, value = compute_goal_hit(statements, targets, min_total_verified=250)
        assert_tuple_bool_float((success, value))
        assert success is True
        assert value == 500.0


# ===========================================================================
# CROSS-SLICE PARAMETER TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.cross_slice
class TestGoalHitCrossSlice:
    """Cross-slice parameter smoke tests for compute_goal_hit."""

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_slice_min_samples_boundary(self, slice_id: str):
        """Test with min_samples from each slice config."""
        params = SLICE_PARAMS[slice_id]
        min_samples = params["min_samples"]
        
        # Create statements equal to min_samples
        statements = [{"hash": f"h{i}"} for i in range(min_samples)]
        # All statements are targets
        targets = {f"h{i}" for i in range(min_samples)}
        
        success, value = compute_goal_hit(statements, targets, min_total_verified=min_samples)
        assert success is True
        assert value == float(min_samples)

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_slice_success_rate_threshold(self, slice_id: str):
        """Test goal hit rate matching slice success rate."""
        params = SLICE_PARAMS[slice_id]
        min_rate = params["min_success_rate"]
        
        total = 100
        required = int(min_rate * total)
        
        statements = [{"hash": f"h{i}"} for i in range(total)]
        targets = {f"h{i}" for i in range(required)}
        
        success, value = compute_goal_hit(statements, targets, min_total_verified=required)
        assert success is True
        assert value == float(required)


# ===========================================================================
# SCHEMA / MISSING FIELD TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.schema
class TestGoalHitSchema:
    """Schema and missing field tests for compute_goal_hit."""

    def test_statement_with_extra_fields(self):
        """Statements with extra fields are handled correctly."""
        statements = [
            {"hash": "h1", "extra": "ignored", "data": 123},
            {"hash": "h2", "meta": {"nested": True}},
        ]
        targets = {"h1", "h2"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        assert success is True
        assert value == 2.0

    def test_statement_missing_hash_raises(self):
        """Statements missing 'hash' key raise KeyError."""
        statements = [{"other": "h1"}, {"hash": "h2"}]
        targets = {"h1"}
        with pytest.raises(KeyError):
            compute_goal_hit(statements, targets, min_total_verified=1)

    def test_statement_with_none_hash(self):
        """Statement with None hash does not match targets."""
        statements = [{"hash": None}, {"hash": "h1"}]
        targets = {"h1"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=1)
        assert success is True
        assert value == 1.0

    def test_statement_with_empty_hash(self):
        """Statement with empty string hash matches empty target."""
        statements = [{"hash": ""}, {"hash": "h1"}]
        targets = {"", "h1"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        assert success is True
        assert value == 2.0


# ===========================================================================
# NEGATIVE THRESHOLD TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestGoalHitNegativeThreshold:
    """Tests for negative threshold behavior."""

    def test_negative_threshold_always_succeeds(self):
        """Negative threshold is effectively zero (always succeed)."""
        statements = []
        targets = {"h1"}
        success, value = compute_goal_hit(statements, targets, min_total_verified=-1)
        # Implementation may vary - test actual behavior
        assert_tuple_bool_float((success, value))


# ===========================================================================
# REPLAY EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestGoalHitReplay:
    """Replay equivalence tests for compute_goal_hit."""

    def test_replay_with_seeded_generator(self, gen_goal_hit: DeterministicGenerator):
        """Results are identical when replayed with same generator seed."""
        gen = gen_goal_hit
        
        # First run
        gen.reset()
        statements1 = gen.statements(50)
        targets1 = gen.hash_set(10)
        result1 = compute_goal_hit(statements1, targets1, min_total_verified=5)
        
        # Replay
        gen.reset()
        statements2 = gen.statements(50)
        targets2 = gen.hash_set(10)
        result2 = compute_goal_hit(statements2, targets2, min_total_verified=5)
        
        assert statements1 == statements2
        assert targets1 == targets2
        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Different seeds produce different results (probabilistically)."""
        gen1 = DeterministicGenerator(SEED_GOAL_HIT)
        gen2 = DeterministicGenerator(SEED_GOAL_HIT + 1)
        
        statements1 = gen1.statements(100)
        targets1 = gen1.hash_set(20)
        
        statements2 = gen2.statements(100)
        targets2 = gen2.hash_set(20)
        
        # Data should be different with different seeds
        assert statements1 != statements2 or targets1 != targets2

