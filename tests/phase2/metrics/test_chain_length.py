# tests/phase2/metrics/test_chain_length.py
"""
Phase II Statistical Test Battery - Chain Length Metric Tests

Tests for compute_chain_success from experiments.slice_success_metrics.

NO UPLIFT INTERPRETATION: These tests verify mechanical correctness only.
All tests are deterministic and self-contained.
"""

import pytest
from typing import Dict, List, Set, Any, Tuple

from experiments.slice_success_metrics import compute_chain_success

from .conftest import (
    DeterministicGenerator,
    SEED_CHAIN_LENGTH,
    assert_tuple_bool_float,
    SLICE_PARAMS,
)


# ===========================================================================
# BASIC FUNCTIONALITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestChainLengthBasic:
    """Basic functionality tests for compute_chain_success."""

    def test_single_node_verified(self):
        """Single verified node has chain length 1."""
        statements = [{"hash": "h0"}]
        graph: Dict[str, List[str]] = {}
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=1)
        assert success is True
        assert value == 1.0
        assert_tuple_bool_float((success, value))

    def test_single_node_not_verified(self):
        """Target not verified has chain length 0."""
        statements: List[Dict[str, Any]] = []
        graph: Dict[str, List[str]] = {}
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=1)
        assert success is False
        assert value == 0.0

    def test_linear_chain_3(self):
        """Linear chain of 3: h0 <- h1 <- h2."""
        graph = {"h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}]
        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=3)
        assert success is True
        assert value == 3.0

    def test_linear_chain_partial(self):
        """Partial verification of linear chain."""
        graph = {"h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h1"}, {"hash": "h2"}]  # h0 missing
        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=3)
        assert success is False
        assert value == 2.0

    def test_diamond_graph(self):
        """Diamond dependency: h3 depends on h1 and h2, both depend on h0."""
        graph = {"h1": ["h0"], "h2": ["h0"], "h3": ["h1", "h2"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        success, value = compute_chain_success(statements, graph, "h3", min_chain_length=3)
        assert success is True
        assert value == 3.0  # h3 -> h1 -> h0 or h3 -> h2 -> h0


# ===========================================================================
# BOUNDARY CONDITION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.boundary
class TestChainLengthBoundary:
    """Boundary condition tests for compute_chain_success."""

    def test_empty_graph_target_verified(self):
        """Empty graph with verified target has length 1."""
        statements = [{"hash": "h0"}]
        graph: Dict[str, List[str]] = {}
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=1)
        assert success is True
        assert value == 1.0

    def test_empty_graph_target_not_verified(self):
        """Empty graph with unverified target has length 0."""
        statements: List[Dict[str, Any]] = []
        graph: Dict[str, List[str]] = {}
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=1)
        assert success is False
        assert value == 0.0

    def test_empty_statements(self):
        """Empty statements list always has length 0."""
        graph = {"h1": ["h0"]}
        statements: List[Dict[str, Any]] = []
        success, value = compute_chain_success(statements, graph, "h1", min_chain_length=1)
        assert success is False
        assert value == 0.0

    def test_min_chain_length_zero(self):
        """Zero min chain length succeeds even with unverified target."""
        # Actually, if target is not verified, chain length is 0
        # 0 >= 0 should succeed
        statements: List[Dict[str, Any]] = []
        graph: Dict[str, List[str]] = {}
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=0)
        assert success is True
        assert value == 0.0

    def test_min_chain_length_one(self):
        """Min chain length 1 requires target to be verified."""
        statements = [{"hash": "h0"}]
        graph: Dict[str, List[str]] = {}
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=1)
        assert success is True
        assert value == 1.0

    def test_chain_exactly_meets_threshold(self):
        """Chain length exactly equals threshold."""
        graph = {"h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}]
        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=3)
        assert success is True
        assert value == 3.0

    def test_chain_one_below_threshold(self):
        """Chain length one below threshold fails."""
        graph = {"h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}]
        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=4)
        assert success is False
        assert value == 3.0


# ===========================================================================
# DEGENERATE CASE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.degenerate
class TestChainLengthDegenerate:
    """Degenerate and edge case tests for compute_chain_success."""

    def test_target_not_in_graph(self):
        """Target not in graph but verified has length 1."""
        graph = {"h1": ["h0"]}
        statements = [{"hash": "h99"}]
        success, value = compute_chain_success(statements, graph, "h99", min_chain_length=1)
        assert success is True
        assert value == 1.0

    def test_cyclic_graph(self):
        """Cyclic graph does not cause infinite recursion."""
        # h0 -> h1 -> h2 -> h0 (cycle)
        graph = {"h0": ["h2"], "h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}]
        # Should complete without hanging
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=1)
        assert_tuple_bool_float((success, value))
        assert success is True

    def test_self_loop(self):
        """Node with self-loop dependency."""
        graph = {"h0": ["h0"]}
        statements = [{"hash": "h0"}]
        success, value = compute_chain_success(statements, graph, "h0", min_chain_length=1)
        assert_tuple_bool_float((success, value))
        assert success is True

    def test_disconnected_components(self):
        """Graph with disconnected components."""
        # h0 <- h1 (component 1)
        # h2 <- h3 (component 2)
        graph = {"h1": ["h0"], "h3": ["h2"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        
        # Target in component 1
        success1, value1 = compute_chain_success(statements, graph, "h1", min_chain_length=2)
        assert success1 is True
        assert value1 == 2.0
        
        # Target in component 2
        success2, value2 = compute_chain_success(statements, graph, "h3", min_chain_length=2)
        assert success2 is True
        assert value2 == 2.0

    def test_multiple_paths_same_length(self):
        """Multiple paths of same length to target."""
        # h0 <- h1 <- h3
        # h0 <- h2 <- h3
        graph = {"h1": ["h0"], "h2": ["h0"], "h3": ["h1", "h2"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        success, value = compute_chain_success(statements, graph, "h3", min_chain_length=3)
        assert success is True
        assert value == 3.0

    def test_multiple_paths_different_lengths(self):
        """Multiple paths of different lengths - longest is chosen."""
        # Path 1: h0 <- h3 (length 2)
        # Path 2: h0 <- h1 <- h2 <- h3 (length 4)
        graph = {"h1": ["h0"], "h2": ["h1"], "h3": ["h2", "h0"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        success, value = compute_chain_success(statements, graph, "h3", min_chain_length=4)
        assert success is True
        assert value == 4.0


# ===========================================================================
# DEEP CHAIN TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.large_scale
class TestChainLengthDeep:
    """Tests for deep dependency chains."""

    def test_chain_depth_10(self, gen_chain_length: DeterministicGenerator):
        """Chain of depth 10."""
        graph, hashes = gen_chain_length.dependency_graph_linear(10)
        statements = [{"hash": h} for h in hashes]
        target = hashes[-1]
        
        success, value = compute_chain_success(statements, graph, target, min_chain_length=10)
        assert success is True
        assert value == 10.0

    def test_chain_depth_50(self, gen_chain_length: DeterministicGenerator):
        """Chain of depth 50."""
        graph, hashes = gen_chain_length.dependency_graph_linear(50)
        statements = [{"hash": h} for h in hashes]
        target = hashes[-1]
        
        success, value = compute_chain_success(statements, graph, target, min_chain_length=50)
        assert success is True
        assert value == 50.0

    def test_chain_depth_100(self, gen_chain_length: DeterministicGenerator):
        """Chain of depth 100."""
        graph, hashes = gen_chain_length.dependency_graph_linear(100)
        statements = [{"hash": h} for h in hashes]
        target = hashes[-1]
        
        success, value = compute_chain_success(statements, graph, target, min_chain_length=100)
        assert success is True
        assert value == 100.0

    def test_chain_depth_200(self, gen_chain_length: DeterministicGenerator):
        """Deep chain of 200 nodes (within recursion limits)."""
        graph, hashes = gen_chain_length.dependency_graph_linear(200)
        statements = [{"hash": h} for h in hashes]
        target = hashes[-1]
        
        success, value = compute_chain_success(statements, graph, target, min_chain_length=200)
        assert success is True
        assert value == 200.0


# ===========================================================================
# TYPE STABILITY TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.type_stability
class TestChainLengthTypeStability:
    """Type stability tests for compute_chain_success."""

    def test_return_type_success(self):
        """Return type is (bool, float) on success."""
        statements = [{"hash": "h0"}]
        result = compute_chain_success(statements, {}, "h0", 1)
        assert_tuple_bool_float(result, "success case")

    def test_return_type_failure(self):
        """Return type is (bool, float) on failure."""
        result = compute_chain_success([], {}, "h0", 1)
        assert_tuple_bool_float(result, "failure case")

    def test_return_type_empty_all(self):
        """Return type is (bool, float) with all empty."""
        result = compute_chain_success([], {}, "h0", 0)
        assert_tuple_bool_float(result, "empty all")

    def test_value_is_float(self):
        """Value is float type."""
        _, value = compute_chain_success([{"hash": "h0"}], {}, "h0", 1)
        assert type(value) is float


# ===========================================================================
# DETERMINISM TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.determinism
class TestChainLengthDeterminism:
    """Determinism tests for compute_chain_success."""

    def test_same_input_same_output_100_runs(self):
        """Same inputs produce same outputs over 100 runs."""
        graph = {"h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}]
        
        results = [
            compute_chain_success(statements, graph, "h2", 3)
            for _ in range(100)
        ]
        assert all(r == results[0] for r in results)

    def test_statement_order_independence(self):
        """Statement order does not affect result."""
        import random
        rng = random.Random(SEED_CHAIN_LENGTH)
        
        graph = {"h1": ["h0"], "h2": ["h1"]}
        base_hashes = ["h0", "h1", "h2"]
        
        results = []
        for _ in range(50):
            shuffled = base_hashes.copy()
            rng.shuffle(shuffled)
            statements = [{"hash": h} for h in shuffled]
            results.append(compute_chain_success(statements, graph, "h2", 3))
        
        assert all(r == results[0] for r in results)

    def test_deterministic_with_random_graph(self, gen_chain_length: DeterministicGenerator):
        """Deterministic with randomly generated graph."""
        gen = gen_chain_length
        
        # Generate once
        gen.reset()
        graph1, hashes1 = gen.dependency_graph_random(20, 0.3)
        statements1 = [{"hash": h} for h in hashes1]
        result1 = compute_chain_success(statements1, graph1, hashes1[-1], 3)
        
        # Regenerate with same seed
        gen.reset()
        graph2, hashes2 = gen.dependency_graph_random(20, 0.3)
        statements2 = [{"hash": h} for h in hashes2]
        result2 = compute_chain_success(statements2, graph2, hashes2[-1], 3)
        
        assert graph1 == graph2
        assert result1 == result2


# ===========================================================================
# CROSS-SLICE PARAMETER TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.cross_slice
class TestChainLengthCrossSlice:
    """Cross-slice parameter smoke tests for compute_chain_success."""

    @pytest.mark.parametrize("slice_id", list(SLICE_PARAMS.keys()))
    def test_slice_with_varying_depths(self, slice_id: str, gen_chain_length: DeterministicGenerator):
        """Test chain success across slices with varying depths."""
        gen = gen_chain_length
        gen.reset()
        
        params = SLICE_PARAMS[slice_id]
        # Use a depth proportional to min_samples (scaled down)
        depth = min(params["min_samples"] // 10, 50)
        
        graph, hashes = gen.dependency_graph_linear(depth)
        statements = [{"hash": h} for h in hashes]
        target = hashes[-1]
        
        success, value = compute_chain_success(statements, graph, target, min_chain_length=depth)
        assert success is True
        assert value == float(depth)


# ===========================================================================
# PARTIAL VERIFICATION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
class TestChainLengthPartialVerification:
    """Tests for partial chain verification scenarios."""

    def test_gap_in_chain(self):
        """Gap in middle of chain stops traversal."""
        graph = {"h1": ["h0"], "h2": ["h1"], "h3": ["h2"]}
        statements = [{"hash": "h0"}, {"hash": "h2"}, {"hash": "h3"}]  # h1 missing
        
        success, value = compute_chain_success(statements, graph, "h3", min_chain_length=4)
        assert success is False
        # h3 -> h2 stops at h1 (unverified), so chain = h3 + h2 = 2
        assert value == 2.0

    def test_only_target_verified(self):
        """Only target verified, dependencies unverified."""
        graph = {"h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h2"}]  # Only h2 verified
        
        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=1)
        assert success is True
        assert value == 1.0

    def test_all_deps_verified_target_not(self):
        """All dependencies verified but not target."""
        graph = {"h1": ["h0"], "h2": ["h1"]}
        statements = [{"hash": "h0"}, {"hash": "h1"}]  # h2 not verified
        
        success, value = compute_chain_success(statements, graph, "h2", min_chain_length=1)
        assert success is False
        assert value == 0.0


# ===========================================================================
# REPLAY EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.replay
class TestChainLengthReplay:
    """Replay equivalence tests for compute_chain_success."""

    def test_replay_linear_chain(self, gen_chain_length: DeterministicGenerator):
        """Linear chain replays identically."""
        gen = gen_chain_length
        
        # First run
        gen.reset()
        graph1, hashes1 = gen.dependency_graph_linear(20)
        statements1 = [{"hash": h} for h in hashes1]
        result1 = compute_chain_success(statements1, graph1, hashes1[-1], 15)
        
        # Replay
        gen.reset()
        graph2, hashes2 = gen.dependency_graph_linear(20)
        statements2 = [{"hash": h} for h in hashes2]
        result2 = compute_chain_success(statements2, graph2, hashes2[-1], 15)
        
        assert result1 == result2

    def test_replay_diamond_chain(self, gen_chain_length: DeterministicGenerator):
        """Diamond chain replays identically."""
        gen = gen_chain_length
        
        # First run
        gen.reset()
        graph1, hashes1 = gen.dependency_graph_diamond()
        statements1 = [{"hash": h} for h in hashes1]
        result1 = compute_chain_success(statements1, graph1, hashes1[-1], 3)
        
        # Replay
        gen.reset()
        graph2, hashes2 = gen.dependency_graph_diamond()
        statements2 = [{"hash": h} for h in hashes2]
        result2 = compute_chain_success(statements2, graph2, hashes2[-1], 3)
        
        assert result1 == result2


# ===========================================================================
# SCHEMA VALIDATION TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.schema
class TestChainLengthSchema:
    """Schema validation tests for compute_chain_success."""

    def test_statement_with_extra_fields(self):
        """Statements with extra fields work correctly."""
        statements = [
            {"hash": "h0", "extra": "ignored"},
            {"hash": "h1", "meta": {"nested": True}},
        ]
        graph = {"h1": ["h0"]}
        success, value = compute_chain_success(statements, graph, "h1", 2)
        assert success is True
        assert value == 2.0

    def test_statement_missing_hash_raises(self):
        """Statement missing hash key raises KeyError."""
        statements = [{"other": "h0"}]
        with pytest.raises(KeyError):
            compute_chain_success(statements, {}, "h0", 1)

    def test_graph_with_empty_deps_list(self):
        """Graph entry with empty dependencies list."""
        graph = {"h0": []}  # Empty deps
        statements = [{"hash": "h0"}]
        success, value = compute_chain_success(statements, graph, "h0", 1)
        assert success is True
        assert value == 1.0


# ===========================================================================
# COMPLEX GRAPH TESTS
# ===========================================================================

@pytest.mark.phase2_metrics
@pytest.mark.large_scale
class TestChainLengthComplexGraphs:
    """Tests with complex graph structures."""

    def test_wide_graph(self, gen_chain_length: DeterministicGenerator):
        """Wide graph with many siblings."""
        # h0 is root
        # h1...h10 all depend on h0
        # h11 depends on all of h1...h10
        graph = {f"h{i}": ["h0"] for i in range(1, 11)}
        graph["h11"] = [f"h{i}" for i in range(1, 11)]
        
        statements = [{"hash": f"h{i}"} for i in range(12)]
        success, value = compute_chain_success(statements, graph, "h11", 3)
        assert success is True
        assert value == 3.0  # h11 -> any h1-h10 -> h0

    def test_random_dag_100_nodes(self, gen_chain_length: DeterministicGenerator):
        """Random DAG with 100 nodes."""
        gen = gen_chain_length
        gen.reset()
        
        graph, hashes = gen.dependency_graph_random(100, 0.1)
        statements = [{"hash": h} for h in hashes]
        
        # Test with reasonable threshold
        success, value = compute_chain_success(statements, graph, hashes[-1], 3)
        assert_tuple_bool_float((success, value))

    def test_random_dag_500_nodes(self, gen_chain_length: DeterministicGenerator):
        """Random DAG with 500 nodes."""
        gen = gen_chain_length
        gen.reset()
        
        graph, hashes = gen.dependency_graph_random(500, 0.05)
        statements = [{"hash": h} for h in hashes]
        
        success, value = compute_chain_success(statements, graph, hashes[-1], 5)
        assert_tuple_bool_float((success, value))

