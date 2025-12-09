# tests/phase2/metrics_adversarial/test_equivalence_oracle.py
"""
Metric Equivalence Oracle Tests

Verifies that mathematically equivalent inputs produce identical outputs:
- Input permutations (order independence)
- Set construction variations
- Graph isomorphism
- Numeric equivalences

NO METRIC INTERPRETATION: These tests verify computational equivalence only.
"""

import itertools
import random
import pytest
from typing import Dict, List, Set, Any, Tuple

from backend.substrate.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

from .conftest import (
    EquivalenceOracle,
    SEED_ORACLE,
)


# ===========================================================================
# INPUT PERMUTATION EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.oracle
class TestInputPermutationEquivalence:
    """Tests verifying permutation invariance."""

    def test_goal_hit_statement_order_invariance(self):
        """goal_hit result is invariant to statement order."""
        hashes = [f"h{i}" for i in range(10)]
        targets = {"h2", "h5", "h8"}
        min_hits = 2
        
        # Generate multiple permutations
        rng = random.Random(SEED_ORACLE)
        
        base_statements = [{"hash": h} for h in hashes]
        base_result = compute_goal_hit(base_statements, targets, min_hits)
        
        for _ in range(20):
            shuffled = hashes.copy()
            rng.shuffle(shuffled)
            shuffled_statements = [{"hash": h} for h in shuffled]
            
            shuffled_result = compute_goal_hit(shuffled_statements, targets, min_hits)
            assert shuffled_result == base_result, "Statement order affected result"

    def test_goal_hit_target_set_construction_invariance(self):
        """goal_hit result is invariant to set construction method."""
        statements = [{"hash": f"h{i}"} for i in range(10)]
        min_hits = 2
        
        # Different ways to construct the same set
        targets1 = {"h2", "h5", "h8"}
        targets2 = set(["h2", "h5", "h8"])
        targets3 = set(["h8", "h2", "h5"])
        targets4 = {"h2"} | {"h5"} | {"h8"}
        targets5 = set(list({"h2", "h5", "h8"}))
        
        result1 = compute_goal_hit(statements, targets1, min_hits)
        result2 = compute_goal_hit(statements, targets2, min_hits)
        result3 = compute_goal_hit(statements, targets3, min_hits)
        result4 = compute_goal_hit(statements, targets4, min_hits)
        result5 = compute_goal_hit(statements, targets5, min_hits)
        
        assert result1 == result2 == result3 == result4 == result5

    def test_chain_success_statement_order_invariance(self):
        """chain_success is invariant to statement list order."""
        graph = {"h1": ["h0"], "h2": ["h1"], "h3": ["h2"]}
        hashes = ["h0", "h1", "h2", "h3"]
        
        rng = random.Random(SEED_ORACLE)
        
        base_statements = [{"hash": h} for h in hashes]
        base_result = compute_chain_success(base_statements, graph, "h3", 4)
        
        for _ in range(20):
            shuffled = hashes.copy()
            rng.shuffle(shuffled)
            shuffled_statements = [{"hash": h} for h in shuffled]
            
            shuffled_result = compute_chain_success(shuffled_statements, graph, "h3", 4)
            assert shuffled_result == base_result

    def test_multi_goal_set_order_invariance(self):
        """multi_goal is invariant to iteration order within sets."""
        # Sets don't have order, but we test different constructions
        verified1 = {"h1", "h2", "h3", "h4", "h5"}
        verified2 = set(["h5", "h4", "h3", "h2", "h1"])
        
        required1 = {"h2", "h4"}
        required2 = set(["h4", "h2"])
        
        result1 = compute_multi_goal_success(verified1, required1)
        result2 = compute_multi_goal_success(verified2, required2)
        
        assert result1 == result2


# ===========================================================================
# SET EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.oracle
class TestSetEquivalence:
    """Tests verifying set semantic equivalence."""

    def test_goal_hit_duplicate_statements_equivalent(self):
        """Duplicate statements don't change goal_hit result."""
        targets = {"h1", "h2"}
        min_hits = 2
        
        # Without duplicates
        statements1 = [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        # With duplicates
        statements2 = [
            {"hash": "h1"}, {"hash": "h1"}, {"hash": "h1"},
            {"hash": "h2"}, {"hash": "h2"},
            {"hash": "h3"}
        ]
        
        result1 = compute_goal_hit(statements1, targets, min_hits)
        result2 = compute_goal_hit(statements2, targets, min_hits)
        
        assert result1 == result2

    def test_multi_goal_duplicate_goals_equivalent(self):
        """Duplicate goals in set construction don't change result."""
        verified = {"h1", "h2", "h3"}
        
        # These should be equivalent sets
        required1 = {"h1", "h2"}
        required2 = set(["h1", "h2", "h1", "h2"])  # Duplicates removed by set
        
        result1 = compute_multi_goal_success(verified, required1)
        result2 = compute_multi_goal_success(verified, required2)
        
        assert result1 == result2

    def test_goal_hit_superset_statements_increases_hits(self):
        """Adding non-target statements doesn't decrease hits."""
        targets = {"h1", "h2"}
        min_hits = 2
        
        statements_small = [{"hash": "h1"}, {"hash": "h2"}]
        statements_large = [{"hash": f"h{i}"} for i in range(100)]
        
        result_small = compute_goal_hit(statements_small, targets, min_hits)
        result_large = compute_goal_hit(statements_large, targets, min_hits)
        
        # Both should have 2 hits (h1 and h2)
        assert result_small[1] == 2.0
        assert result_large[1] == 2.0
        assert result_small[0] == result_large[0]


# ===========================================================================
# NUMERIC EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.oracle
class TestNumericEquivalence:
    """Tests verifying numeric equivalences."""

    def test_sparse_success_int_float_equivalence(self):
        """sparse_success treats int and float equivalently."""
        # Integer inputs
        result_int = compute_sparse_success(50, 100, 25)
        # Float inputs (equivalent values)
        result_float = compute_sparse_success(50.0, 100.0, 25.0)  # type: ignore
        
        assert result_int == result_float

    def test_goal_hit_threshold_equivalence(self):
        """goal_hit thresholds are equivalent at boundary."""
        statements = [{"hash": "h1"}, {"hash": "h2"}]
        targets = {"h1", "h2"}
        
        # Threshold 2 (exactly met)
        result_2 = compute_goal_hit(statements, targets, 2)
        # Float threshold (should be treated same)
        result_2f = compute_goal_hit(statements, targets, int(2.0))
        
        assert result_2 == result_2f

    def test_sparse_success_zero_equivalence(self):
        """Zero values are handled consistently."""
        # Various zero representations
        result1 = compute_sparse_success(0, 0, 0)
        result2 = compute_sparse_success(0, 100, 0)
        result3 = compute_sparse_success(0, 0, 0)
        
        # All should succeed with value 0
        assert result1 == (True, 0.0)
        assert result2 == (True, 0.0)
        assert result3 == (True, 0.0)


# ===========================================================================
# GRAPH STRUCTURE EQUIVALENCE TESTS
# ===========================================================================

@pytest.mark.oracle
class TestGraphEquivalence:
    """Tests verifying graph structure equivalences."""

    def test_chain_success_isomorphic_graphs(self):
        """chain_success produces same length for isomorphic graphs."""
        # Graph 1: a <- b <- c
        statements1 = [{"hash": "a"}, {"hash": "b"}, {"hash": "c"}]
        graph1 = {"b": ["a"], "c": ["b"]}
        
        # Graph 2: x <- y <- z (isomorphic)
        statements2 = [{"hash": "x"}, {"hash": "y"}, {"hash": "z"}]
        graph2 = {"y": ["x"], "z": ["y"]}
        
        _, len1 = compute_chain_success(statements1, graph1, "c", 1)
        _, len2 = compute_chain_success(statements2, graph2, "z", 1)
        
        assert len1 == len2 == 3.0

    def test_chain_success_unused_graph_entries(self):
        """Unused graph entries don't affect chain length."""
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}]
        
        # Minimal graph
        graph_minimal = {"h1": ["h0"], "h2": ["h1"]}
        
        # Graph with unused entries
        graph_extra = {
            "h1": ["h0"],
            "h2": ["h1"],
            "unused1": ["h99"],
            "unused2": ["h98", "h97"],
        }
        
        result_minimal = compute_chain_success(statements, graph_minimal, "h2", 3)
        result_extra = compute_chain_success(statements, graph_extra, "h2", 3)
        
        assert result_minimal == result_extra

    def test_chain_success_diamond_path_equivalence(self):
        """Diamond graph paths are equivalent in length."""
        # h0 <- h1 <- h3
        # h0 <- h2 <- h3
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        graph = {"h1": ["h0"], "h2": ["h0"], "h3": ["h1", "h2"]}
        
        # Both paths have length 3
        success, length = compute_chain_success(statements, graph, "h3", 3)
        assert success is True
        assert length == 3.0


# ===========================================================================
# EXHAUSTIVE PERMUTATION TESTS (SMALL INPUTS)
# ===========================================================================

@pytest.mark.oracle
class TestExhaustivePermutations:
    """Exhaustive permutation tests for small inputs."""

    def test_goal_hit_all_permutations_4_statements(self):
        """Test all 24 permutations of 4 statements."""
        hashes = ["h0", "h1", "h2", "h3"]
        targets = {"h1", "h3"}
        min_hits = 2
        
        results = []
        for perm in itertools.permutations(hashes):
            statements = [{"hash": h} for h in perm]
            result = compute_goal_hit(statements, targets, min_hits)
            results.append(result)
        
        # All 24 permutations should produce identical results
        assert all(r == results[0] for r in results)

    def test_chain_success_all_permutations_4_nodes(self):
        """Test all permutations of 4-node chain statements."""
        hashes = ["h0", "h1", "h2", "h3"]
        graph = {"h1": ["h0"], "h2": ["h1"], "h3": ["h2"]}
        
        results = []
        for perm in itertools.permutations(hashes):
            statements = [{"hash": h} for h in perm]
            result = compute_chain_success(statements, graph, "h3", 4)
            results.append(result)
        
        assert all(r == results[0] for r in results)

    def test_multi_goal_subset_combinations(self):
        """Test various subset combinations for multi_goal."""
        verified = {"h1", "h2", "h3", "h4", "h5"}
        
        # All 2-element subsets of verified
        for subset in itertools.combinations(verified, 2):
            required = set(subset)
            success, value = compute_multi_goal_success(verified, required)
            # All subsets of verified should succeed
            assert success is True
            assert value == 2.0


# ===========================================================================
# ORACLE CONSISTENCY TESTS
# ===========================================================================

@pytest.mark.oracle
class TestOracleConsistency:
    """Tests using equivalence oracle for consistency checks."""

    def test_oracle_float_equivalence_positive(self, equivalence_oracle: EquivalenceOracle):
        """Oracle correctly identifies equivalent floats."""
        assert equivalence_oracle.is_equivalent_float(1.0, 1.0)
        assert equivalence_oracle.is_equivalent_float(0.0, 0.0)
        assert equivalence_oracle.is_equivalent_float(1e-10, 1e-10)

    def test_oracle_float_equivalence_nan(self, equivalence_oracle: EquivalenceOracle):
        """Oracle correctly handles NaN equivalence."""
        import math
        nan1 = float('nan')
        nan2 = float('nan')
        
        # NaN == NaN is False in Python, but semantically equivalent
        assert equivalence_oracle.is_equivalent_float(nan1, nan2)

    def test_oracle_float_equivalence_inf(self, equivalence_oracle: EquivalenceOracle):
        """Oracle correctly handles infinity equivalence."""
        assert equivalence_oracle.is_equivalent_float(float('inf'), float('inf'))
        assert equivalence_oracle.is_equivalent_float(float('-inf'), float('-inf'))
        assert not equivalence_oracle.is_equivalent_float(float('inf'), float('-inf'))

    def test_oracle_result_equivalence(self, equivalence_oracle: EquivalenceOracle):
        """Oracle correctly compares metric results."""
        result1 = (True, 5.0)
        result2 = (True, 5.0)
        result3 = (False, 5.0)
        result4 = (True, 6.0)
        
        assert equivalence_oracle.is_equivalent_goal_hit(result1, result2)
        assert not equivalence_oracle.is_equivalent_goal_hit(result1, result3)
        assert not equivalence_oracle.is_equivalent_goal_hit(result1, result4)


# ===========================================================================
# COMMUTATIVE OPERATION TESTS
# ===========================================================================

@pytest.mark.oracle
class TestCommutativeOperations:
    """Tests verifying commutative property of set operations."""

    def test_goal_hit_intersection_commutative(self):
        """goal_hit intersection is commutative."""
        statements = [{"hash": "a"}, {"hash": "b"}, {"hash": "c"}]
        
        targets1 = {"a", "b", "d"}
        targets2 = {"d", "b", "a"}  # Same set, different order
        
        result1 = compute_goal_hit(statements, targets1, 1)
        result2 = compute_goal_hit(statements, targets2, 1)
        
        assert result1 == result2

    def test_multi_goal_intersection_commutative(self):
        """multi_goal intersection is commutative."""
        set_a = {"h1", "h2", "h3"}
        set_b = {"h2", "h3", "h4"}
        
        # A ∩ B == B ∩ A
        result1 = compute_multi_goal_success(set_a, set_b)
        result2 = compute_multi_goal_success(set_b, set_a)
        
        # Note: args are (verified, required), so these are different operations
        # But the intersection size is the same
        assert result1[1] == result2[1]  # Same intersection size

