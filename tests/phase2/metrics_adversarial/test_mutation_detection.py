# tests/phase2/metrics_adversarial/test_mutation_detection.py
"""
Mutation Detection Harness Tests

Uses shadow implementations to verify production metrics:
- Differential testing against alternate algorithms
- Parameter mutation (±1, ±2, permutations)
- Mismatch detection between implementations

NO METRIC INTERPRETATION: These tests verify implementation correctness only.
"""

import pytest
from typing import Dict, List, Set, Any, Tuple

from backend.substrate.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

from .conftest import (
    ShadowMetrics,
    MutationOperator,
    BatchGenerator,
    SEED_MUTATION,
    SEED_BATCH,
)


# ===========================================================================
# SHADOW VS PRODUCTION DIFFERENTIAL TESTS
# ===========================================================================

@pytest.mark.mutation
class TestShadowDifferential:
    """Tests comparing production metrics against shadow implementations."""

    def test_goal_hit_shadow_equivalence_basic(self, shadow_metrics: ShadowMetrics):
        """goal_hit matches shadow on basic inputs."""
        statements = [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        targets = {"h1", "h3"}
        min_hits = 2
        
        prod_result = compute_goal_hit(statements, targets, min_hits)
        shadow_result = shadow_metrics.compute_goal_hit_shadow(statements, targets, min_hits)
        
        assert prod_result == shadow_result

    def test_goal_hit_shadow_equivalence_empty(self, shadow_metrics: ShadowMetrics):
        """goal_hit matches shadow on empty inputs."""
        prod_result = compute_goal_hit([], set(), 0)
        shadow_result = shadow_metrics.compute_goal_hit_shadow([], set(), 0)
        
        assert prod_result == shadow_result

    def test_goal_hit_shadow_equivalence_duplicates(self, shadow_metrics: ShadowMetrics):
        """goal_hit matches shadow with duplicate hashes."""
        statements = [{"hash": "h1"}, {"hash": "h1"}, {"hash": "h1"}]
        targets = {"h1"}
        min_hits = 1
        
        prod_result = compute_goal_hit(statements, targets, min_hits)
        shadow_result = shadow_metrics.compute_goal_hit_shadow(statements, targets, min_hits)
        
        assert prod_result == shadow_result

    def test_sparse_success_shadow_equivalence(self, shadow_metrics: ShadowMetrics):
        """sparse_success matches shadow across value range."""
        test_cases = [
            (0, 0, 0),
            (10, 20, 5),
            (100, 100, 100),
            (50, 200, 51),
            (1000000, 2000000, 999999),
        ]
        
        for verified, attempted, min_ver in test_cases:
            prod_result = compute_sparse_success(verified, attempted, min_ver)
            shadow_result = shadow_metrics.compute_sparse_success_shadow(verified, attempted, min_ver)
            
            assert prod_result == shadow_result, f"Mismatch at ({verified}, {attempted}, {min_ver})"

    def test_multi_goal_shadow_equivalence(self, shadow_metrics: ShadowMetrics):
        """multi_goal matches shadow on various inputs."""
        test_cases = [
            (set(), set()),
            ({"h1"}, {"h1"}),
            ({"h1", "h2"}, {"h1"}),
            ({"h1"}, {"h1", "h2"}),
            ({f"h{i}" for i in range(100)}, {f"h{i}" for i in range(50)}),
        ]
        
        for verified, required in test_cases:
            prod_result = compute_multi_goal_success(verified, required)
            shadow_result = shadow_metrics.compute_multi_goal_success_shadow(verified, required)
            
            assert prod_result == shadow_result

    def test_chain_success_shadow_linear(self, shadow_metrics: ShadowMetrics):
        """chain_success matches shadow on linear chain."""
        # h0 <- h1 <- h2
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}]
        graph = {"h1": ["h0"], "h2": ["h1"]}
        
        prod_result = compute_chain_success(statements, graph, "h2", 3)
        shadow_result = shadow_metrics.compute_chain_success_shadow(statements, graph, "h2", 3)
        
        assert prod_result == shadow_result

    def test_chain_success_shadow_diamond(self, shadow_metrics: ShadowMetrics):
        """chain_success matches shadow on diamond graph."""
        # h0 <- h1, h0 <- h2, h1/h2 <- h3
        statements = [{"hash": "h0"}, {"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}]
        graph = {"h1": ["h0"], "h2": ["h0"], "h3": ["h1", "h2"]}
        
        prod_result = compute_chain_success(statements, graph, "h3", 3)
        shadow_result = shadow_metrics.compute_chain_success_shadow(statements, graph, "h3", 3)
        
        assert prod_result == shadow_result

    def test_chain_success_shadow_empty(self, shadow_metrics: ShadowMetrics):
        """chain_success matches shadow on empty inputs."""
        prod_result = compute_chain_success([], {}, "h0", 0)
        shadow_result = shadow_metrics.compute_chain_success_shadow([], {}, "h0", 0)
        
        assert prod_result == shadow_result


# ===========================================================================
# PARAMETER MUTATION TESTS
# ===========================================================================

@pytest.mark.mutation
class TestParameterMutation:
    """Tests using mutated parameters to verify sensitivity."""

    def test_goal_hit_min_threshold_mutation(self, mutation_operator: MutationOperator):
        """goal_hit detects ±1/±2 threshold changes."""
        statements = [{"hash": f"h{i}"} for i in range(10)]
        targets = {f"h{i}" for i in range(5)}
        base_min = 3
        
        base_result = compute_goal_hit(statements, targets, base_min)
        
        mutations = mutation_operator.mutate_int(base_min)
        different_results = 0
        
        for mutated_min in mutations:
            mutated_result = compute_goal_hit(statements, targets, mutated_min)
            if mutated_result != base_result:
                different_results += 1
        
        # At least some mutations should produce different results
        assert different_results > 0, "Mutations should affect results"

    def test_sparse_success_threshold_sensitivity(self, mutation_operator: MutationOperator):
        """sparse_success is sensitive to threshold mutations."""
        verified = 50
        attempted = 100
        base_min = 50
        
        base_success, base_value = compute_sparse_success(verified, attempted, base_min)
        
        # +1 should fail (50 < 51)
        plus_one_success, _ = compute_sparse_success(verified, attempted, base_min + 1)
        # -1 should succeed (50 > 49)
        minus_one_success, _ = compute_sparse_success(verified, attempted, base_min - 1)
        
        assert base_success is True
        assert plus_one_success is False
        assert minus_one_success is True

    def test_goal_hit_target_set_mutation(self, mutation_operator: MutationOperator):
        """goal_hit detects target set mutations."""
        statements = [{"hash": f"h{i}"} for i in range(10)]
        targets = {"h2", "h5", "h8"}
        min_hits = 3
        
        base_result = compute_goal_hit(statements, targets, min_hits)
        
        # Mutate targets
        mutations = mutation_operator.mutate_set(targets)
        different_results = 0
        
        for mutated_targets in mutations:
            mutated_result = compute_goal_hit(statements, mutated_targets, min_hits)
            if mutated_result != base_result:
                different_results += 1
        
        # Removing or adding targets should change results
        assert different_results > 0

    def test_multi_goal_required_set_mutation(self, mutation_operator: MutationOperator):
        """multi_goal detects required set mutations."""
        verified = {f"h{i}" for i in range(20)}
        required = {"h5", "h10", "h15"}
        
        base_success, base_value = compute_multi_goal_success(verified, required)
        
        # Remove one required goal
        for goal in required:
            mutated_required = required - {goal}
            mutated_success, mutated_value = compute_multi_goal_success(verified, mutated_required)
            # Removing a required goal that was met should still succeed
            assert mutated_success is True
            # Value should decrease by 1
            assert mutated_value == base_value - 1

    def test_chain_success_min_length_mutation(self, mutation_operator: MutationOperator):
        """chain_success detects min_length mutations at boundary."""
        # Chain of exactly 5
        statements = [{"hash": f"h{i}"} for i in range(5)]
        graph = {f"h{i}": [f"h{i-1}"] for i in range(1, 5)}
        
        # At threshold
        success_5, _ = compute_chain_success(statements, graph, "h4", 5)
        # Below threshold
        success_4, _ = compute_chain_success(statements, graph, "h4", 4)
        # Above threshold
        success_6, _ = compute_chain_success(statements, graph, "h4", 6)
        
        assert success_5 is True
        assert success_4 is True
        assert success_6 is False


# ===========================================================================
# BATCH DIFFERENTIAL TESTS
# ===========================================================================

@pytest.mark.mutation
@pytest.mark.high_volume
class TestBatchDifferential:
    """High-volume batch tests comparing production vs shadow."""

    def test_goal_hit_batch_1000(
        self,
        shadow_metrics: ShadowMetrics,
        batch_generator: BatchGenerator
    ):
        """1000-item batch comparison for goal_hit."""
        batch_generator.reset()
        batch = batch_generator.generate_goal_hit_batch(1000)
        
        mismatches = 0
        for statements, targets, min_hits in batch:
            prod = compute_goal_hit(statements, targets, min_hits)
            shadow = shadow_metrics.compute_goal_hit_shadow(statements, targets, min_hits)
            
            if prod != shadow:
                mismatches += 1
        
        assert mismatches == 0, f"{mismatches} mismatches in batch"

    def test_sparse_success_batch_1000(
        self,
        shadow_metrics: ShadowMetrics,
        batch_generator: BatchGenerator
    ):
        """1000-item batch comparison for sparse_success."""
        batch_generator.reset()
        batch = batch_generator.generate_sparse_success_batch(1000)
        
        mismatches = 0
        for verified, attempted, min_ver in batch:
            prod = compute_sparse_success(verified, attempted, min_ver)
            shadow = shadow_metrics.compute_sparse_success_shadow(verified, attempted, min_ver)
            
            if prod != shadow:
                mismatches += 1
        
        assert mismatches == 0, f"{mismatches} mismatches in batch"

    def test_multi_goal_batch_1000(
        self,
        shadow_metrics: ShadowMetrics,
        batch_generator: BatchGenerator
    ):
        """1000-item batch comparison for multi_goal."""
        batch_generator.reset()
        batch = batch_generator.generate_multi_goal_batch(1000)
        
        mismatches = 0
        for verified, required in batch:
            prod = compute_multi_goal_success(verified, required)
            shadow = shadow_metrics.compute_multi_goal_success_shadow(verified, required)
            
            if prod != shadow:
                mismatches += 1
        
        assert mismatches == 0, f"{mismatches} mismatches in batch"


# ===========================================================================
# MUTATION DETECTION SENSITIVITY TESTS
# ===========================================================================

@pytest.mark.mutation
class TestMutationSensitivity:
    """Tests verifying that metrics detect relevant input changes."""

    def test_goal_hit_single_statement_addition(self):
        """goal_hit detects adding a matching statement."""
        targets = {"h5", "h10"}
        min_hits = 2
        
        # Without h5
        statements1 = [{"hash": f"h{i}"} for i in range(10) if i != 5]
        success1, value1 = compute_goal_hit(statements1, targets, min_hits)
        
        # With h5
        statements2 = [{"hash": f"h{i}"} for i in range(11)]
        success2, value2 = compute_goal_hit(statements2, targets, min_hits)
        
        assert value2 > value1 or success2 != success1

    def test_sparse_success_boundary_detection(self):
        """sparse_success detects boundary crossings."""
        attempted = 100
        min_ver = 50
        
        # Exactly at boundary
        s_at, v_at = compute_sparse_success(50, attempted, min_ver)
        # One below
        s_below, v_below = compute_sparse_success(49, attempted, min_ver)
        # One above
        s_above, v_above = compute_sparse_success(51, attempted, min_ver)
        
        assert s_at is True
        assert s_below is False
        assert s_above is True
        assert v_below < v_at < v_above

    def test_chain_success_partial_chain_detection(self):
        """chain_success detects when chain is broken."""
        # Full chain: h0 <- h1 <- h2 <- h3
        full_statements = [{"hash": f"h{i}"} for i in range(4)]
        graph = {"h1": ["h0"], "h2": ["h1"], "h3": ["h2"]}
        
        # Remove h1 (breaks chain)
        broken_statements = [{"hash": "h0"}, {"hash": "h2"}, {"hash": "h3"}]
        
        full_success, full_len = compute_chain_success(full_statements, graph, "h3", 4)
        broken_success, broken_len = compute_chain_success(broken_statements, graph, "h3", 4)
        
        assert full_success is True
        assert full_len == 4.0
        assert broken_success is False
        assert broken_len < full_len

    def test_multi_goal_single_goal_removal(self):
        """multi_goal detects single required goal removal."""
        verified = {"h1", "h2", "h3", "h4", "h5"}
        required_full = {"h1", "h2", "h3"}
        required_partial = {"h1", "h2"}
        
        full_success, full_value = compute_multi_goal_success(verified, required_full)
        partial_success, partial_value = compute_multi_goal_success(verified, required_partial)
        
        # Both succeed since all are in verified
        assert full_success is True
        assert partial_success is True
        # But values differ
        assert full_value == 3.0
        assert partial_value == 2.0


# ===========================================================================
# NO REGRESSION UNDER MUTATION TESTS (Property-Style Boundary Tests)
# ===========================================================================

@pytest.mark.mutation
class TestNoRegressionUnderMutation:
    """
    Property-style tests verifying:
    - Stable behavior under non-boundary-preserving mutations
    - Sensitive behavior when crossing boundaries
    
    These tests systematically apply ±1, ±2 mutations at boundaries
    and assert mathematically expected behavior changes.
    """

    def test_goal_hit_boundary_regression_systematic(self, mutation_operator: MutationOperator):
        """
        goal_hit exhibits correct boundary behavior under threshold mutation.
        
        Property: For hits=N and threshold=N:
          - threshold ±0 → success=True
          - threshold +1 → success=False (N < N+1)
          - threshold +2 → success=False (N < N+2)
          - threshold -1 → success=True (N >= N-1)
          - threshold -2 → success=True (N >= N-2)
        """
        # Create scenario where exactly N targets are hit
        for n_hits in [1, 3, 5, 10]:
            statements = [{"hash": f"h{i}"} for i in range(n_hits)]
            targets = {f"h{i}" for i in range(n_hits)}
            
            # At boundary (threshold = n_hits)
            at_boundary, _ = compute_goal_hit(statements, targets, n_hits)
            assert at_boundary is True, f"At boundary n={n_hits}"
            
            # Above boundary (should fail)
            for delta in [1, 2]:
                above, _ = compute_goal_hit(statements, targets, n_hits + delta)
                assert above is False, f"Above boundary n={n_hits}, delta={delta}"
            
            # Below boundary (should succeed)
            for delta in [1, 2]:
                below_threshold = max(0, n_hits - delta)
                below, _ = compute_goal_hit(statements, targets, below_threshold)
                assert below is True, f"Below boundary n={n_hits}, delta={delta}"

    def test_sparse_success_boundary_regression_systematic(self, mutation_operator: MutationOperator):
        """
        sparse_success exhibits correct boundary behavior.
        
        Property: For verified=V and min_verified=V:
          - V >= V → True
          - V >= V+1 → False
          - V >= V-1 → True
        """
        for verified in [0, 10, 50, 100, 500]:
            attempted = verified * 2 + 10
            
            # At boundary
            at, _ = compute_sparse_success(verified, attempted, verified)
            assert at is True, f"At boundary v={verified}"
            
            # Above boundary (should fail)
            for delta in [1, 2]:
                above, _ = compute_sparse_success(verified, attempted, verified + delta)
                assert above is False, f"Above boundary v={verified}, delta={delta}"
            
            # Below boundary (should succeed)
            for delta in [1, 2]:
                below_threshold = max(0, verified - delta)
                below, _ = compute_sparse_success(verified, attempted, below_threshold)
                assert below is True, f"Below boundary v={verified}, delta={delta}"

    def test_chain_success_boundary_regression_systematic(self, mutation_operator: MutationOperator):
        """
        chain_success exhibits correct boundary behavior on chain length.
        
        Property: For chain of length L and min_length=L:
          - L >= L → True
          - L >= L+1 → False
          - L >= L-1 → True
        """
        for chain_len in [2, 4, 6, 8]:
            hashes = [f"h{i}" for i in range(chain_len)]
            statements = [{"hash": h} for h in hashes]
            graph = {hashes[i]: [hashes[i-1]] for i in range(1, chain_len)}
            target = hashes[-1]
            
            # At boundary
            at_success, at_len = compute_chain_success(statements, graph, target, chain_len)
            assert at_success is True, f"At boundary len={chain_len}"
            assert at_len == float(chain_len)
            
            # Above boundary (should fail)
            for delta in [1, 2]:
                above, _ = compute_chain_success(statements, graph, target, chain_len + delta)
                assert above is False, f"Above boundary len={chain_len}, delta={delta}"
            
            # Below boundary (should succeed)
            for delta in [1, 2]:
                below_threshold = max(0, chain_len - delta)
                below, _ = compute_chain_success(statements, graph, target, below_threshold)
                assert below is True, f"Below boundary len={chain_len}, delta={delta}"

    def test_multi_goal_boundary_regression_systematic(self, mutation_operator: MutationOperator):
        """
        multi_goal exhibits correct boundary behavior on required goals.
        
        Property: All required goals met → success=True
                  Any required goal not met → success=False
        """
        base_verified = {f"h{i}" for i in range(10)}
        
        for n_required in [1, 3, 5]:
            # All required are in verified
            required_met = {f"h{i}" for i in range(n_required)}
            success_met, value_met = compute_multi_goal_success(base_verified, required_met)
            assert success_met is True
            assert value_met == float(n_required)
            
            # Add one more required that's NOT in verified
            required_unmet = required_met | {"h999"}
            success_unmet, value_unmet = compute_multi_goal_success(base_verified, required_unmet)
            assert success_unmet is False
            assert value_unmet == float(n_required)  # Still met n_required, but not h999

    def test_goal_hit_value_monotonicity_under_mutation(self, mutation_operator: MutationOperator):
        """
        goal_hit value is monotonic with respect to matching targets.
        
        Property: Adding a matching statement cannot decrease the hit count.
        """
        targets = {"h0", "h5", "h10", "h15", "h20"}
        
        for n_statements in range(5, 25, 5):
            statements_small = [{"hash": f"h{i}"} for i in range(n_statements)]
            statements_large = [{"hash": f"h{i}"} for i in range(n_statements + 5)]
            
            _, value_small = compute_goal_hit(statements_small, targets, 0)
            _, value_large = compute_goal_hit(statements_large, targets, 0)
            
            # Larger statement set should have >= hits
            assert value_large >= value_small

    def test_sparse_success_value_identity_under_mutation(self, mutation_operator: MutationOperator):
        """
        sparse_success value always equals verified_count.
        
        Property: For any inputs, value == verified_count
        """
        test_cases = [
            (0, 0, 0),
            (50, 100, 25),
            (100, 100, 100),
            (999, 1000, 500),
        ]
        
        for verified, attempted, min_ver in test_cases:
            _, value = compute_sparse_success(verified, attempted, min_ver)
            assert value == float(verified), f"Value should equal verified={verified}"

    def test_multi_goal_value_bounded_by_required(self, mutation_operator: MutationOperator):
        """
        multi_goal value is bounded by |required|.
        
        Property: 0 <= value <= len(required)
        """
        verified = {f"h{i}" for i in range(50)}
        
        for n_required in [0, 5, 10, 20, 100]:
            required = {f"h{i}" for i in range(n_required)}
            success, value = compute_multi_goal_success(verified, required)
            
            assert 0 <= value <= n_required
            # If success, value == len(required)
            if success:
                assert value == n_required

    def test_all_metrics_deterministic_under_repeated_mutation(self, mutation_operator: MutationOperator):
        """
        All metrics are deterministic: same input → same output always.
        
        Property: f(x) == f(x) for any x, called N times.
        """
        # goal_hit
        stmts = [{"hash": "h1"}, {"hash": "h2"}]
        targets = {"h1", "h2"}
        results_gh = [compute_goal_hit(stmts, targets, 2) for _ in range(100)]
        assert all(r == results_gh[0] for r in results_gh), "goal_hit not deterministic"
        
        # sparse_success
        results_ss = [compute_sparse_success(50, 100, 25) for _ in range(100)]
        assert all(r == results_ss[0] for r in results_ss), "sparse_success not deterministic"
        
        # chain_success
        graph = {"h1": ["h0"]}
        stmts_chain = [{"hash": "h0"}, {"hash": "h1"}]
        results_cs = [compute_chain_success(stmts_chain, graph, "h1", 2) for _ in range(100)]
        assert all(r == results_cs[0] for r in results_cs), "chain_success not deterministic"
        
        # multi_goal
        verified = {"h1", "h2"}
        required = {"h1"}
        results_mg = [compute_multi_goal_success(verified, required) for _ in range(100)]
        assert all(r == results_mg[0] for r in results_mg), "multi_goal not deterministic"
