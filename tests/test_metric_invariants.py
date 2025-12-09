"""
PHASE II — NOT RUN IN PHASE I
Metric Contract Invariant Tests

These tests enforce the invariants specified in docs/METRIC_CORRECTNESS_CONTRACT.md.
Each test is tagged with its corresponding invariant ID (GOAL-*, SPARSE-*, CHAIN-*, MULTI-*).
"""

import unittest
from typing import List, Dict, Set
from experiments.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)


class TestGoalHitInvariants(unittest.TestCase):
    """
    GOAL-* invariants from METRIC_CORRECTNESS_CONTRACT.md Section 5.1.
    """

    def test_GOAL_1_hit_count_lte_target_count(self):
        """GOAL-1: value <= len(target_hashes)"""
        targets = {'t1', 't2', 't3'}
        statements = [{'hash': f'h{i}'} for i in range(10)]
        statements.extend([{'hash': 't1'}, {'hash': 't2'}, {'hash': 't3'}])

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)
        self.assertLessEqual(value, len(targets))

    def test_GOAL_2_hit_count_lte_verified_count(self):
        """GOAL-2: value <= len(verified_statements)"""
        targets = {'t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10'}
        statements = [{'hash': 't1'}, {'hash': 't2'}]

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)
        self.assertLessEqual(value, len(statements))

    def test_GOAL_3_empty_targets_zero_hits(self):
        """GOAL-3: target_hashes = {} -> value = 0.0"""
        statements = [{'hash': f'h{i}'} for i in range(100)]

        _, value = compute_goal_hit(statements, set(), min_total_verified=0)
        self.assertEqual(value, 0.0)

    def test_GOAL_4_disjoint_sets_zero_hits(self):
        """GOAL-4: verified ∩ targets = {} -> value = 0.0"""
        statements = [{'hash': f'v{i}'} for i in range(10)]
        targets = {f't{i}' for i in range(10)}

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)
        self.assertEqual(value, 0.0)

    def test_GOAL_5_success_threshold_exact(self):
        """GOAL-5: success <-> value >= min_total_verified"""
        statements = [{'hash': 't1'}, {'hash': 't2'}, {'hash': 't3'}]
        targets = {'t1', 't2', 't3'}

        # Threshold 3: value=3, should succeed
        success, value = compute_goal_hit(statements, targets, min_total_verified=3)
        self.assertEqual(value, 3.0)
        self.assertTrue(success)

        # Threshold 4: value=3, should fail
        success, value = compute_goal_hit(statements, targets, min_total_verified=4)
        self.assertEqual(value, 3.0)
        self.assertFalse(success)

        # Threshold 2: value=3, should succeed
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        self.assertEqual(value, 3.0)
        self.assertTrue(success)

    def test_GOAL_6_deterministic(self):
        """GOAL-6: compute(I) at t1 = compute(I) at t2"""
        statements = [{'hash': f'h{i}'} for i in range(50)]
        targets = {f'h{i}' for i in range(10, 20)}

        results = [
            compute_goal_hit(statements, targets, min_total_verified=5)
            for _ in range(100)
        ]
        self.assertTrue(all(r == results[0] for r in results))

    def test_GOAL_7_order_independent(self):
        """GOAL-7: compute([a,b]) = compute([b,a])"""
        targets = {'h1', 'h3', 'h5'}

        orderings = [
            [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}, {'hash': 'h4'}, {'hash': 'h5'}],
            [{'hash': 'h5'}, {'hash': 'h4'}, {'hash': 'h3'}, {'hash': 'h2'}, {'hash': 'h1'}],
            [{'hash': 'h3'}, {'hash': 'h1'}, {'hash': 'h5'}, {'hash': 'h2'}, {'hash': 'h4'}],
        ]

        results = [compute_goal_hit(o, targets, 2) for o in orderings]
        self.assertTrue(all(r == results[0] for r in results))


class TestSparseSuccessInvariants(unittest.TestCase):
    """
    SPARSE-* invariants from METRIC_CORRECTNESS_CONTRACT.md Section 5.2.
    """

    def test_SPARSE_1_value_equals_input(self):
        """SPARSE-1: value = float(verified_count)"""
        for v in [0, 1, 5, 10, 100, 1000]:
            _, value = compute_sparse_success(v, 999, min_verified=0)
            self.assertEqual(value, float(v))

    def test_SPARSE_2_success_threshold(self):
        """SPARSE-2: success <-> verified_count >= min_verified"""
        # Exactly at threshold
        success, _ = compute_sparse_success(5, 10, min_verified=5)
        self.assertTrue(success)

        # Below threshold
        success, _ = compute_sparse_success(4, 10, min_verified=5)
        self.assertFalse(success)

        # Above threshold
        success, _ = compute_sparse_success(6, 10, min_verified=5)
        self.assertTrue(success)

    def test_SPARSE_3_attempted_ignored(self):
        """SPARSE-3: compute(v, a1, m) = compute(v, a2, m) for any a1, a2"""
        verified = 7
        min_v = 5

        results = [
            compute_sparse_success(verified, attempted, min_v)
            for attempted in [0, 1, 10, 100, 1000, 999999]
        ]
        self.assertTrue(all(r == results[0] for r in results))

    def test_SPARSE_4_zero_verified(self):
        """SPARSE-4: verified_count = 0 -> value = 0.0"""
        _, value = compute_sparse_success(0, 100, min_verified=0)
        self.assertEqual(value, 0.0)

    def test_SPARSE_5_non_negative(self):
        """SPARSE-5: value >= 0.0"""
        for v in range(100):
            _, value = compute_sparse_success(v, 100, min_verified=0)
            self.assertGreaterEqual(value, 0.0)

    def test_SPARSE_6_deterministic(self):
        """SPARSE-6: compute(I) at t1 = compute(I) at t2"""
        results = [
            compute_sparse_success(42, 100, 30)
            for _ in range(100)
        ]
        self.assertTrue(all(r == results[0] for r in results))


class TestChainSuccessInvariants(unittest.TestCase):
    """
    CHAIN-* invariants from METRIC_CORRECTNESS_CONTRACT.md Section 5.3.
    """

    def test_CHAIN_1_value_lte_verified_count(self):
        """CHAIN-1: value <= len(verified_statements)"""
        # Chain of 5: h0 <- h1 <- h2 <- h3 <- h4
        graph = {f'h{i}': [f'h{i-1}'] for i in range(1, 5)}
        statements = [{'hash': f'h{i}'} for i in range(5)]

        _, value = compute_chain_success(statements, graph, 'h4', min_chain_length=1)
        self.assertLessEqual(value, len(statements))

    def test_CHAIN_2_unverified_target_zero(self):
        """CHAIN-2: target not in verified -> value = 0.0"""
        graph = {'h2': ['h1']}
        statements = [{'hash': 'h1'}]  # h2 not verified

        _, value = compute_chain_success(statements, graph, 'h2', min_chain_length=1)
        self.assertEqual(value, 0.0)

    def test_CHAIN_3_isolated_target_one(self):
        """CHAIN-3: target in verified and deps(target) = {} -> value = 1.0"""
        graph = {}  # No dependencies
        statements = [{'hash': 'isolated'}]

        _, value = compute_chain_success(statements, graph, 'isolated', min_chain_length=1)
        self.assertEqual(value, 1.0)

    def test_CHAIN_4_cycle_safety(self):
        """CHAIN-4: Graph cycles do not cause infinite recursion"""
        # Create a cycle: h1 -> h2 -> h3 -> h1
        cyclic_graph = {
            'h1': ['h3'],
            'h2': ['h1'],
            'h3': ['h2'],
        }
        statements = [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}]

        # Should complete without RecursionError
        try:
            compute_chain_success(statements, cyclic_graph, 'h1', min_chain_length=1)
        except RecursionError:
            self.fail("CHAIN-4 violated: cycle caused infinite recursion")

    def test_CHAIN_5_success_threshold(self):
        """CHAIN-5: success <-> value >= min_chain_length"""
        graph = {'h3': ['h2'], 'h2': ['h1']}
        statements = [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}]

        # Chain length = 3
        success, value = compute_chain_success(statements, graph, 'h3', min_chain_length=3)
        self.assertEqual(value, 3.0)
        self.assertTrue(success)

        success, value = compute_chain_success(statements, graph, 'h3', min_chain_length=4)
        self.assertEqual(value, 3.0)
        self.assertFalse(success)

    def test_CHAIN_6_longest_path(self):
        """CHAIN-6: value = max_chain_length(target, graph, verified)"""
        # Diamond: h4 depends on h2 and h3, both depend on h1
        # Path via h2: h4 -> h2 -> h1 (length 3)
        # Path via h3: h4 -> h3 -> h1 (length 3)
        graph = {'h4': ['h2', 'h3'], 'h2': ['h1'], 'h3': ['h1']}
        statements = [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}, {'hash': 'h4'}]

        _, value = compute_chain_success(statements, graph, 'h4', min_chain_length=1)
        self.assertEqual(value, 3.0)  # Longest path is 3

    def test_CHAIN_7_deterministic(self):
        """CHAIN-7: compute(I) at t1 = compute(I) at t2"""
        graph = {f'h{i}': [f'h{i-1}'] for i in range(1, 10)}
        statements = [{'hash': f'h{i}'} for i in range(10)]

        results = [
            compute_chain_success(statements, graph, 'h9', 5)
            for _ in range(100)
        ]
        self.assertTrue(all(r == results[0] for r in results))

    def test_CHAIN_8_order_independent(self):
        """CHAIN-8: Statement list order does not affect result"""
        graph = {'h3': ['h2'], 'h2': ['h1']}

        orderings = [
            [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}],
            [{'hash': 'h3'}, {'hash': 'h2'}, {'hash': 'h1'}],
            [{'hash': 'h2'}, {'hash': 'h1'}, {'hash': 'h3'}],
        ]

        results = [compute_chain_success(o, graph, 'h3', 3) for o in orderings]
        self.assertTrue(all(r == results[0] for r in results))


class TestMultiGoalSuccessInvariants(unittest.TestCase):
    """
    MULTI-* invariants from METRIC_CORRECTNESS_CONTRACT.md Section 5.4.
    """

    def test_MULTI_1_value_lte_required_count(self):
        """MULTI-1: value <= len(required_goal_hashes)"""
        verified = {f'h{i}' for i in range(100)}
        required = {'h1', 'h2', 'h3'}

        _, value = compute_multi_goal_success(verified, required)
        self.assertLessEqual(value, len(required))

    def test_MULTI_2_all_or_nothing_success(self):
        """MULTI-2: success <-> value = len(required_goal_hashes)"""
        required = {'g1', 'g2', 'g3'}

        # All met
        verified_all = {'g1', 'g2', 'g3', 'extra'}
        success, value = compute_multi_goal_success(verified_all, required)
        self.assertEqual(value, 3.0)
        self.assertTrue(success)

        # Partial
        verified_partial = {'g1', 'g2'}
        success, value = compute_multi_goal_success(verified_partial, required)
        self.assertEqual(value, 2.0)
        self.assertFalse(success)

        # None
        verified_none = {'other1', 'other2'}
        success, value = compute_multi_goal_success(verified_none, required)
        self.assertEqual(value, 0.0)
        self.assertFalse(success)

    def test_MULTI_3_empty_required_success(self):
        """MULTI-3: required = {} -> success = True, value = 0.0"""
        verified = {'h1', 'h2', 'h3'}
        success, value = compute_multi_goal_success(verified, set())
        self.assertTrue(success)
        self.assertEqual(value, 0.0)

    def test_MULTI_4_subset_monotonicity(self):
        """MULTI-4: verified1 ⊆ verified2 -> value1 <= value2"""
        required = {'g1', 'g2', 'g3', 'g4', 'g5'}

        v1 = {'g1', 'g2'}
        v2 = {'g1', 'g2', 'g3'}
        v3 = {'g1', 'g2', 'g3', 'g4', 'g5', 'extra'}

        _, val1 = compute_multi_goal_success(v1, required)
        _, val2 = compute_multi_goal_success(v2, required)
        _, val3 = compute_multi_goal_success(v3, required)

        self.assertLessEqual(val1, val2)
        self.assertLessEqual(val2, val3)

    def test_MULTI_5_goal_counting(self):
        """MULTI-5: value = len(verified ∩ required)"""
        verified = {'a', 'b', 'c', 'd', 'e'}
        required = {'b', 'd', 'f', 'h'}

        expected_intersection = {'b', 'd'}
        _, value = compute_multi_goal_success(verified, required)
        self.assertEqual(value, float(len(expected_intersection)))

    def test_MULTI_6_deterministic(self):
        """MULTI-6: compute(I) at t1 = compute(I) at t2"""
        verified = {f'v{i}' for i in range(50)}
        required = {f'v{i}' for i in range(10, 20)}

        results = [
            compute_multi_goal_success(verified, required)
            for _ in range(100)
        ]
        self.assertTrue(all(r == results[0] for r in results))


class TestNegativeControls(unittest.TestCase):
    """
    Negative control tests from METRIC_CORRECTNESS_CONTRACT.md Section 3.
    """

    def test_NC1_goal_hit_zero_input(self):
        """NC-1: goal_hit with empty verified_statements"""
        success, value = compute_goal_hit([], {'t1', 't2'}, min_total_verified=1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

        # With zero threshold
        success, value = compute_goal_hit([], {'t1', 't2'}, min_total_verified=0)
        self.assertTrue(success)
        self.assertEqual(value, 0.0)

    def test_NC1_sparse_zero_input(self):
        """NC-1: sparse_success with verified_count=0"""
        success, value = compute_sparse_success(0, 100, min_verified=1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

        # With zero threshold
        success, value = compute_sparse_success(0, 100, min_verified=0)
        self.assertTrue(success)
        self.assertEqual(value, 0.0)

    def test_NC1_chain_zero_input(self):
        """NC-1: chain_success with empty verified_statements"""
        success, value = compute_chain_success([], {'h2': ['h1']}, 'h2', min_chain_length=1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

    def test_NC1_multi_goal_zero_input(self):
        """NC-1: multi_goal with empty verified"""
        success, value = compute_multi_goal_success(set(), {'g1', 'g2'})
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

        # Empty required (vacuous truth)
        success, value = compute_multi_goal_success(set(), set())
        self.assertTrue(success)
        self.assertEqual(value, 0.0)

    def test_NC2_impossibility_goal_hit(self):
        """NC-2: goal_hit cannot exceed available targets"""
        statements = [{'hash': f'h{i}'} for i in range(10)]
        targets = {'h0', 'h1', 'h2'}  # Only 3 targets

        _, value = compute_goal_hit(statements, targets, min_total_verified=0)
        self.assertLessEqual(value, 3.0)

    def test_NC2_impossibility_chain(self):
        """NC-2: chain_success respects graph structure"""
        # Only 3 nodes verified, chain length of 5 is impossible
        graph = {f'h{i}': [f'h{i-1}'] for i in range(1, 10)}
        statements = [{'hash': 'h0'}, {'hash': 'h1'}, {'hash': 'h2'}]

        success, value = compute_chain_success(statements, graph, 'h2', min_chain_length=5)
        self.assertFalse(success)
        self.assertLessEqual(value, 3.0)

    def test_NC2_impossibility_multi_goal(self):
        """NC-2: multi_goal cannot exceed verified goals"""
        verified = {'g1'}
        required = {'g1', 'g2', 'g3', 'g4'}

        success, value = compute_multi_goal_success(verified, required)
        self.assertFalse(success)
        self.assertEqual(value, 1.0)

    def test_NC3_threshold_boundary_sparse(self):
        """NC-3: threshold boundary behavior for sparse_success"""
        # T-1
        success, _ = compute_sparse_success(4, 10, min_verified=5)
        self.assertFalse(success)

        # T
        success, _ = compute_sparse_success(5, 10, min_verified=5)
        self.assertTrue(success)

        # T+1
        success, _ = compute_sparse_success(6, 10, min_verified=5)
        self.assertTrue(success)

    def test_NC3_threshold_boundary_goal_hit(self):
        """NC-3: threshold boundary behavior for goal_hit"""
        statements = [{'hash': 't1'}, {'hash': 't2'}, {'hash': 't3'}]
        targets = {'t1', 't2', 't3'}

        # T-1 (value=3, threshold=4)
        success, _ = compute_goal_hit(statements, targets, min_total_verified=4)
        self.assertFalse(success)

        # T (value=3, threshold=3)
        success, _ = compute_goal_hit(statements, targets, min_total_verified=3)
        self.assertTrue(success)

        # T+1 (value=3, threshold=2)
        success, _ = compute_goal_hit(statements, targets, min_total_verified=2)
        self.assertTrue(success)


class TestMonotonicity(unittest.TestCase):
    """
    Monotonicity tests from METRIC_CORRECTNESS_CONTRACT.md Section 4.
    """

    def test_MON1_goal_hit_additive(self):
        """MON-1: Adding more verified items does not decrease value"""
        targets = {'t1', 't2', 't3', 't4', 't5'}

        v1 = [{'hash': 't1'}]
        v2 = [{'hash': 't1'}, {'hash': 't2'}]
        v3 = [{'hash': 't1'}, {'hash': 't2'}, {'hash': 't3'}]

        _, val1 = compute_goal_hit(v1, targets, 0)
        _, val2 = compute_goal_hit(v2, targets, 0)
        _, val3 = compute_goal_hit(v3, targets, 0)

        self.assertLessEqual(val1, val2)
        self.assertLessEqual(val2, val3)

    def test_MON1_sparse_additive(self):
        """MON-1: Increasing verified_count does not decrease value"""
        _, val1 = compute_sparse_success(5, 100, 0)
        _, val2 = compute_sparse_success(10, 100, 0)
        _, val3 = compute_sparse_success(15, 100, 0)

        self.assertLessEqual(val1, val2)
        self.assertLessEqual(val2, val3)

    def test_MON1_multi_goal_additive(self):
        """MON-1: Adding more verified does not decrease value"""
        required = {'g1', 'g2', 'g3', 'g4', 'g5'}

        v1 = {'g1'}
        v2 = {'g1', 'g2'}
        v3 = {'g1', 'g2', 'g3', 'g4', 'g5'}

        _, val1 = compute_multi_goal_success(v1, required)
        _, val2 = compute_multi_goal_success(v2, required)
        _, val3 = compute_multi_goal_success(v3, required)

        self.assertLessEqual(val1, val2)
        self.assertLessEqual(val2, val3)

    def test_MON2_threshold_monotonicity_sparse(self):
        """MON-2: Lower threshold means higher P[success]"""
        # With verified=5, test different thresholds
        s_t3, _ = compute_sparse_success(5, 10, min_verified=3)  # T=3
        s_t5, _ = compute_sparse_success(5, 10, min_verified=5)  # T=5
        s_t7, _ = compute_sparse_success(5, 10, min_verified=7)  # T=7

        # T=3: success, T=5: success, T=7: failure
        self.assertTrue(s_t3)
        self.assertTrue(s_t5)
        self.assertFalse(s_t7)

    def test_MON3_subset_inclusion_multi_goal(self):
        """MON-3: Success with R2 implies success with R1 where R1 ⊆ R2"""
        verified = {'g1', 'g2', 'g3', 'g4', 'g5'}
        r1 = {'g1', 'g2'}
        r2 = {'g1', 'g2', 'g3', 'g4', 'g5'}

        s1, _ = compute_multi_goal_success(verified, r1)
        s2, _ = compute_multi_goal_success(verified, r2)

        # If s2 is True (all of R2 met), then s1 must be True (all of R1 met)
        if s2:
            self.assertTrue(s1)

        # In this case both should succeed
        self.assertTrue(s1)
        self.assertTrue(s2)


class TestReturnTypeContract(unittest.TestCase):
    """
    Return type contract verification.
    All metrics must return Tuple[bool, float].
    """

    def test_goal_hit_return_type(self):
        result = compute_goal_hit([{'hash': 'h1'}], {'h1'}, 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)

    def test_sparse_success_return_type(self):
        result = compute_sparse_success(5, 10, 3)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)

    def test_chain_success_return_type(self):
        result = compute_chain_success([{'hash': 'h1'}], {}, 'h1', 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)

    def test_multi_goal_return_type(self):
        result = compute_multi_goal_success({'h1'}, {'h1'})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
