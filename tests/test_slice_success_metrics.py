"""
PHASE II — NOT USED IN PHASE I
Unit tests for slice_success_metrics.
"""

import unittest
from experiments.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)

class TestSliceSuccessMetrics(unittest.TestCase):

    def test_compute_goal_hit(self):
        statements = [{'hash': f'h{i}'} for i in range(5)]
        targets = {'h1', 'h3', 'h8'}

        # Case 1: Hit threshold
        success, value = compute_goal_hit(statements, targets, min_total_verified=2)
        self.assertTrue(success)
        self.assertEqual(value, 2.0)

        # Case 2: Miss threshold
        success, value = compute_goal_hit(statements, targets, min_total_verified=3)
        self.assertFalse(success)
        self.assertEqual(value, 2.0)

        # Case 3: No hits
        success, value = compute_goal_hit(statements, {'h9'}, min_total_verified=1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

        # Case 4: Empty statements
        success, value = compute_goal_hit([], targets, min_total_verified=1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)
        
        # Case 5: Zero threshold
        success, value = compute_goal_hit(statements, targets, min_total_verified=0)
        self.assertTrue(success)
        self.assertEqual(value, 2.0)

    def test_compute_sparse_success(self):
        # Case 1: Met threshold
        success, value = compute_sparse_success(verified_count=5, attempted_count=10, min_verified=5)
        self.assertTrue(success)
        self.assertEqual(value, 5.0)

        # Case 2: Above threshold
        success, value = compute_sparse_success(verified_count=6, attempted_count=10, min_verified=5)
        self.assertTrue(success)
        self.assertEqual(value, 6.0)

        # Case 3: Below threshold
        success, value = compute_sparse_success(verified_count=4, attempted_count=10, min_verified=5)
        self.assertFalse(success)
        self.assertEqual(value, 4.0)
        
        # Case 4: Zero verified
        success, value = compute_sparse_success(verified_count=0, attempted_count=10, min_verified=1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

    def test_compute_chain_success(self):
        # h1 <- h2 <- h3
        # h4 <- h5
        # h6 (isolated)
        dep_graph = {
            'h3': ['h2'],
            'h2': ['h1'],
            'h5': ['h4'],
        }
        
        # Case 1: Full chain verified
        statements = [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}]
        success, value = compute_chain_success(statements, dep_graph, 'h3', 3)
        self.assertTrue(success)
        self.assertEqual(value, 3.0)

        # Case 2: Partial chain verified, meets lower threshold
        statements = [{'hash': 'h1'}, {'hash': 'h2'}]
        success, value = compute_chain_success(statements, dep_graph, 'h3', 2)
        self.assertFalse(success, "Target h3 is not verified")
        self.assertEqual(value, 0.0, "Chain length should be 0 if target is not verified")

        # Case 2b: Target verified, but chain is too short
        statements = [{'hash': 'h2'}, {'hash': 'h3'}]
        success, value = compute_chain_success(statements, dep_graph, 'h3', 3)
        self.assertFalse(success, "Chain should be too short")
        self.assertEqual(value, 2.0, "Chain should be h3->h2")

        # Case 3: Target not in graph
        statements = [{'hash': 'h1'}]
        success, value = compute_chain_success(statements, dep_graph, 'h99', 1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

        # Case 4: Disconnected dependency
        # h1 verified, h3 verified, but h2 is not.
        statements = [{'hash': 'h1'}, {'hash': 'h3'}]
        success, value = compute_chain_success(statements, dep_graph, 'h3', 2)
        self.assertFalse(success)
        self.assertEqual(value, 1.0, "Chain should be just h3")
        
        # Case 5: Diamond dependency, longest path taken
        # h1 <- h2 <- h4
        # h1 <- h3 <- h4
        diamond_graph = {'h4': ['h2', 'h3'], 'h2':['h1'], 'h3':['h1']}
        statements = [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h4'}] # h3 is missing
        success, value = compute_chain_success(statements, diamond_graph, 'h4', 3)
        self.assertTrue(success, "Longest path is h4->h2->h1")
        self.assertEqual(value, 3.0)

        # Case 6: Target is a leaf with no dependencies in graph
        statements = [{'hash': 'h6'}]
        success, value = compute_chain_success(statements, dep_graph, 'h6', 1)
        self.assertTrue(success)
        self.assertEqual(value, 1.0)

    def test_compute_multi_goal_success(self):
        verified = {'h1', 'h2', 'h3'}
        
        # Case 1: All required goals met
        required = {'h1', 'h3'}
        success, value = compute_multi_goal_success(verified, required)
        self.assertTrue(success)
        self.assertEqual(value, 2.0)
        
        # Case 2: Some required goals missing
        required = {'h1', 'h4'}
        success, value = compute_multi_goal_success(verified, required)
        self.assertFalse(success)
        self.assertEqual(value, 1.0)
        
        # Case 3: No required goals met
        required = {'h4', 'h5'}
        success, value = compute_multi_goal_success(verified, required)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)
        
        # Case 4: Empty required goals (vacuously true)
        required = set()
        success, value = compute_multi_goal_success(verified, required)
        self.assertTrue(success)
        self.assertEqual(value, 0.0)

        # Case 5: Empty verified, non-empty required
        required = {'h1'}
        success, value = compute_multi_goal_success(set(), required)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

class TestDeterminism(unittest.TestCase):
    """
    PHASE II — NOT USED IN PHASE I
    Tests for determinism guarantees required by the RFL protocol.
    """

    def test_goal_hit_determinism(self):
        """Same inputs must produce identical outputs across multiple calls."""
        statements = [{'hash': f'h{i}'} for i in range(10)]
        targets = {'h2', 'h5', 'h8'}

        results = [
            compute_goal_hit(statements, targets, min_total_verified=2)
            for _ in range(100)
        ]

        # All results must be identical
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0], (True, 3.0))

    def test_sparse_success_determinism(self):
        """Sparse success must be deterministic."""
        results = [
            compute_sparse_success(verified_count=7, attempted_count=20, min_verified=5)
            for _ in range(100)
        ]

        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0], (True, 7.0))

    def test_chain_success_determinism(self):
        """Chain success must be deterministic even with complex graphs."""
        # More complex dependency graph
        dep_graph = {
            'h10': ['h7', 'h8', 'h9'],
            'h7': ['h4', 'h5'],
            'h8': ['h5', 'h6'],
            'h9': ['h6'],
            'h4': ['h1', 'h2'],
            'h5': ['h2', 'h3'],
            'h6': ['h3'],
        }
        statements = [{'hash': f'h{i}'} for i in range(1, 11)]

        results = [
            compute_chain_success(statements, dep_graph, 'h10', 4)
            for _ in range(100)
        ]

        self.assertTrue(all(r == results[0] for r in results))
        # Longest path: h10 -> h7 -> h4 -> h1 = 4
        self.assertEqual(results[0], (True, 4.0))

    def test_multi_goal_determinism(self):
        """Multi-goal success must be deterministic."""
        verified = {f'h{i}' for i in range(1, 20)}
        required = {'h3', 'h7', 'h11', 'h15'}

        results = [
            compute_multi_goal_success(verified, required)
            for _ in range(100)
        ]

        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0], (True, 4.0))

    def test_order_independence_goal_hit(self):
        """Goal hit should not depend on statement order."""
        targets = {'h1', 'h3', 'h5'}

        # Different orderings of the same statements
        orderings = [
            [{'hash': 'h0'}, {'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}],
            [{'hash': 'h3'}, {'hash': 'h2'}, {'hash': 'h1'}, {'hash': 'h0'}],
            [{'hash': 'h1'}, {'hash': 'h3'}, {'hash': 'h0'}, {'hash': 'h2'}],
        ]

        results = [
            compute_goal_hit(statements, targets, min_total_verified=2)
            for statements in orderings
        ]

        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0], (True, 2.0))

    def test_order_independence_chain(self):
        """Chain success should not depend on statement list order."""
        dep_graph = {'h3': ['h2'], 'h2': ['h1']}

        orderings = [
            [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}],
            [{'hash': 'h3'}, {'hash': 'h1'}, {'hash': 'h2'}],
            [{'hash': 'h2'}, {'hash': 'h3'}, {'hash': 'h1'}],
        ]

        results = [
            compute_chain_success(statements, dep_graph, 'h3', 3)
            for statements in orderings
        ]

        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(results[0], (True, 3.0))


class TestReturnTypes(unittest.TestCase):
    """
    PHASE II — NOT USED IN PHASE I
    Tests to ensure return types are consistent (pure Python types).
    """

    def test_goal_hit_returns_tuple_of_bool_and_float(self):
        result = compute_goal_hit([{'hash': 'h1'}], {'h1'}, 1)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)

    def test_sparse_success_returns_tuple_of_bool_and_float(self):
        result = compute_sparse_success(5, 10, 3)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)

    def test_chain_success_returns_tuple_of_bool_and_float(self):
        result = compute_chain_success([{'hash': 'h1'}], {}, 'h1', 1)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)

    def test_multi_goal_returns_tuple_of_bool_and_float(self):
        result = compute_multi_goal_success({'h1'}, {'h1'})
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], float)


class TestEdgeCases(unittest.TestCase):
    """
    PHASE II — NOT USED IN PHASE I
    Edge case tests for robustness.
    """

    def test_goal_hit_empty_targets(self):
        """Empty targets should return 0 hits."""
        statements = [{'hash': 'h1'}]
        success, value = compute_goal_hit(statements, set(), min_total_verified=0)
        self.assertTrue(success)
        self.assertEqual(value, 0.0)

        success, value = compute_goal_hit(statements, set(), min_total_verified=1)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

    def test_sparse_success_zero_attempted(self):
        """Zero attempted should still work."""
        success, value = compute_sparse_success(verified_count=0, attempted_count=0, min_verified=0)
        self.assertTrue(success)
        self.assertEqual(value, 0.0)

    def test_chain_empty_graph(self):
        """Empty dependency graph - each node is isolated."""
        statements = [{'hash': 'h1'}, {'hash': 'h2'}]
        success, value = compute_chain_success(statements, {}, 'h1', 1)
        self.assertTrue(success)
        self.assertEqual(value, 1.0)

        success, value = compute_chain_success(statements, {}, 'h1', 2)
        self.assertFalse(success)
        self.assertEqual(value, 1.0)

    def test_chain_circular_dependency_protection(self):
        """Graph with cycle should not cause infinite recursion."""
        # Intentional cycle: h1 -> h2 -> h3 -> h1
        cyclic_graph = {
            'h1': ['h3'],
            'h2': ['h1'],
            'h3': ['h2'],
        }
        statements = [{'hash': 'h1'}, {'hash': 'h2'}, {'hash': 'h3'}]
        # Memoization should prevent infinite recursion
        # Since all are verified and connected, traversal visits each once
        success, value = compute_chain_success(statements, cyclic_graph, 'h1', 1)
        self.assertTrue(success)
        # The exact value depends on traversal order, but should complete without error

    def test_multi_goal_single_goal(self):
        """Single required goal."""
        success, value = compute_multi_goal_success({'h1', 'h2'}, {'h1'})
        self.assertTrue(success)
        self.assertEqual(value, 1.0)

    def test_multi_goal_large_sets(self):
        """Large verified set, small required set."""
        verified = {f'h{i}' for i in range(1000)}
        required = {'h500', 'h750'}
        success, value = compute_multi_goal_success(verified, required)
        self.assertTrue(success)
        self.assertEqual(value, 2.0)

    def test_chain_very_deep(self):
        """Deep chain (100 nodes)."""
        dep_graph = {f'h{i}': [f'h{i-1}'] for i in range(1, 100)}
        statements = [{'hash': f'h{i}'} for i in range(100)]

        success, value = compute_chain_success(statements, dep_graph, 'h99', 50)
        self.assertTrue(success)
        self.assertEqual(value, 100.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)