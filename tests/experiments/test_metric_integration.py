"""
PHASE II — NOT USED IN PHASE I
Metric Engine Integration Tests (Adapted)

Tests the integration between the ADAPTED U2 runner and the
final, canonical metric engine.
"""
import unittest
from typing import Any, Dict, List, Set

from experiments.slice_success_metrics import compute_metric

class TestAdaptedMetricIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a common run-cycle context for all tests."""
        # h1 <- h2 <- h4 (verified)
        # h3 (verified)
        # h5 (not verified)
        derivations: List[Dict[str, Any]] = [
            {"hash": "h1", "premises": []},
            {"hash": "h2", "premises": ["h1"]},
            {"hash": "h3", "premises": []},
            {"hash": "h4", "premises": ["h2"]},
            {"hash": "h5", "premises": []},
        ]
        self.run_data = {
            "verified_hashes": {"h1", "h2", "h3", "h4"},
            "candidates_tried": 200,
            "result": {"derivations": derivations},
        }

    def _run_test_case(self, kind: str, metric_config: Dict[str, Any]) -> Any:
        """Helper to run the metric engine with the common context."""
        args = {
            "kind": kind,
            **self.run_data,
            **metric_config,
        }
        return compute_metric(**args)

    def test_integration_goal_hit_success(self):
        """Test a successful 'goal_hit' integration."""
        metric_config = {
            "target_hashes": {"h1", "h3"},
            "min_goal_hits": 2,
            "min_total_verified": 4,
        }
        success, value, _ = self._run_test_case("goal_hit", metric_config)
        self.assertTrue(success)
        self.assertAlmostEqual(value, 1.0) # 2 hits / 2 targets

    def test_integration_density_success(self):
        """Test a successful 'density' integration."""
        metric_config = {
            "min_verified": 4, # We have 4 verified
            "max_candidates": 500,
        }
        success, value, _ = self._run_test_case("density", metric_config)
        self.assertTrue(success)
        # 4 verified / 200 candidates
        self.assertAlmostEqual(value, 4 / 200)

    def test_integration_chain_length_success(self):
        """Test a successful 'chain_length' integration."""
        metric_config = {
            "chain_target_hash": "h4",
            "min_chain_length": 3,  # h4->h2->h1 is length 3
        }
        success, value, _ = self._run_test_case("chain_length", metric_config)
        self.assertTrue(success)
        self.assertEqual(value, 3.0)

    def test_integration_chain_length_fail_target_not_verified(self):
        """Test a failed 'chain_length' when target is not verified."""
        metric_config = {
            "chain_target_hash": "h5", # not in verified_hashes
            "min_chain_length": 1,
        }
        # Override verified_hashes for this one test
        self.run_data["verified_hashes"] = {"h1", "h2", "h3", "h4"}
        success, value, _ = self._run_test_case("chain_length", metric_config)
        self.assertFalse(success)
        self.assertEqual(value, 0.0)

    def test_integration_multi_goal_success(self):
        """Test a successful 'multi_goal' integration."""
        metric_config = {
            "required_goal_hashes": {"h1", "h4"},
            "min_each_goal": 1,
        }
        success, value, _ = self._run_test_case("multi_goal", metric_config)
        self.assertTrue(success)
        self.assertAlmostEqual(value, 1.0)

    def test_integration_multi_goal_fail(self):
        """Test a failed 'multi_goal' integration."""
        metric_config = {
            "required_goal_hashes": {"h1", "h5"}, # h5 is not verified
            "min_each_goal": 1,
        }
        success, value, _ = self._run_test_case("multi_goal", metric_config)
        self.assertFalse(success)
        self.assertAlmostEqual(value, 0.5) # 1 hit / 2 required

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)