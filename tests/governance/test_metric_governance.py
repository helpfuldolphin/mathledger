"""
PHASE III — LIVE METRIC GOVERNANCE
Tests for the metric governance and promotion logic.
"""
import unittest
import json
import os
import sys
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.metric_governance import (
    can_promote_metric,
    build_metric_conformance_timeline,
    load_promotion_policy
)

class TestMetricGovernance(unittest.TestCase):

    def setUp(self):
        """Set up common snapshots and policy for tests."""
        self.policy = load_promotion_policy()

        # Baseline snapshot, all PASS
        self.baseline = {
            "metric_name": "uplift_u2_density",
            "levels": {
                "L0_reproducible": {"status": "PASS"},
                "L1_deterministic": {"status": "PASS"},
                "L2_domain_coverage": {"status": "PASS"},
                "L3_regression": {"status": "PASS"}
            }
        }
        
        # Candidate with L3 regression
        self.cand_l3_fail = {
            "metric_name": "uplift_u2_density",
            "levels": {
                "L0_reproducible": {"status": "PASS"},
                "L1_deterministic": {"status": "PASS"},
                "L2_domain_coverage": {"status": "PASS"},
                "L3_regression": {"status": "FAIL", "details": "Regression"}
            }
        }

        # Candidate with L2 failure
        self.cand_l2_fail = {
            "metric_name": "uplift_u2_density",
            "levels": {
                "L0_reproducible": {"status": "PASS"},
                "L1_deterministic": {"status": "PASS"},
                "L2_domain_coverage": {"status": "FAIL"},
                "L3_regression": {"status": "PASS"}
            }
        }

        # Perfect candidate
        self.cand_ok = {
            "metric_name": "uplift_u2_density",
            "levels": {
                "L0_reproducible": {"status": "PASS"},
                "L1_deterministic": {"status": "PASS"},
                "L2_domain_coverage": {"status": "PASS"},
                "L3_regression": {"status": "PASS"}
            }
        }

    def test_can_promote_with_l3_regression_allowed(self):
        """A metric with an L3 regression SHOULD pass if the policy allows it."""
        can_promote, reason = can_promote_metric(self.baseline, self.cand_l3_fail, self.policy)
        self.assertTrue(can_promote, f"Should be promotable but was denied for: {reason}")

    def test_can_promote_with_l3_regression_denied(self):
        """A metric with an L3 regression SHOULD FAIL if the policy denies it."""
        # Force use of default policy
        default_policy = {"default": self.policy["default"]}
        can_promote, reason = can_promote_metric(self.baseline, self.cand_l3_fail, default_policy)
        self.assertFalse(can_promote, "Should be denied but was promoted.")
        self.assertIn("L3 regression is not tolerated", reason)

    def test_can_promote_with_l2_failure(self):
        """A metric with a failure at its required level (L2) MUST fail."""
        can_promote, reason = can_promote_metric(self.baseline, self.cand_l2_fail, self.policy)
        self.assertFalse(can_promote, "Should be denied for L2 failure but was promoted.")
        self.assertIn("failed required level L2", reason)

    def test_can_promote_clean_metric(self):
        """A metric with no regressions should always pass."""
        can_promote, reason = can_promote_metric(self.baseline, self.cand_ok, self.policy)
        self.assertTrue(can_promote)

    def test_timeline_builder(self):
        """Tests the logic of the conformance timeline builder."""
        script_dir = os.path.dirname(__file__)
        snapshot_paths = [
            os.path.join(script_dir, '..', '..', 'artifacts', 'snapshots', 'baseline', 'uplift_u2_density.json'),
            os.path.join(script_dir, '..', '..', 'artifacts', 'snapshots', 'candidate', 'uplift_u2_density.json'),
            os.path.join(script_dir, '..', '..', 'artifacts', 'snapshots', 'history_2025_12_01_uplift_u2_density.json'),
        ]
        
        timeline = build_metric_conformance_timeline(snapshot_paths)
        
        self.assertIn("uplift_u2_density", timeline)
        density_timeline = timeline["uplift_u2_density"]
        
        self.assertEqual(density_timeline["total_runs"], 3)
        self.assertEqual(density_timeline["regression_count"], 1)
        self.assertEqual(density_timeline["current_streak"]["status"], "FAIL")
        self.assertEqual(density_timeline["current_streak"]["count"], 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
