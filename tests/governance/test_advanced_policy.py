"""
PHASE IV Tests for Advanced Policy Enforcement & MAAS Summaries
"""
import unittest
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.metric_governance import (
    can_promote_metric,
    build_metric_conformance_timeline,
    summarize_metrics_for_global_console,
    load_promotion_policy
)

class TestAdvancedPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = load_promotion_policy()
        self.baseline = {"metric_name": "test_metric", "levels": {"L0_reproducible": {"status": "PASS"}, "L1_deterministic": {"status": "PASS"}, "L2_domain_coverage": {"status": "PASS"}, "L3_regression": {"status": "PASS"}}}
        self.candidate = {"metric_name": "test_metric", "levels": {"L0_reproducible": {"status": "PASS"}, "L1_deterministic": {"status": "PASS"}, "L2_domain_coverage": {"status": "PASS"}, "L3_regression": {"status": "PASS"}}}

    def test_policy_blocks_on_flapping(self):
        """A metric should be blocked if it exceeds the flapping policy."""
        # This timeline data simulates a metric that is flapping
        timeline_data = {
            "advanced_analytics": {
                "flapping_events_count": 4, # Exceeds default policy max of 2
                "long_term_drift": 0.0,
                "is_regression_outlier": False
            }
        }
        # Use the strict 'default' policy
        policy = {"default": self.policy["default"]}
        can_promote, reason = can_promote_metric(self.baseline, self.candidate, timeline_data, policy)
        self.assertFalse(can_promote)
        self.assertIn("flapping detected", reason)

    def test_policy_blocks_on_drift(self):
        """A metric should be blocked if it exceeds the drift policy."""
        timeline_data = {
            "advanced_analytics": {
                "flapping_events_count": 0,
                "long_term_drift": 0.1, # Exceeds default policy max of 0.05
                "is_regression_outlier": False
            }
        }
        policy = {"default": self.policy["default"]}
        can_promote, reason = can_promote_metric(self.baseline, self.candidate, timeline_data, policy)
        self.assertFalse(can_promote)
        self.assertIn("drift detected", reason)

    def test_policy_allows_drift_for_lenient_policy(self):
        """A drifting metric should be allowed if the policy is lenient."""
        self.candidate["metric_name"] = "legacy_metrics_A" # Match the lenient policy
        timeline_data = {
            "advanced_analytics": {
                "flapping_events_count": 0,
                "long_term_drift": 0.4, # Exceeds default max, but not legacy_metrics max
                "is_regression_outlier": False
            }
        }
        can_promote, reason = can_promote_metric(self.baseline, self.candidate, timeline_data, self.policy)
        self.assertTrue(can_promote)

    def test_maas_summary_ok(self):
        """Test the MAAS summary for a healthy system."""
        timeline = {"metric1": {"advanced_analytics": {"is_flapping": False, "is_regression_outlier": False}}}
        promotions = [{"metric_name": "metric1", "status": "PASS"}]
        summary = summarize_metrics_for_global_console(timeline, promotions)
        self.assertEqual(summary["status"], "OK")
        self.assertTrue(summary["metrics_ok"])
        self.assertIn("stable and conformant", summary["headline"])

    def test_maas_summary_warn_on_flapping(self):
        """Test the MAAS summary for a flapping metric warning."""
        timeline = {"metric1": {"advanced_analytics": {"is_flapping": True, "is_regression_outlier": False}}}
        promotions = [{"metric_name": "metric1", "status": "PASS"}]
        summary = summarize_metrics_for_global_console(timeline, promotions)
        self.assertEqual(summary["status"], "WARN")
        self.assertFalse(summary["metrics_ok"])
        self.assertEqual(summary["flapping_metrics"], ["metric1"])

    def test_maas_summary_block_on_promotion_failure(self):
        """Test the MAAS summary for a blocking promotion failure."""
        timeline = {"metric1": {"advanced_analytics": {"is_flapping": False, "is_regression_outlier": False}}}
        promotions = [{"metric_name": "metric1", "status": "FAIL"}]
        summary = summarize_metrics_for_global_console(timeline, promotions)
        self.assertEqual(summary["status"], "BLOCK")
        self.assertFalse(summary["metrics_ok"])
        self.assertEqual(summary["failing_promotions"], ["metric1"])
        self.assertIn("Action required", summary["headline"])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
