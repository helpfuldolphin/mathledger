# tests/dag/test_posture_analysis.py
"""
PHASE III - Unit tests for DAG posture analysis functions.
"""
import unittest

# Add project root for local imports
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.posture_analysis import evaluate_dag_drift_acceptability, build_dag_posture_timeline

class TestPostureAnalysis(unittest.TestCase):
    """
    Tests the posture analysis and drift gating functions.
    """

    def setUp(self):
        """Set up mock posture snapshots."""
        self.base_posture = {
            "schema_version": "1.0.0", "has_cycles": False, "max_depth": 20,
            "vertex_count": 1000, "edge_count": 1200, "drift_eligible": True
        }

    def test_drift_acceptability_ok(self):
        """Test the case where drift is acceptable."""
        new_posture = self.base_posture.copy()
        new_posture["max_depth"] = 22
        new_posture["vertex_count"] = 1100

        result = evaluate_dag_drift_acceptability(self.base_posture, new_posture)
        self.assertEqual(result["drift_status"], "OK")
        self.assertIn("Depth Growth: Max depth increased by 2.", result["reasons"])
        
    def test_drift_acceptability_warn_depth_regression(self):
        """Test a WARN case for significant depth regression."""
        new_posture = self.base_posture.copy()
        new_posture["max_depth"] = 14 # a drop of 6, which is > threshold of 5

        result = evaluate_dag_drift_acceptability(self.base_posture, new_posture)
        self.assertEqual(result["drift_status"], "WARN")
        self.assertIn("Depth Regression: Max depth decreased by 6.", result["reasons"])

    def test_drift_acceptability_blocked(self):
        """Test a BLOCKED case where depth regresses and eligibility is lost."""
        new_posture = self.base_posture.copy()
        new_posture["max_depth"] = 14
        new_posture["drift_eligible"] = False
        new_posture["drift_ineligibility_reason"] = "FAIL: DRIFT-002 - Vertex Divergence"

        result = evaluate_dag_drift_acceptability(self.base_posture, new_posture)
        self.assertEqual(result["drift_status"], "BLOCKED")
        self.assertIn("CRITICAL REGRESSION", result["reasons"][0])
    
    def test_build_timeline_empty(self):
        """Test the timeline builder with no snapshots."""
        result = build_dag_posture_timeline([])
        self.assertEqual(len(result["timeline"]), 0)
        self.assertEqual(result["aggregates"], {})

    def test_build_timeline_aggregates_and_flags(self):
        """Test timeline aggregation and trend flag detection."""
        # Test data designed to trigger sustained regression and explosive growth
        snapshots = [
            {"timestamp": 1, "max_depth": 10, "vertex_count": 100, "drift_eligible": True},
            {"timestamp": 2, "max_depth": 8,  "vertex_count": 120, "drift_eligible": True},  # depth regression
            {"timestamp": 3, "max_depth": 7,  "vertex_count": 140, "drift_eligible": True},  # depth regression
            {"timestamp": 4, "max_depth": 6,  "vertex_count": 160, "drift_eligible": False}, # depth regression, sustained
            {"timestamp": 5, "max_depth": 15, "vertex_count": 401, "drift_eligible": True},  # explosive growth (>150%)
            {"timestamp": 6, "max_depth": 20, "vertex_count": 1003, "drift_eligible": True}, # explosive growth (>150%)
        ]

        result = build_dag_posture_timeline(snapshots)

        # Test aggregates
        self.assertEqual(result["aggregates"]["total_snapshots"], 6)
        self.assertEqual(result["aggregates"]["eligible_count"], 5)
        self.assertEqual(result["aggregates"]["ineligible_count"], 1)
        self.assertEqual(result["aggregates"]["positive_depth_delta_periods"], 2)
        self.assertEqual(result["aggregates"]["negative_depth_delta_periods"], 3)
        
        # Test trend flags
        self.assertTrue(result["trend_flags"]["sustained_depth_regression"])
        self.assertTrue(result["trend_flags"]["explosive_vertex_growth"])
        
    def test_timeline_no_trends(self):
        """Test that trend flags are false for a stable timeline."""
        snapshots = [
            {"timestamp": 1, "max_depth": 10, "vertex_count": 100, "drift_eligible": True},
            {"timestamp": 2, "max_depth": 11, "vertex_count": 110, "drift_eligible": True},
            {"timestamp": 3, "max_depth": 10, "vertex_count": 120, "drift_eligible": True},
            {"timestamp": 4, "max_depth": 11, "vertex_count": 130, "drift_eligible": True},
        ]
        result = build_dag_posture_timeline(snapshots)
        self.assertFalse(result["trend_flags"]["sustained_depth_regression"])
        self.assertFalse(result["trend_flags"]["explosive_vertex_growth"])

if __name__ == '__main__':
    unittest.main()
