# tests/dag/test_policy_evaluation.py
"""
PHASE III - Unit tests for the policy-driven drift evaluation engine.
"""
import unittest

# Add project root for local imports
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.posture_analysis import evaluate_dag_drift_acceptability

class TestPolicyEvaluation(unittest.TestCase):
    """
    Tests the generic, policy-driven drift evaluation engine.
    """

    def setUp(self):
        """Set up mock postures and policies."""
        self.old_posture = {
            "schema_version": "1.0.0", "has_cycles": False, "max_depth": 20,
            "vertex_count": 100, "edge_count": 120, "drift_eligible": True
        }
        
        # A strict policy that blocks on moderate depth loss
        self.strict_policy = {
            "rules": [{
                "name": "Strict Depth Regression", "status": "BLOCKED",
                "conditions": [{"metric": "comparison.depth_delta", "operator": "<", "value": -3}]
            }],
            "default_status": "OK"
        }
        
        # A lax policy that only warns on the same condition
        self.lax_policy = {
            "rules": [{
                "name": "Lax Depth Regression", "status": "WARN",
                "conditions": [{"metric": "comparison.depth_delta", "operator": "<", "value": -3}]
            }],
            "default_status": "OK"
        }
        
        # A policy that checks for vertex growth ratio
        self.growth_policy = {
            "rules": [{
                "name": "Explosive Vertex Growth", "status": "WARN",
                "conditions": [{"metric": "comparison.vertex_growth_ratio", "operator": ">", "value": 0.5}]
            }],
            "default_status": "OK"
        }
        
        # A policy that checks for cycle introduction
        self.cycle_policy = {
            "rules": [{
                "name": "Cycle Introduced", "status": "BLOCKED",
                "conditions": [
                    {"metric": "new.has_cycles", "operator": "==", "value": True},
                    {"metric": "old.has_cycles", "operator": "==", "value": False}
                ]
            }],
            "default_status": "OK"
        }

    def test_engine_with_strict_policy(self):
        """Test that a moderate regression is BLOCKED by a strict policy."""
        new_posture = self.old_posture.copy()
        new_posture["max_depth"] = 16 # Delta of -4
        
        result = evaluate_dag_drift_acceptability(self.old_posture, new_posture, self.strict_policy)
        
        self.assertEqual(result["drift_status"], "BLOCKED")
        self.assertEqual(result["reasons"][0], "Strict Depth Regression")

    def test_engine_with_lax_policy(self):
        """Test that the same regression is a WARN with a lax policy."""
        new_posture = self.old_posture.copy()
        new_posture["max_depth"] = 16 # Delta of -4
        
        result = evaluate_dag_drift_acceptability(self.old_posture, new_posture, self.lax_policy)
        
        self.assertEqual(result["drift_status"], "WARN")
        self.assertEqual(result["reasons"][0], "Lax Depth Regression")

    def test_default_status(self):
        """Test that the default status is returned when no rules match."""
        new_posture = self.old_posture.copy()
        new_posture["max_depth"] = 19 # Delta of -1, does not meet threshold
        
        result = evaluate_dag_drift_acceptability(self.old_posture, new_posture, self.strict_policy)
        
        self.assertEqual(result["drift_status"], "OK")
        self.assertEqual(result["reasons"][0], "Default status: No rules matched.")
        
    def test_vertex_growth_ratio_rule(self):
        """Test a rule using a computed comparison metric."""
        new_posture = self.old_posture.copy()
        new_posture["vertex_count"] = 160 # 60% growth
        
        result = evaluate_dag_drift_acceptability(self.old_posture, new_posture, self.growth_policy)
        
        self.assertEqual(result["drift_status"], "WARN")
        self.assertAlmostEqual(result["comparison"]["vertex_growth_ratio"], 0.6)
        
    def test_cycle_introduction_rule(self):
        """Test a rule that compares old and new posture states directly."""
        new_posture = self.old_posture.copy()
        new_posture["has_cycles"] = True
        
        result = evaluate_dag_drift_acceptability(self.old_posture, new_posture, self.cycle_policy)
        
        self.assertEqual(result["drift_status"], "BLOCKED")
        self.assertEqual(result["reasons"][0], "Cycle Introduced")

if __name__ == '__main__':
    unittest.main()
