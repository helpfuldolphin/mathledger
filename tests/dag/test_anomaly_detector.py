# tests/dag/test_anomaly_detector.py
"""
PHASE II - Unit tests for the AnomalyDetector.
"""
import unittest
from pathlib import Path
import json
import tempfile

# Add project root for local imports
import sys
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.anomaly_detector import AnomalyDetector
from backend.dag.invariant_guard import ProofDag, SliceProfile

class TestAnomalyDetector(unittest.TestCase):
    """
    Tests the functionality of the AnomalyDetector class.
    """

    def test_proof_chain_collapse(self):
        """Test detection of proof chain collapse."""
        metrics = [
            {"cycle": 0, "MaxDepth(t)": 10, "ΔMaxDepth(t)": 2},
            {"cycle": 1, "MaxDepth(t)": 5, "ΔMaxDepth(t)": -5},
            {"cycle": 2, "MaxDepth(t)": 4, "ΔMaxDepth(t)": -1},
            {"cycle": 3, "MaxDepth(t)": 2, "ΔMaxDepth(t)": -2},
        ]
        # Use a less extreme threshold for the test
        config = {"collapse_threshold": -4}
        detector = AnomalyDetector(metrics, config)
        anomalies = detector.detect_proof_chain_collapse()
        
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["anomaly_type"], "ProofChainCollapse")
        self.assertEqual(anomalies[0]["cycle"], 1)

    def test_depth_stagnation(self):
        """Test detection of depth stagnation."""
        metrics = [{"cycle": i, "ΔMaxDepth(t)": 0} for i in range(15)]
        config = {"stagnation_duration": 5, "stagnation_threshold": 0}
        detector = AnomalyDetector(metrics, config)
        anomalies = detector.detect_depth_stagnation()
        
        # Expect multiple anomaly windows to be detected
        self.assertTrue(len(anomalies) > 0)
        self.assertEqual(anomalies[0]["anomaly_type"], "DepthStagnation")
        self.assertEqual(anomalies[0]["start_cycle"], 0)
        self.assertEqual(anomalies[0]["end_cycle"], 4)

    def test_explosive_branching(self):
        """Test detection of explosive branching."""
        metrics = [
            {"cycle": 0, "GlobalBranchingFactor(t)": 1.5},
            {"cycle": 1, "GlobalBranchingFactor(t)": 1.8},
            {"cycle": 2, "GlobalBranchingFactor(t)": 4.5}, # Spike
        ]
        config = {"branching_factor_std_dev_threshold": 3.0}
        detector = AnomalyDetector(metrics, config)
        anomalies = detector.detect_explosive_branching()
        
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["anomaly_type"], "ExplosiveBranching")
        self.assertEqual(anomalies[0]["cycle"], 2)

    def test_duplicate_proof_patterns(self):
        """Test detection of high duplication rate."""
        metrics = [
            {"cycle": 0, "total_derivations_in_cycle": 10, "duplicate_derivations_in_cycle": 2},
            {"cycle": 1, "total_derivations_in_cycle": 10, "duplicate_derivations_in_cycle": 9},
        ]
        config = {"duplication_ratio_threshold": 0.8}
        detector = AnomalyDetector(metrics, config)
        anomalies = detector.detect_duplicate_proof_patterns()
        
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["anomaly_type"], "DuplicateProofPattern")
        self.assertEqual(anomalies[0]["cycle"], 1)
        self.assertAlmostEqual(anomalies[0]["duplication_ratio"], 0.9)

    def test_no_anomalies(self):
        """Test that no anomalies are detected in healthy metrics."""
        metrics = [
            {"cycle": i, "ΔMaxDepth(t)": 2, "GlobalBranchingFactor(t)": 1.5,
             "total_derivations_in_cycle": 10, "duplicate_derivations_in_cycle": 1}
            for i in range(20)
        ]
        detector = AnomalyDetector(metrics)
        anomalies = detector.detect_all()
        self.assertEqual(len(anomalies), 0)

    def test_invariant_guard_integration(self):
        """Test invariant guard execution alongside anomaly detection."""
        metrics = [
            {"cycle": 0, "MaxDepth(t)": 3, "GlobalBranchingFactor(t)": 1.0},
            {"cycle": 1, "MaxDepth(t)": 6, "GlobalBranchingFactor(t)": 3.2},
        ]
        proof_dag = ProofDag(
            slices={
                "slice-alpha": SliceProfile(
                    slice_id="slice-alpha",
                    max_depth=6,
                    max_branching_factor=2.5,
                    node_kind_counts={"LEMMA": 3, "EXPR": 1},
                )
            },
            metric_ledger=metrics,
        )
        rules = {
            "max_depth_per_slice": {"slice-alpha": 5},
            "max_branching_factor": 2.5,
            "allowed_node_kinds": {"slice-alpha": {"LEMMA"}},
        }
        detector = AnomalyDetector(
            metrics,
            proof_dag=proof_dag,
            invariant_rules=rules,
        )
        detector.detect_all()
        report = detector.invariant_report()
        self.assertIsNotNone(report)
        self.assertEqual(report["status"], "BLOCK")
        self.assertGreaterEqual(len(report["violated_invariants"]), 2)

if __name__ == '__main__':
    unittest.main()
