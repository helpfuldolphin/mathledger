# PHASE IV — SUBSTRATE GOVERNANCE
#
# Unit tests for the Substrate Identity Governance Analyzer.

import unittest
from typing import Any, Dict, List

# Add project root for imports
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.governance.analyzer import (
    analyze_substrate_identity_ledger,
    evaluate_substrate_for_promotion,
    build_substrate_director_panel,
)

# --- Mock Data ---

def create_mock_envelope(
    spec_version="1.0",
    version_hash="hash_stable",
    audit_failures: List[str] = []
) -> Dict[str, Any]:
    """Helper to create a mock identity envelope."""
    audit = {
        "global_rng_check": "PASSED",
        "file_io_check": "PASSED",
        "network_check": "PASSED",
    }
    for failure in audit_failures:
        if failure in audit:
            audit[failure] = "FAILED"

    return {
        "spec_version": spec_version,
        "version_hash": version_hash,
        "forbidden_behavior_audit": audit,
        # Other fields are not needed for this analysis
    }

# A perfect ledger
GOOD_LEDGER = [create_mock_envelope() for _ in range(10)]

# A ledger with substrate version drift
DRIFT_LEDGER = [
    create_mock_envelope(version_hash="hash_A"),
    create_mock_envelope(version_hash="hash_A"),
    create_mock_envelope(version_hash="hash_B"), # Drift!
    create_mock_envelope(version_hash="hash_A"),
]

# A ledger with behavioral flags
BEHAVIOR_LEDGER = [
    create_mock_envelope(),
    create_mock_envelope(audit_failures=["file_io_check"]),
    create_mock_envelope(),
    create_mock_envelope(audit_failures=["file_io_check", "network_check"]),
]


class TestGovernanceAnalyzer(unittest.TestCase):
    
    def test_analyze_good_ledger(self):
        """Test analysis of a perfectly stable and conformant ledger."""
        print("PHASE IV: Testing analyzer with GOOD_LEDGER...")
        analysis = analyze_substrate_identity_ledger(GOOD_LEDGER)
        
        self.assertEqual(analysis["governance_status"], "OK")
        self.assertEqual(analysis["identity_stability_index"], 1.0)
        self.assertFalse(analysis["substrate_version_drift"]["detected"])
        self.assertEqual(len(analysis["substrate_version_drift"]["hashes"]), 1)
        self.assertEqual(analysis["repeated_behavioral_flags"], [])
        print("  -> PASSED")

    def test_analyze_drift_ledger(self):
        """Test analysis of a ledger with version hash drift."""
        print("PHASE IV: Testing analyzer with DRIFT_LEDGER...")
        analysis = analyze_substrate_identity_ledger(DRIFT_LEDGER)
        
        self.assertEqual(analysis["governance_status"], "BLOCK")
        self.assertLess(analysis["identity_stability_index"], 1.0)
        self.assertTrue(analysis["substrate_version_drift"]["detected"])
        self.assertEqual(len(analysis["substrate_version_drift"]["hashes"]), 2)
        self.assertIn("hash_A", analysis["substrate_version_drift"]["hashes"])
        self.assertIn("hash_B", analysis["substrate_version_drift"]["hashes"])
        print("  -> PASSED")

    def test_analyze_behavior_ledger(self):
        """Test analysis of a ledger with forbidden behavior flags."""
        print("PHASE IV: Testing analyzer with BEHAVIOR_LEDGER...")
        analysis = analyze_substrate_identity_ledger(BEHAVIOR_LEDGER)
        
        self.assertEqual(analysis["governance_status"], "BLOCK")
        self.assertEqual(analysis["identity_stability_index"], 1.0) # No drift
        self.assertFalse(analysis["substrate_version_drift"]["detected"])
        
        flags = analysis["repeated_behavioral_flags"]
        self.assertEqual(len(flags), 2)
        
        file_io_flag = next(f for f in flags if f["check"] == "file_io_check")
        network_flag = next(f for f in flags if f["check"] == "network_check")
        
        self.assertEqual(file_io_flag["count"], 2)
        self.assertEqual(network_flag["count"], 1)
        print("  -> PASSED")

    def test_promotion_gate(self):
        """Test the promotion gate logic with different analysis results."""
        print("PHASE IV: Testing promotion gate logic...")
        # Good case
        good_analysis = analyze_substrate_identity_ledger(GOOD_LEDGER)
        good_eval = evaluate_substrate_for_promotion(good_analysis, {})
        self.assertTrue(good_eval["substrate_ok_for_promotion"])
        self.assertEqual(good_eval["status"], "OK")
        
        # Drift case
        drift_analysis = analyze_substrate_identity_ledger(DRIFT_LEDGER)
        drift_eval = evaluate_substrate_for_promotion(drift_analysis, {})
        self.assertFalse(drift_eval["substrate_ok_for_promotion"])
        self.assertEqual(drift_eval["status"], "BLOCK")
        self.assertIn("version drift detected", drift_eval["blocking_reasons"][0])
        
        # Behavior case
        behavior_analysis = analyze_substrate_identity_ledger(BEHAVIOR_LEDGER)
        behavior_eval = evaluate_substrate_for_promotion(behavior_analysis, {})
        self.assertFalse(behavior_eval["substrate_ok_for_promotion"])
        self.assertEqual(behavior_eval["status"], "BLOCK")
        self.assertIn("Forbidden behavior detected", behavior_eval["blocking_reasons"][0])
        print("  -> PASSED")
        
    def test_director_panel(self):
        """Test the director panel generation."""
        print("PHASE IV: Testing director panel generation...")
        # Good case
        good_analysis = analyze_substrate_identity_ledger(GOOD_LEDGER)
        good_eval = evaluate_substrate_for_promotion(good_analysis, {})
        good_panel = build_substrate_director_panel(good_analysis, good_eval)
        self.assertEqual(good_panel["status_light"], "GREEN")
        self.assertEqual(good_panel["identity_stability_index"], 1.0)
        self.assertEqual(good_panel["substrate_hash"], "hash_stable")
        self.assertIn("approved for promotion", good_panel["headline"])

        # Block case
        drift_analysis = analyze_substrate_identity_ledger(DRIFT_LEDGER)
        drift_eval = evaluate_substrate_for_promotion(drift_analysis, {})
        drift_panel = build_substrate_director_panel(drift_analysis, drift_eval)
        self.assertEqual(drift_panel["status_light"], "RED")
        self.assertEqual(drift_panel["substrate_hash"], "INCONSISTENT")
        self.assertIn("BLOCK: Substrate version drift detected", drift_panel["headline"])
        print("  -> PASSED")

if __name__ == '__main__':
    unittest.main()
