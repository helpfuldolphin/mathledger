"""
PHASE II — NOT USED IN PHASE I
Tests for the Metric Integration Ledger (Adapted for Streaming)
"""
import unittest
import json
import io
import copy
import sys
from typing import Any, Dict, List

sys.path.insert(0, '.')
from experiments.metric_consistency_auditor import audit_and_build_from_stream

class TestMetricLedgerStreaming(unittest.TestCase):

    def setUp(self):
        """Set up a canonical valid input set for tests."""
        self.valid_derivations: List[Dict[str, Any]] = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h2", "premises": ["h1"]},
        ]
        self.valid_config: Dict[str, Any] = {
            "metric_kind": "chain_length",
            "chain_target_hash": "h2",
            "min_chain_length": 3,
        }

    def _run_ledger_generation(self, derivations, config):
        """Helper to run the full pipeline from a list of derivations."""
        stream = io.StringIO("\n".join(json.dumps(d) for d in derivations))
        return audit_and_build_from_stream(stream, config, set())

    def test_ledger_determinism(self):
        """Tests that two runs with identical inputs produce identical ledger IDs."""
        print("\nRunning test_ledger_determinism (streaming)...")
        
        ledger1 = self._run_ledger_generation(self.valid_derivations, copy.deepcopy(self.valid_config))
        ledger2 = self._run_ledger_generation(self.valid_derivations, copy.deepcopy(self.valid_config))
        
        self.assertEqual(ledger1["ledger_id"], ledger2["ledger_id"])

    def test_input_tamper_detection(self):
        """Tests that a change in input derivations results in a different ledger_id."""
        print("\nRunning test_input_tamper_detection (streaming)...")
        ledger_original = self._run_ledger_generation(self.valid_derivations, copy.deepcopy(self.valid_config))

        tampered_derivations = copy.deepcopy(self.valid_derivations)
        tampered_derivations[0]["premises"] = ["new_premise"]
        
        ledger_tampered = self._run_ledger_generation(tampered_derivations, copy.deepcopy(self.valid_config))

        self.assertNotEqual(ledger_original["derivation_fingerprint"], ledger_tampered["derivation_fingerprint"])
        self.assertNotEqual(ledger_original["ledger_id"], ledger_tampered["ledger_id"])

    def test_metint_coverage(self):
        """Tests that bad data correctly triggers METINT findings in the ledger."""
        print("\nRunning test_metint_coverage (streaming)...")
        invalid_derivations = [
            {"premises": []}, # METINT-2
            {"hash": "h1", "premises": ["dangling"]}, # METINT-6
        ]
        
        ledger = self._run_ledger_generation(invalid_derivations, copy.deepcopy(self.valid_config))

        self.assertEqual(ledger["auditor_verdict"]["final_status"], "FAILED")
        findings = ledger["auditor_verdict"]["phase_I_structural"]["findings"]
        
        # The auditor correctly finds the malformed entry (METINT-2).
        # Because a hash was missing, it stops checking for dangling edges.
        self.assertIn("METINT-2", findings)
        self.assertNotIn("METINT-6", findings)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)