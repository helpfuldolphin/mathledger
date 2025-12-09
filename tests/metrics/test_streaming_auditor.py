"""
PHASE II — NOT USED IN PHASE I
Tests for the Streaming Metric Consistency Auditor.
"""
import unittest
import io
import json
import sys
from typing import Any, Dict, List

sys.path.insert(0, '.')
from experiments.metric_consistency_auditor import audit_and_build_from_stream

class TestStreamingAuditor(unittest.TestCase):

    def test_streaming_fingerprint_is_deterministic(self):
        """Ensures the streaming fingerprint is deterministic."""
        jsonl_data = [
            {"hash": "h0", "premises": []},
            {"hash": "h1", "premises": ["h0"]},
        ]
        
        # Run 1
        stream1 = io.StringIO("\n".join(json.dumps(d) for d in jsonl_data))
        config1 = {"metric_kind": "goal_hit", "target_hashes": {"h1"}, "min_goal_hits": 1}
        ledger1 = audit_and_build_from_stream(stream1, config1, set())

        # Run 2 (different order in stream)
        jsonl_data_shuffled = [
            {"hash": "h1", "premises": ["h0"]},
            {"hash": "h0", "premises": []},
        ]
        stream2 = io.StringIO("\n".join(json.dumps(d) for d in jsonl_data_shuffled))
        config2 = {"metric_kind": "goal_hit", "target_hashes": {"h1"}, "min_goal_hits": 1}
        ledger2 = audit_and_build_from_stream(stream2, config2, set())
        
        # The fingerprint of the *content* should be the same because the
        # line hashes are sorted before the final hash.
        self.assertEqual(ledger1["derivation_fingerprint"], ledger2["derivation_fingerprint"])
        
        # The final ledger ID should therefore also be the same.
        self.assertEqual(ledger1["ledger_id"], ledger2["ledger_id"])

    def test_streaming_audit_detects_errors(self):
        """Tests that Phase I audit works correctly on a stream."""
        # Malformed line (missing hash) and dangling premise
        jsonl_data = [
            {"hash": "h0", "premises": ["dangling"]},
            {"premises": []}, 
        ]
        stream = io.StringIO("\n".join(json.dumps(d) for d in jsonl_data))
        config = {"metric_kind": "density", "min_verified": 1, "max_candidates": 10}
        
        ledger = audit_and_build_from_stream(stream, config, set())
        
        self.assertEqual(ledger["auditor_verdict"]["final_status"], "FAILED")
        phase_I_findings = ledger["auditor_verdict"]["phase_I_structural"]["findings"]
        
        # The auditor should find the malformed entry (METINT-2).
        # Because a hash was missing, it will not proceed to check for
        # dangling edges (METINT-6), as those checks rely on a complete
        # set of known hashes. This is the correct, specified behavior.
        self.assertIn("METINT-2", phase_I_findings)
        self.assertNotIn("METINT-6", phase_I_findings)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
