# PHASE III — CROSS-RUN HISTORY & INCIDENTS
"""
Tests for the Replay History Ledger, Incident Extractor, and Global Health Hook.
"""
import unittest
import json
import os
from backend.governance.replay_history import (
    ReplayReceipt,
    ReceiptIndexEntry,
    ReceiptSummary,
    build_replay_history,
    save_replay_history,
    load_replay_history,
    extract_replay_incidents,
    summarize_replay_for_global_health,
)

# --- Synthetic Data ---
SYNTHETIC_RECEIPTS = [
    ReplayReceipt(
        run_id="run_001", manifest_path="runs/001/m.json", status="VERIFIED",
        recon_codes=[], expected_hash="h1", actual_hash="h1", timestamp="2025-01-01T10:00:00Z"
    ),
    ReplayReceipt(
        run_id="run_003", manifest_path="runs/003/m.json", status="FAILED",
        recon_codes=["RECON-002"], expected_hash="h3", actual_hash="h3_bad", timestamp="2025-01-03T10:00:00Z"
    ),
    ReplayReceipt(
        run_id="run_002", manifest_path="runs/002/m.json", status="VERIFIED",
        recon_codes=[], expected_hash="h2", actual_hash="h2", timestamp="2025-01-02T10:00:00Z"
    ),
    ReplayReceipt(
        run_id="run_004", manifest_path="runs/004/m.json", status="INCOMPLETE",
        recon_codes=["RECON-001", "RECON-003"], expected_hash="h4", actual_hash="", timestamp="" # No timestamp
    ),
    ReplayReceipt(
        run_id="run_005", manifest_path="runs/005/m.json", status="FAILED",
        recon_codes=["RECON-001"], expected_hash="h5", actual_hash="h5_bad", timestamp="2025-01-05T10:00:00Z"
    ),
]
SYNTHETIC_INDEX = ReceiptIndexEntry(receipts=SYNTHETIC_RECEIPTS)
SYNTHETIC_SUMMARY = ReceiptSummary(
    total_receipts=5, num_verified=2, num_failed=2, num_incomplete=1,
    recon_error_codes=["RECON-001", "RECON-002", "RECON-003"]
)

class TestReplayHistory(unittest.TestCase):

    def test_build_replay_history(self):
        """Verify the history ledger is built correctly."""
        history = build_replay_history(SYNTHETIC_INDEX)

        self.assertEqual(history["schema_version"], "1.0.0")
        self.assertEqual(history["total_receipts"], 5)
        self.assertEqual(history["number_verified"], 2)
        self.assertEqual(history["number_failed"], 2)
        self.assertEqual(history["number_incomplete"], 1)
        self.assertEqual(history["first_successful_replay_at"], "2025-01-01T10:00:00Z")
        self.assertEqual(history["last_failure_at"], "2025-01-05T10:00:00Z")

    def test_save_and_load_replay_history(self):
        """Verify the history can be serialized and deserialized."""
        history = build_replay_history(SYNTHETIC_INDEX)
        path = "test_history.json"
        
        save_replay_history(path, history)
        self.assertTrue(os.path.exists(path))
        
        loaded_history = load_replay_history(path)
        self.assertDictEqual(history, loaded_history)
        
        os.remove(path)

    def test_extract_replay_incidents(self):
        """Verify incident extraction is deterministic and correct."""
        incidents = extract_replay_incidents(SYNTHETIC_RECEIPTS)

        self.assertEqual(len(incidents), 3)
        
        # Check deterministic ordering (by run_id)
        self.assertEqual(incidents[0]["run_id"], "run_003")
        self.assertEqual(incidents[1]["run_id"], "run_004")
        self.assertEqual(incidents[2]["run_id"], "run_005")

        # Check content of an incident
        incident_4 = incidents[1]
        self.assertEqual(incident_4["status"], "INCOMPLETE")
        self.assertEqual(incident_4["expected_hash"], "h4")
        self.assertEqual(incident_4["ht_series_hash"], "")
        self.assertListEqual(incident_4["recon_codes_seen"], ["RECON-001", "RECON-003"])

    def test_global_health_hook_ok(self):
        """Test health hook under normal (OK) conditions."""
        ok_summary = ReceiptSummary(total_receipts=3, num_verified=3, num_failed=0, num_incomplete=0, recon_error_codes=[])
        health = summarize_replay_for_global_health(ok_summary)
        
        self.assertEqual(health["status"], "OK")
        self.assertTrue(health["all_verified"])
        self.assertEqual(health["failure_count"], 0)

    def test_global_health_hook_warn(self):
        """Test health hook under warning (WARN) conditions."""
        warn_summary = ReceiptSummary(total_receipts=4, num_verified=3, num_failed=1, num_incomplete=0, recon_error_codes=["RECON-002"])
        health = summarize_replay_for_global_health(warn_summary)

        self.assertEqual(health["status"], "WARN")
        self.assertFalse(health["all_verified"])
        self.assertEqual(health["failure_count"], 1)
        self.assertListEqual(health["recon_error_codes"], ["RECON-002"])

    def test_global_health_hook_blocked(self):
        """Test health hook under blocked (BLOCKED) conditions."""
        health = summarize_replay_for_global_health(SYNTHETIC_SUMMARY)

        self.assertEqual(health["status"], "BLOCKED")
        self.assertFalse(health["all_verified"])
        self.assertEqual(health["failure_count"], 3)
        self.assertIn("RECON-001", health["recon_error_codes"])

if __name__ == "__main__":
    unittest.main()
