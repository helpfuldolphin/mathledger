# PHASE II — NOT RUN IN PHASE I
"""
Unit tests for the G3/EV3 Determinism Replay gate within the
master verification script, verify_uplift_gates.py.
"""
import unittest
import os
import json
import hashlib
import yaml
import shutil
from scripts.verify_uplift_gates import verify_gates

class TestDeterminismReplay(unittest.TestCase):
    """Validates the replay verification logic."""

    def setUp(self):
        """Set up a valid manifest and consistent/inconsistent replay artifacts."""
        self.run_dir = "test_replay_run_v2"
        self.replay_dir = os.path.join(self.run_dir, "replay")
        self.inconsistent_replay_dir = os.path.join(self.run_dir, "replay_bad")
        os.makedirs(self.replay_dir, exist_ok=True)
        os.makedirs(self.inconsistent_replay_dir, exist_ok=True)

        # --- Artifacts ---
        self.original_results = '{"final_value": 0.9}'
        self.original_telemetry = "TICK 1; TICK 2; FINAL"
        slice_content = '{"p": 1}'

        # --- Hashes ---
        results_hash = hashlib.sha256(self.original_results.encode()).hexdigest()
        telemetry_hash = hashlib.sha256(self.original_telemetry.encode()).hexdigest()
        slice_hash = hashlib.sha256(slice_content.encode()).hexdigest()
        
        # --- Create Preregistration ---
        prereg_entry = {
            "experiment_id": "REPLAY_TEST_01",
            "description": "Replay test",
            "slice_config": "slice.json",
            "seed": 1234,
            "success_metrics": ["final_value"],
            "slice_config_hash": slice_hash
        }
        self.prereg_path = os.path.join(self.run_dir, "prereg.yaml")
        with open(self.prereg_path, "w") as f: yaml.dump([prereg_entry], f)
        prereg_hash = hashlib.sha256(yaml.dump(prereg_entry, sort_keys=True).encode()).hexdigest()

        # --- Create v2 Manifest ---
        self.manifest_data = {
            "manifest_schema_version": "2.0",
            "experiment_id": "REPLAY_TEST_01",
            "preregistration_hash": prereg_hash,
            "slice_config_hash": slice_hash,
            "results_hash": results_hash,
            "ht_binding_hash": telemetry_hash,
            "code_version_hash": "replay-test-commit",
            "deterministic_seed": 2025
        }
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        with open(self.manifest_path, "w") as f: json.dump(self.manifest_data, f)
        
        # --- Create other required files ---
        self.slice_path = os.path.join(self.run_dir, "slice.json")
        self.results_path = os.path.join(self.run_dir, "results.jsonl")
        with open(self.slice_path, "w") as f: f.write(slice_content)
        with open(self.results_path, "w") as f: f.write(self.original_results)

        # --- Create Replay Artifacts (Consistent and Inconsistent) ---
        self.good_replay_results = os.path.join(self.replay_dir, "results.jsonl")
        self.good_replay_telemetry = os.path.join(self.replay_dir, "telemetry.log")
        with open(self.good_replay_results, "w") as f: f.write(self.original_results)
        with open(self.good_replay_telemetry, "w") as f: f.write(self.original_telemetry)

        self.bad_replay_results = os.path.join(self.inconsistent_replay_dir, "results.jsonl")
        self.bad_replay_telemetry = os.path.join(self.inconsistent_replay_dir, "telemetry.log")
        with open(self.bad_replay_results, "w") as f: f.write('{"final_value": 0.90001}')
        with open(self.bad_replay_telemetry, "w") as f: f.write("TICK 1; TICK 3; FINAL")

    def tearDown(self):
        """Clean up the test directory."""
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)

    def test_g3_success_with_consistent_replay(self):
        """G3 must pass when replay artifacts are bit-identical."""
        result = verify_gates(
            manifest_path=self.manifest_path,
            replay_results_file=self.good_replay_results,
            replay_telemetry_file=self.good_replay_telemetry
        )
        self.assertEqual(result["exit_code"], 0, msg=f"STDOUT: {json.dumps(result, indent=2)}")
        self.assertEqual(result["replay_verification"]["status"], "VERIFIED_CONSISTENT")

    def test_g3_failure_with_inconsistent_results(self):
        """G3 must fail with exit code 30 for inconsistent results."""
        result = verify_gates(
            manifest_path=self.manifest_path,
            replay_results_file=self.bad_replay_results,
            replay_telemetry_file=self.good_replay_telemetry
        )
        self.assertEqual(result["exit_code"], 30)
        self.assertEqual(result["replay_verification"]["status"], "FAILED_INCONSISTENT")
        self.assertFalse(result["replay_verification"]["results_hash"]["match"])
        self.assertTrue(result["replay_verification"]["ht_binding_hash"]["match"])

    def test_g3_failure_with_inconsistent_telemetry(self):
        """G3 must fail with exit code 30 for inconsistent telemetry."""
        result = verify_gates(
            manifest_path=self.manifest_path,
            replay_results_file=self.good_replay_results,
            replay_telemetry_file=self.bad_replay_telemetry
        )
        self.assertEqual(result["exit_code"], 30)
        self.assertEqual(result["replay_verification"]["status"], "FAILED_INCONSISTENT")
        self.assertTrue(result["replay_verification"]["results_hash"]["match"])
        self.assertFalse(result["replay_verification"]["ht_binding_hash"]["match"])

    def test_g3_failure_with_missing_replay_file(self):
        """G3 must fail cleanly if a replay file is missing."""
        os.remove(self.good_replay_results)
        result = verify_gates(
            manifest_path=self.manifest_path,
            replay_results_file=self.good_replay_results,
            replay_telemetry_file=self.good_replay_telemetry
        )
        self.assertEqual(result["exit_code"], 30)
        self.assertIn("File not found", result["message"])

if __name__ == "__main__":
    unittest.main()
