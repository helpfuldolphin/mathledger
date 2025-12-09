# PHASE II — NOT RUN IN PHASE I
"""
Unit tests for the verify_uplift_gates.py script.
"""
import unittest
import subprocess
import os
import json
import hashlib
import yaml

class TestVerifyUpliftGates(unittest.TestCase):
    """Tests for the master governance gate verification script."""

    def setUp(self):
        """Set up a complete, valid set of artifacts in a test directory."""
        self.run_dir = "test_run_valid"
        os.makedirs(self.run_dir, exist_ok=True)

        # --- Create Artifacts ---
        slice_content = '{"param": "value"}'
        results_content = '{"metric": 0.95}'

        with open(os.path.join(self.run_dir, "slice.json"), "w") as f: f.write(slice_content)
        with open(os.path.join(self.run_dir, "results.jsonl"), "w") as f: f.write(results_content)

        slice_hash = hashlib.sha256(slice_content.encode()).hexdigest()
        results_hash = hashlib.sha256(results_content.encode()).hexdigest()

        prereg_content_dict = [{
            "experiment_id": "U2_VALID_RUN",
            "description": "Valid run",
            "slice_config": "slice.json",
            "slice_config_hash": slice_hash,
            "seed": 42,
            "success_metrics": ["goal_hit"]
        }]
        with open(os.path.join(self.run_dir, "prereg.yaml"), "w") as f: yaml.dump(prereg_content_dict, f)

        prereg_hash = hashlib.sha256(yaml.dump(prereg_content_dict[0], sort_keys=True).encode()).hexdigest()

        manifest_content = {
            "manifest_schema_version": "1.0",
            "experiment_id": "U2_VALID_RUN",
            "preregistration_hash": prereg_hash,
            "slice_config_hash": slice_hash,
            "results_hash": results_hash,
            "code_version_hash": "git-commit-hash",
            "deterministic_seed": 42
        }
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f: json.dump(manifest_content, f)

    def tearDown(self):
        """Clean up test directory."""
        for root, dirs, files in os.walk(self.run_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.run_dir)

    def run_script(self, args):
        """Helper to run the script."""
        base_command = ["python", "scripts/verify_uplift_gates.py"]
        return subprocess.run(base_command + args, capture_output=True, text=True, check=False)

    def test_success_case(self):
        """Test a fully valid run that should pass all gates."""
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        result = self.run_script(["--manifest", manifest_path, "--json"])
        self.assertEqual(result.returncode, 0, msg=f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "SUCCESS")
        self.assertEqual(output["checked_gates"]["G1_preregistration"], "PASS")

    def test_prereg_gate_failure(self):
        """Test failure when preregistration is invalid."""
        # Corrupt the prereg hash in the manifest
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        with open(manifest_path, "r+") as f:
            data = json.load(f)
            data["preregistration_hash"] = "invalid_hash"
            f.seek(0)
            json.dump(data, f)
            f.truncate()

        result = self.run_script(["--manifest", manifest_path])
        self.assertEqual(result.returncode, 20) # Integrity failure (which is checked after prereg)

    def test_integrity_gate_failure(self):
        """Test failure when a file hash does not match the manifest."""
        # Change the slice file content
        with open(os.path.join(self.run_dir, "slice.json"), "w") as f:
            f.write('{"param": "changed_value"}')

        manifest_path = os.path.join(self.run_dir, "manifest.json")
        result = self.run_script(["--manifest", manifest_path])
        self.assertEqual(result.returncode, 20)

if __name__ == "__main__":
    unittest.main()
