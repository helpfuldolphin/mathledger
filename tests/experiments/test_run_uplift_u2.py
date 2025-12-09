# PHASE II — NOT RUN IN PHASE I
"""
Unit tests for the U2 Uplift Runner.
"""
import unittest
import os
import json
import shutil
import subprocess
import sys

class TestU2Runner(unittest.TestCase):
    """
    Validates the U2 Runner's artifact generation, dual-run execution,
    and determinism reporting.
    """
    def setUp(self):
        self.test_dir = "test_runner_output"
        self.runner_script = "experiments/run_uplift_u2.py"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def run_runner(self, extra_args=None):
        """Helper to run the script via subprocess."""
        args = [sys.executable, self.runner_script, "--run-dir", self.test_dir]
        if extra_args:
            args.extend(extra_args)
        return subprocess.run(args, capture_output=True, text=True, check=False)

    def test_dual_run_execution_and_artifacts(self):
        """
        Verify that a standard run produces all required primary and replay
        artifacts.
        """
        result = self.run_runner()
        self.assertEqual(result.returncode, 0, msg=f"Runner failed:\n{result.stderr}")

        # Check for primary artifacts
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "results.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "telemetry.jsonl")))
        
        # Check for replay artifacts
        replay_dir = os.path.join(self.test_dir, "replay")
        self.assertTrue(os.path.exists(replay_dir))
        self.assertTrue(os.path.exists(os.path.join(replay_dir, "results.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(replay_dir, "telemetry.jsonl")))

        # Check for summary artifacts
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "governance_input.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "replay_determinism.json")))

    def test_manifest_v2_generation(self):
        """
        Verify the manifest contains the mandatory ht_binding_hash.
        """
        self.run_runner()
        manifest_path = os.path.join(self.test_dir, "manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.assertIn("ht_binding_hash", manifest)
        self.assertIsInstance(manifest["ht_binding_hash"], str)
        self.assertEqual(len(manifest["ht_binding_hash"]), 64) # SHA256 hex digest length

    def test_determinism_summary_on_match(self):
        """
        Verify the replay summary reports VERIFIED and ALWAYS_DETERMINISTIC
        when the runs are identical.
        """
        self.run_runner()
        summary_path = os.path.join(self.test_dir, "replay_determinism.json")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            
        self.assertEqual(summary["status"], "VERIFIED")
        self.assertEqual(summary["confidence"], "ALWAYS_DETERMINISTIC")
        self.assertEqual(summary["expected"]["results_hash"], summary["actual"]["results_hash"])

    def test_determinism_summary_on_mismatch(self):
        """
        Verify the summary reports FAILED and POTENTIAL_SUBSTRATE_DRIFT
        when the runs differ. This requires modifying the script's behavior.
        """
        # We need to temporarily modify the script to inject non-determinism
        with open(self.runner_script, "r") as f:
            original_code = f.read()
        
        # Inject randomness into the replay run
        modified_code = original_code.replace(
            "run_single_experiment(args.seed, replay_dir, \"replay\")",
            "run_single_experiment(args.seed + 1, replay_dir, \"replay\")" # Use a different seed for replay
        )
        
        with open(self.runner_script, "w") as f:
            f.write(modified_code)

        try:
            result = self.run_runner()
            # The script itself shouldn't fail, but should report the mismatch
            self.assertEqual(result.returncode, 0)

            summary_path = os.path.join(self.test_dir, "replay_determinism.json")
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            self.assertEqual(summary["status"], "FAILED")
            self.assertEqual(summary["confidence"], "POTENTIAL_SUBSTRATE_DRIFT")
            self.assertNotEqual(summary["expected"]["results_hash"], summary["actual"]["results_hash"])
        finally:
            # Restore the original script content
            with open(self.runner_script, "w") as f:
                f.write(original_code)

if __name__ == "__main__":
    unittest.main()
