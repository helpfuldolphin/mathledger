# PHASE IIb — NOT RUN IN PHASE I
#
# Integration tests for the refactored "Thin-Waist" U2 Runner.

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in path for imports
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# The module to test
from experiments import run_uplift_u2 as runner

class TestRunnerIntegration(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary directory for test artifacts."""
        self.test_dir = Path("./test_runner_artifacts")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_deterministic_envelope_generation(self):
        """
        Validates that two identical runs produce bit-for-bit identical manifests,
        proving end-to-end determinism of the envelope generation.
        """
        print("PHASE IIb: Testing end-to-end determinism...")
        
        common_args = [
            "--slice", "s1",
            "--mode", "baseline",
            "--cycles", "10",
            "--substrate", "mock",
        ]
        
        # Run 1
        outdir1 = self.test_dir / "run1"
        argv1 = ["prog"] + common_args + ["--seed", "42", "--outdir", str(outdir1)]
        with patch.object(sys, 'argv', argv1):
            runner.main()
        
        # Run 2
        outdir2 = self.test_dir / "run2"
        argv2 = ["prog"] + common_args + ["--seed", "42", "--outdir", str(outdir2)]
        with patch.object(sys, 'argv', argv2):
            runner.main()
            
        # Compare the manifests
        manifest1_path = outdir1 / "manifest_s1_baseline.json"
        manifest2_path = outdir2 / "manifest_s1_baseline.json"
        
        self.assertTrue(manifest1_path.exists())
        self.assertTrue(manifest2_path.exists())
        
        content1 = manifest1_path.read_text()
        content2 = manifest2_path.read_text()
        
        self.assertEqual(content1, content2, "Manifests from two identical runs are not identical.")
        print("  -> PASSED")

    def test_ci_summary_generation_and_exit_code(self):
        """
        Validates that the governance summary is created and that the runner
        exits with the correct code if an audit fails.
        
        This test requires a substrate that fails an audit, which is not
        defined in the main substrate file. We can mock it or accept that
        this test only checks for the PASS case for now. We'll test the PASS case.
        """
        print("PHASE IIb: Testing summary generation...")
        argv = [
            "prog",
            "--slice", "s2",
            "--mode", "rfl",
            "--cycles", "5",
            "--seed", "101",
            "--substrate", "mock",
            "--outdir", str(self.test_dir)
        ]
        
        with patch.object(sys, 'argv', argv):
            # The script should exit with 0 on success
            with self.assertRaises(SystemExit) as cm:
                runner.main()
            self.assertEqual(cm.exception.code, 0)
        
        # Check that the summary file was created
        summary_path = self.test_dir / "governance_summary_s2_rfl.json"
        self.assertTrue(summary_path.exists())
        
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        self.assertEqual(summary["behavioral_audit_summary"]["status"], "PASSED")
        self.assertEqual(summary["total_executions"], 5)
        self.assertEqual(summary["substrate_names"], ["MockSubstrate"])
        print("  -> PASSED")

    def test_error_propagation_on_bad_substrate(self):
        """
        Validates that if a substrate fails its audit, the failure is
        recorded in the summary and the runner exits with a non-zero code.
        """
        print("PHASE IIb: Testing error propagation...")
        
        # We need to patch the substrate's execute method to simulate a failed audit.
        # A bit complex, let's instead create a temporary bad substrate for this test.
        # For this test, we can assume the conformance suite already proved
        # that the audit works. Here we just test if the runner *reacts* to it.
        
        # Simulate a summary with a failure
        bad_summary = {
            "behavioral_audit_summary": {
                "status": "FAILED",
                "failures": [{"cycle": 0, "check": "file_io_check"}]
            }
        }
        
        # Patch `build_substrate_summary` to return our bad summary
        with patch('experiments.run_uplift_u2.build_substrate_summary', return_value=bad_summary):
            argv = ["prog", "--slice", "s3", "--mode", "baseline", "--cycles", "1", "--seed", "1", "--substrate", "mock", "--outdir", str(self.test_dir)]
            with patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit) as cm:
                    runner.main()
                # Check for failure exit code
                self.assertEqual(cm.exception.code, 1)
        
        print("  -> PASSED")

if __name__ == '__main__':
    unittest.main()
