import json
import os
import subprocess
import sys
import unittest

class TestDagAudit(unittest.TestCase):

    def test_script_runs_and_produces_json_output(self):
        """
        A simple smoke test to ensure the script runs and outputs valid JSON.
        """
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'scripts', 'proof_dag_u2_audit.py'
        )
        
        # Check if the script exists
        self.assertTrue(os.path.exists(script_path), f"Script not found at {script_path}")

        # Run the script as a subprocess
        process = subprocess.run(
            [sys.executable, script_path, '--exp-id', 'u2_test_exp_1'],
            capture_output=True,
            text=True,
            check=False 
        )

        # Check for errors in script execution
        self.assertEqual(process.returncode, 0, f"Script failed with error:\n{process.stderr}")
        
        # Try to parse the output as JSON
        try:
            output_json = json.loads(process.stdout)
        except json.JSONDecodeError:
            self.fail("Script did not produce valid JSON output.")

        # Check for expected top-level keys
        self.assertIn("audit_metadata", output_json)
        self.assertIn("summary", output_json)
        self.assertIn("invariant_results", output_json)

        # Check that the summary looks reasonable
        self.assertEqual(output_json['summary']['total_invariants'], 11)


if __name__ == '__main__':
    unittest.main()
