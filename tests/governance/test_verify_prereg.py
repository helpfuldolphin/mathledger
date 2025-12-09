# PHASE II — NOT RUN IN PHASE I
"""
Unit tests for the verify_prereg.py script.
"""
import unittest
import subprocess
import os
import json

class TestVerifyPrereg(unittest.TestCase):
    """Tests for the preregistration verification script."""

    def setUp(self):
        """Set up test files."""
        self.valid_prereg_content = """
# PHASE II - NOT USED IN PHASE I
- experiment_id: U2_EXP_001
  description: "Experiment to measure uplift in slice A."
  slice_config: "configs/slice_a.json"
  slice_config_hash: "sha256-abcde"
  seed: 12345
  success_metrics:
    - goal_hit
    - sparse_density
- experiment_id: U2_EXP_002
  description: "Experiment B"
  slice_config: "configs/slice_b.json"
  slice_config_hash: "sha256-fghij"
  seed: 67890
  success_metrics:
    - chain_success
"""
        self.invalid_prereg_content = """
- experiment_id: U2_EXP_003
  description: "Missing seed"
  slice_config: "configs/slice_c.json"
  slice_config_hash: "sha256-klmno"
  # Missing seed
  success_metrics:
    - goal_hit
"""
        self.malformed_yaml_content = "key: value: another value"

        with open("test_valid_prereg.yaml", "w", encoding="utf-8") as f:
            f.write(self.valid_prereg_content)
        with open("test_invalid_prereg.yaml", "w", encoding="utf-8") as f:
            f.write(self.invalid_prereg_content)
        with open("test_malformed.yaml", "w", encoding="utf-8") as f:
            f.write(self.malformed_yaml_content)

    def tearDown(self):
        """Clean up test files."""
        os.remove("test_valid_prereg.yaml")
        os.remove("test_invalid_prereg.yaml")
        os.remove("test_malformed.yaml")

    def run_script(self, args):
        """Helper to run the script and return process info."""
        base_command = ["python", "scripts/verify_prereg.py"]
        return subprocess.run(base_command + args, capture_output=True, text=True, check=False)

    def test_success_case(self):
        """Test a valid preregistration entry."""
        result = self.run_script(["--prereg-file", "test_valid_prereg.yaml", "--experiment-id", "U2_EXP_001", "--json"])
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "SUCCESS")
        self.assertEqual(output["experiment_id"], "U2_EXP_001")
        self.assertIn("preregistration_hash", output)
        self.assertEqual(output["details"]["seed"], 12345)

    def test_file_not_found(self):
        """Test for a missing preregistration file."""
        result = self.run_script(["--prereg-file", "nonexistent.yaml", "--experiment-id", "U2_EXP_001"])
        self.assertEqual(result.returncode, 1)

    def test_malformed_yaml(self):
        """Test for a malformed YAML file."""
        result = self.run_script(["--prereg-file", "test_malformed.yaml", "--experiment-id", "U2_EXP_001"])
        self.assertEqual(result.returncode, 1)

    def test_experiment_not_found(self):
        """Test for an experiment ID that does not exist."""
        result = self.run_script(["--prereg-file", "test_valid_prereg.yaml", "--experiment-id", "U2_EXP_999"])
        self.assertEqual(result.returncode, 2)

    def test_missing_fields(self):
        """Test for an entry with missing required fields."""
        result = self.run_script(["--prereg-file", "test_invalid_prereg.yaml", "--experiment-id", "U2_EXP_003"])
        self.assertEqual(result.returncode, 3)

if __name__ == "__main__":
    unittest.main()
