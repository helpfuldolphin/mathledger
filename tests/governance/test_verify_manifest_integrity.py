# PHASE II — NOT RUN IN PHASE I
"""
Unit tests for the verify_manifest_integrity.py script.
"""
import unittest
import subprocess
import os
import json
import hashlib
import yaml

class TestVerifyManifestIntegrity(unittest.TestCase):
    """Tests for the manifest integrity verification script."""

    def setUp(self):
        """Set up test files."""
        self.prereg_content = """
- experiment_id: U2_GOOD
  description: "Good experiment"
  slice_config: "slice.json"
  slice_config_hash: "{slice_hash}"
  seed: 123
  success_metrics: [goal_hit]
"""
        self.slice_content = '{"param": "value"}'
        self.results_content = '{"metric": 0.95}'

        # Create files
        with open("slice.json", "w") as f: f.write(self.slice_content)
        with open("results.jsonl", "w") as f: f.write(self.results_content)

        # Calculate hashes
        self.slice_hash = hashlib.sha256(self.slice_content.encode()).hexdigest()
        self.results_hash = hashlib.sha256(self.results_content.encode()).hexdigest()

        # Create prereg with correct hash
        self.prereg_content = self.prereg_content.format(slice_hash=self.slice_hash)
        with open("prereg.yaml", "w") as f: f.write(self.prereg_content)

        # Calculate prereg hash
        prereg_dict = yaml.safe_load(self.prereg_content)[0]
        self.prereg_hash = hashlib.sha256(yaml.dump(prereg_dict, sort_keys=True).encode()).hexdigest()

        # Create manifest
        self.manifest_content = {
            "manifest_schema_version": "1.0",
            "experiment_id": "U2_GOOD",
            "preregistration_hash": self.prereg_hash,
            "slice_config_hash": self.slice_hash,
            "results_hash": self.results_hash,
            "code_version_hash": "git-commit-hash",
            "deterministic_seed": 123
        }
        with open("manifest.json", "w") as f: json.dump(self.manifest_content, f)

    def tearDown(self):
        """Clean up test files."""
        for f in ["slice.json", "results.jsonl", "prereg.yaml", "manifest.json", "bad_manifest.json"]:
            if os.path.exists(f):
                os.remove(f)

    def run_script(self, args):
        """Helper to run the script."""
        base_command = ["python", "scripts/verify_manifest_integrity.py"]
        return subprocess.run(base_command + args, capture_output=True, text=True, check=False)

    def test_success_case(self):
        """Test a valid manifest and matching files."""
        args = ["--manifest", "manifest.json", "--prereg-file", "prereg.yaml", "--slice-config", "slice.json", "--results-file", "results.jsonl", "--json"]
        result = self.run_script(args)
        self.assertEqual(result.returncode, 0, msg=f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        output = json.loads(result.stdout)
        self.assertEqual(output["status"], "SUCCESS")
        self.assertListEqual(output["verified_bindings"], ["preregistration", "slice_config", "results"])

    def test_invalid_manifest_schema(self):
        """Test manifest with missing fields."""
        bad_manifest = self.manifest_content.copy()
        del bad_manifest["preregistration_hash"]
        with open("bad_manifest.json", "w") as f: json.dump(bad_manifest, f)
        result = self.run_script(["--manifest", "bad_manifest.json", "--prereg-file", "prereg.yaml", "--slice-config", "slice.json", "--results-file", "results.jsonl"])
        self.assertEqual(result.returncode, 1)

    def test_prereg_hash_mismatch(self):
        """Test manifest with a bad preregistration hash."""
        bad_manifest = self.manifest_content.copy()
        bad_manifest["preregistration_hash"] = "bad_hash"
        with open("bad_manifest.json", "w") as f: json.dump(bad_manifest, f)
        result = self.run_script(["--manifest", "bad_manifest.json", "--prereg-file", "prereg.yaml", "--slice-config", "slice.json", "--results-file", "results.jsonl"])
        self.assertEqual(result.returncode, 2)

    def test_slice_config_hash_mismatch(self):
        """Test manifest with a bad slice config hash."""
        bad_manifest = self.manifest_content.copy()
        bad_manifest["slice_config_hash"] = "bad_hash"
        with open("bad_manifest.json", "w") as f: json.dump(bad_manifest, f)
        result = self.run_script(["--manifest", "bad_manifest.json", "--prereg-file", "prereg.yaml", "--slice-config", "slice.json", "--results-file", "results.jsonl"])
        self.assertEqual(result.returncode, 3)

    def test_results_hash_mismatch(self):
        """Test manifest with a bad results hash."""
        bad_manifest = self.manifest_content.copy()
        bad_manifest["results_hash"] = "bad_hash"
        with open("bad_manifest.json", "w") as f: json.dump(bad_manifest, f)
        result = self.run_script(["--manifest", "bad_manifest.json", "--prereg-file", "prereg.yaml", "--slice-config", "slice.json", "--results-file", "results.jsonl"])
        self.assertEqual(result.returncode, 4)

if __name__ == "__main__":
    unittest.main()
