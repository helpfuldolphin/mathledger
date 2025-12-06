#!/usr/bin/env python3
"""
Unit tests for U2 calibration and verbose-cycles functionality.

Tests the Phase II calibration outputs, determinism verification,
schema validation, and the --verbose-cycles CLI flag.

PHASE II — NOT USED IN PHASE I
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestU2Calibration(unittest.TestCase):
    """Test cases for U2 calibration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(__file__).resolve().parents[2]
        self.calibration_script = self.project_root / "experiments" / "u2_calibration.py"
        self.runner_script = self.project_root / "experiments" / "run_uplift_u2.py"
        self.config_path = self.project_root / "config" / "curriculum_uplift_phase2.yaml"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calibration_output_structure(self):
        """Test that calibration outputs have correct directory structure."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "5",
                "--seed", "42",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        # Check output structure
        slice_dir = Path(self.temp_dir) / "arithmetic_simple"
        self.assertTrue(slice_dir.exists(), "Slice directory not created")
        self.assertTrue((slice_dir / "baseline.jsonl").exists(), "baseline.jsonl not created")
        self.assertTrue((slice_dir / "rfl.jsonl").exists(), "rfl.jsonl not created")
        self.assertTrue((slice_dir / "calibration_summary.json").exists(), "calibration_summary.json not created")

    def test_calibration_summary_schema(self):
        """Test that calibration_summary.json has correct schema."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "10",
                "--seed", "42",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        summary_path = Path(self.temp_dir) / "arithmetic_simple" / "calibration_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        
        # Required fields
        required_fields = {
            "slice",
            "baseline_cycles",
            "baseline_successes",
            "rfl_cycles",
            "rfl_successes",
            "determinism_verified",
            "schema_valid",
            "phase",
        }
        self.assertEqual(set(summary.keys()), required_fields, "Unexpected fields in calibration summary")
        
        # Check field types
        self.assertIsInstance(summary["slice"], str)
        self.assertIsInstance(summary["baseline_cycles"], int)
        self.assertIsInstance(summary["baseline_successes"], int)
        self.assertIsInstance(summary["rfl_cycles"], int)
        self.assertIsInstance(summary["rfl_successes"], int)
        self.assertIsInstance(summary["determinism_verified"], bool)
        self.assertIsInstance(summary["schema_valid"], bool)
        self.assertEqual(summary["phase"], "PHASE II")

    def test_calibration_summary_no_uplift_fields(self):
        """Test that calibration_summary.json has NO uplift statistics."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "10",
                "--seed", "42",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        summary_path = Path(self.temp_dir) / "arithmetic_simple" / "calibration_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        
        # Forbidden uplift fields
        forbidden_fields = ["delta_p", "p_value", "ci_lower", "ci_upper", "uplift", "statistics"]
        for field in forbidden_fields:
            self.assertNotIn(field, summary, f"Forbidden field '{field}' found in calibration summary")

    def test_verbose_cycles_flag(self):
        """Test that --verbose-cycles produces per-cycle output."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "3",
                "--seed", "42",
                "--out", self.temp_dir,
                "--verbose-cycles",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        # Check for verbose cycle output format
        self.assertIn("[cycle=0]", result.stdout, "Verbose cycle output not found")
        self.assertIn("mode=baseline", result.stdout, "Mode not in verbose output")
        self.assertIn("mode=rfl", result.stdout, "RFL mode not in verbose output")
        self.assertIn("success=", result.stdout, "Success not in verbose output")
        self.assertIn("verified=", result.stdout, "Verified not in verbose output")
        self.assertIn("abstained=", result.stdout, "Abstained not in verbose output")
        self.assertIn("item=", result.stdout, "Item hash not in verbose output")

    def test_verbose_cycles_disabled_by_default(self):
        """Test that verbose output is NOT present when flag is not used."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "3",
                "--seed", "42",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        # Verbose cycle output should NOT be present
        self.assertNotIn("[cycle=0]", result.stdout, "Verbose output should not be present by default")

    def test_verbose_cycles_does_not_modify_jsonl(self):
        """Test that --verbose-cycles does NOT modify JSONL output."""
        # Run without verbose
        out_dir_no_verbose = Path(self.temp_dir) / "no_verbose"
        result1 = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "5",
                "--seed", "42",
                "--out", str(out_dir_no_verbose),
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        # Run with verbose
        out_dir_verbose = Path(self.temp_dir) / "verbose"
        result2 = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "5",
                "--seed", "42",
                "--out", str(out_dir_verbose),
                "--verbose-cycles",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result1.returncode, 0, f"Run 1 failed: {result1.stderr}")
        self.assertEqual(result2.returncode, 0, f"Run 2 failed: {result2.stderr}")
        
        # Compare JSONL outputs
        baseline1 = (out_dir_no_verbose / "arithmetic_simple" / "baseline.jsonl").read_text()
        baseline2 = (out_dir_verbose / "arithmetic_simple" / "baseline.jsonl").read_text()
        self.assertEqual(baseline1, baseline2, "JSONL output differs with --verbose-cycles")
        
        rfl1 = (out_dir_no_verbose / "arithmetic_simple" / "rfl.jsonl").read_text()
        rfl2 = (out_dir_verbose / "arithmetic_simple" / "rfl.jsonl").read_text()
        self.assertEqual(rfl1, rfl2, "JSONL output differs with --verbose-cycles")

    def test_determinism_verification(self):
        """Test that determinism verification passes for same seed."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "10",
                "--seed", "42",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        summary_path = Path(self.temp_dir) / "arithmetic_simple" / "calibration_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        
        self.assertTrue(summary["determinism_verified"], "Determinism verification should pass")

    def test_run_uplift_u2_calibrate_mode(self):
        """Test that run_uplift_u2.py supports 'calibrate' mode."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.runner_script),
                "--slice", "arithmetic_simple",
                "--cycles", "5",
                "--seed", "42",
                "--mode", "calibrate",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibrate mode failed: {result.stderr}")
        
        # Check output structure
        slice_dir = Path(self.temp_dir) / "arithmetic_simple"
        self.assertTrue(slice_dir.exists(), "Slice directory not created")
        self.assertTrue((slice_dir / "calibration_summary.json").exists(), "calibration_summary.json not created")

    def test_run_uplift_u2_verbose_cycles_baseline(self):
        """Test that run_uplift_u2.py --verbose-cycles works in baseline mode."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.runner_script),
                "--slice", "arithmetic_simple",
                "--cycles", "3",
                "--seed", "42",
                "--mode", "baseline",
                "--out", self.temp_dir,
                "--verbose-cycles",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Baseline with verbose failed: {result.stderr}")
        self.assertIn("[cycle=0]", result.stdout, "Verbose output not found")
        self.assertIn("mode=baseline", result.stdout, "Mode not in verbose output")

    def test_run_uplift_u2_verbose_cycles_rfl(self):
        """Test that run_uplift_u2.py --verbose-cycles works in RFL mode."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.runner_script),
                "--slice", "arithmetic_simple",
                "--cycles", "3",
                "--seed", "42",
                "--mode", "rfl",
                "--out", self.temp_dir,
                "--verbose-cycles",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"RFL with verbose failed: {result.stderr}")
        self.assertIn("[cycle=0]", result.stdout, "Verbose output not found")
        self.assertIn("mode=rfl", result.stdout, "Mode not in verbose output")

    def test_jsonl_schema_validation(self):
        """Test that JSONL records have required fields."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "5",
                "--seed", "42",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        # Check baseline.jsonl schema
        baseline_path = Path(self.temp_dir) / "arithmetic_simple" / "baseline.jsonl"
        required_fields = {"cycle", "slice", "mode", "seed", "item", "result", "success", "label"}
        
        with open(baseline_path) as f:
            for line in f:
                record = json.loads(line)
                self.assertTrue(required_fields.issubset(record.keys()), f"Missing required fields: {required_fields - set(record.keys())}")
                self.assertIn("PHASE II", record["label"], "PHASE II label not in record")

    def test_phase_ii_label_present(self):
        """Test that all outputs contain PHASE II labels."""
        result = subprocess.run(
            [
                sys.executable,
                str(self.calibration_script),
                "--slice", "arithmetic_simple",
                "--cycles", "3",
                "--seed", "42",
                "--out", self.temp_dir,
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
        )
        
        self.assertEqual(result.returncode, 0, f"Calibration failed: {result.stderr}")
        
        # Check summary
        summary_path = Path(self.temp_dir) / "arithmetic_simple" / "calibration_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        self.assertEqual(summary["phase"], "PHASE II", "Phase label not PHASE II")
        
        # Check stdout contains PHASE II label
        self.assertIn("PHASE II", result.stdout, "PHASE II label not in output")


if __name__ == "__main__":
    unittest.main()
