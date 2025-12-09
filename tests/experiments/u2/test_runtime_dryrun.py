"""
PHASE II — NOT USED IN PHASE I

U2 Runtime Dry-Run Validation Tests
====================================

Tests for the --dry-run-config CLI option in experiments/u2_runtime_inspect.py.

This module verifies:
- Valid configs return exit code 0 and status=OK
- Invalid configs return non-zero exit code with clear error messages
- JSON output includes machine-parseable error codes
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def run_dryrun(config_content: str, use_json: bool = False) -> subprocess.CompletedProcess:
    """
    Helper to run dry-run validation with a temporary config file.
    
    Args:
        config_content: YAML content for the config file.
        use_json: If True, add --json flag.
    
    Returns:
        CompletedProcess with stdout, stderr, returncode.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        cmd = [
            sys.executable,
            "experiments/u2_runtime_inspect.py",
            "--dry-run-config",
            config_path,
        ]
        if use_json:
            cmd.append("--json")
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)


class TestDryRunValidConfig(unittest.TestCase):
    """Tests for valid configuration files."""

    def test_valid_config_exits_zero(self) -> None:
        """Valid config should return exit code 0."""
        config = """
slices:
  test_slice:
    mode: baseline
    cycles: 10
    seed: 42
"""
        result = run_dryrun(config)
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

    def test_valid_config_status_ok(self) -> None:
        """Valid config should return status OK in JSON."""
        config = """
slices:
  test_slice:
    mode: rfl
    cycles: 100
    seed: 12345
"""
        result = run_dryrun(config, use_json=True)
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "OK")
        self.assertEqual(data["errors"], [])

    def test_valid_config_shows_slices_found(self) -> None:
        """Valid config should report number of slices found."""
        config = """
slices:
  slice_a:
    mode: baseline
  slice_b:
    mode: rfl
  slice_c:
    mode: baseline
"""
        result = run_dryrun(config, use_json=True)
        data = json.loads(result.stdout)
        
        self.assertEqual(data["slices_found"], 3)

    def test_empty_slices_gives_warning(self) -> None:
        """Config with no slices should give a warning but not error."""
        config = """
# Empty config
other_field: value
"""
        result = run_dryrun(config, use_json=True)
        
        # Should succeed (exit 0) but with warning
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "OK")
        
        # Should have a warning about no slices
        warning_codes = [w["code"] for w in data["warnings"]]
        self.assertIn("NO_SLICES", warning_codes)


class TestDryRunInvalidConfig(unittest.TestCase):
    """Tests for invalid configuration files."""

    def test_missing_file_exits_nonzero(self) -> None:
        """Missing config file should return non-zero exit code."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--dry-run-config",
                "nonexistent_config_12345.yaml",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_missing_file_error_code(self) -> None:
        """Missing config file should return CONFIG_NOT_FOUND error."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--dry-run-config",
                "nonexistent_config_12345.yaml",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "ERROR")
        
        error_codes = [e["code"] for e in data["errors"]]
        self.assertIn("CONFIG_NOT_FOUND", error_codes)

    def test_invalid_mode_exits_nonzero(self) -> None:
        """Invalid mode should return non-zero exit code."""
        config = """
slices:
  test_slice:
    mode: invalid_mode
    cycles: 10
"""
        result = run_dryrun(config)
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_mode_error_message(self) -> None:
        """Invalid mode should return clear error message."""
        config = """
slices:
  test_slice:
    mode: random_shuffle
    cycles: 10
"""
        result = run_dryrun(config, use_json=True)
        
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "ERROR")
        
        error_codes = [e["code"] for e in data["errors"]]
        self.assertIn("INVALID_MODE", error_codes)
        
        # Error message should mention the slice name
        error_messages = [e["message"] for e in data["errors"]]
        self.assertTrue(
            any("test_slice" in msg for msg in error_messages),
            f"Error messages should mention slice name: {error_messages}"
        )

    def test_negative_cycles_exits_nonzero(self) -> None:
        """Negative cycles should return non-zero exit code."""
        config = """
slices:
  test_slice:
    mode: baseline
    cycles: -5
"""
        result = run_dryrun(config)
        self.assertNotEqual(result.returncode, 0)

    def test_negative_cycles_error_code(self) -> None:
        """Negative cycles should return INVALID_CYCLES error."""
        config = """
slices:
  test_slice:
    mode: baseline
    cycles: -10
"""
        result = run_dryrun(config, use_json=True)
        
        data = json.loads(result.stdout)
        error_codes = [e["code"] for e in data["errors"]]
        self.assertIn("INVALID_CYCLES", error_codes)

    def test_invalid_seed_type(self) -> None:
        """Non-integer seed should return INVALID_SEED error."""
        config = """
slices:
  test_slice:
    mode: baseline
    cycles: 10
    seed: "not_a_number"
"""
        result = run_dryrun(config, use_json=True)
        
        data = json.loads(result.stdout)
        error_codes = [e["code"] for e in data["errors"]]
        self.assertIn("INVALID_SEED", error_codes)


class TestDryRunOutputFormat(unittest.TestCase):
    """Tests for output format of dry-run validation."""

    def test_json_output_has_required_fields(self) -> None:
        """JSON output should have all required fields."""
        config = """
slices:
  test: {}
"""
        result = run_dryrun(config, use_json=True)
        data = json.loads(result.stdout)
        
        required_fields = ["status", "runtime_version", "config_path", "errors", "warnings"]
        for field in required_fields:
            self.assertIn(field, data, f"Missing field: {field}")

    def test_json_output_is_valid_json(self) -> None:
        """Output with --json should always be valid JSON."""
        config = """
slices:
  test:
    mode: invalid
"""
        result = run_dryrun(config, use_json=True)
        
        # Should not raise
        data = json.loads(result.stdout)
        self.assertIsInstance(data, dict)

    def test_text_output_includes_status(self) -> None:
        """Text output should include status."""
        config = """
slices:
  test:
    mode: baseline
"""
        result = run_dryrun(config, use_json=False)
        
        self.assertIn("Status:", result.stdout)
        self.assertIn("OK", result.stdout)

    def test_error_output_includes_code(self) -> None:
        """Error output should include error code."""
        config = """
slices:
  test:
    mode: broken
"""
        result = run_dryrun(config, use_json=False)
        
        # Should show error code in text output
        self.assertIn("INVALID_MODE", result.stdout)


class TestDryRunListStyleSlices(unittest.TestCase):
    """Tests for list-style slice definitions."""

    def test_list_style_slices_work(self) -> None:
        """Config with list-style slices should work."""
        config = """
slices:
  - name: slice_a
    mode: baseline
    cycles: 10
  - name: slice_b
    mode: rfl
    cycles: 20
"""
        result = run_dryrun(config, use_json=True)
        
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "OK")
        self.assertEqual(data["slices_found"], 2)


class TestDryRunNoSideEffects(unittest.TestCase):
    """Tests that dry-run validation has no side effects."""

    def test_no_files_created(self) -> None:
        """Dry-run should not create any files."""
        import os
        
        config = """
slices:
  test:
    mode: baseline
    cycles: 100
"""
        
        # Record files in current directory
        cwd = Path(__file__).parent.parent.parent.parent
        files_before = set(cwd.glob("*"))
        
        run_dryrun(config)
        
        files_after = set(cwd.glob("*"))
        new_files = files_after - files_before
        
        # Filter out any .pyc or __pycache__ that might be created
        new_files = {f for f in new_files if not str(f).endswith(".pyc")}
        new_files = {f for f in new_files if "__pycache__" not in str(f)}
        
        self.assertEqual(new_files, set(), f"Unexpected files created: {new_files}")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

