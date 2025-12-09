"""
PHASE II — NOT USED IN PHASE I

Tests for the U2 Runtime Introspection CLI
==========================================

Tests for experiments/u2_runtime_inspect.py including:
- --show-contract flag
- --show-error-kinds flag
- JSON output modes
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestRuntimeInspectContract(unittest.TestCase):
    """Tests for --show-contract functionality."""

    def test_show_contract_returns_success(self) -> None:
        """--show-contract should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "experiments/u2_runtime_inspect.py", "--show-contract"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

    def test_show_contract_json_is_valid(self) -> None:
        """--show-contract --json should return valid JSON."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-contract",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        
        # Parse JSON
        data = json.loads(result.stdout)
        
        # Verify expected fields
        self.assertIn("version", data)
        self.assertIn("symbols", data)
        self.assertIn("modules", data)
        self.assertIn("invariants", data)
        self.assertIn("guarantees", data)

    def test_show_contract_json_has_version(self) -> None:
        """Contract JSON should include runtime version."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-contract",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        data = json.loads(result.stdout)
        self.assertEqual(data["version"], "1.5.0")

    def test_show_contract_json_has_all_symbols(self) -> None:
        """Contract JSON should list all exported symbols."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-contract",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        data = json.loads(result.stdout)
        
        symbols = data["symbols"]
        
        # Check key symbols are present
        self.assertIn("execute_cycle", symbols)
        self.assertIn("CycleState", symbols)
        self.assertIn("RuntimeErrorKind", symbols)
        self.assertIn("TraceWriter", symbols)
        self.assertIn("generate_seed_schedule", symbols)

    def test_show_contract_json_has_invariants(self) -> None:
        """Contract JSON should document invariants."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-contract",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        data = json.loads(result.stdout)
        
        invariants = data["invariants"]
        
        self.assertIn("INV-RUN-1", invariants)
        self.assertIn("INV-RUN-2", invariants)
        self.assertIn("INV-RUN-3", invariants)
        self.assertIn("INV-RUN-4", invariants)


class TestRuntimeInspectErrorKinds(unittest.TestCase):
    """Tests for --show-error-kinds functionality."""

    def test_show_error_kinds_returns_success(self) -> None:
        """--show-error-kinds should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "experiments/u2_runtime_inspect.py", "--show-error-kinds"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

    def test_show_error_kinds_json_is_valid(self) -> None:
        """--show-error-kinds --json should return valid JSON."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-error-kinds",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        self.assertIn("error_kinds", data)
        self.assertIn("total_count", data)
        self.assertIn("module", data)

    def test_show_error_kinds_includes_all_kinds(self) -> None:
        """Error kinds JSON should include all RuntimeErrorKind values."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-error-kinds",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        data = json.loads(result.stdout)
        
        kind_names = [k["name"] for k in data["error_kinds"]]
        
        expected_kinds = {
            "SUBPROCESS",
            "JSON_DECODE",
            "FILE_NOT_FOUND",
            "VALIDATION",
            "TIMEOUT",
            "UNKNOWN",
        }
        
        for kind in expected_kinds:
            self.assertIn(kind, kind_names)

    def test_show_error_kinds_has_descriptions(self) -> None:
        """Each error kind should have a description."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-error-kinds",
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        data = json.loads(result.stdout)
        
        for kind in data["error_kinds"]:
            self.assertIn("description", kind)
            self.assertIsInstance(kind["description"], str)
            self.assertGreater(len(kind["description"]), 10)


class TestRuntimeInspectConfigMode(unittest.TestCase):
    """Tests for configuration inspection mode."""

    def test_missing_args_shows_error(self) -> None:
        """Missing required args should show helpful error."""
        result = subprocess.run(
            [sys.executable, "experiments/u2_runtime_inspect.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--slice", result.stderr)

    def test_config_inspect_requires_all_args(self) -> None:
        """Config inspection requires --slice, --mode, --cycles."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--slice",
                "test",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_config_inspect_with_all_args_succeeds(self) -> None:
        """Config inspection with all args should succeed."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--slice",
                "test_slice",
                "--mode",
                "baseline",
                "--cycles",
                "10",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")


class TestRuntimeInspectMutualExclusion(unittest.TestCase):
    """Tests for mutually exclusive options."""

    def test_contract_and_error_kinds_mutually_exclusive(self) -> None:
        """--show-contract and --show-error-kinds cannot be used together."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/u2_runtime_inspect.py",
                "--show-contract",
                "--show-error-kinds",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

