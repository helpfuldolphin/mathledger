# PHASE II — NOT USED IN PHASE I
"""
Unit tests for U2 Calibration Fire Harness.

Tests the calibration mode functionality in experiments/u2_calibration.py
and experiments/run_uplift_u2.py.

These tests verify:
- Calibration mode runs both baseline and RFL
- Determinism checks work correctly
- Schema validation catches errors
- JSON summary is correctly generated
- Regular mode still functions properly
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestU2CalibrationHelpers:
    """Tests for u2_calibration.py helper functions."""

    def test_validate_schema_valid_record(self):
        """Test that valid records pass schema validation."""
        from experiments.u2_calibration import validate_schema

        record = {
            "cycle": 0,
            "slice": "arithmetic_simple",
            "mode": "baseline",
            "seed": 42,
            "item": "1 + 1",
            "result": "2",
            "success": True,
            "label": "PHASE II — NOT USED IN PHASE I",
        }

        errors = validate_schema(record)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_validate_schema_missing_field(self):
        """Test that missing required fields are detected."""
        from experiments.u2_calibration import validate_schema

        record = {
            "cycle": 0,
            "slice": "arithmetic_simple",
            # "mode" is missing
            "seed": 42,
            "item": "1 + 1",
            "result": "2",
            "success": True,
            "label": "PHASE II — NOT USED IN PHASE I",
        }

        errors = validate_schema(record)
        assert len(errors) == 1
        assert "mode" in errors[0]

    def test_validate_schema_wrong_type(self):
        """Test that wrong field types are detected."""
        from experiments.u2_calibration import validate_schema

        record = {
            "cycle": "not_an_int",  # Should be int
            "slice": "arithmetic_simple",
            "mode": "baseline",
            "seed": 42,
            "item": "1 + 1",
            "result": "2",
            "success": True,
            "label": "PHASE II — NOT USED IN PHASE I",
        }

        errors = validate_schema(record)
        assert len(errors) >= 1
        assert any("cycle" in e for e in errors)

    def test_validate_manifest_valid(self):
        """Test that valid manifests pass validation."""
        from experiments.u2_calibration import validate_manifest

        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "slice": "arithmetic_simple",
            "mode": "baseline",
            "cycles": 10,
            "initial_seed": 42,
            "slice_config_hash": "abc123",
            "ht_series_hash": "def456",
            "outputs": {
                "results": "path/to/results.jsonl",
                "manifest": "path/to/manifest.json",
            },
        }

        errors = validate_manifest(manifest)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_validate_manifest_missing_field(self):
        """Test that missing manifest fields are detected."""
        from experiments.u2_calibration import validate_manifest

        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "slice": "arithmetic_simple",
            # "mode" is missing
            "cycles": 10,
            "initial_seed": 42,
            "slice_config_hash": "abc123",
            "ht_series_hash": "def456",
            "outputs": {
                "results": "path/to/results.jsonl",
                "manifest": "path/to/manifest.json",
            },
        }

        errors = validate_manifest(manifest)
        assert len(errors) >= 1
        assert any("mode" in e for e in errors)

    def test_count_successes(self):
        """Test counting successes in a log file."""
        from experiments.u2_calibration import count_successes

        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.jsonl"
            records = [
                {"success": True},
                {"success": False},
                {"success": True},
                {"success": True},
            ]
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            count = count_successes(log_path)
            assert count == 3

    def test_count_successes_empty_file(self):
        """Test counting successes in empty file returns 0."""
        from experiments.u2_calibration import count_successes

        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "empty.jsonl"
            log_path.touch()

            count = count_successes(log_path)
            assert count == 0

    def test_count_successes_missing_file(self):
        """Test counting successes with missing file returns 0."""
        from experiments.u2_calibration import count_successes

        count = count_successes(Path("/nonexistent/path.jsonl"))
        assert count == 0


class TestU2CalibrationIntegration:
    """Integration tests for calibration mode via CLI."""

    def test_calibration_mode_runs_both_modes(self):
        """Test that calibration mode runs both baseline and RFL."""
        with TemporaryDirectory() as tmpdir:
            # Run calibration
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                    "--slice", "arithmetic_simple",
                    "--seed", "12345",
                    "--calibration",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            assert result.returncode == 0, f"Calibration failed: {result.stderr}"
            assert "CALIBRATION FIRE HARNESS" in result.stdout
            assert "Overall status: passed" in result.stdout

            # Verify both log files were created
            out_dir = PROJECT_ROOT / "results" / "uplift_u2" / "calibration" / "arithmetic_simple"
            assert (out_dir / "uplift_u2_arithmetic_simple_baseline.jsonl").exists()
            assert (out_dir / "uplift_u2_arithmetic_simple_rfl.jsonl").exists()
            assert (out_dir / "calibration_summary.json").exists()

    def test_calibration_forces_10_cycles_default(self):
        """Test that calibration uses 10 cycles by default."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                "--slice", "arithmetic_simple",
                "--seed", "999",
                "--calibration",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0
        assert "Cycles: 10" in result.stdout

    def test_calibration_allows_custom_cycles(self):
        """Test that calibration mode accepts custom cycle count."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                "--slice", "arithmetic_simple",
                "--seed", "888",
                "--cycles", "5",
                "--calibration",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0
        assert "Cycles: 5" in result.stdout

    def test_calibration_verifies_determinism(self):
        """Test that calibration checks and reports determinism."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                "--slice", "arithmetic_simple",
                "--seed", "777",
                "--calibration",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0
        assert "Baseline deterministic: True" in result.stdout
        assert "RFL deterministic: True" in result.stdout

    def test_calibration_summary_json_format(self):
        """Test that calibration summary JSON has correct format."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                "--slice", "arithmetic_simple",
                "--seed", "666",
                "--calibration",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0

        summary_path = (
            PROJECT_ROOT
            / "results"
            / "uplift_u2"
            / "calibration"
            / "arithmetic_simple"
            / "calibration_summary.json"
        )
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        # Check required fields
        assert summary["calibration_mode"] is True
        assert summary["slice"] == "arithmetic_simple"
        assert "baseline" in summary
        assert "rfl" in summary
        assert "overall_status" in summary
        assert summary["baseline"]["determinism"]["deterministic"] is True
        assert summary["rfl"]["determinism"]["deterministic"] is True

    def test_regular_mode_still_works(self):
        """Test that regular (non-calibration) mode still functions."""
        with TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                    "--slice", "arithmetic_simple",
                    "--seed", "555",
                    "--cycles", "3",
                    "--mode", "baseline",
                    "--out", tmpdir,
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            assert result.returncode == 0
            assert (Path(tmpdir) / "uplift_u2_arithmetic_simple_baseline.jsonl").exists()
            assert (Path(tmpdir) / "uplift_u2_manifest_arithmetic_simple_baseline.json").exists()

    def test_regular_mode_requires_cycles(self):
        """Test that regular mode requires --cycles argument."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                "--slice", "arithmetic_simple",
                "--seed", "444",
                "--mode", "baseline",
                "--out", "/tmp/test",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 2
        assert "--cycles is required" in result.stderr

    def test_regular_mode_requires_mode(self):
        """Test that regular mode requires --mode argument."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                "--slice", "arithmetic_simple",
                "--seed", "333",
                "--cycles", "5",
                "--out", "/tmp/test",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 2
        assert "--mode is required" in result.stderr


class TestU2CalibrationDeterminism:
    """Tests for determinism verification in calibration."""

    def test_same_seed_produces_same_results(self):
        """Test that same seed produces identical results."""
        # Run calibration twice with same seed
        results = []
        for _ in range(2):
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                    "--slice", "arithmetic_simple",
                    "--seed", "123456",
                    "--calibration",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            summary_path = (
                PROJECT_ROOT
                / "results"
                / "uplift_u2"
                / "calibration"
                / "arithmetic_simple"
                / "calibration_summary.json"
            )
            with open(summary_path) as f:
                results.append(json.load(f))

        # Compare hashes (excluding timestamp)
        assert (
            results[0]["baseline"]["determinism"]["original_hash"]
            == results[1]["baseline"]["determinism"]["original_hash"]
        )
        assert (
            results[0]["rfl"]["determinism"]["original_hash"]
            == results[1]["rfl"]["determinism"]["original_hash"]
        )

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        hashes = []
        for seed in [111, 222]:
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "experiments" / "run_uplift_u2.py"),
                    "--slice", "arithmetic_simple",
                    "--seed", str(seed),
                    "--calibration",
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

            summary_path = (
                PROJECT_ROOT
                / "results"
                / "uplift_u2"
                / "calibration"
                / "arithmetic_simple"
                / "calibration_summary.json"
            )
            with open(summary_path) as f:
                summary = json.load(f)
            hashes.append(summary["baseline"]["determinism"]["original_hash"])

        # Hashes should differ for different seeds
        assert hashes[0] != hashes[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
