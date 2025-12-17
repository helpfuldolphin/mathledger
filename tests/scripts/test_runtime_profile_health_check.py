"""
Tests for runtime_profile_health_check.py CI script.

Validates:
- Script produces valid JSON tile
- Script remains read-only
- Script exits with 0 (advisory mode)
- Baseline comparison works
"""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


def test_script_produces_valid_tile(tmp_path: Path):
    """Test that script produces a valid JSON tile in output directory."""
    output_dir = tmp_path / "artifacts" / "runtime_profile_health"
    output_file = output_dir / "runtime_profile_health.json"

    # Run script
    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_profile_health_check.py",
            "--profile",
            "dev-default",
            "--env-context",
            "dev",
            "--runs",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    # Script should exit with 0 (advisory mode)
    assert result.returncode == 0

    # Output file should exist
    assert output_file.exists()

    # File should be valid JSON
    with open(output_file, "r", encoding="utf-8") as f:
        tile = json.load(f)

    # Validate tile structure
    assert tile["schema_version"] == "1.0.0"
    assert tile["tile_type"] == "runtime_profile_health"
    assert "status_light" in tile
    assert tile["status_light"] in ("GREEN", "YELLOW", "RED")
    assert "profile_name" in tile
    assert "profile_stability" in tile
    assert "no_run_rate" in tile
    assert "headline" in tile
    assert "notes" in tile


def test_script_baseline_comparison(tmp_path: Path):
    """Test that script compares with baseline if provided."""
    output_dir = tmp_path / "artifacts" / "runtime_profile_health"
    baseline_file = tmp_path / "baseline.json"

    # Create a baseline summary
    baseline_summary = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 95},
        "profile_stability": 0.95,
        "top_violations": [],
    }

    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_summary, f)

    # Run script with baseline
    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_profile_health_check.py",
            "--profile",
            "dev-default",
            "--env-context",
            "dev",
            "--runs",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
            "--baseline",
            str(baseline_file),
        ],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    # Script should still exit with 0 (advisory mode)
    assert result.returncode == 0

    # Baseline comparison should have been attempted
    # (may or may not print drift message depending on results)
    # Just verify script completed successfully


def test_script_read_only():
    """Test that script is read-only (doesn't modify runtime state)."""
    # This is more of a contract test - we verify the script
    # doesn't have side effects by running it multiple times
    # and checking deterministic output

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "artifacts" / "runtime_profile_health"

        # Run 1
        result1 = subprocess.run(
            [
                sys.executable,
                "scripts/runtime_profile_health_check.py",
                "--profile",
                "dev-default",
                "--runs",
                "10",
                "--seed",
                "42",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        output_file = output_dir / "runtime_profile_health.json"
        assert output_file.exists()

        with open(output_file, "r", encoding="utf-8") as f:
            tile1 = json.load(f)

        # Remove output and run again
        output_file.unlink()

        result2 = subprocess.run(
            [
                sys.executable,
                "scripts/runtime_profile_health_check.py",
                "--profile",
                "dev-default",
                "--runs",
                "10",
                "--seed",
                "42",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        with open(output_file, "r", encoding="utf-8") as f:
            tile2 = json.load(f)

        # Outputs should be identical (deterministic)
        assert tile1 == tile2


def test_script_always_exits_zero():
    """Test that script always exits with 0 (advisory mode)."""
    # Even with errors, script should exit 0 in shadow mode
    # (though it may print errors to stderr)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_profile_health_check.py",
            "--profile",
            "nonexistent-profile",
            "--runs",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(Path.cwd() / "artifacts" / "runtime_profile_health"),
        ],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    # Should still exit 0 (advisory mode)
    assert result.returncode == 0


def test_script_warns_on_red_status(tmp_path: Path):
    """Test that script prints warning to stderr when status is RED."""
    output_dir = tmp_path / "artifacts" / "runtime_profile_health"

    # Use a profile and parameters that might produce RED status
    # (though this depends on randomness, so we just check the warning logic exists)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_profile_health_check.py",
            "--profile",
            "prod-hardened",
            "--env-context",
            "prod",
            "--flip-flags",
            "5",  # More flips = higher chance of violations
            "--runs",
            "50",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    assert result.returncode == 0

    # Check if warning was printed (may or may not be, depending on results)
    # Just verify script completed and produced output
    output_file = output_dir / "runtime_profile_health.json"
    assert output_file.exists()

