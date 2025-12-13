"""
Phase X P3: First-Light E2E Integration Tests

End-to-end tests for the P3 First-Light synthetic experiment.

SHADOW MODE CONTRACT:
- All tests verify observation-only behavior
- Tests run full harness and validate artifact schemas
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from backend.topology.first_light import CycleLogEntry
from tests.factories.first_light_factories import (
    make_synthetic_raw_record,
    make_tda_window,
)

_FACTORY_RAW_PAYLOAD = make_synthetic_raw_record(1, seed=1)
_FACTORY_RAW_FOR_DC = dict(_FACTORY_RAW_PAYLOAD)
_FACTORY_RAW_FOR_DC.pop("abstained", None)
_FACTORY_RAW_RECORD = CycleLogEntry(**_FACTORY_RAW_FOR_DC).to_dict()
_EXPECTED_RAW_KEYS = set(_FACTORY_RAW_RECORD.keys())
_EXPECTED_RUNNER_KEYS = set(_FACTORY_RAW_RECORD["runner"].keys())
_EXPECTED_STATE_KEYS = set(_FACTORY_RAW_RECORD["usla_state"].keys())
_EXPECTED_GOV_KEYS = set(_FACTORY_RAW_RECORD["governance"].keys())
_EXPECTS_ABSTAINED_FIELD = True
_FACTORY_TDA_WINDOW = make_tda_window(window_index=0, seed=10)
_EXPECTED_TDA_METRIC_KEYS = {key.upper() for key in _FACTORY_TDA_WINDOW["trajectories"].keys()}

# Test fixtures
TEST_OUTPUT_DIR = Path("results/test_e2e")


@pytest.fixture(autouse=True)
def cleanup_test_dir():
    """Clean up test output directory before and after tests."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    yield
    # Leave output for debugging if test fails
    # if TEST_OUTPUT_DIR.exists():
    #     shutil.rmtree(TEST_OUTPUT_DIR)


class TestFirstLightE2EValidation:
    """E2E validation tests for P3 First-Light."""

    def test_harness_100_cycle_run(self) -> None:
        """Run harness with 100 cycles and verify all artifacts exist."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_harness.py",
                "--cycles", "100",
                "--seed", "42",
                "--window-size", "20",
                "--output-dir", str(TEST_OUTPUT_DIR),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        assert result.returncode == 0, f"Harness failed: {result.stderr}"

        # Find output directory (contains timestamp)
        output_dirs = list(TEST_OUTPUT_DIR.glob("fl_*"))
        assert len(output_dirs) == 1, f"Expected 1 output dir, found {len(output_dirs)}"
        output_dir = output_dirs[0]

        # Verify all 6 artifacts exist
        expected_files = [
            "synthetic_raw.jsonl",
            "stability_report.json",
            "red_flag_matrix.json",
            "metrics_windows.json",
            "tda_metrics.json",
            "run_config.json",
        ]

        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Missing artifact: {filename}"
            assert filepath.stat().st_size > 0, f"Empty artifact: {filename}"

    def test_synthetic_raw_has_correct_records(self) -> None:
        """Verify synthetic_raw.jsonl has correct number of records."""
        cycles = 100
        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_harness.py",
                "--cycles", str(cycles),
                "--seed", "42",
                "--output-dir", str(TEST_OUTPUT_DIR),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        assert result.returncode == 0

        output_dir = list(TEST_OUTPUT_DIR.glob("fl_*"))[0]
        raw_path = output_dir / "synthetic_raw.jsonl"

        with open(raw_path) as f:
            lines = f.readlines()

        assert len(lines) == cycles, f"Expected {cycles} records, got {len(lines)}"

        expected_keys = _EXPECTED_RAW_KEYS
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert expected_keys.issubset(record.keys()), f"Record {i} missing required keys"
            assert _EXPECTED_RUNNER_KEYS.issubset(record["runner"].keys()), f"Runner block incomplete for record {i}"
            assert _EXPECTED_STATE_KEYS.issubset(record["usla_state"].keys()), f"USLA state block incomplete for record {i}"
            assert _EXPECTED_GOV_KEYS.issubset(record["governance"].keys()), f"Governance block incomplete for record {i}"
            if _EXPECTS_ABSTAINED_FIELD:
                assert "abstained" in record, f"Record {i} missing abstained flag"

    def test_stability_report_has_criteria(self) -> None:
        """Verify stability_report.json has criteria evaluation."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_harness.py",
                "--cycles", "100",
                "--seed", "42",
                "--output-dir", str(TEST_OUTPUT_DIR),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        assert result.returncode == 0

        output_dir = list(TEST_OUTPUT_DIR.glob("fl_*"))[0]
        report_path = output_dir / "stability_report.json"

        with open(report_path) as f:
            report = json.load(f)

        assert "schema_version" in report
        assert report["schema_version"] == "1.0.0"
        assert "criteria_evaluation" in report
        assert "all_passed" in report["criteria_evaluation"]
        assert "criteria" in report["criteria_evaluation"]
        assert len(report["criteria_evaluation"]["criteria"]) > 0

    def test_tda_metrics_has_windows(self) -> None:
        """Verify tda_metrics.json has TDA snapshots."""
        window_size = 20
        cycles = 100
        expected_windows = cycles // window_size  # 5 windows

        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_harness.py",
                "--cycles", str(cycles),
                "--seed", "42",
                "--window-size", str(window_size),
                "--output-dir", str(TEST_OUTPUT_DIR),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        assert result.returncode == 0

        output_dir = list(TEST_OUTPUT_DIR.glob("fl_*"))[0]
        tda_path = output_dir / "tda_metrics.json"

        with open(tda_path) as f:
            tda = json.load(f)

        assert "total_windows" in tda
        assert tda["total_windows"] == expected_windows
        assert "metrics" in tda
        assert len(tda["metrics"]) == expected_windows

        # Verify each TDA metric has required fields
        for i, metric in enumerate(tda["metrics"]):
            assert "window_index" in metric, f"TDA metric {i} missing window_index"
            for key in _EXPECTED_TDA_METRIC_KEYS:
                assert key in metric, f"TDA metric {i} missing {key}"

            # Verify ranges
            assert 0.0 <= metric["SNS"] <= 1.0
            assert 0.0 <= metric["PCS"] <= 1.0
            assert 0.0 <= metric["DRS"] <= 1.0
            assert 0.0 <= metric["HSS"] <= 1.0

    def test_deterministic_with_seed(self) -> None:
        """Verify runs with same seed produce identical results."""
        for i in range(2):
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/usla_first_light_harness.py",
                    "--cycles", "50",
                    "--seed", "12345",
                    "--output-dir", f"{TEST_OUTPUT_DIR}/run{i}",
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent),
            )
            assert result.returncode == 0, f"Run {i} failed"

        # Compare stability reports
        run0_dirs = list(Path(f"{TEST_OUTPUT_DIR}/run0").glob("fl_*"))
        run1_dirs = list(Path(f"{TEST_OUTPUT_DIR}/run1").glob("fl_*"))

        assert len(run0_dirs) == 1 and len(run1_dirs) == 1

        with open(run0_dirs[0] / "stability_report.json") as f:
            report0 = json.load(f)
        with open(run1_dirs[0] / "stability_report.json") as f:
            report1 = json.load(f)

        # Metrics should be identical
        assert report0["metrics"]["success_rate"] == report1["metrics"]["success_rate"]
        assert report0["metrics"]["rsi"]["mean"] == report1["metrics"]["rsi"]["mean"]

    def test_dry_run_does_not_execute(self) -> None:
        """Verify dry run doesn't create output files."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_harness.py",
                "--cycles", "100",
                "--seed", "42",
                "--output-dir", str(TEST_OUTPUT_DIR),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

        # No directories should be created
        output_dirs = list(TEST_OUTPUT_DIR.glob("fl_*"))
        assert len(output_dirs) == 0, "Dry run should not create output"


class TestFirstLight1000CycleSmoke:
    """Smoke test with 1000 cycles (typical First-Light run)."""

    @pytest.mark.slow
    def test_1000_cycle_full_run(self) -> None:
        """Full 1000-cycle run with typical configuration."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/usla_first_light_harness.py",
                "--cycles", "1000",
                "--seed", "42",
                "--window-size", "50",
                "--output-dir", str(TEST_OUTPUT_DIR),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        assert result.returncode == 0, f"1000-cycle run failed: {result.stderr}"

        output_dir = list(TEST_OUTPUT_DIR.glob("fl_*"))[0]

        # Verify file sizes are reasonable
        raw_path = output_dir / "synthetic_raw.jsonl"
        assert raw_path.stat().st_size > 100000  # Should be substantial

        # Verify 20 windows (1000/50)
        with open(output_dir / "tda_metrics.json") as f:
            tda = json.load(f)
        assert tda["total_windows"] == 20

        # Verify stability report
        with open(output_dir / "stability_report.json") as f:
            report = json.load(f)
        assert report["timing"]["cycles_completed"] == 1000
