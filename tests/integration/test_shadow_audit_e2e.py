#!/usr/bin/env python3
"""
Shadow Audit E2E Integration Tests (v0.1)

Aligned with CANONICAL CONTRACT: docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md

FROZEN CLI:
  --input INPUT    (required)
  --output OUTPUT  (required)
  --seed SEED      (optional)
  --verbose, -v    (optional)
  --dry-run        (optional)

FROZEN EXIT CODES:
  0 = OK (script completed - success or warnings)
  1 = FATAL (missing input, crash, exception)
  2 = RESERVED (unused in v0.1)

SHADOW MODE CONTRACT:
  - mode="SHADOW" in all outputs
  - schema_version="1.0.0"
  - shadow_mode_compliance.no_enforcement = true
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def find_script_path() -> Path:
    """Find run_shadow_audit.py from repo root."""
    test_file = Path(__file__).resolve()
    repo_root = test_file.parent.parent.parent
    return repo_root / "scripts" / "run_shadow_audit.py"


SCRIPT_PATH = find_script_path()
PROJECT_ROOT = SCRIPT_PATH.parent.parent if SCRIPT_PATH.exists() else Path.cwd()
TEST_OUTPUT_DIR = Path("results/test_shadow_audit_e2e")


def _script_has_canonical_cli() -> bool:
    """Check if script exists and uses canonical CLI flags."""
    if not SCRIPT_PATH.exists():
        return False
    # Run --help and check for canonical flags
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "--input" in result.stdout and "--output" in result.stdout
    except Exception:
        return False


def run_script(args: list, capture: bool = True) -> subprocess.CompletedProcess:
    """Run the shadow audit script with given arguments."""
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    return subprocess.run(cmd, capture_output=capture, text=True, cwd=str(PROJECT_ROOT))


def assert_shadow_mode_markers(data: Dict[str, Any]) -> None:
    """Assert SHADOW mode markers per canonical contract."""
    assert data.get("mode") == "SHADOW", f"INV-01 violated: mode={data.get('mode')}"
    assert data.get("schema_version") == "1.0.0", f"INV-02 violated: schema_version={data.get('schema_version')}"


@pytest.fixture(autouse=True)
def cleanup_test_dir():
    """Clean up test output directory before and after tests."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    yield
    # Leave output for debugging


def create_mock_shadow_logs(input_dir: Path, num_cycles: int = 10) -> Path:
    """Create minimal shadow log artifacts per canonical input format."""
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create shadow_log_test.jsonl (canonical input format)
    log_file = input_dir / "shadow_log_test.jsonl"
    lines = []

    # Header
    lines.append(json.dumps({
        "_header": True,
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "runner": "test",
        "run_id": "test_run",
    }))

    # Cycles
    for i in range(num_cycles):
        lines.append(json.dumps({
            "cycle": i,
            "runner": "test",
            "input": {"value": i},
            "state": {"H": 0.5 + (i * 0.01), "rho": 0.8},
            "hard_ok": True,
            "in_safe_region": True,
            "real_blocked": False,
            "sim_blocked": False,
            "governance_aligned": True,
        }))

    # Footer
    lines.append(json.dumps({
        "_footer": True,
        "total_cycles": num_cycles,
    }))

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return input_dir


# =============================================================================
# E2E Tests Aligned with Canonical Contract
# =============================================================================

# Cache the canonical CLI check result at module load
_CANONICAL_CLI_AVAILABLE = _script_has_canonical_cli()


def _skip_if_no_canonical_cli() -> None:
    """Skip test if script missing or uses non-canonical CLI."""
    if not SCRIPT_PATH.exists():
        pytest.skip("Script not deployed yet")
    if not _CANONICAL_CLI_AVAILABLE:
        pytest.skip("Script exists but uses non-canonical CLI (transition state)")


class TestShadowAuditE2E:
    """E2E tests for shadow audit orchestrator (canonical contract)."""

    @pytest.mark.integration
    def test_e2e_basic_run(self) -> None:
        """Basic run with --input and --output produces valid run_summary.json."""
        _skip_if_no_canonical_cli()

        # Setup
        input_dir = TEST_OUTPUT_DIR / "fixtures" / "input"
        create_mock_shadow_logs(input_dir, num_cycles=50)
        output_dir = TEST_OUTPUT_DIR / "run1"

        # Run with canonical flags
        result = run_script([
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--seed", "42",
        ])

        # Exit 0 per INV-06 (successful completion)
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stderr}"

        # Find output directory (run_id subdirectory)
        output_dirs = list(output_dir.glob("sha_42_*")) + list(output_dir.glob("run_*"))
        assert len(output_dirs) >= 1, f"No output directory found in {output_dir}"
        run_output = output_dirs[0]

        # Verify run_summary.json exists (MC-03)
        summary_path = run_output / "run_summary.json"
        assert summary_path.exists(), "run_summary.json not created (MC-03 violation)"

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        # Verify SHADOW MODE contract (INV-01, INV-02)
        assert_shadow_mode_markers(summary)

        # Verify shadow_mode_compliance (MC-06)
        compliance = summary.get("shadow_mode_compliance", {})
        assert compliance.get("no_enforcement") is True, "MC-06 violated"

    @pytest.mark.integration
    def test_e2e_deterministic_seed(self) -> None:
        """Same --seed produces identical run_id (INV-04)."""
        _skip_if_no_canonical_cli()

        # Setup
        input_dir = TEST_OUTPUT_DIR / "fixtures_det" / "input"
        create_mock_shadow_logs(input_dir)

        run_ids = []
        for i in range(2):
            output_dir = TEST_OUTPUT_DIR / f"det_run{i}"

            result = run_script([
                "--input", str(input_dir),
                "--output", str(output_dir),
                "--seed", "42",
            ])

            assert result.returncode == 0

            # Find run_id from output directory name
            output_dirs = list(output_dir.glob("sha_42_*")) + list(output_dir.glob("run_*"))
            assert len(output_dirs) >= 1
            run_ids.append(output_dirs[0].name)

        # INV-04: Same seed produces identical run_id
        assert run_ids[0] == run_ids[1], f"INV-04 violated: {run_ids[0]} != {run_ids[1]}"

    @pytest.mark.integration
    def test_e2e_dry_run_no_files(self) -> None:
        """--dry-run creates no files (INV-05)."""
        _skip_if_no_canonical_cli()

        output_dir = TEST_OUTPUT_DIR / "dry_run"

        result = run_script([
            "--input", "results/first_light",  # May not exist, but --dry-run validates only
            "--output", str(output_dir),
            "--dry-run",
        ])

        # --dry-run should exit 0 (validation passed or reported)
        assert result.returncode == 0, f"Dry run failed: {result.stderr}"

        # INV-05: No files created
        if output_dir.exists():
            files = list(output_dir.glob("**/*"))
            assert len(files) == 0, f"INV-05 violated: files created during dry-run: {files}"

    @pytest.mark.integration
    def test_e2e_missing_input_exit_1(self) -> None:
        """Missing input directory exits 1 (INV-07)."""
        _skip_if_no_canonical_cli()

        result = run_script([
            "--input", "/nonexistent/path/xyz123",
            "--output", str(TEST_OUTPUT_DIR / "missing_input"),
        ])

        # INV-07: Missing input → exit 1
        assert result.returncode == 1, f"INV-07 violated: expected exit 1, got {result.returncode}"

    @pytest.mark.integration
    def test_e2e_empty_input_exit_0_warn(self) -> None:
        """Empty input exits 0 with status WARN (INV-06)."""
        _skip_if_no_canonical_cli()

        # Create empty input directory
        input_dir = TEST_OUTPUT_DIR / "empty_input"
        input_dir.mkdir(parents=True)

        output_dir = TEST_OUTPUT_DIR / "empty_output"

        result = run_script([
            "--input", str(input_dir),
            "--output", str(output_dir),
        ])

        # INV-06: Empty input → exit 0
        assert result.returncode == 0, f"INV-06 violated: expected exit 0, got {result.returncode}"

        # Verify status is WARN
        output_dirs = list(output_dir.glob("*"))
        if output_dirs:
            run_output = output_dirs[0]
            summary_path = run_output / "run_summary.json"
            if summary_path.exists():
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
                assert summary.get("status") == "WARN", f"INV-06: expected status=WARN, got {summary.get('status')}"

    @pytest.mark.integration
    def test_e2e_required_artifacts(self) -> None:
        """Both run_summary.json and first_light_status.json are created (MC-03)."""
        _skip_if_no_canonical_cli()

        # Setup
        input_dir = TEST_OUTPUT_DIR / "fixtures_artifacts" / "input"
        create_mock_shadow_logs(input_dir)
        output_dir = TEST_OUTPUT_DIR / "artifacts_test"

        result = run_script([
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--seed", "42",
        ])

        assert result.returncode == 0

        output_dirs = list(output_dir.glob("sha_42_*")) + list(output_dir.glob("run_*"))
        assert len(output_dirs) >= 1
        run_output = output_dirs[0]

        # MC-03: Both required files exist
        assert (run_output / "run_summary.json").exists(), "run_summary.json missing (MC-03)"
        assert (run_output / "first_light_status.json").exists(), "first_light_status.json missing (MC-03)"

        # Verify both have SHADOW markers
        for filename in ["run_summary.json", "first_light_status.json"]:
            with open(run_output / filename, encoding="utf-8") as f:
                data = json.load(f)
            assert_shadow_mode_markers(data)

    @pytest.mark.integration
    def test_e2e_help_output(self) -> None:
        """--help shows canonical flags."""
        _skip_if_no_canonical_cli()

        result = run_script(["--help"])

        assert result.returncode == 0

        # Verify canonical flags are documented
        canonical_flags = ["--input", "--output", "--seed", "--verbose", "--dry-run"]
        for flag in canonical_flags:
            assert flag in result.stdout, f"Canonical flag {flag} not in help output"

    @pytest.mark.integration
    def test_e2e_verbose_mode(self) -> None:
        """--verbose produces extended output."""
        _skip_if_no_canonical_cli()

        input_dir = TEST_OUTPUT_DIR / "fixtures_verbose" / "input"
        create_mock_shadow_logs(input_dir)
        output_dir = TEST_OUTPUT_DIR / "verbose_test"

        result = run_script([
            "--input", str(input_dir),
            "--output", str(output_dir),
            "-v",
        ])

        assert result.returncode == 0
        # Verbose mode should produce stderr output
        # (stdout may also have output depending on implementation)


# =============================================================================
# Sentinel Test: Frozen CLI Flags
# =============================================================================

@pytest.mark.integration
def test_sentinel_frozen_cli_flags() -> None:
    """
    SENTINEL: Fail if script exists but uses non-canonical CLI flags.

    FROZEN flags per canonical contract:
      --input, --output, --seed, --verbose/-v, --dry-run, --help/-h

    FORBIDDEN (not in v0.1 spec):
      --p3-dir, --p4-dir, --output-dir, --deterministic, subcommands
    """
    if not SCRIPT_PATH.exists():
        pytest.skip("Script not deployed yet")

    # Get help output
    result = run_script(["--help"])
    assert result.returncode == 0

    help_text = result.stdout.lower()

    # Canonical flags MUST be present
    canonical_required = ["--input", "--output"]
    canonical_optional = ["--seed", "--verbose", "--dry-run"]

    for flag in canonical_required:
        assert flag in help_text, f"SENTINEL FAIL: Required canonical flag {flag} missing from help"

    for flag in canonical_optional:
        assert flag in help_text, f"SENTINEL FAIL: Optional canonical flag {flag} missing from help"

    # Non-canonical flags MUST NOT be present
    forbidden_flags = ["--p3-dir", "--p4-dir", "--output-dir", "--deterministic"]
    for flag in forbidden_flags:
        assert flag not in help_text, \
            f"SENTINEL FAIL: Non-canonical flag {flag} found in help (violates frozen CLI)"

    # Subcommands MUST NOT be present
    forbidden_subcommands = ["audit", "demo", "verify", "status"]
    for subcmd in forbidden_subcommands:
        # Check for subcommand patterns (word boundary detection)
        if f" {subcmd} " in help_text or f"\n{subcmd}\n" in help_text:
            pytest.fail(f"SENTINEL FAIL: Subcommand '{subcmd}' found (canonical is flags-only)")
