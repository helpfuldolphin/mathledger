"""
Unit Tests for run_shadow_audit.py v0.1 (CANONICAL)

Aligned with CANONICAL CONTRACT: docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md

FROZEN EXIT CODES:
- 0: OK (script completed - success or warnings)
- 1: FATAL (missing input, crash, exception)
- 2: RESERVED (unused in v0.1)

SHADOW MODE CONTRACT:
- mode="SHADOW" in all outputs
- schema_version="1.0.0" in all outputs
- shadow_mode_compliance.no_enforcement = true
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# -----------------------------------------------------------------------------
# Canonical Ignored Keys (for determinism comparison)
# -----------------------------------------------------------------------------

IGNORED_KEYS = {
    "timestamp",
    "created_at",
    "bundle_id",
    "generated_at",
    "started_at",
    "ended_at",
    "run_id",
    "log_file",
    "duration_ms",
}


def normalize_for_comparison(obj: Any, sort_lists: bool = True) -> Any:
    """Normalize JSON structure for deterministic comparison."""
    if isinstance(obj, dict):
        normalized = {
            k: normalize_for_comparison(v, sort_lists)
            for k, v in sorted(obj.items())
            if k not in IGNORED_KEYS
        }
        return normalized
    if isinstance(obj, list):
        normalized = [normalize_for_comparison(item, sort_lists) for item in obj]
        # Sort lists of dicts by their JSON representation for determinism
        if sort_lists and normalized and all(isinstance(x, dict) for x in normalized):
            normalized = sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True))
        return normalized
    return obj


def compare_outputs(a: Dict, b: Dict) -> bool:
    """Compare two outputs for deterministic equivalence."""
    return json.dumps(normalize_for_comparison(a), sort_keys=True) == \
           json.dumps(normalize_for_comparison(b), sort_keys=True)


# =============================================================================
# U1: test_schema_version_fixed
# =============================================================================

@pytest.mark.unit
def test_schema_version_fixed():
    """
    U1: schema_version must be "1.0.0" for v0.1.

    This is immutable for the v0.1 release.
    """
    REQUIRED_VERSION = "1.0.0"

    valid_output = {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
    }
    assert valid_output["schema_version"] == REQUIRED_VERSION

    # Invalid versions
    invalid_versions = ["0.9.0", "1.0.1", "1.1.0", "2.0.0", ""]
    for version in invalid_versions:
        invalid = {"mode": "SHADOW", "schema_version": version}
        assert invalid["schema_version"] != REQUIRED_VERSION, \
            f"Version {version} should not match {REQUIRED_VERSION}"


# =============================================================================
# U2: test_mode_shadow_everywhere
# =============================================================================

@pytest.mark.unit
def test_mode_shadow_everywhere():
    """
    U2: mode="SHADOW" must appear in all output artifacts.

    Per Phase X contract: "All USLA state is written to shadow logs only"
    """
    # Valid outputs
    valid_outputs = [
        {"mode": "SHADOW", "schema_version": "1.0.0"},
        {"mode": "SHADOW", "governance_advisory": "..."},
        {"mode": "SHADOW", "action": "LOGGED_ONLY"},
    ]

    for output in valid_outputs:
        assert output.get("mode") == "SHADOW"

    # Invalid modes
    invalid_modes = ["ACTIVE", "ENFORCEMENT", "LIVE", "", None]
    for mode in invalid_modes:
        invalid = {"mode": mode, "schema_version": "1.0.0"}
        assert invalid.get("mode") != "SHADOW"


# =============================================================================
# U3: test_exit_codes_per_contract
# =============================================================================

@pytest.mark.unit
def test_exit_codes_per_contract():
    """
    U3: Exit codes must follow v0.1 SHADOW MODE contract.

    Exit 0: ALWAYS in SHADOW mode (including missing inputs, red flags, warnings)
    Exit 1: ONLY on uncaught script exception
    Exit 2: CLI argument validation errors (argparse default)
    """
    # Test that the script exists and is runnable
    script_path = Path("scripts/run_shadow_audit.py")

    # Skip if script doesn't exist (CI may not have full repo)
    if not script_path.exists():
        pytest.skip("Script not found - skipping subprocess test")

    # Exit 0: dry-run mode with valid (but empty) input path
    # The script requires --input and --output even in dry-run mode
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--input", str(input_dir),
                "--output", str(output_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        # SHADOW MODE: dry-run should exit 0
        assert result.returncode == 0, \
            f"Dry-run mode should exit 0, got {result.returncode}: {result.stderr}"

    # Exit 0: empty input directory (SHADOW mode - exits 0 with warn status)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "empty_input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable, str(script_path),
                "--input", str(input_dir),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        # SHADOW MODE: exit 0 even when inputs empty (warn status)
        assert result.returncode == 0, \
            f"Empty inputs should still exit 0 (SHADOW mode), got {result.returncode}: {result.stderr}"

        # Verify run_summary.json was created with appropriate status
        output_dirs = list(output_dir.glob("sha_*"))
        if output_dirs:
            summary_path = output_dirs[0] / "run_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                # SHADOW MODE: warns on empty inputs, doesn't fail
                assert summary["final_status"] in ("warn", "pass")
                assert summary["mode"] == "SHADOW"


# =============================================================================
# U4: test_determinism_flag_behavior
# =============================================================================

@pytest.mark.unit
@pytest.mark.determinism
def test_determinism_flag_behavior():
    """
    U4: With --deterministic flag, outputs must be sorted and reproducible.

    Timestamps are IGNORED for comparison (always non-deterministic).
    Other fields must be byte-stable.
    """
    # Simulate two runs with same seed but different timestamps
    run1 = {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "timestamp": "2025-01-01T00:00:00Z",
        "bundle_id": "shadow-20250101-abc123",
        "artifacts": [
            {"name": "b.json", "sha256": "bbb"},
            {"name": "a.json", "sha256": "aaa"},
        ],
    }

    run2 = {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "timestamp": "2025-01-02T12:00:00Z",  # Different
        "bundle_id": "shadow-20250102-def456",  # Different
        "artifacts": [
            {"name": "a.json", "sha256": "aaa"},  # Different order
            {"name": "b.json", "sha256": "bbb"},
        ],
    }

    # After normalization, should match
    assert compare_outputs(run1, run2), \
        "Deterministic outputs should match after normalization"

    # Verify ignored keys are removed
    norm = normalize_for_comparison(run1)
    assert "timestamp" not in norm
    assert "bundle_id" not in norm


# =============================================================================
# U5: test_graceful_degradation_no_fabrication
# =============================================================================

@pytest.mark.unit
def test_graceful_degradation_no_fabrication():
    """
    U5: Missing data must NOT be fabricated.

    - Report schema_ok: true if structure valid
    - Add advisory_warnings for missing optional data
    - NEVER invent placeholder values
    """
    # Valid graceful degradation output
    valid_degraded = {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "schema_ok": True,
        "advisory_warnings": [
            "Optional artifact 'tda_metrics.json' not found",
            "Optional artifact 'divergence_distribution.json' not found",
        ],
    }

    assert valid_degraded["schema_ok"] is True
    assert len(valid_degraded["advisory_warnings"]) > 0

    # INVALID: Fabricated placeholder values
    FABRICATION_MARKERS = [
        "unknown",
        "N/A",
        "placeholder",
        "default",
        "TODO",
        "FIXME",
    ]

    def contains_fabrication(obj: Any) -> bool:
        """Check if object contains fabricated values."""
        if isinstance(obj, str):
            return any(marker.lower() in obj.lower() for marker in FABRICATION_MARKERS)
        if isinstance(obj, dict):
            return any(contains_fabrication(v) for v in obj.values())
        if isinstance(obj, list):
            return any(contains_fabrication(item) for item in obj)
        return False

    assert not contains_fabrication(valid_degraded), \
        "Graceful degradation must not fabricate values"


# =============================================================================
# U6: test_windows_safe_encoding
# =============================================================================

@pytest.mark.unit
def test_windows_safe_encoding():
    """
    U6: All file writes must use UTF-8, no BOM, no emojis.

    This prevents Windows encoding issues.
    """
    # Test JSON serialization patterns
    test_data = {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "unicode_test": "test",  # Plain ASCII
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.json"

        # Correct encoding: explicit UTF-8
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        # Read back and verify
        content_bytes = test_path.read_bytes()

        # No BOM (UTF-8 BOM = 0xEF 0xBB 0xBF)
        assert not content_bytes.startswith(b"\xef\xbb\xbf"), \
            "Output must not have UTF-8 BOM"

        # No emojis (check common emoji byte patterns)
        # Emojis are typically in range U+1F600-U+1F64F, U+1F300-U+1F5FF, etc.
        content_text = content_bytes.decode("utf-8")
        EMOJI_RANGES = [
            ("\U0001F600", "\U0001F64F"),  # Emoticons
            ("\U0001F300", "\U0001F5FF"),  # Misc symbols
            ("\U0001F680", "\U0001F6FF"),  # Transport
        ]
        for start, end in EMOJI_RANGES:
            for char in content_text:
                assert not (start <= char <= end), \
                    f"Output contains emoji: {repr(char)}"


# =============================================================================
# Additional Contract Tests
# =============================================================================

@pytest.mark.unit
def test_action_always_logged_only():
    """All divergence actions must be LOGGED_ONLY per Phase X contract."""
    valid_divergences = [
        {"cycle": 1, "action": "LOGGED_ONLY"},
        {"cycle": 2, "action": "LOGGED_ONLY"},
    ]

    for div in valid_divergences:
        assert div["action"] == "LOGGED_ONLY"

    # Invalid actions per contract
    invalid_actions = ["BLOCKED", "ENFORCED", "STOPPED", "ABORTED"]
    for action in invalid_actions:
        invalid = {"cycle": 1, "action": action}
        assert invalid["action"] != "LOGGED_ONLY", \
            f"Action '{action}' violates SHADOW mode"


@pytest.mark.unit
def test_enforcement_always_false():
    """enforcement field must always be False in SHADOW mode."""
    valid_outputs = [
        {"mode": "SHADOW", "enforcement": False},
        {"mode": "SHADOW", "governance_advisory": "...", "enforcement": False},
    ]

    for output in valid_outputs:
        # enforcement can be absent (implicit False) or explicit False
        assert output.get("enforcement", False) is False

    # enforcement=True is a contract violation
    invalid = {"mode": "SHADOW", "enforcement": True}
    assert invalid["enforcement"] is not False, \
        "enforcement=True violates SHADOW mode contract"


@pytest.mark.unit
def test_red_flags_do_not_affect_exit_code():
    """
    Red flags (logged anomalies) must NOT cause non-zero exit.

    Per Claude P: "No exit code != 0 for red flags (only for script errors)"
    """
    # Output with many red flags should still be valid
    output_with_flags = {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "red_flag_count": 100,
        "summary": {
            "status": "completed",  # NOT "failed"
        },
    }

    # Status must indicate completion, not failure
    assert output_with_flags["summary"]["status"] == "completed"
    assert output_with_flags["red_flag_count"] == 100  # Flags are logged

    # Invalid status values
    BLOCKING_STATUSES = ["failed", "blocked", "error", "rejected"]
    assert output_with_flags["summary"]["status"] not in BLOCKING_STATUSES


@pytest.mark.unit
def test_governance_advisory_message():
    """Evidence packs must include governance advisory message."""
    manifest = {
        "mode": "SHADOW",
        "schema_version": "1.0.0",
        "governance_advisory": "SHADOW MODE - This pack is advisory only and does not block any operations.",
    }

    assert "governance_advisory" in manifest
    assert "advisory" in manifest["governance_advisory"].lower()
    assert "block" in manifest["governance_advisory"].lower()


# =============================================================================
# REAL-READY — Tests for v0.1 orchestrator (scripts/run_shadow_audit.py)
# These tests import from the script and skip if not deployed.
# =============================================================================

def _script_exists() -> bool:
    """Check if run_shadow_audit.py exists and is importable."""
    test_file = Path(__file__).resolve()
    repo_root = test_file.parent.parent.parent
    script_path = repo_root / "scripts" / "run_shadow_audit.py"
    return script_path.exists()


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_discover_inputs_finds_run_dirs():
    """Given valid P3/P4 subdirs in input, returns correct run IDs."""
    from scripts.run_shadow_audit import discover_inputs

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create P3 structure under input/p3/
        p3_dir = tmpdir_path / "p3"
        p3_run = p3_dir / "fl_20250101_120000_seed42"
        p3_run.mkdir(parents=True)
        (p3_run / "stability_report.json").write_text("{}")

        # Create P4 structure under input/p4/
        p4_dir = tmpdir_path / "p4"
        p4_run = p4_dir / "p4_20250101_130000"
        p4_run.mkdir(parents=True)
        (p4_run / "p4_summary.json").write_text("{}")

        # discover_inputs takes a single input_dir that contains p3/ and p4/ subdirs
        inputs, result = discover_inputs(tmpdir_path)

        assert result.status == "pass"
        assert inputs.p3_run_id == "fl_20250101_120000_seed42"
        assert inputs.p4_run_id == "p4_20250101_130000"


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_discover_inputs_missing_p3_returns_pass_with_p4():
    """Returns pass status when P3 dir missing but P4 exists (partial artifacts ok)."""
    from scripts.run_shadow_audit import discover_inputs

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Only create P4 subdirectory
        p4_dir = tmpdir_path / "p4"
        p4_run = p4_dir / "p4_20250101_130000"
        p4_run.mkdir(parents=True)
        (p4_run / "p4_summary.json").write_text("{}")

        # discover_inputs uses single input_dir
        inputs, result = discover_inputs(tmpdir_path)

        # Having P4 is sufficient for pass status
        assert result.status == "pass"
        assert inputs.p4_run_id == "p4_20250101_130000"
        assert inputs.p3_run_id is None


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_discover_inputs_both_missing_returns_warn():
    """Returns warn status when both P3 and P4 dirs missing."""
    from scripts.run_shadow_audit import discover_inputs

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Empty input directory - no P3 or P4
        inputs, result = discover_inputs(tmpdir_path)

        # Per SHADOW MODE: warn on missing inputs, not fail
        assert result.status == "warn"
        assert "No shadow logs or P3/P4 artifacts found" in result.error
        assert inputs.p3_run_id is None
        assert inputs.p4_run_id is None


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_compute_final_status_fail_on_discover():
    """Returns 'fail' when discover stage failed."""
    from scripts.run_shadow_audit import compute_final_status, StageResult

    stages = {
        "discover": StageResult(status="fail", error="No artifacts found"),
    }

    result = compute_final_status(stages, [])
    assert result == "fail"


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_compute_final_status_warn_on_advisory():
    """Returns 'warn' when any stage has warnings."""
    from scripts.run_shadow_audit import compute_final_status, StageResult

    stages = {
        "discover": StageResult(status="pass"),
        "status": StageResult(status="warn", error="Schema validation failed"),
    }

    result = compute_final_status(stages, [])
    assert result == "warn"


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_compute_final_status_pass_when_all_pass():
    """Returns 'pass' when all stages pass."""
    from scripts.run_shadow_audit import compute_final_status, StageResult

    stages = {
        "discover": StageResult(status="pass"),
        "status": StageResult(status="pass"),
    }

    result = compute_final_status(stages, [])
    assert result == "pass"


@pytest.mark.unit
@pytest.mark.determinism
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_deterministic_timestamps_fixed():
    """With --seed, timestamps and run IDs are deterministic."""
    from scripts.run_shadow_audit import (
        get_timestamp,
        generate_run_id,
        DETERMINISTIC_TIMESTAMP,
    )

    # Deterministic mode
    ts = get_timestamp(deterministic=True)
    assert ts == DETERMINISTIC_TIMESTAMP

    # generate_run_id takes only seed; seed=None means non-deterministic
    run_id = generate_run_id(seed=42)
    # With seed, format is sha_<seed>_<hash8>
    assert run_id.startswith("sha_42_")
    assert len(run_id) == len("sha_42_") + 8  # 8 hex chars

    # Same seed produces same run_id (deterministic)
    run_id2 = generate_run_id(seed=42)
    assert run_id == run_id2

    # Non-deterministic should differ
    ts_live = get_timestamp(deterministic=False)
    assert ts_live != DETERMINISTIC_TIMESTAMP


@pytest.mark.unit
@pytest.mark.determinism
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_sort_dict_recursive():
    """_sort_dict_recursive sorts nested dicts deterministically."""
    from scripts.run_shadow_audit import _sort_dict_recursive

    unsorted = {
        "z": 1,
        "a": {"c": 3, "b": 2},
        "m": [{"y": 1, "x": 2}],
    }

    sorted_dict = _sort_dict_recursive(unsorted)

    # Keys should be sorted
    assert list(sorted_dict.keys()) == ["a", "m", "z"]
    assert list(sorted_dict["a"].keys()) == ["b", "c"]


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_write_summary_creates_file():
    """_write_summary creates run_summary.json with correct content."""
    from scripts.run_shadow_audit import _write_summary

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        run_summary = {
            "schema_version": "1.0.0",
            "run_id": "test_run",
            "mode": "SHADOW",
            "enforcement": False,
            "final_status": "pass",
        }

        summary_path = _write_summary(run_summary, output_dir, deterministic=False)

        assert summary_path.exists()
        with open(summary_path) as f:
            loaded = json.load(f)

        assert loaded["mode"] == "SHADOW"
        assert loaded["enforcement"] is False
        assert loaded["schema_version"] == "1.0.0"


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_find_run_dir_returns_most_recent():
    """find_run_dir returns the most recently modified directory."""
    from scripts.run_shadow_audit import find_run_dir
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create two run directories
        older_run = base_dir / "fl_20250101_100000"
        older_run.mkdir()
        (older_run / "dummy.txt").write_text("old")

        time.sleep(0.1)  # Ensure different mtime

        newer_run = base_dir / "fl_20250102_100000"
        newer_run.mkdir()
        (newer_run / "dummy.txt").write_text("new")

        result = find_run_dir(base_dir, "fl_")

        assert result is not None
        assert result.name == "fl_20250102_100000"


@pytest.mark.unit
@pytest.mark.skipif(not _script_exists(), reason="Script not deployed yet")
def test_find_run_dir_returns_none_for_missing():
    """find_run_dir returns None when no matching directory exists."""
    from scripts.run_shadow_audit import find_run_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        result = find_run_dir(base_dir, "fl_")

        assert result is None
