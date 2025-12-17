"""
Unit tests for scripts/verify_cal_exp_2_run.py

Tests the CAL-EXP-2 run verifier against synthetic fixtures.
Does not require actual run artifacts to exist.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Import the verifier module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from verify_cal_exp_2_run import (
    CheckResult,
    VerificationReport,
    load_json_safe,
    validate_jsonl,
    check_divergence_actions,
    verify_run,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_run_config() -> dict:
    """Canonical valid run_config.json content."""
    return {
        "schema_version": "1.4.0",
        "mode": "SHADOW",
        "phase": "P4",
        "run_id": "test_run",
        "parameters": {
            "cycles": 1000,
            "seed": 42,
        },
        "twin_lr_overrides": {
            "H": 0.20,
            "rho": 0.15,
            "tau": 0.02,
            "beta": 0.12,
        },
    }


@pytest.fixture
def valid_metadata() -> dict:
    """Canonical valid RUN_METADATA.json content."""
    return {
        "status": "completed",
        "enforcement": False,
        "total_cycles_requested": 1000,
        "cycles_completed": 1000,
    }


@pytest.fixture
def valid_divergence_log() -> list:
    """Canonical valid divergence_log.jsonl entries."""
    return [
        {"cycle": i, "field": "H", "action": "LOGGED_ONLY"}
        for i in range(10)
    ]


# =============================================================================
# CheckResult Tests
# =============================================================================

@pytest.mark.unit
def test_check_result_pass_str():
    """CheckResult formats PASS correctly."""
    result = CheckResult(
        name="test_check",
        passed=True,
        expected="foo",
        actual="foo",
        invalidates=True,
    )
    assert "[PASS]" in str(result)
    assert "test_check" in str(result)


@pytest.mark.unit
def test_check_result_fail_str():
    """CheckResult formats FAIL correctly for invalidating checks."""
    result = CheckResult(
        name="test_check",
        passed=False,
        expected="foo",
        actual="bar",
        invalidates=True,
    )
    assert "[FAIL]" in str(result)


@pytest.mark.unit
def test_check_result_warn_str():
    """CheckResult formats WARN correctly for non-invalidating checks."""
    result = CheckResult(
        name="test_check",
        passed=False,
        expected="foo",
        actual="bar",
        invalidates=False,
    )
    assert "[WARN]" in str(result)


@pytest.mark.unit
def test_check_result_to_dict():
    """CheckResult.to_dict() serializes correctly."""
    result = CheckResult(
        name="test_check",
        passed=True,
        expected="foo",
        actual="foo",
        invalidates=True,
    )
    d = result.to_dict()
    assert d["name"] == "test_check"
    assert d["status"] == "PASS"
    assert d["passed"] is True
    assert d["expected"] == "foo"
    assert d["actual"] == "foo"
    assert d["invalidates"] is True


@pytest.mark.unit
def test_check_result_to_dict_fail():
    """CheckResult.to_dict() formats FAIL status correctly."""
    result = CheckResult(
        name="fail_check",
        passed=False,
        expected="x",
        actual="y",
        invalidates=True,
    )
    d = result.to_dict()
    assert d["status"] == "FAIL"
    assert d["passed"] is False


@pytest.mark.unit
def test_check_result_to_dict_warn():
    """CheckResult.to_dict() formats WARN status correctly."""
    result = CheckResult(
        name="warn_check",
        passed=False,
        expected="x",
        actual="y",
        invalidates=False,
    )
    d = result.to_dict()
    assert d["status"] == "WARN"
    assert d["passed"] is False


# =============================================================================
# VerificationReport Tests
# =============================================================================

@pytest.mark.unit
def test_report_passed_all_pass():
    """Report passes when all checks pass."""
    report = VerificationReport(run_dir="test")
    report.add(CheckResult("a", True, "x", "x", True))
    report.add(CheckResult("b", True, "y", "y", True))
    assert report.passed is True
    assert report.fail_count == 0


@pytest.mark.unit
def test_report_fails_on_invalidating_failure():
    """Report fails when an invalidating check fails."""
    report = VerificationReport(run_dir="test")
    report.add(CheckResult("a", True, "x", "x", True))
    report.add(CheckResult("b", False, "y", "z", True))  # invalidates=True
    assert report.passed is False
    assert report.fail_count == 1


@pytest.mark.unit
def test_report_passes_with_warn_only_failure():
    """Report passes when only warn-only checks fail."""
    report = VerificationReport(run_dir="test")
    report.add(CheckResult("a", True, "x", "x", True))
    report.add(CheckResult("b", False, "y", "z", False))  # invalidates=False
    assert report.passed is True
    assert report.warn_count == 1


@pytest.mark.unit
def test_report_pass_count():
    """Report correctly counts passing checks."""
    report = VerificationReport(run_dir="test")
    report.add(CheckResult("a", True, "x", "x", True))
    report.add(CheckResult("b", True, "y", "y", True))
    report.add(CheckResult("c", False, "z", "w", False))  # WARN
    assert report.pass_count == 2
    assert report.warn_count == 1
    assert report.fail_count == 0


@pytest.mark.unit
def test_report_to_dict():
    """VerificationReport.to_dict() serializes correctly."""
    report = VerificationReport(run_dir="/test/run/dir")
    report.add(CheckResult("check_a", True, "x", "x", True))
    report.add(CheckResult("check_b", False, "y", "z", True))

    d = report.to_dict()

    assert d["schema_version"] == "1.0.0"
    assert d["verifier"] == "verify_cal_exp_2_run.py"
    assert "canonical_contract" in d
    assert d["run_dir"] == "/test/run/dir"
    assert "timestamp" in d
    assert d["verdict"] == "FAIL"  # One invalidating failure
    assert d["summary"]["total_checks"] == 2
    assert d["summary"]["pass_count"] == 1
    assert d["summary"]["fail_count"] == 1
    assert d["summary"]["warn_count"] == 0
    assert len(d["checks"]) == 2


@pytest.mark.unit
def test_report_to_dict_pass_verdict():
    """VerificationReport.to_dict() shows PASS verdict when all pass."""
    report = VerificationReport(run_dir="test")
    report.add(CheckResult("a", True, "x", "x", True))
    report.add(CheckResult("b", True, "y", "y", True))

    d = report.to_dict()
    assert d["verdict"] == "PASS"


@pytest.mark.unit
def test_report_write_json():
    """VerificationReport.write_json() writes valid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = VerificationReport(run_dir="/test/run")
        report.add(CheckResult("test_check", True, "a", "a", True))

        report_path = Path(tmpdir) / "verification_report.json"
        report.write_json(report_path)

        assert report_path.exists()

        with open(report_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["verdict"] == "PASS"
        assert loaded["summary"]["total_checks"] == 1
        assert len(loaded["checks"]) == 1
        assert loaded["checks"][0]["name"] == "test_check"


@pytest.mark.unit
def test_report_write_json_creates_parent_dirs():
    """VerificationReport.write_json() creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = VerificationReport(run_dir="test")
        report.add(CheckResult("check", True, "x", "x", True))

        # Nested path that doesn't exist
        report_path = Path(tmpdir) / "nested" / "subdir" / "report.json"
        report.write_json(report_path)

        assert report_path.exists()


# =============================================================================
# JSON Loading Tests
# =============================================================================

@pytest.mark.unit
def test_load_json_safe_valid():
    """load_json_safe returns data for valid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "value"}, f)
        f.flush()
        path = Path(f.name)

    data, err = load_json_safe(path)
    assert err is None
    assert data == {"key": "value"}
    path.unlink()


@pytest.mark.unit
def test_load_json_safe_missing_file():
    """load_json_safe returns error for missing file."""
    data, err = load_json_safe(Path("/nonexistent/file.json"))
    assert data is None
    assert "not found" in err.lower()


@pytest.mark.unit
def test_load_json_safe_invalid_json():
    """load_json_safe returns error for invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json {{{")
        f.flush()
        path = Path(f.name)

    data, err = load_json_safe(path)
    assert data is None
    assert "parse error" in err.lower() or "decode" in err.lower()
    path.unlink()


# =============================================================================
# JSONL Validation Tests
# =============================================================================

@pytest.mark.unit
def test_validate_jsonl_valid():
    """validate_jsonl succeeds for valid JSONL."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"a": 1}\n')
        f.write('{"b": 2}\n')
        f.write('{"c": 3}\n')
        f.flush()
        path = Path(f.name)

    count, err = validate_jsonl(path)
    assert err is None
    assert count == 3
    path.unlink()


@pytest.mark.unit
def test_validate_jsonl_invalid_line():
    """validate_jsonl returns error for invalid line."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"a": 1}\n')
        f.write('not json\n')
        f.write('{"c": 3}\n')
        f.flush()
        path = Path(f.name)

    count, err = validate_jsonl(path)
    assert err is not None
    assert "line 2" in err.lower()
    path.unlink()


# =============================================================================
# Divergence Actions Tests
# =============================================================================

@pytest.mark.unit
def test_check_divergence_actions_all_logged():
    """check_divergence_actions passes when all actions are LOGGED_ONLY."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"cycle": 0, "action": "LOGGED_ONLY"}\n')
        f.write('{"cycle": 1, "action": "LOGGED_ONLY"}\n')
        f.flush()
        path = Path(f.name)

    ok, msg = check_divergence_actions(path)
    assert ok is True
    assert "LOGGED_ONLY" in msg
    path.unlink()


@pytest.mark.unit
def test_check_divergence_actions_violation():
    """check_divergence_actions fails when action is not LOGGED_ONLY."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"cycle": 0, "action": "LOGGED_ONLY"}\n')
        f.write('{"cycle": 1, "action": "ENFORCED"}\n')
        f.flush()
        path = Path(f.name)

    ok, msg = check_divergence_actions(path)
    assert ok is False
    assert "violation" in msg.lower()
    path.unlink()


# =============================================================================
# Full Verification Tests
# =============================================================================

@pytest.mark.unit
def test_verify_run_valid(valid_run_config, valid_metadata, valid_divergence_log):
    """verify_run passes for a fully valid run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Write run_config.json
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(valid_run_config, f)

        # Write RUN_METADATA.json
        with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
            json.dump(valid_metadata, f)

        # Write divergence_log.jsonl
        with open(run_dir / "divergence_log.jsonl", "w", encoding="utf-8") as f:
            for entry in valid_divergence_log:
                f.write(json.dumps(entry) + "\n")

        report = verify_run(run_dir)
        assert report.passed is True, f"Expected PASS, got FAIL: {report.fail_count} failures"


@pytest.mark.unit
def test_verify_run_fails_on_wrong_mode(valid_run_config, valid_metadata):
    """verify_run fails when mode is not SHADOW."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Modify mode
        config = valid_run_config.copy()
        config["mode"] = "ACTIVE"

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
            json.dump(valid_metadata, f)

        report = verify_run(run_dir)
        assert report.passed is False

        mode_check = next((c for c in report.checks if c.name == "mode"), None)
        assert mode_check is not None
        assert mode_check.passed is False


@pytest.mark.unit
def test_verify_run_fails_on_enforcement_true(valid_run_config, valid_metadata):
    """verify_run fails when enforcement is true."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(valid_run_config, f)

        # Modify enforcement
        metadata = valid_metadata.copy()
        metadata["enforcement"] = True

        with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        report = verify_run(run_dir)
        assert report.passed is False

        enforcement_check = next((c for c in report.checks if c.name == "enforcement"), None)
        assert enforcement_check is not None
        assert enforcement_check.passed is False


@pytest.mark.unit
def test_verify_run_fails_on_blocking_status(valid_run_config, valid_metadata):
    """verify_run fails when status is a blocking value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(valid_run_config, f)

        # Modify status
        metadata = valid_metadata.copy()
        metadata["status"] = "blocked"

        with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        report = verify_run(run_dir)
        assert report.passed is False


@pytest.mark.unit
def test_verify_run_fails_on_lr_out_of_bounds(valid_run_config, valid_metadata):
    """verify_run fails when LR is outside [0, 1]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Modify LR to invalid value
        config = valid_run_config.copy()
        config["twin_lr_overrides"] = {"H": 1.5}  # Out of bounds

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
            json.dump(valid_metadata, f)

        report = verify_run(run_dir)
        assert report.passed is False


@pytest.mark.unit
def test_verify_run_warns_on_non_canonical_seed(valid_run_config, valid_metadata):
    """verify_run warns (but passes) on non-canonical seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Modify seed
        config = valid_run_config.copy()
        config["parameters"]["seed"] = 123  # Non-canonical

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
            json.dump(valid_metadata, f)

        report = verify_run(run_dir)
        # Should still pass (seed is warn-only)
        assert report.passed is True
        assert report.warn_count >= 1


@pytest.mark.unit
def test_verify_run_with_json_report(valid_run_config, valid_metadata, valid_divergence_log):
    """verify_run generates valid JSON report when write_json is called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Write run_config.json
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(valid_run_config, f)

        # Write RUN_METADATA.json
        with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
            json.dump(valid_metadata, f)

        # Write divergence_log.jsonl
        with open(run_dir / "divergence_log.jsonl", "w", encoding="utf-8") as f:
            for entry in valid_divergence_log:
                f.write(json.dumps(entry) + "\n")

        report = verify_run(run_dir)

        # Write report to run directory (default behavior)
        report_path = run_dir / "cal_exp_2_verification_report.json"
        report.write_json(report_path)

        assert report_path.exists()

        with open(report_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["verdict"] == "PASS"
        assert loaded["schema_version"] == "1.0.0"
        assert loaded["verifier"] == "verify_cal_exp_2_run.py"
        assert str(run_dir) in loaded["run_dir"]
        assert loaded["summary"]["fail_count"] == 0


# =============================================================================
# Integration Test with Real Fixture (if exists)
# =============================================================================

@pytest.mark.unit
def test_verify_real_cal_exp_2_run_if_exists():
    """Test against real CAL-EXP-2 run if it exists."""
    real_run_dir = Path("results/cal_exp_2/p4_20251212_103832")

    if not real_run_dir.exists():
        pytest.skip(f"Real run directory not found: {real_run_dir}")

    report = verify_run(real_run_dir)

    # Real run should pass all checks
    assert report.passed is True, (
        f"Real CAL-EXP-2 run failed verification with {report.fail_count} failures"
    )
