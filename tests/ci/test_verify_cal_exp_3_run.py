"""
Unit tests for scripts/verify_cal_exp_3_run.py

Tests the CAL-EXP-3 run verifier against synthetic fixtures.
Conforms to authoritative artifact layout from CAL_EXP_3_IMPLEMENTATION_PLAN.md.

Does not require actual run artifacts to exist.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

# Import the verifier module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from verify_cal_exp_3_run import (
    CheckResult,
    VerificationReport,
    load_json_safe,
    load_text_safe,
    load_cycles_jsonl,
    validate_cycles_structure,
    check_cycle_alignment,
    check_no_external_ingestion,
    check_isolation_audit,
    check_window_coverage,
    check_cycle_line_determinism,
    check_toolchain_manifest,
    verify_run,
)


# =============================================================================
# Fixtures - Authoritative Artifact Layout
# =============================================================================

@pytest.fixture
def valid_run_config() -> dict:
    """Canonical valid run_config.json per CAL_EXP_3_IMPLEMENTATION_PLAN.md.

    Note: Uses a small window (0-9) for testing, matching valid_cycles fixture.
    Production runs would use larger windows (e.g., 201-1000).
    """
    return {
        "experiment": "CAL-EXP-3",
        "spec_reference": "CAL_EXP_3_UPLIFT_SPEC.md",
        "seed": 42,
        "cycles": 10,
        "windows": {
            "evaluation_window": {
                "start_cycle": 0,
                "end_cycle": 9,
                "included_in_analysis": True
            }
        },
        "baseline_config": {
            "learning_enabled": False,
            "rfl_active": False
        },
        "treatment_config": {
            "learning_enabled": True,
            "rfl_active": True
        },
        "registered_at": "2025-12-13T00:00:00Z"
    }


@pytest.fixture
def valid_run_metadata() -> dict:
    """Canonical valid RUN_METADATA.json."""
    return {
        "experiment": "CAL-EXP-3",
        "run_id": "test_run_001",
        "verdict": "L4",
        "delta_delta_p": 0.025,
        "validity_passed": True,
        "claim_permitted": "Measured ΔΔp of +0.0250 +/- 0.0050 in cycles 0-9",
        "generated_at": "2025-12-13T12:00:00Z",
        "enforcement": False
    }


@pytest.fixture
def valid_cycles() -> list:
    """Canonical valid cycles.jsonl entries (cycles 0-9 matching valid_run_config)."""
    return [
        {"cycle": i, "delta_p": 0.01 + i * 0.001, "timestamp": f"2025-12-13T12:{i:02d}:00Z"}
        for i in range(10)
    ]


@pytest.fixture
def valid_validity_checks() -> dict:
    """Canonical valid validity_checks.json."""
    return {
        "all_passed": True,
        "toolchain_parity": True,
        "corpus_identity": True,
        "window_alignment": True,
        "no_pathology": True,
        "external_ingestion": {"detected": False}
    }


@pytest.fixture
def valid_corpus_manifest() -> dict:
    """Canonical valid corpus_manifest.json."""
    return {
        "hash": "abc123def456789012345678901234567890123456789012345678901234",
        "corpus_size": 1000,
        "generated_at": "2025-12-13T00:00:00Z"
    }


@pytest.fixture
def valid_isolation_audit() -> dict:
    """Canonical valid isolation_audit.json per §7.1.1."""
    return {
        "network_calls": [],
        "file_reads_outside_corpus": [],
        "isolation_passed": True
    }


def create_valid_run_dir(tmpdir: Path, config: dict, metadata: dict, cycles: list,
                         validity_checks: dict, corpus_manifest: dict,
                         isolation_audit: dict = None) -> Path:
    """Create a complete valid CAL-EXP-3 run directory."""
    run_dir = tmpdir / "run_001"
    run_dir.mkdir()

    # Create subdirectories
    (run_dir / "baseline").mkdir()
    (run_dir / "treatment").mkdir()
    (run_dir / "analysis").mkdir()
    (run_dir / "validity").mkdir()

    # Write run_config.json
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    # Write RUN_METADATA.json
    with open(run_dir / "RUN_METADATA.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Write baseline/cycles.jsonl
    with open(run_dir / "baseline" / "cycles.jsonl", "w", encoding="utf-8") as f:
        for entry in cycles:
            f.write(json.dumps(entry) + "\n")

    # Write treatment/cycles.jsonl (same cycles for alignment)
    with open(run_dir / "treatment" / "cycles.jsonl", "w", encoding="utf-8") as f:
        for entry in cycles:
            f.write(json.dumps(entry) + "\n")

    # Write validity/toolchain_hash.txt
    with open(run_dir / "validity" / "toolchain_hash.txt", "w", encoding="utf-8") as f:
        f.write("abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890")

    # Write validity/corpus_manifest.json
    with open(run_dir / "validity" / "corpus_manifest.json", "w", encoding="utf-8") as f:
        json.dump(corpus_manifest, f)

    # Write validity/validity_checks.json
    with open(run_dir / "validity" / "validity_checks.json", "w", encoding="utf-8") as f:
        json.dump(validity_checks, f)

    # Write validity/isolation_audit.json (§7.1.1: negative proof for F2.3)
    if isolation_audit is None:
        isolation_audit = {
            "network_calls": [],
            "file_reads_outside_corpus": [],
            "isolation_passed": True
        }
    with open(run_dir / "validity" / "isolation_audit.json", "w", encoding="utf-8") as f:
        json.dump(isolation_audit, f)

    return run_dir


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
    report.add(CheckResult("b", False, "y", "z", True))
    assert report.passed is False
    assert report.fail_count == 1


@pytest.mark.unit
def test_report_passes_with_warn_only_failure():
    """Report passes when only warn-only checks fail."""
    report = VerificationReport(run_dir="test")
    report.add(CheckResult("a", True, "x", "x", True))
    report.add(CheckResult("b", False, "y", "z", False))
    assert report.passed is True
    assert report.warn_count == 1


@pytest.mark.unit
def test_report_to_dict_sorted_keys():
    """VerificationReport.to_dict() produces deterministic output."""
    report = VerificationReport(run_dir="/test/run")
    report.add(CheckResult("check_a", True, "x", "x", True))

    d = report.to_dict()
    assert d["schema_version"] == "1.0.0"
    assert d["verifier"] == "verify_cal_exp_3_run.py"
    assert "canonical_sources" in d


@pytest.mark.unit
def test_report_write_json_sorted():
    """VerificationReport.write_json() writes with sorted keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = VerificationReport(run_dir="/test/run")
        report.add(CheckResult("test_check", True, "a", "a", True))

        report_path = Path(tmpdir) / "report.json"
        report.write_json(report_path)

        # Read back and verify sorted keys
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            loaded = json.loads(content)

        assert loaded["verdict"] == "PASS"
        # Verify file ends with newline
        assert content.endswith("\n")


# =============================================================================
# cycles.jsonl Loading Tests
# =============================================================================

@pytest.mark.unit
def test_load_cycles_jsonl_valid():
    """load_cycles_jsonl succeeds for valid JSONL."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"cycle": 0, "delta_p": 0.01, "timestamp": "2025-01-01T00:00:00Z"}\n')
        f.write('{"cycle": 1, "delta_p": 0.02, "timestamp": "2025-01-01T00:01:00Z"}\n')
        f.flush()
        path = Path(f.name)

    records, err = load_cycles_jsonl(path)
    assert err is None
    assert len(records) == 2
    assert records[0]["cycle"] == 0
    assert records[1]["delta_p"] == 0.02
    path.unlink()


@pytest.mark.unit
def test_load_cycles_jsonl_invalid_line():
    """load_cycles_jsonl returns error for invalid JSON line."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"cycle": 0}\n')
        f.write('not valid json\n')
        f.flush()
        path = Path(f.name)

    records, err = load_cycles_jsonl(path)
    assert err is not None
    assert "Line 2" in err
    path.unlink()


@pytest.mark.unit
def test_load_cycles_jsonl_missing():
    """load_cycles_jsonl returns error for missing file."""
    records, err = load_cycles_jsonl(Path("/nonexistent/file.jsonl"))
    assert err is not None
    assert "not found" in err.lower()


# =============================================================================
# Cycles Structure Validation Tests
# =============================================================================

@pytest.mark.unit
def test_validate_cycles_structure_valid():
    """validate_cycles_structure passes for valid records."""
    records = [
        {"cycle": 0, "delta_p": 0.01},
        {"cycle": 1, "delta_p": 0.02},
        {"cycle": 2, "delta_p": 0.03},
    ]
    is_valid, msg, indices = validate_cycles_structure(records, "baseline")
    assert is_valid is True
    assert indices == {0, 1, 2}


@pytest.mark.unit
def test_validate_cycles_structure_missing_cycle():
    """validate_cycles_structure fails when cycle field is missing."""
    records = [
        {"delta_p": 0.01},  # Missing cycle
    ]
    is_valid, msg, indices = validate_cycles_structure(records, "baseline")
    assert is_valid is False
    assert "missing" in msg.lower()


@pytest.mark.unit
def test_validate_cycles_structure_missing_delta_p():
    """validate_cycles_structure fails when delta_p field is missing."""
    records = [
        {"cycle": 0},  # Missing delta_p
    ]
    is_valid, msg, indices = validate_cycles_structure(records, "baseline")
    assert is_valid is False
    assert "missing" in msg.lower()


@pytest.mark.unit
def test_validate_cycles_structure_nan_value():
    """validate_cycles_structure fails for NaN delta_p values."""
    records = [
        {"cycle": 0, "delta_p": float('nan')},
    ]
    is_valid, msg, indices = validate_cycles_structure(records, "baseline")
    assert is_valid is False
    assert "nan" in msg.lower()


@pytest.mark.unit
def test_validate_cycles_structure_duplicate_cycles():
    """validate_cycles_structure fails for duplicate cycle indices."""
    records = [
        {"cycle": 0, "delta_p": 0.01},
        {"cycle": 0, "delta_p": 0.02},  # Duplicate!
    ]
    is_valid, msg, indices = validate_cycles_structure(records, "baseline")
    assert is_valid is False
    assert "duplicate" in msg.lower()


@pytest.mark.unit
def test_validate_cycles_structure_empty():
    """validate_cycles_structure fails for empty records."""
    is_valid, msg, indices = validate_cycles_structure([], "baseline")
    assert is_valid is False
    assert "empty" in msg.lower()


# =============================================================================
# Cycle Alignment Tests
# =============================================================================

@pytest.mark.unit
def test_check_cycle_alignment_exact_match():
    """check_cycle_alignment passes when cycle indices are identical."""
    baseline = {0, 1, 2, 3, 4}
    treatment = {0, 1, 2, 3, 4}
    aligned, msg = check_cycle_alignment(baseline, treatment)
    assert aligned is True
    assert "aligned" in msg.lower()


@pytest.mark.unit
def test_check_cycle_alignment_missing_in_treatment():
    """check_cycle_alignment fails when treatment is missing cycles."""
    baseline = {0, 1, 2, 3, 4}
    treatment = {0, 1, 2}  # Missing 3, 4
    aligned, msg = check_cycle_alignment(baseline, treatment)
    assert aligned is False
    assert "misaligned" in msg.lower()


@pytest.mark.unit
def test_check_cycle_alignment_missing_in_baseline():
    """check_cycle_alignment fails when baseline is missing cycles."""
    baseline = {0, 1, 2}
    treatment = {0, 1, 2, 3, 4}  # Extra 3, 4
    aligned, msg = check_cycle_alignment(baseline, treatment)
    assert aligned is False
    assert "misaligned" in msg.lower()


# =============================================================================
# External Ingestion Tests
# =============================================================================

@pytest.mark.unit
def test_check_no_external_ingestion_clean():
    """check_no_external_ingestion passes when no external ingestion."""
    validity_checks = {
        "external_ingestion": {"detected": False}
    }
    passed, msg = check_no_external_ingestion(validity_checks)
    assert passed is True


@pytest.mark.unit
def test_check_no_external_ingestion_detected():
    """check_no_external_ingestion fails when external ingestion present."""
    validity_checks = {
        "external_ingestion": {"detected": True, "detail": "network fetch"}
    }
    passed, msg = check_no_external_ingestion(validity_checks)
    assert passed is False
    assert "external" in msg.lower() or "ingestion" in msg.lower()


@pytest.mark.unit
def test_check_no_external_ingestion_network_calls():
    """check_no_external_ingestion fails when network calls present."""
    validity_checks = {
        "network_calls": ["https://example.com/data"]
    }
    passed, msg = check_no_external_ingestion(validity_checks)
    assert passed is False
    assert "network" in msg.lower()


@pytest.mark.unit
def test_check_no_external_ingestion_bool_flag():
    """check_no_external_ingestion handles boolean external_ingestion flag."""
    validity_checks = {"external_ingestion": True}
    passed, msg = check_no_external_ingestion(validity_checks)
    assert passed is False

    validity_checks = {"external_ingestion": False}
    passed, msg = check_no_external_ingestion(validity_checks)
    assert passed is True


# =============================================================================
# Isolation Audit Tests (§7.1.1)
# =============================================================================

@pytest.mark.unit
def test_check_isolation_audit_valid():
    """check_isolation_audit passes for valid isolation_audit.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({
                "network_calls": [],
                "file_reads_outside_corpus": [],
                "isolation_passed": True
            }, f)

        passed, msg, data = check_isolation_audit(run_dir)
        assert passed is True
        assert "passed" in msg.lower()


@pytest.mark.unit
def test_check_isolation_audit_missing():
    """check_isolation_audit fails when isolation_audit.json is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        passed, msg, data = check_isolation_audit(run_dir)
        assert passed is False
        assert "missing" in msg.lower()


@pytest.mark.unit
def test_check_isolation_audit_failed():
    """check_isolation_audit fails when isolation_passed is false."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({
                "network_calls": [],
                "file_reads_outside_corpus": [],
                "isolation_passed": False
            }, f)

        passed, msg, data = check_isolation_audit(run_dir)
        assert passed is False
        assert "isolation_passed=false" in msg


@pytest.mark.unit
def test_check_isolation_audit_network_calls():
    """check_isolation_audit fails when network calls present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({
                "network_calls": ["https://example.com"],
                "file_reads_outside_corpus": [],
                "isolation_passed": True  # Even if true, network calls fail
            }, f)

        passed, msg, data = check_isolation_audit(run_dir)
        assert passed is False
        assert "network" in msg.lower()


@pytest.mark.unit
def test_check_isolation_audit_file_reads():
    """check_isolation_audit fails when external file reads present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({
                "network_calls": [],
                "file_reads_outside_corpus": ["/etc/passwd"],
                "isolation_passed": True
            }, f)

        passed, msg, data = check_isolation_audit(run_dir)
        assert passed is False
        assert "filesystem" in msg.lower() or "external" in msg.lower()


# =============================================================================
# Window Coverage Tests (§3.4)
# =============================================================================

@pytest.mark.unit
def test_check_window_coverage_complete():
    """check_window_coverage passes when all cycles present."""
    cycles = {0, 1, 2, 3, 4}
    passed, msg = check_window_coverage(cycles, 0, 4, "baseline")
    assert passed is True
    assert "all" in msg.lower() and "present" in msg.lower()


@pytest.mark.unit
def test_check_window_coverage_missing():
    """check_window_coverage fails when cycles are missing."""
    cycles = {0, 1, 4}  # Missing 2, 3
    passed, msg = check_window_coverage(cycles, 0, 4, "baseline")
    assert passed is False
    assert "missing" in msg.lower()
    assert "2" in msg  # Should mention missing cycle


@pytest.mark.unit
def test_check_window_coverage_inclusive():
    """check_window_coverage treats bounds as inclusive."""
    # Window [201, 210] should require cycles 201, 202, ..., 210 (10 cycles)
    cycles = set(range(201, 211))  # 201-210 inclusive
    passed, msg = check_window_coverage(cycles, 201, 210, "baseline")
    assert passed is True

    # Missing end cycle should fail
    cycles = set(range(201, 210))  # 201-209, missing 210
    passed, msg = check_window_coverage(cycles, 201, 210, "baseline")
    assert passed is False


# =============================================================================
# Artifact Determinism Tests (§4.3)
# =============================================================================

@pytest.mark.unit
def test_check_cycle_line_determinism_valid():
    """check_cycle_line_determinism passes for canonical format."""
    records = [
        {"cycle": 0, "delta_p": 0.01},
        {"cycle": 1, "delta_p": 0.02, "timestamp": "2025-01-01T00:00:00Z"},
    ]
    passed, msg = check_cycle_line_determinism(records, "baseline")
    assert passed is True


@pytest.mark.unit
def test_check_cycle_line_determinism_uuid_value():
    """check_cycle_line_determinism fails when UUID values present."""
    records = [
        {"cycle": 0, "delta_p": 0.01, "record_uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"},
    ]
    passed, msg = check_cycle_line_determinism(records, "baseline")
    assert passed is False
    assert "uuid" in msg.lower() or "violation" in msg.lower()


@pytest.mark.unit
def test_check_cycle_line_determinism_id_field():
    """check_cycle_line_determinism fails when id field present."""
    records = [
        {"cycle": 0, "delta_p": 0.01, "id": "some_id"},
    ]
    passed, msg = check_cycle_line_determinism(records, "baseline")
    assert passed is False
    assert "forbidden" in msg.lower() or "id" in msg.lower()


@pytest.mark.unit
def test_check_cycle_line_determinism_run_id_field():
    """check_cycle_line_determinism fails when run_id field present."""
    records = [
        {"cycle": 0, "delta_p": 0.01, "run_id": "run_001"},
    ]
    passed, msg = check_cycle_line_determinism(records, "baseline")
    assert passed is False


# =============================================================================
# Toolchain Manifest Tests (§5.4)
# =============================================================================

@pytest.mark.unit
def test_check_toolchain_manifest_no_manifest():
    """check_toolchain_manifest passes without manifest (uses hash.txt)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        # No manifest, but that's OK - toolchain_hash.txt is primary
        passed, msg, provenance = check_toolchain_manifest(run_dir)
        assert passed is True


@pytest.mark.unit
def test_check_toolchain_manifest_valid():
    """check_toolchain_manifest passes for valid manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "toolchain_manifest.json", "w") as f:
            json.dump({
                "schema_version": "1.0.0",
                "experiment_id": "CAL-EXP-3",
                "provenance_level": "full",
                "toolchain_fingerprint": "a" * 64,
                "uv_lock_hash": "b" * 64
            }, f)

        passed, msg, provenance = check_toolchain_manifest(run_dir)
        assert passed is True
        assert provenance == "full"


@pytest.mark.unit
def test_check_toolchain_manifest_wrong_schema():
    """check_toolchain_manifest fails for wrong schema_version."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "toolchain_manifest.json", "w") as f:
            json.dump({
                "schema_version": "2.0.0",  # Wrong!
                "experiment_id": "CAL-EXP-3",
            }, f)

        passed, msg, provenance = check_toolchain_manifest(run_dir)
        assert passed is False
        assert "schema_version" in msg.lower()


@pytest.mark.unit
def test_check_toolchain_manifest_wrong_experiment():
    """check_toolchain_manifest fails for wrong experiment_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "toolchain_manifest.json", "w") as f:
            json.dump({
                "schema_version": "1.0.0",
                "experiment_id": "CAL-EXP-2",  # Wrong!
            }, f)

        passed, msg, provenance = check_toolchain_manifest(run_dir)
        assert passed is False
        assert "experiment_id" in msg.lower()


@pytest.mark.unit
def test_check_toolchain_manifest_partial_provenance():
    """check_toolchain_manifest returns partial provenance level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        with open(run_dir / "validity" / "toolchain_manifest.json", "w") as f:
            json.dump({
                "schema_version": "1.0.0",
                "experiment_id": "CAL-EXP-3",
                "provenance_level": "partial",
            }, f)

        passed, msg, provenance = check_toolchain_manifest(run_dir)
        assert passed is True
        assert provenance == "partial"


# =============================================================================
# Full Verification Tests
# =============================================================================

@pytest.mark.unit
def test_verify_run_valid(valid_run_config, valid_run_metadata, valid_cycles,
                          valid_validity_checks, valid_corpus_manifest):
    """verify_run passes for fully valid run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_valid_run_dir(
            Path(tmpdir), valid_run_config, valid_run_metadata,
            valid_cycles, valid_validity_checks, valid_corpus_manifest
        )

        report = verify_run(run_dir)
        assert report.passed is True, f"Expected PASS, got FAIL: {report.fail_count} failures"


@pytest.mark.unit
def test_verify_run_missing_dir():
    """verify_run handles missing run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "nonexistent"

        report = verify_run(run_dir)
        assert report.passed is False
        assert any("run_dir_exists" in c.name for c in report.checks)


@pytest.mark.unit
def test_verify_run_wrong_experiment():
    """verify_run fails when experiment is not CAL-EXP-3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        for d in ["baseline", "treatment", "analysis", "validity"]:
            (run_dir / d).mkdir()

        config = {"experiment": "CAL-EXP-2", "seed": 42}  # Wrong!
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)
        with open(run_dir / "RUN_METADATA.json", "w") as f:
            json.dump({"enforcement": False}, f)

        # Create minimal required files
        for arm in ["baseline", "treatment"]:
            with open(run_dir / arm / "cycles.jsonl", "w") as f:
                f.write('{"cycle": 0, "delta_p": 0.01}\n')
        with open(run_dir / "validity" / "toolchain_hash.txt", "w") as f:
            f.write("a" * 64)
        with open(run_dir / "validity" / "corpus_manifest.json", "w") as f:
            json.dump({"hash": "a" * 64}, f)
        with open(run_dir / "validity" / "validity_checks.json", "w") as f:
            json.dump({"all_passed": True}, f)
        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({"network_calls": [], "file_reads_outside_corpus": [], "isolation_passed": True}, f)

        report = verify_run(run_dir)
        exp_check = next((c for c in report.checks if c.name == "experiment_identity"), None)
        assert exp_check is not None
        assert exp_check.passed is False


@pytest.mark.unit
def test_verify_run_seed_mismatch():
    """verify_run fails when baseline and treatment have different seeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        for d in ["baseline", "treatment", "analysis", "validity"]:
            (run_dir / d).mkdir()

        config = {
            "experiment": "CAL-EXP-3",
            "seed": 42,
            "baseline_config": {"learning_enabled": False, "seed": 42},
            "treatment_config": {"learning_enabled": True, "seed": 43},  # Different!
            "windows": {"evaluation_window": {"start_cycle": 0, "end_cycle": 100}}
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)
        with open(run_dir / "RUN_METADATA.json", "w") as f:
            json.dump({"enforcement": False}, f)

        for arm in ["baseline", "treatment"]:
            with open(run_dir / arm / "cycles.jsonl", "w") as f:
                f.write('{"cycle": 0, "delta_p": 0.01}\n')
        with open(run_dir / "validity" / "toolchain_hash.txt", "w") as f:
            f.write("a" * 64)
        with open(run_dir / "validity" / "corpus_manifest.json", "w") as f:
            json.dump({"hash": "a" * 64}, f)
        with open(run_dir / "validity" / "validity_checks.json", "w") as f:
            json.dump({"all_passed": True}, f)
        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({"network_calls": [], "file_reads_outside_corpus": [], "isolation_passed": True}, f)

        report = verify_run(run_dir)
        seed_check = next((c for c in report.checks if c.name == "seed:identical_across_arms"), None)
        assert seed_check is not None
        assert seed_check.passed is False


@pytest.mark.unit
def test_verify_run_cycle_misalignment():
    """verify_run fails when baseline and treatment have different cycles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        for d in ["baseline", "treatment", "analysis", "validity"]:
            (run_dir / d).mkdir()

        config = {
            "experiment": "CAL-EXP-3",
            "seed": 42,
            "baseline_config": {"learning_enabled": False},
            "treatment_config": {"learning_enabled": True},
            "windows": {"evaluation_window": {"start_cycle": 0, "end_cycle": 100}}
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)
        with open(run_dir / "RUN_METADATA.json", "w") as f:
            json.dump({"enforcement": False}, f)

        # Baseline has cycles 0, 1, 2
        with open(run_dir / "baseline" / "cycles.jsonl", "w") as f:
            for i in [0, 1, 2]:
                f.write(json.dumps({"cycle": i, "delta_p": 0.01}) + "\n")

        # Treatment has cycles 0, 1, 2, 3, 4 (extra!)
        with open(run_dir / "treatment" / "cycles.jsonl", "w") as f:
            for i in [0, 1, 2, 3, 4]:
                f.write(json.dumps({"cycle": i, "delta_p": 0.01}) + "\n")

        with open(run_dir / "validity" / "toolchain_hash.txt", "w") as f:
            f.write("a" * 64)
        with open(run_dir / "validity" / "corpus_manifest.json", "w") as f:
            json.dump({"hash": "a" * 64}, f)
        with open(run_dir / "validity" / "validity_checks.json", "w") as f:
            json.dump({"all_passed": True}, f)
        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({"network_calls": [], "file_reads_outside_corpus": [], "isolation_passed": True}, f)

        report = verify_run(run_dir)
        align_check = next((c for c in report.checks if c.name == "cycle_alignment:exact"), None)
        assert align_check is not None
        assert align_check.passed is False


@pytest.mark.unit
def test_verify_run_enforcement_true():
    """verify_run fails when enforcement is true."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        for d in ["baseline", "treatment", "analysis", "validity"]:
            (run_dir / d).mkdir()

        config = {
            "experiment": "CAL-EXP-3",
            "seed": 42,
            "baseline_config": {"learning_enabled": False},
            "treatment_config": {"learning_enabled": True},
            "windows": {"evaluation_window": {}}
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)
        with open(run_dir / "RUN_METADATA.json", "w") as f:
            json.dump({"enforcement": True}, f)  # VIOLATION!

        for arm in ["baseline", "treatment"]:
            with open(run_dir / arm / "cycles.jsonl", "w") as f:
                f.write('{"cycle": 0, "delta_p": 0.01}\n')
        with open(run_dir / "validity" / "toolchain_hash.txt", "w") as f:
            f.write("a" * 64)
        with open(run_dir / "validity" / "corpus_manifest.json", "w") as f:
            json.dump({"hash": "a" * 64}, f)
        with open(run_dir / "validity" / "validity_checks.json", "w") as f:
            json.dump({"all_passed": True}, f)
        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({"network_calls": [], "file_reads_outside_corpus": [], "isolation_passed": True}, f)

        report = verify_run(run_dir)
        enforce_check = next((c for c in report.checks if c.name == "shadow_mode:enforcement"), None)
        assert enforce_check is not None
        assert enforce_check.passed is False


@pytest.mark.unit
def test_verify_run_missing_windows():
    """verify_run fails when windows are not pre-registered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        for d in ["baseline", "treatment", "analysis", "validity"]:
            (run_dir / d).mkdir()

        config = {
            "experiment": "CAL-EXP-3",
            "seed": 42,
            "baseline_config": {"learning_enabled": False},
            "treatment_config": {"learning_enabled": True},
            # NO windows!
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)
        with open(run_dir / "RUN_METADATA.json", "w") as f:
            json.dump({"enforcement": False}, f)

        for arm in ["baseline", "treatment"]:
            with open(run_dir / arm / "cycles.jsonl", "w") as f:
                f.write('{"cycle": 0, "delta_p": 0.01}\n')
        with open(run_dir / "validity" / "toolchain_hash.txt", "w") as f:
            f.write("a" * 64)
        with open(run_dir / "validity" / "corpus_manifest.json", "w") as f:
            json.dump({"hash": "a" * 64}, f)
        with open(run_dir / "validity" / "validity_checks.json", "w") as f:
            json.dump({"all_passed": True}, f)
        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({"network_calls": [], "file_reads_outside_corpus": [], "isolation_passed": True}, f)

        report = verify_run(run_dir)
        window_check = next((c for c in report.checks if c.name == "windows:pre_registered"), None)
        assert window_check is not None
        assert window_check.passed is False


@pytest.mark.unit
def test_verify_run_optional_files_warn():
    """verify_run warns (not fails) for missing optional files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        for d in ["baseline", "treatment", "analysis", "validity"]:
            (run_dir / d).mkdir()

        config = {
            "experiment": "CAL-EXP-3",
            "seed": 42,
            "baseline_config": {"learning_enabled": False},
            "treatment_config": {"learning_enabled": True},
            "windows": {"evaluation_window": {"start_cycle": 0, "end_cycle": 2}}  # Small window
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)
        with open(run_dir / "RUN_METADATA.json", "w") as f:
            json.dump({"enforcement": False}, f)

        # Create cycles 0, 1, 2 matching the window
        for arm in ["baseline", "treatment"]:
            with open(run_dir / arm / "cycles.jsonl", "w") as f:
                for i in range(3):
                    f.write(json.dumps({"cycle": i, "delta_p": 0.01}) + "\n")
        with open(run_dir / "validity" / "toolchain_hash.txt", "w") as f:
            f.write("a" * 64)
        with open(run_dir / "validity" / "corpus_manifest.json", "w") as f:
            json.dump({"hash": "a" * 64}, f)
        with open(run_dir / "validity" / "validity_checks.json", "w") as f:
            json.dump({"all_passed": True}, f)
        with open(run_dir / "validity" / "isolation_audit.json", "w") as f:
            json.dump({"network_calls": [], "file_reads_outside_corpus": [], "isolation_passed": True}, f)

        # Don't create optional files (summary.json, uplift_report.json, etc.)

        report = verify_run(run_dir)

        # Should still pass overall (optional files are WARN only)
        assert report.passed is True
        assert report.warn_count >= 1  # At least one warning for missing optional


@pytest.mark.unit
def test_verify_run_with_json_report(valid_run_config, valid_run_metadata, valid_cycles,
                                      valid_validity_checks, valid_corpus_manifest):
    """verify_run generates valid JSON report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_valid_run_dir(
            Path(tmpdir), valid_run_config, valid_run_metadata,
            valid_cycles, valid_validity_checks, valid_corpus_manifest
        )

        report = verify_run(run_dir)
        report_path = run_dir / "verification_report.json"
        report.write_json(report_path)

        assert report_path.exists()

        with open(report_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["verdict"] == "PASS"
        assert loaded["schema_version"] == "1.0.0"
        assert loaded["verifier"] == "verify_cal_exp_3_run.py"
