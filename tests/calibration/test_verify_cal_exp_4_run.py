#!/usr/bin/env python3
"""
Unit tests for CAL-EXP-4 Run Verifier

Tests verify_cal_exp_4_run.py against:
- Happy path (all thresholds pass)
- Each F5.x failure individually
- Schema malformed → fail-close
- Missing audit → F5.4 cap

SHADOW MODE: These tests validate the verifier script, not CAL-EXP-4 runs.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

# Import verifier functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from verify_cal_exp_4_run import (
    CheckResult,
    VerificationReport,
    load_json_safe,
    validate_schema_version,
    validate_experiment_id,
    validate_required_fields,
    check_for_nan_inf,
    verify_run,
    SCHEMA_VERSION,
    EXPERIMENT_ID,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_temporal_audit():
    """Valid temporal_structure_audit.json content."""
    return {
        "schema_version": "1.0.0",
        "experiment_id": "CAL-EXP-4",
        "baseline_arm": {
            "cycle_count": 800,
            "cycle_min": 201,
            "cycle_max": 1000,
            "cycle_gap_max": 1,
            "cycle_gap_mean": 1.0,
            "monotonic_cycle_indices": True,
            "timestamp_monotonic": True,
            "temporal_coverage_ratio": 1.0,
            "missing_cycles": None,
            "duplicate_cycles": None,
        },
        "treatment_arm": {
            "cycle_count": 800,
            "cycle_min": 201,
            "cycle_max": 1000,
            "cycle_gap_max": 1,
            "cycle_gap_mean": 1.0,
            "monotonic_cycle_indices": True,
            "timestamp_monotonic": True,
            "temporal_coverage_ratio": 1.0,
            "missing_cycles": None,
            "duplicate_cycles": None,
        },
        "comparability": {
            "cycle_count_match": True,
            "cycle_indices_identical": True,
            "coverage_ratio_match": True,
            "gap_structure_compatible": True,
            "temporal_structure_compatible": True,
            "temporal_structure_pass": True,
            "mismatch_details": None,
        },
        "thresholds": {
            "min_coverage_ratio": 1.0,
            "max_gap_ratio_divergence": 0.1,
        },
        "evaluation_window": {
            "start_cycle": 201,
            "end_cycle": 1000,
            "expected_cycle_count": 800,
        },
        "generated_at": "2025-12-17T00:00:00Z",
    }


@pytest.fixture
def valid_variance_audit():
    """Valid variance_profile_audit.json content."""
    return {
        "schema_version": "1.0.0",
        "experiment_id": "CAL-EXP-4",
        "baseline_arm": {
            "delta_p_count": 800,
            "delta_p_mean": 0.85,
            "delta_p_variance": 0.01,
            "delta_p_std": 0.1,
            "delta_p_iqr": 0.12,
            "delta_p_range": 0.5,
            "delta_p_min": 0.6,
            "delta_p_max": 1.1,
            "delta_p_median": 0.85,
            "delta_p_q1": 0.79,
            "delta_p_q3": 0.91,
            "has_nan": False,
            "has_inf": False,
            "windowed_variances": [0.01, 0.01, 0.01, 0.01],
        },
        "treatment_arm": {
            "delta_p_count": 800,
            "delta_p_mean": 0.88,
            "delta_p_variance": 0.012,
            "delta_p_std": 0.11,
            "delta_p_iqr": 0.13,
            "delta_p_range": 0.55,
            "delta_p_min": 0.62,
            "delta_p_max": 1.17,
            "delta_p_median": 0.88,
            "delta_p_q1": 0.81,
            "delta_p_q3": 0.94,
            "has_nan": False,
            "has_inf": False,
            "windowed_variances": [0.012, 0.011, 0.013, 0.012],
        },
        "comparability": {
            "variance_ratio": 1.2,
            "variance_ratio_acceptable": True,
            "windowed_variance_drift": 0.002,
            "windowed_drift_acceptable": True,
            "iqr_ratio": 1.08,
            "iqr_ratio_acceptable": True,
            "profile_compatible": True,
            "variance_profile_pass": True,
            "claim_cap_applied": False,
            "claim_cap_level": None,
            "mismatch_details": None,
        },
        "thresholds": {
            "variance_ratio_max": 2.0,
            "variance_ratio_min": 0.5,
            "windowed_drift_max": 0.05,
            "iqr_ratio_max": 2.0,
            "claim_cap_threshold": 3.0,
        },
        "evaluation_window": {
            "start_cycle": 201,
            "end_cycle": 1000,
        },
        "generated_at": "2025-12-17T00:00:00Z",
    }


def create_run_dir(tmpdir, temporal_audit, variance_audit):
    """Helper to create a valid run directory structure."""
    run_dir = Path(tmpdir) / "run"
    run_dir.mkdir()
    (run_dir / "validity").mkdir()

    if temporal_audit is not None:
        with open(run_dir / "validity" / "temporal_structure_audit.json", "w") as f:
            json.dump(temporal_audit, f)

    if variance_audit is not None:
        with open(run_dir / "validity" / "variance_profile_audit.json", "w") as f:
            json.dump(variance_audit, f)

    return run_dir


# =============================================================================
# Unit Tests: CheckResult
# =============================================================================

@pytest.mark.unit
def test_check_result_pass_str():
    """CheckResult.__str__ shows PASS for passed checks."""
    cr = CheckResult("test", True, "expected", "actual", True)
    assert "[PASS]" in str(cr) or "PASS:" in str(cr)


@pytest.mark.unit
def test_check_result_fail_str():
    """CheckResult.__str__ shows FAIL for failed invalidating checks."""
    cr = CheckResult("test", False, "expected", "actual", True)
    assert "[FAIL]" in str(cr) or "FAIL:" in str(cr)


@pytest.mark.unit
def test_check_result_warn_str():
    """CheckResult.__str__ shows WARN for failed non-invalidating checks."""
    cr = CheckResult("test", False, "expected", "actual", False)
    assert "[WARN]" in str(cr) or "WARN:" in str(cr)


@pytest.mark.unit
def test_check_result_f5_code_in_str():
    """CheckResult.__str__ includes F5 code when present and failed."""
    cr = CheckResult("test", False, "expected", "actual", True, f5_code="F5.1")
    assert "F5.1" in str(cr)


@pytest.mark.unit
def test_check_result_to_dict():
    """CheckResult.to_dict() includes all fields."""
    cr = CheckResult("test", True, "expected", "actual", True, f5_code="F5.1")
    d = cr.to_dict()
    assert d["name"] == "test"
    assert d["passed"] is True
    assert d["f5_code"] == "F5.1"


# =============================================================================
# Unit Tests: VerificationReport
# =============================================================================

@pytest.mark.unit
def test_report_passed_all_pass():
    """Report.passed is True when all checks pass."""
    report = VerificationReport("test")
    report.add(CheckResult("a", True, "", "", True))
    report.add(CheckResult("b", True, "", "", True))
    assert report.passed is True


@pytest.mark.unit
def test_report_fails_on_invalidating_failure():
    """Report.passed is False when invalidating check fails."""
    report = VerificationReport("test")
    report.add(CheckResult("a", True, "", "", True))
    report.add(CheckResult("b", False, "", "", True))  # invalidates=True
    assert report.passed is False


@pytest.mark.unit
def test_report_passes_with_warn_only_failure():
    """Report.passed is True when only non-invalidating checks fail."""
    report = VerificationReport("test")
    report.add(CheckResult("a", True, "", "", True))
    report.add(CheckResult("b", False, "", "", False))  # WARN only
    assert report.passed is True


@pytest.mark.unit
def test_report_f5_failure_codes():
    """Report.f5_failure_codes collects unique F5 codes."""
    report = VerificationReport("test")
    report.add(CheckResult("a", False, "", "", True, f5_code="F5.1"))
    report.add(CheckResult("b", False, "", "", True, f5_code="F5.2"))
    report.add(CheckResult("c", False, "", "", True, f5_code="F5.1"))  # duplicate
    report.add(CheckResult("d", True, "", "", True))  # passed, no code
    assert report.f5_failure_codes == ["F5.1", "F5.2"]


@pytest.mark.unit
def test_report_to_dict_has_required_fields():
    """Report.to_dict() includes all required output fields."""
    report = VerificationReport("test")
    report.temporal_comparability = True
    report.variance_comparability = True
    report.claim_cap_applied = False
    report.claim_cap_level = None
    d = report.to_dict()

    assert "temporal_comparability" in d
    assert "variance_comparability" in d
    assert "f5_failure_codes" in d
    assert "claim_cap_applied" in d
    assert "claim_cap_level" in d
    assert "verdict" in d


# =============================================================================
# Unit Tests: Utility Functions
# =============================================================================

@pytest.mark.unit
def test_load_json_safe_missing_file():
    """load_json_safe returns error for missing file."""
    data, error = load_json_safe(Path("/nonexistent/file.json"))
    assert data is None
    assert "not found" in error


@pytest.mark.unit
def test_load_json_safe_invalid_json():
    """load_json_safe returns error for invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json")
        f.flush()
        data, error = load_json_safe(Path(f.name))
    assert data is None
    assert "invalid JSON" in error


@pytest.mark.unit
def test_load_json_safe_valid():
    """load_json_safe returns data for valid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "value"}, f)
        f.flush()
        data, error = load_json_safe(Path(f.name))
    assert data == {"key": "value"}
    assert error is None


@pytest.mark.unit
def test_validate_schema_version_pass():
    """validate_schema_version passes for correct version."""
    passed, msg = validate_schema_version({"schema_version": "1.0.0"}, "1.0.0")
    assert passed is True


@pytest.mark.unit
def test_validate_schema_version_fail():
    """validate_schema_version fails for wrong version."""
    passed, msg = validate_schema_version({"schema_version": "2.0.0"}, "1.0.0")
    assert passed is False


@pytest.mark.unit
def test_validate_experiment_id_pass():
    """validate_experiment_id passes for correct ID."""
    passed, msg = validate_experiment_id({"experiment_id": "CAL-EXP-4"}, "CAL-EXP-4")
    assert passed is True


@pytest.mark.unit
def test_validate_experiment_id_fail():
    """validate_experiment_id fails for wrong ID."""
    passed, msg = validate_experiment_id({"experiment_id": "CAL-EXP-3"}, "CAL-EXP-4")
    assert passed is False


@pytest.mark.unit
def test_check_for_nan_inf_clean():
    """check_for_nan_inf returns empty for clean data."""
    data = {"a": 1.0, "b": {"c": 2.0}, "d": [3.0, 4.0]}
    issues = check_for_nan_inf(data, "root")
    assert issues == []


@pytest.mark.unit
def test_check_for_nan_inf_detects_nan():
    """check_for_nan_inf detects NaN values."""
    data = {"a": float("nan")}
    issues = check_for_nan_inf(data, "root")
    assert len(issues) == 1
    assert "NaN" in issues[0]


@pytest.mark.unit
def test_check_for_nan_inf_detects_inf():
    """check_for_nan_inf detects Inf values."""
    data = {"a": float("inf")}
    issues = check_for_nan_inf(data, "root")
    assert len(issues) == 1
    assert "Inf" in issues[0]


# =============================================================================
# Integration Tests: Happy Path
# =============================================================================

@pytest.mark.unit
def test_happy_path_all_pass(valid_temporal_audit, valid_variance_audit):
    """Happy path: all thresholds pass, report.passed=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is True
        assert report.temporal_comparability is True
        assert report.variance_comparability is True
        assert report.claim_cap_applied is False
        assert report.claim_cap_level is None
        assert report.f5_failure_codes == []


@pytest.mark.unit
def test_happy_path_output_format(valid_temporal_audit, valid_variance_audit):
    """Happy path: to_dict() has correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)
        d = report.to_dict()

        assert d["verdict"] == "PASS"
        assert d["temporal_comparability"] is True
        assert d["variance_comparability"] is True
        assert d["f5_failure_codes"] == []
        assert d["claim_cap_applied"] is False


# =============================================================================
# Integration Tests: F5.1 Temporal Structure Failures
# =============================================================================

@pytest.mark.unit
def test_f5_1_cycle_count_mismatch(valid_temporal_audit, valid_variance_audit):
    """F5.1: cycle_count_match=false triggers failure."""
    valid_temporal_audit["comparability"]["cycle_count_match"] = False
    valid_temporal_audit["comparability"]["temporal_structure_pass"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.1" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_1_cycle_indices_not_identical(valid_temporal_audit, valid_variance_audit):
    """F5.1: cycle_indices_identical=false triggers failure."""
    valid_temporal_audit["comparability"]["cycle_indices_identical"] = False
    valid_temporal_audit["comparability"]["temporal_structure_pass"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.1" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_1_baseline_not_monotonic(valid_temporal_audit, valid_variance_audit):
    """F5.1: baseline monotonic_cycle_indices=false triggers failure."""
    valid_temporal_audit["baseline_arm"]["monotonic_cycle_indices"] = False
    valid_temporal_audit["comparability"]["temporal_structure_pass"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.1" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_1_treatment_not_monotonic(valid_temporal_audit, valid_variance_audit):
    """F5.1: treatment monotonic_cycle_indices=false triggers failure."""
    valid_temporal_audit["treatment_arm"]["monotonic_cycle_indices"] = False
    valid_temporal_audit["comparability"]["temporal_structure_pass"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.1" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_1_temporal_structure_pass_false(valid_temporal_audit, valid_variance_audit):
    """F5.1: temporal_structure_pass=false triggers failure."""
    valid_temporal_audit["comparability"]["temporal_structure_pass"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.1" in report.f5_failure_codes


# =============================================================================
# Integration Tests: F5.2 Variance Ratio Failures
# =============================================================================

@pytest.mark.unit
def test_f5_2_variance_ratio_not_acceptable(valid_temporal_audit, valid_variance_audit):
    """F5.2: variance_ratio_acceptable=false triggers failure."""
    valid_variance_audit["comparability"]["variance_ratio_acceptable"] = False
    valid_variance_audit["comparability"]["variance_ratio"] = 3.5
    valid_variance_audit["comparability"]["profile_compatible"] = False
    valid_variance_audit["comparability"]["variance_profile_pass"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.variance_comparability is False
        assert "F5.2" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_2_profile_not_compatible(valid_temporal_audit, valid_variance_audit):
    """F5.2: profile_compatible=false triggers fail-close."""
    valid_variance_audit["comparability"]["profile_compatible"] = False
    valid_variance_audit["comparability"]["variance_profile_pass"] = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.variance_comparability is False
        assert "F5.2" in report.f5_failure_codes


# =============================================================================
# Integration Tests: F5.3 Windowed Drift Failures
# =============================================================================

@pytest.mark.unit
def test_f5_3_windowed_drift_not_acceptable(valid_temporal_audit, valid_variance_audit):
    """F5.3: windowed_drift_acceptable=false triggers failure."""
    valid_variance_audit["comparability"]["windowed_drift_acceptable"] = False
    valid_variance_audit["comparability"]["windowed_variance_drift"] = 0.15
    # Note: profile_compatible stays True, so this is a WARN not FAIL
    # To make it fail-close, set profile_compatible=False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.variance_comparability is False
        assert "F5.3" in report.f5_failure_codes


# =============================================================================
# Integration Tests: F5.4 Missing Audit Artifact
# =============================================================================

@pytest.mark.unit
def test_f5_4_missing_temporal_audit(valid_variance_audit):
    """F5.4: Missing temporal_structure_audit.json triggers failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, None, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.4" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_4_missing_variance_audit(valid_temporal_audit):
    """F5.4: Missing variance_profile_audit.json triggers failure and cap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, None)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.variance_comparability is False
        assert "F5.4" in report.f5_failure_codes
        assert report.claim_cap_applied is True
        assert report.claim_cap_level == "L3"


@pytest.mark.unit
def test_f5_4_missing_both_audits():
    """F5.4: Missing both audits triggers failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, None, None)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert report.variance_comparability is False
        assert "F5.4" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_4_run_dir_not_found():
    """F5.4: Non-existent run directory triggers failure."""
    report = verify_run(Path("/nonexistent/run/dir"))

    assert report.passed is False
    assert "F5.4" in report.f5_failure_codes


# =============================================================================
# Integration Tests: F5.5 Schema Validation Failures
# =============================================================================

@pytest.mark.unit
def test_f5_5_wrong_schema_version(valid_temporal_audit, valid_variance_audit):
    """F5.5: Wrong schema_version triggers failure."""
    valid_temporal_audit["schema_version"] = "2.0.0"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert "F5.5" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_5_wrong_experiment_id(valid_temporal_audit, valid_variance_audit):
    """F5.5: Wrong experiment_id triggers failure."""
    valid_temporal_audit["experiment_id"] = "CAL-EXP-3"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert "F5.5" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_5_missing_required_field(valid_temporal_audit, valid_variance_audit):
    """F5.5: Missing required field triggers failure."""
    del valid_temporal_audit["comparability"]

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert "F5.5" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_5_missing_arm_field(valid_temporal_audit, valid_variance_audit):
    """F5.5: Missing arm field triggers failure."""
    del valid_temporal_audit["baseline_arm"]["cycle_count"]

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert "F5.5" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_5_malformed_json():
    """F5.5: Malformed JSON triggers failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir()
        (run_dir / "validity").mkdir()

        # Write malformed JSON
        with open(run_dir / "validity" / "temporal_structure_audit.json", "w") as f:
            f.write("{not valid json}")

        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False


# =============================================================================
# Integration Tests: F5.6 Pathological Data (NaN/Inf)
# =============================================================================

@pytest.mark.unit
def test_f5_6_nan_in_temporal_audit(valid_temporal_audit, valid_variance_audit):
    """F5.6: NaN in temporal audit triggers failure."""
    valid_temporal_audit["baseline_arm"]["cycle_gap_mean"] = float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.6" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_6_inf_in_temporal_audit(valid_temporal_audit, valid_variance_audit):
    """F5.6: Inf in temporal audit triggers failure."""
    valid_temporal_audit["treatment_arm"]["temporal_coverage_ratio"] = float("inf")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.temporal_comparability is False
        assert "F5.6" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_6_nan_in_variance_audit(valid_temporal_audit, valid_variance_audit):
    """F5.6: NaN in variance audit triggers failure."""
    valid_variance_audit["baseline_arm"]["delta_p_variance"] = float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.variance_comparability is False
        assert "F5.6" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_6_has_nan_flag_baseline(valid_temporal_audit, valid_variance_audit):
    """F5.6: has_nan=true in baseline triggers failure."""
    valid_variance_audit["baseline_arm"]["has_nan"] = True

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.variance_comparability is False
        assert "F5.6" in report.f5_failure_codes


@pytest.mark.unit
def test_f5_6_has_inf_flag_treatment(valid_temporal_audit, valid_variance_audit):
    """F5.6: has_inf=true in treatment triggers failure."""
    valid_variance_audit["treatment_arm"]["has_inf"] = True

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert report.variance_comparability is False
        assert "F5.6" in report.f5_failure_codes


# =============================================================================
# Integration Tests: F5.7 IQR Ratio Failures
# =============================================================================

@pytest.mark.unit
def test_f5_7_iqr_ratio_not_acceptable(valid_temporal_audit, valid_variance_audit):
    """F5.7: iqr_ratio_acceptable=false triggers failure."""
    valid_variance_audit["comparability"]["iqr_ratio_acceptable"] = False
    valid_variance_audit["comparability"]["iqr_ratio"] = 3.5

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.variance_comparability is False
        assert "F5.7" in report.f5_failure_codes


# =============================================================================
# Integration Tests: Claim Capping
# =============================================================================

@pytest.mark.unit
def test_claim_cap_from_audit(valid_temporal_audit, valid_variance_audit):
    """Claim cap from audit is propagated to report."""
    valid_variance_audit["comparability"]["claim_cap_applied"] = True
    valid_variance_audit["comparability"]["claim_cap_level"] = "L2"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.claim_cap_applied is True
        assert report.claim_cap_level == "L2"


@pytest.mark.unit
def test_claim_cap_l3_on_missing_variance_audit(valid_temporal_audit):
    """Missing variance audit triggers L3 claim cap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, None)
        report = verify_run(run_dir)

        assert report.claim_cap_applied is True
        assert report.claim_cap_level == "L3"


# =============================================================================
# Integration Tests: JSON Report Output
# =============================================================================

@pytest.mark.unit
def test_json_report_write(valid_temporal_audit, valid_variance_audit):
    """write_json creates valid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        report_path = Path(tmpdir) / "report.json"
        report.write_json(report_path)

        assert report_path.exists()
        with open(report_path) as f:
            loaded = json.load(f)

        assert loaded["verdict"] == "PASS"
        assert loaded["verifier"] == "verify_cal_exp_4_run.py"
        assert "temporal_comparability" in loaded
        assert "f5_failure_codes" in loaded


# =============================================================================
# Adversarial Tests
# =============================================================================

@pytest.mark.unit
def test_adversarial_negative_variance(valid_temporal_audit, valid_variance_audit):
    """Negative variance value in arm data."""
    # Note: Schema allows minimum: 0, so -1 would be invalid
    # But verifier reads from artifact, doesn't recompute
    valid_variance_audit["baseline_arm"]["delta_p_variance"] = -0.01

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        # Verifier doesn't validate numeric ranges, just reads flags
        # This should still pass if flags are true
        assert report.passed is True


@pytest.mark.unit
def test_adversarial_empty_comparability(valid_temporal_audit, valid_variance_audit):
    """Empty comparability object triggers schema failure."""
    valid_temporal_audit["comparability"] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        assert report.passed is False
        assert "F5.5" in report.f5_failure_codes


@pytest.mark.unit
def test_adversarial_extra_fields_allowed(valid_temporal_audit, valid_variance_audit):
    """Extra fields in audit should be ignored (not fail)."""
    valid_temporal_audit["extra_field"] = "ignored"
    valid_temporal_audit["baseline_arm"]["extra_field"] = "ignored"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_run_dir(tmpdir, valid_temporal_audit, valid_variance_audit)
        report = verify_run(run_dir)

        # Verifier ignores extra fields
        assert report.passed is True


# =============================================================================
# Red Team Fixture Test
# =============================================================================

@pytest.mark.unit
def test_adversarial_fixture_triggers_all_f5_failures():
    """
    Red Team fixture must trigger F5.1, F5.2, F5.3, F5.7 failures.

    This test uses the pre-built adversarial fixture at:
    tests/fixtures/cal_exp_4_adversarial/

    Expected outcomes:
    - temporal_comparability = False
    - variance_comparability = False
    - f5_failure_codes includes F5.1, F5.2, F5.3, F5.7
    - claim_cap_applied = True
    - claim_cap_level = "L3"
    - VERDICT = FAIL
    - No verifier crash
    """
    fixture_dir = Path(__file__).parent.parent / "fixtures" / "cal_exp_4_adversarial"

    # Skip if fixture doesn't exist (CI may not have it)
    if not fixture_dir.exists():
        pytest.skip("Adversarial fixture not found")

    report = verify_run(fixture_dir)

    # Verifier must not crash
    assert report is not None

    # Must FAIL
    assert report.passed is False, "Adversarial fixture must trigger FAIL"

    # temporal_comparability must be False
    assert report.temporal_comparability is False, (
        "temporal_comparability must be False for adversarial fixture"
    )

    # variance_comparability must be False
    assert report.variance_comparability is False, (
        "variance_comparability must be False for adversarial fixture"
    )

    # F5.1 must be triggered (temporal structure failures)
    assert "F5.1" in report.f5_failure_codes, (
        f"F5.1 not in failure codes: {report.f5_failure_codes}"
    )

    # F5.2 must be triggered (variance ratio failure)
    assert "F5.2" in report.f5_failure_codes, (
        f"F5.2 not in failure codes: {report.f5_failure_codes}"
    )

    # F5.3 must be triggered (windowed drift failure)
    assert "F5.3" in report.f5_failure_codes, (
        f"F5.3 not in failure codes: {report.f5_failure_codes}"
    )

    # F5.7 must be triggered (IQR ratio failure)
    assert "F5.7" in report.f5_failure_codes, (
        f"F5.7 not in failure codes: {report.f5_failure_codes}"
    )

    # Claim must be capped to L3
    assert report.claim_cap_applied is True, "claim_cap_applied must be True"
    assert report.claim_cap_level == "L3", (
        f"claim_cap_level must be L3, got {report.claim_cap_level}"
    )

    # Verify check counts
    assert report.fail_count >= 9, f"Expected >= 9 FAIL, got {report.fail_count}"
    assert report.warn_count >= 3, f"Expected >= 3 WARN, got {report.warn_count}"
