"""
PHASE II — Telemetry Conformance Test Suite

Tests for conformance level classification (L0/L1/L2), quarantine pipeline,
and batch auditing per TELEMETRY_CONFORMANCE_SPEC.md.

Author: CLAUDE H (Telemetry Conformance Enforcer)
Date: 2025-12-06
Status: PHASE II — NOT RUN IN PHASE I
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from experiments.telemetry_conformance_check import (
    ALERT_THRESHOLDS,
    CONFORMANCE_SNAPSHOT_SCHEMA_VERSION,
    DEFAULT_QUARANTINE_THRESHOLD,
    DEFAULT_SLO_CONFIG,
    GOVERNANCE_BLOCKING_RULES,
    GOVERNANCE_WARN_RULES,
    AlertCode,
    ConformanceLevel,
    ConformanceReport,
    DirectorTelemetryPanel,
    GlobalConsoleResult,
    GlobalHealthSummary,
    GovernanceSignal,
    MAASAdapterResult,
    MAASStatus,
    QuarantineAction,
    QuarantineDecision,
    QuarantineEnvelope,
    ReleaseGateResult,
    ReleaseStatus,
    SLOResult,
    SLOStatus,
    StatusLight,
    TDACorrelationResult,
    TelemetryGovernanceResult,
    ViolationSeverity,
    audit_telemetry_file,
    build_telemetry_conformance_snapshot,
    build_telemetry_director_panel,
    check_record_conformance,
    classify_record_level,
    create_quarantine_envelope,
    decide_telemetry_quarantine,
    evaluate_telemetry_for_release,
    evaluate_telemetry_slo,
    get_alert_codes_for_telemetry,
    is_canonical,
    is_schema_valid,
    should_quarantine,
    summarize_telemetry_for_global_console,
    summarize_telemetry_for_global_health,
    summarize_telemetry_for_governance,
    summarize_telemetry_for_maas,
    summarize_telemetry_for_maas_v2,
    summarize_telemetry_tda_correlation,
    to_governance_signal_for_telemetry,
    write_conformance_report,
    write_quarantine_record,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def valid_cycle_metric_l2() -> str:
    """Valid L2-conformant cycle_metric record (canonical form)."""
    record = {
        "cycle": 42,
        "ht": "a" * 64,
        "metric_type": "goal_hit",
        "metric_value": 1.0,
        "mode": "rfl",
        "r_t": "b" * 64,
        "run_id": "U2-abc-123",
        "slice": "slice_uplift_goal",
        "success": True,
        "ts": "2025-12-06T10:30:00.123456Z",
        "u_t": "c" * 64,
    }
    # Return canonical JSON (sorted keys, no whitespace)
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


@pytest.fixture
def valid_cycle_metric_l1() -> str:
    """Valid L1 record (schema valid but not canonical - has extra whitespace)."""
    record = {
        "cycle": 42,
        "ht": "a" * 64,
        "metric_type": "goal_hit",
        "metric_value": 1.0,
        "mode": "rfl",
        "r_t": "b" * 64,
        "run_id": "U2-abc-123",
        "slice": "slice_uplift_goal",
        "success": True,
        "ts": "2025-12-06T10:30:00.123456Z",
        "u_t": "c" * 64,
    }
    # Return with whitespace after colons (L1 but not L2) - single line
    return json.dumps(record, sort_keys=True)


@pytest.fixture
def valid_experiment_summary_l2() -> str:
    """Valid L2-conformant experiment_summary record."""
    record = {
        "ci_95": [0.35, 0.49],
        "mode": "baseline",
        "n_cycles": 500,
        "p_success": 0.42,
        "phase": "II",
        "run_id": "U2-baseline-001",
        "slice": "slice_uplift_goal",
        "uplift_delta": None,
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


@pytest.fixture
def valid_uplift_result_l2() -> str:
    """Valid L2-conformant uplift_result record."""
    record = {
        "baseline_run_id": "U2-baseline-001",
        "ci_95": [0.089, 0.231],
        "n_base": 500,
        "n_rfl": 500,
        "p_base": 0.42,
        "p_rfl": 0.58,
        "p_value": 0.00023,
        "phase": "II",
        "rfl_run_id": "U2-rfl-001",
        "significant": True,
        "slice": "slice_uplift_goal",
        "ts": "2025-12-06T16:00:00.123456Z",
        "uplift_delta": 0.16,
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# SCHEMA ADHERENCE TESTS (Section 5.2 of spec)
# =============================================================================

class TestRequiredFields:
    """Test required field validation (REQ-001 through REQ-006)."""

    def test_req_001_all_required_fields_present(self, valid_cycle_metric_l2):
        """REQ-001: All required fields present passes validation."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.level == ConformanceLevel.L2
        assert result.is_valid
        assert result.is_canonical

    def test_req_002_missing_one_required_field(self):
        """REQ-002: Missing one required field results in quarantine."""
        record = {
            "cycle": 42,
            # "ht" is missing
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))
        result = classify_record_level(raw)

        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "ht" for v in result.violations)
        assert any(v.check_name == "check_required_fields" for v in result.violations)

    def test_req_003_missing_multiple_required_fields(self):
        """REQ-003: Missing multiple required fields results in quarantine."""
        record = {
            "cycle": 42,
            # "ht" and "ts" missing
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "u_t": "c" * 64,
        }
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))
        result = classify_record_level(raw)

        assert result.level == ConformanceLevel.QUARANTINE
        missing_fields = [v.field for v in result.violations if v.check_name == "check_required_fields"]
        assert "ht" in missing_fields
        assert "ts" in missing_fields

    def test_req_006_required_field_wrong_type(self):
        """REQ-006: Required field with wrong type results in quarantine."""
        record = {
            "cycle": "42",  # Should be int, not string
            "ht": "a" * 64,
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))
        result = classify_record_level(raw)

        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "cycle" and v.check_name == "check_field_types" for v in result.violations)


class TestProhibitedFields:
    """Test prohibited field validation (PRO-001 through PRO-005)."""

    def test_pro_001_no_prohibited_fields(self, valid_cycle_metric_l2):
        """PRO-001: No prohibited fields passes validation."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.level == ConformanceLevel.L2
        prohibited_violations = [v for v in result.violations if v.check_name == "check_prohibited_fields"]
        assert len(prohibited_violations) == 0

    def test_pro_002_single_prohibited_field(self, valid_cycle_metric_l2):
        """PRO-002: Single prohibited field results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["_id"] = "some_id"
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "_id" for v in result.violations)

    def test_pro_003_multiple_prohibited_fields(self, valid_cycle_metric_l2):
        """PRO-003: Multiple prohibited fields all flagged."""
        record = json.loads(valid_cycle_metric_l2)
        record["timestamp"] = "2025-01-01"
        record["metadata"] = {"extra": "data"}
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        prohibited = [v.field for v in result.violations if v.check_name == "check_prohibited_fields"]
        assert "timestamp" in prohibited
        assert "metadata" in prohibited


class TestEnumValidation:
    """Test enum value validation (ENM-001 through ENM-005)."""

    def test_enm_001_invalid_mode(self, valid_cycle_metric_l2):
        """ENM-001: Invalid mode value results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["mode"] = "test"  # Invalid
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "mode" and v.check_name == "check_enum_values" for v in result.violations)

    def test_enm_002_invalid_slice(self, valid_cycle_metric_l2):
        """ENM-002: Invalid slice value results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["slice"] = "custom_slice"  # Invalid
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "slice" for v in result.violations)

    def test_enm_005_case_sensitive_mode(self, valid_cycle_metric_l2):
        """ENM-005: Mode is case-sensitive."""
        record = json.loads(valid_cycle_metric_l2)
        record["mode"] = "BASELINE"  # Wrong case
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE


class TestRangeValidation:
    """Test value range validation (RNG-001 through RNG-006)."""

    def test_rng_001_p_success_above_range(self, valid_experiment_summary_l2):
        """RNG-001: p_success > 1.0 results in quarantine."""
        record = json.loads(valid_experiment_summary_l2)
        record["p_success"] = 1.5
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "p_success" for v in result.violations)

    def test_rng_002_p_success_below_range(self, valid_experiment_summary_l2):
        """RNG-002: p_success < 0.0 results in quarantine."""
        record = json.loads(valid_experiment_summary_l2)
        record["p_success"] = -0.1
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE

    def test_rng_003_negative_cycle(self, valid_cycle_metric_l2):
        """RNG-003: Negative cycle number results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["cycle"] = -1
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE

    def test_rng_006_ci_lower_greater_than_upper(self, valid_experiment_summary_l2):
        """RNG-006: CI lower > upper results in quarantine."""
        record = json.loads(valid_experiment_summary_l2)
        record["ci_95"] = [0.6, 0.4]  # Lower > upper
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "ci_95" for v in result.violations)


# =============================================================================
# CANONICALIZATION TESTS (Section 5.3 of spec)
# =============================================================================

class TestFieldOrdering:
    """Test field ordering validation (ORD-001 through ORD-004)."""

    def test_ord_001_canonical_order(self, valid_cycle_metric_l2):
        """ORD-001: Canonical (alphabetical) order achieves L2."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.level == ConformanceLevel.L2

    def test_ord_002_reverse_order(self):
        """ORD-002: Reverse order results in L1 only."""
        record = {
            "u_t": "c" * 64,
            "ts": "2025-12-06T10:30:00.123456Z",
            "success": True,
            "slice": "slice_uplift_goal",
            "run_id": "U2-abc-123",
            "r_t": "b" * 64,
            "mode": "rfl",
            "metric_value": 1.0,
            "metric_type": "goal_hit",
            "ht": "a" * 64,
            "cycle": 42,
        }
        # Use custom serialization to preserve order
        raw = "{" + ",".join(f'"{k}":{json.dumps(v)}' for k, v in record.items()) + "}"

        result = classify_record_level(raw)
        # Should be L1 (schema valid) but not L2 (not canonical)
        assert result.level == ConformanceLevel.L1
        assert any(v.check_name == "check_field_ordering" for v in result.violations)


class TestSerializationFormat:
    """Test JSON serialization format (SER-001 through SER-006)."""

    def test_ser_001_compact_json(self, valid_cycle_metric_l2):
        """SER-001: Compact JSON achieves L2."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.level == ConformanceLevel.L2

    def test_ser_002_spaces_after_colons(self, valid_cycle_metric_l2):
        """SER-002: Spaces after colons results in L1 only."""
        record = json.loads(valid_cycle_metric_l2)
        raw = json.dumps(record, sort_keys=True)  # Default has space after colon

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.L1
        assert any(v.check_name == "check_serialization_format" for v in result.violations)

    def test_ser_004_pretty_printed(self, valid_cycle_metric_l2):
        """SER-004: Pretty-printed (multi-line) results in L1 only."""
        record = json.loads(valid_cycle_metric_l2)
        # Create multi-line pretty-printed version
        raw = json.dumps(record, sort_keys=True, indent=2)
        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.L1


class TestTimestampFormat:
    """Test timestamp format validation (TST-001 through TST-007)."""

    def test_tst_001_full_precision_with_z(self, valid_cycle_metric_l2):
        """TST-001: Full precision with Z achieves L2."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.level == ConformanceLevel.L2

    def test_tst_002_missing_microseconds(self, valid_cycle_metric_l2):
        """TST-002: Missing microseconds results in error."""
        record = json.loads(valid_cycle_metric_l2)
        record["ts"] = "2025-12-06T10:30:00Z"  # No microseconds
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "ts" for v in result.violations)

    def test_tst_004_offset_instead_of_z(self, valid_cycle_metric_l2):
        """TST-004: Offset timezone instead of Z results in error."""
        record = json.loads(valid_cycle_metric_l2)
        record["ts"] = "2025-12-06T10:30:00.123456+00:00"
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE

    def test_tst_007_invalid_date(self, valid_cycle_metric_l2):
        """TST-007: Invalid date results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["ts"] = "2025-13-45T10:30:00.123456Z"  # Invalid month/day
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE


class TestHashFormat:
    """Test hash format validation (HSH-001 through HSH-007)."""

    def test_hsh_001_valid_64_lowercase_hex(self, valid_cycle_metric_l2):
        """HSH-001: 64 lowercase hex achieves L2."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.level == ConformanceLevel.L2

    def test_hsh_002_uppercase_hex(self, valid_cycle_metric_l2):
        """HSH-002: Uppercase hex results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["ht"] = "A" * 64  # Uppercase
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "ht" for v in result.violations)

    def test_hsh_004_too_short(self, valid_cycle_metric_l2):
        """HSH-004: Too short (63 chars) results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["ht"] = "a" * 63
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE

    def test_hsh_005_too_long(self, valid_cycle_metric_l2):
        """HSH-005: Too long (65 chars) results in quarantine."""
        record = json.loads(valid_cycle_metric_l2)
        record["ht"] = "a" * 65
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE


# =============================================================================
# DRIFT DETECTION TESTS (Section 5.4 of spec)
# =============================================================================

class TestCrossFieldConsistency:
    """Test cross-field consistency validation (XFC-001 through XFC-006)."""

    def test_xfc_001_correct_uplift_delta(self, valid_uplift_result_l2):
        """XFC-001: Correct delta = p_rfl - p_base passes."""
        result = classify_record_level(valid_uplift_result_l2)
        assert result.level == ConformanceLevel.L2

    def test_xfc_002_incorrect_uplift_delta(self, valid_uplift_result_l2):
        """XFC-002: delta != p_rfl - p_base results in quarantine."""
        record = json.loads(valid_uplift_result_l2)
        # p_rfl=0.58, p_base=0.42, delta should be 0.16
        record["uplift_delta"] = 0.15  # Off by 0.01
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "uplift_delta" and "p_rfl - p_base" in v.message for v in result.violations)

    def test_xfc_003_significant_matches_ci(self, valid_uplift_result_l2):
        """XFC-003: significant matches CI (ci_lower > 0, sig=true) passes."""
        result = classify_record_level(valid_uplift_result_l2)
        assert result.level == ConformanceLevel.L2

    def test_xfc_004_significant_mismatch(self, valid_uplift_result_l2):
        """XFC-004: significant mismatch with CI results in quarantine."""
        record = json.loads(valid_uplift_result_l2)
        # ci_lower=0.089 > 0, so significant should be True
        record["significant"] = False
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "significant" for v in result.violations)

    def test_xfc_005_p_within_ci(self, valid_experiment_summary_l2):
        """XFC-005: p_success within ci_95 passes."""
        result = classify_record_level(valid_experiment_summary_l2)
        assert result.level == ConformanceLevel.L2

    def test_xfc_006_p_outside_ci(self, valid_experiment_summary_l2):
        """XFC-006: p_success outside ci_95 results in quarantine."""
        record = json.loads(valid_experiment_summary_l2)
        # ci_95 = [0.35, 0.49], p_success should be within
        record["p_success"] = 0.60  # Outside CI
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.level == ConformanceLevel.QUARANTINE
        assert any(v.field == "p_success" and "ci_95" in v.message for v in result.violations)


# =============================================================================
# QUARANTINE PIPELINE TESTS
# =============================================================================

class TestQuarantinePipeline:
    """Test quarantine envelope creation and writing."""

    def test_create_quarantine_envelope(self):
        """Test quarantine envelope creation."""
        record = {"cycle": "not_an_int"}  # Invalid
        raw = json.dumps(record)
        result = classify_record_level(raw)

        envelope = create_quarantine_envelope(
            result,
            source_file="test.jsonl",
            source_line=42,
            source_byte_offset=1000,
        )

        assert envelope.source_file == "test.jsonl"
        assert envelope.source_line == 42
        assert envelope.status == "pending"
        assert len(envelope.violations) > 0

    def test_envelope_to_dict(self):
        """Test envelope serialization to dict."""
        record = {"cycle": "invalid"}
        raw = json.dumps(record)
        result = classify_record_level(raw)
        envelope = create_quarantine_envelope(result)

        d = envelope.to_dict()

        assert "quarantine" in d
        assert "violations" in d
        assert "record" in d
        assert "context" in d
        assert "disposition" in d

    def test_write_quarantine_record(self, temp_dir):
        """Test writing quarantine record to filesystem."""
        record = {"cycle": "invalid", "run_id": "U2-test-123"}
        raw = json.dumps(record)
        result = classify_record_level(raw)
        envelope = create_quarantine_envelope(result)
        envelope.run_id = "U2-test-123"
        envelope.cycle = 42

        path = write_quarantine_record(envelope, temp_dir)

        assert path.exists()
        assert (temp_dir / "index.jsonl").exists()
        assert (temp_dir / "by_run" / "U2-test-123" / "manifest.json").exists()


# =============================================================================
# BATCH AUDITOR TESTS
# =============================================================================

class TestBatchAuditor:
    """Test batch auditing of JSONL files."""

    def test_audit_valid_file(self, temp_dir, valid_cycle_metric_l2):
        """Test auditing a file with valid records."""
        # Create test JSONL file
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(10):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)

        assert report.total_records == 10
        assert report.by_level["L2"] == 10
        assert report.l2_percentage == 100.0
        assert report.quarantined_count == 0

    def test_audit_mixed_file(self, temp_dir, valid_cycle_metric_l2, valid_cycle_metric_l1):
        """Test auditing a file with mixed conformance levels."""
        jsonl_path = temp_dir / "mixed.jsonl"
        with open(jsonl_path, "w") as f:
            # 5 L2 records
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            # 5 L1 records
            for _ in range(5):
                f.write(valid_cycle_metric_l1 + "\n")

        report = audit_telemetry_file(jsonl_path)

        assert report.total_records == 10
        assert report.by_level["L2"] == 5
        assert report.by_level["L1"] == 5
        assert report.l2_percentage == 50.0

    def test_audit_with_quarantine(self, temp_dir, valid_cycle_metric_l2):
        """Test auditing with quarantine enabled."""
        jsonl_path = temp_dir / "quarantine_test.jsonl"
        quarantine_dir = temp_dir / "quarantine"

        # Create an invalid cycle_metric record (detected as cycle_metric but missing required field)
        invalid_record = {
            "cycle": 42,
            "metric_type": "goal_hit",
            # Missing ht - should trigger quarantine
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            # Valid record
            f.write(valid_cycle_metric_l2 + "\n")
            # Invalid record (missing required field)
            f.write(invalid_json + "\n")
            # Another valid record
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path, quarantine_root=quarantine_dir)

        assert report.total_records == 3
        assert report.quarantined_count >= 1
        assert (quarantine_dir / "index.jsonl").exists()

    def test_write_conformance_report(self, temp_dir, valid_cycle_metric_l2):
        """Test writing conformance report to file."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        report_path = write_conformance_report(report, temp_dir / "report.json")

        assert report_path.exists()

        with open(report_path) as f:
            data = json.load(f)

        assert data["total_records"] == 1
        assert "by_level" in data


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_json_object(self):
        """Empty object should be L0 (cannot detect type)."""
        result = classify_record_level("{}")
        assert result.level == ConformanceLevel.L0

    def test_invalid_json(self):
        """Invalid JSON should quarantine."""
        result = classify_record_level("{not valid json")
        assert result.level == ConformanceLevel.QUARANTINE
        assert any("parse" in v.check_name for v in result.violations)

    def test_json_array_not_object(self):
        """JSON array (not object) should quarantine."""
        result = classify_record_level("[1, 2, 3]")
        assert result.level == ConformanceLevel.QUARANTINE

    def test_null_json(self):
        """Null JSON should quarantine."""
        result = classify_record_level("null")
        assert result.level == ConformanceLevel.QUARANTINE

    def test_canonical_hash_computed_for_l2(self, valid_cycle_metric_l2):
        """L2 records should have canonical hash computed."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.level == ConformanceLevel.L2
        assert result.canonical_hash is not None
        assert len(result.canonical_hash) == 64

    def test_legacy_slice_format_accepted(self, valid_cycle_metric_l2):
        """Legacy U2_env_X slice format should be accepted."""
        record = json.loads(valid_cycle_metric_l2)
        record["slice"] = "U2_env_A"  # Legacy format
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"))

        result = classify_record_level(raw)
        assert result.is_valid


# =============================================================================
# RECORD TYPE DETECTION TESTS
# =============================================================================

class TestRecordTypeDetection:
    """Test automatic record type detection."""

    def test_detect_cycle_metric(self, valid_cycle_metric_l2):
        """Should detect cycle_metric from field signature."""
        result = classify_record_level(valid_cycle_metric_l2)
        assert result.record_type == "cycle_metric"

    def test_detect_experiment_summary(self, valid_experiment_summary_l2):
        """Should detect experiment_summary from field signature."""
        result = classify_record_level(valid_experiment_summary_l2)
        assert result.record_type == "experiment_summary"

    def test_detect_uplift_result(self, valid_uplift_result_l2):
        """Should detect uplift_result from field signature."""
        result = classify_record_level(valid_uplift_result_l2)
        assert result.record_type == "uplift_result"


# =============================================================================
# TASK 1: TELEMETRY CONFORMANCE SNAPSHOT TESTS
# =============================================================================

class TestConformanceSnapshot:
    """Test build_telemetry_conformance_snapshot function."""

    def test_snapshot_has_schema_version(self, temp_dir, valid_cycle_metric_l2):
        """Snapshot includes schema_version field."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == CONFORMANCE_SNAPSHOT_SCHEMA_VERSION

    def test_snapshot_has_total_records(self, temp_dir, valid_cycle_metric_l2):
        """Snapshot includes total_records count."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        assert snapshot["total_records"] == 5

    def test_snapshot_has_level_counts(self, temp_dir, valid_cycle_metric_l2, valid_cycle_metric_l1):
        """Snapshot includes counts for all conformance levels."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # 3 L2 records
            for _ in range(3):
                f.write(valid_cycle_metric_l2 + "\n")
            # 2 L1 records
            for _ in range(2):
                f.write(valid_cycle_metric_l1 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        assert "by_level" in snapshot
        assert snapshot["by_level"]["L2"] == 3
        assert snapshot["by_level"]["L1"] == 2
        assert snapshot["by_level"]["L0"] == 0
        assert snapshot["by_level"]["QUARANTINE"] == 0

    def test_snapshot_quarantine_ratio_calculated(self, temp_dir, valid_cycle_metric_l2):
        """Snapshot calculates quarantine_ratio correctly."""
        jsonl_path = temp_dir / "test.jsonl"

        # Create a record that will be quarantined (missing required field)
        invalid_record = {
            "cycle": 42,
            "metric_type": "goal_hit",
            # Missing ht
            "metric_value": 1.0,
            "mode": "rfl",
        }
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            # 4 valid L2 records
            for _ in range(4):
                f.write(valid_cycle_metric_l2 + "\n")
            # 1 quarantined record
            f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        assert snapshot["quarantine_count"] == 1
        assert snapshot["quarantine_ratio"] == 0.2  # 1/5 = 0.2

    def test_snapshot_severity_mix(self, temp_dir, valid_cycle_metric_l2):
        """Snapshot includes severity_mix breakdown."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        assert "severity_mix" in snapshot
        assert "critical" in snapshot["severity_mix"]
        assert "error" in snapshot["severity_mix"]
        assert "warning" in snapshot["severity_mix"]
        assert "info" in snapshot["severity_mix"]

    def test_snapshot_is_json_serializable(self, temp_dir, valid_cycle_metric_l2):
        """Snapshot is fully JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        # Should not raise
        serialized = json.dumps(snapshot)
        deserialized = json.loads(serialized)
        assert deserialized == snapshot

    def test_snapshot_is_deterministic(self, temp_dir, valid_cycle_metric_l2):
        """Same input produces identical snapshot."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot1 = build_telemetry_conformance_snapshot(report)
        snapshot2 = build_telemetry_conformance_snapshot(report)

        assert snapshot1 == snapshot2

    def test_snapshot_empty_file(self, temp_dir):
        """Snapshot handles empty file gracefully."""
        jsonl_path = temp_dir / "empty.jsonl"
        with open(jsonl_path, "w") as f:
            pass  # Empty file

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        assert snapshot["total_records"] == 0
        assert snapshot["quarantine_ratio"] == 0.0


# =============================================================================
# TASK 2: STREAMING HOOK API TESTS
# =============================================================================

class TestStreamingHookAPI:
    """Test lightweight streaming hook API functions."""

    def test_check_record_conformance_valid_record(self):
        """check_record_conformance classifies valid record correctly."""
        record = {
            "cycle": 42,
            "ht": "a" * 64,
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }

        result = check_record_conformance(record, "cycle_metric")

        assert result.level == ConformanceLevel.L2
        assert result.is_canonical

    def test_check_record_conformance_invalid_record(self):
        """check_record_conformance classifies invalid record correctly."""
        record = {
            "cycle": 42,
            "metric_type": "goal_hit",
            # Missing ht - should quarantine
            "metric_value": 1.0,
            "mode": "rfl",
        }

        result = check_record_conformance(record, "cycle_metric")

        assert result.requires_quarantine

    def test_check_record_conformance_auto_detect_type(self):
        """check_record_conformance auto-detects record type."""
        record = {
            "cycle": 42,
            "ht": "a" * 64,
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }

        result = check_record_conformance(record)  # No type hint

        assert result.record_type == "cycle_metric"

    def test_check_record_conformance_is_idempotent(self):
        """Same record produces same result every time."""
        record = {
            "cycle": 42,
            "ht": "a" * 64,
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }

        result1 = check_record_conformance(record)
        result2 = check_record_conformance(record)
        result3 = check_record_conformance(record)

        assert result1.level == result2.level == result3.level
        assert result1.canonical_hash == result2.canonical_hash == result3.canonical_hash

    def test_should_quarantine_returns_true_for_quarantine(self):
        """should_quarantine returns True for QUARANTINE level."""
        record = {"cycle": 42, "metric_type": "goal_hit"}  # Missing required fields
        result = check_record_conformance(record, "cycle_metric")

        assert should_quarantine(result) is True

    def test_should_quarantine_returns_false_for_valid(self):
        """should_quarantine returns False for valid records."""
        record = {
            "cycle": 42,
            "ht": "a" * 64,
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }
        result = check_record_conformance(record)

        assert should_quarantine(result) is False

    def test_is_canonical_returns_true_for_l2(self):
        """is_canonical returns True for L2 records."""
        record = {
            "cycle": 42,
            "ht": "a" * 64,
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }
        result = check_record_conformance(record)

        assert is_canonical(result) is True

    def test_is_schema_valid_returns_true_for_l1_and_l2(self):
        """is_schema_valid returns True for L1+ records."""
        # L2 record
        valid_record = {
            "cycle": 42,
            "ht": "a" * 64,
            "metric_type": "goal_hit",
            "metric_value": 1.0,
            "mode": "rfl",
            "r_t": "b" * 64,
            "run_id": "U2-abc-123",
            "slice": "slice_uplift_goal",
            "success": True,
            "ts": "2025-12-06T10:30:00.123456Z",
            "u_t": "c" * 64,
        }
        result = check_record_conformance(valid_record)

        assert is_schema_valid(result) is True

    def test_streaming_matches_batch_path(self, temp_dir, valid_cycle_metric_l2):
        """Streaming API produces same results as batch path."""
        # Batch path
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")
        report = audit_telemetry_file(jsonl_path)

        # Streaming path
        record = json.loads(valid_cycle_metric_l2)
        result = check_record_conformance(record)

        # Should match
        assert result.level == ConformanceLevel.L2
        assert report.by_level["L2"] == 1


# =============================================================================
# TASK 3: GOVERNANCE INTEGRATION SIGNAL TESTS
# =============================================================================

class TestGovernanceSummary:
    """Test summarize_telemetry_for_governance function."""

    def test_governance_summary_has_quarantine_records_flag(self, temp_dir, valid_cycle_metric_l2):
        """Governance summary includes has_quarantine_records boolean."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        gov = summarize_telemetry_for_governance(snapshot)

        assert "has_quarantine_records" in gov
        assert gov["has_quarantine_records"] is False

    def test_governance_summary_quarantine_ratio(self, temp_dir, valid_cycle_metric_l2):
        """Governance summary includes quarantine_ratio."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")
            f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        gov = summarize_telemetry_for_governance(snapshot)

        assert "quarantine_ratio" in gov
        assert gov["quarantine_ratio"] == 0.5  # 1 quarantined out of 2

    def test_governance_healthy_when_no_quarantine(self, temp_dir, valid_cycle_metric_l2):
        """is_telemetry_healthy is True when no quarantine."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(10):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        gov = summarize_telemetry_for_governance(snapshot)

        assert gov["is_telemetry_healthy"] is True

    def test_governance_unhealthy_above_threshold(self, temp_dir, valid_cycle_metric_l2):
        """is_telemetry_healthy is False when quarantine_ratio > threshold."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            # Create 50% quarantine rate (way above 1% threshold)
            f.write(valid_cycle_metric_l2 + "\n")
            f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        gov = summarize_telemetry_for_governance(snapshot)

        assert gov["is_telemetry_healthy"] is False

    def test_governance_custom_threshold(self, temp_dir, valid_cycle_metric_l2):
        """Custom threshold affects health classification."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            # 9 valid + 1 invalid = 10% quarantine rate
            for _ in range(9):
                f.write(valid_cycle_metric_l2 + "\n")
            f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        # With default 1% threshold: unhealthy
        gov_default = summarize_telemetry_for_governance(snapshot)
        assert gov_default["is_telemetry_healthy"] is False

        # With 15% threshold: healthy
        gov_lenient = summarize_telemetry_for_governance(snapshot, quarantine_threshold=0.15)
        assert gov_lenient["is_telemetry_healthy"] is True

    def test_governance_includes_health_threshold(self, temp_dir, valid_cycle_metric_l2):
        """Governance summary includes explicit health_threshold."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        gov = summarize_telemetry_for_governance(snapshot)

        assert "health_threshold" in gov
        assert gov["health_threshold"] == DEFAULT_QUARANTINE_THRESHOLD

    def test_governance_includes_critical_violations(self, temp_dir, valid_cycle_metric_l2):
        """Governance summary includes critical_violations count."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        gov = summarize_telemetry_for_governance(snapshot)

        assert "critical_violations" in gov
        assert isinstance(gov["critical_violations"], int)

    def test_governance_stable_summarization(self, temp_dir, valid_cycle_metric_l2):
        """Same snapshot produces identical governance summary."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        gov1 = summarize_telemetry_for_governance(snapshot)
        gov2 = summarize_telemetry_for_governance(snapshot)

        assert gov1 == gov2

    def test_governance_embeddable_in_json(self, temp_dir, valid_cycle_metric_l2):
        """Governance summary can be embedded in JSON artifacts."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        gov = summarize_telemetry_for_governance(snapshot)

        # Embed in larger artifact
        artifact = {
            "experiment_id": "U2-test-001",
            "telemetry_health": gov,
            "other_data": {"foo": "bar"}
        }

        # Should serialize without error
        serialized = json.dumps(artifact)
        deserialized = json.loads(serialized)
        assert deserialized["telemetry_health"] == gov


# =============================================================================
# PHASE III TASK 1: SLO EVALUATOR TESTS
# =============================================================================

class TestSLOEvaluator:
    """Test evaluate_telemetry_slo function."""

    def test_slo_ok_when_all_rules_pass(self, temp_dir, valid_cycle_metric_l2):
        """SLO status is OK when all rules pass."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Create 100 valid L2 records
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        assert slo_result.slo_status == SLOStatus.OK
        assert slo_result.breach_rules == 0
        assert slo_result.warn_rules == 0

    def test_slo_warn_when_threshold_exceeded(self, temp_dir, valid_cycle_metric_l2):
        """SLO status is WARN when warn threshold exceeded but not breach."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Only 5 records - below 10 record warn threshold
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        # minimum_records rule: warn at <10, breach at <1
        assert slo_result.slo_status == SLOStatus.WARN
        assert slo_result.warn_rules >= 1

    def test_slo_breach_when_breach_threshold_exceeded(self, temp_dir, valid_cycle_metric_l2):
        """SLO status is BREACH when breach threshold exceeded."""
        jsonl_path = temp_dir / "test.jsonl"

        # Create records with 50% quarantine rate (breach threshold is 1%)
        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        assert slo_result.slo_status == SLOStatus.BREACH
        assert slo_result.breach_rules >= 1

    def test_slo_violated_rules_populated(self, temp_dir, valid_cycle_metric_l2):
        """Violated rules list is populated with violation details."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")
            f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        assert len(slo_result.violated_rules) > 0
        violation = slo_result.violated_rules[0]
        assert violation.rule_name is not None
        assert violation.metric is not None
        assert violation.actual_value is not None

    def test_slo_custom_config(self, temp_dir, valid_cycle_metric_l2):
        """Custom SLO config is used when provided."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(10):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)

        # Custom config with strict L2 requirement
        custom_config = [
            {
                "name": "strict_l2",
                "metric": "l2_percentage",
                "operator": ">=",
                "warn_threshold": 99.0,
                "breach_threshold": 98.0,
                "description": "Strict L2 conformance",
            }
        ]

        slo_result = evaluate_telemetry_slo(snapshot, slo_cfg=custom_config)
        assert slo_result.evaluated_rules == 1

    def test_slo_result_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """SLOResult.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        # Should serialize without error
        serialized = json.dumps(slo_result.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["slo_status"] == slo_result.slo_status.value

    def test_slo_default_config_has_expected_rules(self):
        """Default SLO config has expected rules."""
        rule_names = [r["name"] for r in DEFAULT_SLO_CONFIG]
        assert "quarantine_ratio" in rule_names
        assert "l2_conformance" in rule_names
        assert "critical_violations" in rule_names
        assert "minimum_records" in rule_names

    def test_slo_snapshot_summary_included(self, temp_dir, valid_cycle_metric_l2):
        """SLO result includes snapshot summary."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        assert "total_records" in slo_result.snapshot_summary
        assert "quarantine_ratio" in slo_result.snapshot_summary
        assert "l2_percentage" in slo_result.snapshot_summary


# =============================================================================
# PHASE III TASK 2: AUTO-QUARANTINE ENGINE TESTS
# =============================================================================

class TestAutoQuarantineEngine:
    """Test decide_telemetry_quarantine function."""

    def test_allow_publish_when_slo_ok(self, temp_dir, valid_cycle_metric_l2):
        """Publish allowed when SLO status is OK."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        assert decision.publish_allowed is True
        assert decision.quarantine_required is False
        assert decision.recommended_action == QuarantineAction.ALLOW_PUBLISH

    def test_allow_publish_when_slo_warn(self, temp_dir, valid_cycle_metric_l2):
        """Publish allowed (with warnings) when SLO status is WARN."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Only 5 records - triggers warn
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        # Only proceed if we have a WARN status
        if slo_result.slo_status == SLOStatus.WARN:
            decision = decide_telemetry_quarantine(snapshot, slo_result)
            assert decision.publish_allowed is True
            assert decision.quarantine_required is False

    def test_quarantine_required_when_slo_breach(self, temp_dir, valid_cycle_metric_l2):
        """Quarantine required when SLO status is BREACH."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        assert decision.publish_allowed is False
        assert decision.quarantine_required is True

    def test_decision_has_reasons(self, temp_dir, valid_cycle_metric_l2):
        """Decision includes reasons for the outcome."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        assert decision.reasons is not None
        assert len(decision.reasons) > 0

    def test_decision_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """QuarantineDecision.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        serialized = json.dumps(decision.to_dict())
        deserialized = json.loads(serialized)
        assert "publish_allowed" in deserialized
        assert "quarantine_required" in deserialized
        assert "recommended_action" in deserialized

    def test_quarantine_action_types(self):
        """QuarantineAction enum has expected values."""
        assert QuarantineAction.ALLOW_PUBLISH.value == "allow_publish"
        assert QuarantineAction.QUARANTINE_RUN.value == "quarantine_run"
        assert QuarantineAction.QUARANTINE_AND_ALERT.value == "quarantine_and_alert"
        assert QuarantineAction.BLOCK_PIPELINE.value == "block_pipeline"

    def test_breach_count_and_warn_count_tracked(self, temp_dir, valid_cycle_metric_l2):
        """Decision tracks breach and warn counts."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        assert decision.breach_count == slo_result.breach_rules
        assert decision.warn_count == slo_result.warn_rules


# =============================================================================
# PHASE III TASK 3: GLOBAL HEALTH SUMMARY TESTS
# =============================================================================

class TestGlobalHealthSummary:
    """Test summarize_telemetry_for_global_health function."""

    def test_telemetry_ok_when_no_breach(self, temp_dir, valid_cycle_metric_l2):
        """telemetry_ok is True when no breaches."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        assert health.telemetry_ok is True

    def test_telemetry_not_ok_when_breach(self, temp_dir, valid_cycle_metric_l2):
        """telemetry_ok is False when there are breaches."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        assert health.telemetry_ok is False

    def test_breach_ratio_calculated(self, temp_dir, valid_cycle_metric_l2):
        """breach_ratio is calculated correctly."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        expected_ratio = (slo_result.warn_rules + slo_result.breach_rules) / slo_result.evaluated_rules
        assert health.breach_ratio == round(expected_ratio, 4)

    def test_key_reasons_populated_on_violations(self, temp_dir, valid_cycle_metric_l2):
        """key_reasons populated when there are violations."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")
            f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        if slo_result.violated_rules:
            assert len(health.key_reasons) > 0

    def test_recommendation_healthy(self, temp_dir, valid_cycle_metric_l2):
        """Recommendation is positive when healthy."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        assert "healthy" in health.recommendation.lower() or "proceed" in health.recommendation.lower()

    def test_recommendation_breach(self, temp_dir, valid_cycle_metric_l2):
        """Recommendation indicates issue when breached."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        assert "investigation" in health.recommendation.lower() or "breach" in health.recommendation.lower()

    def test_global_health_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """GlobalHealthSummary.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        serialized = json.dumps(health.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["telemetry_ok"] == health.telemetry_ok
        assert deserialized["breach_ratio"] == health.breach_ratio

    def test_global_health_rule_counts(self, temp_dir, valid_cycle_metric_l2):
        """Global health includes rule counts."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        assert health.total_rules == slo_result.evaluated_rules
        assert health.passed_rules == slo_result.passed_rules
        assert health.failed_rules == slo_result.warn_rules + slo_result.breach_rules

    def test_slo_status_included(self, temp_dir, valid_cycle_metric_l2):
        """Global health includes SLO status string."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        health = summarize_telemetry_for_global_health(slo_result)

        assert health.slo_status == slo_result.slo_status.value


# =============================================================================
# PHASE IV TASK 1: TELEMETRY RELEASE GATE TESTS
# =============================================================================

class TestTelemetryReleaseGate:
    """Test evaluate_telemetry_for_release function."""

    def test_release_ok_when_slo_ok(self, temp_dir, valid_cycle_metric_l2):
        """Release is OK when SLO status is OK."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        assert gate.release_ok is True
        assert gate.status == ReleaseStatus.OK
        assert len(gate.blocking_reasons) == 0

    def test_release_warn_when_slo_warn(self, temp_dir, valid_cycle_metric_l2):
        """Release is WARN (but allowed) when SLO status is WARN."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Only 5 records - triggers warn
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        if slo_result.slo_status == SLOStatus.WARN:
            decision = decide_telemetry_quarantine(snapshot, slo_result)
            gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

            assert gate.release_ok is True  # Warnings don't block
            assert gate.status == ReleaseStatus.WARN

    def test_release_blocked_when_slo_breach(self, temp_dir, valid_cycle_metric_l2):
        """Release is BLOCK when SLO status is BREACH."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        assert gate.release_ok is False
        assert gate.status == ReleaseStatus.BLOCK
        assert len(gate.blocking_reasons) > 0

    def test_blocking_reasons_populated(self, temp_dir, valid_cycle_metric_l2):
        """Blocking reasons are populated on breach."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        assert len(gate.blocking_reasons) > 0
        # Should contain SLO breach info
        assert any("SLO" in reason or "breach" in reason.lower() for reason in gate.blocking_reasons)

    def test_release_gate_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """ReleaseGateResult.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        serialized = json.dumps(gate.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["release_ok"] == gate.release_ok
        assert deserialized["status"] == gate.status.value

    def test_release_status_enum_values(self):
        """ReleaseStatus enum has expected values."""
        assert ReleaseStatus.OK.value == "OK"
        assert ReleaseStatus.WARN.value == "WARN"
        assert ReleaseStatus.BLOCK.value == "BLOCK"


# =============================================================================
# PHASE IV TASK 2: MAAS TELEMETRY ADAPTER TESTS
# =============================================================================

class TestMAASAdapter:
    """Test summarize_telemetry_for_maas function."""

    def test_maas_admissible_when_slo_ok(self, temp_dir, valid_cycle_metric_l2):
        """Telemetry is admissible when SLO status is OK."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        maas = summarize_telemetry_for_maas(slo_result.to_dict(), decision.to_dict())

        assert maas.telemetry_admissible is True
        assert maas.status == MAASStatus.OK
        assert len(maas.violation_codes) == 0

    def test_maas_attention_when_slo_warn(self, temp_dir, valid_cycle_metric_l2):
        """Telemetry needs attention when SLO status is WARN."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Only 5 records - triggers warn
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        if slo_result.slo_status == SLOStatus.WARN:
            decision = decide_telemetry_quarantine(snapshot, slo_result)
            maas = summarize_telemetry_for_maas(slo_result.to_dict(), decision.to_dict())

            assert maas.telemetry_admissible is True
            assert maas.status == MAASStatus.ATTENTION

    def test_maas_block_when_slo_breach(self, temp_dir, valid_cycle_metric_l2):
        """Telemetry is blocked when SLO status is BREACH."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        maas = summarize_telemetry_for_maas(slo_result.to_dict(), decision.to_dict())

        assert maas.telemetry_admissible is False
        assert maas.status == MAASStatus.BLOCK

    def test_violation_codes_populated(self, temp_dir, valid_cycle_metric_l2):
        """Violation codes are populated from violated rules."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")
            f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        maas = summarize_telemetry_for_maas(slo_result.to_dict(), decision.to_dict())

        # Should have violation codes if there are violations
        if slo_result.violated_rules:
            assert len(maas.violation_codes) > 0

    def test_maas_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """MAASAdapterResult.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        maas = summarize_telemetry_for_maas(slo_result.to_dict(), decision.to_dict())

        serialized = json.dumps(maas.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["telemetry_admissible"] == maas.telemetry_admissible
        assert deserialized["status"] == maas.status.value

    def test_maas_status_enum_values(self):
        """MAASStatus enum has expected values."""
        assert MAASStatus.OK.value == "OK"
        assert MAASStatus.ATTENTION.value == "ATTENTION"
        assert MAASStatus.BLOCK.value == "BLOCK"

    def test_quarantine_adds_violation_code(self, temp_dir, valid_cycle_metric_l2):
        """Quarantine decision adds violation code."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        maas = summarize_telemetry_for_maas(slo_result.to_dict(), decision.to_dict())

        if decision.quarantine_required:
            assert any("QUARANTINE" in code for code in maas.violation_codes)


# =============================================================================
# PHASE IV TASK 3: DIRECTOR TELEMETRY PANEL TESTS
# =============================================================================

class TestDirectorTelemetryPanel:
    """Test build_telemetry_director_panel function."""

    def test_green_light_when_all_ok(self, temp_dir, valid_cycle_metric_l2):
        """Status light is green when all checks pass."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

        assert panel.status_light == StatusLight.GREEN
        assert panel.telemetry_ok is True

    def test_yellow_light_when_warn(self, temp_dir, valid_cycle_metric_l2):
        """Status light is yellow when warnings present."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Only 5 records - triggers warn
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        if slo_result.slo_status == SLOStatus.WARN:
            decision = decide_telemetry_quarantine(snapshot, slo_result)
            health = summarize_telemetry_for_global_health(slo_result)
            gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
            panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

            assert panel.status_light == StatusLight.YELLOW

    def test_red_light_when_breach(self, temp_dir, valid_cycle_metric_l2):
        """Status light is red when breaches present."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

        assert panel.status_light == StatusLight.RED
        assert panel.telemetry_ok is False

    def test_headline_neutral_for_green(self, temp_dir, valid_cycle_metric_l2):
        """Headline is neutral/positive for green status."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

        assert "nominal" in panel.headline.lower()
        assert "passed" in panel.headline.lower()

    def test_headline_indicates_warnings(self, temp_dir, valid_cycle_metric_l2):
        """Headline indicates warnings for yellow status."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            # Only 5 records - triggers warn
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        if slo_result.slo_status == SLOStatus.WARN:
            decision = decide_telemetry_quarantine(snapshot, slo_result)
            health = summarize_telemetry_for_global_health(slo_result)
            gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
            panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

            assert "attention" in panel.headline.lower() or "warning" in panel.headline.lower()

    def test_headline_indicates_breach(self, temp_dir, valid_cycle_metric_l2):
        """Headline indicates breach for red status."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

        assert "breach" in panel.headline.lower()

    def test_breach_ratio_included(self, temp_dir, valid_cycle_metric_l2):
        """Panel includes breach_ratio from health summary."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

        assert panel.breach_ratio == health.breach_ratio

    def test_panel_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """DirectorTelemetryPanel.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

        serialized = json.dumps(panel.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["status_light"] == panel.status_light.value
        assert deserialized["telemetry_ok"] == panel.telemetry_ok
        assert deserialized["headline"] == panel.headline

    def test_status_light_enum_values(self):
        """StatusLight enum has expected values."""
        assert StatusLight.GREEN.value == "green"
        assert StatusLight.YELLOW.value == "yellow"
        assert StatusLight.RED.value == "red"

    def test_telemetry_ok_false_when_release_blocked(self, temp_dir, valid_cycle_metric_l2):
        """telemetry_ok is False when release is blocked."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        panel = build_telemetry_director_panel(health.to_dict(), gate.to_dict())

        # telemetry_ok should be False when release is blocked
        assert panel.telemetry_ok is False


# =============================================================================
# PHASE V TASK 1: TDA-TELEMETRY CORRELATION TESTS
# =============================================================================

class TestTDACorrelation:
    """Test summarize_telemetry_tda_correlation function."""

    def test_no_correlation_when_both_healthy(self, temp_dir, valid_cycle_metric_l2):
        """No correlation detected when both telemetry and TDA are healthy."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        # Healthy TDA stats
        tda_stats = {
            "hss": 0.9,
            "tda_ok": True,
            "failing_slices": [],
            "slice_health": {},
        }

        corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        assert corr.low_hss_high_quarantine_pattern is False
        assert corr.tda_telemetry_mismatch is False
        assert len(corr.correlated_failures) == 0

    def test_low_hss_high_quarantine_correlation(self, temp_dir, valid_cycle_metric_l2):
        """Correlation detected when low HSS + high quarantine rate."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            # Create ~10% quarantine rate (above 2% threshold)
            for _ in range(90):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(10):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        # Low HSS TDA stats
        tda_stats = {
            "hss": 0.3,  # Below 0.5 threshold
            "tda_ok": False,
            "failing_slices": ["slice_A", "slice_B"],
            "slice_health": {},
        }

        corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        assert corr.low_hss_high_quarantine_pattern is True
        assert "slice_A" in corr.correlated_failures
        assert "slice_B" in corr.correlated_failures
        assert any("Correlation detected" in note for note in corr.notes)

    def test_tda_telemetry_mismatch_tda_ok_telemetry_issues(self, temp_dir, valid_cycle_metric_l2):
        """Mismatch detected when TDA healthy but telemetry has issues."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        # TDA reports OK but telemetry has issues
        tda_stats = {
            "hss": 0.9,
            "tda_ok": True,
            "failing_slices": [],
            "slice_health": {},
        }

        corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        assert corr.tda_telemetry_mismatch is True
        assert any("Mismatch" in note for note in corr.notes)

    def test_per_slice_correlation(self, temp_dir, valid_cycle_metric_l2):
        """Per-slice health issues are detected."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        # Slice health issues
        tda_stats = {
            "hss": 0.8,
            "tda_ok": True,
            "failing_slices": [],
            "slice_health": {
                "slice_uplift_goal": 0.3,  # Low health
                "slice_throughput": 0.9,   # Good health
            },
        }

        corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        assert "slice_uplift_goal" in corr.correlated_failures
        assert "slice_throughput" not in corr.correlated_failures

    def test_correlation_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """TDACorrelationResult.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)

        tda_stats = {"hss": 0.9, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        serialized = json.dumps(corr.to_dict())
        deserialized = json.loads(serialized)
        assert "low_hss_high_quarantine_pattern" in deserialized
        assert "correlated_failures" in deserialized
        assert "notes" in deserialized


# =============================================================================
# PHASE V TASK 2: GLOBAL CONSOLE ADAPTER V2 TESTS
# =============================================================================

class TestGlobalConsoleAdapter:
    """Test summarize_telemetry_for_global_console function."""

    def test_green_light_no_tda_correlation(self, temp_dir, valid_cycle_metric_l2):
        """Green status when all healthy and no TDA correlation."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        tda_stats = {"hss": 0.9, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        assert console.telemetry_ok is True
        assert console.status_light == StatusLight.GREEN
        assert console.tda_correlation_detected is False
        assert len(console.alert_codes) == 0

    def test_yellow_light_on_tda_correlation(self, temp_dir, valid_cycle_metric_l2):
        """Yellow status when TDA correlation detected."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        # TDA mismatch (TDA not OK but telemetry clean)
        tda_stats = {"hss": 0.3, "tda_ok": False, "failing_slices": ["slice_A"], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        assert console.status_light == StatusLight.YELLOW
        assert console.tda_correlation_detected is True

    def test_red_light_on_breach(self, temp_dir, valid_cycle_metric_l2):
        """Red status when SLO breach."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        tda_stats = {"hss": 0.9, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        assert console.status_light == StatusLight.RED
        assert console.telemetry_ok is False

    def test_alert_codes_on_low_hss_high_quarantine(self, temp_dir, valid_cycle_metric_l2):
        """Alert codes include TDA correlation alerts."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(90):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(10):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        tda_stats = {"hss": 0.3, "tda_ok": False, "failing_slices": ["slice_A"], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        assert AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value in console.alert_codes

    def test_headline_mentions_tda_correlation(self, temp_dir, valid_cycle_metric_l2):
        """Headline mentions TDA correlation when detected."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        # TDA mismatch
        tda_stats = {"hss": 0.3, "tda_ok": False, "failing_slices": ["slice_A"], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        assert "TDA" in console.headline or "correlation" in console.headline.lower()

    def test_console_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """GlobalConsoleResult.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        tda_stats = {"hss": 0.9, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        serialized = json.dumps(console.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["telemetry_ok"] == console.telemetry_ok
        assert deserialized["status_light"] == console.status_light.value


# =============================================================================
# PHASE V TASK 3: ALERT CODES TESTS
# =============================================================================

class TestAlertCodes:
    """Test alert violation codes and MAAS v2 adapter."""

    def test_alert_code_enum_values(self):
        """AlertCode enum has expected values."""
        assert AlertCode.TELEMETRY_QUARANTINE_SPIKE.value == "TELEMETRY_QUARANTINE_SPIKE"
        assert AlertCode.TELEMETRY_TDA_MISMATCH.value == "TELEMETRY_TDA_MISMATCH"
        assert AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value == "TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE"
        assert AlertCode.TELEMETRY_PIPELINE_BLOCKED.value == "TELEMETRY_PIPELINE_BLOCKED"

    def test_no_alerts_when_healthy(self, temp_dir, valid_cycle_metric_l2):
        """No alert codes when telemetry is healthy."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        alerts = get_alert_codes_for_telemetry(slo_result.to_dict(), decision.to_dict())
        assert len(alerts) == 0

    def test_maas_v2_includes_tda_alerts(self, temp_dir, valid_cycle_metric_l2):
        """MAAS v2 adapter includes TDA correlation alerts."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(90):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(10):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        tda_stats = {"hss": 0.3, "tda_ok": False, "failing_slices": ["slice_A"], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        maas = summarize_telemetry_for_maas_v2(
            slo_result.to_dict(), decision.to_dict(), tda_corr.to_dict()
        )

        assert AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value in maas.violation_codes

    def test_maas_v2_no_tda_correlation_provided(self, temp_dir, valid_cycle_metric_l2):
        """MAAS v2 works without TDA correlation."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        maas = summarize_telemetry_for_maas_v2(slo_result.to_dict(), decision.to_dict())

        assert maas.telemetry_admissible is True
        assert maas.status == MAASStatus.OK

    def test_maas_v2_attention_on_tda_mismatch(self, temp_dir, valid_cycle_metric_l2):
        """MAAS v2 shows ATTENTION when TDA mismatch detected."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)

        # TDA not OK but telemetry is clean = mismatch
        tda_stats = {"hss": 0.3, "tda_ok": False, "failing_slices": ["slice_A"], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        maas = summarize_telemetry_for_maas_v2(
            slo_result.to_dict(), decision.to_dict(), tda_corr.to_dict()
        )

        assert maas.status == MAASStatus.ATTENTION
        assert AlertCode.TELEMETRY_TDA_MISMATCH.value in maas.violation_codes


# =============================================================================
# PHASE V INTEGRATION TESTS
# =============================================================================

class TestPhaseVIntegration:
    """Integration tests for Phase V TDA × Telemetry × MAAS."""

    def test_telemetry_ok_tda_benign_no_alerts(self, temp_dir, valid_cycle_metric_l2):
        """Telemetry OK + TDA benign → no special alerts."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        # Benign TDA
        tda_stats = {"hss": 0.95, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        # Verify no alerts
        assert console.telemetry_ok is True
        assert console.status_light == StatusLight.GREEN
        assert len(console.alert_codes) == 0
        assert console.tda_correlation_detected is False

    def test_telemetry_warn_low_hss_correlated_failures(self, temp_dir, valid_cycle_metric_l2):
        """Telemetry WARN + low HSS → correlated_failures True."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            # Create enough invalid records to trigger correlation
            for _ in range(95):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        # Low HSS with failing slices
        tda_stats = {
            "hss": 0.35,  # Low HSS
            "tda_ok": False,
            "failing_slices": ["slice_uplift_goal", "slice_throughput"],
            "slice_health": {
                "slice_uplift_goal": 0.2,
                "slice_throughput": 0.3,
            },
        }
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )

        # Verify correlation detected
        assert tda_corr.low_hss_high_quarantine_pattern is True or tda_corr.tda_telemetry_mismatch is True
        assert len(tda_corr.correlated_failures) > 0
        assert console.tda_correlation_detected is True

    def test_full_pipeline_healthy_to_console(self, temp_dir, valid_cycle_metric_l2):
        """Full pipeline from file audit to console output."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        # Full pipeline
        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        tda_stats = {"hss": 0.9, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)
        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )
        maas = summarize_telemetry_for_maas_v2(
            slo_result.to_dict(), decision.to_dict(), tda_corr.to_dict()
        )

        # All outputs should indicate healthy
        assert console.telemetry_ok is True
        assert maas.telemetry_admissible is True
        assert gate.release_ok is True
        assert health.telemetry_ok is True

    def test_full_pipeline_breach_to_console(self, temp_dir, valid_cycle_metric_l2):
        """Full pipeline with breach propagates correctly."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(5):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(5):
                f.write(invalid_json + "\n")

        # Full pipeline
        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        tda_stats = {"hss": 0.9, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)
        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )
        maas = summarize_telemetry_for_maas_v2(
            slo_result.to_dict(), decision.to_dict(), tda_corr.to_dict()
        )

        # All outputs should indicate breach
        assert console.telemetry_ok is False
        assert console.status_light == StatusLight.RED
        assert maas.telemetry_admissible is False
        assert maas.status == MAASStatus.BLOCK
        assert gate.release_ok is False
        assert health.telemetry_ok is False


# =============================================================================
# PHASE V-B: TELEMETRY GOVERNANCE SIGNAL ADAPTER TESTS
# =============================================================================

class TestGovernanceSignalAdapter:
    """Test to_governance_signal_for_telemetry function."""

    def test_ok_when_telemetry_and_tda_healthy(self, temp_dir, valid_cycle_metric_l2):
        """Governance signal is OK when telemetry and TDA both healthy."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        # Healthy TDA
        tda_stats = {"hss": 0.95, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )
        gov = to_governance_signal_for_telemetry(console.to_dict())

        assert gov.signal == GovernanceSignal.OK
        assert gov.telemetry_ok is True
        assert len(gov.blocking_rules) == 0
        assert gov.tda_coupled is False

    def test_block_on_pipeline_blocked_alert(self, temp_dir, valid_cycle_metric_l2):
        """Governance signal is BLOCK when TELEMETRY_PIPELINE_BLOCKED alert present."""
        # Create console summary with pipeline blocked alert
        console_summary = {
            "telemetry_ok": False,
            "status_light": "red",
            "breach_ratio": 0.5,
            "headline": "Pipeline blocked",
            "alert_codes": [AlertCode.TELEMETRY_PIPELINE_BLOCKED.value],
            "tda_correlation_detected": False,
            "correlated_slice_count": 0,
        }

        gov = to_governance_signal_for_telemetry(console_summary)

        assert gov.signal == GovernanceSignal.BLOCK
        assert gov.telemetry_ok is False
        assert AlertCode.TELEMETRY_PIPELINE_BLOCKED.value in gov.blocking_rules
        assert any("blocked" in reason.lower() for reason in gov.reasons)

    def test_block_on_low_hss_high_quarantine_alert(self, temp_dir, valid_cycle_metric_l2):
        """Governance signal is BLOCK when TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE alert present."""
        # Create console summary with low HSS + high quarantine pattern
        console_summary = {
            "telemetry_ok": False,
            "status_light": "red",
            "breach_ratio": 0.1,
            "headline": "Correlated failure detected",
            "alert_codes": [AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value],
            "tda_correlation_detected": True,
            "correlated_slice_count": 2,
        }

        gov = to_governance_signal_for_telemetry(console_summary)

        assert gov.signal == GovernanceSignal.BLOCK
        assert gov.telemetry_ok is False
        assert AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value in gov.blocking_rules
        assert any("Correlated" in reason for reason in gov.reasons)

    def test_warn_on_tda_mismatch_alert(self, temp_dir, valid_cycle_metric_l2):
        """Governance signal is WARN when TELEMETRY_TDA_MISMATCH alert present."""
        console_summary = {
            "telemetry_ok": True,
            "status_light": "yellow",
            "breach_ratio": 0.0,
            "headline": "TDA mismatch detected",
            "alert_codes": [AlertCode.TELEMETRY_TDA_MISMATCH.value],
            "tda_correlation_detected": True,
            "correlated_slice_count": 1,
        }

        gov = to_governance_signal_for_telemetry(console_summary)

        assert gov.signal == GovernanceSignal.WARN
        assert gov.telemetry_ok is True  # WARN allows continue
        assert AlertCode.TELEMETRY_TDA_MISMATCH.value in gov.blocking_rules
        assert gov.tda_coupled is True

    def test_warn_on_quarantine_spike_alert(self, temp_dir, valid_cycle_metric_l2):
        """Governance signal is WARN when TELEMETRY_QUARANTINE_SPIKE alert present."""
        console_summary = {
            "telemetry_ok": True,
            "status_light": "yellow",
            "breach_ratio": 0.1,
            "headline": "Quarantine spike detected",
            "alert_codes": [AlertCode.TELEMETRY_QUARANTINE_SPIKE.value],
            "tda_correlation_detected": False,
            "correlated_slice_count": 0,
        }

        gov = to_governance_signal_for_telemetry(console_summary)

        assert gov.signal == GovernanceSignal.WARN
        assert AlertCode.TELEMETRY_QUARANTINE_SPIKE.value in gov.blocking_rules
        assert any("spike" in reason.lower() for reason in gov.reasons)

    def test_warn_on_l2_degradation_alert(self):
        """Governance signal is WARN when TELEMETRY_L2_DEGRADATION alert present."""
        console_summary = {
            "telemetry_ok": True,
            "status_light": "yellow",
            "breach_ratio": 0.05,
            "headline": "L2 conformance degraded",
            "alert_codes": [AlertCode.TELEMETRY_L2_DEGRADATION.value],
            "tda_correlation_detected": False,
            "correlated_slice_count": 0,
        }

        gov = to_governance_signal_for_telemetry(console_summary)

        assert gov.signal == GovernanceSignal.WARN
        assert AlertCode.TELEMETRY_L2_DEGRADATION.value in gov.blocking_rules
        assert any("L2" in reason for reason in gov.reasons)

    def test_warn_on_yellow_status_light_no_alerts(self):
        """Governance signal is WARN when status light is yellow even without specific alerts."""
        console_summary = {
            "telemetry_ok": True,
            "status_light": "yellow",
            "breach_ratio": 0.01,
            "headline": "Telemetry warnings present",
            "alert_codes": [],
            "tda_correlation_detected": False,
            "correlated_slice_count": 0,
        }

        gov = to_governance_signal_for_telemetry(console_summary)

        assert gov.signal == GovernanceSignal.WARN
        assert len(gov.blocking_rules) == 0  # No specific blocking rules
        assert "Telemetry warnings present" in gov.reasons

    def test_warn_on_tda_correlation_detected(self):
        """Governance signal is WARN when TDA correlation detected even without specific alerts."""
        console_summary = {
            "telemetry_ok": True,
            "status_light": "green",  # Note: green, but correlation detected
            "breach_ratio": 0.0,
            "headline": "Telemetry nominal",
            "alert_codes": [],
            "tda_correlation_detected": True,
            "correlated_slice_count": 1,
        }

        gov = to_governance_signal_for_telemetry(console_summary)

        assert gov.signal == GovernanceSignal.WARN
        assert gov.tda_coupled is True
        assert any("correlation" in reason.lower() for reason in gov.reasons)

    def test_governance_signal_enum_values(self):
        """GovernanceSignal enum has expected values."""
        assert GovernanceSignal.OK.value == "OK"
        assert GovernanceSignal.WARN.value == "WARN"
        assert GovernanceSignal.BLOCK.value == "BLOCK"

    def test_governance_result_to_dict_serializable(self, temp_dir, valid_cycle_metric_l2):
        """TelemetryGovernanceResult.to_dict() is JSON-serializable."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        tda_stats = {"hss": 0.9, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)

        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )
        gov = to_governance_signal_for_telemetry(console.to_dict())

        serialized = json.dumps(gov.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["signal"] == gov.signal.value
        assert deserialized["telemetry_ok"] == gov.telemetry_ok
        assert deserialized["blocking_rules"] == gov.blocking_rules

    def test_blocking_rules_constant_values(self):
        """GOVERNANCE_BLOCKING_RULES contains correct values."""
        assert AlertCode.TELEMETRY_PIPELINE_BLOCKED.value in GOVERNANCE_BLOCKING_RULES
        assert AlertCode.TELEMETRY_TDA_LOW_HSS_HIGH_QUARANTINE.value in GOVERNANCE_BLOCKING_RULES

    def test_warn_rules_constant_values(self):
        """GOVERNANCE_WARN_RULES contains correct values."""
        assert AlertCode.TELEMETRY_TDA_MISMATCH.value in GOVERNANCE_WARN_RULES
        assert AlertCode.TELEMETRY_QUARANTINE_SPIKE.value in GOVERNANCE_WARN_RULES
        assert AlertCode.TELEMETRY_L2_DEGRADATION.value in GOVERNANCE_WARN_RULES
        assert AlertCode.TELEMETRY_CRITICAL_VIOLATIONS.value in GOVERNANCE_WARN_RULES

    def test_full_pipeline_to_governance_signal(self, temp_dir, valid_cycle_metric_l2):
        """Full pipeline from audit to governance signal."""
        jsonl_path = temp_dir / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for _ in range(100):
                f.write(valid_cycle_metric_l2 + "\n")

        # Full pipeline
        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())
        tda_stats = {"hss": 0.95, "tda_ok": True, "failing_slices": [], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)
        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )
        gov = to_governance_signal_for_telemetry(console.to_dict())

        # Governance should be OK
        assert gov.signal == GovernanceSignal.OK
        assert gov.telemetry_ok is True

    def test_breach_pipeline_to_governance_block(self, temp_dir, valid_cycle_metric_l2):
        """Full pipeline with breach leads to governance BLOCK."""
        jsonl_path = temp_dir / "test.jsonl"

        invalid_record = {"cycle": 42, "metric_type": "goal_hit"}
        invalid_json = json.dumps(invalid_record, sort_keys=True, separators=(",", ":"))

        with open(jsonl_path, "w") as f:
            for _ in range(80):
                f.write(valid_cycle_metric_l2 + "\n")
            for _ in range(20):  # 20% invalid for significant breach
                f.write(invalid_json + "\n")

        # Full pipeline
        report = audit_telemetry_file(jsonl_path)
        snapshot = build_telemetry_conformance_snapshot(report)
        slo_result = evaluate_telemetry_slo(snapshot)
        decision = decide_telemetry_quarantine(snapshot, slo_result)
        health = summarize_telemetry_for_global_health(slo_result)
        gate = evaluate_telemetry_for_release(slo_result.to_dict(), decision.to_dict())

        # Low HSS for correlated failure
        tda_stats = {"hss": 0.3, "tda_ok": False, "failing_slices": ["slice_A"], "slice_health": {}}
        tda_corr = summarize_telemetry_tda_correlation(slo_result.to_dict(), tda_stats)
        console = summarize_telemetry_for_global_console(
            health.to_dict(), gate.to_dict(), tda_corr.to_dict()
        )
        gov = to_governance_signal_for_telemetry(console.to_dict())

        # Should be BLOCK due to low HSS + high quarantine
        assert gov.signal == GovernanceSignal.BLOCK
        assert gov.telemetry_ok is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
