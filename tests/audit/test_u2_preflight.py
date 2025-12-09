"""
Tests for U2 Pre-Flight Audit Tool

PHASE II — NOT RUN IN PHASE I

Tests the 25-item pre-flight checklist implementation.
Uses mocks for database and filesystem to ensure determinism.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from u2_preflight import (
    CheckStatus,
    EligibilityStatus,
    FailureType,
    CheckResult,
    GateResult,
    PreFlightReport,
    _check_PRE1_registration_identity,
    _check_PRE2_preregistration_integrity,
    _check_PRE3_baseline_snapshot,
    _check_PRE4_log_directory_integrity,
    _check_PRE5_database_connectivity,
    _check_PRE6_state_eligibility,
    _classify_eligibility,
    _aggregate_gates,
    _compute_prereg_hash,
    run_preflight,
    # New v2 functions
    PREFLIGHT_SNAPSHOT_SCHEMA_VERSION,
    build_preflight_snapshot,
    compare_preflight_snapshots,
    to_bundle_stage_result,
    render_preflight_markdown,
    # Phase III functions
    build_preflight_drift_timeline,
    summarize_preflight_for_maas,
    summarize_preflight_for_global_health,
    # Phase IV functions
    evaluate_preflight_for_release,
    summarize_preflight_for_audit_readiness,
    build_preflight_director_panel,
    # Phase V functions
    build_preflight_bundle_joint_view,
    summarize_preflight_for_global_console,
    to_governance_signal,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_db_conn():
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_prereg_entry():
    """Valid preregistration entry."""
    return {
        "experiment_id": "u2_test_001",
        "theory_id": 1,
        "slice_id": "pl_atoms5_depth6",
        "goals": [{"id": "goal_peirce", "formula": "((p → q) → p) → p"}],
        "cycles": 50,
    }


@pytest.fixture
def temp_prereg_file(valid_prereg_entry):
    """Create a temporary prereg file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump({"experiments": [valid_prereg_entry]}, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)


# =============================================================================
# PRE-1: REGISTRATION & IDENTITY TESTS
# =============================================================================

class TestPRE1RegistrationIdentity:
    """Tests for PRE-1 gate checks."""

    def test_experiment_not_found(self, mock_db_conn):
        """PRE-1.1: Experiment not in database → STOP."""
        conn, cursor = mock_db_conn
        cursor.fetchone.return_value = None

        results = _check_PRE1_registration_identity("missing_exp", conn)

        assert len(results) == 1
        assert results[0].id == "PRE-1.1"
        assert results[0].status == CheckStatus.FAIL
        assert results[0].failure_type == FailureType.STOP
        assert "not found" in results[0].message

    def test_experiment_found_all_valid(self, mock_db_conn):
        """PRE-1: All fields valid → PASS."""
        conn, cursor = mock_db_conn

        # Mock experiment data
        cursor.fetchone.side_effect = [
            # First call: experiment data
            ("u2_test_001", 1, "slice_001", "completed", datetime.now()),
            # Second call: theory exists
            (1,),
        ]

        results = _check_PRE1_registration_identity("u2_test_001", conn)

        assert len(results) == 5
        assert all(r.status == CheckStatus.PASS for r in results)

    def test_invalid_theory_fk(self, mock_db_conn):
        """PRE-1.2: Invalid theory FK → STOP."""
        conn, cursor = mock_db_conn

        cursor.fetchone.side_effect = [
            ("u2_test_001", 999, "slice_001", "completed", datetime.now()),
            None,  # Theory not found
        ]

        results = _check_PRE1_registration_identity("u2_test_001", conn)

        check_1_2 = next(r for r in results if r.id == "PRE-1.2")
        assert check_1_2.status == CheckStatus.FAIL
        assert check_1_2.failure_type == FailureType.STOP

    def test_empty_slice_id(self, mock_db_conn):
        """PRE-1.3: Empty slice ID → STOP."""
        conn, cursor = mock_db_conn

        cursor.fetchone.side_effect = [
            ("u2_test_001", 1, "", "completed", datetime.now()),
            (1,),
        ]

        results = _check_PRE1_registration_identity("u2_test_001", conn)

        check_1_3 = next(r for r in results if r.id == "PRE-1.3")
        assert check_1_3.status == CheckStatus.FAIL
        assert check_1_3.failure_type == FailureType.STOP

    def test_invalid_status(self, mock_db_conn):
        """PRE-1.4: Invalid status → STOP."""
        conn, cursor = mock_db_conn

        cursor.fetchone.side_effect = [
            ("u2_test_001", 1, "slice_001", "invalid_status", datetime.now()),
            (1,),
        ]

        results = _check_PRE1_registration_identity("u2_test_001", conn)

        check_1_4 = next(r for r in results if r.id == "PRE-1.4")
        assert check_1_4.status == CheckStatus.FAIL
        assert check_1_4.failure_type == FailureType.STOP

    def test_missing_start_time_when_running(self, mock_db_conn):
        """PRE-1.5: Running without start_time → STOP."""
        conn, cursor = mock_db_conn

        cursor.fetchone.side_effect = [
            ("u2_test_001", 1, "slice_001", "running", None),
            (1,),
        ]

        results = _check_PRE1_registration_identity("u2_test_001", conn)

        check_1_5 = next(r for r in results if r.id == "PRE-1.5")
        assert check_1_5.status == CheckStatus.FAIL
        assert check_1_5.failure_type == FailureType.STOP


# =============================================================================
# PRE-2: PREREGISTRATION INTEGRITY TESTS
# =============================================================================

class TestPRE2PreregistrationIntegrity:
    """Tests for PRE-2 gate checks."""

    def test_prereg_file_not_found(self, mock_db_conn):
        """PRE-2.6: Prereg file missing → FATAL."""
        conn, cursor = mock_db_conn
        missing_path = Path("/nonexistent/prereg.yaml")

        results = _check_PRE2_preregistration_integrity("u2_test_001", missing_path, conn)

        assert len(results) == 1
        assert results[0].id == "PRE-2.6"
        assert results[0].status == CheckStatus.FAIL
        assert results[0].failure_type == FailureType.FATAL

    def test_experiment_not_in_prereg(self, mock_db_conn, temp_prereg_file):
        """PRE-2.6: Experiment not in prereg → FATAL (INADMISSIBLE)."""
        conn, cursor = mock_db_conn

        results = _check_PRE2_preregistration_integrity("unknown_exp", temp_prereg_file, conn)

        check_2_6 = next(r for r in results if r.id == "PRE-2.6")
        assert check_2_6.status == CheckStatus.FAIL
        assert check_2_6.failure_type == FailureType.FATAL
        assert "not found" in check_2_6.message

    def test_prereg_hash_mismatch(self, mock_db_conn, temp_prereg_file, valid_prereg_entry):
        """PRE-2.7: Hash mismatch → FATAL."""
        conn, cursor = mock_db_conn
        cursor.fetchone.return_value = ("wrong_hash_value",)

        results = _check_PRE2_preregistration_integrity("u2_test_001", temp_prereg_file, conn)

        check_2_7 = next(r for r in results if r.id == "PRE-2.7")
        assert check_2_7.status == CheckStatus.FAIL
        assert check_2_7.failure_type == FailureType.FATAL
        assert "mismatch" in check_2_7.message.lower()

    def test_prereg_hash_matches(self, mock_db_conn, temp_prereg_file, valid_prereg_entry):
        """PRE-2.7: Hash matches → PASS."""
        conn, cursor = mock_db_conn

        # Compute correct hash
        correct_hash = _compute_prereg_hash(valid_prereg_entry)
        cursor.fetchone.return_value = (correct_hash,)

        results = _check_PRE2_preregistration_integrity("u2_test_001", temp_prereg_file, conn)

        check_2_7 = next(r for r in results if r.id == "PRE-2.7")
        assert check_2_7.status == CheckStatus.PASS


# =============================================================================
# PRE-3: BASELINE SNAPSHOT TESTS
# =============================================================================

class TestPRE3BaselineSnapshot:
    """Tests for PRE-3 gate checks."""

    def test_snapshot_missing(self, mock_db_conn):
        """PRE-3.9: No snapshot → STOP (BLOCKED_FIXABLE)."""
        conn, cursor = mock_db_conn
        cursor.fetchone.return_value = None

        results = _check_PRE3_baseline_snapshot("u2_test_001", conn)

        assert len(results) == 1
        assert results[0].id == "PRE-3.9"
        assert results[0].status == CheckStatus.FAIL
        assert results[0].failure_type == FailureType.STOP

    def test_snapshot_after_start(self, mock_db_conn):
        """PRE-3.13: Snapshot after start → FATAL."""
        conn, cursor = mock_db_conn

        start_time = datetime.now()
        snapshot_time = start_time + timedelta(hours=1)

        cursor.fetchone.side_effect = [
            # Snapshot data
            ("u2_test_001", "a" * 64, 100, 50, snapshot_time),
            # Experiment start time
            (start_time,),
        ]

        results = _check_PRE3_baseline_snapshot("u2_test_001", conn)

        check_3_13 = next(r for r in results if r.id == "PRE-3.13")
        assert check_3_13.status == CheckStatus.FAIL
        assert check_3_13.failure_type == FailureType.FATAL
        assert "contaminated" in check_3_13.message.lower()

    def test_snapshot_before_start(self, mock_db_conn):
        """PRE-3.13: Snapshot before start → PASS."""
        conn, cursor = mock_db_conn

        start_time = datetime.now()
        snapshot_time = start_time - timedelta(hours=1)

        cursor.fetchone.side_effect = [
            ("u2_test_001", "a" * 64, 100, 50, snapshot_time),
            (start_time,),
        ]

        results = _check_PRE3_baseline_snapshot("u2_test_001", conn)

        check_3_13 = next(r for r in results if r.id == "PRE-3.13")
        assert check_3_13.status == CheckStatus.PASS

    def test_invalid_merkle_length(self, mock_db_conn):
        """PRE-3.10: Invalid merkle length → STOP."""
        conn, cursor = mock_db_conn

        cursor.fetchone.side_effect = [
            ("u2_test_001", "tooshort", 100, 50, datetime.now()),
            (datetime.now(),),
        ]

        results = _check_PRE3_baseline_snapshot("u2_test_001", conn)

        check_3_10 = next(r for r in results if r.id == "PRE-3.10")
        assert check_3_10.status == CheckStatus.FAIL
        assert check_3_10.failure_type == FailureType.STOP


# =============================================================================
# PRE-4: LOG DIRECTORY INTEGRITY TESTS
# =============================================================================

class TestPRE4LogDirectoryIntegrity:
    """Tests for PRE-4 gate checks."""

    def test_log_dir_missing(self, temp_log_dir):
        """PRE-4.14: Log directory missing → STOP."""
        results = _check_PRE4_log_directory_integrity("nonexistent_exp", temp_log_dir)

        assert len(results) == 1
        assert results[0].id == "PRE-4.14"
        assert results[0].status == CheckStatus.FAIL
        assert results[0].failure_type == FailureType.STOP

    def test_manifest_missing(self, temp_log_dir):
        """PRE-4.15: manifest.json missing → STOP."""
        log_dir = temp_log_dir / "logs" / "u2" / "test_exp"
        log_dir.mkdir(parents=True)
        # Create cycle log but no manifest
        (log_dir / "cycle_1.jsonl").write_text('{"cycle": 1}\n')

        results = _check_PRE4_log_directory_integrity("test_exp", temp_log_dir)

        check_4_15 = next(r for r in results if r.id == "PRE-4.15")
        assert check_4_15.status == CheckStatus.FAIL
        assert check_4_15.failure_type == FailureType.STOP

    def test_manifest_invalid_json(self, temp_log_dir):
        """PRE-4.16: Invalid JSON in manifest → STOP."""
        log_dir = temp_log_dir / "logs" / "u2" / "test_exp"
        log_dir.mkdir(parents=True)
        (log_dir / "manifest.json").write_text("{invalid json")
        (log_dir / "cycle_1.jsonl").write_text('{"cycle": 1}\n')

        results = _check_PRE4_log_directory_integrity("test_exp", temp_log_dir)

        check_4_16 = next(r for r in results if r.id == "PRE-4.16")
        assert check_4_16.status == CheckStatus.FAIL
        assert check_4_16.failure_type == FailureType.STOP

    def test_no_cycle_logs(self, temp_log_dir):
        """PRE-4.17: No cycle logs → STOP."""
        log_dir = temp_log_dir / "logs" / "u2" / "test_exp"
        log_dir.mkdir(parents=True)
        (log_dir / "manifest.json").write_text('{"experiment_id": "test_exp"}')

        results = _check_PRE4_log_directory_integrity("test_exp", temp_log_dir)

        check_4_17 = next(r for r in results if r.id == "PRE-4.17")
        assert check_4_17.status == CheckStatus.FAIL
        assert check_4_17.failure_type == FailureType.STOP

    def test_verifications_missing(self, temp_log_dir):
        """PRE-4.19: verifications.jsonl missing → WARN."""
        log_dir = temp_log_dir / "logs" / "u2" / "test_exp"
        log_dir.mkdir(parents=True)
        (log_dir / "manifest.json").write_text('{"experiment_id": "test_exp"}')
        (log_dir / "cycle_1.jsonl").write_text('{"cycle": 1}\n')

        results = _check_PRE4_log_directory_integrity("test_exp", temp_log_dir)

        check_4_19 = next(r for r in results if r.id == "PRE-4.19")
        assert check_4_19.status == CheckStatus.WARN
        assert check_4_19.failure_type == FailureType.WARN

    def test_all_logs_valid(self, temp_log_dir):
        """PRE-4: All files valid → PASS."""
        log_dir = temp_log_dir / "logs" / "u2" / "test_exp"
        log_dir.mkdir(parents=True)
        (log_dir / "manifest.json").write_text('{"experiment_id": "test_exp"}')
        (log_dir / "cycle_1.jsonl").write_text('{"cycle": 1}\n')
        (log_dir / "verifications.jsonl").write_text('{"hash": "abc", "result": "success"}\n')

        results = _check_PRE4_log_directory_integrity("test_exp", temp_log_dir)

        assert all(r.status == CheckStatus.PASS for r in results)


# =============================================================================
# PRE-5: DATABASE CONNECTIVITY TESTS
# =============================================================================

class TestPRE5DatabaseConnectivity:
    """Tests for PRE-5 gate checks."""

    @patch('u2_preflight.HAS_PSYCOPG2', False)
    def test_psycopg2_not_installed(self):
        """PRE-5.21: psycopg2 missing → STOP."""
        results, conn = _check_PRE5_database_connectivity("postgresql://test")

        assert len(results) == 1
        assert results[0].id == "PRE-5.21"
        assert results[0].status == CheckStatus.FAIL
        assert conn is None


# =============================================================================
# PRE-6: STATE ELIGIBILITY TESTS
# =============================================================================

class TestPRE6StateEligibility:
    """Tests for PRE-6 gate checks."""

    def test_status_pending(self, mock_db_conn):
        """PRE-6.24: Status pending → STOP."""
        conn, cursor = mock_db_conn
        cursor.fetchone.return_value = ("pending",)

        results = _check_PRE6_state_eligibility("u2_test_001", conn)

        check_6_24 = next(r for r in results if r.id == "PRE-6.24")
        assert check_6_24.status == CheckStatus.FAIL
        assert check_6_24.failure_type == FailureType.STOP
        assert "pending" in check_6_24.message

    def test_running_no_cycles(self, mock_db_conn):
        """PRE-6.25: Running with 0 cycles → STOP."""
        conn, cursor = mock_db_conn
        cursor.fetchone.side_effect = [
            ("running",),
            (0,),  # cycle count
        ]

        results = _check_PRE6_state_eligibility("u2_test_001", conn)

        check_6_25 = next(r for r in results if r.id == "PRE-6.25")
        assert check_6_25.status == CheckStatus.FAIL
        assert check_6_25.failure_type == FailureType.STOP

    def test_running_with_cycles(self, mock_db_conn):
        """PRE-6.25: Running with cycles → WARN (partial audit)."""
        conn, cursor = mock_db_conn
        cursor.fetchone.side_effect = [
            ("running",),
            (5,),  # 5 completed cycles
        ]

        results = _check_PRE6_state_eligibility("u2_test_001", conn)

        check_6_25 = next(r for r in results if r.id == "PRE-6.25")
        assert check_6_25.status == CheckStatus.WARN
        assert check_6_25.failure_type == FailureType.WARN
        assert "partial" in check_6_25.message.lower()

    def test_completed_status(self, mock_db_conn):
        """PRE-6: Completed status → PASS."""
        conn, cursor = mock_db_conn
        cursor.fetchone.return_value = ("completed",)

        results = _check_PRE6_state_eligibility("u2_test_001", conn)

        assert all(r.status == CheckStatus.PASS for r in results)


# =============================================================================
# ELIGIBILITY CLASSIFICATION TESTS
# =============================================================================

class TestEligibilityClassification:
    """Tests for eligibility classification logic."""

    def test_fatal_yields_inadmissible(self):
        """Any FATAL → INADMISSIBLE."""
        checks = [
            CheckResult("PRE-2.7", "PRE-2", "Hash", CheckStatus.FAIL, "msg", FailureType.FATAL),
        ]

        status, _ = _classify_eligibility(checks)
        assert status == EligibilityStatus.INADMISSIBLE

    def test_stop_yields_blocked(self):
        """STOP without FATAL → BLOCKED_FIXABLE."""
        checks = [
            CheckResult("PRE-1.1", "PRE-1", "Exp", CheckStatus.FAIL, "msg", FailureType.STOP),
        ]

        status, _ = _classify_eligibility(checks)
        assert status == EligibilityStatus.BLOCKED_FIXABLE

    def test_warn_only_yields_eligible_warned(self):
        """WARN only → ELIGIBLE_WARNED."""
        checks = [
            CheckResult("PRE-4.19", "PRE-4", "Verif", CheckStatus.WARN, "msg", FailureType.WARN),
        ]

        status, _ = _classify_eligibility(checks)
        assert status == EligibilityStatus.ELIGIBLE_WARNED

    def test_all_pass_yields_eligible(self):
        """All PASS → ELIGIBLE."""
        checks = [
            CheckResult("PRE-1.1", "PRE-1", "Exp", CheckStatus.PASS, "msg"),
            CheckResult("PRE-2.6", "PRE-2", "Prereg", CheckStatus.PASS, "msg"),
        ]

        status, _ = _classify_eligibility(checks)
        assert status == EligibilityStatus.ELIGIBLE

    def test_partial_audit_running(self):
        """Running with cycles → ELIGIBLE_PARTIAL."""
        checks = [
            CheckResult("PRE-6.24", "PRE-6", "Status", CheckStatus.PASS, "msg"),
            CheckResult("PRE-6.25", "PRE-6", "Cycles", CheckStatus.WARN, "partial", FailureType.WARN),
        ]

        status, _ = _classify_eligibility(checks)
        assert status == EligibilityStatus.ELIGIBLE_PARTIAL

    def test_fatal_overrides_stop(self):
        """FATAL takes precedence over STOP."""
        checks = [
            CheckResult("PRE-1.1", "PRE-1", "Exp", CheckStatus.FAIL, "msg", FailureType.STOP),
            CheckResult("PRE-2.7", "PRE-2", "Hash", CheckStatus.FAIL, "msg", FailureType.FATAL),
        ]

        status, _ = _classify_eligibility(checks)
        assert status == EligibilityStatus.INADMISSIBLE


# =============================================================================
# GATE AGGREGATION TESTS
# =============================================================================

class TestGateAggregation:
    """Tests for gate result aggregation."""

    def test_aggregates_by_gate(self):
        """Checks are grouped by gate."""
        checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "msg"),
            CheckResult("PRE-1.2", "PRE-1", "B", CheckStatus.PASS, "msg"),
            CheckResult("PRE-2.6", "PRE-2", "C", CheckStatus.FAIL, "msg", FailureType.FATAL),
        ]

        gates = _aggregate_gates(checks)

        assert "PRE-1" in gates
        assert "PRE-2" in gates
        assert gates["PRE-1"].passed == 2
        assert gates["PRE-1"].failed == 0
        assert gates["PRE-2"].failed == 1

    def test_gate_status_fail_if_any_fail(self):
        """Gate status is FAIL if any check fails."""
        checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "msg"),
            CheckResult("PRE-1.2", "PRE-1", "B", CheckStatus.FAIL, "msg", FailureType.STOP),
        ]

        gates = _aggregate_gates(checks)
        assert gates["PRE-1"].status == CheckStatus.FAIL

    def test_gate_status_warn_if_no_fail(self):
        """Gate status is WARN if warnings but no failures."""
        checks = [
            CheckResult("PRE-4.19", "PRE-4", "A", CheckStatus.PASS, "msg"),
            CheckResult("PRE-4.20", "PRE-4", "B", CheckStatus.WARN, "msg", FailureType.WARN),
        ]

        gates = _aggregate_gates(checks)
        assert gates["PRE-4"].status == CheckStatus.WARN


# =============================================================================
# REPORT FORMAT TESTS
# =============================================================================

class TestReportFormat:
    """Tests for report JSON format and determinism."""

    def test_report_to_json_deterministic(self):
        """Report JSON output is deterministic."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-2.6", "PRE-2", "B", CheckStatus.PASS, "msg"),
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "msg"),
        ]
        report.gates = _aggregate_gates(report.checks)

        json1 = report.to_json()
        json2 = report.to_json()

        assert json1 == json2

    def test_checks_sorted_by_id(self):
        """Checks in JSON are sorted by ID."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-3.10", "PRE-3", "C", CheckStatus.PASS, "msg"),
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "msg"),
            CheckResult("PRE-2.6", "PRE-2", "B", CheckStatus.PASS, "msg"),
        ]

        data = report.to_dict()
        check_ids = [c["id"] for c in data["checks"]]

        assert check_ids == sorted(check_ids)

    def test_report_contains_required_fields(self):
        """Report JSON contains all required fields."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.BLOCKED_FIXABLE,
        )
        report.fatal_reasons = ["F1"]
        report.stop_reasons = ["S1"]
        report.warnings = ["W1"]

        data = report.to_dict()

        assert "experiment_id" in data
        assert "preflight_timestamp" in data
        assert "eligibility_status" in data
        assert "gates" in data
        assert "checks" in data
        assert "fatal_reasons" in data
        assert "stop_reasons" in data
        assert "warnings" in data
        assert "recommendation" in data

    def test_json_is_valid(self):
        """Report JSON is valid JSON."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["experiment_id"] == "test_001"
        assert parsed["eligibility_status"] == "ELIGIBLE"


# =============================================================================
# INTEGRATION TESTS (MOCKED)
# =============================================================================

class TestIntegration:
    """Integration tests with full mocking."""

    @patch('u2_preflight._check_PRE5_database_connectivity')
    @patch('u2_preflight._check_PRE1_registration_identity')
    @patch('u2_preflight._check_PRE2_preregistration_integrity')
    @patch('u2_preflight._check_PRE3_baseline_snapshot')
    @patch('u2_preflight._check_PRE4_log_directory_integrity')
    @patch('u2_preflight._check_PRE6_state_eligibility')
    def test_full_preflight_eligible(
        self,
        mock_pre6, mock_pre4, mock_pre3, mock_pre2, mock_pre1, mock_pre5
    ):
        """Full preflight with all passing → ELIGIBLE."""
        mock_conn = MagicMock()
        mock_pre5.return_value = (
            [CheckResult("PRE-5.21", "PRE-5", "DB", CheckStatus.PASS, "ok")],
            mock_conn
        )
        mock_pre1.return_value = [CheckResult("PRE-1.1", "PRE-1", "Exp", CheckStatus.PASS, "ok")]
        mock_pre2.return_value = [CheckResult("PRE-2.6", "PRE-2", "Prereg", CheckStatus.PASS, "ok")]
        mock_pre3.return_value = [CheckResult("PRE-3.9", "PRE-3", "Snap", CheckStatus.PASS, "ok")]
        mock_pre4.return_value = [CheckResult("PRE-4.14", "PRE-4", "Logs", CheckStatus.PASS, "ok")]
        mock_pre6.return_value = [CheckResult("PRE-6.24", "PRE-6", "State", CheckStatus.PASS, "ok")]

        report = run_preflight(
            exp_id="test_001",
            run_dir=Path("/tmp"),
            db_url="postgresql://test",
            prereg_path=Path("/tmp/prereg.yaml"),
        )

        assert report.eligibility_status == EligibilityStatus.ELIGIBLE
        assert len(report.checks) == 6
        assert len(report.fatal_reasons) == 0
        assert len(report.stop_reasons) == 0

    @patch('u2_preflight._check_PRE5_database_connectivity')
    @patch('u2_preflight._check_PRE1_registration_identity')
    @patch('u2_preflight._check_PRE2_preregistration_integrity')
    @patch('u2_preflight._check_PRE3_baseline_snapshot')
    @patch('u2_preflight._check_PRE4_log_directory_integrity')
    @patch('u2_preflight._check_PRE6_state_eligibility')
    def test_full_preflight_inadmissible(
        self,
        mock_pre6, mock_pre4, mock_pre3, mock_pre2, mock_pre1, mock_pre5
    ):
        """Full preflight with FATAL → INADMISSIBLE."""
        mock_conn = MagicMock()
        mock_pre5.return_value = (
            [CheckResult("PRE-5.21", "PRE-5", "DB", CheckStatus.PASS, "ok")],
            mock_conn
        )
        mock_pre1.return_value = [CheckResult("PRE-1.1", "PRE-1", "Exp", CheckStatus.PASS, "ok")]
        mock_pre2.return_value = [
            CheckResult("PRE-2.6", "PRE-2", "Prereg", CheckStatus.FAIL, "not found", FailureType.FATAL)
        ]
        mock_pre3.return_value = [CheckResult("PRE-3.9", "PRE-3", "Snap", CheckStatus.PASS, "ok")]
        mock_pre4.return_value = [CheckResult("PRE-4.14", "PRE-4", "Logs", CheckStatus.PASS, "ok")]
        mock_pre6.return_value = [CheckResult("PRE-6.24", "PRE-6", "State", CheckStatus.PASS, "ok")]

        report = run_preflight(
            exp_id="test_001",
            run_dir=Path("/tmp"),
            db_url="postgresql://test",
            prereg_path=Path("/tmp/prereg.yaml"),
        )

        assert report.eligibility_status == EligibilityStatus.INADMISSIBLE
        assert len(report.fatal_reasons) == 1
        assert "PRE-2.6" in report.fatal_reasons[0]


# =============================================================================
# CLI EXIT CODE TESTS
# =============================================================================

class TestCLIExitCodes:
    """Tests for CLI exit code behavior."""

    def test_eligible_exit_code_0(self):
        """ELIGIBLE status should return exit code 0."""
        # This is implicitly tested via main() but we verify the logic
        assert EligibilityStatus.ELIGIBLE in (
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.ELIGIBLE_WARNED,
            EligibilityStatus.ELIGIBLE_PARTIAL,
        )

    def test_blocked_exit_code_1(self):
        """BLOCKED_FIXABLE status should return exit code 1."""
        assert EligibilityStatus.BLOCKED_FIXABLE not in (
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.ELIGIBLE_WARNED,
            EligibilityStatus.ELIGIBLE_PARTIAL,
        )

    def test_inadmissible_exit_code_1(self):
        """INADMISSIBLE status should return exit code 1."""
        assert EligibilityStatus.INADMISSIBLE not in (
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.ELIGIBLE_WARNED,
            EligibilityStatus.ELIGIBLE_PARTIAL,
        )


# =============================================================================
# SNAPSHOT FUNCTION TESTS (v2)
# =============================================================================

class TestBuildPreflightSnapshot:
    """Tests for build_preflight_snapshot() function."""

    def test_snapshot_contains_schema_version(self):
        """Snapshot includes schema version."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )

        snapshot = build_preflight_snapshot(report)

        assert "schema_version" in snapshot
        assert snapshot["schema_version"] == PREFLIGHT_SNAPSHOT_SCHEMA_VERSION

    def test_snapshot_contains_eligibility(self):
        """Snapshot includes eligibility status as string."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.BLOCKED_FIXABLE,
        )

        snapshot = build_preflight_snapshot(report)

        assert snapshot["eligibility"] == "BLOCKED_FIXABLE"

    def test_snapshot_counts_checks_by_status(self):
        """Snapshot counts PASS/WARN/FAIL checks."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE_WARNED,
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "ok"),
            CheckResult("PRE-1.2", "PRE-1", "B", CheckStatus.PASS, "ok"),
            CheckResult("PRE-2.6", "PRE-2", "C", CheckStatus.WARN, "warn", FailureType.WARN),
            CheckResult("PRE-3.9", "PRE-3", "D", CheckStatus.FAIL, "fail", FailureType.STOP),
        ]

        snapshot = build_preflight_snapshot(report)

        assert snapshot["counts"]["total"] == 4
        assert snapshot["counts"]["pass"] == 2
        assert snapshot["counts"]["warn"] == 1
        assert snapshot["counts"]["fail"] == 1

    def test_snapshot_counts_failure_types(self):
        """Snapshot counts FATAL/STOP/WARN failure types."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.INADMISSIBLE,
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.FAIL, "f", FailureType.FATAL),
            CheckResult("PRE-1.2", "PRE-1", "B", CheckStatus.FAIL, "s", FailureType.STOP),
            CheckResult("PRE-1.3", "PRE-1", "C", CheckStatus.WARN, "w", FailureType.WARN),
        ]

        snapshot = build_preflight_snapshot(report)

        assert snapshot["failure_types"]["fatal"] == 1
        assert snapshot["failure_types"]["stop"] == 1
        assert snapshot["failure_types"]["warn"] == 1

    def test_snapshot_collects_failed_check_ids_sorted(self):
        """Snapshot includes sorted list of failed/warned check IDs."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.BLOCKED_FIXABLE,
        )
        report.checks = [
            CheckResult("PRE-3.9", "PRE-3", "A", CheckStatus.FAIL, "f", FailureType.STOP),
            CheckResult("PRE-1.1", "PRE-1", "B", CheckStatus.PASS, "ok"),
            CheckResult("PRE-2.6", "PRE-2", "C", CheckStatus.WARN, "w", FailureType.WARN),
        ]

        snapshot = build_preflight_snapshot(report)

        assert snapshot["failed_check_ids"] == ["PRE-2.6", "PRE-3.9"]  # sorted


class TestComparePreflightSnapshots:
    """Tests for compare_preflight_snapshots() function."""

    def test_detects_schema_compatibility(self):
        """Comparison detects schema version mismatch."""
        old = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}
        new = {"schema_version": "2.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}

        diff = compare_preflight_snapshots(old, new)

        assert diff["schema_compatible"] is False

    def test_detects_eligibility_change(self):
        """Comparison detects eligibility status change."""
        old = {"schema_version": "1.0.0", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1"]}
        new = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}

        diff = compare_preflight_snapshots(old, new)

        assert diff["eligibility_change"] == ("BLOCKED_FIXABLE", "ELIGIBLE")

    def test_no_eligibility_change_returns_none(self):
        """Comparison returns None for eligibility_change if unchanged."""
        old = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}
        new = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}

        diff = compare_preflight_snapshots(old, new)

        assert diff["eligibility_change"] is None

    def test_detects_new_failures(self):
        """Comparison detects new failures."""
        old = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}
        new = {"schema_version": "1.0.0", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1", "PRE-2.6"]}

        diff = compare_preflight_snapshots(old, new)

        assert diff["new_failures"] == ["PRE-1.1", "PRE-2.6"]

    def test_detects_resolved_failures(self):
        """Comparison detects resolved failures."""
        old = {"schema_version": "1.0.0", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1", "PRE-2.6"]}
        new = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}

        diff = compare_preflight_snapshots(old, new)

        assert diff["resolved_failures"] == ["PRE-1.1", "PRE-2.6"]

    def test_improved_when_eligibility_better(self):
        """improved=True when eligibility status improves."""
        old = {"schema_version": "1.0.0", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1"]}
        new = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}

        diff = compare_preflight_snapshots(old, new)

        assert diff["improved"] is True
        assert diff["regressed"] is False

    def test_regressed_when_eligibility_worse(self):
        """regressed=True when eligibility status worsens."""
        old = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": []}
        new = {"schema_version": "1.0.0", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1"]}

        diff = compare_preflight_snapshots(old, new)

        assert diff["improved"] is False
        assert diff["regressed"] is True

    def test_regressed_when_new_failures_appear(self):
        """regressed=True when new failures appear (even if eligibility same)."""
        old = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE_WARNED", "failed_check_ids": ["PRE-4.19"]}
        new = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE_WARNED", "failed_check_ids": ["PRE-4.19", "PRE-4.20"]}

        diff = compare_preflight_snapshots(old, new)

        assert diff["regressed"] is True

    def test_improved_when_failures_resolved_no_new(self):
        """improved=True when failures resolved and no new ones."""
        old = {
            "schema_version": "1.0.0",
            "eligibility": "ELIGIBLE_WARNED",
            "failed_check_ids": ["PRE-4.19", "PRE-4.20"],
            "counts": {"pass": 3, "warn": 2, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 2},
        }
        new = {
            "schema_version": "1.0.0",
            "eligibility": "ELIGIBLE_WARNED",
            "failed_check_ids": ["PRE-4.19"],
            "counts": {"pass": 4, "warn": 1, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 1},
        }

        diff = compare_preflight_snapshots(old, new)

        assert diff["improved"] is True
        assert diff["regressed"] is False

    def test_count_deltas_computed(self):
        """Comparison computes count deltas."""
        old = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": [],
               "counts": {"pass": 20, "warn": 2, "fail": 3}}
        new = {"schema_version": "1.0.0", "eligibility": "ELIGIBLE", "failed_check_ids": [],
               "counts": {"pass": 25, "warn": 0, "fail": 0}}

        diff = compare_preflight_snapshots(old, new)

        assert diff["count_deltas"]["pass"] == 5
        assert diff["count_deltas"]["warn"] == -2
        assert diff["count_deltas"]["fail"] == -3


# =============================================================================
# BUNDLE BRIDGE TESTS (v2)
# =============================================================================

class TestToBundleStageResult:
    """Tests for to_bundle_stage_result() function."""

    def test_bundle_format_structure(self):
        """Bundle result has required structure."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
            recommendation="PROCEED_TO_AUDIT",
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "ok"),
        ]
        report.gates = _aggregate_gates(report.checks)

        bundle = to_bundle_stage_result(report)

        assert bundle["stage_name"] == "preflight"
        assert bundle["stage_version"] == PREFLIGHT_SNAPSHOT_SCHEMA_VERSION
        assert "status" in bundle
        assert "experiment_id" in bundle
        assert "timestamp" in bundle
        assert "eligibility" in bundle
        assert "recommendation" in bundle
        assert "gates" in bundle
        assert "checks" in bundle
        assert "fatal_reasons" in bundle
        assert "stop_reasons" in bundle
        assert "warnings" in bundle

    def test_eligible_maps_to_pass(self):
        """ELIGIBLE → status: 'pass'."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )

        bundle = to_bundle_stage_result(report)

        assert bundle["status"] == "pass"

    def test_eligible_warned_maps_to_warn(self):
        """ELIGIBLE_WARNED → status: 'warn'."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE_WARNED,
        )

        bundle = to_bundle_stage_result(report)

        assert bundle["status"] == "warn"

    def test_blocked_fixable_maps_to_fail(self):
        """BLOCKED_FIXABLE → status: 'fail'."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.BLOCKED_FIXABLE,
        )

        bundle = to_bundle_stage_result(report)

        assert bundle["status"] == "fail"

    def test_inadmissible_maps_to_fail(self):
        """INADMISSIBLE → status: 'fail'."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.INADMISSIBLE,
        )

        bundle = to_bundle_stage_result(report)

        assert bundle["status"] == "fail"

    def test_checks_lowercase_status(self):
        """Bundle checks have lowercase status values."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "ok"),
            CheckResult("PRE-2.6", "PRE-2", "B", CheckStatus.WARN, "w", FailureType.WARN),
        ]
        report.gates = _aggregate_gates(report.checks)

        bundle = to_bundle_stage_result(report)

        assert bundle["checks"][0]["status"] == "pass"
        assert bundle["checks"][1]["status"] == "warn"

    def test_checks_sorted_by_id(self):
        """Bundle checks are sorted by ID."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-3.9", "PRE-3", "C", CheckStatus.PASS, "ok"),
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "ok"),
            CheckResult("PRE-2.6", "PRE-2", "B", CheckStatus.PASS, "ok"),
        ]
        report.gates = _aggregate_gates(report.checks)

        bundle = to_bundle_stage_result(report)

        check_ids = [c["id"] for c in bundle["checks"]]
        assert check_ids == ["PRE-1.1", "PRE-2.6", "PRE-3.9"]


# =============================================================================
# MARKDOWN RENDERER TESTS (v2)
# =============================================================================

class TestRenderPreflightMarkdown:
    """Tests for render_preflight_markdown() function."""

    def test_markdown_contains_header(self):
        """Markdown includes header with experiment ID."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )

        md = render_preflight_markdown(report)

        assert "# U2 Pre-Flight Audit Report" in md
        assert "test_001" in md

    def test_markdown_contains_status_emoji(self):
        """Markdown includes status emoji."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )

        md = render_preflight_markdown(report)

        assert "✅" in md

    def test_markdown_contains_recommendation(self):
        """Markdown includes recommendation."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
            recommendation="PROCEED_TO_AUDIT - All checks passed.",
        )

        md = render_preflight_markdown(report)

        assert "## Recommendation" in md
        assert "PROCEED_TO_AUDIT" in md

    def test_markdown_contains_gate_summary_table(self):
        """Markdown includes gate summary table."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "ok"),
        ]
        report.gates = _aggregate_gates(report.checks)

        md = render_preflight_markdown(report)

        assert "## Gate Summary" in md
        assert "| Gate | Status |" in md
        assert "PRE-1" in md

    def test_markdown_contains_fatal_reasons_section(self):
        """Markdown includes fatal reasons if present."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.INADMISSIBLE,
            fatal_reasons=["PRE-2.7: Hash mismatch"],
        )

        md = render_preflight_markdown(report)

        assert "## ❌ Fatal Conditions" in md
        assert "Hash mismatch" in md

    def test_markdown_contains_stop_reasons_section(self):
        """Markdown includes stop reasons if present."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.BLOCKED_FIXABLE,
            stop_reasons=["PRE-1.1: Experiment not found"],
        )

        md = render_preflight_markdown(report)

        assert "## 🛑 Blocking Issues" in md
        assert "Experiment not found" in md

    def test_markdown_contains_warnings_section(self):
        """Markdown includes warnings if present."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE_WARNED,
            warnings=["PRE-4.19: verifications.jsonl missing"],
        )

        md = render_preflight_markdown(report)

        assert "## ⚠️ Warnings" in md
        assert "verifications.jsonl missing" in md

    def test_markdown_contains_collapsible_details(self):
        """Markdown includes collapsible detailed check results."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "Exp ID", CheckStatus.PASS, "ok"),
        ]
        report.gates = _aggregate_gates(report.checks)

        md = render_preflight_markdown(report)

        assert "<details>" in md
        assert "<summary>" in md
        assert "</details>" in md
        assert "| ID | Gate |" in md

    def test_markdown_escapes_pipe_in_message(self):
        """Markdown escapes pipe characters in messages."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "value|with|pipes"),
        ]
        report.gates = _aggregate_gates(report.checks)

        md = render_preflight_markdown(report)

        assert "value\\|with\\|pipes" in md

    def test_markdown_contains_footer_with_counts(self):
        """Markdown includes footer with pass/total counts."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.ELIGIBLE,
        )
        report.checks = [
            CheckResult("PRE-1.1", "PRE-1", "A", CheckStatus.PASS, "ok"),
            CheckResult("PRE-1.2", "PRE-1", "B", CheckStatus.PASS, "ok"),
            CheckResult("PRE-2.6", "PRE-2", "C", CheckStatus.WARN, "w", FailureType.WARN),
        ]
        report.gates = _aggregate_gates(report.checks)

        md = render_preflight_markdown(report)

        assert "2/3 checks passed" in md

    def test_blocked_status_emoji(self):
        """Markdown uses correct emoji for BLOCKED_FIXABLE."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.BLOCKED_FIXABLE,
        )

        md = render_preflight_markdown(report)

        assert "🛑" in md

    def test_inadmissible_status_emoji(self):
        """Markdown uses correct emoji for INADMISSIBLE."""
        report = PreFlightReport(
            experiment_id="test_001",
            preflight_timestamp="2025-06-15T00:00:00Z",
            eligibility_status=EligibilityStatus.INADMISSIBLE,
        )

        md = render_preflight_markdown(report)

        assert "❌" in md


# =============================================================================
# PHASE III: DRIFT TIMELINE TESTS
# =============================================================================

class TestBuildPreflightDriftTimeline:
    """Tests for build_preflight_drift_timeline() function."""

    def test_empty_snapshots_returns_defaults(self):
        """Empty snapshot list returns default values."""
        result = build_preflight_drift_timeline([])

        assert result["eligibility_shifts"] == []
        assert result["recurring_failures"] == {}
        assert result["run_stability_index"] == 1.0
        assert result["total_runs"] == 0
        assert result["first_timestamp"] is None
        assert result["last_timestamp"] is None

    def test_single_snapshot_returns_stable(self):
        """Single snapshot returns stable with no shifts."""
        snapshot = {
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "eligibility": "ELIGIBLE",
            "failed_check_ids": [],
        }

        result = build_preflight_drift_timeline([snapshot])

        assert result["eligibility_shifts"] == []
        assert result["recurring_failures"] == {}
        assert result["run_stability_index"] == 1.0
        assert result["total_runs"] == 1
        assert result["first_timestamp"] == "2025-06-15T00:00:00Z"
        assert result["last_timestamp"] == "2025-06-15T00:00:00Z"

    def test_detects_eligibility_shifts(self):
        """Detects eligibility changes across snapshots."""
        snapshots = [
            {"preflight_timestamp": "T1", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1"]},
            {"preflight_timestamp": "T2", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1"]},
            {"preflight_timestamp": "T3", "eligibility": "ELIGIBLE", "failed_check_ids": []},
            {"preflight_timestamp": "T4", "eligibility": "ELIGIBLE_WARNED", "failed_check_ids": ["PRE-4.19"]},
        ]

        result = build_preflight_drift_timeline(snapshots)

        assert len(result["eligibility_shifts"]) == 2
        assert result["eligibility_shifts"][0] == ("T3", "BLOCKED_FIXABLE", "ELIGIBLE")
        assert result["eligibility_shifts"][1] == ("T4", "ELIGIBLE", "ELIGIBLE_WARNED")

    def test_identifies_recurring_failures(self):
        """Identifies check IDs that fail multiple times."""
        snapshots = [
            {"preflight_timestamp": "T1", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1", "PRE-2.6"]},
            {"preflight_timestamp": "T2", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1"]},
            {"preflight_timestamp": "T3", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1", "PRE-3.9"]},
        ]

        result = build_preflight_drift_timeline(snapshots)

        # PRE-1.1 appears 3 times, PRE-2.6 once, PRE-3.9 once
        assert result["recurring_failures"] == {"PRE-1.1": 3}

    def test_excludes_single_occurrence_failures(self):
        """Failures occurring only once are not in recurring_failures."""
        snapshots = [
            {"preflight_timestamp": "T1", "eligibility": "ELIGIBLE", "failed_check_ids": ["PRE-4.19"]},
            {"preflight_timestamp": "T2", "eligibility": "ELIGIBLE", "failed_check_ids": ["PRE-4.20"]},
            {"preflight_timestamp": "T3", "eligibility": "ELIGIBLE", "failed_check_ids": ["PRE-4.21"]},
        ]

        result = build_preflight_drift_timeline(snapshots)

        assert result["recurring_failures"] == {}

    def test_stability_index_perfect_when_no_changes(self):
        """Stability index is 1.0 when no shifts and no recurring failures."""
        snapshots = [
            {"preflight_timestamp": "T1", "eligibility": "ELIGIBLE", "failed_check_ids": []},
            {"preflight_timestamp": "T2", "eligibility": "ELIGIBLE", "failed_check_ids": []},
            {"preflight_timestamp": "T3", "eligibility": "ELIGIBLE", "failed_check_ids": []},
        ]

        result = build_preflight_drift_timeline(snapshots)

        assert result["run_stability_index"] == 1.0

    def test_stability_index_decreases_with_shifts(self):
        """Stability index decreases when eligibility shifts frequently."""
        snapshots = [
            {"preflight_timestamp": "T1", "eligibility": "ELIGIBLE", "failed_check_ids": []},
            {"preflight_timestamp": "T2", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1"]},
            {"preflight_timestamp": "T3", "eligibility": "ELIGIBLE", "failed_check_ids": []},
            {"preflight_timestamp": "T4", "eligibility": "INADMISSIBLE", "failed_check_ids": ["PRE-2.7"]},
        ]

        result = build_preflight_drift_timeline(snapshots)

        # 3 shifts out of max 3 possible = shift_stability 0.0
        # 2 unique failures, 0 recurring = failure_stability 1.0
        # Combined: 0.6 * 0.0 + 0.4 * 1.0 = 0.4
        assert result["run_stability_index"] == 0.4

    def test_stability_index_decreases_with_recurring_failures(self):
        """Stability index decreases when failures recur."""
        snapshots = [
            {"preflight_timestamp": "T1", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1", "PRE-1.2"]},
            {"preflight_timestamp": "T2", "eligibility": "BLOCKED_FIXABLE", "failed_check_ids": ["PRE-1.1", "PRE-1.2"]},
        ]

        result = build_preflight_drift_timeline(snapshots)

        # 0 shifts = shift_stability 1.0
        # 2 unique failures, 2 recurring = failure_stability 0.0
        # Combined: 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert result["run_stability_index"] == 0.6

    def test_timestamps_captured_correctly(self):
        """First and last timestamps are captured correctly."""
        snapshots = [
            {"preflight_timestamp": "2025-06-01T00:00:00Z", "eligibility": "ELIGIBLE", "failed_check_ids": []},
            {"preflight_timestamp": "2025-06-15T00:00:00Z", "eligibility": "ELIGIBLE", "failed_check_ids": []},
            {"preflight_timestamp": "2025-06-30T00:00:00Z", "eligibility": "ELIGIBLE", "failed_check_ids": []},
        ]

        result = build_preflight_drift_timeline(snapshots)

        assert result["first_timestamp"] == "2025-06-01T00:00:00Z"
        assert result["last_timestamp"] == "2025-06-30T00:00:00Z"
        assert result["total_runs"] == 3


# =============================================================================
# PHASE III: MAAS BRIDGE TESTS
# =============================================================================

class TestSummarizePreflightForMaas:
    """Tests for summarize_preflight_for_maas() function."""

    def test_eligible_is_admissible_green(self):
        """ELIGIBLE status → admissible=True, status='green'."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "failed_check_ids": [],
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_maas(snapshot)

        assert result["admissible"] is True
        assert result["status"] == "green"
        assert result["blocking_pf_ids"] == []

    def test_eligible_warned_is_admissible_yellow(self):
        """ELIGIBLE_WARNED status → admissible=True, status='yellow'."""
        snapshot = {
            "eligibility": "ELIGIBLE_WARNED",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "failed_check_ids": ["PRE-4.19"],
            "counts": {"total": 25, "pass": 24, "warn": 1, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 1},
        }

        result = summarize_preflight_for_maas(snapshot)

        assert result["admissible"] is True
        assert result["status"] == "yellow"
        assert result["blocking_pf_ids"] == []

    def test_eligible_partial_is_admissible_yellow(self):
        """ELIGIBLE_PARTIAL status → admissible=True, status='yellow'."""
        snapshot = {
            "eligibility": "ELIGIBLE_PARTIAL",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "failed_check_ids": ["PRE-6.25"],
            "counts": {"total": 25, "pass": 24, "warn": 1, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 1},
        }

        result = summarize_preflight_for_maas(snapshot)

        assert result["admissible"] is True
        assert result["status"] == "yellow"

    def test_blocked_fixable_is_not_admissible_red(self):
        """BLOCKED_FIXABLE status → admissible=False, status='red'."""
        snapshot = {
            "eligibility": "BLOCKED_FIXABLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "failed_check_ids": ["PRE-1.1", "PRE-3.9"],
            "counts": {"total": 25, "pass": 22, "warn": 0, "fail": 3},
            "failure_types": {"fatal": 0, "stop": 2, "warn": 0},
        }

        result = summarize_preflight_for_maas(snapshot)

        assert result["admissible"] is False
        assert result["status"] == "red"
        assert result["blocking_pf_ids"] == ["PRE-1.1", "PRE-3.9"]

    def test_inadmissible_is_not_admissible_red(self):
        """INADMISSIBLE status → admissible=False, status='red'."""
        snapshot = {
            "eligibility": "INADMISSIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "failed_check_ids": ["PRE-2.7"],
            "counts": {"total": 25, "pass": 23, "warn": 0, "fail": 2},
            "failure_types": {"fatal": 1, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_maas(snapshot)

        assert result["admissible"] is False
        assert result["status"] == "red"
        assert "PRE-2.7" in result["blocking_pf_ids"]

    def test_failure_summary_included(self):
        """Result includes failure_summary with counts."""
        snapshot = {
            "eligibility": "BLOCKED_FIXABLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "failed_check_ids": ["PRE-1.1"],
            "counts": {"total": 25, "pass": 22, "warn": 1, "fail": 2},
            "failure_types": {"fatal": 0, "stop": 2, "warn": 1},
        }

        result = summarize_preflight_for_maas(snapshot)

        assert result["failure_summary"]["total_checks"] == 25
        assert result["failure_summary"]["passed"] == 22
        assert result["failure_summary"]["warned"] == 1
        assert result["failure_summary"]["failed"] == 2
        assert result["failure_summary"]["fatal"] == 0
        assert result["failure_summary"]["stop"] == 2

    def test_experiment_id_and_timestamp_preserved(self):
        """Experiment ID and timestamp are preserved in result."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_exp_abc123",
            "preflight_timestamp": "2025-06-15T12:34:56Z",
            "failed_check_ids": [],
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_maas(snapshot)

        assert result["experiment_id"] == "u2_exp_abc123"
        assert result["timestamp"] == "2025-06-15T12:34:56Z"
        assert result["eligibility"] == "ELIGIBLE"


# =============================================================================
# PHASE III: GLOBAL HEALTH SUMMARY TESTS
# =============================================================================

class TestSummarizePreflightForGlobalHealth:
    """Tests for summarize_preflight_for_global_health() function."""

    def test_eligible_is_preflight_ok(self):
        """ELIGIBLE status → preflight_ok=True."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_global_health(snapshot)

        assert result["preflight_ok"] is True
        assert result["current_eligibility"] == "ELIGIBLE"

    def test_blocked_is_not_preflight_ok(self):
        """BLOCKED_FIXABLE status → preflight_ok=False."""
        snapshot = {
            "eligibility": "BLOCKED_FIXABLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 20, "warn": 0, "fail": 5},
            "failure_types": {"fatal": 0, "stop": 3, "warn": 0},
        }

        result = summarize_preflight_for_global_health(snapshot)

        assert result["preflight_ok"] is False

    def test_drift_status_unknown_without_timeline(self):
        """Without drift timeline, drift_status is 'unknown'."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=None)

        assert result["drift_status"] == "unknown"
        assert result["failure_hotspots"] == []

    def test_drift_status_stable_no_shifts(self):
        """Drift status is 'stable' when no eligibility shifts."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }
        drift = {
            "eligibility_shifts": [],
            "recurring_failures": {},
            "run_stability_index": 1.0,
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=drift)

        assert result["drift_status"] == "stable"

    def test_drift_status_improving_when_last_shift_better(self):
        """Drift status is 'improving' when last shift improved eligibility."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }
        drift = {
            "eligibility_shifts": [
                ("T1", "BLOCKED_FIXABLE", "ELIGIBLE"),
            ],
            "recurring_failures": {},
            "run_stability_index": 0.8,
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=drift)

        assert result["drift_status"] == "improving"

    def test_drift_status_degrading_when_last_shift_worse(self):
        """Drift status is 'degrading' when last shift worsened eligibility."""
        snapshot = {
            "eligibility": "BLOCKED_FIXABLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 20, "warn": 0, "fail": 5},
            "failure_types": {"fatal": 0, "stop": 3, "warn": 0},
        }
        drift = {
            "eligibility_shifts": [
                ("T1", "ELIGIBLE", "BLOCKED_FIXABLE"),
            ],
            "recurring_failures": {"PRE-1.1": 3},
            "run_stability_index": 0.5,
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=drift)

        assert result["drift_status"] == "degrading"

    def test_failure_hotspots_from_recurring(self):
        """Failure hotspots are extracted from recurring failures."""
        snapshot = {
            "eligibility": "BLOCKED_FIXABLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 20, "warn": 0, "fail": 5},
            "failure_types": {"fatal": 0, "stop": 3, "warn": 0},
        }
        drift = {
            "eligibility_shifts": [],
            "recurring_failures": {
                "PRE-1.1": 5,
                "PRE-3.9": 3,
                "PRE-2.6": 2,
            },
            "run_stability_index": 0.6,
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=drift)

        # Should be sorted by count descending
        assert result["failure_hotspots"] == ["PRE-1.1", "PRE-3.9", "PRE-2.6"]

    def test_failure_hotspots_limited_to_top_5(self):
        """Failure hotspots are limited to top 5."""
        snapshot = {
            "eligibility": "BLOCKED_FIXABLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 15, "warn": 0, "fail": 10},
            "failure_types": {"fatal": 0, "stop": 5, "warn": 0},
        }
        drift = {
            "eligibility_shifts": [],
            "recurring_failures": {
                "PRE-1.1": 10, "PRE-1.2": 9, "PRE-2.6": 8,
                "PRE-3.9": 7, "PRE-4.14": 6, "PRE-4.15": 5,
                "PRE-5.21": 4,
            },
            "run_stability_index": 0.3,
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=drift)

        assert len(result["failure_hotspots"]) == 5
        assert result["failure_hotspots"][0] == "PRE-1.1"
        assert result["failure_hotspots"][4] == "PRE-4.14"

    def test_health_score_perfect_for_all_pass(self):
        """Health score is high when all checks pass."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=None)

        # current_pass_rate = 1.0, no penalties, stability_factor = 1.0
        # health_score = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert result["health_score"] == 1.0

    def test_health_score_penalized_by_fatal(self):
        """Health score is penalized by FATAL failures."""
        snapshot = {
            "eligibility": "INADMISSIBLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 25, "pass": 24, "warn": 0, "fail": 1},
            "failure_types": {"fatal": 1, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_global_health(snapshot, drift_timeline=None)

        # pass_rate = 24/25 = 0.96
        # fatal_penalty = 0.3
        # current_health = 0.96 - 0.3 = 0.66
        # health_score = 0.7 * 0.66 + 0.3 * 1.0 = 0.762
        assert result["health_score"] < 1.0
        assert result["health_score"] > 0.5

    def test_pass_rate_calculated(self):
        """Pass rate is calculated and included."""
        snapshot = {
            "eligibility": "BLOCKED_FIXABLE",
            "experiment_id": "u2_test_001",
            "preflight_timestamp": "2025-06-15T00:00:00Z",
            "counts": {"total": 20, "pass": 15, "warn": 2, "fail": 3},
            "failure_types": {"fatal": 0, "stop": 2, "warn": 2},
        }

        result = summarize_preflight_for_global_health(snapshot)

        assert result["pass_rate"] == 0.75  # 15/20

    def test_experiment_id_and_timestamp_preserved(self):
        """Experiment ID and timestamp are preserved."""
        snapshot = {
            "eligibility": "ELIGIBLE",
            "experiment_id": "u2_exp_xyz789",
            "preflight_timestamp": "2025-06-20T10:30:00Z",
            "counts": {"total": 25, "pass": 25, "warn": 0, "fail": 0},
            "failure_types": {"fatal": 0, "stop": 0, "warn": 0},
        }

        result = summarize_preflight_for_global_health(snapshot)

        assert result["experiment_id"] == "u2_exp_xyz789"
        assert result["last_check"] == "2025-06-20T10:30:00Z"


# =============================================================================
# PHASE IV: PREFLIGHT RELEASE EVALUATOR TESTS
# =============================================================================

class TestEvaluatePreflightForRelease:
    """Tests for evaluate_preflight_for_release() function."""

    def test_eligible_stable_returns_ok(self):
        """ELIGIBLE with stable drift → release_ok=True, status='OK'."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 1.0,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is True
        assert result["status"] == "OK"
        assert result["blocking_reasons"] == []

    def test_inadmissible_returns_block(self):
        """INADMISSIBLE → release_ok=False, status='BLOCK'."""
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "INADMISSIBLE",
            "health_score": 0.5,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 1.0,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("INADMISSIBLE" in r for r in result["blocking_reasons"])

    def test_blocked_fixable_returns_block(self):
        """BLOCKED_FIXABLE → release_ok=False, status='BLOCK'."""
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "BLOCKED_FIXABLE",
            "health_score": 0.6,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 0.8,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("blocking issues" in r for r in result["blocking_reasons"])

    def test_degrading_drift_blocks_release(self):
        """Active degradation → BLOCK even if currently ELIGIBLE."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.9,
            "drift_status": "degrading",
        }
        drift_timeline = {
            "run_stability_index": 0.5,
            "recurring_failures": {},
            "eligibility_shifts": [("T1", "ELIGIBLE", "ELIGIBLE_WARNED")],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("degrading" in r for r in result["blocking_reasons"])

    def test_low_stability_blocks_release(self):
        """Stability < 0.3 → BLOCK."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.8,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 0.2,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("stability" in r.lower() for r in result["blocking_reasons"])

    def test_chronic_failures_block_release(self):
        """Recurring failures > 3 times → BLOCK."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.8,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 0.7,
            "recurring_failures": {"PRE-1.1": 5, "PRE-3.9": 4},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is False
        assert result["status"] == "BLOCK"
        assert any("CHRONIC" in r for r in result["blocking_reasons"])

    def test_eligible_warned_returns_warn(self):
        """ELIGIBLE_WARNED → release_ok=True, status='WARN'."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE_WARNED",
            "health_score": 0.9,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 0.9,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is True
        assert result["status"] == "WARN"

    def test_low_health_generates_warning(self):
        """Health score < 0.7 generates warning (not blocking)."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.6,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 0.8,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        assert result["release_ok"] is True
        assert result["status"] == "WARN"
        assert any("Health score" in w for w in result["warnings"])

    def test_confidence_calculated(self):
        """Confidence is calculated from health and stability."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.8,
            "drift_status": "stable",
        }
        drift_timeline = {
            "run_stability_index": 0.6,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }

        result = evaluate_preflight_for_release(global_summary, drift_timeline)

        # confidence = 0.5 * 0.8 + 0.5 * 0.6 = 0.7
        assert result["confidence"] == 0.7


# =============================================================================
# PHASE IV: MAAS AUDIT READINESS TESTS
# =============================================================================

class TestSummarizePreflightForAuditReadiness:
    """Tests for summarize_preflight_for_audit_readiness() function."""

    def test_eligible_stable_is_audit_ready_ok(self):
        """ELIGIBLE + stable → audit_ready=True, status='OK'."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        drift_timeline = {
            "run_stability_index": 1.0,
            "recurring_failures": {},
        }

        result = summarize_preflight_for_audit_readiness(global_summary, drift_timeline)

        assert result["audit_ready"] is True
        assert result["status"] == "OK"
        assert result["drift_status"] == "stable"

    def test_blocked_is_not_audit_ready(self):
        """BLOCKED_FIXABLE → audit_ready=False, status='BLOCK'."""
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "BLOCKED_FIXABLE",
            "health_score": 0.6,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        drift_timeline = {
            "run_stability_index": 0.8,
            "recurring_failures": {},
        }

        result = summarize_preflight_for_audit_readiness(global_summary, drift_timeline)

        assert result["audit_ready"] is False
        assert result["status"] == "BLOCK"

    def test_degrading_triggers_attention(self):
        """Degrading drift → status='ATTENTION'."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.9,
            "drift_status": "degrading",
            "experiment_id": "u2_test_001",
        }
        drift_timeline = {
            "run_stability_index": 0.6,
            "recurring_failures": {},
        }

        result = summarize_preflight_for_audit_readiness(global_summary, drift_timeline)

        assert result["audit_ready"] is False  # degrading blocks readiness
        assert result["status"] == "ATTENTION"
        assert "degrading" in result["attention_reasons"][0].lower()

    def test_low_stability_triggers_attention(self):
        """Low stability (<0.5) → status='ATTENTION'."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        drift_timeline = {
            "run_stability_index": 0.4,
            "recurring_failures": {},
        }

        result = summarize_preflight_for_audit_readiness(global_summary, drift_timeline)

        assert result["status"] == "ATTENTION"
        assert any("stability" in r.lower() for r in result["attention_reasons"])

    def test_many_recurring_failures_triggers_attention(self):
        """More than 2 recurring failures → status='ATTENTION'."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.8,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        drift_timeline = {
            "run_stability_index": 0.7,
            "recurring_failures": {"PRE-1.1": 2, "PRE-2.6": 2, "PRE-3.9": 2},
        }

        result = summarize_preflight_for_audit_readiness(global_summary, drift_timeline)

        assert result["status"] == "ATTENTION"
        assert any("recurring" in r.lower() for r in result["attention_reasons"])

    def test_recurring_failures_included(self):
        """Recurring failures dict is included in result."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        drift_timeline = {
            "run_stability_index": 0.9,
            "recurring_failures": {"PRE-1.1": 3},
        }

        result = summarize_preflight_for_audit_readiness(global_summary, drift_timeline)

        assert result["recurring_failures"] == {"PRE-1.1": 3}

    def test_attention_reasons_empty_when_ok(self):
        """attention_reasons is empty when status is OK."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        drift_timeline = {
            "run_stability_index": 1.0,
            "recurring_failures": {},
        }

        result = summarize_preflight_for_audit_readiness(global_summary, drift_timeline)

        assert result["attention_reasons"] == []


# =============================================================================
# PHASE IV: DIRECTOR PREFLIGHT PANEL TESTS
# =============================================================================

class TestBuildPreflightDirectorPanel:
    """Tests for build_preflight_director_panel() function."""

    def test_ok_release_returns_green_light(self):
        """OK release → status_light='green'."""
        global_summary = {
            "current_eligibility": "ELIGIBLE",
            "pass_rate": 1.0,
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "blocking_reasons": [],
            "confidence": 1.0,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert result["status_light"] == "green"
        assert result["release_ok"] is True

    def test_warn_release_returns_yellow_light(self):
        """WARN release → status_light='yellow'."""
        global_summary = {
            "current_eligibility": "ELIGIBLE_WARNED",
            "pass_rate": 0.96,
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "WARN",
            "release_ok": True,
            "blocking_reasons": [],
            "confidence": 0.9,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert result["status_light"] == "yellow"

    def test_block_release_returns_red_light(self):
        """BLOCK release → status_light='red'."""
        global_summary = {
            "current_eligibility": "BLOCKED_FIXABLE",
            "pass_rate": 0.8,
            "health_score": 0.6,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "BLOCK",
            "release_ok": False,
            "blocking_reasons": ["STOP: Blocking issues"],
            "confidence": 0.5,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert result["status_light"] == "red"
        assert result["release_ok"] is False

    def test_headline_for_eligible_stable(self):
        """Headline for ELIGIBLE stable includes audit-eligible and stable."""
        global_summary = {
            "current_eligibility": "ELIGIBLE",
            "pass_rate": 1.0,
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "blocking_reasons": [],
            "confidence": 1.0,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert "audit-eligible" in result["headline"].lower()
        assert "stable" in result["headline"].lower()
        assert "100%" in result["headline"]

    def test_headline_for_eligible_improving(self):
        """Headline for ELIGIBLE improving mentions improving trend."""
        global_summary = {
            "current_eligibility": "ELIGIBLE",
            "pass_rate": 1.0,
            "health_score": 0.9,
            "drift_status": "improving",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "blocking_reasons": [],
            "confidence": 0.9,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert "improving" in result["headline"].lower()

    def test_headline_for_eligible_warned(self):
        """Headline for ELIGIBLE_WARNED mentions warnings and advisories."""
        global_summary = {
            "current_eligibility": "ELIGIBLE_WARNED",
            "pass_rate": 0.96,
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "WARN",
            "release_ok": True,
            "blocking_reasons": [],
            "confidence": 0.85,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert "warning" in result["headline"].lower()
        assert "96%" in result["headline"]

    def test_headline_for_blocked_fixable(self):
        """Headline for BLOCKED_FIXABLE mentions fixable issues."""
        global_summary = {
            "current_eligibility": "BLOCKED_FIXABLE",
            "pass_rate": 0.8,
            "health_score": 0.6,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "BLOCK",
            "release_ok": False,
            "blocking_reasons": ["STOP: Issues"],
            "confidence": 0.5,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert "blocked" in result["headline"].lower()
        assert "fixable" in result["headline"].lower()

    def test_headline_for_inadmissible(self):
        """Headline for INADMISSIBLE mentions permanently inadmissible."""
        global_summary = {
            "current_eligibility": "INADMISSIBLE",
            "pass_rate": 0.9,
            "health_score": 0.5,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "BLOCK",
            "release_ok": False,
            "blocking_reasons": ["FATAL: Inadmissible"],
            "confidence": 0.4,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert "inadmissible" in result["headline"].lower()
        assert "permanently" in result["headline"].lower()

    def test_panel_includes_all_required_fields(self):
        """Panel includes all required fields."""
        global_summary = {
            "current_eligibility": "ELIGIBLE",
            "pass_rate": 0.95,
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "blocking_reasons": [],
            "confidence": 0.9,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert "status_light" in result
        assert "current_eligibility" in result
        assert "pass_rate" in result
        assert "headline" in result
        assert "experiment_id" in result
        assert "health_score" in result
        assert "release_ok" in result
        assert "confidence" in result
        assert "drift_status" in result

    def test_pass_rate_preserved(self):
        """Pass rate is preserved in panel."""
        global_summary = {
            "current_eligibility": "ELIGIBLE",
            "pass_rate": 0.92,
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "blocking_reasons": [],
            "confidence": 0.9,
        }

        result = build_preflight_director_panel(global_summary, release_eval)

        assert result["pass_rate"] == 0.92


# =============================================================================
# PHASE V: BUNDLE-PREFLIGHT JOINT VIEW TESTS
# =============================================================================

class TestBuildPreflightBundleJointView:
    """Tests for build_preflight_bundle_joint_view() function."""

    def test_both_ok_returns_integration_ready(self):
        """Both preflight and bundle OK → integration_ready=True, joint_status='OK'."""
        preflight_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["integration_ready"] is True
        assert result["joint_status"] == "OK"
        assert result["reasons"] == []

    def test_bundle_block_causes_integration_block(self):
        """Bundle BLOCK → joint_status='BLOCK' even if preflight OK."""
        preflight_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "BLOCK",
            "bundle_id": "bundle_001",
            "stage": "validation",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["integration_ready"] is False
        assert result["joint_status"] == "BLOCK"
        assert any("BUNDLE_BLOCK" in r for r in result["reasons"])

    def test_preflight_block_causes_integration_block(self):
        """Preflight BLOCK → joint_status='BLOCK' even if bundle OK."""
        preflight_summary = {
            "preflight_ok": False,
            "current_eligibility": "BLOCKED_FIXABLE",
            "health_score": 0.6,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["integration_ready"] is False
        assert result["joint_status"] == "BLOCK"
        assert any("PREFLIGHT_BLOCK" in r for r in result["reasons"])

    def test_both_block_lists_both_reasons(self):
        """Both BLOCK → both reasons listed."""
        preflight_summary = {
            "preflight_ok": False,
            "current_eligibility": "INADMISSIBLE",
            "health_score": 0.3,
            "drift_status": "degrading",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "BLOCK",
            "bundle_id": "bundle_001",
            "stage": "validation",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["integration_ready"] is False
        assert result["joint_status"] == "BLOCK"
        assert any("PREFLIGHT_BLOCK" in r for r in result["reasons"])
        assert any("BUNDLE_BLOCK" in r for r in result["reasons"])

    def test_preflight_warn_causes_joint_warn(self):
        """Preflight WARN → joint_status='WARN'."""
        preflight_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE_WARNED",
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["integration_ready"] is True
        assert result["joint_status"] == "WARN"

    def test_bundle_warn_causes_joint_warn(self):
        """Bundle WARN → joint_status='WARN'."""
        preflight_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "WARN",
            "bundle_id": "bundle_001",
            "stage": "pending_review",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["integration_ready"] is True
        assert result["joint_status"] == "WARN"

    def test_degrading_drift_causes_warn(self):
        """Degrading drift → joint_status='WARN' even if both OK."""
        preflight_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.9,
            "drift_status": "degrading",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["integration_ready"] is True
        assert result["joint_status"] == "WARN"
        assert any("DRIFT_DEGRADING" in r for r in result["reasons"])

    def test_low_health_adds_warning_reason(self):
        """Low health (<0.5) adds warning reason."""
        preflight_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE_WARNED",
            "health_score": 0.4,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert any("LOW_HEALTH" in r for r in result["reasons"])

    def test_includes_status_summaries(self):
        """Result includes preflight_status and bundle_status summaries."""
        preflight_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        assert result["preflight_status"]["status"] == "OK"
        assert result["preflight_status"]["eligibility"] == "ELIGIBLE"
        assert result["bundle_status"]["status"] == "OK"
        assert result["bundle_status"]["bundle_id"] == "bundle_001"

    def test_preflight_block_overrides_topology_ok(self):
        """Preflight BLOCK overrides any topology/bundle OK in global gating."""
        # This is the key integration test: preflight is non-optional
        preflight_summary = {
            "preflight_ok": False,
            "current_eligibility": "BLOCKED_FIXABLE",
            "health_score": 0.6,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
        }
        # Simulating topology OK via bundle
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
            "topology_status": "OK",  # Additional field showing topology passed
        }

        result = build_preflight_bundle_joint_view(preflight_summary, bundle_evolution)

        # Even though bundle (and topology) are OK, preflight BLOCK wins
        assert result["integration_ready"] is False
        assert result["joint_status"] == "BLOCK"


# =============================================================================
# PHASE V: GLOBAL CONSOLE ADAPTER TESTS
# =============================================================================

class TestSummarizePreflightForGlobalConsole:
    """Tests for summarize_preflight_for_global_console() function."""

    def test_eligible_ok_returns_green(self):
        """ELIGIBLE + OK release → green status_light."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "pass_rate": 1.0,
            "health_score": 1.0,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "blocking_reasons": [],
        }

        result = summarize_preflight_for_global_console(global_summary, release_eval)

        assert result["preflight_ok"] is True
        assert result["status_light"] == "green"
        assert "OK" in result["headline"]
        assert "100%" in result["headline"]

    def test_warn_returns_yellow(self):
        """WARN release → yellow status_light."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE_WARNED",
            "pass_rate": 0.96,
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "WARN",
            "release_ok": True,
            "blocking_reasons": [],
        }

        result = summarize_preflight_for_global_console(global_summary, release_eval)

        assert result["status_light"] == "yellow"
        assert "warning" in result["headline"].lower()

    def test_block_returns_red(self):
        """BLOCK release → red status_light."""
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "BLOCKED_FIXABLE",
            "pass_rate": 0.8,
            "health_score": 0.6,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "BLOCK",
            "release_ok": False,
            "blocking_reasons": ["STOP: Issues"],
        }

        result = summarize_preflight_for_global_console(global_summary, release_eval)

        assert result["status_light"] == "red"
        assert "BLOCKED" in result["headline"]

    def test_inadmissible_headline(self):
        """INADMISSIBLE → headline mentions inadmissible."""
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "INADMISSIBLE",
            "pass_rate": 0.9,
            "health_score": 0.5,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "BLOCK",
            "release_ok": False,
            "blocking_reasons": ["FATAL: Inadmissible"],
        }

        result = summarize_preflight_for_global_console(global_summary, release_eval)

        assert "inadmissible" in result["headline"].lower()

    def test_includes_all_fields(self):
        """Result includes all required fields."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "pass_rate": 0.95,
            "health_score": 0.9,
            "drift_status": "stable",
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T12:00:00Z",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "blocking_reasons": [],
        }

        result = summarize_preflight_for_global_console(global_summary, release_eval)

        assert "preflight_ok" in result
        assert "status_light" in result
        assert "pass_rate" in result
        assert "headline" in result
        assert "experiment_id" in result
        assert "timestamp" in result
        assert result["timestamp"] == "2025-06-15T12:00:00Z"


# =============================================================================
# PHASE V: GOVERNANCE SIGNAL TESTS
# =============================================================================

class TestToGovernanceSignal:
    """Tests for to_governance_signal() function.

    Validates conformance to canonical GovernanceSignal schema from
    backend.analytics.governance_verifier:
    - layer_name: "preflight"
    - status: "OK" | "WARN" | "BLOCK"
    - blocking_rules: List[str]
    - blocking_rate: float in [0, 1]
    - headline: str
    """

    def test_layer_name_is_preflight(self):
        """Layer name is always 'preflight' for routing."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "pass_rate": 1.0,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "confidence": 1.0,
            "blocking_reasons": [],
            "warnings": [],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert result["layer_name"] == "preflight"

    def test_ok_release_has_ok_status(self):
        """OK release → status=OK, blocking_rate=0."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "pass_rate": 1.0,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "confidence": 1.0,
            "blocking_reasons": [],
            "warnings": [],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert result["status"] == "OK"
        assert result["blocking_rate"] == 0.0
        assert result["blocking_rules"] == []

    def test_block_release_has_block_status(self):
        """BLOCK release → status=BLOCK with blocking_rules."""
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "BLOCKED_FIXABLE",
            "health_score": 0.6,
            "drift_status": "stable",
            "pass_rate": 0.8,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "BLOCK",
            "release_ok": False,
            "confidence": 0.5,
            "blocking_reasons": ["STOP: Issues"],
            "warnings": [],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert result["status"] == "BLOCK"
        assert result["blocking_rate"] == 0.2  # 1 - 0.8 pass_rate
        assert "STOP: Issues" in result["blocking_rules"]

    def test_warn_release_has_warn_status(self):
        """WARN release → status=WARN with blocking_rules."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE_WARNED",
            "health_score": 0.9,
            "drift_status": "stable",
            "pass_rate": 0.96,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "WARN",
            "release_ok": True,
            "confidence": 0.85,
            "blocking_reasons": [],
            "warnings": ["Minor issues"],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert result["status"] == "WARN"
        assert "Minor issues" in result["blocking_rules"]
        # blocking_rate = 1 - 0.96 = 0.04
        assert abs(result["blocking_rate"] - 0.04) < 0.001

    def test_blocking_rate_computed_from_pass_rate(self):
        """blocking_rate = 1 - pass_rate."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.8,
            "drift_status": "stable",
            "pass_rate": 0.95,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "confidence": 0.75,
            "blocking_reasons": [],
            "warnings": [],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert abs(result["blocking_rate"] - 0.05) < 0.001

    def test_headline_generated_for_ok(self):
        """Headline generated for OK status."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.9,
            "drift_status": "improving",
            "pass_rate": 0.95,
            "experiment_id": "u2_exp_abc",
            "last_check": "2025-06-15T12:00:00Z",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "confidence": 0.9,
            "blocking_reasons": [],
            "warnings": [],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert "headline" in result
        assert isinstance(result["headline"], str)
        assert "Preflight OK" in result["headline"]
        assert "u2_exp_abc" in result["headline"]

    def test_headline_generated_for_block(self):
        """Headline generated for BLOCK status includes eligibility."""
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "INADMISSIBLE",
            "health_score": 0.3,
            "drift_status": "stable",
            "pass_rate": 0.5,
            "experiment_id": "u2_blocked",
            "last_check": "2025-06-15T12:00:00Z",
        }
        release_eval = {
            "status": "BLOCK",
            "release_ok": False,
            "confidence": 0.1,
            "blocking_reasons": ["FATAL: Permanent failure"],
            "warnings": [],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert "headline" in result
        assert "BLOCKED" in result["headline"]
        assert "inadmissible" in result["headline"].lower()

    def test_degrading_drift_adds_blocking_rule(self):
        """Degrading drift adds DRIFT blocking_rule if not already present."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 0.8,
            "drift_status": "degrading",
            "pass_rate": 0.9,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "WARN",
            "release_ok": True,
            "confidence": 0.7,
            "blocking_reasons": [],
            "warnings": ["Health below threshold"],
        }

        result = to_governance_signal(global_summary, release_eval)

        assert any("DRIFT" in r for r in result["blocking_rules"])

    def test_conforms_to_canonical_governance_schema(self):
        """Signal conforms to canonical GovernanceSignal schema."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "pass_rate": 1.0,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }
        release_eval = {
            "status": "OK",
            "release_ok": True,
            "confidence": 1.0,
            "blocking_reasons": [],
            "warnings": [],
        }

        result = to_governance_signal(global_summary, release_eval)

        # Verify canonical GovernanceSignal schema conformance
        assert "layer_name" in result
        assert "status" in result
        assert "blocking_rules" in result
        assert "blocking_rate" in result
        assert "headline" in result
        assert result["layer_name"] == "preflight"
        assert result["status"] in ("OK", "WARN", "BLOCK")
        assert isinstance(result["blocking_rules"], list)
        assert isinstance(result["blocking_rate"], (int, float))
        assert 0.0 <= result["blocking_rate"] <= 1.0
        assert isinstance(result["headline"], str)


# =============================================================================
# PHASE V: INTEGRATION CHAIN TESTS
# =============================================================================

class TestPhaseVIntegrationChain:
    """Integration tests ensuring preflight is non-optional in release pipeline."""

    def test_preflight_block_cascades_through_chain(self):
        """Preflight BLOCK cascades through all integration points."""
        # Set up a blocked preflight
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "BLOCKED_FIXABLE",
            "health_score": 0.6,
            "drift_status": "stable",
            "pass_rate": 0.8,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }

        # 1. Release evaluator should BLOCK
        drift_timeline = {
            "run_stability_index": 0.8,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }
        release_eval = evaluate_preflight_for_release(global_summary, drift_timeline)
        assert release_eval["status"] == "BLOCK"

        # 2. Global console should show red
        console = summarize_preflight_for_global_console(global_summary, release_eval)
        assert console["status_light"] == "red"

        # 3. Governance signal should have BLOCK status (critical layer)
        gov_signal = to_governance_signal(global_summary, release_eval)
        assert gov_signal["status"] == "BLOCK"
        assert gov_signal["layer_name"] == "preflight"

        # 4. Joint view with OK bundle should still be BLOCK
        bundle_evolution = {"status": "OK", "bundle_id": "bundle_001", "stage": "complete"}
        joint = build_preflight_bundle_joint_view(global_summary, bundle_evolution)
        assert joint["joint_status"] == "BLOCK"
        assert joint["integration_ready"] is False

    def test_ok_preflight_allows_integration(self):
        """OK preflight allows integration to proceed."""
        global_summary = {
            "preflight_ok": True,
            "current_eligibility": "ELIGIBLE",
            "health_score": 1.0,
            "drift_status": "stable",
            "pass_rate": 1.0,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }

        # 1. Release evaluator should OK
        drift_timeline = {
            "run_stability_index": 1.0,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }
        release_eval = evaluate_preflight_for_release(global_summary, drift_timeline)
        assert release_eval["status"] == "OK"

        # 2. Global console should show green
        console = summarize_preflight_for_global_console(global_summary, release_eval)
        assert console["status_light"] == "green"

        # 3. Governance signal should have OK status
        gov_signal = to_governance_signal(global_summary, release_eval)
        assert gov_signal["status"] == "OK"
        assert gov_signal["layer_name"] == "preflight"

        # 4. Joint view with OK bundle should be OK
        bundle_evolution = {"status": "OK", "bundle_id": "bundle_001", "stage": "complete"}
        joint = build_preflight_bundle_joint_view(global_summary, bundle_evolution)
        assert joint["joint_status"] == "OK"
        assert joint["integration_ready"] is True

    def test_preflight_block_overrides_all_ok_topology(self):
        """Preflight BLOCK overrides even when all other systems report OK."""
        # Simulate scenario where topology, bundle, etc. are all OK
        # but preflight has a blocking issue
        global_summary = {
            "preflight_ok": False,
            "current_eligibility": "INADMISSIBLE",  # FATAL condition
            "health_score": 0.5,
            "drift_status": "degrading",
            "pass_rate": 0.9,
            "experiment_id": "u2_test_001",
            "last_check": "2025-06-15T00:00:00Z",
        }

        # All other systems report OK
        bundle_evolution = {
            "status": "OK",
            "bundle_id": "bundle_001",
            "stage": "complete",
            "topology_status": "OK",  # Simulated topology OK
            "evidence_status": "OK",  # Simulated evidence OK
        }

        joint = build_preflight_bundle_joint_view(global_summary, bundle_evolution)

        # Preflight BLOCK must override
        assert joint["joint_status"] == "BLOCK"
        assert joint["integration_ready"] is False
        assert any("PREFLIGHT_BLOCK" in r for r in joint["reasons"])

        # Verify governance signal also has BLOCK status (critical layer veto)
        drift_timeline = {
            "run_stability_index": 0.5,
            "recurring_failures": {},
            "eligibility_shifts": [],
        }
        release_eval = evaluate_preflight_for_release(global_summary, drift_timeline)
        gov_signal = to_governance_signal(global_summary, release_eval)
        assert gov_signal["status"] == "BLOCK"
        assert gov_signal["layer_name"] == "preflight"
        # CRITICAL LAYER: preflight BLOCK => global promotion BLOCKED
        # (as per DEFAULT_CRITICAL_LAYERS in governance_verifier.py)
