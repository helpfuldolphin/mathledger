#!/usr/bin/env python3
"""
Tests for the U2 Uplift Experiment Auditor and Manifest Verifier.

These tests verify:
- ManifestVerifier correctly validates manifest bindings
- audit_uplift_u2.py discovers artifacts correctly
- Phase II labels are validated
- Raw counts are computed (no interpretation)
"""

import json
import tempfile
from pathlib import Path

import pytest

from experiments.manifest_verifier import (
    CheckResult,
    ManifestVerifier,
    VerificationReport,
)
from experiments.audit_uplift_u2 import (
    PHASE_II_LABEL,
    check_log_phase_ii_labels,
    count_log_records,
    discover_artifacts,
)


class TestManifestVerifier:
    """Tests for the ManifestVerifier class."""

    def test_valid_manifest(self, tmp_path: Path):
        """Test verification of a valid manifest."""
        manifest_data = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": 10,
            "initial_seed": 42,
            "slice_config_hash": "abc123",
            "outputs": {"results": "test.jsonl"},
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        verifier = ManifestVerifier(manifest_path, tmp_path)
        report = verifier.validate_all()

        assert report.overall_pass is True
        assert any(c.name == "manifest_exists" and c.passed for c in report.checks)
        assert any(c.name == "phase_ii_label" and c.passed for c in report.checks)
        assert any(c.name == "required_fields" and c.passed for c in report.checks)

    def test_missing_phase_ii_label(self, tmp_path: Path):
        """Test that missing Phase II label is detected."""
        manifest_data = {
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": 10,
            "initial_seed": 42,
            "slice_config_hash": "abc123",
            "outputs": {},
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        verifier = ManifestVerifier(manifest_path, tmp_path)
        report = verifier.validate_all()

        assert report.overall_pass is False
        label_check = next((c for c in report.checks if c.name == "phase_ii_label"), None)
        assert label_check is not None
        assert label_check.passed is False

    def test_missing_required_fields(self, tmp_path: Path):
        """Test that missing required fields are detected."""
        manifest_data = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "slice": "test_slice",
            # Missing: mode, cycles, initial_seed, slice_config_hash, outputs
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        verifier = ManifestVerifier(manifest_path, tmp_path)
        report = verifier.validate_all()

        assert report.overall_pass is False
        fields_check = next((c for c in report.checks if c.name == "required_fields"), None)
        assert fields_check is not None
        assert fields_check.passed is False

    def test_nonexistent_manifest(self, tmp_path: Path):
        """Test behavior with nonexistent manifest."""
        manifest_path = tmp_path / "nonexistent.json"
        verifier = ManifestVerifier(manifest_path, tmp_path)
        report = verifier.validate_all()

        assert report.overall_pass is False
        assert any(c.name == "manifest_exists" and not c.passed for c in report.checks)

    def test_invalid_json(self, tmp_path: Path):
        """Test behavior with invalid JSON."""
        manifest_path = tmp_path / "invalid.json"
        manifest_path.write_text("{ invalid json }")

        verifier = ManifestVerifier(manifest_path, tmp_path)
        report = verifier.validate_all()

        assert report.overall_pass is False

    def test_report_to_dict(self, tmp_path: Path):
        """Test that report can be converted to dict."""
        manifest_data = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "slice": "test",
            "mode": "baseline",
            "cycles": 1,
            "initial_seed": 1,
            "slice_config_hash": "hash",
            "outputs": {},
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        verifier = ManifestVerifier(manifest_path, tmp_path)
        report = verifier.validate_all()
        report_dict = report.to_dict()

        assert "manifest_path" in report_dict
        assert "overall_pass" in report_dict
        assert "checks" in report_dict
        assert isinstance(report_dict["checks"], list)

    def test_report_to_markdown(self, tmp_path: Path):
        """Test that report can be converted to Markdown."""
        manifest_data = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "slice": "test",
            "mode": "baseline",
            "cycles": 1,
            "initial_seed": 1,
            "slice_config_hash": "hash",
            "outputs": {},
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        verifier = ManifestVerifier(manifest_path, tmp_path)
        report = verifier.validate_all()
        markdown = report.to_markdown()

        assert "# Manifest Verification Report" in markdown
        assert "Overall Result" in markdown


class TestAuditHelpers:
    """Tests for audit helper functions."""

    def test_discover_artifacts_all_present(self, tmp_path: Path):
        """Test artifact discovery when all files present."""
        (tmp_path / "experiment_manifest.json").write_text("{}")
        (tmp_path / "baseline_log.jsonl").write_text("{}\n")
        (tmp_path / "rfl_log.jsonl").write_text("{}\n")
        (tmp_path / "ht_series.json").write_text("[]")

        manifest, baseline, rfl, ht_series, notes = discover_artifacts(tmp_path)

        assert manifest is not None
        assert baseline is not None
        assert rfl is not None
        assert ht_series is not None

    def test_discover_artifacts_missing(self, tmp_path: Path):
        """Test artifact discovery with missing files."""
        manifest, baseline, rfl, ht_series, notes = discover_artifacts(tmp_path)

        assert manifest is None
        assert baseline is None
        assert rfl is None
        assert ht_series is None
        assert any("WARNING" in note or "INFO" in note for note in notes)

    def test_discover_artifacts_nonexistent_dir(self, tmp_path: Path):
        """Test artifact discovery with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        manifest, baseline, rfl, ht_series, notes = discover_artifacts(nonexistent)

        assert manifest is None
        assert any("ERROR" in note for note in notes)

    def test_count_log_records(self, tmp_path: Path):
        """Test log record counting."""
        log_path = tmp_path / "test.jsonl"
        log_path.write_text(
            '{"cycle": 0, "success": true}\n'
            '{"cycle": 1, "success": true}\n'
            '{"cycle": 2, "success": false}\n'
        )

        counts = count_log_records(log_path)

        assert counts["total_records"] == 3
        assert counts["cycles"] == 3
        assert counts["successes"] == 2
        assert counts["abstentions"] == 1

    def test_count_log_records_empty(self, tmp_path: Path):
        """Test log record counting with empty file."""
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")

        counts = count_log_records(log_path)

        assert counts["total_records"] == 0
        assert counts["cycles"] == 0

    def test_count_log_records_nonexistent(self, tmp_path: Path):
        """Test log record counting with nonexistent file."""
        log_path = tmp_path / "nonexistent.jsonl"
        counts = count_log_records(log_path)

        assert counts["total_records"] == 0

    def test_check_log_phase_ii_labels_all_present(self, tmp_path: Path):
        """Test Phase II label check when all records have labels."""
        log_path = tmp_path / "test.jsonl"
        log_path.write_text(
            '{"label": "PHASE II — NOT USED IN PHASE I"}\n'
            '{"label": "PHASE II — NOT USED IN PHASE I"}\n'
        )

        result = check_log_phase_ii_labels(log_path)

        assert result.passed is True
        assert "All 2 records" in result.message

    def test_check_log_phase_ii_labels_missing(self, tmp_path: Path):
        """Test Phase II label check when labels are missing."""
        log_path = tmp_path / "test.jsonl"
        log_path.write_text('{"cycle": 0}\n{"cycle": 1}\n')

        result = check_log_phase_ii_labels(log_path)

        assert result.passed is False
        assert "No records have PHASE II label" in result.message

    def test_check_log_phase_ii_labels_partial(self, tmp_path: Path):
        """Test Phase II label check with partial labels."""
        log_path = tmp_path / "test.jsonl"
        log_path.write_text(
            '{"label": "PHASE II — NOT USED IN PHASE I"}\n'
            '{"cycle": 1}\n'
        )

        result = check_log_phase_ii_labels(log_path)

        assert result.passed is False
        assert "1/2" in result.message


class TestCheckResult:
    """Tests for the CheckResult dataclass."""

    def test_check_result_pass(self):
        """Test passing check result."""
        result = CheckResult(
            name="test_check",
            passed=True,
            message="Test passed",
        )
        assert result.passed is True
        assert result.name == "test_check"

    def test_check_result_fail(self):
        """Test failing check result with expected/actual."""
        result = CheckResult(
            name="hash_check",
            passed=False,
            message="Hash mismatch",
            expected="abc123",
            actual="xyz789",
        )
        assert result.passed is False
        assert result.expected == "abc123"
        assert result.actual == "xyz789"


class TestVerificationReport:
    """Tests for the VerificationReport class."""

    def test_add_check_pass(self):
        """Test adding passing check."""
        report = VerificationReport(manifest_path="/test/path")
        report.add_check(CheckResult("test", True, "passed"))

        assert report.overall_pass is True
        assert len(report.checks) == 1

    def test_add_check_fail(self):
        """Test adding failing check sets overall_pass to False."""
        report = VerificationReport(manifest_path="/test/path")
        report.add_check(CheckResult("test1", True, "passed"))
        report.add_check(CheckResult("test2", False, "failed"))

        assert report.overall_pass is False
        assert len(report.checks) == 2
