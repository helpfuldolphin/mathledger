#!/usr/bin/env python3
"""
Tests for Phase II Manifest Verifier and Audit Uplift U2

These tests validate the structural integrity checking functionality
without computing uplift or modifying any files.

Exit Codes tested:
    0 - PASS: All checks OK
    1 - FAIL: Structural/cryptographic failure
    2 - MISSING: Missing or ambiguous artifacts
"""

import json
import tempfile
from pathlib import Path

import pytest

from experiments.manifest_verifier import (
    ManifestVerifier,
    ManifestVerificationReport,
    VerificationResult,
    verify_manifest_file,
)
from experiments.audit_uplift_u2 import (
    EXIT_PASS,
    EXIT_FAIL,
    EXIT_MISSING,
    audit_experiment,
    count_jsonl_records,
    discover_artifacts,
    generate_audit_report,
    validate_log_structure,
)


@pytest.fixture
def temp_experiment_dir():
    """Create a temporary experiment directory with test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestManifestVerifier:
    """Tests for ManifestVerifier class."""

    def test_verify_missing_manifest(self, temp_experiment_dir):
        """Test verification when manifest file doesn't exist."""
        manifest_path = temp_experiment_dir / "nonexistent_manifest.json"
        verifier = ManifestVerifier(manifest_path)
        report = verifier.verify_all()
        
        assert report.overall_status == "MISSING"
        assert not report.all_passed()

    def test_verify_invalid_json_manifest(self, temp_experiment_dir):
        """Test verification with invalid JSON manifest."""
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest_path.write_text("{ invalid json }")
        
        verifier = ManifestVerifier(manifest_path)
        report = verifier.verify_all()
        
        assert report.overall_status == "FAIL"
        assert any("Invalid JSON" in r.message for r in report.results)

    def test_verify_valid_manifest_cycle_count_match(self, temp_experiment_dir):
        """Test cycle count verification with matching counts."""
        # Create baseline log with 10 records
        baseline_log = temp_experiment_dir / "uplift_u2_slice_baseline.jsonl"
        with open(baseline_log, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I",
                    "success": True
                }) + "\n")
        
        # Create RFL log with 10 records
        rfl_log = temp_experiment_dir / "uplift_u2_slice_rfl.jsonl"
        with open(rfl_log, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I",
                    "success": True
                }) + "\n")
        
        # Create manifest with matching cycle counts
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "n_cycles": {
                "baseline": 10,
                "rfl": 10
            },
            "outputs": {
                "baseline_log": str(baseline_log),
                "rfl_log": str(rfl_log)
            },
            "prereg_hash": "abc123"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path, temp_experiment_dir)
        report = verifier.verify_all()
        
        # Find cycle count result
        cycle_result = next(r for r in report.results if r.check_name == "cycle_count")
        assert cycle_result.passed
        assert "match" in cycle_result.message.lower() or cycle_result.passed

    def test_verify_cycle_count_mismatch(self, temp_experiment_dir):
        """Test cycle count verification with mismatching counts."""
        # Create baseline log with 5 records (mismatch)
        baseline_log = temp_experiment_dir / "uplift_u2_slice_baseline.jsonl"
        with open(baseline_log, 'w') as f:
            for i in range(5):
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I"
                }) + "\n")
        
        # Create manifest expecting 10 cycles
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "n_cycles": {
                "baseline": 10
            },
            "outputs": {
                "baseline_log": str(baseline_log)
            },
            "prereg_hash": "abc123"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path, temp_experiment_dir)
        report = verifier.verify_all()
        
        # Find cycle count result
        cycle_result = next(r for r in report.results if r.check_name == "cycle_count")
        assert not cycle_result.passed
        assert "mismatch" in cycle_result.message.lower()

    def test_verify_ht_series_length_match(self, temp_experiment_dir):
        """Test ht_series verification with matching length."""
        # Create ht_series.json with 10 entries - using simple string hashes
        ht_series_path = temp_experiment_dir / "ht_series.json"
        ht_series = [f"hash_{i}" for i in range(10)]
        with open(ht_series_path, 'w') as f:
            json.dump(ht_series, f)
        
        # Create manifest with matching cycle count - ht_first/ht_last as simple strings
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "cycles": 10,
            "prereg_hash": "abc123",
            "ht_series": {
                "ht_first": "hash_0",
                "ht_last": "hash_9"
            }
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path, temp_experiment_dir)
        report = verifier.verify_all()
        
        # Find ht_series result
        ht_result = next(r for r in report.results if r.check_name == "ht_series")
        assert ht_result.passed

    def test_verify_ht_series_length_mismatch(self, temp_experiment_dir):
        """Test ht_series verification with mismatching length."""
        # Create ht_series.json with 5 entries
        ht_series_path = temp_experiment_dir / "ht_series.json"
        ht_series = [{"h_t": f"hash_{i}"} for i in range(5)]
        with open(ht_series_path, 'w') as f:
            json.dump(ht_series, f)
        
        # Create manifest expecting 10 cycles
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "cycles": 10,
            "prereg_hash": "abc123"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path, temp_experiment_dir)
        report = verifier.verify_all()
        
        # Find ht_series result
        ht_result = next(r for r in report.results if r.check_name == "ht_series")
        assert not ht_result.passed
        assert "length" in ht_result.message.lower()

    def test_verify_label_constraint_pass(self, temp_experiment_dir):
        """Test label constraint verification passes with Phase II label."""
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "prereg_hash": "abc123"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path)
        report = verifier.verify_all()
        
        label_result = next(r for r in report.results if r.check_name == "label_constraint")
        assert label_result.passed

    def test_verify_label_constraint_fail(self, temp_experiment_dir):
        """Test label constraint verification fails without Phase II label."""
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "Some other label",
            "prereg_hash": "abc123"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path)
        report = verifier.verify_all()
        
        label_result = next(r for r in report.results if r.check_name == "label_constraint")
        assert not label_result.passed

    def test_verify_binding_pass(self, temp_experiment_dir):
        """Test binding verification passes with prereg_hash."""
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "prereg_hash": "abc123def456"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path)
        report = verifier.verify_all()
        
        binding_result = next(r for r in report.results if r.check_name == "binding")
        assert binding_result.passed

    def test_verify_binding_fail(self, temp_experiment_dir):
        """Test binding verification fails without binding hash."""
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path)
        report = verifier.verify_all()
        
        binding_result = next(r for r in report.results if r.check_name == "binding")
        assert not binding_result.passed

    def test_verify_artifact_hashes_match(self, temp_experiment_dir):
        """Test artifact hash verification with matching hashes."""
        import hashlib
        
        # Create a log file
        log_path = temp_experiment_dir / "test_log.jsonl"
        log_content = '{"cycle": 0}\n{"cycle": 1}\n'
        log_path.write_text(log_content)
        
        # Compute actual hash
        actual_hash = hashlib.sha256(log_content.encode()).hexdigest()
        
        # Create manifest with correct hash
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "prereg_hash": "abc123",
            "artifacts": {
                "logs": [
                    {
                        "path": str(log_path),
                        "sha256": actual_hash,
                        "type": "jsonl"
                    }
                ]
            }
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path, temp_experiment_dir)
        report = verifier.verify_all()
        
        hash_result = next(r for r in report.results if r.check_name == "artifact_hashes")
        assert hash_result.passed

    def test_verify_artifact_hashes_mismatch(self, temp_experiment_dir):
        """Test artifact hash verification with mismatching hashes."""
        # Create a log file
        log_path = temp_experiment_dir / "test_log.jsonl"
        log_path.write_text('{"cycle": 0}\n')
        
        # Create manifest with wrong hash
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "prereg_hash": "abc123",
            "artifacts": {
                "logs": [
                    {
                        "path": str(log_path),
                        "sha256": "wrong_hash_value_that_does_not_match",
                        "type": "jsonl"
                    }
                ]
            }
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        verifier = ManifestVerifier(manifest_path, temp_experiment_dir)
        report = verifier.verify_all()
        
        hash_result = next(r for r in report.results if r.check_name == "artifact_hashes")
        assert not hash_result.passed
        assert "mismatch" in hash_result.message.lower()


class TestAuditUpliftU2:
    """Tests for audit_uplift_u2 module functions."""

    def test_count_jsonl_records(self, temp_experiment_dir):
        """Test counting JSONL records."""
        log_path = temp_experiment_dir / "test.jsonl"
        with open(log_path, 'w') as f:
            for i in range(15):
                f.write(json.dumps({"cycle": i}) + "\n")
        
        count = count_jsonl_records(log_path)
        assert count == 15

    def test_count_jsonl_records_missing_file(self, temp_experiment_dir):
        """Test counting JSONL records for missing file."""
        log_path = temp_experiment_dir / "nonexistent.jsonl"
        count = count_jsonl_records(log_path)
        assert count is None

    def test_discover_artifacts_with_manifest(self, temp_experiment_dir):
        """Test artifact discovery with manifest."""
        # Create manifest
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest_path.write_text("{}")
        
        # Create logs
        baseline_log = temp_experiment_dir / "uplift_u2_slice_baseline.jsonl"
        baseline_log.write_text('{"cycle": 0}\n')
        
        rfl_log = temp_experiment_dir / "uplift_u2_slice_rfl.jsonl"
        rfl_log.write_text('{"cycle": 0}\n')
        
        artifacts = discover_artifacts(temp_experiment_dir)
        
        assert artifacts["manifest"] is not None
        assert artifacts["baseline_log"] is not None
        assert artifacts["rfl_log"] is not None

    def test_validate_log_structure_valid(self, temp_experiment_dir):
        """Test log validation with valid log."""
        log_path = temp_experiment_dir / "test.jsonl"
        with open(log_path, 'w') as f:
            for i in range(5):
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I"
                }) + "\n")
        
        result = validate_log_structure(log_path)
        
        assert result["exists"]
        assert result["record_count"] == 5
        assert result["has_phase_ii_label"]
        assert result["has_cycle_field"]
        assert result["is_valid"]

    def test_validate_log_structure_missing_label(self, temp_experiment_dir):
        """Test log validation with missing Phase II label."""
        log_path = temp_experiment_dir / "test.jsonl"
        with open(log_path, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"cycle": i}) + "\n")
        
        result = validate_log_structure(log_path)
        
        assert not result["has_phase_ii_label"]
        assert not result["is_valid"]
        assert any("Phase II label" in issue for issue in result["issues"])

    def test_validate_log_structure_empty_file(self, temp_experiment_dir):
        """Test log validation with empty file."""
        log_path = temp_experiment_dir / "test.jsonl"
        log_path.write_text("")
        
        result = validate_log_structure(log_path)
        
        assert not result["is_valid"]
        assert any("empty" in issue.lower() for issue in result["issues"])


class TestAuditExperimentIntegration:
    """Integration tests for audit_experiment function."""

    def test_audit_experiment_pass(self, temp_experiment_dir):
        """Test full audit with passing experiment."""
        # Create valid baseline log
        baseline_log = temp_experiment_dir / "uplift_u2_slice_baseline.jsonl"
        with open(baseline_log, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I"
                }) + "\n")
        
        # Create valid RFL log
        rfl_log = temp_experiment_dir / "uplift_u2_slice_rfl.jsonl"
        with open(rfl_log, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I"
                }) + "\n")
        
        # Create valid manifest
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "n_cycles": {
                "baseline": 10,
                "rfl": 10
            },
            "outputs": {
                "baseline_log": str(baseline_log),
                "rfl_log": str(rfl_log)
            },
            "prereg_hash": "abc123"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        exit_code = audit_experiment(experiment_dir=temp_experiment_dir)
        
        assert exit_code == EXIT_PASS
        
        # Verify reports were generated
        assert (temp_experiment_dir / "audit_report.json").exists()
        assert (temp_experiment_dir / "audit_report.md").exists()

    def test_audit_experiment_fail_cycle_mismatch(self, temp_experiment_dir):
        """Test full audit with cycle count mismatch."""
        # Create baseline log with wrong count
        baseline_log = temp_experiment_dir / "uplift_u2_slice_baseline.jsonl"
        with open(baseline_log, 'w') as f:
            for i in range(5):  # Only 5, but manifest says 10
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I"
                }) + "\n")
        
        # Create manifest expecting 10 cycles
        manifest_path = temp_experiment_dir / "manifest.json"
        manifest = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "n_cycles": {
                "baseline": 10
            },
            "outputs": {
                "baseline_log": str(baseline_log)
            },
            "prereg_hash": "abc123"
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        exit_code = audit_experiment(experiment_dir=temp_experiment_dir)
        
        assert exit_code == EXIT_FAIL

    def test_audit_experiment_missing_manifest(self, temp_experiment_dir):
        """Test full audit with missing manifest."""
        # Create log without manifest
        baseline_log = temp_experiment_dir / "uplift_u2_slice_baseline.jsonl"
        with open(baseline_log, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "cycle": i,
                    "label": "PHASE II — NOT USED IN PHASE I"
                }) + "\n")
        
        exit_code = audit_experiment(experiment_dir=temp_experiment_dir)
        
        # Should return MISSING since no manifest found
        assert exit_code == EXIT_MISSING


class TestExitCodes:
    """Test that exit codes are correctly defined."""

    def test_exit_code_values(self):
        """Verify exit code constants have correct values."""
        assert EXIT_PASS == 0
        assert EXIT_FAIL == 1
        assert EXIT_MISSING == 2


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_verification_result_creation(self):
        """Test creating a VerificationResult."""
        result = VerificationResult(
            check_name="test_check",
            passed=True,
            message="Test passed",
            details={"key": "value"}
        )
        
        assert result.check_name == "test_check"
        assert result.passed
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}


class TestManifestVerificationReport:
    """Tests for ManifestVerificationReport dataclass."""

    def test_all_passed_true(self):
        """Test all_passed returns True when all checks pass."""
        report = ManifestVerificationReport(manifest_path="test.json")
        report.results = [
            VerificationResult("check1", True, "OK"),
            VerificationResult("check2", True, "OK"),
        ]
        
        assert report.all_passed()

    def test_all_passed_false(self):
        """Test all_passed returns False when any check fails."""
        report = ManifestVerificationReport(manifest_path="test.json")
        report.results = [
            VerificationResult("check1", True, "OK"),
            VerificationResult("check2", False, "Failed"),
        ]
        
        assert not report.all_passed()

    def test_compute_overall_status_pass(self):
        """Test overall status is PASS when all checks pass."""
        report = ManifestVerificationReport(manifest_path="test.json")
        report.results = [
            VerificationResult("check1", True, "OK"),
        ]
        
        status = report.compute_overall_status()
        assert status == "PASS"

    def test_compute_overall_status_missing(self):
        """Test overall status is MISSING when artifacts not found."""
        report = ManifestVerificationReport(manifest_path="test.json")
        report.results = [
            VerificationResult("check1", False, "File not found"),
        ]
        
        status = report.compute_overall_status()
        assert status == "MISSING"

    def test_compute_overall_status_fail(self):
        """Test overall status is FAIL for other failures."""
        report = ManifestVerificationReport(manifest_path="test.json")
        report.results = [
            VerificationResult("check1", False, "Mismatch detected"),
        ]
        
        status = report.compute_overall_status()
        assert status == "FAIL"

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = ManifestVerificationReport(manifest_path="test.json")
        report.results = [
            VerificationResult("check1", True, "OK", {"key": "value"}),
        ]
        report.overall_status = "PASS"
        
        d = report.to_dict()
        
        assert d["manifest_path"] == "test.json"
        assert d["overall_status"] == "PASS"
        assert len(d["results"]) == 1
        assert d["results"][0]["check_name"] == "check1"
