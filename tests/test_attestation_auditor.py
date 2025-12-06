"""
Tests for Attestation Auditor - Phase III Evidence Chain Ledger
================================================================

Comprehensive test suite covering:
- Manifest verification and hash utilities
- Single experiment auditing
- Multi-experiment auditing
- Evidence chain ledger construction
- CI hard gate evaluation
- Markdown rendering
"""

import json
import tempfile
from pathlib import Path

import pytest

from attestation.manifest_verifier import (
    compute_sha256_file,
    compute_sha256_string,
    compute_sha256_bytes,
    compute_sha256_json,
    verify_manifest_file_hash,
    load_and_verify_json,
)
from attestation.audit_uplift_u2 import (
    audit_experiment,
    render_audit_json,
    render_audit_markdown,
)
from attestation.audit_uplift_u2_all import (
    audit_all_experiments,
    aggregate_audit_summary,
    render_aggregate_json,
    render_aggregate_markdown,
)
from attestation.evidence_chain import (
    build_evidence_chain_ledger,
    evaluate_evidence_chain_for_ci,
    render_evidence_chain_section,
)


class TestManifestVerifier:
    """Test manifest verification and hash utilities."""
    
    def test_compute_sha256_string(self):
        """Test string hashing."""
        result = compute_sha256_string("hello world")
        assert len(result) == 64
        assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    
    def test_compute_sha256_bytes(self):
        """Test bytes hashing."""
        result = compute_sha256_bytes(b"hello world")
        assert len(result) == 64
        assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    
    def test_compute_sha256_json(self):
        """Test JSON hashing with canonical serialization."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        
        hash1 = compute_sha256_json(data1)
        hash2 = compute_sha256_json(data2)
        
        assert hash1 == hash2  # Order shouldn't matter
        assert len(hash1) == 64
    
    def test_compute_sha256_file(self, tmp_path):
        """Test file hashing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        
        result = compute_sha256_file(test_file)
        assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    
    def test_compute_sha256_file_nonexistent(self, tmp_path):
        """Test file hashing with nonexistent file."""
        result = compute_sha256_file(tmp_path / "nonexistent.txt")
        assert result is None
    
    def test_verify_manifest_file_hash(self, tmp_path):
        """Test manifest file hash verification."""
        test_file = tmp_path / "manifest.json"
        test_file.write_text('{"test": true}')
        
        expected_hash = compute_sha256_file(test_file)
        result = verify_manifest_file_hash(test_file, expected_hash)
        
        assert result["exists"] is True
        assert result["hash_match"] is True
        assert result["error"] is None
    
    def test_verify_manifest_file_hash_mismatch(self, tmp_path):
        """Test manifest file hash verification with mismatch."""
        test_file = tmp_path / "manifest.json"
        test_file.write_text('{"test": true}')
        
        result = verify_manifest_file_hash(test_file, "wrong_hash")
        
        assert result["exists"] is True
        assert result["hash_match"] is False
        assert "Hash mismatch" in result["error"]
    
    def test_load_and_verify_json(self, tmp_path):
        """Test JSON loading and verification."""
        test_file = tmp_path / "test.json"
        test_data = {"key": "value", "number": 42}
        test_file.write_text(json.dumps(test_data))
        
        result = load_and_verify_json(test_file)
        
        assert result["valid"] is True
        assert result["data"] == test_data
        assert result["error"] is None
        assert len(result["sha256"]) == 64
    
    def test_load_and_verify_json_invalid(self, tmp_path):
        """Test JSON loading with invalid JSON."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("not valid json {")
        
        result = load_and_verify_json(test_file)
        
        assert result["valid"] is False
        assert result["data"] is None
        assert "Invalid JSON" in result["error"]


class TestSingleExperimentAudit:
    """Test single experiment auditing."""
    
    def test_audit_experiment_pass(self, tmp_path):
        """Test auditing a valid experiment."""
        # Create experiment structure
        exp_dir = tmp_path / "experiments" / "EXP_001"
        exp_dir.mkdir(parents=True)
        
        # Create log file
        log_dir = tmp_path / "results"
        log_dir.mkdir()
        log_file = log_dir / "exp_001.jsonl"
        log_file.write_text('{"result": "success"}\n{"result": "success"}\n')
        log_hash = compute_sha256_file(log_file)
        
        # Create manifest
        manifest = {
            "experiment_id": "EXP_001",
            "artifacts": {
                "logs": [
                    {
                        "path": "results/exp_001.jsonl",
                        "type": "jsonl",
                        "sha256": log_hash
                    }
                ],
                "figures": []
            }
        }
        manifest_path = exp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        # Audit
        result = audit_experiment(exp_dir, tmp_path)
        
        assert result.status == "PASS"
        assert result.manifest_valid is True
        assert len(result.issues) == 0
        assert len(result.artifacts_checked) == 1
    
    def test_audit_experiment_empty_log(self, tmp_path):
        """Test auditing experiment with empty log file."""
        exp_dir = tmp_path / "experiments" / "EXP_002"
        exp_dir.mkdir(parents=True)
        
        # Create empty log file
        log_dir = tmp_path / "results"
        log_dir.mkdir()
        log_file = log_dir / "exp_002.jsonl"
        log_file.write_text("")  # Empty
        log_hash = compute_sha256_file(log_file)
        
        # Create manifest
        manifest = {
            "experiment_id": "EXP_002",
            "artifacts": {
                "logs": [
                    {
                        "path": "results/exp_002.jsonl",
                        "type": "jsonl",
                        "sha256": log_hash
                    }
                ],
                "figures": []
            }
        }
        manifest_path = exp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        # Audit
        result = audit_experiment(exp_dir, tmp_path)
        
        assert result.status == "FAIL"
        assert any("empty" in issue.lower() for issue in result.issues)
    
    def test_audit_experiment_hash_mismatch(self, tmp_path):
        """Test auditing experiment with hash mismatch."""
        exp_dir = tmp_path / "experiments" / "EXP_003"
        exp_dir.mkdir(parents=True)
        
        # Create log file
        log_dir = tmp_path / "results"
        log_dir.mkdir()
        log_file = log_dir / "exp_003.jsonl"
        log_file.write_text('{"result": "success"}\n')
        
        # Create manifest with wrong hash
        manifest = {
            "experiment_id": "EXP_003",
            "artifacts": {
                "logs": [
                    {
                        "path": "results/exp_003.jsonl",
                        "type": "jsonl",
                        "sha256": "wrong_hash_value"
                    }
                ],
                "figures": []
            }
        }
        manifest_path = exp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        # Audit
        result = audit_experiment(exp_dir, tmp_path)
        
        assert result.status == "FAIL"
        assert any("mismatch" in issue.lower() for issue in result.issues)
    
    def test_render_audit_json(self, tmp_path):
        """Test JSON rendering of audit result."""
        exp_dir = tmp_path / "experiments" / "EXP_004"
        exp_dir.mkdir(parents=True)
        
        manifest = {
            "experiment_id": "EXP_004",
            "artifacts": {"logs": [], "figures": []}
        }
        manifest_path = exp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        result = audit_experiment(exp_dir, tmp_path)
        output = render_audit_json(result)
        
        assert isinstance(output, str)
        parsed = json.loads(output)
        assert parsed["experiment_id"] == "EXP_004"
        assert parsed["status"] in ["PASS", "FAIL", "SKIP"]
    
    def test_render_audit_markdown(self, tmp_path):
        """Test Markdown rendering of audit result."""
        exp_dir = tmp_path / "experiments" / "EXP_005"
        exp_dir.mkdir(parents=True)
        
        manifest = {
            "experiment_id": "EXP_005",
            "artifacts": {"logs": [], "figures": []}
        }
        manifest_path = exp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        
        result = audit_experiment(exp_dir, tmp_path)
        output = render_audit_markdown(result)
        
        assert isinstance(output, str)
        assert "# Audit Report: EXP_005" in output
        assert "Status:" in output


class TestMultiExperimentAudit:
    """Test multi-experiment auditing."""
    
    def test_audit_all_experiments(self, tmp_path):
        """Test auditing multiple experiments."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        
        # Create two valid experiments
        for i in range(1, 3):
            exp_dir = experiments_dir / f"EXP_{i:03d}"
            exp_dir.mkdir()
            
            manifest = {
                "experiment_id": f"EXP_{i:03d}",
                "artifacts": {"logs": [], "figures": []}
            }
            manifest_path = exp_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
        
        results = audit_all_experiments(experiments_dir, tmp_path)
        
        assert len(results) == 2
        assert all(r.status == "PASS" for r in results)
    
    def test_aggregate_audit_summary(self, tmp_path):
        """Test aggregate summary generation."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        
        # Create mixed results
        for i in range(1, 4):
            exp_dir = experiments_dir / f"EXP_{i:03d}"
            exp_dir.mkdir()
            
            manifest = {
                "experiment_id": f"EXP_{i:03d}",
                "artifacts": {"logs": [], "figures": []}
            }
            manifest_path = exp_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
        
        results = audit_all_experiments(experiments_dir, tmp_path)
        summary = aggregate_audit_summary(results)
        
        assert summary["total_experiments"] == 3
        assert summary["passed"] + summary["failed"] + summary["skipped"] == 3
        assert 0.0 <= summary["pass_rate"] <= 1.0
    
    def test_render_aggregate_json(self, tmp_path):
        """Test JSON rendering of aggregate results."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        
        exp_dir = experiments_dir / "EXP_001"
        exp_dir.mkdir()
        manifest = {
            "experiment_id": "EXP_001",
            "artifacts": {"logs": [], "figures": []}
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest))
        
        results = audit_all_experiments(experiments_dir, tmp_path)
        output = render_aggregate_json(results)
        
        assert isinstance(output, str)
        parsed = json.loads(output)
        assert "summary" in parsed
        assert "experiments" in parsed
    
    def test_render_aggregate_markdown(self, tmp_path):
        """Test Markdown rendering of aggregate results."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        
        exp_dir = experiments_dir / "EXP_001"
        exp_dir.mkdir()
        manifest = {
            "experiment_id": "EXP_001",
            "artifacts": {"logs": [], "figures": []}
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest))
        
        results = audit_all_experiments(experiments_dir, tmp_path)
        output = render_aggregate_markdown(results)
        
        assert isinstance(output, str)
        assert "# Multi-Experiment Audit Report" in output
        assert "## Summary" in output


class TestEvidenceChainLedger:
    """Test evidence chain ledger functionality."""
    
    def test_build_evidence_chain_ledger_all_pass(self):
        """Test building ledger with all passing experiments."""
        audit_results = [
            {
                "experiment_id": "EXP_001",
                "status": "PASS",
                "manifest_path": "exp1/manifest.json",
                "manifest_hash": "hash1",
                "artifacts": [
                    {"path": "results/exp1.jsonl", "hash": "artifact_hash_1"}
                ]
            },
            {
                "experiment_id": "EXP_002",
                "status": "PASS",
                "manifest_path": "exp2/manifest.json",
                "manifest_hash": "hash2",
                "artifacts": [
                    {"path": "results/exp2.jsonl", "hash": "artifact_hash_2"}
                ]
            }
        ]
        
        ledger = build_evidence_chain_ledger(audit_results)
        
        assert ledger["schema_version"] == "1.0"
        assert ledger["experiment_count"] == 2
        assert ledger["global_status"] == "PASS"
        assert len(ledger["experiments"]) == 2
        assert len(ledger["ledger_hash"]) == 64  # SHA-256 hex length
    
    def test_build_evidence_chain_ledger_with_failures(self):
        """Test building ledger with failed experiments."""
        audit_results = [
            {
                "experiment_id": "EXP_001",
                "status": "PASS",
                "manifest_path": "exp1/manifest.json",
                "manifest_hash": "hash1",
                "artifacts": []
            },
            {
                "experiment_id": "EXP_002",
                "status": "FAIL",
                "manifest_path": "exp2/manifest.json",
                "manifest_hash": "hash2",
                "artifacts": []
            }
        ]
        
        ledger = build_evidence_chain_ledger(audit_results)
        
        assert ledger["global_status"] == "FAIL"
        assert ledger["experiment_count"] == 2
    
    def test_build_evidence_chain_ledger_deterministic(self):
        """Test that ledger hash is deterministic."""
        audit_results = [
            {
                "experiment_id": "EXP_001",
                "status": "PASS",
                "manifest_path": "exp1/manifest.json",
                "manifest_hash": "hash1",
                "artifacts": [{"path": "test.jsonl", "hash": "abc123"}]
            }
        ]
        
        ledger1 = build_evidence_chain_ledger(audit_results)
        ledger2 = build_evidence_chain_ledger(audit_results)
        
        assert ledger1["ledger_hash"] == ledger2["ledger_hash"]
    
    def test_evaluate_evidence_chain_for_ci_pass(self):
        """Test CI gate evaluation with passing status."""
        ledger = {"global_status": "PASS"}
        exit_code = evaluate_evidence_chain_for_ci(ledger)
        assert exit_code == 0
    
    def test_evaluate_evidence_chain_for_ci_partial(self):
        """Test CI gate evaluation with partial status."""
        ledger = {"global_status": "PARTIAL"}
        exit_code = evaluate_evidence_chain_for_ci(ledger)
        assert exit_code == 1
    
    def test_evaluate_evidence_chain_for_ci_fail(self):
        """Test CI gate evaluation with failed status."""
        ledger = {"global_status": "FAIL"}
        exit_code = evaluate_evidence_chain_for_ci(ledger)
        assert exit_code == 2
    
    def test_evaluate_evidence_chain_for_ci_unknown(self):
        """Test CI gate evaluation with unknown status."""
        ledger = {"global_status": "UNKNOWN"}
        exit_code = evaluate_evidence_chain_for_ci(ledger)
        assert exit_code == 2
    
    def test_render_evidence_chain_section(self):
        """Test Markdown rendering of evidence chain section."""
        ledger = {
            "schema_version": "1.0",
            "experiment_count": 2,
            "global_status": "PASS",
            "ledger_hash": "abc123def456",
            "experiments": [
                {
                    "id": "EXP_001",
                    "status": "PASS",
                    "report_path": "exp1/manifest.json",
                    "artifact_hashes": {
                        "exp1/manifest.json": "hash1",
                        "results/exp1.jsonl": "hash2"
                    }
                },
                {
                    "id": "EXP_002",
                    "status": "FAIL",
                    "report_path": "exp2/manifest.json",
                    "artifact_hashes": {}
                }
            ]
        }
        
        output = render_evidence_chain_section(ledger)
        
        assert isinstance(output, str)
        assert "## Evidence Chain" in output
        assert "EXP_001" in output
        assert "EXP_002" in output
        assert "SHA-256" in output
        assert "ledger_hash" in output.lower()
        assert "✓ PASS" in output
        assert "✗ FAIL" in output


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_audit_to_ledger_workflow(self, tmp_path):
        """Test complete workflow from experiments to evidence chain ledger."""
        # Setup experiment structure
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # Create two experiments
        for i in range(1, 3):
            exp_dir = experiments_dir / f"EXP_{i:03d}"
            exp_dir.mkdir()
            
            # Create log file
            log_file = results_dir / f"exp_{i:03d}.jsonl"
            log_file.write_text('{"cycle": 1}\n{"cycle": 2}\n')
            log_hash = compute_sha256_file(log_file)
            
            # Create manifest
            manifest = {
                "experiment_id": f"EXP_{i:03d}",
                "artifacts": {
                    "logs": [{
                        "path": f"results/exp_{i:03d}.jsonl",
                        "type": "jsonl",
                        "sha256": log_hash
                    }],
                    "figures": []
                }
            }
            (exp_dir / "manifest.json").write_text(json.dumps(manifest))
        
        # Run multi-experiment audit
        results = audit_all_experiments(experiments_dir, tmp_path)
        
        # Build evidence chain ledger
        audit_results = [
            {
                "experiment_id": r.experiment_id,
                "status": r.status,
                "manifest_path": r.manifest_path,
                "manifest_hash": r.manifest_hash,
                "artifacts": [
                    {"path": a.path, "hash": a.actual_hash}
                    for a in r.artifacts_checked
                ]
            }
            for r in results
        ]
        
        ledger = build_evidence_chain_ledger(audit_results)
        
        # Verify ledger structure
        assert ledger["experiment_count"] == 2
        assert ledger["global_status"] == "PASS"
        assert len(ledger["ledger_hash"]) == 64
        
        # Verify CI gate
        exit_code = evaluate_evidence_chain_for_ci(ledger)
        assert exit_code == 0
        
        # Verify rendering
        markdown = render_evidence_chain_section(ledger)
        assert "EXP_001" in markdown
        assert "EXP_002" in markdown
