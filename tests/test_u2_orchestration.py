# PHASE II — NOT USED IN PHASE I
# Tests for U2 orchestration and evidence feed

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_uplift_u2 import (
    build_u2_run_summary,
    summarize_u2_run_for_evidence,
)


class TestU2RunSummary:
    """Tests for build_u2_run_summary function."""
    
    def test_basic_run_summary(self):
        """Test basic run summary generation."""
        config = {"slices": {"test_slice": {"items": ["a", "b", "c"]}}}
        manifest = {
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": 50,
        }
        calibration_summary = None
        verbose_paths = {
            "baseline_jsonl": "/tmp/baseline.jsonl",
            "rfl_jsonl": "/tmp/rfl.jsonl",
            "manifest_json": "/tmp/manifest.json",
        }
        
        summary = build_u2_run_summary(
            config=config,
            manifest=manifest,
            calibration_summary=calibration_summary,
            verbose_paths=verbose_paths,
        )
        
        assert summary["schema_version"] == "1.0.0"
        assert summary["slice_name"] == "test_slice"
        assert summary["mode"] == "baseline"
        assert summary["calibration_used"] is False
        assert summary["cycles_requested"] == 50
        assert summary["cycles_completed"] == 50
        assert summary["paths"]["baseline_jsonl"] == "/tmp/baseline.jsonl"
        assert summary["paths"]["calibration_summary_json"] is None
        assert summary["label"] == "PHASE II — NOT USED IN PHASE I"
    
    def test_run_summary_with_calibration(self):
        """Test run summary with calibration enabled."""
        config = {"slices": {"test_slice": {"items": ["a", "b", "c"]}}}
        manifest = {
            "slice": "test_slice",
            "mode": "rfl",
            "cycles": 100,
        }
        calibration_summary = {
            "slice_name": "test_slice",
            "calibration_cycles": 10,
        }
        verbose_paths = {
            "baseline_jsonl": "/tmp/baseline.jsonl",
            "rfl_jsonl": "/tmp/rfl.jsonl",
            "calibration_summary_json": "/tmp/calibration.json",
            "manifest_json": "/tmp/manifest.json",
            "determinism_verified": True,
        }
        
        summary = build_u2_run_summary(
            config=config,
            manifest=manifest,
            calibration_summary=calibration_summary,
            verbose_paths=verbose_paths,
        )
        
        assert summary["calibration_used"] is True
        assert summary["paths"]["calibration_summary_json"] == "/tmp/calibration.json"
        assert summary["determinism_verified"] is True


class TestEvidenceSummary:
    """Tests for summarize_u2_run_for_evidence function."""
    
    def test_complete_run_ready_for_bootstrap(self):
        """Test evidence summary for complete run."""
        run_summary = {
            "schema_version": "1.0.0",
            "slice_name": "test_slice",
            "mode": "baseline",
            "calibration_used": False,
            "cycles_requested": 50,
            "cycles_completed": 50,
            "paths": {
                "baseline_jsonl": "/tmp/baseline.jsonl",
                "rfl_jsonl": "/tmp/rfl.jsonl",
                "manifest_json": "/tmp/manifest.json",
                "calibration_summary_json": None,
            },
            "determinism_verified": False,
        }
        
        evidence = summarize_u2_run_for_evidence(run_summary)
        
        assert evidence["schema_version"] == "1.0.0"
        assert evidence["has_all_required_artifacts"] is True
        assert evidence["calibration_ok"] is True
        assert evidence["ready_for_bootstrap"] is True
        assert "ready for statistical analysis" in evidence["notes"]
        assert evidence["label"] == "PHASE II — Evidence feed, no uplift claims"
    
    def test_missing_baseline_artifact(self):
        """Test evidence summary with missing baseline."""
        run_summary = {
            "schema_version": "1.0.0",
            "slice_name": "test_slice",
            "mode": "rfl",
            "calibration_used": False,
            "cycles_requested": 50,
            "cycles_completed": 50,
            "paths": {
                "baseline_jsonl": None,
                "rfl_jsonl": "/tmp/rfl.jsonl",
                "manifest_json": "/tmp/manifest.json",
                "calibration_summary_json": None,
            },
            "determinism_verified": False,
        }
        
        evidence = summarize_u2_run_for_evidence(run_summary)
        
        assert evidence["has_all_required_artifacts"] is False
        assert evidence["ready_for_bootstrap"] is False
        assert "baseline_jsonl" in evidence["notes"]
    
    def test_missing_calibration_when_required(self):
        """Test evidence summary when calibration required but missing."""
        run_summary = {
            "schema_version": "1.0.0",
            "slice_name": "test_slice",
            "mode": "rfl",
            "calibration_used": True,
            "cycles_requested": 50,
            "cycles_completed": 50,
            "paths": {
                "baseline_jsonl": "/tmp/baseline.jsonl",
                "rfl_jsonl": "/tmp/rfl.jsonl",
                "manifest_json": "/tmp/manifest.json",
                "calibration_summary_json": None,  # Missing!
            },
            "determinism_verified": False,
        }
        
        evidence = summarize_u2_run_for_evidence(run_summary)
        
        assert evidence["calibration_ok"] is False
        assert evidence["ready_for_bootstrap"] is False
        assert "Calibration was required" in evidence["notes"]
    
    def test_incomplete_run(self):
        """Test evidence summary for incomplete run."""
        run_summary = {
            "schema_version": "1.0.0",
            "slice_name": "test_slice",
            "mode": "baseline",
            "calibration_used": False,
            "cycles_requested": 100,
            "cycles_completed": 50,  # Incomplete!
            "paths": {
                "baseline_jsonl": "/tmp/baseline.jsonl",
                "rfl_jsonl": "/tmp/rfl.jsonl",
                "manifest_json": "/tmp/manifest.json",
                "calibration_summary_json": None,
            },
            "determinism_verified": False,
        }
        
        evidence = summarize_u2_run_for_evidence(run_summary)
        
        assert "incomplete" in evidence["notes"].lower()
    
    def test_multiple_issues(self):
        """Test evidence summary with multiple issues."""
        run_summary = {
            "schema_version": "1.0.0",
            "slice_name": "test_slice",
            "mode": "rfl",
            "calibration_used": True,
            "cycles_requested": 100,
            "cycles_completed": 50,
            "paths": {
                "baseline_jsonl": None,
                "rfl_jsonl": "/tmp/rfl.jsonl",
                "manifest_json": None,
                "calibration_summary_json": None,
            },
            "determinism_verified": False,
        }
        
        evidence = summarize_u2_run_for_evidence(run_summary)
        
        assert evidence["has_all_required_artifacts"] is False
        assert evidence["calibration_ok"] is False
        assert evidence["ready_for_bootstrap"] is False
        # Should have multiple notes
        assert "baseline_jsonl" in evidence["notes"]
        assert "manifest_json" in evidence["notes"]


class TestU2RunSummaryContract:
    """Test the complete U2 run summary contract."""
    
    def test_summary_has_all_required_fields(self):
        """Verify run summary contains all contract fields."""
        config = {"slices": {"test_slice": {"items": ["a", "b", "c"]}}}
        manifest = {
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": 50,
        }
        calibration_summary = None
        verbose_paths = {
            "baseline_jsonl": "/tmp/baseline.jsonl",
            "rfl_jsonl": "/tmp/rfl.jsonl",
            "manifest_json": "/tmp/manifest.json",
        }
        
        summary = build_u2_run_summary(
            config=config,
            manifest=manifest,
            calibration_summary=calibration_summary,
            verbose_paths=verbose_paths,
        )
        
        # Verify all required fields exist
        required_fields = [
            "schema_version",
            "slice_name",
            "mode",
            "calibration_used",
            "cycles_requested",
            "cycles_completed",
            "paths",
            "determinism_verified",
            "label",
        ]
        
        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"
        
        # Verify paths structure
        assert "baseline_jsonl" in summary["paths"]
        assert "rfl_jsonl" in summary["paths"]
        assert "calibration_summary_json" in summary["paths"]
        assert "manifest_json" in summary["paths"]


class TestEvidenceFeedContract:
    """Test the evidence feed contract."""
    
    def test_evidence_has_all_required_fields(self):
        """Verify evidence summary contains all contract fields."""
        run_summary = {
            "schema_version": "1.0.0",
            "slice_name": "test_slice",
            "mode": "baseline",
            "calibration_used": False,
            "cycles_requested": 50,
            "cycles_completed": 50,
            "paths": {
                "baseline_jsonl": "/tmp/baseline.jsonl",
                "rfl_jsonl": "/tmp/rfl.jsonl",
                "manifest_json": "/tmp/manifest.json",
                "calibration_summary_json": None,
            },
            "determinism_verified": False,
        }
        
        evidence = summarize_u2_run_for_evidence(run_summary)
        
        # Verify all required fields exist
        required_fields = [
            "schema_version",
            "has_all_required_artifacts",
            "calibration_ok",
            "ready_for_bootstrap",
            "notes",
            "label",
        ]
        
        for field in required_fields:
            assert field in evidence, f"Missing required field: {field}"
        
        # Verify field types
        assert isinstance(evidence["has_all_required_artifacts"], bool)
        assert isinstance(evidence["calibration_ok"], bool)
        assert isinstance(evidence["ready_for_bootstrap"], bool)
        assert isinstance(evidence["notes"], str)


class TestNoUpliftClaims:
    """Verify no uplift claims in outputs."""
    
    def test_run_summary_no_uplift_claims(self):
        """Verify run summary contains no uplift claims."""
        config = {"slices": {"test_slice": {"items": ["a", "b", "c"]}}}
        manifest = {
            "slice": "test_slice",
            "mode": "rfl",
            "cycles": 50,
        }
        calibration_summary = None
        verbose_paths = {
            "baseline_jsonl": "/tmp/baseline.jsonl",
            "rfl_jsonl": "/tmp/rfl.jsonl",
            "manifest_json": "/tmp/manifest.json",
        }
        
        summary = build_u2_run_summary(
            config=config,
            manifest=manifest,
            calibration_summary=calibration_summary,
            verbose_paths=verbose_paths,
        )
        
        # Convert to JSON string and check for uplift claims
        summary_json = json.dumps(summary)
        forbidden_terms = ["uplift", "improvement", "better", "superior", "outperform"]
        
        # Check for presence of Phase II label
        assert "PHASE II" in summary["label"]
        
        # These terms should not appear in summary
        for term in forbidden_terms:
            assert term.lower() not in summary_json.lower(), f"Found forbidden term: {term}"
    
    def test_evidence_summary_no_uplift_claims(self):
        """Verify evidence summary contains no uplift claims."""
        run_summary = {
            "schema_version": "1.0.0",
            "slice_name": "test_slice",
            "mode": "baseline",
            "calibration_used": False,
            "cycles_requested": 50,
            "cycles_completed": 50,
            "paths": {
                "baseline_jsonl": "/tmp/baseline.jsonl",
                "rfl_jsonl": "/tmp/rfl.jsonl",
                "manifest_json": "/tmp/manifest.json",
                "calibration_summary_json": None,
            },
            "determinism_verified": False,
        }
        
        evidence = summarize_u2_run_for_evidence(run_summary)
        
        # Check for presence of Phase II label
        assert "PHASE II" in evidence["label"]
        assert "no uplift claims" in evidence["label"]
        
        # These terms should not appear in notes (excluding the disclaimer in label)
        forbidden_terms = ["improvement", "better", "superior", "outperform"]
        notes = evidence["notes"].lower()
        
        for term in forbidden_terms:
            assert term not in notes, f"Found forbidden term in notes: {term}"
        
        # Verify no claim of statistical significance
        assert "significant" not in notes
        assert "95% ci" not in notes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
