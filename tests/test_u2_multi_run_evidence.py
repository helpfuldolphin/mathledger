"""
PHASE II — NOT USED IN PHASE I

Tests for U2 Multi-Run Evidence Fusion

Verifies:
- Fusion logic correctness
- Ordering detection
- Determinism agreement
- Missing artifact detection
- Exit code correctness
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

from experiments.u2.evidence_fusion import (
    fuse_evidence_summaries,
    check_determinism_violations,
    check_missing_artifacts,
    check_conflicting_slice_names,
    check_run_ordering_anomalies,
    check_rfl_policy_completeness,
    inject_multi_run_fusion_into_evidence,
    validate_run_summary,
    FusedEvidenceSummary,
)
from experiments.u2.cli import promotion_precheck


def create_minimal_run_summary(
    slice_name: str = "SLICE_A",
    mode: str = "baseline",
    cycles: int = 100,
    seed: int = 42,
    ht_series_hash: str = "abc123",
    slice_config_hash: str = "def456",
    results_path: str = "/tmp/results.jsonl",
    manifest_path: str = "/tmp/manifest.json",
) -> Dict[str, Any]:
    """Create a minimal valid run summary for testing."""
    summary = {
        "label": "PHASE II — NOT USED IN PHASE I",
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": slice_config_hash,
        "prereg_hash": "N/A",
        "ht_series_hash": ht_series_hash,
        "ht_series_length": cycles,
        "outputs": {
            "results": results_path,
            "manifest": manifest_path,
        },
    }
    
    if mode == "rfl":
        summary["policy_stats"] = {
            "update_count": 10,
            "success_count": {"item1": 5, "item2": 3},
            "attempt_count": {"item1": 8, "item2": 5},
        }
    
    return summary


class TestValidateRunSummary:
    """Test run summary validation."""
    
    def test_valid_summary(self):
        """Valid summary has no errors."""
        summary = create_minimal_run_summary()
        errors = validate_run_summary(summary)
        assert errors == []
    
    def test_missing_required_field(self):
        """Missing required field is detected."""
        summary = create_minimal_run_summary()
        del summary["slice"]
        errors = validate_run_summary(summary)
        assert any("slice" in e for e in errors)
    
    def test_missing_outputs_section(self):
        """Missing outputs section is detected."""
        summary = create_minimal_run_summary()
        del summary["outputs"]
        errors = validate_run_summary(summary)
        assert any("outputs" in e for e in errors)
    
    def test_missing_hash_fields(self):
        """Missing hash fields are detected."""
        summary = create_minimal_run_summary()
        del summary["slice_config_hash"]
        del summary["ht_series_hash"]
        errors = validate_run_summary(summary)
        assert any("slice_config_hash" in e for e in errors)
        assert any("ht_series_hash" in e for e in errors)


class TestDeterminismViolations:
    """Test determinism violation detection."""
    
    def test_no_violations_single_run(self):
        """Single run has no violations."""
        summaries = [create_minimal_run_summary()]
        violations = check_determinism_violations(summaries)
        assert len(violations) == 0
    
    def test_no_violations_matching_hashes(self):
        """Matching hashes have no violations."""
        summaries = [
            create_minimal_run_summary(ht_series_hash="abc123"),
            create_minimal_run_summary(ht_series_hash="abc123"),
        ]
        violations = check_determinism_violations(summaries)
        assert len(violations) == 0
    
    def test_violation_different_hashes(self):
        """Different hashes trigger violation."""
        summaries = [
            create_minimal_run_summary(ht_series_hash="abc123"),
            create_minimal_run_summary(ht_series_hash="xyz789"),
        ]
        violations = check_determinism_violations(summaries)
        assert len(violations) == 1
        assert "abc123" in violations[0].trace_hashes
        assert "xyz789" in violations[0].trace_hashes
    
    def test_no_violation_different_seeds(self):
        """Different seeds don't trigger violation."""
        summaries = [
            create_minimal_run_summary(seed=42, ht_series_hash="abc123"),
            create_minimal_run_summary(seed=99, ht_series_hash="xyz789"),
        ]
        violations = check_determinism_violations(summaries)
        assert len(violations) == 0


class TestMissingArtifacts:
    """Test missing artifact detection."""
    
    def test_no_missing_with_temp_files(self):
        """No missing artifacts when files exist."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as results_file:
            with tempfile.NamedTemporaryFile(suffix=".json") as manifest_file:
                summaries = [
                    create_minimal_run_summary(
                        results_path=results_file.name,
                        manifest_path=manifest_file.name,
                    )
                ]
                missing = check_missing_artifacts(summaries)
                assert len(missing) == 0
    
    def test_missing_results_file(self):
        """Missing results file is detected."""
        summaries = [
            create_minimal_run_summary(
                results_path="/nonexistent/results.jsonl",
                manifest_path="/nonexistent/manifest.json",
            )
        ]
        missing = check_missing_artifacts(summaries)
        assert len(missing) >= 1
        assert any(a.artifact_type == "results" for a in missing)


class TestConflictingSliceNames:
    """Test conflicting slice name detection."""
    
    def test_no_conflicts_same_slice(self):
        """Same slice names have no conflicts."""
        summaries = [
            create_minimal_run_summary(slice_name="SLICE_A"),
            create_minimal_run_summary(slice_name="SLICE_A"),
        ]
        conflicts = check_conflicting_slice_names(summaries)
        assert len(conflicts) == 0
    
    def test_no_conflicts_different_configs(self):
        """Different configs with different slices have no conflicts."""
        summaries = [
            create_minimal_run_summary(
                slice_name="SLICE_A",
                slice_config_hash="hash1",
            ),
            create_minimal_run_summary(
                slice_name="SLICE_B",
                slice_config_hash="hash2",
            ),
        ]
        conflicts = check_conflicting_slice_names(summaries)
        assert len(conflicts) == 0
    
    def test_conflict_same_config_different_slice(self):
        """Same config with different slices triggers conflict."""
        summaries = [
            create_minimal_run_summary(
                slice_name="SLICE_A",
                slice_config_hash="same_hash",
            ),
            create_minimal_run_summary(
                slice_name="SLICE_B",
                slice_config_hash="same_hash",
            ),
        ]
        conflicts = check_conflicting_slice_names(summaries)
        assert len(conflicts) == 1
        assert "SLICE_A" in conflicts[0].slice_names
        assert "SLICE_B" in conflicts[0].slice_names


class TestRunOrderingAnomalies:
    """Test run ordering anomaly detection."""
    
    def test_no_anomalies_baseline_and_rfl(self):
        """Baseline and RFL runs have no anomalies."""
        summaries = [
            create_minimal_run_summary(slice_name="SLICE_A", mode="baseline"),
            create_minimal_run_summary(slice_name="SLICE_A", mode="rfl"),
        ]
        anomalies = check_run_ordering_anomalies(summaries)
        assert len(anomalies) == 0
    
    def test_anomaly_rfl_without_baseline(self):
        """RFL without baseline triggers anomaly."""
        summaries = [
            create_minimal_run_summary(slice_name="SLICE_A", mode="rfl"),
        ]
        anomalies = check_run_ordering_anomalies(summaries)
        assert len(anomalies) == 1
        assert "SLICE_A" in anomalies[0].description
        assert "no baseline" in anomalies[0].description.lower()
    
    def test_no_anomaly_baseline_only(self):
        """Baseline only has no anomalies."""
        summaries = [
            create_minimal_run_summary(slice_name="SLICE_A", mode="baseline"),
        ]
        anomalies = check_run_ordering_anomalies(summaries)
        assert len(anomalies) == 0


class TestRFLPolicyCompleteness:
    """Test RFL policy completeness check."""
    
    def test_complete_baseline_only(self):
        """Baseline runs are always complete."""
        summaries = [
            create_minimal_run_summary(mode="baseline"),
        ]
        complete = check_rfl_policy_completeness(summaries)
        assert complete is True
    
    def test_complete_rfl_with_stats(self):
        """RFL with policy_stats is complete."""
        summaries = [
            create_minimal_run_summary(mode="rfl"),
        ]
        complete = check_rfl_policy_completeness(summaries)
        assert complete is True
    
    def test_incomplete_rfl_missing_stats(self):
        """RFL without policy_stats is incomplete."""
        summary = create_minimal_run_summary(mode="rfl")
        del summary["policy_stats"]
        summaries = [summary]
        complete = check_rfl_policy_completeness(summaries)
        assert complete is False
    
    def test_incomplete_rfl_missing_field(self):
        """RFL with incomplete policy_stats is incomplete."""
        summary = create_minimal_run_summary(mode="rfl")
        del summary["policy_stats"]["update_count"]
        summaries = [summary]
        complete = check_rfl_policy_completeness(summaries)
        assert complete is False


class TestFuseEvidenceSummaries:
    """Test evidence fusion logic."""
    
    def test_empty_runs_blocks(self):
        """Empty runs list blocks."""
        fused = fuse_evidence_summaries([])
        assert fused.pass_status == "BLOCK"
        assert fused.run_count == 0
    
    def test_valid_single_run_passes(self):
        """Valid single run passes."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as results_file:
            with tempfile.NamedTemporaryFile(suffix=".json") as manifest_file:
                summaries = [
                    create_minimal_run_summary(
                        results_path=results_file.name,
                        manifest_path=manifest_file.name,
                    )
                ]
                fused = fuse_evidence_summaries(summaries)
                assert fused.pass_status == "PASS"
                assert fused.run_count == 1
    
    def test_determinism_violation_blocks(self):
        """Determinism violation blocks."""
        summaries = [
            create_minimal_run_summary(ht_series_hash="abc123"),
            create_minimal_run_summary(ht_series_hash="xyz789"),
        ]
        fused = fuse_evidence_summaries(summaries)
        assert fused.pass_status == "BLOCK"
        assert len(fused.determinism_violations) == 1
    
    def test_missing_artifact_blocks(self):
        """Missing artifact blocks."""
        summaries = [
            create_minimal_run_summary(
                results_path="/nonexistent/results.jsonl",
            )
        ]
        fused = fuse_evidence_summaries(summaries)
        assert fused.pass_status == "BLOCK"
        assert len(fused.missing_artifacts) >= 1
    
    def test_ordering_anomaly_warns(self):
        """Ordering anomaly warns."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as results_file:
            with tempfile.NamedTemporaryFile(suffix=".json") as manifest_file:
                summaries = [
                    create_minimal_run_summary(
                        mode="rfl",
                        results_path=results_file.name,
                        manifest_path=manifest_file.name,
                    )
                ]
                fused = fuse_evidence_summaries(summaries)
                assert fused.pass_status == "WARN"
                assert len(fused.run_ordering_anomalies) == 1
    
    def test_to_dict_serializable(self):
        """to_dict produces serializable output."""
        summaries = [create_minimal_run_summary()]
        fused = fuse_evidence_summaries(summaries)
        data = fused.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(data)
        assert json_str is not None
        
        # Check structure
        assert "label" in data
        assert "run_count" in data
        assert "pass_status" in data


class TestInjectMultiRunFusion:
    """Test evidence injection."""
    
    def test_inject_adds_fusion_data(self):
        """Injection adds fusion data to evidence."""
        summary = {"existing": "data"}
        fused = FusedEvidenceSummary(
            run_count=2,
            pass_status="PASS",
        )
        
        combined = inject_multi_run_fusion_into_evidence(summary, fused)
        
        assert "multi_run_fusion" in combined
        assert combined["multi_run_fusion"]["run_count"] == 2
        assert combined["multi_run_fusion"]["pass_status"] == "PASS"
        assert combined["label"] == "PHASE II — NOT USED IN PHASE I"
    
    def test_inject_preserves_existing_data(self):
        """Injection preserves existing data."""
        summary = {"existing": "data", "more": "stuff"}
        fused = FusedEvidenceSummary(run_count=1, pass_status="PASS")
        
        combined = inject_multi_run_fusion_into_evidence(summary, fused)
        
        assert combined["existing"] == "data"
        assert combined["more"] == "stuff"


class TestPromotionPrecheckCLI:
    """Test promotion precheck CLI."""
    
    def test_pass_exit_code(self):
        """PASS returns exit code 0."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as results_file:
                summary = create_minimal_run_summary(
                    results_path=results_file.name,
                    manifest_path=f.name,
                )
                json.dump(summary, f)
                f.flush()
                
                try:
                    exit_code = promotion_precheck([f.name])
                    assert exit_code == 0
                finally:
                    Path(f.name).unlink(missing_ok=True)
                    Path(results_file.name).unlink(missing_ok=True)
    
    def test_warn_exit_code(self):
        """WARN returns exit code 1."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as results_file:
                summary = create_minimal_run_summary(
                    mode="rfl",
                    results_path=results_file.name,
                    manifest_path=f.name,
                )
                json.dump(summary, f)
                f.flush()
                
                try:
                    exit_code = promotion_precheck([f.name])
                    assert exit_code == 1
                finally:
                    Path(f.name).unlink(missing_ok=True)
                    Path(results_file.name).unlink(missing_ok=True)
    
    def test_block_exit_code(self):
        """BLOCK returns exit code 2."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            summary = create_minimal_run_summary(
                results_path="/nonexistent/results.jsonl",
            )
            json.dump(summary, f)
            f.flush()
            
            try:
                exit_code = promotion_precheck([f.name])
                assert exit_code == 2
            finally:
                Path(f.name).unlink(missing_ok=True)
    
    def test_nonexistent_manifest_blocks(self):
        """Nonexistent manifest returns exit code 2."""
        exit_code = promotion_precheck(["/nonexistent/manifest.json"])
        assert exit_code == 2
