import json
from pathlib import Path
import pytest
import sys
import argparse
import hashlib

# Add project root to path to resolve imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.run_uplift_governance_pipeline import run_pipeline

#region Fixtures and Test Data

def get_base_summary_data() -> dict:
    """Returns a base summary that can be modified for tests."""
    # This structure must be kept in sync with the verifier's expectations
    return {
        "experiment_id": "u2-test-run",
        "reproducibility": {"bootstrap_seed": 42, "n_bootstrap": 10000, "confidence": 0.95},
        "governance": {"recommendation": "PROCEED", "all_slices_pass": True, "passing_slices": ["prop_depth4"], "failing_slices": []},
        "slices": {
            "prop_depth4": {
                "sample_size": {"baseline": 500, "rfl": 500},
                "metrics": {"success_rate": {"rfl": 0.99}, "abstention_rate": {"rfl": 0.01}, "throughput": {"delta_pct": 10.0, "delta_ci_low": 6.0, "delta_ci_high": 14.0}}
            }
        }
    }

def get_base_manifest_data(log_files: dict) -> dict:
    """Returns a base manifest that can be modified for tests."""
    return {
        "experiment_id": "u2-test-run",
        "prereg_ref": "PREREG_UPLIFT_U2.yaml",
        "config": {"seed_baseline": 1, "seed_rfl": 2, "slices": {"prop_depth4": {"derivation_params": {"steps": 1, "depth": 1, "breadth": 1, "total": 1}}}},
        "artifacts": {
            "baseline_logs": [log_files["baseline"].name],
            "rfl_logs": [log_files["rfl"].name],
        },
        "checksums": {
            log_files["baseline"].name: hashlib.sha256(log_files["baseline"].read_bytes()).hexdigest(),
            log_files["rfl"].name: hashlib.sha256(log_files["rfl"].read_bytes()).hexdigest(),
        }
    }

@pytest.fixture
def artifact_paths(tmp_path: Path) -> dict:
    """Creates a full set of dummy artifacts for a test run."""
    paths = {
        "summary": tmp_path / "summary.json",
        "baseline_log": tmp_path / "baseline.jsonl",
        "rfl_log": tmp_path / "rfl.jsonl",
        "manifest": tmp_path / "manifest.json",
        "telemetry": tmp_path / "telemetry.json",
        "prereg": tmp_path / "PREREG_UPLIFT_U2.yaml",
        "output": tmp_path / "report.json",
    }
    
    # Create dummy log files
    log_record = {"cycle": 1, "metrics": {"abstention_rate": 0.1}, "policy": {"theta": [0.0]}}
    paths["baseline_log"].write_text(json.dumps(log_record))
    paths["rfl_log"].write_text(json.dumps(log_record))

    # Create other dummy files
    paths["telemetry"].write_text(json.dumps({"cycles":[]}))
    paths["prereg"].write_text("confidence_level: 0.95")
    
    return paths

#endregion

def test_pipeline_pass_path(artifact_paths: dict):
    """Tests the full pipeline with a set of fully valid and consistent artifacts."""
    # Create valid summary and manifest
    summary_data = get_base_summary_data()
    # To pass GOV-9 and GOV-4, all slices must be present and accounted for
    for slice_id in ["fol_eq_group", "fol_eq_ring", "linear_arith"]:
        summary_data["slices"][slice_id] = summary_data["slices"]["prop_depth4"]
    summary_data["governance"]["passing_slices"] = list(summary_data["slices"].keys())
    artifact_paths["summary"].write_text(json.dumps(summary_data))
    
    log_files = {"baseline": artifact_paths["baseline_log"], "rfl": artifact_paths["rfl_log"]}
    manifest_data = get_base_manifest_data(log_files)
    # To pass MAN-4, all slice configs must be present
    for slice_id in ["fol_eq_group", "fol_eq_ring", "linear_arith"]:
        manifest_data["config"]["slices"][slice_id] = manifest_data["config"]["slices"]["prop_depth4"]
    artifact_paths["manifest"].write_text(json.dumps(manifest_data))

    args = argparse.Namespace(
        summary_path=str(artifact_paths["summary"]),
        baseline_log_path=str(artifact_paths["baseline_log"]),
        rfl_log_path=str(artifact_paths["rfl_log"]),
        manifest_path=str(artifact_paths["manifest"]),
        telemetry_path=str(artifact_paths["telemetry"]),
        prereg_path=str(artifact_paths["prereg"]),
        output_path=str(artifact_paths["output"]),
    )

    is_admissible = run_pipeline(args)

    assert is_admissible is True, "Pipeline should be admissible for a passing report"
    report = json.loads(artifact_paths["output"].read_text())
    
    assert report["status"] == "PASS"
    assert report["governance_verdict"]["status"] == "PASS"
    assert not report["governance_verdict"]["invalidating_rules"]
    assert report["conjecture_report"] is not None

def test_pipeline_fail_path_manifest_checksum(artifact_paths: dict):
    """Tests the hard gate with a manifest that has a bad checksum (MAN-6)."""
    summary_data = get_base_summary_data()
    artifact_paths["summary"].write_text(json.dumps(summary_data))
    
    log_files = {"baseline": artifact_paths["baseline_log"], "rfl": artifact_paths["rfl_log"]}
    manifest_data = get_base_manifest_data(log_files)
    manifest_data["checksums"][log_files["baseline"].name] = "bad_hash" # Tamper with the hash
    artifact_paths["manifest"].write_text(json.dumps(manifest_data))

    args = argparse.Namespace(
        summary_path=str(artifact_paths["summary"]),
        baseline_log_path=str(artifact_paths["baseline_log"]),
        rfl_log_path=str(artifact_paths["rfl_log"]),
        manifest_path=str(artifact_paths["manifest"]),
        telemetry_path=None, prereg_path=None, # Test without optional files
        output_path=str(artifact_paths["output"]),
    )

    is_admissible = run_pipeline(args)

    assert is_admissible is False, "Pipeline should fail on a checksum mismatch"
    report = json.loads(artifact_paths["output"].read_text())
    
    assert report["status"] == "FAILED"
    assert report["governance_verdict"]["status"] == "FAIL"
    assert "MAN-6" in report["governance_verdict"]["invalidating_rules"]
    assert report["conjecture_report"] is None
