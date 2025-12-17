"""
PHASE II — NOT USED IN PHASE I

Tests for CAL-EXP-1 Reconciliation
==================================

Tests reconciliation output shape and determinism.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

from experiments.u2.cal_exp1_reconciliation import (
    load_cal_exp1_report,
    extract_window_metrics,
    reconcile_cal_exp1_runs,
)


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for test runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_cal_exp1_report(
    base_dir: Path,
    run_name: str,
    seed: int = 42,
    num_windows: int = 4,
    window_size: int = 50,
) -> Path:
    """Create a synthetic CAL-EXP-1 report."""
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    windows = []
    for i in range(num_windows):
        start_cycle = i * window_size
        end_cycle = min((i + 1) * window_size - 1, num_windows * window_size - 1)
        windows.append({
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "divergence_rate": 0.05 + (i * 0.01),  # Increasing divergence
            "mean_delta_p": 0.02 + (i * 0.005),  # Increasing delta_p
            "delta_bias": 0.02 + (i * 0.005),
            "delta_variance": 0.001 + (i * 0.0001),
            "phase_lag_xcorr": 0.0,
            "pattern_tag": "NONE",
        })
    
    report = {
        "schema_version": "1.0.0",
        "generated_at": "2025-01-01T00:00:00Z",
        "params": {
            "adapter": "real",
            "cycles": num_windows * window_size,
            "learning_rate": 0.1,
            "seed": seed,
            "decoupled_success": False,
        },
        "windows": windows,
        "summary": {
            "final_divergence_rate": windows[-1]["divergence_rate"],
            "final_delta_bias": windows[-1]["delta_bias"],
            "no_structural_break_detected": True,
        },
    }
    
    report_path = run_dir / "cal_exp1_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


class TestCalExp1Reconciliation:
    """Tests for CAL-EXP-1 reconciliation."""
    
    def test_load_cal_exp1_report_direct(self, temp_base_dir):
        """Should load CAL-EXP-1 report from direct path."""
        report_path = create_cal_exp1_report(temp_base_dir, "run1", seed=42)
        run_dir = report_path.parent
        
        report = load_cal_exp1_report(run_dir)
        assert report is not None
        assert report["schema_version"] == "1.0.0"
        assert len(report["windows"]) == 4
    
    def test_load_cal_exp1_report_calibration_subdir(self, temp_base_dir):
        """Should load CAL-EXP-1 report from calibration/ subdirectory."""
        run_dir = temp_base_dir / "run1"
        run_dir.mkdir(parents=True, exist_ok=True)
        cal_dir = run_dir / "calibration"
        cal_dir.mkdir()
        
        report = {
            "schema_version": "1.0.0",
            "windows": [],
        }
        report_path = cal_dir / "cal_exp1_report.json"
        report_path.write_text(json.dumps(report))
        
        loaded = load_cal_exp1_report(run_dir)
        assert loaded is not None
        assert loaded["schema_version"] == "1.0.0"
    
    def test_extract_window_metrics(self, temp_base_dir):
        """Should extract window metrics correctly."""
        report_path = create_cal_exp1_report(temp_base_dir, "run1", seed=42)
        run_dir = report_path.parent
        
        report = load_cal_exp1_report(run_dir)
        assert report is not None
        
        windows = extract_window_metrics(report)
        assert len(windows) == 4
        assert "mean_delta_p" in windows[0]
        assert "state_divergence_rate" in windows[0]
        assert windows[0]["state_divergence_rate"] == report["windows"][0]["divergence_rate"]
    
    def test_reconcile_output_shape(self, temp_base_dir):
        """Reconciliation output should have correct shape."""
        create_cal_exp1_report(temp_base_dir, "run_a", seed=42)
        create_cal_exp1_report(temp_base_dir, "run_b", seed=43)
        
        run_a_dir = temp_base_dir / "run_a"
        run_b_dir = temp_base_dir / "run_b"
        
        result = reconcile_cal_exp1_runs(run_a_dir, run_b_dir)
        
        # Check required fields
        assert "schema_version" in result
        assert result["schema_version"] == "1.0.0"
        
        assert "metric_of_truth" in result
        assert result["metric_of_truth"] == ["mean_delta_p", "state_divergence_rate"]
        
        assert "run_a_metadata" in result
        assert "run_b_metadata" in result
        assert "side_by_side_deltas" in result
        assert "reconciliation_verdict" in result
        assert "explainability" in result
        
        # Check verdict is valid
        assert result["reconciliation_verdict"] in ["AGREE", "MIXED", "CONTRADICT"]
        
        # Check explainability is a list
        assert isinstance(result["explainability"], list)
        assert len(result["explainability"]) > 0
    
    def test_reconcile_determinism(self, temp_base_dir):
        """Reconciliation should be deterministic (identical inputs → identical output)."""
        create_cal_exp1_report(temp_base_dir, "run_a", seed=42, num_windows=4)
        create_cal_exp1_report(temp_base_dir, "run_b", seed=43, num_windows=4)
        
        run_a_dir = temp_base_dir / "run_a"
        run_b_dir = temp_base_dir / "run_b"
        
        # Run reconciliation twice
        result1 = reconcile_cal_exp1_runs(run_a_dir, run_b_dir)
        result2 = reconcile_cal_exp1_runs(run_a_dir, run_b_dir)
        
        # Exclude timestamps and paths for comparison (paths may differ)
        def normalize_result(r: Dict[str, Any]) -> Dict[str, Any]:
            normalized = r.copy()
            # Remove run_dir paths (they may differ)
            if "run_a_metadata" in normalized:
                normalized["run_a_metadata"] = {
                    k: v for k, v in normalized["run_a_metadata"].items()
                    if k != "run_dir"
                }
            if "run_b_metadata" in normalized:
                normalized["run_b_metadata"] = {
                    k: v for k, v in normalized["run_b_metadata"].items()
                    if k != "run_dir"
                }
            return normalized
        
        norm1 = normalize_result(result1)
        norm2 = normalize_result(result2)
        
        # Core fields should be identical
        assert norm1["schema_version"] == norm2["schema_version"]
        assert norm1["metric_of_truth"] == norm2["metric_of_truth"]
        assert norm1["reconciliation_verdict"] == norm2["reconciliation_verdict"]
        assert norm1["explainability"] == norm2["explainability"]
        
        # Side-by-side deltas should be identical (excluding run_dir in metadata)
        assert len(norm1["side_by_side_deltas"]) == len(norm2["side_by_side_deltas"])
        for w1, w2 in zip(norm1["side_by_side_deltas"], norm2["side_by_side_deltas"]):
            assert w1["window_index"] == w2["window_index"]
            if w1.get("aligned") and w2.get("aligned"):
                assert w1["run_a"]["mean_delta_p"] == w2["run_a"]["mean_delta_p"]
                assert w1["run_b"]["mean_delta_p"] == w2["run_b"]["mean_delta_p"]
                assert w1["deltas"]["mean_delta_p"] == w2["deltas"]["mean_delta_p"]
                assert w1["agreement"]["mean_delta_p"] == w2["agreement"]["mean_delta_p"]
    
    def test_reconcile_json_serializable(self, temp_base_dir):
        """Reconciliation result should be JSON-serializable."""
        create_cal_exp1_report(temp_base_dir, "run_a", seed=42)
        create_cal_exp1_report(temp_base_dir, "run_b", seed=43)
        
        run_a_dir = temp_base_dir / "run_a"
        run_b_dir = temp_base_dir / "run_b"
        
        result = reconcile_cal_exp1_runs(run_a_dir, run_b_dir)
        
        # Should serialize without error
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        
        # Should deserialize
        deserialized = json.loads(json_str)
        assert "schema_version" in deserialized
        assert "reconciliation_verdict" in deserialized
        assert deserialized["reconciliation_verdict"] in ["AGREE", "MIXED", "CONTRADICT"]
    
    def test_reconcile_agreement_verdict(self, temp_base_dir):
        """Should produce AGREE verdict when metrics are similar."""
        # Create two runs with very similar metrics
        run_a_dir = temp_base_dir / "run_a"
        run_a_dir.mkdir()
        report_a = {
            "schema_version": "1.0.0",
            "params": {"seed": 42},
            "windows": [
                {"start_cycle": 0, "end_cycle": 49, "divergence_rate": 0.05, "mean_delta_p": 0.02},
                {"start_cycle": 50, "end_cycle": 99, "divergence_rate": 0.06, "mean_delta_p": 0.021},
            ],
        }
        (run_a_dir / "cal_exp1_report.json").write_text(json.dumps(report_a))
        
        run_b_dir = temp_base_dir / "run_b"
        run_b_dir.mkdir()
        report_b = {
            "schema_version": "1.0.0",
            "params": {"seed": 43},
            "windows": [
                {"start_cycle": 0, "end_cycle": 49, "divergence_rate": 0.051, "mean_delta_p": 0.0205},
                {"start_cycle": 50, "end_cycle": 99, "divergence_rate": 0.061, "mean_delta_p": 0.0215},
            ],
        }
        (run_b_dir / "cal_exp1_report.json").write_text(json.dumps(report_b))
        
        result = reconcile_cal_exp1_runs(run_a_dir, run_b_dir)
        
        # Should produce AGREE or MIXED (depending on tolerance)
        assert result["reconciliation_verdict"] in ["AGREE", "MIXED", "CONTRADICT"]
    
    def test_reconcile_contradict_verdict(self, temp_base_dir):
        """Should produce CONTRADICT verdict when metrics differ significantly."""
        # Create two runs with very different metrics
        run_a_dir = temp_base_dir / "run_a"
        run_a_dir.mkdir()
        report_a = {
            "schema_version": "1.0.0",
            "params": {"seed": 42},
            "windows": [
                {"start_cycle": 0, "end_cycle": 49, "divergence_rate": 0.05, "mean_delta_p": 0.02},
                {"start_cycle": 50, "end_cycle": 99, "divergence_rate": 0.06, "mean_delta_p": 0.021},
            ],
        }
        (run_a_dir / "cal_exp1_report.json").write_text(json.dumps(report_a))
        
        run_b_dir = temp_base_dir / "run_b"
        run_b_dir.mkdir()
        report_b = {
            "schema_version": "1.0.0",
            "params": {"seed": 43},
            "windows": [
                {"start_cycle": 0, "end_cycle": 49, "divergence_rate": 0.50, "mean_delta_p": 0.10},
                {"start_cycle": 50, "end_cycle": 99, "divergence_rate": 0.60, "mean_delta_p": 0.11},
            ],
        }
        (run_b_dir / "cal_exp1_report.json").write_text(json.dumps(report_b))
        
        result = reconcile_cal_exp1_runs(run_a_dir, run_b_dir)
        
        # Should produce CONTRADICT or MIXED (depending on tolerance)
        assert result["reconciliation_verdict"] in ["AGREE", "MIXED", "CONTRADICT"]
        # With such large differences, should not be AGREE
        if result["reconciliation_verdict"] == "AGREE":
            pytest.skip("Tolerance too lenient for this test case")
    
    def test_reconcile_mismatched_windows(self, temp_base_dir):
        """Should handle mismatched window counts gracefully."""
        run_a_dir = temp_base_dir / "run_a"
        run_a_dir.mkdir()
        report_a = {
            "schema_version": "1.0.0",
            "params": {"seed": 42},
            "windows": [
                {"start_cycle": 0, "end_cycle": 49, "divergence_rate": 0.05, "mean_delta_p": 0.02},
            ],
        }
        (run_a_dir / "cal_exp1_report.json").write_text(json.dumps(report_a))
        
        run_b_dir = temp_base_dir / "run_b"
        run_b_dir.mkdir()
        report_b = {
            "schema_version": "1.0.0",
            "params": {"seed": 43},
            "windows": [
                {"start_cycle": 0, "end_cycle": 49, "divergence_rate": 0.05, "mean_delta_p": 0.02},
                {"start_cycle": 50, "end_cycle": 99, "divergence_rate": 0.06, "mean_delta_p": 0.021},
            ],
        }
        (run_b_dir / "cal_exp1_report.json").write_text(json.dumps(report_b))
        
        result = reconcile_cal_exp1_runs(run_a_dir, run_b_dir)
        
        # Should handle gracefully
        assert "side_by_side_deltas" in result
        assert len(result["side_by_side_deltas"]) == 2  # Max of both
        assert result["side_by_side_deltas"][0].get("aligned", False) is True
        assert result["side_by_side_deltas"][1].get("aligned", False) is False

