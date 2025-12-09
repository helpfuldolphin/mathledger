
import json
import pytest
from pathlib import Path

from backend.metrics.u2_analysis import (
    U2ExperimentData,
    U2UpliftResult,
    load_u2_experiment,
    compute_uplift_metrics,
    render_u2_summary,
    compute_wilson_ci,
    bootstrap_delta,
)

@pytest.fixture
def experiment_data(tmp_path: Path) -> dict:
    """Create dummy log files and return paths."""
    baseline_log = tmp_path / "baseline.jsonl"
    rfl_log = tmp_path / "rfl.jsonl"

    baseline_data = [
        {"slice": "prop_depth4", "cycle": i, "success": True, "duration_seconds": 1.2, "abstention_count": 0}
        for i in range(100)
    ]
    rfl_data = [
        {"slice": "prop_depth4", "cycle": i, "success": True, "duration_seconds": 0.9, "abstention_count": 0}
        for i in range(100)
    ]
    
    with open(baseline_log, "w") as f:
        for record in baseline_data:
            f.write(json.dumps(record) + "\n")
            
    with open(rfl_log, "w") as f:
        for record in rfl_data:
            f.write(json.dumps(record) + "\n")

    return {"baseline": str(baseline_log), "rfl": str(rfl_log)}


def test_load_u2_experiment(experiment_data: dict):
    data = load_u2_experiment(experiment_data["baseline"], experiment_data["rfl"])
    assert isinstance(data, U2ExperimentData)
    assert data.slice_id == "prop_depth4"
    assert len(data.baseline_records) == 100
    assert len(data.rfl_records) == 100
    assert data.baseline_records[0]["duration_seconds"] == 1.2
    assert data.rfl_records[0]["duration_seconds"] == 0.9


def test_compute_wilson_ci():
    # Test case from fo_analytics.py
    ci_low, ci_high = compute_wilson_ci(80, 100, confidence=0.95)
    assert ci_low == pytest.approx(0.711, abs=1e-2)
    assert ci_high == pytest.approx(0.869, abs=1e-2)

    # Edge case: 0 successes
    ci_low, ci_high = compute_wilson_ci(0, 100)
    assert ci_low == pytest.approx(0.0)
    assert ci_high > 0.0

    # Edge case: 100 successes
    ci_low, ci_high = compute_wilson_ci(100, 100)
    assert ci_low < 1.0
    assert ci_high == 1.0


def test_bootstrap_delta_deterministic():
    baseline = [1.2, 1.4, 1.1, 1.3, 1.5]
    rfl = [0.9, 1.0, 0.8, 1.1, 0.95]
    
    result1 = bootstrap_delta(baseline, rfl, seed=42)
    result2 = bootstrap_delta(baseline, rfl, seed=42)
    
    assert result1 == result2

    result3 = bootstrap_delta(baseline, rfl, seed=123)
    assert result1 != result3


def test_compute_uplift_metrics(experiment_data: dict):
    data = load_u2_experiment(experiment_data["baseline"], experiment_data["rfl"])
    result = compute_uplift_metrics(data, bootstrap_seed=42)

    assert isinstance(result, U2UpliftResult)
    assert result.slice_id == "prop_depth4"
    assert result.n_baseline == 100
    assert result.n_rfl == 100
    assert result.passes_governance is False # Should fail on min_samples
    assert result.governance_details["sample_size_passed"] is False

    # Check throughput uplift
    assert "throughput" in result.metrics
    assert result.metrics["throughput"]["delta_pct"] > 0
    assert result.metrics["throughput"]["significant"]


def test_render_u2_summary(experiment_data: dict):
    data = load_u2_experiment(experiment_data["baseline"], experiment_data["rfl"])
    result = compute_uplift_metrics(data, bootstrap_seed=42)
    summary = render_u2_summary(result)

    assert isinstance(summary, dict)
    assert summary["slice_id"] == "prop_depth4"
    assert summary["governance"]["passed"] is False
    assert "throughput" in summary["metrics"]

