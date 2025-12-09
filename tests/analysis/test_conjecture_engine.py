
import json
import pytest
from pathlib import Path

from analysis.conjecture_engine import run_conjecture_engine, BehaviorClassification

# Helper function from the other test, adapted for this suite
def get_base_summary_data() -> dict:
    return {
        "slice_id": "prop_depth4",
        "sample_size": {"baseline": 500, "rfl": 500},
        "metrics": {
            "success_rate": {"baseline": 0.95, "rfl": 0.98, "delta": 0.03, "ci": [0.01, 0.05]},
            "abstention_rate": {"baseline": 0.02, "rfl": 0.01, "delta": -0.01, "ci": [-0.02, 0.0]},
            "throughput": {
                "baseline_stat": 10.0,
                "treatment_stat": 12.0,
                "delta": 2.0,
                "delta_ci_low": 5.5,
                "delta_ci_high": 6.5,
                "delta_pct": 20.0,
                "significant": True,
            },
        },
        "governance": {
            "passed": True,
            "details": {
                "sample_size_passed": True,
                "success_rate_passed": True,
                "abstention_rate_passed": True,
                "throughput_uplift_passed": True,
            },
        },
        "reproducibility": {"bootstrap_seed": 42, "n_bootstrap": 10000},
    }

@pytest.fixture
def summary_file(tmp_path: Path) -> Path:
    summary_path = tmp_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(get_base_summary_data(), f)
    return summary_path


def test_strong_uplift(summary_file: Path):
    report = run_conjecture_engine(str(summary_file))
    assert report["behavior_classification"] == BehaviorClassification.STRONG_UPLIFT.name
    assert "more efficient and effective" in report["derived_conjecture"]

def test_weak_uplift(tmp_path: Path):
    data = get_base_summary_data()
    data["governance"]["passed"] = False # Fails governance but still has uplift
    summary_path = tmp_path / "summary.json"
    with open(summary_path, "w") as f: json.dump(data, f)
    
    report = run_conjecture_engine(str(summary_path))
    assert report["behavior_classification"] == BehaviorClassification.WEAK_UPLIFT.name
    assert "marginally better" in report["derived_conjecture"]

def test_inconclusive(tmp_path: Path):
    data = get_base_summary_data()
    data["metrics"]["throughput"]["significant"] = False
    summary_path = tmp_path / "summary.json"
    with open(summary_path, "w") as f: json.dump(data, f)
    
    report = run_conjecture_engine(str(summary_path))
    assert report["behavior_classification"] == BehaviorClassification.INCONCLUSIVE.name
    assert "insufficient to either support or refute" in report["derived_conjecture"]

def test_harmful_regression(tmp_path: Path):
    data = get_base_summary_data()
    data["metrics"]["throughput"]["significant"] = True
    data["metrics"]["throughput"]["delta_pct"] = -10.0 # Negative uplift
    summary_path = tmp_path / "summary.json"
    with open(summary_path, "w") as f: json.dump(data, f)
    
    report = run_conjecture_engine(str(summary_path))
    assert report["behavior_classification"] == BehaviorClassification.HARMFUL_REGRESSION.name
    assert "strictly worse" in report["derived_conjecture"]

def test_regression_in_success(tmp_path: Path):
    data = get_base_summary_data()
    data["governance"]["passed"] = False
    data["metrics"]["success_rate"]["delta"] = -0.05 # Success rate decreased
    summary_path = tmp_path / "summary.json"
    with open(summary_path, "w") as f: json.dump(data, f)
    
    report = run_conjecture_engine(str(summary_path))
    assert report["behavior_classification"] == BehaviorClassification.REGRESSION_IN_SUCCESS.name
    assert "over-optimized for speed" in report["derived_conjecture"]

def test_missing_file():
    with pytest.raises(ValueError, match="Failed to load or parse"):
        run_conjecture_engine("non_existent_summary.json")

def test_invalid_json(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    with open(summary_path, "w") as f: f.write("{,}")
    
    with pytest.raises(ValueError, match="Failed to load or parse"):
        run_conjecture_engine(str(summary_path))

