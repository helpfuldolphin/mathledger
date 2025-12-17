"""
Tests for budget vs divergence cross-plot generator.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from derivation.budget_invariants import build_budget_invariant_snapshot
from derivation.budget_cal_exp_integration import (
    annotate_cal_exp_windows_with_budget_storyline,
    attach_budget_storyline_to_cal_exp_report,
    extract_budget_storyline_from_cal_exp_report,
)
from derivation.budget_invariants import (
    build_budget_invariant_timeline,
    build_budget_storyline,
    project_budget_stability_horizon,
)


# Mock PipelineStats-like objects
class MockStats:
    def __init__(
        self,
        budget_exhausted: bool = False,
        max_candidates_hit: bool = False,
        timeout_abstentions: int = 0,
        statements_skipped: int = 0,
        candidates_considered: int = 0,
        budget_remaining_s: float | None = None,
        post_exhaustion_candidates: int = 0,
    ):
        self.budget_exhausted = budget_exhausted
        self.max_candidates_hit = max_candidates_hit
        self.timeout_abstentions = timeout_abstentions
        self.statements_skipped = statements_skipped
        self.candidates_considered = candidates_considered
        self.budget_remaining_s = budget_remaining_s
        self.post_exhaustion_candidates = post_exhaustion_candidates


def test_extract_cross_plot_data_shape(tmp_path: Path):
    """Test cross-plot data extraction produces correct shape."""
    # Build synthetic CAL-EXP-1 report with budget storyline
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(5)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(5)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    report = {
        "schema_version": "1.0.0",
        "windows": [
            {"divergence_rate": 0.05, "delta_bias": 0.01},
            {"divergence_rate": 0.06, "delta_bias": 0.02},
            {"divergence_rate": 0.04, "delta_bias": 0.01},
        ],
    }
    
    enriched = attach_budget_storyline_to_cal_exp_report(
        report, timeline, storyline, projection, "CAL-EXP-1", "run1"
    )
    
    # Import and test extract function
    from scripts.plot_budget_vs_divergence import extract_cross_plot_data
    
    data = extract_cross_plot_data([enriched])
    
    assert len(data) == 3  # One per window
    assert all("window_idx" in d for d in data)
    assert all("stability_index" in d for d in data)
    assert all("divergence_rate" in d for d in data)
    assert all("experiment_id" in d for d in data)
    assert all("run_id" in d for d in data)
    
    # Verify values
    assert data[0]["window_idx"] == 0
    assert data[0]["divergence_rate"] == 0.05
    assert data[0]["stability_index"] == timeline.get("stability_index", 0.0)
    assert data[0]["experiment_id"] == "CAL-EXP-1"
    
    # Verify new budget fields
    assert "budget_combined_status" in data[0]
    assert "budget_projection_class" in data[0]
    assert "budget_confounded" in data[0]
    assert "budget_confound_reason" in data[0]
    assert "stability_index" in data[0]


def test_cross_plot_csv_output(tmp_path: Path):
    """Test CSV output format."""
    from scripts.plot_budget_vs_divergence import extract_cross_plot_data, write_csv_output
    
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    report = {
        "schema_version": "1.0.0",
        "windows": [
            {"divergence_rate": 0.05},
            {"divergence_rate": 0.06},
        ],
    }
    
    enriched = attach_budget_storyline_to_cal_exp_report(
        report, timeline, storyline, projection, "CAL-EXP-1", "run1"
    )
    
    data = extract_cross_plot_data([enriched])
    output_path = tmp_path / "output.csv"
    write_csv_output(data, output_path)
    
    # Verify CSV structure
    assert output_path.exists()
    with open(output_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert "window_idx" in rows[0]
        assert "stability_index" in rows[0]
        assert "divergence_rate" in rows[0]
        assert "budget_combined_status" in rows[0]
        assert "budget_projection_class" in rows[0]
        assert "budget_confounded" in rows[0]
        assert "budget_confound_reason" in rows[0]
        assert "stability_index" in rows[0]
        assert "experiment_id" in rows[0]
        assert "run_id" in rows[0]


def test_cross_plot_json_output(tmp_path: Path):
    """Test JSON output format."""
    from scripts.plot_budget_vs_divergence import extract_cross_plot_data, write_json_output
    
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    report = {
        "schema_version": "1.0.0",
        "windows": [{"divergence_rate": 0.05}],
    }
    
    enriched = attach_budget_storyline_to_cal_exp_report(
        report, timeline, storyline, projection, "CAL-EXP-1", "run1"
    )
    
    data = extract_cross_plot_data([enriched])
    output_path = tmp_path / "output.json"
    write_json_output(data, output_path)
    
    # Verify JSON structure
    assert output_path.exists()
    with open(output_path, "r") as f:
        output = json.load(f)
        assert output["schema_version"] == "1.0.0"
        assert "data" in output
        assert "summary" in output
        assert len(output["data"]) == 1
        assert output["summary"]["total_points"] == 1
    
    # Verify new budget fields in JSON output
    assert "budget_combined_status" in output["data"][0]
    assert "budget_projection_class" in output["data"][0]
    assert "budget_confounded" in output["data"][0]
    assert "budget_confound_reason" in output["data"][0]
    assert "stability_index" in output["data"][0]


def test_cross_plot_with_annotated_windows(tmp_path: Path):
    """Test cross-plot extraction with pre-annotated windows."""
    from scripts.plot_budget_vs_divergence import extract_cross_plot_data
    
    snapshots = [build_budget_invariant_snapshot(MockStats()) for _ in range(3)]
    timeline = build_budget_invariant_timeline(snapshots)
    health_history = [{"health_score": 90.0, "trend_status": "STABLE"} for _ in range(3)]
    storyline = build_budget_storyline(timeline, health_history)
    projection = project_budget_stability_horizon(health_history)
    
    report = {
        "schema_version": "1.0.0",
        "windows": [
            {"divergence_rate": 0.05},
            {"divergence_rate": 0.06},
        ],
    }
    
    enriched = attach_budget_storyline_to_cal_exp_report(
        report, timeline, storyline, projection, "CAL-EXP-1", "run1"
    )
    
    # Annotate windows with budget storyline
    budget_storyline = extract_budget_storyline_from_cal_exp_report(enriched)
    annotated_report = annotate_cal_exp_windows_with_budget_storyline(
        enriched, budget_storyline
    )
    
    # Extract cross-plot data
    data = extract_cross_plot_data([annotated_report])
    
    # Verify window-level budget fields are used
    assert len(data) == 2
    assert all("budget_combined_status" in d for d in data)
    assert all("budget_projection_class" in d for d in data)
    assert all("budget_confounded" in d for d in data)



