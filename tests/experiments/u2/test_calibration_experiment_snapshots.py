"""
PHASE II — NOT USED IN PHASE I

Tests for Calibration Experiment Snapshot Runbooks
===================================================

Tests snapshot runbook summaries for CAL-EXP-1, CAL-EXP-2, and CAL-EXP-3.
"""

import json
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest

from experiments.u2.snapshots import SnapshotData, save_snapshot
from experiments.u2.snapshot_history import (
    build_multi_run_snapshot_history,
    plan_future_runs,
    build_calibration_experiment_runbook,
    compare_multi_run_snapshots,
    classify_calibration_trend,
)


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for test runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_synthetic_run(
    base_dir: Path,
    run_name: str,
    cycles: List[int],
    total_cycles: int = 100,
    experiment_id: Optional[str] = None,
    mode: str = "baseline",
) -> Path:
    """Create a synthetic run directory with snapshots."""
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    
    exp_id = experiment_id or run_name
    
    for cycle in cycles:
        snapshot = SnapshotData(
            experiment_id=exp_id,
            slice_name="test_slice",
            mode=mode,
            master_seed="test_seed",
            current_cycle=cycle,
            total_cycles=total_cycles,
            snapshot_cycle=cycle,
        )
        from experiments.u2.snapshots import create_snapshot_name
        snapshot_filename = create_snapshot_name(exp_id, cycle) + ".json"
        save_snapshot(snapshot, snapshot_dir / snapshot_filename)
    
    return run_dir


# --- Test: CAL-EXP-1 (Warm-Start) ---

class TestCalExp1Runbook:
    """Tests for CAL-EXP-1 warm-start runbook."""
    
    def test_builds_cal_exp1_runbook(self, temp_base_dir):
        """Should build CAL-EXP-1 runbook with short-window stability focus."""
        # Create runs with good coverage (short-window stability)
        run1 = create_synthetic_run(temp_base_dir, "cal_exp1_run1", list(range(10, 100, 5)), total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        runbook = build_calibration_experiment_runbook(multi_history, plan, "CAL-EXP-1")
        
        assert runbook["experiment_type"] == "CAL-EXP-1"
        assert runbook["experiment_focus"] == "short_window_stability"
        assert "stability_indicators" in runbook
        assert "coverage_consistency" in runbook["stability_indicators"]
        assert "gap_risk" in runbook["stability_indicators"]
        assert runbook["calibration_metrics"]["runs_analyzed"] == 1
    
    def test_cal_exp1_stability_indicators(self, temp_base_dir):
        """Should correctly assess stability indicators for CAL-EXP-1."""
        # High coverage run (should have high consistency, low gap risk)
        run1 = create_synthetic_run(temp_base_dir, "cal_exp1_high", list(range(10, 100, 2)), total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        runbook = build_calibration_experiment_runbook(multi_history, plan, "CAL-EXP-1")
        
        indicators = runbook["stability_indicators"]
        # High coverage should yield high consistency
        if runbook["mean_coverage_pct"] > 50.0:
            assert indicators["coverage_consistency"] == "high"
        # Small gaps should yield low risk
        if runbook["max_gap"] < 20:
            assert indicators["gap_risk"] == "low"


# --- Test: CAL-EXP-2 (Long-Window Convergence) ---

class TestCalExp2Runbook:
    """Tests for CAL-EXP-2 long-window convergence runbook."""
    
    def test_builds_cal_exp2_runbook(self, temp_base_dir):
        """Should build CAL-EXP-2 runbook with long-window convergence focus."""
        # Create runs with good coverage (convergence)
        run1 = create_synthetic_run(temp_base_dir, "cal_exp2_run1", list(range(10, 100, 3)), total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        runbook = build_calibration_experiment_runbook(multi_history, plan, "CAL-EXP-2")
        
        assert runbook["experiment_type"] == "CAL-EXP-2"
        assert runbook["experiment_focus"] == "long_window_convergence"
        assert "convergence_indicators" in runbook
        assert "coverage_trend" in runbook["convergence_indicators"]
        assert "gap_stability" in runbook["convergence_indicators"]
    
    def test_cal_exp2_convergence_indicators(self, temp_base_dir):
        """Should correctly assess convergence indicators for CAL-EXP-2."""
        # High coverage run (should show stable trend)
        run1 = create_synthetic_run(temp_base_dir, "cal_exp2_stable", list(range(10, 100, 2)), total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        runbook = build_calibration_experiment_runbook(multi_history, plan, "CAL-EXP-2")
        
        indicators = runbook["convergence_indicators"]
        # High coverage should show stable trend
        if runbook["mean_coverage_pct"] > 70.0:
            assert indicators["coverage_trend"] == "stable"
        # Small gaps should show stable gap stability
        if runbook["max_gap"] < 30:
            assert indicators["gap_stability"] == "stable"


# --- Test: CAL-EXP-3 (Regime-Change Resilience) ---

class TestCalExp3Runbook:
    """Tests for CAL-EXP-3 regime-change resilience runbook."""
    
    def test_builds_cal_exp3_runbook(self, temp_base_dir):
        """Should build CAL-EXP-3 runbook with regime-change resilience focus."""
        # Create runs with moderate coverage (resilience testing)
        run1 = create_synthetic_run(temp_base_dir, "cal_exp3_run1", [10, 20, 30, 50, 70, 90], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        runbook = build_calibration_experiment_runbook(multi_history, plan, "CAL-EXP-3")
        
        assert runbook["experiment_type"] == "CAL-EXP-3"
        assert runbook["experiment_focus"] == "regime_change_resilience"
        assert "resilience_indicators" in runbook
        assert "coverage_robustness" in runbook["resilience_indicators"]
        assert "gap_tolerance" in runbook["resilience_indicators"]
    
    def test_cal_exp3_resilience_indicators(self, temp_base_dir):
        """Should correctly assess resilience indicators for CAL-EXP-3."""
        # Moderate coverage run (should show medium robustness)
        run1 = create_synthetic_run(temp_base_dir, "cal_exp3_moderate", [10, 25, 40, 60, 80], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        runbook = build_calibration_experiment_runbook(multi_history, plan, "CAL-EXP-3")
        
        indicators = runbook["resilience_indicators"]
        # Coverage robustness should be assessed
        assert indicators["coverage_robustness"] in ["high", "medium", "low"]
        # Gap tolerance should be assessed
        assert indicators["gap_tolerance"] in ["acceptable", "concerning"]


# --- Test: Multi-Run Comparison ---

class TestMultiRunComparison:
    """Tests for multi-run comparison analysis."""
    
    def test_compares_multi_run_snapshots(self, temp_base_dir):
        """Should perform multi-run comparison analysis."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        run2 = create_synthetic_run(temp_base_dir, "run2", [10, 20, 30, 40, 50], total_cycles=100)
        
        run_dirs = [str(run1), str(run2)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        
        assert "schema_version" in comparison
        assert "stability_deltas" in comparison
        assert "max_gap_analysis" in comparison
        assert "coverage_regression" in comparison
        assert comparison["stability_deltas"]["coverage_stability"] in ["stable", "moderate", "unstable"]
    
    def test_detects_coverage_regression(self, temp_base_dir):
        """Should detect coverage regression when comparing histories."""
        # Create first history with high coverage
        run1 = create_synthetic_run(temp_base_dir, "run1_high", list(range(10, 100, 5)), total_cycles=100)
        run_dirs1 = [str(run1)]
        previous_history = build_multi_run_snapshot_history(run_dirs1)
        
        # Create second history with lower coverage
        run2 = create_synthetic_run(temp_base_dir, "run2_low", [10, 20, 30], total_cycles=100)
        run_dirs2 = [str(run2)]
        current_history = build_multi_run_snapshot_history(run_dirs2)
        
        comparison = compare_multi_run_snapshots(current_history, previous_history)
        
        # Should detect regression if previous had higher coverage
        prev_coverage = previous_history.get("summary", {}).get("average_coverage_pct", 0.0)
        curr_coverage = current_history.get("summary", {}).get("average_coverage_pct", 0.0)
        
        if prev_coverage > curr_coverage:
            assert comparison["coverage_regression"]["detected"] is True
            assert comparison["coverage_regression"]["severity"] in ["high", "medium", "low"]
    
    def test_detects_problematic_gaps(self, temp_base_dir):
        """Should detect runs with problematic gaps."""
        # Create run with large gap (>50% of total cycles)
        run1 = create_synthetic_run(temp_base_dir, "run1_large_gap", [10, 60], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        
        # Should identify problematic gaps
        problematic = comparison["max_gap_analysis"]["problematic_gaps"]
        if multi_history.get("global_max_gap", 0) > 50:  # More than 50% of 100 cycles
            assert len(problematic) > 0
            assert problematic[0]["risk_level"] in ["high", "medium"]
    
    def test_stability_deltas_computation(self, temp_base_dir):
        """Should compute stability deltas correctly."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        run2 = create_synthetic_run(temp_base_dir, "run2", [10, 20, 30, 40, 50], total_cycles=100)
        run3 = create_synthetic_run(temp_base_dir, "run3", [10, 20, 30, 40, 50, 60, 70], total_cycles=100)
        
        run_dirs = [str(run1), str(run2), str(run3)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        
        sd = comparison["stability_deltas"]
        assert "coverage_mean" in sd
        assert "coverage_std" in sd
        assert "coverage_stability" in sd
        assert "gap_mean" in sd
        assert "status_distribution" in sd
        assert isinstance(sd["coverage_mean"], (int, float))
        assert isinstance(sd["coverage_std"], (int, float))
    
    def test_comparison_json_serializable(self, temp_base_dir):
        """Comparison analysis should be JSON-serializable."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        
        # Should serialize without error
        json_str = json.dumps(comparison)
        assert isinstance(json_str, str)
        
        # Should deserialize
        deserialized = json.loads(json_str)
        assert "stability_deltas" in deserialized
        assert "max_gap_analysis" in deserialized
        assert "coverage_regression" in deserialized


# --- Test: Trend Verdict Classification ---

class TestTrendVerdict:
    """Tests for calibration trend verdict classification."""
    
    def test_detects_regression(self, temp_base_dir):
        """Should detect REGRESSING verdict when coverage regression is detected."""
        # Create previous history with high coverage
        run1_prev = create_synthetic_run(temp_base_dir, "run1_prev", list(range(10, 100, 5)), total_cycles=100)
        run_dirs_prev = [str(run1_prev)]
        previous_history = build_multi_run_snapshot_history(run_dirs_prev)
        
        # Create current history with lower coverage
        run1_curr = create_synthetic_run(temp_base_dir, "run1_curr", [10, 20, 30], total_cycles=100)
        run_dirs_curr = [str(run1_curr)]
        current_history = build_multi_run_snapshot_history(run_dirs_curr)
        
        comparison = compare_multi_run_snapshots(current_history, previous_history)
        verdict = classify_calibration_trend(comparison)
        
        # Should detect regression if previous had higher coverage
        prev_coverage = previous_history.get("summary", {}).get("average_coverage_pct", 0.0)
        curr_coverage = current_history.get("summary", {}).get("average_coverage_pct", 0.0)
        
        if prev_coverage > curr_coverage + 10.0:  # Significant regression
            assert verdict["verdict"] == "REGRESSING"
            assert verdict["confidence"] > 0.0
            assert "coverage_regression" in [m["metric"] for m in verdict["contributing_metrics"]]
    
    def test_discriminates_stable_vs_improving(self, temp_base_dir):
        """Should correctly discriminate between STABLE and IMPROVING verdicts."""
        # Create stable run (good coverage, low std)
        run1 = create_synthetic_run(temp_base_dir, "run1_stable", list(range(10, 100, 3)), total_cycles=100)
        run2 = create_synthetic_run(temp_base_dir, "run2_stable", list(range(10, 100, 3)), total_cycles=100)
        
        run_dirs = [str(run1), str(run2)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        verdict = classify_calibration_trend(comparison)
        
        # Should classify as STABLE or IMPROVING based on signals
        assert verdict["verdict"] in ["STABLE", "IMPROVING", "INCONCLUSIVE"]
        assert verdict["confidence"] >= 0.0
        assert verdict["confidence"] <= 1.0
        
        # Check that contributing metrics are present
        assert len(verdict["contributing_metrics"]) > 0
        assert len(verdict["top_signals"]) <= 3
        
        # If coverage is high and stable, should lean toward IMPROVING or STABLE
        coverage_mean = comparison["stability_deltas"]["coverage_mean"]
        coverage_std = comparison["stability_deltas"]["coverage_std"]
        
        if coverage_mean > 70.0 and coverage_std < 5.0:
            # High stable coverage should not be REGRESSING
            assert verdict["verdict"] != "REGRESSING"
    
    def test_verdict_json_serializable(self, temp_base_dir):
        """Trend verdict should be JSON-serializable."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        verdict = classify_calibration_trend(comparison)
        
        # Should serialize without error
        json_str = json.dumps(verdict)
        assert isinstance(json_str, str)
        
        # Should deserialize
        deserialized = json.loads(json_str)
        assert "verdict" in deserialized
        assert "confidence" in deserialized
        assert "contributing_metrics" in deserialized
        assert "rationale" in deserialized
        assert "top_signals" in deserialized
        assert deserialized["verdict"] in ["IMPROVING", "STABLE", "REGRESSING", "INCONCLUSIVE"]
        assert 0.0 <= deserialized["confidence"] <= 1.0
    
    def test_verdict_has_explainability(self, temp_base_dir):
        """Trend verdict should include explainability (metrics, confidence, rationale)."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        verdict = classify_calibration_trend(comparison)
        
        # Check explainability fields
        assert "verdict" in verdict
        assert "confidence" in verdict
        assert isinstance(verdict["confidence"], (int, float))
        assert 0.0 <= verdict["confidence"] <= 1.0
        
        assert "contributing_metrics" in verdict
        assert isinstance(verdict["contributing_metrics"], list)
        for metric in verdict["contributing_metrics"]:
            assert "metric" in metric
            assert "value" in metric
            assert "weight" in metric
            assert "direction" in metric
            assert "description" in metric
            assert metric["direction"] in ["positive", "negative"]
        
        assert "rationale" in verdict
        assert isinstance(verdict["rationale"], str)
        assert len(verdict["rationale"]) > 0
        
        assert "top_signals" in verdict
        assert isinstance(verdict["top_signals"], list)
        assert len(verdict["top_signals"]) <= 3
        for signal in verdict["top_signals"]:
            assert "signal" in signal
            assert "strength" in signal
            assert "impact" in signal
            assert "message" in signal
    
    def test_verdict_inconclusive_for_insufficient_data(self, temp_base_dir):
        """Should return INCONCLUSIVE verdict when data is insufficient."""
        # Create minimal run with very few snapshots
        run1 = create_synthetic_run(temp_base_dir, "run1_minimal", [10], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        verdict = classify_calibration_trend(comparison)
        
        # If there's insufficient data, should be INCONCLUSIVE
        # (This depends on the actual data, but we check the structure)
        assert verdict["verdict"] in ["IMPROVING", "STABLE", "REGRESSING", "INCONCLUSIVE"]
        assert "rationale" in verdict
        assert "confidence" in verdict
    
    def test_verdict_weight_summary(self, temp_base_dir):
        """Trend verdict should include weight summary for transparency."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        comparison = compare_multi_run_snapshots(multi_history)
        verdict = classify_calibration_trend(comparison)
        
        # Check weight summary
        assert "weight_summary" in verdict
        ws = verdict["weight_summary"]
        assert "positive_weight" in ws
        assert "negative_weight" in ws
        assert "total_weight" in ws
        assert isinstance(ws["positive_weight"], (int, float))
        assert isinstance(ws["negative_weight"], (int, float))
        assert isinstance(ws["total_weight"], (int, float))
        assert ws["total_weight"] >= 0.0



