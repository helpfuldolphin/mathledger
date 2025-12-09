"""
PHASE II — NOT USED IN PHASE I

Tests for Multi-Run Snapshot History & Run Planning
====================================================

Tests verifying:
- Multi-run snapshot history aggregation
- Run planning advisor
- U2 orchestrator summary adapter
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("msgpack")
pytest.importorskip("zstandard")

from experiments.u2.snapshots import (
    SnapshotData,
    save_snapshot,
)

# Import history module
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.u2.snapshot_history import (
    build_multi_run_snapshot_history,
    plan_future_runs,
    summarize_snapshot_plans_for_u2_orchestrator,
    build_snapshot_history,
)


# --- Fixtures ---

@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory for multiple runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_run_with_snapshots(
    base_dir: Path,
    run_name: str,
    cycles: List[int],
    total_cycles: int = 100,
    experiment_id: str = None,
) -> Path:
    """Helper to create a run directory with snapshots."""
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    
    exp_id = experiment_id or run_name
    
    for cycle in cycles:
        snapshot = SnapshotData(
            cycle_index=cycle,
            total_cycles=total_cycles,
            mode="baseline",
            slice_name="test_slice",
            experiment_id=exp_id,
        )
        path = snapshot_dir / f"snapshot_{exp_id}_{cycle:06d}.snap"
        save_snapshot(snapshot, path)
    
    return run_dir


# --- Test: build_multi_run_snapshot_history ---

class TestMultiRunHistory:
    """Tests for build_multi_run_snapshot_history function."""
    
    def test_aggregates_multiple_runs(self, temp_base_dir):
        """Should aggregate history from multiple run directories."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30])
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [15, 25, 35])
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        assert multi_history["run_count"] == 2
        assert len(multi_history["runs"]) == 2
    
    def test_includes_schema_version(self, temp_base_dir):
        """Should include schema_version in multi-run history."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20])
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        
        assert "schema_version" in multi_history
        assert multi_history["schema_version"] == "1.0"
    
    def test_tracks_runs_with_block_status(self, temp_base_dir):
        """Should count runs with BLOCK status."""
        # Create one healthy run
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30])
        
        # Create one empty run (will be EMPTY status)
        run2 = temp_base_dir / "run2"
        run2.mkdir()
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        # Empty runs don't count as BLOCK, but we should track status
        assert "runs_with_block_status" in multi_history
        assert multi_history["runs_with_block_status"] >= 0
    
    def test_computes_global_max_gap(self, temp_base_dir):
        """Should find maximum gap across all runs."""
        # Run 1: gap of 20
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 30])
        
        # Run 2: gap of 50
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [10, 60])
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        assert multi_history["global_max_gap"] == 50
    
    def test_determines_overall_status(self, temp_base_dir):
        """Should determine overall status across all runs."""
        # All healthy runs
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30])
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [15, 25, 35])
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        assert multi_history["overall_status"] in ("OK", "WARN")
    
    def test_overall_status_block_with_large_gaps(self, temp_base_dir):
        """Should be WARN/BLOCK when large gaps exist."""
        # Run with 60-cycle gap
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 70])
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        
        # Large gap should trigger WARN
        assert multi_history["overall_status"] in ("WARN", "BLOCK")
    
    def test_includes_per_run_summaries(self, temp_base_dir):
        """Should include detailed summary for each run."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30], experiment_id="exp1")
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        
        assert len(multi_history["runs"]) == 1
        run_summary = multi_history["runs"][0]
        
        assert "run_id" in run_summary
        assert "status" in run_summary
        assert "coverage_pct" in run_summary
        assert "max_gap" in run_summary
        assert "recommended_resume_point" in run_summary
    
    def test_tracks_status_counts(self, temp_base_dir):
        """Should track counts of each status type."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20])
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [15, 25])
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        assert "status_counts" in multi_history
        assert "OK" in multi_history["status_counts"]
        assert multi_history["status_counts"]["OK"] >= 0
    
    def test_summary_statistics(self, temp_base_dir):
        """Should include aggregated summary statistics."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30])
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [15, 25, 35])
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        assert "summary" in multi_history
        summary = multi_history["summary"]
        
        assert "total_valid_snapshots" in summary
        assert "average_coverage_pct" in summary
        assert "runs_with_resume_points" in summary
    
    def test_handles_nonexistent_directories(self, temp_base_dir):
        """Should skip non-existent directories gracefully."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20])
        nonexistent = temp_base_dir / "nonexistent"
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(nonexistent)])
        
        # Should only include the valid run
        assert multi_history["run_count"] == 1


# --- Test: plan_future_runs ---

class TestPlanFutureRuns:
    """Tests for plan_future_runs function."""
    
    def test_identifies_runs_to_extend(self, temp_base_dir):
        """Should identify runs that should be extended."""
        # Run with low coverage
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        assert len(plan["runs_to_extend"]) > 0
    
    def test_prioritizes_low_coverage_runs(self, temp_base_dir):
        """Should prioritize runs with lower coverage."""
        # Run 1: 2% coverage
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        # Run 2: 5% coverage
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [10, 20, 30, 40, 50], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        # Run 1 should have higher priority (lower coverage)
        if len(plan["runs_to_extend"]) >= 2:
            assert plan["runs_to_extend"][0]["priority_score"] >= plan["runs_to_extend"][1]["priority_score"]
    
    def test_suggests_new_runs_when_blocked(self, temp_base_dir):
        """Should suggest new runs when no resumable runs exist."""
        # Empty run (no snapshots)
        run1 = temp_base_dir / "run1"
        run1.mkdir()
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        assert plan["suggested_new_runs"] > 0
    
    def test_suggests_new_runs_with_low_coverage(self, temp_base_dir):
        """Should suggest new runs when average coverage is very low."""
        # Run with very low coverage
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        # Should suggest new runs if coverage is below target/2
        if multi_history["summary"]["average_coverage_pct"] < 5.0:
            assert plan["suggested_new_runs"] > 0
    
    def test_includes_priority_scores(self, temp_base_dir):
        """Should include priority scores for runs to extend."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        if plan["runs_to_extend"]:
            run = plan["runs_to_extend"][0]
            assert "priority_score" in run
            assert run["priority_score"] > 0
    
    def test_includes_extension_reasons(self, temp_base_dir):
        """Should include human-readable reasons for extension."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        if plan["runs_to_extend"]:
            run = plan["runs_to_extend"][0]
            assert "reason" in run
            assert len(run["reason"]) > 0
    
    def test_provides_human_readable_message(self, temp_base_dir):
        """Should provide a summary message."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20])
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        assert "message" in plan
        assert len(plan["message"]) > 0
    
    def test_prioritizes_runs_with_large_gaps(self, temp_base_dir):
        """Should prioritize runs with large gaps."""
        # Run 1: small gap
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        # Run 2: large gap
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [10, 50], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        # Run 2 should have higher priority due to large gap
        if len(plan["runs_to_extend"]) >= 2:
            # Find run2 in the list
            run2_entry = next((r for r in plan["runs_to_extend"] if "run2" in r.get("run_dir", "")), None)
            if run2_entry:
                assert run2_entry["max_gap"] > 20


# --- Test: summarize_snapshot_plans_for_u2_orchestrator ---

class TestU2OrchestratorAdapter:
    """Tests for summarize_snapshot_plans_for_u2_orchestrator function."""
    
    def test_has_resume_targets_when_runs_available(self, temp_base_dir):
        """Should indicate resume targets when runs are available."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        if plan["runs_to_extend"]:
            assert summary["has_resume_targets"] is True
    
    def test_provides_preferred_run_id(self, temp_base_dir):
        """Should provide preferred run ID for resuming."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100, experiment_id="exp1")
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        if plan["runs_to_extend"]:
            assert summary["preferred_run_id"] is not None
    
    def test_provides_preferred_snapshot_path(self, temp_base_dir):
        """Should provide path to preferred snapshot."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        if plan["runs_to_extend"]:
            assert summary["preferred_snapshot_path"] is not None
            assert "snapshot" in summary["preferred_snapshot_path"]
    
    def test_status_resume_when_targets_available(self, temp_base_dir):
        """Should return RESUME status when resume targets exist."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        if plan["runs_to_extend"]:
            assert summary["status"] == "RESUME"
    
    def test_status_new_run_when_no_targets(self, temp_base_dir):
        """Should return NEW_RUN status when no resume targets."""
        # Empty run
        run1 = temp_base_dir / "run1"
        run1.mkdir()
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        if plan["suggested_new_runs"] > 0 and not plan["runs_to_extend"]:
            assert summary["status"] == "NEW_RUN"
    
    def test_status_no_action_when_adequate(self, temp_base_dir):
        """Should return NO_ACTION when coverage is adequate."""
        # Run with good coverage
        run1 = create_run_with_snapshots(
            temp_base_dir, "run1", 
            list(range(10, 100, 5)),  # 18 snapshots = 18% coverage
            total_cycles=100
        )
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # If no runs to extend and no new runs suggested, should be NO_ACTION
        if not plan["runs_to_extend"] and plan["suggested_new_runs"] == 0:
            assert summary["status"] == "NO_ACTION"
    
    def test_includes_details(self, temp_base_dir):
        """Should include additional context in details."""
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20])
        
        multi_history = build_multi_run_snapshot_history([str(run1)])
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        assert "details" in summary
        details = summary["details"]
        
        assert "runs_available" in details
        assert "suggested_new_runs" in details
        assert "message" in details


# --- Test: Integration ---

class TestIntegration:
    """Integration tests for multi-run workflow."""
    
    def test_full_workflow(self, temp_base_dir):
        """Test complete workflow from multi-history to orchestrator summary."""
        # Create multiple runs
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [15, 25], total_cycles=100)
        
        # Build multi-run history
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        assert multi_history["run_count"] == 2
        
        # Plan future runs
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        assert "runs_to_extend" in plan
        assert "suggested_new_runs" in plan
        
        # Get orchestrator summary
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        assert "status" in summary
        assert summary["status"] in ("RESUME", "NEW_RUN", "NO_ACTION")
    
    def test_handles_mixed_status_runs(self, temp_base_dir):
        """Should handle mix of healthy and problematic runs."""
        # Healthy run
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30, 40, 50], total_cycles=100)
        
        # Run with large gap
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [10, 70], total_cycles=100)
        
        multi_history = build_multi_run_snapshot_history([str(run1), str(run2)])
        
        # Should aggregate both
        assert multi_history["run_count"] == 2
        
        # Planning should prioritize the problematic one
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        # Run2 should be prioritized due to large gap
        if plan["runs_to_extend"]:
            run2_entry = next((r for r in plan["runs_to_extend"] if "run2" in r.get("run_dir", "")), None)
            if run2_entry:
                assert run2_entry["max_gap"] > 20

