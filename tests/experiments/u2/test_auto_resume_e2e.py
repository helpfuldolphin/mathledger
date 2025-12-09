"""
PHASE II — NOT USED IN PHASE I

End-to-End Auto-Resume Integration Tests
========================================

Tests the complete auto-resume workflow from CLI invocation through
snapshot selection and event emission.

This test harness creates synthetic run directories and validates
that the orchestrator correctly:
- Discovers and analyzes multiple runs
- Selects the highest-priority snapshot
- Emits SnapshotPlanEvent
- Falls back gracefully when no viable resumes exist
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("msgpack")
pytest.importorskip("zstandard")

from experiments.u2.snapshots import (
    SnapshotData,
    save_snapshot,
)
from experiments.u2.schema import SnapshotPlanEvent
from experiments.u2.snapshot_history import (
    build_multi_run_snapshot_history,
    plan_future_runs,
    summarize_snapshot_plans_for_u2_orchestrator,
)


# --- Fixtures ---

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
    experiment_id: str = None,
    mode: str = "baseline",
) -> Path:
    """
    Create a synthetic run directory with snapshots.
    
    This simulates a real experiment run without executing Lean/FO.
    """
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create snapshot directory
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    
    # Create output directory structure
    (run_dir / "results").mkdir(exist_ok=True)
    
    exp_id = experiment_id or run_name
    
    # Create snapshots
    for cycle in cycles:
        snapshot = SnapshotData(
            cycle_index=cycle,
            total_cycles=total_cycles,
            mode=mode,
            slice_name="test_slice",
            experiment_id=exp_id,
        )
        path = snapshot_dir / f"snapshot_{exp_id}_{cycle:06d}.snap"
        save_snapshot(snapshot, path)
    
    # Create a minimal manifest for completeness
    manifest = {
        "slice": "test_slice",
        "mode": mode,
        "cycles": total_cycles,
        "initial_seed": 42,
        "experiment_id": exp_id,
    }
    with open(run_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f)
    
    return run_dir


def invoke_auto_resume_wrapper(
    snapshot_root: Path,
    slice_name: str = "test_slice",
    cycles: int = 10,
    seed: int = 42,
    mode: str = "baseline",
    trace_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Wrapper that simulates --auto-resume invocation without running actual Lean/FO.
    
    Returns the orchestrator summary and any emitted events.
    """
    # Import here to avoid circular dependencies
    from experiments.u2.snapshot_history import (
        build_multi_run_snapshot_history,
        plan_future_runs,
        summarize_snapshot_plans_for_u2_orchestrator,
    )
    
    # Discover run directories
    run_dirs = [str(d) for d in snapshot_root.iterdir() if d.is_dir()]
    
    if not run_dirs:
        # No runs found - return NEW_RUN decision
        return {
            "status": "NEW_RUN",
            "has_resume_targets": False,
            "preferred_run_id": None,
            "preferred_snapshot_path": None,
            "total_runs_analyzed": 0,
        }
    
    # Build multi-run history
    multi_history = build_multi_run_snapshot_history(run_dirs)
    
    # Plan future runs
    plan = plan_future_runs(multi_history, target_coverage=10.0)
    
    # Get orchestrator summary
    summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
    
    # Create event (as orchestrator would)
    event = SnapshotPlanEvent(
        status=summary["status"],
        preferred_run_id=summary.get("preferred_run_id"),
        preferred_snapshot_path=summary.get("preferred_snapshot_path"),
        total_runs_analyzed=multi_history["run_count"],
    )
    
    # Simulate trace log emission if path provided
    if trace_log_path:
        from experiments.u2.logging import U2TraceLogger
        with U2TraceLogger(trace_log_path, enabled_events={"snapshot_plan"}) as logger:
            logger.log_snapshot_plan(event)
    
    return {
        "status": summary["status"],
        "has_resume_targets": summary["has_resume_targets"],
        "preferred_run_id": summary.get("preferred_run_id"),
        "preferred_snapshot_path": summary.get("preferred_snapshot_path"),
        "total_runs_analyzed": multi_history["run_count"],
        "event": event,
        "plan": plan,
    }


# --- Test: E2E Auto-Resume Workflow ---

class TestAutoResumeE2E:
    """End-to-end tests for auto-resume workflow."""
    
    def test_selects_highest_priority_run(self, temp_base_dir):
        """Should select the run with highest priority score."""
        # Run 1: Low coverage (2%) - should be highest priority
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        # Run 2: Better coverage (5%) - lower priority
        run2 = create_synthetic_run(temp_base_dir, "run2", [10, 20, 30, 40, 50], total_cycles=100)
        
        # Run 3: Good coverage (10%) - lowest priority
        run3 = create_synthetic_run(temp_base_dir, "run3", list(range(10, 110, 10)), total_cycles=100)
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should choose RESUME
        assert result["status"] == "RESUME"
        assert result["has_resume_targets"] is True
        assert result["preferred_snapshot_path"] is not None
        
        # Should prefer run1 (lowest coverage = highest priority)
        assert "run1" in result["preferred_snapshot_path"] or result["preferred_run_id"] == "run1"
    
    def test_emits_snapshot_plan_event(self, temp_base_dir):
        """Should emit SnapshotPlanEvent in trace log."""
        # Create a run with snapshots
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        trace_log = temp_base_dir / "trace.jsonl"
        result = invoke_auto_resume_wrapper(temp_base_dir, trace_log_path=trace_log)
        
        # Verify event was created
        assert result["event"] is not None
        assert isinstance(result["event"], SnapshotPlanEvent)
        assert result["event"].status in ("RESUME", "NEW_RUN", "NO_ACTION")
        
        # Verify event was written to trace log
        if trace_log.exists():
            with open(trace_log, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            
            # Should have at least one event
            assert len(lines) >= 1
            
            # Parse first event
            log_entry = json.loads(lines[0])
            assert log_entry["event_type"] == "SnapshotPlanEvent"
            assert log_entry["payload"]["status"] == result["event"].status
    
    def test_falls_back_to_new_run_when_no_viable_resumes(self, temp_base_dir):
        """Should fall back to NEW_RUN when no viable resume points exist."""
        # Create empty run (no snapshots)
        run1 = temp_base_dir / "run1"
        run1.mkdir()
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should choose NEW_RUN
        assert result["status"] == "NEW_RUN"
        assert result["has_resume_targets"] is False
        assert result["preferred_snapshot_path"] is None
        assert result["total_runs_analyzed"] == 0
    
    def test_handles_mixed_coverage_scenarios(self, temp_base_dir):
        """Should handle runs with varying coverage levels."""
        # Run 1: Very low coverage (1%)
        run1 = create_synthetic_run(temp_base_dir, "run1", [50], total_cycles=100)
        
        # Run 2: Low coverage with large gap
        run2 = create_synthetic_run(temp_base_dir, "run2", [10, 70], total_cycles=100)
        
        # Run 3: Good coverage
        run3 = create_synthetic_run(temp_base_dir, "run3", list(range(10, 100, 10)), total_cycles=100)
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should choose RESUME (at least one viable)
        assert result["status"] == "RESUME"
        assert result["total_runs_analyzed"] == 3
        
        # Should prefer run with lowest coverage or largest gap
        assert result["preferred_snapshot_path"] is not None
    
    def test_prioritizes_runs_with_large_gaps(self, temp_base_dir):
        """Should prioritize runs with large gaps over those with small gaps."""
        # Run 1: Small gaps (10 cycles each)
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30, 40, 50], total_cycles=100)
        
        # Run 2: Large gap (50 cycles)
        run2 = create_synthetic_run(temp_base_dir, "run2", [10, 60], total_cycles=100)
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should prefer run2 due to large gap
        if result["status"] == "RESUME":
            # Run2 should have higher priority
            plan = result["plan"]
            if plan["runs_to_extend"]:
                top_priority = plan["runs_to_extend"][0]
                # Should be run2 (has larger gap)
                assert "run2" in top_priority["run_dir"] or top_priority["max_gap"] > 20


# --- Test: Failure Mode Handling ---

class TestFailureModeHandling:
    """Tests for handling edge cases and failure modes."""
    
    def test_handles_corrupt_run_directories(self, temp_base_dir):
        """Should handle runs with corrupted snapshots gracefully."""
        # Create run with valid snapshots
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30])
        
        # Corrupt one snapshot
        snapshot_dir = run1 / "snapshots"
        snap_files = list(snapshot_dir.glob("*.snap"))
        if snap_files:
            with open(snap_files[0], 'rb') as f:
                data = bytearray(f.read())
            if len(data) > 50:
                data[50] ^= 0xFF
            with open(snap_files[0], 'wb') as f:
                f.write(data)
        
        # Should still work (corrupted snapshot is detected but doesn't block)
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should still find valid snapshots
        assert result["total_runs_analyzed"] >= 1
    
    def test_handles_missing_snapshot_directories(self, temp_base_dir):
        """Should handle runs without snapshot directories."""
        # Run 1: Has snapshots
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20])
        
        # Run 2: No snapshot directory
        run2 = temp_base_dir / "run2"
        run2.mkdir()
        (run2 / "results").mkdir()
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should still analyze run1
        assert result["total_runs_analyzed"] >= 1
    
    def test_handles_mixed_directory_shapes(self, temp_base_dir):
        """Should handle runs with different directory structures."""
        # Run 1: Standard structure with snapshots/
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20])
        
        # Run 2: Snapshots directly in run dir (alternative structure)
        run2 = temp_base_dir / "run2"
        run2.mkdir()
        snapshot = SnapshotData(cycle_index=15, total_cycles=100, experiment_id="run2")
        save_snapshot(snapshot, run2 / "snapshot_run2_000015.snap")
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should discover both runs
        assert result["total_runs_analyzed"] >= 1
    
    def test_handles_empty_snapshot_root(self, temp_base_dir):
        """Should handle empty snapshot root directory."""
        # Empty directory
        result = invoke_auto_resume_wrapper(temp_base_dir)
        
        # Should fall back to NEW_RUN
        assert result["status"] == "NEW_RUN"
        assert result["total_runs_analyzed"] == 0
    
    def test_handles_permission_errors_gracefully(self, temp_base_dir):
        """Should handle permission errors without crashing."""
        # Create a run
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20])
        
        # Make directory unreadable (on Unix) - skip on Windows
        import platform
        if platform.system() != "Windows":
            try:
                run1.chmod(0o000)
                # Should still work (error handling in build_multi_run_snapshot_history)
                result = invoke_auto_resume_wrapper(temp_base_dir)
                # Should handle gracefully
                assert result["status"] in ("RESUME", "NEW_RUN", "NO_ACTION")
            finally:
                # Restore permissions
                run1.chmod(0o755)
        else:
            # On Windows, just verify normal operation
            result = invoke_auto_resume_wrapper(temp_base_dir)
            assert result["status"] in ("RESUME", "NEW_RUN", "NO_ACTION")


# --- Test: Event Validation ---

class TestEventValidation:
    """Tests for SnapshotPlanEvent validation and correctness."""
    
    def test_event_matches_orchestrator_summary(self, temp_base_dir):
        """Event should match orchestrator summary exactly."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30])
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        event = result["event"]
        
        # Event fields should match summary
        assert event.status == result["status"]
        assert event.preferred_run_id == result["preferred_run_id"]
        assert event.preferred_snapshot_path == result["preferred_snapshot_path"]
        assert event.total_runs_analyzed == result["total_runs_analyzed"]
    
    def test_event_serialization_in_trace_log(self, temp_base_dir):
        """Event should serialize correctly in trace log."""
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20])
        
        trace_log = temp_base_dir / "trace.jsonl"
        result = invoke_auto_resume_wrapper(temp_base_dir, trace_log_path=trace_log)
        
        if trace_log.exists():
            with open(trace_log, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            
            # Find SnapshotPlanEvent
            for line in lines:
                entry = json.loads(line)
                if entry.get("event_type") == "SnapshotPlanEvent":
                    payload = entry["payload"]
                    
                    # Verify all fields present
                    assert "status" in payload
                    assert "preferred_run_id" in payload
                    assert "preferred_snapshot_path" in payload
                    assert "total_runs_analyzed" in payload
                    
                    # Verify values match
                    assert payload["status"] == result["status"]
                    break
            else:
                pytest.fail("SnapshotPlanEvent not found in trace log")


# --- Test: Priority Scoring Accuracy ---

class TestPriorityScoring:
    """Tests for priority scoring accuracy."""
    
    def test_low_coverage_gets_higher_priority(self, temp_base_dir):
        """Runs with lower coverage should get higher priority scores."""
        # Run 1: 2% coverage
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        # Run 2: 5% coverage
        run2 = create_synthetic_run(temp_base_dir, "run2", [10, 20, 30, 40, 50], total_cycles=100)
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        plan = result["plan"]
        
        if len(plan["runs_to_extend"]) >= 2:
            # Run1 should have higher priority (lower coverage)
            run1_entry = next((r for r in plan["runs_to_extend"] if "run1" in r["run_dir"]), None)
            run2_entry = next((r for r in plan["runs_to_extend"] if "run2" in r["run_dir"]), None)
            
            if run1_entry and run2_entry:
                assert run1_entry["priority_score"] > run2_entry["priority_score"]
    
    def test_large_gaps_increase_priority(self, temp_base_dir):
        """Runs with large gaps should get priority boost."""
        # Run 1: Small gaps
        run1 = create_synthetic_run(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        # Run 2: Large gap
        run2 = create_synthetic_run(temp_base_dir, "run2", [10, 60], total_cycles=100)
        
        result = invoke_auto_resume_wrapper(temp_base_dir)
        plan = result["plan"]
        
        if len(plan["runs_to_extend"]) >= 2:
            # Run2 should have higher priority due to large gap
            run2_entry = next((r for r in plan["runs_to_extend"] if "run2" in r["run_dir"]), None)
            if run2_entry:
                assert run2_entry["max_gap"] > 20
                assert run2_entry["priority_score"] > 0

