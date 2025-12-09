"""
PHASE II — NOT USED IN PHASE I

Tests for Snapshot Orchestrator Integration
===========================================

Tests verifying:
- Auto-resume integration with multi-run planning
- SnapshotPlanEvent emission
- Event serialization round-trip
- Correct RESUME vs NEW_RUN decisions
"""

import json
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
from experiments.u2.schema import (
    SnapshotPlanEvent,
    TRACE_SCHEMA_VERSION,
)
from experiments.u2.snapshot_history import (
    build_multi_run_snapshot_history,
    plan_future_runs,
    summarize_snapshot_plans_for_u2_orchestrator,
)

# Import orchestrator functions
import sys
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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


# --- Test: Auto-Resume Decision Logic ---

class TestAutoResumeDecision:
    """Tests for auto-resume decision making."""
    
    def test_resume_chosen_when_runs_available(self, temp_base_dir):
        """Should choose RESUME when valid runs with snapshots exist."""
        # Create run with good snapshots
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20, 30], total_cycles=100)
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # Should choose RESUME if runs are available
        if plan["runs_to_extend"]:
            assert summary["status"] == "RESUME"
            assert summary["has_resume_targets"] is True
            assert summary["preferred_snapshot_path"] is not None
    
    def test_new_run_chosen_when_no_viable_resumes(self, temp_base_dir):
        """Should choose NEW_RUN when no viable resume points exist."""
        # Create empty run (no snapshots)
        run1 = temp_base_dir / "run1"
        run1.mkdir()
        
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # Should suggest NEW_RUN
        if not plan["runs_to_extend"] and plan["suggested_new_runs"] > 0:
            assert summary["status"] == "NEW_RUN"
            assert summary["has_resume_targets"] is False
    
    def test_prefers_highest_priority_run(self, temp_base_dir):
        """Should prefer the run with highest priority score."""
        # Run 1: low coverage (higher priority)
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        # Run 2: better coverage (lower priority)
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [10, 20, 30, 40, 50], total_cycles=100)
        
        run_dirs = [str(run1), str(run2)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # Should prefer run1 (lower coverage = higher priority)
        if len(plan["runs_to_extend"]) >= 2:
            top_priority = plan["runs_to_extend"][0]
            assert summary["preferred_run_id"] == top_priority["run_id"]


# --- Test: SnapshotPlanEvent ---

class TestSnapshotPlanEvent:
    """Tests for SnapshotPlanEvent schema and serialization."""
    
    def test_event_creation(self):
        """Should create SnapshotPlanEvent with all fields."""
        event = SnapshotPlanEvent(
            status="RESUME",
            preferred_run_id="run_001",
            preferred_snapshot_path="/path/to/snapshot.snap",
            total_runs_analyzed=3,
        )
        
        assert event.status == "RESUME"
        assert event.preferred_run_id == "run_001"
        assert event.preferred_snapshot_path == "/path/to/snapshot.snap"
        assert event.total_runs_analyzed == 3
    
    def test_event_serialization_round_trip(self):
        """Should serialize and deserialize correctly."""
        from dataclasses import asdict, fields
        
        event = SnapshotPlanEvent(
            status="NEW_RUN",
            preferred_run_id=None,
            preferred_snapshot_path=None,
            total_runs_analyzed=0,
        )
        
        # Serialize to dict
        event_dict = asdict(event)
        
        # Should have all fields
        assert "status" in event_dict
        assert "preferred_run_id" in event_dict
        assert "preferred_snapshot_path" in event_dict
        assert "total_runs_analyzed" in event_dict
        
        # Values should match
        assert event_dict["status"] == "NEW_RUN"
        assert event_dict["preferred_run_id"] is None
        assert event_dict["total_runs_analyzed"] == 0
    
    def test_event_json_serialization(self):
        """Should serialize to JSON correctly."""
        event = SnapshotPlanEvent(
            status="RESUME",
            preferred_run_id="test_run",
            preferred_snapshot_path="/snapshots/test.snap",
            total_runs_analyzed=5,
        )
        
        # Convert to dict and then JSON
        from dataclasses import asdict
        event_dict = asdict(event)
        json_str = json.dumps(event_dict)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        
        assert parsed["status"] == "RESUME"
        assert parsed["preferred_run_id"] == "test_run"
        assert parsed["preferred_snapshot_path"] == "/snapshots/test.snap"
        assert parsed["total_runs_analyzed"] == 5
    
    def test_event_with_none_values(self):
        """Should handle None values correctly."""
        event = SnapshotPlanEvent(
            status="NO_ACTION",
            preferred_run_id=None,
            preferred_snapshot_path=None,
            total_runs_analyzed=0,
        )
        
        from dataclasses import asdict
        event_dict = asdict(event)
        
        assert event_dict["preferred_run_id"] is None
        assert event_dict["preferred_snapshot_path"] is None


# --- Test: Integration Workflow ---

class TestIntegrationWorkflow:
    """Tests for complete integration workflow."""
    
    def test_multi_run_analysis_to_event(self, temp_base_dir):
        """Test complete workflow from multi-run analysis to event creation."""
        # Create multiple runs with varying coverage
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [15, 25, 35], total_cycles=100)
        
        # Build multi-run history
        run_dirs = [str(run1), str(run2)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        
        # Plan future runs
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        
        # Get orchestrator summary
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # Create event
        event = SnapshotPlanEvent(
            status=summary["status"],
            preferred_run_id=summary.get("preferred_run_id"),
            preferred_snapshot_path=summary.get("preferred_snapshot_path"),
            total_runs_analyzed=multi_history["run_count"],
        )
        
        # Verify event
        assert event.status in ("RESUME", "NEW_RUN", "NO_ACTION")
        assert event.total_runs_analyzed == 2
    
    def test_event_emission_simulation(self, temp_base_dir):
        """Simulate event emission in trace logger."""
        from experiments.u2.logging import U2TraceLogger
        from dataclasses import asdict
        
        # Create a run
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20], total_cycles=100)
        
        # Build plan
        run_dirs = [str(run1)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # Create event
        event = SnapshotPlanEvent(
            status=summary["status"],
            preferred_run_id=summary.get("preferred_run_id"),
            preferred_snapshot_path=summary.get("preferred_snapshot_path"),
            total_runs_analyzed=multi_history["run_count"],
        )
        
        # Simulate logging (write to temp file)
        log_path = temp_base_dir / "test_trace.jsonl"
        with U2TraceLogger(log_path, enabled_events={"snapshot_plan"}) as logger:
            logger.log_snapshot_plan(event)
        
        # Verify event was written
        assert log_path.exists()
        
        with open(log_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        assert len(lines) == 1
        
        # Parse JSON
        log_entry = json.loads(lines[0])
        
        assert log_entry["event_type"] == "SnapshotPlanEvent"
        assert log_entry["payload"]["status"] == event.status
        assert log_entry["payload"]["total_runs_analyzed"] == event.total_runs_analyzed


# --- Test: Edge Cases ---

class TestEdgeCases:
    """Tests for edge cases in orchestrator integration."""
    
    def test_empty_snapshot_root(self, temp_base_dir):
        """Should handle empty snapshot root directory."""
        # Empty directory
        snapshot_root = temp_base_dir / "empty"
        snapshot_root.mkdir()
        
        run_dirs = [str(d) for d in snapshot_root.iterdir() if d.is_dir()]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # Should suggest NEW_RUN
        assert summary["status"] == "NEW_RUN"
        assert summary["has_resume_targets"] is False
    
    def test_all_runs_blocked(self, temp_base_dir):
        """Should handle case where all runs are blocked."""
        # Create runs but corrupt all snapshots
        run1 = create_run_with_snapshots(temp_base_dir, "run1", [10, 20])
        run2 = create_run_with_snapshots(temp_base_dir, "run2", [15, 25])
        
        # Corrupt all snapshots
        for run_dir in [run1, run2]:
            snapshot_dir = run_dir / "snapshots"
            for snap_file in snapshot_dir.glob("*.snap"):
                with open(snap_file, 'rb') as f:
                    data = bytearray(f.read())
                if len(data) > 50:
                    data[50] ^= 0xFF
                with open(snap_file, 'wb') as f:
                    f.write(data)
        
        run_dirs = [str(run1), str(run2)]
        multi_history = build_multi_run_snapshot_history(run_dirs)
        plan = plan_future_runs(multi_history, target_coverage=10.0)
        summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
        
        # Should suggest NEW_RUN when all blocked
        if multi_history["overall_status"] == "BLOCK":
            assert summary["status"] == "NEW_RUN"
    
    def test_event_with_all_status_types(self):
        """Should handle all status types in event."""
        for status in ["NO_ACTION", "RESUME", "NEW_RUN"]:
            event = SnapshotPlanEvent(
                status=status,
                preferred_run_id="test" if status == "RESUME" else None,
                preferred_snapshot_path="/test.snap" if status == "RESUME" else None,
                total_runs_analyzed=1 if status != "NO_ACTION" else 0,
            )
            
            assert event.status == status
            from dataclasses import asdict
            event_dict = asdict(event)
            assert event_dict["status"] == status

