"""
U2 Planner Cycle Replay Tests

Verifies:
- Trace-based replay
- Snapshot-based replay
- Determinism across replays
- Hash verification
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any, Tuple

from rfl.prng import DeterministicPRNG
from experiments.u2 import (
    U2Runner,
    U2Config,
    run_with_traces,
    load_experiment_trace,
    verify_trace_determinism,
    extract_telemetry_from_trace,
    compare_telemetry,
    save_snapshot,
    load_snapshot,
)


class TestTraceReplay:
    """Test trace-based replay."""
    
    def create_mock_execute_fn(self, seed: int):
        """Create deterministic mock execution function."""
        prng = DeterministicPRNG(seed)
        
        def execute(item: Any, cycle_seed: int) -> Tuple[bool, Any]:
            # Deterministic based on item and cycle
            exec_prng = prng.for_path("execute", str(item), str(cycle_seed))
            success = exec_prng.random() > 0.4
            result = {
                "outcome": "VERIFIED" if success else "FAILED",
                "item": str(item),
            }
            return success, result
        
        return execute
    
    def test_trace_generation(self):
        """Trace file is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            trace_path = tmppath / "trace.jsonl"
            
            config = U2Config(
                experiment_id="test_trace_gen",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=3,
                master_seed=42,
                max_beam_width=10,
            )
            
            # Run with traces
            runner = U2Runner(config)
            runner.frontier.push("seed_item", priority=1.0, depth=0)
            
            execute_fn = self.create_mock_execute_fn(42)
            
            from experiments.u2.logging import U2TraceLogger
            from experiments.u2.runner import TracedExperimentContext
            
            with U2TraceLogger(
                output_path=trace_path,
                experiment_id=config.experiment_id,
                slice_name=config.slice_name,
                mode=config.mode,
                master_seed="0x000000000000002a",
            ) as logger:
                trace_ctx = TracedExperimentContext(trace_logger=logger)
                
                for cycle in range(config.total_cycles):
                    runner.run_cycle(cycle, execute_fn, trace_ctx)
            
            # Verify trace file exists
            assert trace_path.exists()
            
            # Load and verify trace
            trace = load_experiment_trace(trace_path)
            assert trace.experiment_id == config.experiment_id
            assert len(trace.cycles) == config.total_cycles
    
    def test_replay_determinism(self):
        """Replaying same experiment produces identical trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            trace1_path = tmppath / "trace1.jsonl"
            trace2_path = tmppath / "trace2.jsonl"
            
            config = U2Config(
                experiment_id="test_replay_det",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=5,
                master_seed=12345,
                max_beam_width=10,
            )
            
            # Run experiment twice
            for trace_path in [trace1_path, trace2_path]:
                runner = U2Runner(config)
                runner.frontier.push("seed_item", priority=1.0, depth=0)
                
                execute_fn = self.create_mock_execute_fn(12345)
                
                from experiments.u2.logging import U2TraceLogger
                from experiments.u2.runner import TracedExperimentContext
                
                with U2TraceLogger(
                    output_path=trace_path,
                    experiment_id=config.experiment_id,
                    slice_name=config.slice_name,
                    mode=config.mode,
                    master_seed="0x0000000000003039",
                ) as logger:
                    trace_ctx = TracedExperimentContext(trace_logger=logger)
                    
                    for cycle in range(config.total_cycles):
                        runner.run_cycle(cycle, execute_fn, trace_ctx)
            
            # Verify traces are identical
            is_deterministic = verify_trace_determinism(trace1_path, trace2_path)
            assert is_deterministic, "Replayed experiment must produce identical trace"
    
    def test_trace_hash_stability(self):
        """Trace hashes are stable across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            trace_path = tmppath / "trace.jsonl"
            
            config = U2Config(
                experiment_id="test_hash_stable",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=3,
                master_seed=99999,
                max_beam_width=10,
            )
            
            # Run experiment
            runner = U2Runner(config)
            runner.frontier.push("seed_item", priority=1.0, depth=0)
            
            execute_fn = self.create_mock_execute_fn(99999)
            
            from experiments.u2.logging import U2TraceLogger
            from experiments.u2.runner import TracedExperimentContext
            
            with U2TraceLogger(
                output_path=trace_path,
                experiment_id=config.experiment_id,
                slice_name=config.slice_name,
                mode=config.mode,
                master_seed="0x000000000001869f",
            ) as logger:
                trace_ctx = TracedExperimentContext(trace_logger=logger)
                
                for cycle in range(config.total_cycles):
                    runner.run_cycle(cycle, execute_fn, trace_ctx)
            
            # Load trace and get hashes
            trace = load_experiment_trace(trace_path)
            hashes1 = [cycle.hash() for cycle in trace.cycles]
            
            # Reload and recompute hashes
            trace2 = load_experiment_trace(trace_path)
            hashes2 = [cycle.hash() for cycle in trace2.cycles]
            
            assert hashes1 == hashes2, "Trace hashes must be stable"


class TestSnapshotReplay:
    """Test snapshot-based replay."""
    
    def test_snapshot_save_load(self):
        """Snapshot can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            snapshot_path = tmppath / "snapshot.json"
            
            config = U2Config(
                experiment_id="test_snapshot_sl",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=10,
                master_seed=42,
                snapshot_dir=tmppath,
            )
            
            runner = U2Runner(config)
            runner.frontier.push("seed_item", priority=1.0, depth=0)
            
            # Run some cycles
            execute_fn = lambda item, seed: (True, {"result": "ok"})
            for cycle in range(5):
                runner.run_cycle(cycle, execute_fn)
            
            # Save snapshot
            from experiments.u2.snapshots import SnapshotData
            snapshot = SnapshotData(
                experiment_id=config.experiment_id,
                slice_name=config.slice_name,
                mode=config.mode,
                master_seed="0x000000000000002a",
                current_cycle=5,
                total_cycles=config.total_cycles,
                frontier_state=runner.frontier.get_state(),
                prng_state=runner.slice_prng.get_state(),
                stats=runner.stats,
            )
            
            save_snapshot(snapshot, snapshot_path)
            
            # Load snapshot
            loaded = load_snapshot(snapshot_path, verify_hash=True)
            
            assert loaded.experiment_id == config.experiment_id
            assert loaded.current_cycle == 5
    
    def test_resume_from_snapshot(self):
        """Experiment can resume from snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            config = U2Config(
                experiment_id="test_resume",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=10,
                master_seed=42,
                snapshot_dir=tmppath,
            )
            
            # Run first half
            runner1 = U2Runner(config)
            runner1.frontier.push("seed_item", priority=1.0, depth=0)
            
            execute_fn = lambda item, seed: (True, {"result": "ok"})
            
            for cycle in range(5):
                runner1.run_cycle(cycle, execute_fn)
            
            # Save snapshot
            snapshot_hash = runner1.save_snapshot(5)
            
            # Create new runner and restore
            runner2 = U2Runner(config)
            
            from experiments.u2.snapshots import load_snapshot, find_latest_snapshot
            snapshot_path = find_latest_snapshot(tmppath, config.experiment_id)
            assert snapshot_path is not None
            
            snapshot = load_snapshot(snapshot_path)
            runner2.restore_state(snapshot)
            
            # Verify state
            assert runner2.current_cycle == 5
            assert runner2.stats["total_candidates_processed"] == runner1.stats["total_candidates_processed"]


class TestTelemetryReplay:
    """Test telemetry extraction and comparison."""
    
    def test_telemetry_extraction(self):
        """Telemetry can be extracted from trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            trace_path = tmppath / "trace.jsonl"
            
            config = U2Config(
                experiment_id="test_telemetry",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=3,
                master_seed=42,
                max_beam_width=10,
            )
            
            # Run experiment
            runner = U2Runner(config)
            runner.frontier.push("seed_item", priority=1.0, depth=0)
            
            execute_fn = lambda item, seed: (True, {"result": "ok"})
            
            from experiments.u2.logging import U2TraceLogger
            from experiments.u2.runner import TracedExperimentContext
            
            with U2TraceLogger(
                output_path=trace_path,
                experiment_id=config.experiment_id,
                slice_name=config.slice_name,
                mode=config.mode,
                master_seed="0x000000000000002a",
            ) as logger:
                trace_ctx = TracedExperimentContext(trace_logger=logger)
                
                for cycle in range(config.total_cycles):
                    runner.run_cycle(cycle, execute_fn, trace_ctx)
            
            # Extract telemetry
            telemetry = extract_telemetry_from_trace(trace_path)
            
            assert telemetry.experiment_id == config.experiment_id
            assert telemetry.total_cycles == config.total_cycles
            assert len(telemetry.cycle_stats) == config.total_cycles
    
    def test_telemetry_comparison(self):
        """Telemetry from identical runs can be compared."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            config = U2Config(
                experiment_id="test_telemetry_cmp",
                slice_name="test_slice",
                mode="baseline",
                total_cycles=3,
                master_seed=42,
                max_beam_width=10,
            )
            
            # Run twice and extract telemetry
            telemetries = []
            
            for run_id in [1, 2]:
                trace_path = tmppath / f"trace_{run_id}.jsonl"
                
                runner = U2Runner(config)
                runner.frontier.push("seed_item", priority=1.0, depth=0)
                
                execute_fn = lambda item, seed: (True, {"result": "ok"})
                
                from experiments.u2.logging import U2TraceLogger
                from experiments.u2.runner import TracedExperimentContext
                
                with U2TraceLogger(
                    output_path=trace_path,
                    experiment_id=config.experiment_id,
                    slice_name=config.slice_name,
                    mode=config.mode,
                    master_seed="0x000000000000002a",
                ) as logger:
                    trace_ctx = TracedExperimentContext(trace_logger=logger)
                    
                    for cycle in range(config.total_cycles):
                        runner.run_cycle(cycle, execute_fn, trace_ctx)
                
                telemetries.append(extract_telemetry_from_trace(trace_path))
            
            # Compare telemetries
            comparison = compare_telemetry(telemetries[0], telemetries[1])
            
            assert comparison["deterministic"], "Identical runs must produce identical telemetry"
            assert len(comparison["differences"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
