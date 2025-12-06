#!/usr/bin/env python3.11
"""
U2 Planner Demonstration

Shows:
- Basic experiment setup
- Deterministic execution
- Trace logging
- Snapshot/restore
- Telemetry export
"""

import sys
import tempfile
from pathlib import Path
from typing import Any, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfl.prng import DeterministicPRNG
from experiments.u2 import (
    U2Runner,
    U2Config,
    U2TraceLogger,
    TracedExperimentContext,
)
from experiments.u2.telemetry import (
    extract_telemetry_from_trace,
    export_telemetry,
    create_evidence_pack,
)
from experiments.u2.snapshots import load_snapshot, find_latest_snapshot


def create_mock_execute_fn(seed: int):
    """
    Create a deterministic mock execution function.
    
    In a real scenario, this would call the derivation engine.
    """
    prng = DeterministicPRNG(seed)
    
    def execute(item: Any, cycle_seed: int) -> Tuple[bool, Any]:
        # Deterministic execution based on item and cycle
        exec_prng = prng.for_path("execute", str(item), str(cycle_seed))
        
        # Simulate success/failure
        success = exec_prng.random() > 0.3
        
        result = {
            "outcome": "VERIFIED" if success else "FAILED",
            "item": str(item),
            "cycle": cycle_seed,
        }
        
        return success, result
    
    return execute


def demo_basic_experiment():
    """Demonstrate basic experiment execution."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Experiment")
    print("=" * 60)
    
    config = U2Config(
        experiment_id="demo_basic",
        slice_name="demo_slice",
        mode="baseline",
        total_cycles=5,
        master_seed=42,
        max_beam_width=10,
    )
    
    runner = U2Runner(config)
    
    # Seed initial frontier
    runner.frontier.push("axiom_0", priority=1.0, depth=0)
    print(f"✓ Initialized frontier with seed item")
    
    # Create execution function
    execute_fn = create_mock_execute_fn(42)
    
    # Run cycles
    print(f"\nRunning {config.total_cycles} cycles...")
    for cycle in range(config.total_cycles):
        result = runner.run_cycle(cycle, execute_fn)
        print(f"  Cycle {cycle}: {result.candidates_processed} processed, "
              f"{result.candidates_generated} generated")
    
    # Show final stats
    state = runner.get_state()
    print(f"\n✓ Experiment complete")
    print(f"  Total processed: {state['stats']['total_candidates_processed']}")
    print(f"  Total generated: {state['stats']['total_candidates_generated']}")
    print(f"  Frontier size: {state['frontier_stats']['current_size']}")


def demo_trace_logging():
    """Demonstrate trace logging and telemetry."""
    print("\n" + "=" * 60)
    print("DEMO 2: Trace Logging & Telemetry")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        trace_path = tmppath / "trace.jsonl"
        
        config = U2Config(
            experiment_id="demo_trace",
            slice_name="demo_slice",
            mode="baseline",
            total_cycles=3,
            master_seed=99,
            max_beam_width=10,
        )
        
        runner = U2Runner(config)
        runner.frontier.push("axiom_0", priority=1.0, depth=0)
        
        execute_fn = create_mock_execute_fn(99)
        
        # Run with trace logging
        print(f"✓ Running with trace logging to {trace_path.name}")
        
        with U2TraceLogger(
            output_path=trace_path,
            experiment_id=config.experiment_id,
            slice_name=config.slice_name,
            mode=config.mode,
            master_seed="0x0000000000000063",
        ) as logger:
            trace_ctx = TracedExperimentContext(trace_logger=logger)
            
            for cycle in range(config.total_cycles):
                runner.run_cycle(cycle, execute_fn, trace_ctx)
        
        print(f"✓ Trace written to {trace_path}")
        
        # Extract telemetry
        print(f"\n✓ Extracting telemetry...")
        telemetry = extract_telemetry_from_trace(trace_path)
        
        print(f"  Experiment: {telemetry.experiment_id}")
        print(f"  Mode: {telemetry.mode}")
        print(f"  Cycles: {telemetry.total_cycles}")
        print(f"  Candidates processed: {telemetry.total_candidates_processed}")
        print(f"  Trace hashes: {len(telemetry.trace_hashes)}")
        
        # Export telemetry
        telemetry_path = tmppath / "telemetry.json"
        export_telemetry(telemetry, telemetry_path)
        print(f"✓ Telemetry exported to {telemetry_path.name}")
        
        # Create evidence pack
        pack_dir = create_evidence_pack(
            trace_path=trace_path,
            output_dir=tmppath,
            include_trace=True,
        )
        print(f"✓ Evidence pack created at {pack_dir.name}")


def demo_determinism():
    """Demonstrate deterministic replay."""
    print("\n" + "=" * 60)
    print("DEMO 3: Deterministic Replay")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        config = U2Config(
            experiment_id="demo_determinism",
            slice_name="demo_slice",
            mode="baseline",
            total_cycles=5,
            master_seed=12345,
            max_beam_width=10,
        )
        
        execute_fn = create_mock_execute_fn(12345)
        
        # Run experiment twice
        print("✓ Running experiment twice with same seed...")
        
        results = []
        for run_id in [1, 2]:
            runner = U2Runner(config)
            runner.frontier.push("axiom_0", priority=1.0, depth=0)
            
            run_results = []
            for cycle in range(config.total_cycles):
                result = runner.run_cycle(cycle, execute_fn)
                run_results.append(result)
            
            results.append(run_results)
            print(f"  Run {run_id}: {len(run_results)} cycles completed")
        
        # Compare results
        print(f"\n✓ Comparing results...")
        
        match = True
        for cycle in range(config.total_cycles):
            r1 = results[0][cycle]
            r2 = results[1][cycle]
            
            if (r1.candidates_processed != r2.candidates_processed or
                r1.candidates_generated != r2.candidates_generated):
                match = False
                print(f"  ✗ Cycle {cycle}: MISMATCH")
            else:
                print(f"  ✓ Cycle {cycle}: MATCH")
        
        if match:
            print(f"\n✅ DETERMINISM VERIFIED: Both runs produced identical results")
        else:
            print(f"\n❌ DETERMINISM FAILED: Results differ")


def demo_snapshot_restore():
    """Demonstrate snapshot and restore."""
    print("\n" + "=" * 60)
    print("DEMO 4: Snapshot & Restore")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        config = U2Config(
            experiment_id="demo_snapshot",
            slice_name="demo_slice",
            mode="baseline",
            total_cycles=10,
            master_seed=42,
            max_beam_width=10,
            snapshot_dir=tmppath,
        )
        
        execute_fn = create_mock_execute_fn(42)
        
        # Run first half
        print("✓ Running first 5 cycles...")
        runner1 = U2Runner(config)
        runner1.frontier.push("axiom_0", priority=1.0, depth=0)
        
        for cycle in range(5):
            result = runner1.run_cycle(cycle, execute_fn)
            print(f"  Cycle {cycle}: {result.candidates_processed} processed")
        
        # Save snapshot
        print(f"\n✓ Saving snapshot at cycle 5...")
        snapshot_hash = runner1.save_snapshot(5)
        print(f"  Snapshot hash: {snapshot_hash[:16]}...")
        
        state1 = runner1.get_state()
        
        # Create new runner and restore
        print(f"\n✓ Creating new runner and restoring from snapshot...")
        runner2 = U2Runner(config)
        
        snapshot_path = find_latest_snapshot(tmppath, config.experiment_id)
        snapshot = load_snapshot(snapshot_path, verify_hash=False)
        runner2.restore_state(snapshot)
        
        print(f"  Restored to cycle {snapshot.current_cycle}")
        
        # Verify state matches
        state2 = runner2.get_state()
        
        if (state1["stats"]["total_candidates_processed"] == 
            state2["stats"]["total_candidates_processed"]):
            print(f"✅ SNAPSHOT VERIFIED: State restored correctly")
        else:
            print(f"❌ SNAPSHOT FAILED: State mismatch")
        
        # Continue from snapshot
        print(f"\n✓ Continuing from cycle 5...")
        for cycle in range(5, 10):
            result = runner2.run_cycle(cycle, execute_fn)
            print(f"  Cycle {cycle}: {result.candidates_processed} processed")
        
        final_state = runner2.get_state()
        print(f"\n✓ Experiment complete")
        print(f"  Total processed: {final_state['stats']['total_candidates_processed']}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("U2 PLANNER DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo showcases the U2 planner capabilities:")
    print("  1. Basic experiment execution")
    print("  2. Trace logging and telemetry export")
    print("  3. Deterministic replay verification")
    print("  4. Snapshot and restore")
    
    try:
        demo_basic_experiment()
        demo_trace_logging()
        demo_determinism()
        demo_snapshot_restore()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
