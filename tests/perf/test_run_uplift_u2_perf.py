"""
Performance Microbenchmark for U2 Uplift Runner

This test ensures the refactored U2 runner maintains acceptable performance
for large cycle counts. The benchmark is deterministic and robust across
different CI environments.

PHASE II — NOT USED IN PHASE I

Performance Thresholds:
- The default threshold assumes a reasonable baseline for CI hardware
- Thresholds can be adjusted via environment variable U2_PERF_THRESHOLD_MS
- The test is designed to catch significant performance regressions
- If CI hardware changes, update the threshold documentation below

To adjust threshold:
  export U2_PERF_THRESHOLD_MS=5000  # 5 seconds for 1000 cycles

To skip on specific environments:
  export U2_SKIP_PERF_TESTS=1
"""

import os
import time
import pytest
from pathlib import Path
from typing import Any, Tuple

from experiments.u2.runner import U2Runner, U2Config, CycleResult


# Default threshold: 3000ms for 1000 cycles (3ms per cycle)
# This is conservative and should work on most CI hardware
DEFAULT_THRESHOLD_MS = 3000

# Configurable threshold via environment variable
THRESHOLD_MS = int(os.environ.get('U2_PERF_THRESHOLD_MS', DEFAULT_THRESHOLD_MS))

# Option to skip perf tests in certain environments
SKIP_PERF_TESTS = os.environ.get('U2_SKIP_PERF_TESTS', '0') == '1'


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "u2_perf_test"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def simple_curriculum():
    """Simple curriculum for benchmarking."""
    return {
        "slices": {
            "perf_test": {
                "name": "perf_test",
                "items": [f"item_{i}" for i in range(10)],
            }
        }
    }


@pytest.fixture
def mock_execute_fn():
    """Mock execution function with minimal overhead."""
    import hashlib
    
    def execute(item: str, seed: int) -> Tuple[bool, Any]:
        # Minimal execution: just return success based on hash
        h = int(hashlib.sha256(f"{item}{seed}".encode()).hexdigest()[:8], 16)
        success = (h % 2) == 0
        return success, {"outcome": "mock"}
    return execute


@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Perf tests disabled via U2_SKIP_PERF_TESTS")
def test_runner_cycle_performance_baseline(temp_output_dir, simple_curriculum, mock_execute_fn):
    """
    Benchmark baseline mode runner performance.
    
    This test runs 1000 cycles and ensures the total time is within threshold.
    It measures the core cycle execution loop without I/O overhead.
    """
    # Configuration
    num_cycles = 1000
    config = U2Config(
        experiment_id="perf_test_baseline",
        slice_name="perf_test",
        mode="baseline",
        total_cycles=num_cycles,
        master_seed=42,
        snapshot_interval=0,  # Disable snapshots for pure cycle perf
        snapshot_dir=temp_output_dir / "snapshots",
        output_dir=temp_output_dir,
        slice_config=simple_curriculum["slices"]["perf_test"],
    )
    
    runner = U2Runner(config)
    items = config.slice_config["items"]
    
    # Warm-up: run a few cycles to ensure JIT compilation, etc.
    for _ in range(10):
        runner.run_cycle(items, mock_execute_fn)
    
    # Reset state after warm-up
    runner.cycle_index = 0
    runner.ht_series = []
    
    # Benchmark: run cycles and measure time
    start_time = time.perf_counter()
    
    for _ in range(num_cycles):
        result = runner.run_cycle(items, mock_execute_fn)
        assert isinstance(result, CycleResult)
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Verify correctness
    assert runner.cycle_index == num_cycles
    assert len(runner.ht_series) == num_cycles
    
    # Performance assertion
    print(f"\nPerformance: {num_cycles} cycles in {elapsed_ms:.2f}ms ({elapsed_ms/num_cycles:.2f}ms per cycle)")
    print(f"Threshold: {THRESHOLD_MS}ms")
    
    assert elapsed_ms < THRESHOLD_MS, (
        f"Performance regression: {elapsed_ms:.2f}ms > {THRESHOLD_MS}ms threshold. "
        f"Consider optimizing or adjusting threshold via U2_PERF_THRESHOLD_MS."
    )


@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Perf tests disabled via U2_SKIP_PERF_TESTS")
def test_runner_cycle_performance_rfl(temp_output_dir, simple_curriculum, mock_execute_fn):
    """
    Benchmark RFL mode runner performance.
    
    RFL mode includes policy updates, so we allow a slightly higher threshold.
    """
    # Configuration
    num_cycles = 1000
    # RFL mode may be slightly slower due to policy updates
    rfl_threshold_ms = int(THRESHOLD_MS * 1.5)  # 50% tolerance for policy overhead
    
    config = U2Config(
        experiment_id="perf_test_rfl",
        slice_name="perf_test",
        mode="rfl",
        total_cycles=num_cycles,
        master_seed=42,
        snapshot_interval=0,  # Disable snapshots for pure cycle perf
        snapshot_dir=temp_output_dir / "snapshots",
        output_dir=temp_output_dir,
        slice_config=simple_curriculum["slices"]["perf_test"],
    )
    
    runner = U2Runner(config)
    items = config.slice_config["items"]
    
    # Warm-up
    for _ in range(10):
        runner.run_cycle(items, mock_execute_fn)
    
    # Reset state after warm-up
    runner.cycle_index = 0
    runner.ht_series = []
    runner.policy_update_count = 0
    runner.success_count = {}
    runner.attempt_count = {}
    
    # Benchmark
    start_time = time.perf_counter()
    
    for _ in range(num_cycles):
        result = runner.run_cycle(items, mock_execute_fn)
        assert isinstance(result, CycleResult)
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    # Verify correctness
    assert runner.cycle_index == num_cycles
    assert len(runner.ht_series) == num_cycles
    assert runner.policy_update_count == num_cycles  # One update per cycle
    
    # Performance assertion (with RFL tolerance)
    print(f"\nPerformance (RFL): {num_cycles} cycles in {elapsed_ms:.2f}ms ({elapsed_ms/num_cycles:.2f}ms per cycle)")
    print(f"Threshold: {rfl_threshold_ms}ms (1.5x baseline)")
    
    assert elapsed_ms < rfl_threshold_ms, (
        f"Performance regression (RFL): {elapsed_ms:.2f}ms > {rfl_threshold_ms}ms threshold. "
        f"Consider optimizing or adjusting threshold via U2_PERF_THRESHOLD_MS."
    )


@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Perf tests disabled via U2_SKIP_PERF_TESTS")
def test_telemetry_record_serialization_performance(temp_output_dir):
    """
    Benchmark TelemetryRecord creation and JSON serialization.
    
    This ensures the dataclass-based telemetry doesn't introduce overhead.
    """
    import json
    from dataclasses import asdict
    from experiments.u2.runner import TelemetryRecord
    
    num_records = 10000
    threshold_ms = 500  # 500ms for 10k records (0.05ms per record)
    
    # Benchmark
    start_time = time.perf_counter()
    
    for i in range(num_records):
        record = TelemetryRecord(
            cycle=i,
            slice="perf_test",
            mode="baseline",
            seed=12345 + i,
            item=f"item_{i % 10}",
            result="mock_result",
            success=(i % 2) == 0,
        )
        # Serialize to JSON (common operation in telemetry)
        _ = json.dumps(asdict(record))
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    print(f"\nSerialization: {num_records} records in {elapsed_ms:.2f}ms ({elapsed_ms/num_records:.3f}ms per record)")
    print(f"Threshold: {threshold_ms}ms")
    
    assert elapsed_ms < threshold_ms, (
        f"Serialization performance regression: {elapsed_ms:.2f}ms > {threshold_ms}ms threshold."
    )


@pytest.mark.skipif(SKIP_PERF_TESTS, reason="Perf tests disabled via U2_SKIP_PERF_TESTS")
def test_runner_determinism_with_same_seed(temp_output_dir, simple_curriculum, mock_execute_fn):
    """
    Verify determinism: same seed produces same results.
    
    This is not strictly a performance test, but ensures the optimized runner
    maintains the determinism contract.
    """
    num_cycles = 100
    
    def run_experiment(seed: int):
        config = U2Config(
            experiment_id=f"determinism_test_{seed}",
            slice_name="perf_test",
            mode="baseline",
            total_cycles=num_cycles,
            master_seed=seed,
            snapshot_interval=0,
            snapshot_dir=temp_output_dir / "snapshots",
            output_dir=temp_output_dir,
            slice_config=simple_curriculum["slices"]["perf_test"],
        )
        
        runner = U2Runner(config)
        items = config.slice_config["items"]
        
        ht_series = []
        for _ in range(num_cycles):
            result = runner.run_cycle(items, mock_execute_fn)
            ht_series.append(result.ht)
        
        return ht_series
    
    # Run twice with same seed
    seed = 42
    ht_series_1 = run_experiment(seed)
    ht_series_2 = run_experiment(seed)
    
    # Verify determinism
    assert ht_series_1 == ht_series_2, "Determinism violated: same seed produced different H_t series"
    
    # Run with different seed
    ht_series_3 = run_experiment(seed + 1)
    
    # Verify different seeds produce different results
    assert ht_series_1 != ht_series_3, "Different seeds should produce different results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
