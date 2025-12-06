"""
Performance Guardrail Tests for U2

Validates that U2 experiments complete within acceptable performance bounds
and that the safety envelope correctly identifies performance issues.
"""

import pytest
import time
from typing import Tuple, Any

from experiments.u2.runner import U2Config, U2Runner
from experiments.u2.safety_envelope import build_u2_safety_envelope
from experiments.u2.entrypoint import run_u2_experiment


class TestPerformanceThresholds:
    """Tests for performance threshold detection."""
    
    def test_fast_cycles_pass(self):
        """Test that fast cycles pass performance checks."""
        config = U2Config(
            experiment_id="fast_test",
            slice_name="perf_test",
            mode="baseline",
            total_cycles=5,
            master_seed=42,
        )
        
        def fast_execute(item: str, seed: int) -> Tuple[bool, Any]:
            # Very fast execution
            return True, {"ok": True}
        
        results, envelope = run_u2_experiment(
            config=config,
            items=["a", "b", "c"],
            execute_fn=fast_execute,
        )
        
        assert envelope.perf_ok
        assert envelope.safety_status == "OK"
        assert all(r.duration_ms < 1000 for r in results if r.duration_ms)
    
    def test_slow_cycles_warn(self):
        """Test that slow cycles trigger warnings."""
        config = U2Config(
            experiment_id="slow_test",
            slice_name="perf_test",
            mode="baseline",
            total_cycles=3,
            master_seed=42,
        )
        
        def slow_execute(item: str, seed: int) -> Tuple[bool, Any]:
            # Simulate slow execution (exceeds threshold)
            time.sleep(0.1)  # 100ms - but we'll fake stats to trigger warning
            return True, {"ok": True}
        
        # Run experiment
        runner = U2Runner(config)
        results = []
        for _ in range(3):
            result = runner.run_cycle(["a", "b", "c"], slow_execute)
            results.append(result)
        
        # Manually create high performance stats to trigger warning
        perf_stats = {
            "cycle_durations_ms": [6000.0, 7000.0, 6500.0],  # Exceeds threshold
            "max_cycle_duration_ms": 7000.0,
            "avg_cycle_duration_ms": 6500.0,
        }
        
        envelope = build_u2_safety_envelope(config, perf_stats, [])
        
        assert not envelope.perf_ok
        assert envelope.safety_status == "WARN"
        assert len(envelope.warnings) > 0
        assert any("Max cycle duration" in w for w in envelope.warnings)


class TestCycleDurationTracking:
    """Tests for cycle duration tracking."""
    
    def test_duration_is_recorded(self):
        """Test that cycle durations are recorded."""
        config = U2Config(
            experiment_id="duration_test",
            slice_name="test",
            mode="baseline",
            total_cycles=5,
            master_seed=42,
        )
        
        runner = U2Runner(config)
        
        def execute(item: str, seed: int) -> Tuple[bool, Any]:
            return True, {"ok": True}
        
        for _ in range(5):
            runner.run_cycle(["a", "b", "c"], execute)
        
        assert len(runner.cycle_durations_ms) == 5
        assert all(d > 0 for d in runner.cycle_durations_ms)
        assert all(d < 1000 for d in runner.cycle_durations_ms)  # Should be fast
    
    def test_duration_in_cycle_result(self):
        """Test that duration is included in CycleResult."""
        config = U2Config(
            experiment_id="duration_test",
            slice_name="test",
            mode="baseline",
            total_cycles=1,
            master_seed=42,
        )
        
        runner = U2Runner(config)
        
        def execute(item: str, seed: int) -> Tuple[bool, Any]:
            return True, {"ok": True}
        
        result = runner.run_cycle(["a", "b", "c"], execute)
        
        assert result.duration_ms is not None
        assert result.duration_ms > 0


class TestSafetyEnvelopePerfIntegration:
    """Tests for safety envelope performance integration."""
    
    def test_envelope_includes_perf_stats(self):
        """Test that envelope includes performance statistics."""
        config = U2Config(
            experiment_id="perf_stats_test",
            slice_name="test",
            mode="baseline",
            total_cycles=5,
            master_seed=42,
        )
        
        def execute(item: str, seed: int) -> Tuple[bool, Any]:
            return True, {"ok": True}
        
        results, envelope = run_u2_experiment(
            config=config,
            items=["a", "b", "c"],
            execute_fn=execute,
        )
        
        assert "max_cycle_duration_ms" in envelope.perf_stats
        assert "avg_cycle_duration_ms" in envelope.perf_stats
        assert "cycle_count" in envelope.perf_stats
        assert envelope.perf_stats["cycle_count"] == 5
    
    def test_custom_perf_thresholds(self):
        """Test using custom performance thresholds."""
        config = U2Config(
            experiment_id="custom_threshold_test",
            slice_name="test",
            mode="baseline",
            total_cycles=3,
            master_seed=42,
        )
        
        # Simulate moderate performance
        perf_stats = {
            "cycle_durations_ms": [100.0, 150.0, 120.0],
            "max_cycle_duration_ms": 150.0,
            "avg_cycle_duration_ms": 123.3,
        }
        
        # Test with strict thresholds
        strict_thresholds = {
            "max_cycle_duration_ms": 100.0,  # Very strict
            "max_avg_cycle_duration_ms": 100.0,
            "max_eval_lint_issues": 10,
        }
        
        envelope = build_u2_safety_envelope(
            config,
            perf_stats,
            [],
            perf_thresholds=strict_thresholds,
        )
        
        # Should fail with strict thresholds
        assert not envelope.perf_ok
        assert envelope.safety_status == "WARN"


class TestPerformanceRegression:
    """Tests to detect performance regressions."""
    
    def test_baseline_performance_benchmark(self):
        """Benchmark baseline performance for regression detection."""
        config = U2Config(
            experiment_id="benchmark",
            slice_name="test",
            mode="baseline",
            total_cycles=100,
            master_seed=42,
        )
        
        def fast_execute(item: str, seed: int) -> Tuple[bool, Any]:
            # Minimal work
            return True, {"ok": True}
        
        start = time.time()
        results, envelope = run_u2_experiment(
            config=config,
            items=["a", "b", "c", "d", "e"],
            execute_fn=fast_execute,
        )
        end = time.time()
        
        total_time = end - start
        avg_cycle_time = total_time / 100
        
        # Benchmark: 100 cycles should complete in under 1 second
        assert total_time < 1.0, f"Performance regression: 100 cycles took {total_time:.2f}s"
        
        # Each cycle should be very fast
        assert avg_cycle_time < 0.01, f"Avg cycle time too high: {avg_cycle_time:.4f}s"
        
        # Safety envelope should pass
        assert envelope.perf_ok
        assert envelope.safety_status == "OK"
    
    def test_rfl_mode_performance(self):
        """Test that RFL mode has acceptable performance."""
        config = U2Config(
            experiment_id="rfl_perf",
            slice_name="test",
            mode="rfl",
            total_cycles=50,
            master_seed=42,
        )
        
        def execute(item: str, seed: int) -> Tuple[bool, Any]:
            return True, {"ok": True}
        
        start = time.time()
        results, envelope = run_u2_experiment(
            config=config,
            items=["a", "b", "c", "d", "e"],
            execute_fn=execute,
        )
        end = time.time()
        
        total_time = end - start
        
        # RFL mode should still be fast
        assert total_time < 0.5, f"RFL mode too slow: {total_time:.2f}s for 50 cycles"
        
        # Safety envelope should pass
        assert envelope.perf_ok


@pytest.mark.unit
class TestPerformanceMonitoring:
    """Integration tests for performance monitoring."""
    
    def test_performance_envelope_workflow(self):
        """Test complete performance monitoring workflow."""
        # Step 1: Run experiment
        config = U2Config(
            experiment_id="monitoring_test",
            slice_name="test",
            mode="baseline",
            total_cycles=10,
            master_seed=42,
        )
        
        def execute(item: str, seed: int) -> Tuple[bool, Any]:
            return True, {"ok": True}
        
        results, envelope = run_u2_experiment(
            config=config,
            items=["a", "b", "c"],
            execute_fn=execute,
        )
        
        # Step 2: Verify performance data is collected
        assert len(results) == 10
        assert all(r.duration_ms is not None for r in results)
        
        # Step 3: Verify envelope contains perf data
        assert envelope.perf_stats["cycle_count"] == 10
        assert envelope.perf_stats["max_cycle_duration_ms"] > 0
        assert envelope.perf_stats["avg_cycle_duration_ms"] > 0
        
        # Step 4: Verify safety status
        assert envelope.perf_ok
        assert envelope.safety_status == "OK"
    
    def test_performance_warning_propagation(self):
        """Test that performance warnings are properly propagated."""
        config = U2Config(
            experiment_id="warning_test",
            slice_name="test",
            mode="baseline",
            total_cycles=1,
            master_seed=42,
        )
        
        # Create artificially slow performance stats
        perf_stats = {
            "cycle_durations_ms": [10000.0],  # 10 seconds - way too slow
            "max_cycle_duration_ms": 10000.0,
            "avg_cycle_duration_ms": 10000.0,
        }
        
        envelope = build_u2_safety_envelope(config, perf_stats, [])
        
        # Should have warnings
        assert not envelope.perf_ok
        assert len(envelope.warnings) > 0
        assert envelope.safety_status == "WARN"
        
        # Warnings should mention duration
        warning_text = " ".join(envelope.warnings)
        assert "duration" in warning_text.lower() or "threshold" in warning_text.lower()
