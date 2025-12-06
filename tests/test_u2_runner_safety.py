"""
Tests for U2 Runner and Safety Envelope

Validates type-verified runner, safety envelope, and performance monitoring.
"""

import pytest
from pathlib import Path
from typing import Tuple, Any

from experiments.u2.runner import U2Config, U2Runner, CycleResult
from experiments.u2.safety_envelope import (
    U2SafetyEnvelope,
    build_u2_safety_envelope,
    evaluate_safety_status,
)
from experiments.u2.u2_safe_eval import SafeEvalLintResult
from experiments.u2.entrypoint import run_u2_experiment


@pytest.fixture
def basic_config():
    """Fixture for basic U2 config."""
    return U2Config(
        experiment_id="test_exp",
        slice_name="test_slice",
        mode="baseline",
        total_cycles=5,
        master_seed=42,
    )


@pytest.fixture
def mock_execute_fn():
    """Fixture for mock execution function."""
    def execute(item: str, seed: int) -> Tuple[bool, Any]:
        # Simple deterministic success based on seed
        success = (seed % 2) == 0
        return success, {"outcome": "VERIFIED" if success else "FAILED"}
    return execute


class TestU2Config:
    """Tests for U2Config."""
    
    def test_config_creation(self, basic_config):
        """Test basic config creation."""
        assert basic_config.experiment_id == "test_exp"
        assert basic_config.slice_name == "test_slice"
        assert basic_config.mode == "baseline"
        assert basic_config.total_cycles == 5
        assert basic_config.master_seed == 42
    
    def test_config_to_safe_dict(self, basic_config):
        """Test conversion to safe dictionary."""
        safe_dict = basic_config.to_safe_dict()
        
        assert "experiment_id" in safe_dict
        assert "master_seed" in safe_dict
        assert safe_dict["experiment_id"] == "test_exp"
        assert safe_dict["master_seed"] == 42
    
    def test_config_with_snapshot_dir(self, tmp_path):
        """Test config with snapshot directory."""
        config = U2Config(
            experiment_id="snap_test",
            slice_name="test",
            mode="baseline",
            total_cycles=10,
            master_seed=123,
            snapshot_interval=5,
            snapshot_dir=tmp_path / "snapshots",
        )
        
        assert config.snapshot_interval == 5
        assert config.snapshot_dir == tmp_path / "snapshots"


class TestU2Runner:
    """Tests for U2Runner."""
    
    def test_runner_initialization(self, basic_config):
        """Test runner initialization."""
        runner = U2Runner(basic_config)
        
        assert runner.config == basic_config
        assert runner.cycle_index == 0
        assert len(runner.ht_series) == 0
        assert runner.policy_update_count == 0
    
    def test_runner_baseline_cycle(self, basic_config, mock_execute_fn):
        """Test running a baseline cycle."""
        runner = U2Runner(basic_config)
        items = ["item1", "item2", "item3"]
        
        result = runner.run_cycle(items, mock_execute_fn)
        
        assert isinstance(result, CycleResult)
        assert result.cycle_index == 0
        assert result.slice_name == "test_slice"
        assert result.mode == "baseline"
        assert result.item in items
        assert isinstance(result.success, bool)
        assert result.duration_ms is not None
    
    def test_runner_multiple_cycles(self, basic_config, mock_execute_fn):
        """Test running multiple cycles."""
        runner = U2Runner(basic_config)
        items = ["item1", "item2", "item3"]
        
        results = []
        for _ in range(5):
            result = runner.run_cycle(items, mock_execute_fn)
            results.append(result)
        
        assert len(results) == 5
        assert runner.cycle_index == 5
        assert len(runner.ht_series) == 5
    
    def test_runner_rfl_mode(self, mock_execute_fn):
        """Test runner in RFL mode."""
        config = U2Config(
            experiment_id="rfl_test",
            slice_name="test",
            mode="rfl",
            total_cycles=10,
            master_seed=42,
        )
        runner = U2Runner(config)
        items = ["item1", "item2", "item3"]
        
        # Run several cycles
        for _ in range(10):
            runner.run_cycle(items, mock_execute_fn)
        
        # Check that RFL policy was updated
        assert runner.policy_update_count > 0
        assert len(runner.success_count) > 0
        assert len(runner.attempt_count) > 0


class TestSafetyEnvelope:
    """Tests for safety envelope."""
    
    def test_build_envelope_ok_status(self, basic_config):
        """Test building envelope with OK status."""
        perf_stats = {
            "cycle_durations_ms": [100.0, 150.0, 120.0],
            "max_cycle_duration_ms": 150.0,
            "avg_cycle_duration_ms": 123.3,
        }
        lint_results = []
        
        envelope = build_u2_safety_envelope(basic_config, perf_stats, lint_results)
        
        assert envelope.safety_status == "OK"
        assert envelope.perf_ok
        assert envelope.eval_lint_issues == 0
        assert envelope.schema_version == "1.0.0"
    
    def test_build_envelope_warn_perf(self, basic_config):
        """Test building envelope with WARN status due to performance."""
        perf_stats = {
            "cycle_durations_ms": [6000.0, 7000.0, 5500.0],  # Exceeds threshold
            "max_cycle_duration_ms": 7000.0,
            "avg_cycle_duration_ms": 6166.7,
        }
        lint_results = []
        
        envelope = build_u2_safety_envelope(basic_config, perf_stats, lint_results)
        
        assert envelope.safety_status == "WARN"
        assert not envelope.perf_ok
        assert len(envelope.warnings) > 0
    
    def test_build_envelope_block_unsafe_eval(self, basic_config):
        """Test building envelope with BLOCK status due to unsafe eval."""
        perf_stats = {
            "cycle_durations_ms": [100.0],
            "max_cycle_duration_ms": 100.0,
            "avg_cycle_duration_ms": 100.0,
        }
        lint_results = [
            SafeEvalLintResult(
                is_safe=False,
                issues=["Dangerous operation: Import"],
                dangerous_nodes=["Import"],
                expression="import os",
            )
        ]
        
        envelope = build_u2_safety_envelope(basic_config, perf_stats, lint_results)
        
        assert envelope.safety_status == "BLOCK"
        assert envelope.eval_lint_issues > 0
        assert len(envelope.top_eval_issues) > 0
    
    def test_envelope_to_dict(self, basic_config):
        """Test converting envelope to dictionary."""
        perf_stats = {
            "cycle_durations_ms": [100.0],
            "max_cycle_duration_ms": 100.0,
            "avg_cycle_duration_ms": 100.0,
        }
        envelope = build_u2_safety_envelope(basic_config, perf_stats, [])
        
        envelope_dict = envelope.to_dict()
        
        assert "schema_version" in envelope_dict
        assert "safety_status" in envelope_dict
        assert "perf_ok" in envelope_dict
        assert envelope_dict["safety_status"] == "OK"
    
    def test_evaluate_safety_status(self, basic_config):
        """Test evaluating safety status."""
        perf_stats = {
            "cycle_durations_ms": [100.0],
            "max_cycle_duration_ms": 100.0,
            "avg_cycle_duration_ms": 100.0,
        }
        
        # OK status should pass
        envelope_ok = build_u2_safety_envelope(basic_config, perf_stats, [])
        assert evaluate_safety_status(envelope_ok)
        
        # BLOCK status should fail
        lint_results = [
            SafeEvalLintResult(
                is_safe=False,
                issues=["Dangerous"],
                dangerous_nodes=["Import"],
                expression="import os",
            )
        ]
        envelope_block = build_u2_safety_envelope(basic_config, perf_stats, lint_results)
        assert not evaluate_safety_status(envelope_block)


class TestRunU2Experiment:
    """Tests for run_u2_experiment entry point."""
    
    def test_run_experiment_basic(self, basic_config, mock_execute_fn):
        """Test running a basic experiment."""
        items = ["item1", "item2", "item3"]
        
        results, envelope = run_u2_experiment(
            config=basic_config,
            items=items,
            execute_fn=mock_execute_fn,
        )
        
        assert len(results) == basic_config.total_cycles
        assert all(isinstance(r, CycleResult) for r in results)
        assert isinstance(envelope, U2SafetyEnvelope)
        assert envelope.safety_status in ["OK", "WARN", "BLOCK"]
    
    def test_run_experiment_with_lint(self, basic_config, mock_execute_fn):
        """Test running experiment with expression linting."""
        items = ["item1", "item2", "item3"]
        lint_expressions = ["1 + 1", "2 * 3"]
        
        results, envelope = run_u2_experiment(
            config=basic_config,
            items=items,
            execute_fn=mock_execute_fn,
            lint_expressions=lint_expressions,
        )
        
        assert len(results) == basic_config.total_cycles
        assert envelope.safety_status == "OK"
        assert envelope.eval_lint_issues == 0
    
    def test_run_experiment_performance_tracking(self, basic_config, mock_execute_fn):
        """Test that performance is tracked."""
        items = ["item1", "item2", "item3"]
        
        results, envelope = run_u2_experiment(
            config=basic_config,
            items=items,
            execute_fn=mock_execute_fn,
        )
        
        assert "max_cycle_duration_ms" in envelope.perf_stats
        assert "avg_cycle_duration_ms" in envelope.perf_stats
        assert envelope.perf_stats["cycle_count"] == basic_config.total_cycles


@pytest.mark.unit
class TestU2Integration:
    """Integration tests for U2 infrastructure."""
    
    def test_end_to_end_baseline(self, tmp_path, mock_execute_fn):
        """Test end-to-end baseline experiment."""
        config = U2Config(
            experiment_id="e2e_baseline",
            slice_name="integration_test",
            mode="baseline",
            total_cycles=10,
            master_seed=12345,
            output_dir=tmp_path,
        )
        
        items = ["a", "b", "c", "d", "e"]
        
        results, envelope = run_u2_experiment(
            config=config,
            items=items,
            execute_fn=mock_execute_fn,
        )
        
        assert len(results) == 10
        assert envelope.safety_status in ["OK", "WARN"]
        assert envelope.perf_ok
    
    def test_end_to_end_rfl(self, tmp_path, mock_execute_fn):
        """Test end-to-end RFL experiment."""
        config = U2Config(
            experiment_id="e2e_rfl",
            slice_name="integration_test",
            mode="rfl",
            total_cycles=20,
            master_seed=54321,
            output_dir=tmp_path,
        )
        
        items = ["a", "b", "c", "d", "e"]
        
        results, envelope = run_u2_experiment(
            config=config,
            items=items,
            execute_fn=mock_execute_fn,
        )
        
        assert len(results) == 20
        assert envelope.safety_status in ["OK", "WARN"]
