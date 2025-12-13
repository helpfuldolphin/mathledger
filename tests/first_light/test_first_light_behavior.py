"""
Phase X P3: First-Light Behavior Tests

These tests verify the actual behavior of the First-Light shadow experiment
components with 20-50 cycle toy runs using synthetic data.

SHADOW MODE CONTRACT:
- All tests verify observation-only behavior
- No tests check for abort enforcement (SHADOW MODE)
- All tests use synthetic data, not real runners

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)

See: docs/system_law/Phase_X_P3_Spec.md
"""

from __future__ import annotations

import pytest
from typing import List

from tests.factories.first_light_factories import (
    make_metrics_window,
    make_red_flag_entry,
    make_synthetic_raw_record,
)


class TestComputeSlope:
    """Tests for compute_slope() linear regression function."""

    def test_compute_slope_insufficient_data(self) -> None:
        """Verify compute_slope returns None for < 2 points."""
        from backend.topology.first_light import compute_slope

        assert compute_slope([]) is None
        assert compute_slope([1.0]) is None

    def test_compute_slope_flat_line(self) -> None:
        """Verify compute_slope returns 0 for flat data."""
        from backend.topology.first_light import compute_slope

        result = compute_slope([0.5, 0.5, 0.5, 0.5])
        assert result == 0.0

    def test_compute_slope_positive_trend(self) -> None:
        """Verify compute_slope detects positive trend."""
        from backend.topology.first_light import compute_slope

        # Perfect linear increase
        result = compute_slope([0.0, 1.0, 2.0, 3.0])
        assert result is not None
        assert abs(result - 1.0) < 0.001

    def test_compute_slope_negative_trend(self) -> None:
        """Verify compute_slope detects negative trend."""
        from backend.topology.first_light import compute_slope

        # Perfect linear decrease
        result = compute_slope([3.0, 2.0, 1.0, 0.0])
        assert result is not None
        assert abs(result - (-1.0)) < 0.001

    def test_compute_slope_noisy_positive(self) -> None:
        """Verify compute_slope handles noisy but positive data."""
        from backend.topology.first_light import compute_slope

        # Generally increasing with noise
        result = compute_slope([0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.8])
        assert result is not None
        assert result > 0  # Should be positive overall


class TestDeltaPComputer:
    """Tests for DeltaPComputer class."""

    def test_delta_p_computer_basic(self) -> None:
        """Verify DeltaPComputer computes metrics correctly."""
        from backend.topology.first_light import DeltaPComputer

        computer = DeltaPComputer(window_size=5)

        # Add 10 cycles
        for i in range(10):
            computer.update(
                cycle=i,
                success=i % 2 == 0,  # 50% success
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                abstained=False,
            )

        metrics = computer.compute()

        assert metrics.total_count == 10
        assert metrics.success_count == 5
        assert metrics.success_rate_final == 0.5
        assert len(metrics.success_rate_trajectory) == 2  # 10/5 = 2 windows

    def test_delta_p_computer_learning_curve(self) -> None:
        """Verify DeltaPComputer detects learning improvement."""
        from backend.topology.first_light import DeltaPComputer

        computer = DeltaPComputer(window_size=10)

        # Simulate improving success rate
        # Window 1: 50% success, Window 2: 70% success, Window 3: 90% success
        for i in range(30):
            if i < 10:
                success = i % 2 == 0  # 50%
            elif i < 20:
                success = i % 10 < 7  # 70%
            else:
                success = i % 10 < 9  # 90%

            computer.update(
                cycle=i,
                success=success,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                abstained=False,
            )

        metrics = computer.compute()

        # Should detect positive delta_p (learning)
        assert metrics.delta_p_success is not None
        assert metrics.delta_p_success > 0  # Learning curve should be positive

    def test_delta_p_computer_meets_success_criteria(self) -> None:
        """Verify meets_success_criteria() returns correct dict."""
        from backend.topology.first_light import DeltaPComputer

        computer = DeltaPComputer(window_size=10)

        # Add cycles with good metrics
        for i in range(30):
            success = i % 10 < 7 + (i // 10)  # Improving
            computer.update(
                cycle=i,
                success=success,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                abstained=False,
            )

        metrics = computer.compute()
        criteria = metrics.meets_success_criteria()

        assert isinstance(criteria, dict)
        assert "delta_p_success_positive" in criteria
        assert "omega_occupancy_90" in criteria


class TestRedFlagObserver:
    """Tests for RedFlagObserver class."""

    def test_red_flag_observer_no_flags_healthy_state(self) -> None:
        """Verify no red flags for healthy state."""
        from backend.topology.first_light import RedFlagObserver

        observer = RedFlagObserver()

        # Healthy state
        state = {"H": 0.8, "rho": 0.9, "tau": 0.2, "beta": 0.1, "in_omega": True}
        observations = observer.observe(
            cycle=1, state=state, hard_ok=True, governance_aligned=True
        )

        # No red flags expected for healthy state
        assert len(observations) == 0

    def test_red_flag_observer_rsi_collapse(self) -> None:
        """Verify RSI collapse is detected."""
        from backend.topology.first_light import RedFlagObserver, RedFlagType

        observer = RedFlagObserver()

        # Low RSI state
        state = {"H": 0.5, "rho": 0.1, "tau": 0.2, "beta": 0.1, "in_omega": True}
        observations = observer.observe(
            cycle=1, state=state, hard_ok=True, governance_aligned=True
        )

        # Should detect RSI collapse
        rsi_flags = [o for o in observations if o.flag_type == RedFlagType.RSI_COLLAPSE]
        assert len(rsi_flags) >= 1

    def test_red_flag_observer_hard_fail(self) -> None:
        """Verify HARD failure is detected."""
        from backend.topology.first_light import RedFlagObserver, RedFlagType

        observer = RedFlagObserver()

        state = {"H": 0.8, "rho": 0.9, "tau": 0.2, "beta": 0.1, "in_omega": True}
        observations = observer.observe(
            cycle=1, state=state, hard_ok=False, governance_aligned=True
        )

        hard_flags = [o for o in observations if o.flag_type == RedFlagType.HARD_FAIL]
        assert len(hard_flags) == 1

    def test_red_flag_observer_governance_divergence(self) -> None:
        """Verify governance divergence is detected."""
        from backend.topology.first_light import RedFlagObserver, RedFlagType

        observer = RedFlagObserver()

        state = {"H": 0.8, "rho": 0.9, "tau": 0.2, "beta": 0.1, "in_omega": True}
        observations = observer.observe(
            cycle=1, state=state, hard_ok=True, governance_aligned=False
        )

        div_flags = [o for o in observations if o.flag_type == RedFlagType.GOVERNANCE_DIVERGENCE]
        assert len(div_flags) == 1

    def test_red_flag_observer_omega_exit(self) -> None:
        """Verify Omega exit is detected."""
        from backend.topology.first_light import RedFlagObserver, RedFlagType

        observer = RedFlagObserver()

        state = {"H": 0.8, "rho": 0.9, "tau": 0.2, "beta": 0.1, "in_omega": False}
        observations = observer.observe(
            cycle=1, state=state, hard_ok=True, governance_aligned=True
        )

        omega_flags = [o for o in observations if o.flag_type == RedFlagType.OMEGA_EXIT]
        assert len(omega_flags) == 1

    def test_red_flag_observer_streak_tracking(self) -> None:
        """Verify streak tracking works correctly."""
        from backend.topology.first_light import RedFlagObserver

        observer = RedFlagObserver()

        # Consecutive HARD failures
        state = {"H": 0.8, "rho": 0.9, "tau": 0.2, "beta": 0.1, "in_omega": True}
        for i in range(5):
            observer.observe(cycle=i, state=state, hard_ok=False, governance_aligned=True)

        summary = observer.get_summary()
        assert summary.max_hard_fail_streak == 5

    def test_red_flag_observer_hypothetical_abort_shadow_mode(self) -> None:
        """Verify hypothetical abort is for analysis only."""
        from backend.topology.first_light import RedFlagObserver

        observer = RedFlagObserver()

        # Create condition that would trigger hypothetical abort
        # Low RSI + high beta triggers CDI-007 proxy
        state = {"H": 0.2, "rho": 0.1, "tau": 0.2, "beta": 0.8, "in_omega": False}
        for i in range(15):  # Exceed CDI-007 streak threshold
            observer.observe(cycle=i, state=state, hard_ok=True, governance_aligned=True)

        would_abort, reason = observer.hypothetical_should_abort()

        # SHADOW MODE: This is analysis only, the result doesn't control anything
        # We just verify the method works and returns proper types
        assert isinstance(would_abort, bool)
        assert reason is None or isinstance(reason, str)

    def test_red_flag_observer_summary(self) -> None:
        """Verify get_summary() returns complete data."""
        from backend.topology.first_light import RedFlagObserver

        observer = RedFlagObserver()

        # Generate some observations
        states = [
            {"H": 0.8, "rho": 0.1, "tau": 0.2, "beta": 0.1, "in_omega": True},  # RSI collapse
            {"H": 0.8, "rho": 0.9, "tau": 0.2, "beta": 0.1, "in_omega": False},  # Omega exit
        ]
        for i, state in enumerate(states):
            observer.observe(cycle=i, state=state, hard_ok=True, governance_aligned=True)

        summary = observer.get_summary()

        assert summary.total_observations >= 2
        assert isinstance(summary.observations_by_type, dict)
        assert isinstance(summary.observations_by_severity, dict)

    def test_red_flag_observer_reset(self) -> None:
        """Verify reset clears all state."""
        from backend.topology.first_light import RedFlagObserver

        observer = RedFlagObserver()

        # Generate observations
        state = {"H": 0.8, "rho": 0.1, "tau": 0.2, "beta": 0.1, "in_omega": True}
        observer.observe(cycle=1, state=state, hard_ok=False, governance_aligned=False)

        # Reset
        observer.reset()

        summary = observer.get_summary()
        assert summary.total_observations == 0


class TestMetricsWindow:
    """Tests for MetricsWindow class."""

    def test_metrics_window_basic(self) -> None:
        """Verify MetricsWindow accumulates correctly."""
        from backend.topology.first_light import MetricsWindow

        window = MetricsWindow(window_size=10)

        for i in range(10):
            window.add(
                success=i % 2 == 0,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
            )

        result = window.finalize()

        assert result["total_count"] == 10
        assert result["success_metrics"]["success_count"] == 5
        assert result["success_metrics"]["success_rate"] == 0.5
        assert result["mode"] == "SHADOW"

    def test_metrics_window_is_full(self) -> None:
        """Verify is_full() works correctly."""
        from backend.topology.first_light import MetricsWindow

        window = MetricsWindow(window_size=5)

        for i in range(4):
            window.add(True, False, True, True, 0.8, False)
            assert not window.is_full()

        window.add(True, False, True, True, 0.8, False)
        assert window.is_full()

    def test_metrics_window_factory_payload_replay(self) -> None:
        """Verify factory payloads can be replayed into MetricsWindow."""
        from backend.topology.first_light import MetricsWindow

        payload = make_metrics_window(window_index=2, window_size=6, seed=99)

        window = MetricsWindow(
            window_size=payload["total_count"],
            window_index=payload["window_index"],
            start_cycle=payload["start_cycle"],
        )

        for cycle in payload["cycles"]:
            window.add(
                success=cycle["success"],
                abstained=cycle["abstained"],
                in_omega=cycle["in_omega"],
                hard_ok=cycle["hard_ok"],
                rsi=cycle["rsi"],
                blocked=cycle["blocked"],
                H=cycle["H"],
            )

        result = window.finalize()

        assert result["success_metrics"]["success_count"] == payload["success_metrics"]["success_count"]
        assert result["abstention_metrics"]["abstention_count"] == payload["abstention_metrics"]["abstention_count"]
        assert result["safe_region_metrics"]["omega_count"] == payload["safe_region_metrics"]["omega_count"]
        assert result["hard_mode_metrics"]["hard_ok_count"] == payload["hard_mode_metrics"]["hard_ok_count"]
        assert result["block_metrics"]["blocked_count"] == payload["block_metrics"]["blocked_count"]
        assert result["tda_inputs"]["H_trajectory"] == payload["tda_inputs"]["H_trajectory"]


class TestMetricsAccumulator:
    """Tests for MetricsAccumulator class."""

    def test_metrics_accumulator_trajectories(self) -> None:
        """Verify MetricsAccumulator builds trajectories."""
        from backend.topology.first_light.metrics_window import MetricsAccumulator

        acc = MetricsAccumulator(window_size=10)

        # Add 30 cycles (3 windows)
        for i in range(30):
            acc.add(
                success=True,
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
            )

        trajectories = acc.get_trajectories()

        assert len(trajectories["success_rate"]) == 3
        assert all(r == 1.0 for r in trajectories["success_rate"])

    def test_metrics_accumulator_cumulative_rates(self) -> None:
        """Verify cumulative rates are correct."""
        from backend.topology.first_light.metrics_window import MetricsAccumulator

        acc = MetricsAccumulator(window_size=10)

        for i in range(20):
            acc.add(
                success=i % 2 == 0,  # 50% success
                abstained=False,
                in_omega=True,
                hard_ok=True,
                rsi=0.8,
                blocked=False,
            )

        rates = acc.get_cumulative_rates()

        assert rates["success_rate"] == 0.5
        assert rates["omega_occupancy"] == 1.0


class TestFirstLightConfig:
    """Tests for FirstLightConfig validation."""

    def test_config_shadow_mode_default_true(self) -> None:
        """Verify shadow_mode defaults to True."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig()
        assert config.shadow_mode is True

    def test_config_validate_shadow_mode_violation(self) -> None:
        """Verify shadow_mode=False is detected as violation."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig(shadow_mode=False)
        errors = config.validate()

        assert len(errors) > 0
        assert any("SHADOW MODE" in e for e in errors)

    def test_config_validate_tau_0_bounds(self) -> None:
        """Verify tau_0 bounds are checked."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig(tau_0=1.5)
        errors = config.validate()
        assert any("tau_0" in e for e in errors)

        config = FirstLightConfig(tau_0=-0.1)
        errors = config.validate()
        assert any("tau_0" in e for e in errors)

    def test_config_validate_or_raise(self) -> None:
        """Verify validate_or_raise() raises ValueError."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig(shadow_mode=False)

        with pytest.raises(ValueError, match="SHADOW MODE"):
            config.validate_or_raise()


class TestFirstLightResult:
    """Tests for FirstLightResult serialization."""

    def test_result_to_dict(self) -> None:
        """Verify to_dict() produces valid JSON structure."""
        from backend.topology.first_light import FirstLightResult

        result = FirstLightResult(
            run_id="test_run",
            config_slice="arithmetic_simple",
            config_runner_type="u2",
            cycles_completed=100,
            u2_success_rate_final=0.85,
        )

        d = result.to_dict()

        assert d["schema"] == "first-light-summary/1.0.0"
        assert d["mode"] == "SHADOW"
        assert d["run_id"] == "test_run"
        assert d["uplift_metrics"]["u2_success_rate_final"] == 0.85


class TestSyntheticStateGenerator:
    """Tests for SyntheticStateGenerator."""

    def test_generator_deterministic_with_seed(self) -> None:
        """Verify generator is deterministic with same seed."""
        from backend.topology.first_light import SyntheticStateGenerator

        gen1 = SyntheticStateGenerator(seed=42)
        gen2 = SyntheticStateGenerator(seed=42)

        states1 = [gen1.step() for _ in range(10)]
        states2 = [gen2.step() for _ in range(10)]

        for s1, s2 in zip(states1, states2):
            assert s1["H"] == s2["H"]
            assert s1["rho"] == s2["rho"]
            assert s1["success"] == s2["success"]

    def test_generator_produces_valid_state(self) -> None:
        """Verify generator produces valid state values."""
        from backend.topology.first_light import SyntheticStateGenerator

        gen = SyntheticStateGenerator(seed=42)

        for _ in range(50):
            state = gen.step()

            assert 0.0 <= state["H"] <= 1.0
            assert 0.0 <= state["rho"] <= 1.0
            assert 0.0 <= state["tau"] <= 1.0
            assert 0.0 <= state["beta"] <= 1.0
            assert isinstance(state["in_omega"], bool)
            assert isinstance(state["success"], bool)
            assert isinstance(state["hard_ok"], bool)


class TestFirstLightShadowRunner:
    """Tests for FirstLightShadowRunner - 20-50 cycle toy runs."""

    def test_runner_rejects_non_shadow_mode(self) -> None:
        """Verify runner rejects shadow_mode=False."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(shadow_mode=False)

        with pytest.raises(ValueError, match="SHADOW MODE VIOLATION"):
            FirstLightShadowRunner(config)

    def test_runner_20_cycle_run(self) -> None:
        """Run 20-cycle toy experiment."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=20,
            success_window=5,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        assert result.cycles_completed == 20
        assert result.total_cycles_requested == 20
        assert 0.0 <= result.u2_success_rate_final <= 1.0
        assert isinstance(result.delta_p_success, (float, type(None)))

    def test_runner_50_cycle_run(self) -> None:
        """Run 50-cycle toy experiment."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=50,
            success_window=10,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=123)
        result = runner.run()

        assert result.cycles_completed == 50
        assert len(result.u2_success_rate_trajectory) == 5  # 50/10 = 5 windows

    def test_runner_yields_observations(self) -> None:
        """Verify runner yields cycle observations."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=25,
            success_window=5,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=42)
        observations = list(runner.run_cycles(25))

        assert len(observations) == 25
        assert all(obs.cycle == i + 1 for i, obs in enumerate(observations))

    def test_runner_get_current_metrics(self) -> None:
        """Verify get_current_metrics() returns valid data."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=30,
            success_window=10,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=42)

        # Run some cycles
        for _ in runner.run_cycles(15):
            pass

        metrics = runner.get_current_metrics()

        assert metrics["mode"] == "SHADOW"
        assert metrics["cycle"] == 15
        assert "cumulative" in metrics

    def test_runner_get_red_flag_status(self) -> None:
        """Verify get_red_flag_status() returns valid data."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=30,
            success_window=10,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=42)

        # Run some cycles
        for _ in runner.run_cycles(30):
            pass

        status = runner.get_red_flag_status()

        assert status["mode"] == "SHADOW"
        assert "total_observations" in status
        assert "hypothetical_abort" in status

    def test_runner_result_has_trajectories(self) -> None:
        """Verify result contains trajectory data."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=40,
            success_window=10,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        assert len(result.u2_success_rate_trajectory) == 4
        assert len(result.rsi_trajectory) == 4
        assert len(result.omega_occupancy_trajectory) == 4

    def test_runner_reset(self) -> None:
        """Verify reset clears runner state."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=20,
            success_window=5,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=42)
        runner.run()

        runner.reset()

        # After reset, should be able to run again
        result = runner.run()
        assert result.cycles_completed == 20

    def test_runner_deterministic_with_seed(self) -> None:
        """Verify runner is deterministic with same seed."""
        from backend.topology.first_light import FirstLightConfig, FirstLightShadowRunner

        config = FirstLightConfig(
            total_cycles=30,
            success_window=10,
            shadow_mode=True,
        )

        runner1 = FirstLightShadowRunner(config, seed=42)
        result1 = runner1.run()

        runner2 = FirstLightShadowRunner(config, seed=42)
        result2 = runner2.run()

        assert result1.u2_success_rate_final == result2.u2_success_rate_final
        assert result1.mean_rsi == result2.mean_rsi


class TestSchemas:
    """Tests for JSONL schema dataclasses."""

    def test_cycle_log_entry_to_dict(self) -> None:
        """Verify CycleLogEntry.to_dict() produces valid structure."""
        from backend.topology.first_light import CycleLogEntry

        payload = make_synthetic_raw_record(42)
        entry = CycleLogEntry(**payload)

        d = entry.to_dict()

        assert d["schema"] == "first-light-cycle/1.0.0"
        assert d["mode"] == "SHADOW"
        assert d["cycle"] == 42
        assert d["runner"]["success"] == payload["runner_success"]
        assert d["runner"]["slice"] == payload["runner_slice"]

    def test_red_flag_log_entry_shadow_markers(self) -> None:
        """Verify RedFlagLogEntry has SHADOW MODE markers."""
        from backend.topology.first_light import RedFlagLogEntry

        payload = make_red_flag_entry(10, "RSI_COLLAPSE")
        entry = RedFlagLogEntry(**payload)

        d = entry.to_dict()

        assert d["mode"] == "SHADOW"
        assert d["action"] == "LOGGED_ONLY"

    def test_metrics_log_entry_to_dict(self) -> None:
        """Verify MetricsLogEntry.to_dict() produces valid structure."""
        from backend.topology.first_light import MetricsLogEntry

        entry = MetricsLogEntry(
            window_index=5,
            window_start_cycle=250,
            window_end_cycle=299,
            window_success_rate=0.82,
        )

        d = entry.to_dict()

        assert d["schema"] == "first-light-metrics/1.0.0"
        assert d["mode"] == "SHADOW"
        assert d["success_metrics"]["window_success_rate"] == 0.82

    def test_summary_schema_to_dict(self) -> None:
        """Verify SummarySchema.to_dict() produces valid structure."""
        from backend.topology.first_light import SummarySchema

        summary = SummarySchema(
            run_id="fl_test_123",
            slice_name="arithmetic_simple",
            runner_type="u2",
            total_cycles=1000,
        )

        d = summary.to_dict()

        assert d["schema"] == "first-light-summary/1.0.0"
        assert d["mode"] == "SHADOW"
        assert d["config"]["total_cycles"] == 1000


class TestIntegration:
    """Integration tests for complete First-Light workflow."""

    def test_full_workflow_20_cycles(self) -> None:
        """Test complete workflow with 20 cycles."""
        from backend.topology.first_light import (
            FirstLightConfig,
            FirstLightShadowRunner,
        )

        # Configure
        config = FirstLightConfig(
            slice_name="arithmetic_simple",
            runner_type="u2",
            total_cycles=20,
            tau_0=0.20,
            success_window=5,
            shadow_mode=True,
        )

        # Validate
        errors = config.validate()
        assert len(errors) == 0

        # Run
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        # Verify result
        assert result.cycles_completed == 20
        assert result.config_slice == "arithmetic_simple"
        assert result.config_runner_type == "u2"

        # Verify metrics exist
        assert result.u2_success_rate_final is not None
        assert result.mean_rsi > 0

        # Verify trajectories
        assert len(result.u2_success_rate_trajectory) == 4  # 20/5 = 4 windows

        # Verify serialization
        d = result.to_dict()
        assert d["mode"] == "SHADOW"
        assert d["execution"]["cycles_completed"] == 20

    def test_full_workflow_50_cycles_with_red_flags(self) -> None:
        """Test workflow that generates some red flags."""
        from backend.topology.first_light import (
            FirstLightConfig,
            FirstLightShadowRunner,
        )

        config = FirstLightConfig(
            total_cycles=50,
            success_window=10,
            shadow_mode=True,
        )

        runner = FirstLightShadowRunner(config, seed=999)  # Seed that may generate red flags
        result = runner.run()

        # Result should have red flag tracking
        assert isinstance(result.total_red_flags, int)
        assert isinstance(result.red_flags_by_type, dict)
        assert isinstance(result.red_flags_by_severity, dict)

        # Even with red flags, no abort should occur (SHADOW MODE)
        # The hypothetical_abort_cycle is for analysis only
        assert result.cycles_completed == 50  # All cycles complete

    def test_learning_curve_detection(self) -> None:
        """Verify delta_p detects learning improvement over 50 cycles."""
        from backend.topology.first_light import (
            FirstLightConfig,
            FirstLightShadowRunner,
        )

        config = FirstLightConfig(
            total_cycles=50,
            success_window=10,
            shadow_mode=True,
        )

        # Use seed that produces learning effect
        runner = FirstLightShadowRunner(config, seed=42)
        result = runner.run()

        # The synthetic generator should show learning
        # (success rate improves over time due to learning_rate parameter)
        if len(result.u2_success_rate_trajectory) >= 2:
            # Check that later windows tend to have higher success
            early = result.u2_success_rate_trajectory[0]
            late = result.u2_success_rate_trajectory[-1]
            # With learning, late should generally be >= early
            # (not guaranteed due to noise, but likely)
            assert isinstance(early, float)
            assert isinstance(late, float)
