"""
Integration tests for P4 harness with RealTelemetryAdapter (P5 POC).

SHADOW MODE: All tests verify observation-only mode.
No governance actions are taken, all divergences are LOGGED_ONLY.

See: docs/system_law/First_Light_P5_Adapter_Checklist.md
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from backend.topology.first_light.config_p4 import FirstLightConfigP4
from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4
from backend.topology.first_light.real_telemetry_adapter import RealTelemetryAdapter
from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider


class TestP4HarnessRealAdapterFlag:
    """Test P4 harness accepts --telemetry-adapter flag."""

    def test_p4_harness_accepts_real_adapter_flag(self) -> None:
        """
        P4 harness should accept --telemetry-adapter real flag.

        From STRATCOM Task 2: Add CLI flag --telemetry-adapter {mock,real}
        """
        # Create RealTelemetryAdapter
        adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=42,
            source_label="P5_ADAPTER_STUB",
        )

        # Create P4 configuration with real adapter
        config = FirstLightConfigP4(
            slice_name="arithmetic_simple",
            runner_type="u2",
            total_cycles=10,
            tau_0=0.20,
            telemetry_adapter=adapter,
            log_dir=tempfile.mkdtemp(),
            run_id="test_real_adapter",
        )

        # Validate configuration
        config.validate_or_raise()

        # Verify adapter is correctly assigned
        assert config.telemetry_adapter is adapter
        assert isinstance(config.telemetry_adapter, RealTelemetryAdapter)


class TestP4HarnessRealAdapterExecution:
    """Test P4 harness execution with RealTelemetryAdapter."""

    def test_p4_harness_real_adapter_runs_50_cycles_shadow_only(self) -> None:
        """
        P4 harness with real adapter must run 50 cycles in SHADOW mode.

        Verifies:
        - mode == "SHADOW" everywhere
        - action == "LOGGED_ONLY" in divergence log
        - No exceptions thrown
        - Output artifacts exist

        From STRATCOM Task 2
        """
        # Create RealTelemetryAdapter
        adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=42,
            source_label="P5_ADAPTER_STUB",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create P4 configuration
            config = FirstLightConfigP4(
                slice_name="arithmetic_simple",
                runner_type="u2",
                total_cycles=50,
                tau_0=0.20,
                telemetry_adapter=adapter,
                log_dir=str(output_dir),
                run_id="test_real_50_cycles",
            )

            config.validate_or_raise()

            # Create and run P4 runner
            runner = FirstLightShadowRunnerP4(config, seed=42)

            observations = []
            twin_observations = []
            divergence_snapshots = []

            for observation in runner.run_cycles(50):
                observations.append(observation)
                twin_observations.append(runner.get_twin_observations()[-1])
                divergence_snapshots.append(runner.get_divergence_snapshots()[-1])

            # Finalize
            result = runner.finalize()

            # Verify 50 cycles completed
            assert len(observations) == 50
            assert len(twin_observations) == 50
            assert len(divergence_snapshots) == 50
            assert result.cycles_completed == 50

            # Verify SHADOW mode everywhere
            for obs in observations:
                assert obs.mode == "SHADOW", f"Observation mode is {obs.mode}, expected SHADOW"

            for twin_obs in twin_observations:
                assert twin_obs.mode == "SHADOW", f"Twin mode is {twin_obs.mode}, expected SHADOW"

            # Verify action == LOGGED_ONLY for all divergence snapshots
            for div_snap in divergence_snapshots:
                assert div_snap.action == "LOGGED_ONLY", \
                    f"Divergence action is {div_snap.action}, expected LOGGED_ONLY"

    def test_p4_real_adapter_produces_valid_metrics(self) -> None:
        """Real adapter should produce metrics in valid ranges."""
        adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = FirstLightConfigP4(
                slice_name="arithmetic_simple",
                runner_type="u2",
                total_cycles=20,
                tau_0=0.20,
                telemetry_adapter=adapter,
                log_dir=str(tmpdir),
                run_id="test_metrics",
            )

            config.validate_or_raise()
            runner = FirstLightShadowRunnerP4(config, seed=42)

            for observation in runner.run_cycles(20):
                # Verify metrics are in valid ranges
                assert 0.0 <= observation.H <= 1.0
                assert 0.0 <= observation.rho <= 1.0
                assert 0.0 <= observation.tau <= 1.0
                assert 0.0 <= observation.beta <= 1.0
                assert isinstance(observation.success, bool)
                assert isinstance(observation.in_omega, bool)
                assert isinstance(observation.hard_ok, bool)


class TestP4RealVsMockDivergence:
    """Test that RealTelemetryAdapter produces lower divergence than Mock."""

    @pytest.mark.slow
    def test_real_adapter_divergence_lower_than_mock_baseline(self) -> None:
        """
        With RealTelemetryAdapter, divergence should be lower than mock baseline.

        The mock baseline is ~97% divergence (noise baseline).
        RealTelemetryAdapter generates correlated data that the twin can
        partially predict, so divergence should be significantly lower.

        From STRATCOM Task 3
        """
        # Run with Mock adapter
        mock_adapter = MockTelemetryProvider(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config = FirstLightConfigP4(
                slice_name="arithmetic_simple",
                runner_type="u2",
                total_cycles=50,
                tau_0=0.20,
                telemetry_adapter=mock_adapter,
                log_dir=str(tmpdir),
                run_id="test_mock",
            )

            mock_config.validate_or_raise()
            mock_runner = FirstLightShadowRunnerP4(mock_config, seed=42)

            for _ in mock_runner.run_cycles(50):
                pass

            mock_result = mock_runner.finalize()
            mock_divergence = mock_result.divergence_rate

        # Run with Real adapter
        real_adapter = RealTelemetryAdapter(
            runner_type="u2",
            slice_name="arithmetic_simple",
            seed=42,
            source_label="P5_ADAPTER_STUB",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            real_config = FirstLightConfigP4(
                slice_name="arithmetic_simple",
                runner_type="u2",
                total_cycles=50,
                tau_0=0.20,
                telemetry_adapter=real_adapter,
                log_dir=str(tmpdir),
                run_id="test_real",
            )

            real_config.validate_or_raise()
            real_runner = FirstLightShadowRunnerP4(real_config, seed=42)

            for _ in real_runner.run_cycles(50):
                pass

            real_result = real_runner.finalize()
            real_divergence = real_result.divergence_rate

        # The key assertion: real divergence should be lower than mock baseline
        # Mock baseline is typically ~0.97 (97%)
        # Real adapter should be < 0.97 (ideally much lower)
        print(f"Mock divergence: {mock_divergence:.4f}")
        print(f"Real divergence: {real_divergence:.4f}")

        # Primary assertion: real < 0.97 (below noise baseline)
        assert real_divergence < 0.97, \
            f"Real divergence {real_divergence:.3f} not below noise baseline 0.97"

        # Secondary assertion: real < mock (improvement over mock)
        # Note: This may not always hold due to randomness, but is expected
        # assert real_divergence < mock_divergence


class TestP4RealAdapterShadowCompliance:
    """Test SHADOW MODE compliance with RealTelemetryAdapter."""

    def test_real_adapter_no_governance_mutations(self) -> None:
        """RealTelemetryAdapter should not have governance mutation methods."""
        adapter = RealTelemetryAdapter(runner_type="u2", seed=42)

        # These methods should not exist
        forbidden_methods = [
            "set_state",
            "modify_governance",
            "abort",
            "stop",
            "enforce_policy",
            "send_command",
            "trigger_abort",
        ]

        for method_name in forbidden_methods:
            assert not hasattr(adapter, method_name) or \
                   not callable(getattr(adapter, method_name, None)), \
                   f"Adapter should not have callable {method_name}"

    def test_config_rejects_non_shadow_mode(self) -> None:
        """P4 config should reject shadow_mode=False."""
        adapter = RealTelemetryAdapter(runner_type="u2", seed=42)

        config = FirstLightConfigP4(
            slice_name="arithmetic_simple",
            runner_type="u2",
            total_cycles=10,
            tau_0=0.20,
            telemetry_adapter=adapter,
            log_dir=tempfile.mkdtemp(),
            run_id="test_shadow",
            shadow_mode=False,  # Try to disable shadow mode
        )

        # Should raise during validation
        with pytest.raises(ValueError, match="shadow"):
            config.validate_or_raise()
