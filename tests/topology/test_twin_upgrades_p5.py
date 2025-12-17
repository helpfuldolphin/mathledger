from __future__ import annotations

import statistics
from pathlib import Path

from backend.topology.first_light.config_p4 import FirstLightConfigP4
from backend.topology.first_light.runner_p4 import FirstLightShadowRunnerP4, TwinRunner
from backend.topology.first_light.telemetry_adapter import MockTelemetryProvider
from backend.topology.first_light.data_structures_p4 import RealCycleObservation


def test_decoupled_success_improves_success_prediction_accuracy_on_synthetic_trace() -> None:
    base_cfg = FirstLightConfigP4()
    base_cfg.total_cycles = 60
    base_cfg.telemetry_adapter = MockTelemetryProvider(seed=999)

    cfg_baseline = base_cfg
    cfg_decoupled = FirstLightConfigP4()
    cfg_decoupled.total_cycles = 60
    cfg_decoupled.telemetry_adapter = MockTelemetryProvider(seed=999)
    cfg_decoupled.use_decoupled_success = True

    runner_baseline = FirstLightShadowRunnerP4(cfg_baseline, seed=123)
    runner_decoupled = FirstLightShadowRunnerP4(cfg_decoupled, seed=123)

    runner_baseline.run()
    runner_decoupled.run()

    acc_base = runner_baseline._divergence_analyzer.get_summary().success_accuracy  # type: ignore[attr-defined]
    acc_decoupled = runner_decoupled._divergence_analyzer.get_summary().success_accuracy  # type: ignore[attr-defined]

    assert acc_decoupled >= acc_base


def test_per_component_lr_reduces_H_tracking_error() -> None:
    # Synthetic trace where H jumps quickly; per-component LR should track H better.
    observations = [
        RealCycleObservation(
            cycle=i,
            timestamp=str(i),
            runner_type="u2",
            slice_name="arithmetic_simple",
            success=True,
            H=0.2 * i,
            rho=0.7,
            tau=0.2,
            beta=0.1,
            in_omega=True,
            hard_ok=True,
        )
        for i in range(1, 6)
    ]

    twin_default = TwinRunner(learning_rate=0.1, noise_scale=0.0, use_decoupled_success=False)
    twin_overrides = TwinRunner(
        learning_rate=0.1,
        noise_scale=0.0,
        use_decoupled_success=False,
        lr_overrides={"H": 0.5},
    )

    # Initialize from first snapshot for fairness
    twin_default.initialize_from_snapshot(
        RealCycleObservation(
            cycle=0,
            timestamp="0",
            runner_type="u2",
            slice_name="arithmetic_simple",
            success=True,
            H=0.5,
            rho=0.7,
            tau=0.2,
            beta=0.1,
            in_omega=True,
            hard_ok=True,
        )
    )
    twin_overrides.initialize_from_snapshot(
        RealCycleObservation(
            cycle=0,
            timestamp="0",
            runner_type="u2",
            slice_name="arithmetic_simple",
            success=True,
            H=0.5,
            rho=0.7,
            tau=0.2,
            beta=0.1,
            in_omega=True,
            hard_ok=True,
        )
    )

    errors_default = []
    errors_override = []

    for obs in observations:
        twin_default.update_state(obs)
        twin_overrides.update_state(obs)
        errors_default.append(abs(twin_default._H - obs.H))  # type: ignore[attr-defined]
        errors_override.append(abs(twin_overrides._H - obs.H))  # type: ignore[attr-defined]

    mean_default = statistics.mean(errors_default)
    mean_override = statistics.mean(errors_override)

    assert mean_override <= mean_default
