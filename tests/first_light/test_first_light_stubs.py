"""
Phase X P3: First-Light Import and Signature Tests

These tests verify that all First-Light modules import correctly and
all classes/functions exist with expected signatures.

Note: The "stub" tests that expected NotImplementedError have been
replaced by the behavior tests in test_first_light_behavior.py now
that P3 implementation is complete.

SHADOW MODE CONTRACT:
- No governance control or abort logic is tested
- Tests verify import compatibility and SHADOW MODE markers

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)

See: docs/system_law/Phase_X_P3_Spec.md
"""

from __future__ import annotations

import pytest


class TestFirstLightImports:
    """
    Phase X P3: Import smoke tests.

    These tests verify that modules can be imported.
    """

    def test_import_config_module(self) -> None:
        """Verify config module imports without error."""
        from backend.topology.first_light import config

        assert config is not None

    def test_import_runner_module(self) -> None:
        """Verify runner module imports without error."""
        from backend.topology.first_light import runner

        assert runner is not None

    def test_import_red_flag_observer_module(self) -> None:
        """Verify red_flag_observer module imports without error."""
        from backend.topology.first_light import red_flag_observer

        assert red_flag_observer is not None

    def test_import_delta_p_computer_module(self) -> None:
        """Verify delta_p_computer module imports without error."""
        from backend.topology.first_light import delta_p_computer

        assert delta_p_computer is not None

    def test_import_metrics_window_module(self) -> None:
        """Verify metrics_window module imports without error."""
        from backend.topology.first_light import metrics_window

        assert metrics_window is not None

    def test_import_schemas_module(self) -> None:
        """Verify schemas module imports without error."""
        from backend.topology.first_light import schemas

        assert schemas is not None


class TestFirstLightClassesExist:
    """
    Phase X P3: Class existence tests.

    These tests verify that classes can be imported.
    """

    def test_first_light_config_exists(self) -> None:
        """Verify FirstLightConfig class exists."""
        from backend.topology.first_light import FirstLightConfig

        assert FirstLightConfig is not None

    def test_first_light_result_exists(self) -> None:
        """Verify FirstLightResult class exists."""
        from backend.topology.first_light import FirstLightResult

        assert FirstLightResult is not None

    def test_first_light_shadow_runner_exists(self) -> None:
        """Verify FirstLightShadowRunner class exists."""
        from backend.topology.first_light import FirstLightShadowRunner

        assert FirstLightShadowRunner is not None

    def test_red_flag_type_exists(self) -> None:
        """Verify RedFlagType enum exists."""
        from backend.topology.first_light import RedFlagType

        assert RedFlagType is not None
        # Verify it's an enum with expected values
        assert hasattr(RedFlagType, "CDI_010")
        assert hasattr(RedFlagType, "RSI_COLLAPSE")

    def test_red_flag_severity_exists(self) -> None:
        """Verify RedFlagSeverity enum exists."""
        from backend.topology.first_light import RedFlagSeverity

        assert RedFlagSeverity is not None
        assert hasattr(RedFlagSeverity, "INFO")
        assert hasattr(RedFlagSeverity, "WARNING")
        assert hasattr(RedFlagSeverity, "CRITICAL")

    def test_red_flag_observation_exists(self) -> None:
        """Verify RedFlagObservation dataclass exists."""
        from backend.topology.first_light import RedFlagObservation

        assert RedFlagObservation is not None

    def test_red_flag_summary_exists(self) -> None:
        """Verify RedFlagSummary dataclass exists."""
        from backend.topology.first_light import RedFlagSummary

        assert RedFlagSummary is not None

    def test_red_flag_observer_exists(self) -> None:
        """Verify RedFlagObserver class exists."""
        from backend.topology.first_light import RedFlagObserver

        assert RedFlagObserver is not None

    def test_delta_p_metrics_exists(self) -> None:
        """Verify DeltaPMetrics dataclass exists."""
        from backend.topology.first_light import DeltaPMetrics

        assert DeltaPMetrics is not None

    def test_delta_p_computer_exists(self) -> None:
        """Verify DeltaPComputer class exists."""
        from backend.topology.first_light import DeltaPComputer

        assert DeltaPComputer is not None

    def test_compute_slope_exists(self) -> None:
        """Verify compute_slope function exists."""
        from backend.topology.first_light import compute_slope

        assert compute_slope is not None
        assert callable(compute_slope)

    def test_metrics_window_exists(self) -> None:
        """Verify MetricsWindow class exists."""
        from backend.topology.first_light import MetricsWindow

        assert MetricsWindow is not None


class TestFirstLightSchemasExist:
    """
    Phase X P3: Schema existence tests.

    These tests verify that schema classes exist.
    """

    def test_cycle_log_entry_exists(self) -> None:
        """Verify CycleLogEntry schema exists."""
        from backend.topology.first_light import CycleLogEntry

        assert CycleLogEntry is not None

    def test_red_flag_log_entry_exists(self) -> None:
        """Verify RedFlagLogEntry schema exists."""
        from backend.topology.first_light import RedFlagLogEntry

        assert RedFlagLogEntry is not None

    def test_metrics_log_entry_exists(self) -> None:
        """Verify MetricsLogEntry schema exists."""
        from backend.topology.first_light import MetricsLogEntry

        assert MetricsLogEntry is not None

    def test_summary_schema_exists(self) -> None:
        """Verify SummarySchema exists."""
        from backend.topology.first_light import SummarySchema

        assert SummarySchema is not None

    def test_schema_versions_defined(self) -> None:
        """Verify schema version constants are defined."""
        from backend.topology.first_light.schemas import (
            CYCLE_LOG_SCHEMA_VERSION,
            RED_FLAG_LOG_SCHEMA_VERSION,
            METRICS_LOG_SCHEMA_VERSION,
            SUMMARY_SCHEMA_VERSION,
        )

        assert CYCLE_LOG_SCHEMA_VERSION == "first-light-cycle/1.0.0"
        assert RED_FLAG_LOG_SCHEMA_VERSION == "first-light-red-flag/1.0.0"
        assert METRICS_LOG_SCHEMA_VERSION == "first-light-metrics/1.0.0"
        assert SUMMARY_SCHEMA_VERSION == "first-light-summary/1.0.0"


class TestFirstLightDataclassInstantiation:
    """
    Phase X P3: Dataclass instantiation tests.

    These tests verify dataclasses can be instantiated with defaults.
    """

    def test_first_light_config_instantiation(self) -> None:
        """Verify FirstLightConfig can be instantiated with defaults."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig()
        assert config is not None
        assert config.shadow_mode is True  # Must always be True

    def test_first_light_result_instantiation(self) -> None:
        """Verify FirstLightResult can be instantiated with defaults."""
        from backend.topology.first_light import FirstLightResult

        result = FirstLightResult()
        assert result is not None

    def test_red_flag_observation_instantiation(self) -> None:
        """Verify RedFlagObservation can be instantiated."""
        from backend.topology.first_light import (
            RedFlagObservation,
            RedFlagType,
            RedFlagSeverity,
        )

        obs = RedFlagObservation()
        assert obs is not None
        assert obs.action_taken == "LOGGED_ONLY"  # SHADOW MODE marker

    def test_red_flag_summary_instantiation(self) -> None:
        """Verify RedFlagSummary can be instantiated."""
        from backend.topology.first_light import RedFlagSummary

        summary = RedFlagSummary()
        assert summary is not None

    def test_delta_p_metrics_instantiation(self) -> None:
        """Verify DeltaPMetrics can be instantiated."""
        from backend.topology.first_light import DeltaPMetrics

        metrics = DeltaPMetrics()
        assert metrics is not None

    def test_metrics_window_instantiation(self) -> None:
        """Verify MetricsWindow can be instantiated."""
        from backend.topology.first_light import MetricsWindow

        window = MetricsWindow()
        assert window is not None

    def test_cycle_log_entry_instantiation(self) -> None:
        """Verify CycleLogEntry can be instantiated."""
        from backend.topology.first_light import CycleLogEntry

        entry = CycleLogEntry()
        assert entry is not None
        assert entry.mode == "SHADOW"  # SHADOW MODE marker

    def test_red_flag_log_entry_instantiation(self) -> None:
        """Verify RedFlagLogEntry can be instantiated."""
        from backend.topology.first_light import RedFlagLogEntry

        entry = RedFlagLogEntry()
        assert entry is not None
        assert entry.mode == "SHADOW"  # SHADOW MODE marker
        assert entry.action == "LOGGED_ONLY"  # SHADOW MODE marker


class TestFirstLightConfigValidation:
    """
    Phase X P3: Config validation tests.

    These tests verify basic config validation works.
    """

    def test_config_validate_returns_list(self) -> None:
        """Verify validate() returns a list."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig()
        errors = config.validate()
        assert isinstance(errors, list)

    def test_config_shadow_mode_violation_detected(self) -> None:
        """Verify shadow_mode=False is detected as violation."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig(shadow_mode=False)
        errors = config.validate()
        assert len(errors) > 0
        assert any("SHADOW MODE" in e for e in errors)


class TestFirstLightShadowModeEnforcement:
    """
    Phase X P3: SHADOW MODE enforcement tests.

    These tests verify that SHADOW MODE is enforced.
    """

    def test_runner_rejects_non_shadow_mode(self) -> None:
        """Verify FirstLightShadowRunner rejects shadow_mode=False."""
        from backend.topology.first_light import (
            FirstLightConfig,
            FirstLightShadowRunner,
        )

        config = FirstLightConfig(shadow_mode=False)

        with pytest.raises(ValueError, match="SHADOW MODE VIOLATION"):
            FirstLightShadowRunner(config)

    def test_runner_accepts_shadow_mode(self) -> None:
        """Verify FirstLightShadowRunner accepts shadow_mode=True."""
        from backend.topology.first_light import (
            FirstLightConfig,
            FirstLightShadowRunner,
        )

        config = FirstLightConfig(shadow_mode=True, total_cycles=10)
        runner = FirstLightShadowRunner(config, seed=42)
        assert runner is not None

    def test_red_flag_observer_methods_work(self) -> None:
        """Verify RedFlagObserver methods work (P3 implementation)."""
        from backend.topology.first_light import RedFlagObserver

        observer = RedFlagObserver()

        # Methods should work, not raise NotImplementedError
        state = {"H": 0.8, "rho": 0.9, "tau": 0.2, "beta": 0.1, "in_omega": True}
        observations = observer.observe(cycle=1, state=state, hard_ok=True, governance_aligned=True)
        assert isinstance(observations, list)

        would_abort, reason = observer.hypothetical_should_abort()
        assert isinstance(would_abort, bool)

        summary = observer.get_summary()
        assert summary is not None

        observer.reset()
        assert observer.get_summary().total_observations == 0

    def test_delta_p_computer_methods_work(self) -> None:
        """Verify DeltaPComputer methods work (P3 implementation)."""
        from backend.topology.first_light import DeltaPComputer

        computer = DeltaPComputer()

        # Methods should work, not raise NotImplementedError
        computer.update(cycle=1, success=True, in_omega=True, hard_ok=True, rsi=0.8)

        metrics = computer.compute()
        assert metrics is not None

        # get_trajectory_point raises IndexError for invalid index, not NotImplementedError
        with pytest.raises(IndexError):
            computer.get_trajectory_point(999)

    def test_compute_slope_works(self) -> None:
        """Verify compute_slope works for real input (P3 implementation)."""
        from backend.topology.first_light import compute_slope

        # Should return None for insufficient data
        result = compute_slope([])
        assert result is None

        result = compute_slope([1.0])
        assert result is None

        # Should compute real slope for valid input
        result = compute_slope([1.0, 2.0, 3.0])
        assert result is not None
        assert abs(result - 1.0) < 0.001
