"""
Phase X P4: Stub Tests for Design Verification

These tests verify that all P4 modules import correctly and all
classes/functions exist with expected signatures. All implementation
methods MUST raise NotImplementedError until P4 is authorized.

SHADOW MODE CONTRACT:
- No governance control or abort logic is tested
- Tests verify import compatibility and stub status
- Tests verify SHADOW MODE markers are present

Status: P4 DESIGN FREEZE (STUBS ONLY)

See: docs/system_law/Phase_X_P4_Spec.md
"""

from __future__ import annotations

import pytest

from tests.factories.first_light_factories import (
    make_divergence_entry,
    make_tda_window,
    make_real_telemetry_snapshot,
)


class TestP4ModuleImports:
    """
    Phase X P4: Import smoke tests.

    These tests verify that P4 modules can be imported.
    """

    def test_import_config_p4_module(self) -> None:
        """Verify config_p4 module imports without error."""
        from backend.topology.first_light import config_p4

        assert config_p4 is not None

    def test_import_runner_p4_module(self) -> None:
        """Verify runner_p4 module imports without error."""
        from backend.topology.first_light import runner_p4

        assert runner_p4 is not None

    def test_import_telemetry_adapter_module(self) -> None:
        """Verify telemetry_adapter module imports without error."""
        from backend.topology.first_light import telemetry_adapter

        assert telemetry_adapter is not None

    def test_import_divergence_analyzer_module(self) -> None:
        """Verify divergence_analyzer module imports without error."""
        from backend.topology.first_light import divergence_analyzer

        assert divergence_analyzer is not None

    def test_import_data_structures_p4_module(self) -> None:
        """Verify data_structures_p4 module imports without error."""
        from backend.topology.first_light import data_structures_p4

        assert data_structures_p4 is not None

    def test_import_schemas_p4_module(self) -> None:
        """Verify schemas_p4 module imports without error."""
        from backend.topology.first_light import schemas_p4

        assert schemas_p4 is not None


class TestP4ClassesExist:
    """
    Phase X P4: Class existence tests.

    These tests verify that P4 classes can be imported.
    """

    def test_first_light_config_p4_exists(self) -> None:
        """Verify FirstLightConfigP4 class exists."""
        from backend.topology.first_light import FirstLightConfigP4

        assert FirstLightConfigP4 is not None

    def test_first_light_result_p4_exists(self) -> None:
        """Verify FirstLightResultP4 class exists."""
        from backend.topology.first_light import FirstLightResultP4

        assert FirstLightResultP4 is not None

    def test_first_light_shadow_runner_p4_exists(self) -> None:
        """Verify FirstLightShadowRunnerP4 class exists."""
        from backend.topology.first_light import FirstLightShadowRunnerP4

        assert FirstLightShadowRunnerP4 is not None

    def test_twin_runner_exists(self) -> None:
        """Verify TwinRunner class exists."""
        from backend.topology.first_light import TwinRunner

        assert TwinRunner is not None

    def test_telemetry_provider_interface_exists(self) -> None:
        """Verify TelemetryProviderInterface class exists."""
        from backend.topology.first_light import TelemetryProviderInterface

        assert TelemetryProviderInterface is not None

    def test_usla_integration_adapter_exists(self) -> None:
        """Verify USLAIntegrationAdapter class exists."""
        from backend.topology.first_light import USLAIntegrationAdapter

        assert USLAIntegrationAdapter is not None

    def test_mock_telemetry_provider_exists(self) -> None:
        """Verify MockTelemetryProvider class exists."""
        from backend.topology.first_light import MockTelemetryProvider

        assert MockTelemetryProvider is not None

    def test_divergence_analyzer_exists(self) -> None:
        """Verify DivergenceAnalyzer class exists."""
        from backend.topology.first_light import DivergenceAnalyzer

        assert DivergenceAnalyzer is not None

    def test_divergence_summary_exists(self) -> None:
        """Verify DivergenceSummary class exists."""
        from backend.topology.first_light import DivergenceSummary

        assert DivergenceSummary is not None

    def test_divergence_thresholds_exists(self) -> None:
        """Verify DivergenceThresholds class exists."""
        from backend.topology.first_light import DivergenceThresholds

        assert DivergenceThresholds is not None


class TestP4DataStructuresExist:
    """
    Phase X P4: Data structure existence tests.
    """

    def test_telemetry_snapshot_exists(self) -> None:
        """Verify TelemetrySnapshot class exists."""
        from backend.topology.first_light import TelemetrySnapshot

        assert TelemetrySnapshot is not None

    def test_real_cycle_observation_exists(self) -> None:
        """Verify RealCycleObservation class exists."""
        from backend.topology.first_light import RealCycleObservation

        assert RealCycleObservation is not None

    def test_twin_cycle_observation_exists(self) -> None:
        """Verify TwinCycleObservation class exists."""
        from backend.topology.first_light import TwinCycleObservation

        assert TwinCycleObservation is not None

    def test_divergence_snapshot_exists(self) -> None:
        """Verify DivergenceSnapshot class exists."""
        from backend.topology.first_light import DivergenceSnapshot

        assert DivergenceSnapshot is not None


class TestP4SchemasExist:
    """
    Phase X P4: Schema existence tests.
    """

    def test_real_cycle_log_entry_exists(self) -> None:
        """Verify RealCycleLogEntry class exists."""
        from backend.topology.first_light import RealCycleLogEntry

        assert RealCycleLogEntry is not None

    def test_twin_cycle_log_entry_exists(self) -> None:
        """Verify TwinCycleLogEntry class exists."""
        from backend.topology.first_light import TwinCycleLogEntry

        assert TwinCycleLogEntry is not None

    def test_divergence_log_entry_exists(self) -> None:
        """Verify DivergenceLogEntry class exists."""
        from backend.topology.first_light import DivergenceLogEntry

        assert DivergenceLogEntry is not None

    def test_p4_metrics_log_entry_exists(self) -> None:
        """Verify P4MetricsLogEntry class exists."""
        from backend.topology.first_light import P4MetricsLogEntry

        assert P4MetricsLogEntry is not None

    def test_p4_summary_schema_exists(self) -> None:
        """Verify P4SummarySchema class exists."""
        from backend.topology.first_light import P4SummarySchema

        assert P4SummarySchema is not None

    def test_p4_schema_versions_defined(self) -> None:
        """Verify P4 schema version constants are defined."""
        from backend.topology.first_light.schemas_p4 import (
            REAL_CYCLE_SCHEMA_VERSION,
            TWIN_CYCLE_SCHEMA_VERSION,
            DIVERGENCE_SCHEMA_VERSION,
            P4_METRICS_SCHEMA_VERSION,
            P4_SUMMARY_SCHEMA_VERSION,
        )

        assert REAL_CYCLE_SCHEMA_VERSION == "first-light-p4-real-cycle/1.0.0"
        assert TWIN_CYCLE_SCHEMA_VERSION == "first-light-p4-twin-cycle/1.0.0"
        assert DIVERGENCE_SCHEMA_VERSION == "first-light-p4-divergence/1.0.0"
        assert P4_METRICS_SCHEMA_VERSION == "first-light-p4-metrics/1.0.0"
        assert P4_SUMMARY_SCHEMA_VERSION == "first-light-p4-summary/1.0.0"


class TestP4DataclassInstantiation:
    """
    Phase X P4: Dataclass instantiation tests.

    These tests verify P4 dataclasses can be instantiated with defaults.
    """

    def test_first_light_config_p4_instantiation(self) -> None:
        """Verify FirstLightConfigP4 can be instantiated with defaults."""
        from backend.topology.first_light import FirstLightConfigP4

        config = FirstLightConfigP4()
        assert config is not None
        assert config.shadow_mode is True  # Must always be True

    def test_first_light_result_p4_instantiation(self) -> None:
        """Verify FirstLightResultP4 can be instantiated with defaults."""
        from backend.topology.first_light import FirstLightResultP4

        result = FirstLightResultP4()
        assert result is not None

    def test_telemetry_snapshot_instantiation(self) -> None:
        """Verify TelemetrySnapshot can be instantiated with defaults."""
        from backend.topology.first_light import TelemetrySnapshot

        snapshot = TelemetrySnapshot()
        assert snapshot is not None

    def test_real_cycle_observation_instantiation(self) -> None:
        """Verify RealCycleObservation can be instantiated with defaults."""
        from backend.topology.first_light import RealCycleObservation

        obs = RealCycleObservation()
        assert obs is not None
        assert obs.source == "REAL_RUNNER"

    def test_twin_cycle_observation_instantiation(self) -> None:
        """Verify TwinCycleObservation can be instantiated with defaults."""
        from backend.topology.first_light import TwinCycleObservation

        obs = TwinCycleObservation()
        assert obs is not None
        assert obs.source == "SHADOW_TWIN"

    def test_divergence_snapshot_instantiation(self) -> None:
        """Verify DivergenceSnapshot can be instantiated with defaults."""
        from backend.topology.first_light import DivergenceSnapshot

        snapshot = DivergenceSnapshot()
        assert snapshot is not None
        assert snapshot.action == "LOGGED_ONLY"  # SHADOW MODE marker

    def test_telemetry_snapshot_factory_payload(self) -> None:
        """Verify TelemetrySnapshot accepts factory payloads."""
        from datetime import datetime
        from backend.topology.first_light import TelemetrySnapshot

        payload = make_real_telemetry_snapshot(cycle=7, seed=123)
        snapshot = TelemetrySnapshot(
            cycle=payload["cycle"],
            timestamp=datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00")),
            runner_type=payload["runner_type"],
            slice_name=payload["slice_name"],
            success=payload["success"],
            depth=payload["depth"],
            proof_hash=payload["proof_hash"],
            H=payload["H"],
            rho=payload["rho"],
            tau=payload["tau"],
            beta=payload["beta"],
            in_omega=payload["in_omega"],
            real_blocked=payload["real_blocked"],
            governance_aligned=payload["governance_aligned"],
            governance_reason=payload["governance_reason"],
            hard_ok=payload["hard_ok"],
            abstained=payload["abstained"],
            abstention_reason=payload["abstention_reason"],
            reasoning_graph_hash=payload["reasoning_graph_hash"],
            proof_dag_size=payload["proof_dag_size"],
            snapshot_hash=payload["snapshot_hash"],
        )

        assert snapshot.runner_type == payload["runner_type"]
        assert snapshot.in_omega == payload["in_omega"]

    def test_divergence_thresholds_instantiation(self) -> None:
        """Verify DivergenceThresholds can be instantiated with defaults."""
        from backend.topology.first_light import DivergenceThresholds

        thresholds = DivergenceThresholds()
        assert thresholds is not None

    def test_divergence_summary_instantiation(self) -> None:
        """Verify DivergenceSummary can be instantiated with defaults."""
        from backend.topology.first_light import DivergenceSummary

        summary = DivergenceSummary()
        assert summary is not None

    def test_real_cycle_log_entry_instantiation(self) -> None:
        """Verify RealCycleLogEntry can be instantiated with defaults."""
        from backend.topology.first_light import RealCycleLogEntry

        entry = RealCycleLogEntry()
        assert entry is not None
        assert entry.mode == "SHADOW"  # SHADOW MODE marker
        assert entry.source == "REAL_RUNNER"

    def test_twin_cycle_log_entry_instantiation(self) -> None:
        """Verify TwinCycleLogEntry can be instantiated with defaults."""
        from backend.topology.first_light import TwinCycleLogEntry

        entry = TwinCycleLogEntry()
        assert entry is not None
        assert entry.mode == "SHADOW"  # SHADOW MODE marker
        assert entry.source == "SHADOW_TWIN"

    def test_divergence_log_entry_instantiation(self) -> None:
        """Verify DivergenceLogEntry can be instantiated with defaults."""
        from backend.topology.first_light import DivergenceLogEntry

        entry = DivergenceLogEntry()
        assert entry is not None
        assert entry.mode == "SHADOW"  # SHADOW MODE marker
        assert entry.action == "LOGGED_ONLY"  # SHADOW MODE marker

    def test_divergence_log_entry_factory_payload(self) -> None:
        """Verify DivergenceLogEntry accepts factory payloads."""
        from backend.topology.first_light import DivergenceLogEntry

        payload = make_divergence_entry(12)
        entry = DivergenceLogEntry(**payload)

        assert entry.cycle == 12
        assert entry.mode == "SHADOW"
        assert entry.action == "LOGGED_ONLY"
        assert entry.severity == payload["severity"]

    def test_p4_metrics_log_entry_instantiation(self) -> None:
        """Verify P4MetricsLogEntry can be instantiated with defaults."""
        from backend.topology.first_light import P4MetricsLogEntry

        entry = P4MetricsLogEntry()
        assert entry is not None
        assert entry.mode == "SHADOW"  # SHADOW MODE marker

    def test_p4_summary_schema_instantiation(self) -> None:
        """Verify P4SummarySchema can be instantiated with defaults."""
        from backend.topology.first_light import P4SummarySchema

        schema = P4SummarySchema()
        assert schema is not None
        assert schema.mode == "SHADOW"  # SHADOW MODE marker


class TestP4StubsRaiseNotImplementedError:
    """
    Phase X P4: Stub verification tests.

    These tests verify that all P4 implementation methods raise
    NotImplementedError until implementation is authorized.

    CRITICAL: All these tests MUST pass. If any test fails because
    a method does NOT raise NotImplementedError, it means unauthorized
    implementation has occurred.
    """

    def test_config_p4_validate_raises(self) -> None:
        """Verify FirstLightConfigP4.validate() raises NotImplementedError."""
        from backend.topology.first_light import FirstLightConfigP4

        config = FirstLightConfigP4()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            config.validate()

    def test_config_p4_validate_or_raise_raises(self) -> None:
        """Verify FirstLightConfigP4.validate_or_raise() raises NotImplementedError."""
        from backend.topology.first_light import FirstLightConfigP4

        config = FirstLightConfigP4()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            config.validate_or_raise()

    def test_result_p4_to_dict_raises(self) -> None:
        """Verify FirstLightResultP4.to_dict() raises NotImplementedError."""
        from backend.topology.first_light import FirstLightResultP4

        result = FirstLightResultP4()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            result.to_dict()

    def test_result_p4_meets_success_criteria_raises(self) -> None:
        """Verify FirstLightResultP4.meets_success_criteria() raises NotImplementedError."""
        from backend.topology.first_light import FirstLightResultP4

        result = FirstLightResultP4()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            result.meets_success_criteria()

    def test_runner_p4_init_raises(self) -> None:
        """Verify FirstLightShadowRunnerP4.__init__() raises NotImplementedError."""
        from backend.topology.first_light import FirstLightShadowRunnerP4, FirstLightConfigP4

        config = FirstLightConfigP4()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            FirstLightShadowRunnerP4(config)

    def test_twin_runner_init_raises(self) -> None:
        """Verify TwinRunner.__init__() raises NotImplementedError."""
        from backend.topology.first_light import TwinRunner

        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            TwinRunner()

    def test_usla_integration_adapter_init_raises(self) -> None:
        """Verify USLAIntegrationAdapter.__init__() raises NotImplementedError."""
        from backend.topology.first_light import USLAIntegrationAdapter

        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            USLAIntegrationAdapter(None, "u2")

    def test_mock_telemetry_provider_init_raises(self) -> None:
        """Verify MockTelemetryProvider.__init__() raises NotImplementedError."""
        from backend.topology.first_light import MockTelemetryProvider

        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            MockTelemetryProvider()

    def test_divergence_analyzer_init_raises(self) -> None:
        """Verify DivergenceAnalyzer.__init__() raises NotImplementedError."""
        from backend.topology.first_light import DivergenceAnalyzer

        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            DivergenceAnalyzer()

    def test_divergence_summary_to_dict_raises(self) -> None:
        """Verify DivergenceSummary.to_dict() raises NotImplementedError."""
        from backend.topology.first_light import DivergenceSummary

        summary = DivergenceSummary()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            summary.to_dict()

    def test_real_cycle_observation_to_dict_raises(self) -> None:
        """Verify RealCycleObservation.to_dict() raises NotImplementedError."""
        from backend.topology.first_light import RealCycleObservation

        obs = RealCycleObservation()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            obs.to_dict()

    def test_real_cycle_observation_from_snapshot_raises(self) -> None:
        """Verify RealCycleObservation.from_snapshot() raises NotImplementedError."""
        from backend.topology.first_light import RealCycleObservation, TelemetrySnapshot

        snapshot = TelemetrySnapshot()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            RealCycleObservation.from_snapshot(snapshot)

    def test_twin_cycle_observation_to_dict_raises(self) -> None:
        """Verify TwinCycleObservation.to_dict() raises NotImplementedError."""
        from backend.topology.first_light import TwinCycleObservation

        obs = TwinCycleObservation()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            obs.to_dict()

    def test_divergence_snapshot_to_dict_raises(self) -> None:
        """Verify DivergenceSnapshot.to_dict() raises NotImplementedError."""
        from backend.topology.first_light import DivergenceSnapshot

        snapshot = DivergenceSnapshot()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            snapshot.to_dict()

    def test_divergence_snapshot_is_diverged_raises(self) -> None:
        """Verify DivergenceSnapshot.is_diverged() raises NotImplementedError."""
        from backend.topology.first_light import DivergenceSnapshot

        snapshot = DivergenceSnapshot()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            snapshot.is_diverged()

    def test_divergence_snapshot_from_observations_raises(self) -> None:
        """Verify DivergenceSnapshot.from_observations() raises NotImplementedError."""
        from backend.topology.first_light import (
            DivergenceSnapshot,
            RealCycleObservation,
            TwinCycleObservation,
        )

        real = RealCycleObservation()
        twin = TwinCycleObservation()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            DivergenceSnapshot.from_observations(real, twin, {})

    def test_real_cycle_log_entry_to_dict_raises(self) -> None:
        """Verify RealCycleLogEntry.to_dict() raises NotImplementedError."""
        from backend.topology.first_light import RealCycleLogEntry

        entry = RealCycleLogEntry()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            entry.to_dict()

    def test_real_cycle_log_entry_to_json_line_raises(self) -> None:
        """Verify RealCycleLogEntry.to_json_line() raises NotImplementedError."""
        from backend.topology.first_light import RealCycleLogEntry

        entry = RealCycleLogEntry()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            entry.to_json_line()

    def test_p4_summary_schema_to_dict_raises(self) -> None:
        """Verify P4SummarySchema.to_dict() raises NotImplementedError."""
        from backend.topology.first_light import P4SummarySchema

        schema = P4SummarySchema()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            schema.to_dict()

    def test_p4_summary_schema_to_json_raises(self) -> None:
        """Verify P4SummarySchema.to_json() raises NotImplementedError."""
        from backend.topology.first_light import P4SummarySchema

        schema = P4SummarySchema()
        with pytest.raises(NotImplementedError, match="P4 implementation not yet activated"):
            schema.to_json()


class TestP4ShadowModeMarkers:
    """
    Phase X P4: SHADOW MODE marker verification.

    These tests verify that all P4 data structures include proper
    SHADOW MODE markers (mode="SHADOW", action="LOGGED_ONLY").
    """

    def test_config_p4_default_shadow_mode_true(self) -> None:
        """Verify FirstLightConfigP4 defaults to shadow_mode=True."""
        from backend.topology.first_light import FirstLightConfigP4

        config = FirstLightConfigP4()
        assert config.shadow_mode is True

    def test_real_cycle_observation_source_marker(self) -> None:
        """Verify RealCycleObservation has source='REAL_RUNNER'."""
        from backend.topology.first_light import RealCycleObservation

        obs = RealCycleObservation()
        assert obs.source == "REAL_RUNNER"

    def test_twin_cycle_observation_source_marker(self) -> None:
        """Verify TwinCycleObservation has source='SHADOW_TWIN'."""
        from backend.topology.first_light import TwinCycleObservation

        obs = TwinCycleObservation()
        assert obs.source == "SHADOW_TWIN"

    def test_divergence_snapshot_action_marker(self) -> None:
        """Verify DivergenceSnapshot has action='LOGGED_ONLY'."""
        from backend.topology.first_light import DivergenceSnapshot

        snapshot = DivergenceSnapshot()
        assert snapshot.action == "LOGGED_ONLY"

    def test_real_cycle_log_entry_mode_marker(self) -> None:
        """Verify RealCycleLogEntry has mode='SHADOW'."""
        from backend.topology.first_light import RealCycleLogEntry

        entry = RealCycleLogEntry()
        assert entry.mode == "SHADOW"

    def test_twin_cycle_log_entry_mode_marker(self) -> None:
        """Verify TwinCycleLogEntry has mode='SHADOW'."""
        from backend.topology.first_light import TwinCycleLogEntry

        entry = TwinCycleLogEntry()
        assert entry.mode == "SHADOW"

    def test_divergence_log_entry_mode_marker(self) -> None:
        """Verify DivergenceLogEntry has mode='SHADOW'."""
        from backend.topology.first_light import DivergenceLogEntry

        entry = DivergenceLogEntry()
        assert entry.mode == "SHADOW"

    def test_divergence_log_entry_action_marker(self) -> None:
        """Verify DivergenceLogEntry has action='LOGGED_ONLY'."""
        from backend.topology.first_light import DivergenceLogEntry

        entry = DivergenceLogEntry()
        assert entry.action == "LOGGED_ONLY"

    def test_p4_metrics_log_entry_mode_marker(self) -> None:
        """Verify P4MetricsLogEntry has mode='SHADOW'."""
        from backend.topology.first_light import P4MetricsLogEntry

        entry = P4MetricsLogEntry()
        assert entry.mode == "SHADOW"

    def test_p4_summary_schema_mode_marker(self) -> None:
        """Verify P4SummarySchema has mode='SHADOW'."""
        from backend.topology.first_light import P4SummarySchema

        schema = P4SummarySchema()
        assert schema.mode == "SHADOW"


class TestP4TDAWindowFactory:
    """Validate factory-generated TDA windows."""

    def test_tda_window_factory_payload_structure(self) -> None:
        payload = make_tda_window(window_index=3, seed=2025)

        assert payload["mode"] == "SHADOW"
        assert "sns" in payload and "pcs" in payload and "hss" in payload and "drs" in payload
        assert payload["sns"]["max"] >= payload["sns"]["min"]
        assert payload["pcs"]["max"] >= payload["pcs"]["min"]
        assert payload["hss"]["min"] <= payload["hss"]["max"]
        assert payload["drs"]["max"] >= payload["drs"]["min"]
        assert payload["red_flags"]["tda_sns_anomaly"] >= 0
        assert len(payload["trajectories"]["sns"]) == len(payload["trajectories"]["drs"])


class TestP4InterfaceContract:
    """
    Phase X P4: Interface contract tests.

    These tests verify that abstract interfaces are properly defined.
    """

    def test_telemetry_provider_is_abstract(self) -> None:
        """Verify TelemetryProviderInterface is abstract."""
        from abc import ABC
        from backend.topology.first_light import TelemetryProviderInterface

        assert issubclass(TelemetryProviderInterface, ABC)

    def test_telemetry_provider_has_required_methods(self) -> None:
        """Verify TelemetryProviderInterface has required abstract methods."""
        from backend.topology.first_light import TelemetryProviderInterface

        required_methods = [
            "get_current_snapshot",
            "get_historical_snapshots",
            "is_available",
            "get_runner_type",
            "get_current_cycle",
        ]

        for method_name in required_methods:
            assert hasattr(TelemetryProviderInterface, method_name), f"Missing method: {method_name}"

    def test_usla_integration_adapter_inherits_interface(self) -> None:
        """Verify USLAIntegrationAdapter inherits from TelemetryProviderInterface."""
        from backend.topology.first_light import (
            USLAIntegrationAdapter,
            TelemetryProviderInterface,
        )

        assert issubclass(USLAIntegrationAdapter, TelemetryProviderInterface)

    def test_mock_telemetry_provider_inherits_interface(self) -> None:
        """Verify MockTelemetryProvider inherits from TelemetryProviderInterface."""
        from backend.topology.first_light import (
            MockTelemetryProvider,
            TelemetryProviderInterface,
        )

        assert issubclass(MockTelemetryProvider, TelemetryProviderInterface)

    def test_telemetry_snapshot_is_frozen(self) -> None:
        """Verify TelemetrySnapshot is frozen (immutable)."""
        from backend.topology.first_light import TelemetrySnapshot
        from dataclasses import fields

        snapshot = TelemetrySnapshot()
        # Frozen dataclasses raise FrozenInstanceError on attribute assignment
        with pytest.raises(Exception):  # FrozenInstanceError
            snapshot.cycle = 999  # type: ignore
