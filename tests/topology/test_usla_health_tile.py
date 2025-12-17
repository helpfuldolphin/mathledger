"""
Phase X P1: USLA Health Tile Tests

Tests for the USLAHealthTileProducer to verify:
1. Tile can be produced from USLAState
2. Required fields are present
3. Output is JSON-serializable
4. Headlines are deterministic for given inputs
"""

import json

import pytest


class TestUSLAHealthTileImport:
    """Test that health tile components can be imported."""

    def test_import_health_tile_producer(self):
        """Verify USLAHealthTileProducer can be imported."""
        from backend.topology.usla_health_tile import (
            USLAHealthTileProducer,
            USLAHealthTile,
            HEALTH_TILE_SCHEMA_VERSION,
        )
        assert USLAHealthTileProducer is not None
        assert USLAHealthTile is not None
        assert HEALTH_TILE_SCHEMA_VERSION == "1.0.0"


class TestUSLAHealthTileProducer:
    """Tests for USLAHealthTileProducer."""

    def test_produce_from_mock_state(self):
        """Produce a health tile from a mock state."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState, ConvergenceClass

        producer = USLAHealthTileProducer()

        # Create a mock state with reasonable values
        state = USLAState(
            H=0.75,
            D=5,
            D_dot=0.5,
            B=2.0,
            S=0.1,
            C=ConvergenceClass.CONVERGING,
            rho=0.85,
            tau=0.21,
            J=2.5,
            W=False,
            beta=0.05,
            kappa=0.8,
            nu=0.001,
            delta=0,
            Gamma=0.88,
            cycle=42,
            blocked=False,
            active_cdis=[],
            invariant_violations=[],
        )

        tile = producer.produce(state=state, hard_ok=True)

        # Verify tile is a dict
        assert isinstance(tile, dict)

        # Verify required fields exist
        assert "schema_version" in tile
        assert "tile_type" in tile
        assert "timestamp" in tile
        assert "state_summary" in tile
        assert "hard_mode_status" in tile
        assert "headline" in tile

        # Verify state summary fields
        state_summary = tile["state_summary"]
        assert "H" in state_summary
        assert "rho" in state_summary
        assert "tau" in state_summary
        assert "beta" in state_summary
        assert "J" in state_summary
        assert "C" in state_summary
        assert "Gamma" in state_summary

        # Verify values match input
        assert state_summary["H"] == pytest.approx(0.75, rel=0.01)
        assert state_summary["rho"] == pytest.approx(0.85, rel=0.01)
        assert state_summary["C"] == "CONVERGING"

    def test_tile_is_json_serializable(self):
        """Verify tile can be serialized to JSON."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState, ConvergenceClass

        producer = USLAHealthTileProducer()

        state = USLAState(
            H=0.7,
            rho=0.8,
            tau=0.2,
            C=ConvergenceClass.OSCILLATING,
            cycle=100,
        )

        tile = producer.produce(state=state, hard_ok=True)

        # This should not raise
        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed["state_summary"]["H"] == pytest.approx(0.7, rel=0.01)

    def test_headline_nominal(self):
        """Verify nominal headline when all is well."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState

        producer = USLAHealthTileProducer()

        state = USLAState(
            H=0.8,
            rho=0.9,
            active_cdis=[],
            invariant_violations=[],
        )

        tile = producer.produce(state=state, hard_ok=True)

        assert tile["headline"] == "Topology stable; monitoring active"

    def test_headline_cdis_detected(self):
        """Verify headline when CDIs are active."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState

        producer = USLAHealthTileProducer()

        state = USLAState(
            H=0.7,
            rho=0.8,
            active_cdis=["CDI-001", "CDI-003"],
            invariant_violations=[],
        )

        tile = producer.produce(state=state, hard_ok=True)

        assert "CDI" in tile["headline"] or "degraded" in tile["headline"].lower()

    def test_headline_invariant_violations(self):
        """Verify headline when invariants are violated."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState

        producer = USLAHealthTileProducer()

        state = USLAState(
            H=0.6,
            rho=0.7,
            active_cdis=[],
            invariant_violations=["INV-001", "INV-003"],
        )

        tile = producer.produce(state=state, hard_ok=True)

        assert "invariant" in tile["headline"].lower() or "degraded" in tile["headline"].lower()

    def test_headline_hard_fail(self):
        """Verify headline when HARD mode fails."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState

        producer = USLAHealthTileProducer()

        state = USLAState(
            H=0.5,
            rho=0.6,
            active_cdis=[],
            invariant_violations=[],
        )

        tile = producer.produce(state=state, hard_ok=False)

        assert "HARD" in tile["headline"] or "inactive" in tile["headline"].lower()
        assert tile["hard_mode_status"] == "FAIL"

    def test_headline_deterministic(self):
        """Verify headline is deterministic for same input."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState, ConvergenceClass

        producer = USLAHealthTileProducer()

        state = USLAState(
            H=0.72,
            D=4,
            rho=0.81,
            tau=0.19,
            C=ConvergenceClass.CONVERGING,
            active_cdis=[],
            invariant_violations=[],
        )

        # Produce multiple times
        tile1 = producer.produce(state=state, hard_ok=True)
        tile2 = producer.produce(state=state, hard_ok=True)
        tile3 = producer.produce(state=state, hard_ok=True)

        # Headlines should be identical
        assert tile1["headline"] == tile2["headline"]
        assert tile2["headline"] == tile3["headline"]

        # State summaries should match (excluding timestamp)
        assert tile1["state_summary"] == tile2["state_summary"]


class TestUSLAHealthTileWithMonitor:
    """Tests for health tile with divergence monitor."""

    def test_produce_with_monitor(self):
        """Produce health tile with divergence monitor."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState
        from backend.topology.divergence_monitor import DivergenceMonitor

        producer = USLAHealthTileProducer()
        monitor = DivergenceMonitor()

        # Simulate some divergence
        monitor.check(cycle=1, real_blocked=False, sim_blocked=True)
        monitor.check(cycle=2, real_blocked=False, sim_blocked=True)

        state = USLAState(H=0.7, rho=0.8)

        tile = producer.produce(state=state, hard_ok=True, monitor=monitor)

        # Verify divergence summary
        assert "divergence_summary" in tile
        div_summary = tile["divergence_summary"]
        assert div_summary["governance_aligned"] is False
        assert div_summary["consecutive_divergence"] == 2

    def test_produce_aligned_with_monitor(self):
        """Produce health tile when monitor shows alignment."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_simulator import USLAState
        from backend.topology.divergence_monitor import DivergenceMonitor

        producer = USLAHealthTileProducer()
        monitor = DivergenceMonitor()

        # All aligned
        monitor.check(cycle=1, real_blocked=False, sim_blocked=False)
        monitor.check(cycle=2, real_blocked=True, sim_blocked=True)

        state = USLAState(H=0.8, rho=0.9)

        tile = producer.produce(state=state, hard_ok=True, monitor=monitor)

        assert tile["divergence_summary"]["governance_aligned"] is True
        assert tile["divergence_summary"]["consecutive_divergence"] == 0


class TestUSLAHealthTileFromIntegration:
    """Tests for producing health tile from USLAIntegration."""

    def test_produce_from_disabled_integration(self):
        """Produce returns None for disabled integration."""
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        producer = USLAHealthTileProducer()

        integration = USLAIntegration.create_for_runner(
            runner_type=RunnerType.RFL,
            runner_id="test",
            enabled=False,
        )

        tile = producer.produce_from_integration(integration)
        assert tile is None

    def test_produce_from_enabled_integration(self):
        """Produce health tile from enabled integration."""
        import tempfile
        from backend.topology.usla_health_tile import USLAHealthTileProducer
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        producer = USLAHealthTileProducer()

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.RFL,
                runner_id="test",
                enabled=True,
                log_dir=tmpdir,
            )

            tile = producer.produce_from_integration(integration)

            assert tile is not None
            assert "state_summary" in tile
            assert "headline" in tile

            integration.close()


class TestTDATelemetryProvider:
    """Tests for TDATelemetryProvider."""

    def test_import_telemetry_provider(self):
        """Verify TDATelemetryProvider can be imported."""
        from backend.topology.tda_telemetry_provider import (
            TDATelemetryProvider,
            TDATelemetrySnapshot,
            TDATelemetryConfig,
        )
        assert TDATelemetryProvider is not None
        assert TDATelemetrySnapshot is not None

    def test_capture_synthetic(self):
        """Test synthetic telemetry capture."""
        from backend.topology.tda_telemetry_provider import TDATelemetryProvider

        provider = TDATelemetryProvider()

        snapshot = provider.capture_synthetic(
            hss=0.75,
            depth=5,
            blocked=False,
            threshold=0.2,
        )

        assert snapshot.blocked is False
        assert snapshot.threshold == 0.2
        assert snapshot.hss_by_depth == {5: 0.75}
        assert snapshot.source == "synthetic"

    def test_capture_from_rfl_mock(self):
        """Test RFL telemetry capture with mock data."""
        from backend.topology.tda_telemetry_provider import TDATelemetryProvider
        from dataclasses import dataclass

        @dataclass
        class MockLedgerEntry:
            abstention_fraction: float = 0.15

        @dataclass
        class MockAttestation:
            abstention_rate: float = 0.15
            metadata: dict = None

            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {"max_depth": 6}

        provider = TDATelemetryProvider()

        snapshot = provider.capture_from_rfl(
            ledger_entry=MockLedgerEntry(),
            attestation=MockAttestation(),
            blocked=False,
            governance_threshold=0.18,
        )

        assert snapshot.blocked is False
        assert snapshot.threshold == 0.18
        assert snapshot.source == "rfl"
        assert snapshot.hss_by_depth is not None
        assert 6 in snapshot.hss_by_depth

    def test_capture_from_u2_mock(self):
        """Test U2 telemetry capture with mock data."""
        from backend.topology.tda_telemetry_provider import TDATelemetryProvider

        provider = TDATelemetryProvider()

        cycle_result = {
            "success": True,
            "depth": 4,
        }

        # Build up success history
        success_history = [True, True, False, True, True]

        snapshot = provider.capture_from_u2(
            cycle_result=cycle_result,
            blocked=False,
            governance_threshold=0.22,
            success_history=success_history,
        )

        assert snapshot.blocked is False
        assert snapshot.threshold == 0.22
        assert snapshot.source == "u2"

    def test_telemetry_snapshot_to_dict(self):
        """Test TDATelemetrySnapshot.to_dict()."""
        from backend.topology.tda_telemetry_provider import TDATelemetrySnapshot

        snapshot = TDATelemetrySnapshot(
            blocked=True,
            threshold=0.25,
            hss_by_depth={3: 0.8, 4: 0.75, 5: 0.7},
            real_rsi=0.82,
            real_block_rate=0.12,
            source="test",
        )

        d = snapshot.to_dict()

        assert d["blocked"] is True
        assert d["threshold"] == 0.25
        assert d["hss_by_depth"] == {3: 0.8, 4: 0.75, 5: 0.7}
        assert d["real_rsi"] == 0.82
        assert d["source"] == "test"
