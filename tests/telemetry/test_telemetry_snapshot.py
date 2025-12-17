"""
Tests for Telemetry Snapshot Correctness and Hash Chain Determinism

Phase X: Telemetry Canonical Interface

These tests verify:
1. TelemetrySnapshot immutability and correctness
2. Hash computation determinism
3. Hash chain integrity
4. Schema conformance of serialized output

SHADOW MODE: All tests operate in observation-only mode.

See: docs/system_law/Telemetry_PhaseX_Contract.md
"""

import json
import pytest
from datetime import datetime, timezone

from backend.topology.first_light.data_structures_p4 import (
    TelemetrySnapshot,
    RealCycleObservation,
    TwinCycleObservation,
    DivergenceSnapshot,
)
from backend.topology.first_light.telemetry_adapter import (
    MockTelemetryProvider,
    TelemetryProviderInterface,
)


class TestTelemetrySnapshotCorrectness:
    """Tests for TelemetrySnapshot correctness."""

    def test_snapshot_is_frozen(self):
        """Verify TelemetrySnapshot is immutable (frozen dataclass)."""
        snapshot = TelemetrySnapshot(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            runner_type="u2",
            H=0.8,
        )

        # Should raise FrozenInstanceError
        with pytest.raises(Exception):  # FrozenInstanceError
            snapshot.cycle = 2

    def test_snapshot_to_dict(self):
        """Verify to_dict() produces correct structure."""
        snapshot = TelemetrySnapshot(
            cycle=42,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            runner_type="u2",
            slice_name="arithmetic_simple",
            success=True,
            depth=5,
            H=0.85,
            rho=0.92,
            tau=0.20,
            beta=0.05,
            in_omega=True,
            real_blocked=False,
            governance_aligned=True,
            hard_ok=True,
        )

        d = snapshot.to_dict()

        assert d["cycle"] == 42
        assert d["runner_type"] == "u2"
        assert d["success"] is True
        assert d["H"] == 0.85
        assert d["in_omega"] is True

    def test_snapshot_hash_determinism(self):
        """Verify hash computation is deterministic."""
        data = {
            "cycle": 1,
            "timestamp": "2025-12-10T12:00:00.000000+00:00",
            "runner_type": "u2",
            "H": 0.8,
            "rho": 0.9,
            "tau": 0.2,
            "beta": 0.1,
        }

        hash1 = TelemetrySnapshot.compute_hash(data)
        hash2 = TelemetrySnapshot.compute_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_snapshot_hash_changes_with_data(self):
        """Verify hash changes when data changes."""
        data1 = {"cycle": 1, "H": 0.8}
        data2 = {"cycle": 1, "H": 0.81}

        hash1 = TelemetrySnapshot.compute_hash(data1)
        hash2 = TelemetrySnapshot.compute_hash(data2)

        assert hash1 != hash2

    def test_snapshot_hash_order_independent(self):
        """Verify hash is independent of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        hash1 = TelemetrySnapshot.compute_hash(data1)
        hash2 = TelemetrySnapshot.compute_hash(data2)

        assert hash1 == hash2


class TestMockTelemetryProvider:
    """Tests for MockTelemetryProvider."""

    def test_provider_initialization(self):
        """Verify provider initializes correctly."""
        provider = MockTelemetryProvider(runner_type="u2", seed=42)

        assert provider.get_runner_type() == "u2"
        assert provider.get_current_cycle() == 0
        assert provider.is_available() is True

    def test_provider_invalid_runner_type(self):
        """Verify provider rejects invalid runner type."""
        with pytest.raises(ValueError):
            MockTelemetryProvider(runner_type="invalid")

    def test_provider_snapshot_generation(self):
        """Verify provider generates valid snapshots."""
        provider = MockTelemetryProvider(runner_type="u2", seed=42)

        snapshot = provider.get_snapshot()

        assert snapshot is not None
        assert snapshot.cycle == 1
        assert snapshot.runner_type == "u2"
        assert 0.0 <= snapshot.H <= 1.0
        assert 0.0 <= snapshot.rho <= 1.0
        assert snapshot.snapshot_hash != ""

    def test_provider_snapshot_sequence(self):
        """Verify provider generates sequential snapshots."""
        provider = MockTelemetryProvider(runner_type="u2", seed=42)

        s1 = provider.get_snapshot()
        s2 = provider.get_snapshot()
        s3 = provider.get_snapshot()

        assert s1.cycle == 1
        assert s2.cycle == 2
        assert s3.cycle == 3

    def test_provider_determinism_with_seed(self):
        """Verify provider is deterministic with same seed."""
        provider1 = MockTelemetryProvider(runner_type="u2", seed=42)
        provider2 = MockTelemetryProvider(runner_type="u2", seed=42)

        s1 = provider1.get_snapshot()
        s2 = provider2.get_snapshot()

        assert s1.H == s2.H
        assert s1.rho == s2.rho
        assert s1.success == s2.success

    def test_provider_different_with_different_seed(self):
        """Verify different seeds produce different results."""
        provider1 = MockTelemetryProvider(runner_type="u2", seed=42)
        provider2 = MockTelemetryProvider(runner_type="u2", seed=123)

        s1 = provider1.get_snapshot()
        s2 = provider2.get_snapshot()

        # With different seeds, values should differ
        # (statistically very unlikely to be equal)
        assert s1.snapshot_hash != s2.snapshot_hash

    def test_provider_availability_control(self):
        """Verify availability can be controlled."""
        provider = MockTelemetryProvider(runner_type="u2")

        assert provider.is_available() is True
        assert provider.get_snapshot() is not None

        provider.set_available(False)
        assert provider.is_available() is False
        assert provider.get_snapshot() is None

    def test_provider_reset(self):
        """Verify provider reset works."""
        provider = MockTelemetryProvider(runner_type="u2", seed=42)

        provider.get_snapshot()
        provider.get_snapshot()
        assert provider.get_current_cycle() == 2

        provider.reset()
        assert provider.get_current_cycle() == 0

        # After reset, should produce same sequence
        s1 = provider.get_snapshot()
        provider.reset()
        s2 = provider.get_snapshot()

        assert s1.H == s2.H

    def test_provider_historical_snapshots(self):
        """Verify historical snapshot retrieval."""
        provider = MockTelemetryProvider(runner_type="u2", seed=42)

        # Generate some snapshots
        for _ in range(10):
            provider.get_snapshot()

        # Retrieve history
        history = list(provider.get_historical_snapshots(3, 7))

        assert len(history) == 5
        assert history[0].cycle == 3
        assert history[-1].cycle == 7


class TestRealCycleObservation:
    """Tests for RealCycleObservation."""

    def test_observation_from_snapshot(self):
        """Verify observation can be created from snapshot."""
        snapshot = TelemetrySnapshot(
            cycle=42,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            runner_type="u2",
            slice_name="arithmetic_simple",
            success=True,
            depth=5,
            H=0.85,
            rho=0.92,
            tau=0.20,
            beta=0.05,
            in_omega=True,
            real_blocked=False,
            governance_aligned=True,
            hard_ok=True,
        )

        obs = RealCycleObservation.from_snapshot(snapshot)

        assert obs.source == "REAL_RUNNER"
        assert obs.mode == "SHADOW"
        assert obs.cycle == 42
        assert obs.success is True
        assert obs.H == 0.85
        assert obs.in_omega is True

    def test_observation_to_dict(self):
        """Verify to_dict produces schema-conformant output."""
        obs = RealCycleObservation(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            runner_type="u2",
            slice_name="test",
            success=True,
            H=0.8,
            rho=0.9,
            tau=0.2,
            beta=0.1,
            in_omega=True,
        )

        d = obs.to_dict()

        assert d["source"] == "REAL_RUNNER"
        assert d["mode"] == "SHADOW"
        assert "usla_state" in d
        assert d["usla_state"]["H"] == 0.8


class TestDivergenceSnapshot:
    """Tests for DivergenceSnapshot."""

    def test_divergence_from_observations(self):
        """Verify divergence can be computed from observations."""
        real = RealCycleObservation(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            success=True,
            H=0.8,
            rho=0.9,
            tau=0.2,
            beta=0.1,
            in_omega=True,
            real_blocked=False,
            hard_ok=True,
        )

        twin = TwinCycleObservation(
            real_cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            predicted_success=True,
            predicted_blocked=False,
            predicted_in_omega=True,
            predicted_hard_ok=True,
            twin_H=0.79,
            twin_rho=0.88,
            twin_tau=0.2,
            twin_beta=0.11,
        )

        div = DivergenceSnapshot.from_observations(real, twin)

        assert div.cycle == 1
        assert div.success_diverged is False
        assert div.blocked_diverged is False
        assert abs(div.H_delta - 0.01) < 0.001
        assert div.action == "LOGGED_ONLY"

    def test_divergence_detects_outcome_divergence(self):
        """Verify outcome divergence is detected."""
        real = RealCycleObservation(
            cycle=1,
            success=True,  # Real: success
            H=0.8,
            rho=0.9,
            tau=0.2,
            beta=0.1,
            in_omega=True,
            real_blocked=False,
            hard_ok=True,
        )

        twin = TwinCycleObservation(
            real_cycle=1,
            predicted_success=False,  # Twin predicted: failure
            predicted_blocked=False,
            predicted_in_omega=True,
            predicted_hard_ok=True,
            twin_H=0.8,
            twin_rho=0.9,
            twin_tau=0.2,
            twin_beta=0.1,
        )

        div = DivergenceSnapshot.from_observations(real, twin)

        assert div.success_diverged is True
        assert div.divergence_type in ("OUTCOME", "BOTH")

    def test_divergence_severity_classification(self):
        """Verify severity is classified correctly."""
        # Small delta -> NONE
        real1 = RealCycleObservation(H=0.8, rho=0.9, tau=0.2, beta=0.1)
        twin1 = TwinCycleObservation(twin_H=0.8, twin_rho=0.9, twin_tau=0.2, twin_beta=0.1)
        div1 = DivergenceSnapshot.from_observations(real1, twin1)
        assert div1.divergence_severity == "NONE"

        # Large delta -> CRITICAL
        real2 = RealCycleObservation(H=0.8, rho=0.9, tau=0.2, beta=0.1)
        twin2 = TwinCycleObservation(twin_H=0.2, twin_rho=0.3, twin_tau=0.5, twin_beta=0.7)
        div2 = DivergenceSnapshot.from_observations(real2, twin2)
        assert div2.divergence_severity == "CRITICAL"

    def test_divergence_is_diverged(self):
        """Verify is_diverged() helper."""
        # No divergence
        div1 = DivergenceSnapshot(divergence_severity="NONE")
        assert div1.is_diverged() is False

        # Has divergence
        div2 = DivergenceSnapshot(success_diverged=True, divergence_severity="WARN")
        assert div2.is_diverged() is True


class TestHashChainIntegrity:
    """Tests for hash chain integrity across snapshots."""

    def test_consecutive_snapshots_form_chain(self):
        """Verify consecutive snapshots can form a hash chain."""
        provider = MockTelemetryProvider(runner_type="u2", seed=42)

        snapshots = [provider.get_snapshot() for _ in range(5)]

        # Each snapshot should have unique hash
        hashes = [s.snapshot_hash for s in snapshots]
        assert len(set(hashes)) == 5  # All unique

    def test_snapshot_hash_includes_cycle(self):
        """Verify snapshot hash includes cycle number."""
        # Same state but different cycles should have different hashes
        data1 = {"cycle": 1, "H": 0.8}
        data2 = {"cycle": 2, "H": 0.8}

        hash1 = TelemetrySnapshot.compute_hash(data1)
        hash2 = TelemetrySnapshot.compute_hash(data2)

        assert hash1 != hash2

    def test_hash_chain_tamper_detection(self):
        """Verify tampering can be detected via hash mismatch."""
        data = {"cycle": 1, "H": 0.8, "rho": 0.9}
        original_hash = TelemetrySnapshot.compute_hash(data)

        # Tamper with data
        data["H"] = 0.81
        tampered_hash = TelemetrySnapshot.compute_hash(data)

        assert original_hash != tampered_hash


class TestSchemaConformance:
    """Tests for schema conformance of serialized output."""

    def test_snapshot_json_serializable(self):
        """Verify snapshot can be serialized to JSON."""
        snapshot = TelemetrySnapshot(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            runner_type="u2",
            H=0.8,
        )

        d = snapshot.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["cycle"] == 1

    def test_observation_json_serializable(self):
        """Verify observation can be serialized to JSON."""
        obs = RealCycleObservation(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            runner_type="u2",
            success=True,
            H=0.8,
            rho=0.9,
            tau=0.2,
            beta=0.1,
        )

        d = obs.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["source"] == "REAL_RUNNER"
        assert parsed["mode"] == "SHADOW"

    def test_divergence_json_serializable(self):
        """Verify divergence can be serialized to JSON."""
        div = DivergenceSnapshot(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            H_delta=0.01,
            divergence_severity="NONE",
        )

        d = div.to_dict()
        json_str = json.dumps(d)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["action"] == "LOGGED_ONLY"
        assert parsed["mode"] == "SHADOW"
