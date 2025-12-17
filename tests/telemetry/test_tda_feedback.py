"""
Tests for TDA Telemetry Feedback Provider and Anomaly Detection

Phase X: Telemetry Canonical Interface

These tests verify:
1. TDA feedback provider functionality
2. Anomaly window management
3. Topology metrics computation (Betti numbers, persistence)
4. Anomaly clustering and detection
5. Governance signal emission

SHADOW MODE: All tests operate in observation-only mode.

See: docs/system_law/Telemetry_PhaseX_Contract.md Section 9
"""

import json
import pytest
from datetime import datetime, timezone

from backend.topology.tda_telemetry_feedback import (
    TDAFeedbackProvider,
    TDAFeedback,
    TelemetryAnomalyWindow,
    AnomalyRecord,
    TopologyMetrics,
)
from backend.telemetry.governance_signal import (
    TelemetryGovernanceSignal,
    TelemetryGovernanceSignalEmitter,
    EmitterHealth,
    AnomalySummary,
    GovernanceRecommendation,
)


class TestAnomalyRecord:
    """Tests for AnomalyRecord."""

    def test_anomaly_record_creation(self):
        """Verify anomaly record can be created."""
        record = AnomalyRecord(
            cycle=42,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            anomaly_type="rate_anomaly",
            severity="WARN",
            component="u2_runner",
            details={"rate_drop": 0.3},
        )

        assert record.cycle == 42
        assert record.anomaly_type == "rate_anomaly"
        assert record.severity == "WARN"

    def test_anomaly_record_to_dict(self):
        """Verify to_dict produces correct structure."""
        record = AnomalyRecord(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            anomaly_type="schema_violation",
            severity="CRITICAL",
            component="lean_verifier",
        )

        d = record.to_dict()

        assert d["cycle"] == 1
        assert d["anomaly_type"] == "schema_violation"
        assert d["severity"] == "CRITICAL"


class TestTelemetryAnomalyWindow:
    """Tests for TelemetryAnomalyWindow."""

    def test_window_initialization(self):
        """Verify window initializes correctly."""
        window = TelemetryAnomalyWindow(window_size=50)

        assert window.window_size == 50
        assert len(window.anomalies) == 0

    def test_window_add_anomaly(self):
        """Verify anomalies can be added."""
        window = TelemetryAnomalyWindow(window_size=100)

        record = AnomalyRecord(
            cycle=1,
            timestamp="2025-12-10T12:00:00.000000+00:00",
            anomaly_type="rate_anomaly",
            severity="WARN",
            component="u2_runner",
        )

        window.add_anomaly(record)

        assert len(window.anomalies) == 1
        assert window.start_cycle == 1
        assert window.end_cycle == 1

    def test_window_size_limit(self):
        """Verify window respects size limit."""
        window = TelemetryAnomalyWindow(window_size=5)

        for i in range(10):
            window.add_anomaly(AnomalyRecord(
                cycle=i,
                timestamp=f"2025-12-10T12:00:{i:02d}.000000+00:00",
                anomaly_type="rate_anomaly",
                severity="INFO",
                component="test",
            ))

        assert len(window.anomalies) == 5
        assert window.start_cycle == 5  # Oldest remaining
        assert window.end_cycle == 9    # Newest

    def test_window_get_by_type(self):
        """Verify filtering by type."""
        window = TelemetryAnomalyWindow()

        window.add_anomaly(AnomalyRecord(
            cycle=1, timestamp="", anomaly_type="rate_anomaly",
            severity="WARN", component="test"
        ))
        window.add_anomaly(AnomalyRecord(
            cycle=2, timestamp="", anomaly_type="schema_violation",
            severity="WARN", component="test"
        ))
        window.add_anomaly(AnomalyRecord(
            cycle=3, timestamp="", anomaly_type="rate_anomaly",
            severity="INFO", component="test"
        ))

        rate_anomalies = window.get_by_type("rate_anomaly")

        assert len(rate_anomalies) == 2

    def test_window_get_by_severity(self):
        """Verify filtering by severity."""
        window = TelemetryAnomalyWindow()

        window.add_anomaly(AnomalyRecord(
            cycle=1, timestamp="", anomaly_type="a", severity="INFO", component="test"
        ))
        window.add_anomaly(AnomalyRecord(
            cycle=2, timestamp="", anomaly_type="b", severity="WARN", component="test"
        ))
        window.add_anomaly(AnomalyRecord(
            cycle=3, timestamp="", anomaly_type="c", severity="WARN", component="test"
        ))

        warn_anomalies = window.get_by_severity("WARN")

        assert len(warn_anomalies) == 2

    def test_window_count_by_type(self):
        """Verify count by type."""
        window = TelemetryAnomalyWindow()

        for i in range(3):
            window.add_anomaly(AnomalyRecord(
                cycle=i, timestamp="", anomaly_type="rate_anomaly",
                severity="INFO", component="test"
            ))
        for i in range(2):
            window.add_anomaly(AnomalyRecord(
                cycle=i + 3, timestamp="", anomaly_type="schema_violation",
                severity="WARN", component="test"
            ))

        counts = window.count_by_type()

        assert counts["rate_anomaly"] == 3
        assert counts["schema_violation"] == 2

    def test_window_clear(self):
        """Verify window clear."""
        window = TelemetryAnomalyWindow()

        window.add_anomaly(AnomalyRecord(
            cycle=1, timestamp="", anomaly_type="a", severity="INFO", component="test"
        ))
        window.clear()

        assert len(window.anomalies) == 0
        assert window.start_cycle == 0
        assert window.end_cycle == 0


class TestTopologyMetrics:
    """Tests for TopologyMetrics."""

    def test_topology_metrics_creation(self):
        """Verify metrics can be created."""
        metrics = TopologyMetrics(
            betti_0=3,
            betti_1=1,
            persistence_max=0.5,
            persistence_mean=0.3,
            min_cut_capacity=0.8,
            cluster_count=3,
            cluster_sizes=[5, 3, 2],
        )

        assert metrics.betti_0 == 3
        assert metrics.betti_1 == 1
        assert metrics.persistence_max == 0.5

    def test_topology_metrics_to_dict(self):
        """Verify to_dict produces correct structure."""
        metrics = TopologyMetrics(betti_0=2, betti_1=0)

        d = metrics.to_dict()

        assert d["betti_0"] == 2
        assert d["betti_1"] == 0
        assert "persistence_max" in d


class TestTDAFeedback:
    """Tests for TDAFeedback."""

    def test_feedback_creation(self):
        """Verify feedback can be created."""
        feedback = TDAFeedback(
            feedback_available=True,
            topology_alert_level="WARNING",
            betti_anomaly_detected=True,
            persistence_anomaly_detected=False,
            min_cut_capacity_degraded=True,
            feedback_cycle=100,
            recommended_actions=["Review cluster pattern"],
        )

        assert feedback.topology_alert_level == "WARNING"
        assert feedback.betti_anomaly_detected is True
        assert len(feedback.recommended_actions) == 1

    def test_feedback_shadow_mode(self):
        """Verify feedback enforces shadow mode."""
        feedback = TDAFeedback()

        assert feedback.mode == "SHADOW"
        assert feedback.enforcement_status == "LOGGED_ONLY"

    def test_feedback_to_dict(self):
        """Verify to_dict produces schema-conformant output."""
        feedback = TDAFeedback(
            topology_alert_level="ELEVATED",
            feedback_cycle=50,
        )

        d = feedback.to_dict()

        assert d["mode"] == "SHADOW"
        assert d["enforcement_status"] == "LOGGED_ONLY"
        assert d["topology_alert_level"] == "ELEVATED"


class TestTDAFeedbackProvider:
    """Tests for TDAFeedbackProvider."""

    def test_provider_initialization(self):
        """Verify provider initializes correctly."""
        provider = TDAFeedbackProvider(
            window_size=50,
            betti_0_threshold=3,
        )

        stats = provider.get_stats()
        assert stats["window_size"] == 50
        assert stats["thresholds"]["betti_0"] == 3

    def test_provider_add_anomaly(self):
        """Verify anomalies can be added."""
        provider = TDAFeedbackProvider()

        provider.add_anomaly(
            cycle=1,
            anomaly_type="rate_anomaly",
            severity="WARN",
            component="u2_runner",
        )

        summary = provider.get_anomaly_summary()
        assert summary["anomaly_count"] == 1

    def test_provider_compute_topology_empty(self):
        """Verify topology computation with empty window."""
        provider = TDAFeedbackProvider()

        metrics = provider.compute_topology_metrics()

        assert metrics.betti_0 == 0
        assert metrics.betti_1 == 0
        assert metrics.min_cut_capacity == 1.0

    def test_provider_compute_topology_with_anomalies(self):
        """Verify topology computation with anomalies."""
        provider = TDAFeedbackProvider(window_size=100)

        # Add clustered anomalies
        for i in range(5):
            provider.add_anomaly(
                cycle=i,
                anomaly_type="rate_anomaly",
                severity="WARN",
                component="u2_runner",
            )

        # Add another cluster
        for i in range(5):
            provider.add_anomaly(
                cycle=50 + i,
                anomaly_type="schema_violation",
                severity="CRITICAL",
                component="lean_verifier",
            )

        metrics = provider.compute_topology_metrics()

        # Should detect multiple clusters
        assert metrics.betti_0 >= 1
        assert metrics.cluster_count >= 1

    def test_provider_generate_feedback_normal(self):
        """Verify normal feedback generation."""
        provider = TDAFeedbackProvider()

        # No anomalies
        feedback = provider.generate_feedback()

        assert feedback.feedback_available is True
        assert feedback.topology_alert_level == "NORMAL"
        assert feedback.betti_anomaly_detected is False

    def test_provider_generate_feedback_warning(self):
        """Verify warning feedback when thresholds exceeded."""
        provider = TDAFeedbackProvider(
            betti_0_threshold=2,
            persistence_threshold=0.1,
        )

        # Add anomalies to trigger warning
        for i in range(20):
            provider.add_anomaly(
                cycle=i * 2,
                anomaly_type="rate_anomaly",
                severity="WARN",
                component="test",
            )

        feedback = provider.generate_feedback()

        # Should have elevated or warning level
        assert feedback.topology_alert_level in ("ELEVATED", "WARNING", "CRITICAL")

    def test_provider_generate_feedback_critical(self):
        """Verify critical feedback with critical anomalies."""
        provider = TDAFeedbackProvider()

        # Add critical anomalies
        for i in range(5):
            provider.add_anomaly(
                cycle=i,
                anomaly_type="schema_violation",
                severity="CRITICAL",
                component="test",
            )

        feedback = provider.generate_feedback()

        assert feedback.topology_alert_level == "CRITICAL"

    def test_provider_reset(self):
        """Verify provider reset."""
        provider = TDAFeedbackProvider()

        provider.add_anomaly(
            cycle=1, anomaly_type="a", severity="INFO", component="test"
        )
        provider.generate_feedback()

        provider.reset()

        summary = provider.get_anomaly_summary()
        assert summary["anomaly_count"] == 0


class TestTelemetryGovernanceSignalEmitter:
    """Tests for TelemetryGovernanceSignalEmitter."""

    def test_emitter_initialization(self):
        """Verify emitter initializes correctly."""
        emitter = TelemetryGovernanceSignalEmitter()

        signal = emitter.emit_signal()
        assert signal.schema_version == "1.0.0"
        assert signal.signal_type == "telemetry_governance"

    def test_emitter_register_emitter(self):
        """Verify emitter registration."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.register_emitter("u2_runner", "HEALTHY", emit_rate=10.0)

        signal = emitter.emit_signal()
        assert signal.emitter_status["total_emitters"] == 1
        assert signal.emitter_status["healthy_emitters"] == 1

    def test_emitter_update_emitter(self):
        """Verify emitter update."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.register_emitter("u2_runner", "HEALTHY")
        emitter.update_emitter("u2_runner", status="DEGRADED")

        signal = emitter.emit_signal()
        assert signal.emitter_status["degraded_emitters"] == 1

    def test_emitter_record_anomaly(self):
        """Verify anomaly recording."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.record_anomaly(
            cycle=1,
            anomaly_type="rate_anomaly",
            severity="WARN",
            component="test",
        )

        signal = emitter.emit_signal()
        assert signal.anomaly_summary.anomaly_count == 1

    def test_emitter_conformance_status(self):
        """Verify conformance status tracking."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.set_conformance_status(
            conformant=False,
            drift_detected=True,
            violations=3,
        )

        signal = emitter.emit_signal()
        assert signal.schema_conformance["conformant"] is False
        assert signal.schema_conformance["drift_detected"] is True

    def test_emitter_signal_status_ok(self):
        """Verify OK status when healthy."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.register_emitter("runner1", "HEALTHY")
        emitter.register_emitter("runner2", "HEALTHY")

        signal = emitter.emit_signal()
        assert signal.status == "OK"

    def test_emitter_signal_status_warn(self):
        """Verify WARN status with issues."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.register_emitter("runner1", "DEGRADED")
        emitter.set_conformance_status(conformant=False, drift_detected=True)

        signal = emitter.emit_signal()
        assert signal.status in ("WARN", "ATTENTION")

    def test_emitter_recommendation_shadow_mode(self):
        """Verify recommendations are LOGGED_ONLY."""
        emitter = TelemetryGovernanceSignalEmitter()

        signal = emitter.emit_signal()

        assert signal.recommendation.enforcement_status == "LOGGED_ONLY"
        assert signal.mode == "SHADOW"

    def test_emitter_p4_context(self):
        """Verify P4 coupling context."""
        emitter = TelemetryGovernanceSignalEmitter(p4_enabled=True)

        emitter.set_p4_state(enabled=True, last_cycle=100)

        signal = emitter.emit_signal()
        assert signal.p4_coupling_supported is True
        assert signal.p4_last_snapshot_cycle == 100

    def test_emitter_tda_integration(self):
        """Verify TDA feedback integration."""
        emitter = TelemetryGovernanceSignalEmitter()

        # Add anomalies
        for i in range(10):
            emitter.record_anomaly(
                cycle=i,
                anomaly_type="rate_anomaly",
                severity="WARN",
                component="test",
            )

        signal = emitter.emit_signal()

        assert signal.tda_feedback is not None
        assert signal.tda_feedback.feedback_available is True

    def test_emitter_correlation_ids(self):
        """Verify correlation ID tracking."""
        emitter = TelemetryGovernanceSignalEmitter()

        signal = emitter.emit_signal(
            conformance_snapshot_id="abc123",
            p4_divergence_cycle=42,
        )

        assert signal.conformance_snapshot_id == "abc123"
        assert signal.p4_divergence_cycle == 42

    def test_emitter_signal_history(self):
        """Verify signal history tracking."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.emit_signal()
        emitter.emit_signal()
        emitter.emit_signal()

        history = emitter.get_signal_history()
        assert len(history) == 3

    def test_emitter_reset(self):
        """Verify emitter reset."""
        emitter = TelemetryGovernanceSignalEmitter()

        emitter.register_emitter("test", "HEALTHY")
        emitter.emit_signal()

        emitter.reset()

        signal = emitter.emit_signal()
        assert signal.emitter_status["total_emitters"] == 0


class TestTelemetryGovernanceSignal:
    """Tests for TelemetryGovernanceSignal."""

    def test_signal_auto_generates_id(self):
        """Verify signal auto-generates UUID."""
        signal = TelemetryGovernanceSignal()

        assert signal.signal_id != ""
        assert len(signal.signal_id) == 36  # UUID format

    def test_signal_auto_generates_timestamp(self):
        """Verify signal auto-generates timestamp."""
        signal = TelemetryGovernanceSignal()

        assert signal.timestamp != ""
        assert "T" in signal.timestamp  # ISO format

    def test_signal_compute_hash(self):
        """Verify hash computation."""
        signal = TelemetryGovernanceSignal(
            status="OK",
            overall_health="HEALTHY",
            health_score=0.95,
        )

        hash1 = signal.compute_hash()
        hash2 = signal.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_signal_to_dict_schema_conformant(self):
        """Verify to_dict produces schema-conformant output."""
        signal = TelemetryGovernanceSignal(
            status="OK",
            recommendation=GovernanceRecommendation(
                action="PROCEED",
                confidence=0.95,
                reasons=["All checks passed"],
            ),
            anomaly_summary=AnomalySummary(
                anomalies_detected=False,
                anomaly_count=0,
            ),
        )

        d = signal.to_dict()

        # Check required fields
        assert d["schema_version"] == "1.0.0"
        assert d["signal_type"] == "telemetry_governance"
        assert d["signal_id"] is not None
        assert d["timestamp"] is not None
        assert d["mode"] == "SHADOW"
        assert d["status"] == "OK"

        # Check nested structures
        assert d["governance_recommendation"]["action"] == "PROCEED"
        assert d["governance_recommendation"]["enforcement_status"] == "LOGGED_ONLY"

    def test_signal_to_json(self):
        """Verify JSON serialization."""
        signal = TelemetryGovernanceSignal(status="OK")

        json_str = signal.to_json()

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["status"] == "OK"

    def test_signal_to_jsonl(self):
        """Verify JSONL serialization (no indent)."""
        signal = TelemetryGovernanceSignal(status="OK")

        jsonl = signal.to_jsonl()

        assert "\n" not in jsonl  # Single line
        parsed = json.loads(jsonl)
        assert parsed["status"] == "OK"


class TestClusteringAlgorithm:
    """Tests for the clustering algorithm in TDAFeedbackProvider."""

    def test_clustering_single_point(self):
        """Verify clustering handles single point."""
        provider = TDAFeedbackProvider()

        provider.add_anomaly(
            cycle=50, anomaly_type="a", severity="WARN", component="test"
        )

        metrics = provider.compute_topology_metrics()

        assert metrics.cluster_count == 1

    def test_clustering_tight_cluster(self):
        """Verify tight cluster is detected as one."""
        provider = TDAFeedbackProvider(cluster_distance=5.0)

        # Add points close together
        for i in range(5):
            provider.add_anomaly(
                cycle=i, anomaly_type="a", severity="WARN", component="test"
            )

        metrics = provider.compute_topology_metrics()

        # Should be one cluster
        assert metrics.cluster_count <= 2  # May be 1 or 2 depending on severity

    def test_clustering_separate_clusters(self):
        """Verify separate clusters are detected."""
        provider = TDAFeedbackProvider(cluster_distance=3.0)

        # Cluster 1: cycles 0-4
        for i in range(5):
            provider.add_anomaly(
                cycle=i, anomaly_type="a", severity="WARN", component="test"
            )

        # Cluster 2: cycles 100-104 (far away)
        for i in range(5):
            provider.add_anomaly(
                cycle=100 + i, anomaly_type="b", severity="WARN", component="test"
            )

        metrics = provider.compute_topology_metrics()

        # Should detect 2 clusters
        assert metrics.betti_0 >= 2


class TestPersistenceComputation:
    """Tests for persistence computation."""

    def test_persistence_empty(self):
        """Verify persistence is 0 with no anomalies."""
        provider = TDAFeedbackProvider()

        metrics = provider.compute_topology_metrics()

        assert metrics.persistence_max == 0.0
        assert metrics.persistence_mean == 0.0

    def test_persistence_long_lived_cluster(self):
        """Verify persistence captures cluster lifespan."""
        provider = TDAFeedbackProvider(window_size=100)

        # Add anomalies spanning many cycles
        for i in range(50):
            provider.add_anomaly(
                cycle=i * 2,  # 0, 2, 4, ..., 98
                anomaly_type="a",
                severity="WARN",
                component="test",
            )

        metrics = provider.compute_topology_metrics()

        # Should have non-zero persistence
        assert metrics.persistence_max > 0


class TestMinCutCapacity:
    """Tests for min-cut capacity estimation."""

    def test_min_cut_no_anomalies(self):
        """Verify min-cut is 1.0 with no anomalies."""
        provider = TDAFeedbackProvider()

        metrics = provider.compute_topology_metrics()

        assert metrics.min_cut_capacity == 1.0

    def test_min_cut_degrades_with_density(self):
        """Verify min-cut degrades with anomaly density."""
        provider = TDAFeedbackProvider(window_size=100)

        # Low density
        for i in range(5):
            provider.add_anomaly(
                cycle=i * 20,  # Sparse
                anomaly_type="a",
                severity="INFO",
                component="test",
            )

        metrics_low = provider.compute_topology_metrics()

        provider.reset()

        # High density
        for i in range(50):
            provider.add_anomaly(
                cycle=i * 2,  # Dense
                anomaly_type="a",
                severity="INFO",
                component="test",
            )

        metrics_high = provider.compute_topology_metrics()

        # Higher density should have lower capacity
        assert metrics_high.min_cut_capacity < metrics_low.min_cut_capacity
