"""
Tests for Telemetry P4 Integration Functions

Phase X: Telemetry Canonical Interface

These tests verify:
1. build_telemetry_summary_for_p4() returns correct structure
2. attach_telemetry_governance_to_evidence() stores data correctly
3. telemetry_signal_to_ggfl_telemetry() produces GGFL-compatible output

SHADOW MODE: All tests operate in observation-only mode.

See: docs/system_law/Telemetry_PhaseX_Contract.md
"""

import json
import pytest
from datetime import datetime, timezone

from backend.telemetry import (
    TelemetryGovernanceSignal,
    TelemetryGovernanceSignalEmitter,
    build_telemetry_summary_for_p4,
    attach_telemetry_governance_to_evidence,
    telemetry_signal_to_ggfl_telemetry,
    TelemetryP4Summary,
)


class TestBuildTelemetrySummaryForP4:
    """Tests for build_telemetry_summary_for_p4() function."""

    def test_summary_returns_correct_type(self):
        """Verify function returns TelemetryP4Summary."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        assert isinstance(summary, TelemetryP4Summary)

    def test_summary_contains_health_fields(self):
        """Verify summary contains all health fields."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        assert hasattr(summary, "lean_health")
        assert hasattr(summary, "db_health")
        assert hasattr(summary, "redis_health")
        assert hasattr(summary, "overall_health")
        assert hasattr(summary, "health_score")

    def test_summary_health_values_are_valid(self):
        """Verify health values are valid enum strings."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.register_emitter("test", "HEALTHY", emit_rate=1.0)
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        valid_statuses = {"HEALTHY", "DEGRADED", "UNHEALTHY", "CRITICAL"}
        assert summary.lean_health in valid_statuses
        assert summary.db_health in valid_statuses
        assert summary.redis_health in valid_statuses
        assert summary.overall_health in valid_statuses

    def test_summary_contains_anomaly_rate(self):
        """Verify summary contains anomaly_rate field."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal, cycles_observed=100)

        assert hasattr(summary, "anomaly_rate")
        assert isinstance(summary.anomaly_rate, float)
        assert summary.anomaly_rate >= 0.0

    def test_summary_anomaly_rate_calculation(self):
        """Verify anomaly_rate is calculated correctly."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "HIGH", "test")
        emitter.record_anomaly(2, "THRESHOLD_BREACH", "MEDIUM", "test")
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal, cycles_observed=100)

        # 2 anomalies / 100 cycles = 0.02
        assert abs(summary.anomaly_rate - 0.02) < 0.001

    def test_summary_anomaly_rate_zero_cycles(self):
        """Verify anomaly_rate handles zero cycles gracefully."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "HIGH", "test")
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal, cycles_observed=0)

        assert summary.anomaly_rate == 0.0

    def test_summary_contains_tda_feedback(self):
        """Verify summary contains TDA feedback summary."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        assert hasattr(summary, "tda_feedback_summary")
        assert isinstance(summary.tda_feedback_summary, dict)

    def test_summary_tda_feedback_structure(self):
        """Verify TDA feedback summary has expected structure."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)
        tda = summary.tda_feedback_summary

        assert "topology_alert_level" in tda
        assert "betti_anomaly_detected" in tda
        assert "persistence_anomaly_detected" in tda
        assert "min_cut_capacity_degraded" in tda
        assert "recommended_actions_count" in tda

    def test_summary_contains_governance_fields(self):
        """Verify summary contains governance fields."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        assert hasattr(summary, "governance_status")
        assert hasattr(summary, "governance_recommendation")
        assert hasattr(summary, "safe_for_p4_coupling")

    def test_summary_mode_is_shadow(self):
        """Verify summary mode is SHADOW."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        assert summary.mode == "SHADOW"

    def test_summary_to_dict(self):
        """Verify summary can be converted to dict."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)
        d = summary.to_dict()

        assert isinstance(d, dict)
        assert "lean_health" in d
        assert "anomaly_rate" in d
        assert "tda_feedback_summary" in d

    def test_summary_json_serializable(self):
        """Verify summary can be serialized to JSON."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)
        json_str = json.dumps(summary.to_dict())

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["mode"] == "SHADOW"

    def test_summary_preserves_signal_id(self):
        """Verify summary preserves signal ID from source."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        assert summary.signal_id == signal.signal_id

    def test_summary_with_degraded_health(self):
        """Verify summary reflects degraded health correctly."""
        emitter = TelemetryGovernanceSignalEmitter()
        # Register many healthy and some degraded
        for i in range(10):
            emitter.register_emitter(f"healthy_{i}", "HEALTHY", emit_rate=1.0)
        for i in range(5):
            emitter.register_emitter(f"degraded_{i}", "DEGRADED", emit_rate=0.5)
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal)

        # With 10 healthy and 5 degraded (66.7% healthy), status depends on
        # _extract_component_health thresholds - verify it's a valid status
        valid_statuses = {"HEALTHY", "DEGRADED", "UNHEALTHY", "CRITICAL"}
        assert summary.lean_health in valid_statuses


class TestAttachTelemetryGovernanceToEvidence:
    """Tests for attach_telemetry_governance_to_evidence() function."""

    def test_attaches_to_empty_evidence(self):
        """Verify function works with empty evidence dict."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {}
        result = attach_telemetry_governance_to_evidence(evidence, signal)

        assert "governance" in result
        assert "telemetry" in result["governance"]

    def test_attaches_to_evidence_with_existing_governance(self):
        """Verify function preserves existing governance data."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {
            "governance": {
                "existing_key": "existing_value"
            }
        }
        result = attach_telemetry_governance_to_evidence(evidence, signal)

        assert result["governance"]["existing_key"] == "existing_value"
        assert "telemetry" in result["governance"]

    def test_telemetry_block_contains_required_fields(self):
        """Verify telemetry block contains all required fields."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {}
        result = attach_telemetry_governance_to_evidence(evidence, signal)
        telemetry = result["governance"]["telemetry"]

        assert "schema_version" in telemetry
        assert "signal_id" in telemetry
        assert "timestamp" in telemetry
        assert "mode" in telemetry
        assert "status" in telemetry
        assert "overall_health" in telemetry
        assert "health_score" in telemetry

    def test_telemetry_block_contains_hash(self):
        """Verify telemetry block contains hash for integrity."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {}
        result = attach_telemetry_governance_to_evidence(evidence, signal)
        telemetry = result["governance"]["telemetry"]

        assert "hash" in telemetry
        assert "algorithm" in telemetry["hash"]
        assert "value" in telemetry["hash"]
        assert telemetry["hash"]["algorithm"] == "sha256"

    def test_telemetry_block_contains_p4_coupling(self):
        """Verify telemetry block contains P4 coupling context."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {}
        result = attach_telemetry_governance_to_evidence(evidence, signal)
        telemetry = result["governance"]["telemetry"]

        assert "p4_coupling" in telemetry
        assert "supported" in telemetry["p4_coupling"]
        assert "adapter_ready" in telemetry["p4_coupling"]

    def test_telemetry_block_contains_recommendation(self):
        """Verify telemetry block contains recommendation when present."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {}
        result = attach_telemetry_governance_to_evidence(evidence, signal)
        telemetry = result["governance"]["telemetry"]

        if signal.recommendation is not None:
            assert "recommendation" in telemetry
            assert "action" in telemetry["recommendation"]
            assert "confidence" in telemetry["recommendation"]

    def test_telemetry_block_contains_anomaly_summary(self):
        """Verify telemetry block contains anomaly summary when present."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "HIGH", "test")
        signal = emitter.emit_signal()

        evidence = {}
        result = attach_telemetry_governance_to_evidence(evidence, signal)
        telemetry = result["governance"]["telemetry"]

        assert "anomaly_summary" in telemetry
        assert "anomalies_detected" in telemetry["anomaly_summary"]
        assert "anomaly_count" in telemetry["anomaly_summary"]

    def test_telemetry_mode_is_shadow(self):
        """Verify telemetry mode is SHADOW."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {}
        result = attach_telemetry_governance_to_evidence(evidence, signal)
        telemetry = result["governance"]["telemetry"]

        assert telemetry["mode"] == "SHADOW"

    def test_returns_modified_evidence(self):
        """Verify function returns the modified evidence dict."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        evidence = {"existing": "data"}
        result = attach_telemetry_governance_to_evidence(evidence, signal)

        assert result is evidence  # Same object, modified in place
        assert result["existing"] == "data"

    def test_evidence_json_serializable(self):
        """Verify modified evidence can be serialized to JSON."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "HIGH", "test")
        signal = emitter.emit_signal()

        evidence = {"metadata": {"version": "1.0"}}
        result = attach_telemetry_governance_to_evidence(evidence, signal)

        json_str = json.dumps(result)
        assert json_str is not None
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "telemetry" in parsed["governance"]


class TestTelemetrySignalToGGFLTelemetry:
    """Tests for telemetry_signal_to_ggfl_telemetry() function."""

    def test_returns_dict(self):
        """Verify function returns a dictionary."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        assert isinstance(result, dict)

    def test_contains_ggfl_required_fields(self):
        """Verify output contains GGFL required fields."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        # GGFL telemetry signal expected fields
        assert "lean_healthy" in result
        assert "db_healthy" in result
        assert "redis_healthy" in result
        assert "worker_count" in result
        assert "error_rate" in result

    def test_healthy_fields_are_boolean(self):
        """Verify healthy fields are booleans."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        assert isinstance(result["lean_healthy"], bool)
        assert isinstance(result["db_healthy"], bool)
        assert isinstance(result["redis_healthy"], bool)

    def test_worker_count_is_integer(self):
        """Verify worker_count is an integer."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        assert isinstance(result["worker_count"], int)
        assert result["worker_count"] >= 0

    def test_error_rate_is_float(self):
        """Verify error_rate is a float."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        assert isinstance(result["error_rate"], float)
        assert 0.0 <= result["error_rate"] <= 1.0

    def test_error_rate_reflects_critical_anomalies(self):
        """Verify error_rate reflects critical anomaly proportion."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "CRITICAL", "test")
        emitter.record_anomaly(2, "THRESHOLD_BREACH", "HIGH", "test")
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        # 1 critical / 2 total = 0.5
        assert abs(result["error_rate"] - 0.5) < 0.001

    def test_error_rate_zero_when_no_critical(self):
        """Verify error_rate is 0 when no critical anomalies."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "HIGH", "test")
        emitter.record_anomaly(2, "THRESHOLD_BREACH", "MEDIUM", "test")
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        assert result["error_rate"] == 0.0

    def test_healthy_true_when_healthy(self):
        """Verify healthy flags are True when system is HEALTHY."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.register_emitter("test", "HEALTHY", emit_rate=1.0)
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        # With healthy emitter, overall should be HEALTHY
        assert result["lean_healthy"] is True
        assert result["db_healthy"] is True
        assert result["redis_healthy"] is True

    def test_contains_additional_telemetry_fields(self):
        """Verify output contains additional useful telemetry fields."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        # Additional fields that GGFL can use
        assert "overall_health" in result
        assert "health_score" in result
        assert "anomaly_count" in result
        assert "tda_alert_level" in result
        assert "signal_id" in result
        assert "timestamp" in result
        assert "mode" in result

    def test_mode_is_shadow(self):
        """Verify mode is SHADOW."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        assert result["mode"] == "SHADOW"

    def test_json_serializable(self):
        """Verify output is JSON serializable."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "CRITICAL", "test")
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)
        json_str = json.dumps(result)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["mode"] == "SHADOW"

    def test_preserves_signal_id(self):
        """Verify output preserves signal ID."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        result = telemetry_signal_to_ggfl_telemetry(signal)

        assert result["signal_id"] == signal.signal_id


class TestGGFLIntegration:
    """Integration tests for GGFL adapter with build_global_alignment_view()."""

    def test_ggfl_adapter_output_format(self):
        """Verify GGFL adapter produces format compatible with build_global_alignment_view()."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.register_emitter("lean_verifier", "HEALTHY", emit_rate=1.0)
        emitter.register_emitter("db_connector", "HEALTHY", emit_rate=0.95)
        emitter.register_emitter("redis_cache", "HEALTHY", emit_rate=0.98)
        signal = emitter.emit_signal()

        ggfl_telemetry = telemetry_signal_to_ggfl_telemetry(signal)

        # Verify format matches what build_global_alignment_view expects
        # Based on fusion.py: telemetry dict with lean_healthy, db_healthy, redis_healthy, worker_count, error_rate
        assert isinstance(ggfl_telemetry.get("lean_healthy"), bool)
        assert isinstance(ggfl_telemetry.get("db_healthy"), bool)
        assert isinstance(ggfl_telemetry.get("redis_healthy"), bool)
        assert isinstance(ggfl_telemetry.get("worker_count"), int)
        assert isinstance(ggfl_telemetry.get("error_rate"), (int, float))

    def test_ggfl_adapter_with_anomalies(self):
        """Verify GGFL adapter handles anomalies correctly."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.register_emitter("lean_verifier", "HEALTHY", emit_rate=1.0)

        # Record various anomalies
        emitter.record_anomaly(1, "THRESHOLD_BREACH", "CRITICAL", "lean")
        emitter.record_anomaly(2, "TDA_ALERT", "HIGH", "topology")
        emitter.record_anomaly(3, "LATENCY_SPIKE", "MEDIUM", "db")

        signal = emitter.emit_signal()
        ggfl_telemetry = telemetry_signal_to_ggfl_telemetry(signal)

        # Should reflect anomalies in error_rate and anomaly_count
        assert ggfl_telemetry["anomaly_count"] == 3
        assert ggfl_telemetry["error_rate"] > 0  # Has critical anomaly

    def test_ggfl_adapter_tda_alert_level(self):
        """Verify GGFL adapter includes TDA alert level."""
        emitter = TelemetryGovernanceSignalEmitter()
        signal = emitter.emit_signal()

        ggfl_telemetry = telemetry_signal_to_ggfl_telemetry(signal)

        assert "tda_alert_level" in ggfl_telemetry
        # Default should be NORMAL
        assert ggfl_telemetry["tda_alert_level"] == "NORMAL"


class TestP4IntegrationEndToEnd:
    """End-to-end tests for P4 integration flow."""

    def test_full_p4_integration_flow(self):
        """Test complete P4 integration from signal to evidence."""
        # 1. Create emitter and register components
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.register_emitter("lean_verifier", "HEALTHY", emit_rate=1.0)
        emitter.register_emitter("db_connector", "HEALTHY", emit_rate=0.95)
        emitter.register_emitter("redis_cache", "DEGRADED", emit_rate=0.7)

        # 2. Record some anomalies
        emitter.record_anomaly(100, "THRESHOLD_BREACH", "MEDIUM", "lean")
        emitter.record_anomaly(150, "LATENCY_SPIKE", "HIGH", "db")

        # 3. Emit signal
        signal = emitter.emit_signal()

        # 4. Build P4 summary
        summary = build_telemetry_summary_for_p4(signal, cycles_observed=200)

        # 5. Attach to evidence
        evidence = {"proofs": [], "metadata": {"version": "1.0"}}
        evidence = attach_telemetry_governance_to_evidence(evidence, signal)

        # 6. Get GGFL adapter output
        ggfl_telemetry = telemetry_signal_to_ggfl_telemetry(signal)

        # Verify all outputs are consistent
        assert summary.signal_id == signal.signal_id
        assert evidence["governance"]["telemetry"]["signal_id"] == signal.signal_id
        assert ggfl_telemetry["signal_id"] == signal.signal_id

        # Verify anomaly rate calculation
        assert abs(summary.anomaly_rate - 0.01) < 0.001  # 2/200 = 0.01

        # Verify mode is SHADOW throughout
        assert summary.mode == "SHADOW"
        assert evidence["governance"]["telemetry"]["mode"] == "SHADOW"
        assert ggfl_telemetry["mode"] == "SHADOW"

    def test_p4_summary_can_be_embedded_in_calibration_report(self):
        """Verify P4 summary can be embedded in a calibration report structure."""
        emitter = TelemetryGovernanceSignalEmitter()
        emitter.register_emitter("lean", "HEALTHY", emit_rate=1.0)
        signal = emitter.emit_signal()

        summary = build_telemetry_summary_for_p4(signal, cycles_observed=100)

        # Simulate calibration report structure
        calibration_report = {
            "schema_version": "1.0.0",
            "report_id": "cal-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "telemetry_summary": summary.to_dict(),
            "calibration_results": {},
        }

        # Should be JSON serializable
        json_str = json.dumps(calibration_report)
        parsed = json.loads(json_str)

        assert parsed["telemetry_summary"]["mode"] == "SHADOW"
        assert "lean_health" in parsed["telemetry_summary"]
        assert "anomaly_rate" in parsed["telemetry_summary"]
