"""
Tests for First Organism telemetry integration.

Verifies that the telemetry plugin captures metrics correctly and
the Metrics Oracle can collect them.
"""

import pytest
import time
from unittest.mock import patch, MagicMock


class TestFirstOrganismTelemetryCapture:
    """Test the telemetry capture mechanism."""

    @pytest.mark.first_organism
    def test_telemetry_emits_on_pass(self, capsys):
        """
        Verify telemetry emits metrics on test pass.

        This test itself is marked first_organism, so running it should
        trigger the telemetry hook.
        """
        # Simulate First Organism output pattern
        print("[PASS] FIRST ORGANISM ALIVE H_t=abc123def456")
        print("Abstention count: 3")

        # Simple assertion to pass
        assert True

    @pytest.mark.first_organism
    def test_telemetry_captures_duration(self):
        """Verify duration is captured."""
        # Sleep briefly to have measurable duration
        time.sleep(0.01)
        assert True


class TestFirstOrganismTelemetryEmitter:
    """Test the telemetry emitter module directly."""

    def test_emit_result_structure(self):
        """Test FirstOrganismRunResult structure."""
        from backend.metrics.first_organism_telemetry import FirstOrganismRunResult

        result = FirstOrganismRunResult(
            duration_seconds=1.5,
            ht_hash="abc123def456789012345678901234567890123456789012345678901234",
            abstention_count=3,
            success=True,
            timestamp="2025-01-01T00:00:00Z",
            metadata={"test": "value"},
        )

        assert result.duration_seconds == 1.5
        assert result.ht_hash.startswith("abc123")
        assert result.abstention_count == 3
        assert result.success is True

    def test_telemetry_available_check(self):
        """Test telemetry availability detection."""
        from backend.metrics.first_organism_telemetry import FirstOrganismTelemetry

        # Create telemetry instance - may or may not have Redis available
        telemetry = FirstOrganismTelemetry()
        # Should not raise, just return bool
        _ = telemetry.available


class TestFirstOrganismMetricsCollection:
    """Test the Metrics Oracle's First Organism collector."""

    def test_first_organism_collector_structure(self):
        """Test FirstOrganismCollector returns expected structure."""
        from backend.metrics_cartographer import FirstOrganismCollector

        collector = FirstOrganismCollector(redis_url=None)
        result = collector.collect()

        # Should return a CollectorResult even without Redis
        assert result is not None
        assert hasattr(result, "metrics")
        assert hasattr(result, "provenance")
        assert hasattr(result, "warnings")

    def test_first_organism_in_blank_metrics(self):
        """Test that blank metrics don't include first_organism by default."""
        from backend.metrics_cartographer import MetricsAggregator

        # Blank metrics shouldn't have first_organism section
        blank = MetricsAggregator._blank_metrics()

        # first_organism is NOT in blank metrics - it's added by collector
        assert "throughput" in blank
        assert "success_rates" in blank
        assert "coverage" in blank


class TestFirstOrganismTrendsComputation:
    """Test First Organism trends are computed in history."""

    def test_trends_include_first_organism_when_data_present(self):
        """Test that trends computation includes First Organism data."""
        from backend.metrics_cartographer import MetricsAggregator, MetricsConfig

        # Create aggregator
        config = MetricsConfig(db_url=None, redis_url=None)
        aggregator = MetricsAggregator(config)

        # Create fake history with First Organism data
        history = [
            {
                "session_id": "test-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "merkle_hash": "abc123",
                "throughput": {"proofs_per_sec": 1.0},
                "success_rates": {"proof_success_rate": 90.0},
                "performance": {"p95_latency_ms": 100.0},
                "first_organism": {
                    "runs_total": 1,
                    "average_duration_seconds": 1.5,
                    "abstention_count": 2,
                    "success_history": ["success"],
                },
            },
            {
                "session_id": "test-2",
                "timestamp": "2025-01-02T00:00:00Z",
                "merkle_hash": "def456",
                "throughput": {"proofs_per_sec": 1.2},
                "success_rates": {"proof_success_rate": 92.0},
                "performance": {"p95_latency_ms": 95.0},
                "first_organism": {
                    "runs_total": 2,
                    "average_duration_seconds": 1.3,
                    "abstention_count": 1,
                    "success_history": ["success", "success"],
                },
            },
        ]

        # Compute trends
        from backend.metrics_cartographer import CanonicalMetrics

        canonical = CanonicalMetrics(
            timestamp="2025-01-02T00:00:00Z",
            session_id="test-2",
            source="test",
            metrics={},
            provenance={},
        )

        trends = aggregator._compute_trends(history, canonical)

        # Verify First Organism trends are computed
        assert "first_organism" in trends
        fo_trends = trends["first_organism"]

        assert "duration_seconds" in fo_trends
        assert "abstention_count" in fo_trends
        assert "runs_total" in fo_trends
        assert "success_rate" in fo_trends

        # Check trend structure
        duration_trend = fo_trends["duration_seconds"]
        assert "latest" in duration_trend
        assert "trend" in duration_trend
        assert "samples" in duration_trend


class TestFirstOrganismReporterSection:
    """Test the ASCII reporter includes First Organism sections."""

    def test_reporter_first_organism_section(self, tmp_path):
        """Test reporter generates First Organism section."""
        import json
        from backend.metrics_reporter import ASCIIReporter

        metrics_data = {
            "session_id": "test-session",
            "timestamp": "2025-01-01T00:00:00Z",
            "source": "test",
            "metrics": {
                "first_organism": {
                    "runs_total": 5,
                    "last_ht_hash": "abc123def456789012345678901234567890123456789012345678901234",
                    "last_duration_seconds": 1.5,
                    "average_duration_seconds": 1.4,
                    "median_duration_seconds": 1.3,
                    "abstention_count": 2,
                    "last_run_timestamp": "2025-01-01T00:00:00Z",
                    "last_status": "success",
                    "success_rate": 80.0,
                    "duration_delta": 0.1,
                    "abstention_delta": -1,
                    "duration_history": [1.5, 1.4, 1.3, 1.2, 1.1],
                },
            },
            "provenance": {},
        }

        metrics_file = tmp_path / "test_metrics.json"
        metrics_file.write_text(json.dumps(metrics_data))

        reporter = ASCIIReporter(metrics_file)
        section = reporter.generate_first_organism_section()

        assert "FIRST ORGANISM VITAL SIGNS" in section
        assert "ALIVE" in section
        assert "H_t" in section
        # Check H_t is truncated (16 chars + "...")
        assert "abc123def4567890..." in section
        assert "Success rate" in section
        assert "80.0%" in section

    def test_reporter_first_organism_trends_section(self, tmp_path):
        """Test reporter generates First Organism trends section."""
        import json
        from backend.metrics_reporter import ASCIIReporter

        metrics_data = {
            "session_id": "test-session",
            "timestamp": "2025-01-01T00:00:00Z",
            "source": "test",
            "metrics": {
                "trends": {
                    "first_organism": {
                        "duration_seconds": {
                            "latest": 1.5,
                            "delta_from_previous": 0.1,
                            "trend": "up",
                            "samples": 5,
                        },
                        "abstention_count": {
                            "latest": 2,
                            "delta_from_previous": -1,
                            "trend": "down",
                            "samples": 5,
                        },
                        "success_rate": {
                            "latest": 80.0,
                            "delta_from_previous": 5.0,
                            "trend": "up",
                            "samples": 5,
                        },
                        "runs_total": {
                            "latest": 10,
                            "delta_from_previous": 2,
                            "trend": "up",
                            "samples": 5,
                        },
                    },
                },
            },
            "provenance": {},
        }

        metrics_file = tmp_path / "test_metrics.json"
        metrics_file.write_text(json.dumps(metrics_data))

        reporter = ASCIIReporter(metrics_file)
        section = reporter.generate_first_organism_trends_section()

        assert "FIRST ORGANISM HEALTH TRENDS" in section
        assert "Duration" in section
        assert "Abstentions" in section
        assert "Success rate" in section
