"""
Tests for integration metrics and latency tracking.
"""

import pytest
import time
from backend.integration.metrics import (
    LatencyMeasurement,
    LatencyTracker,
    IntegrationMetrics
)


def test_latency_measurement_creation():
    """Test latency measurement creation."""
    measurement = LatencyMeasurement(
        operation="test_op",
        start_time=1.0,
        end_time=2.0,
        duration_ms=1000.0,
        success=True,
        error=None,
        metadata={"key": "value"}
    )

    assert measurement.operation == "test_op"
    assert measurement.duration_ms == 1000.0
    assert measurement.success is True
    assert measurement.metadata["key"] == "value"


def test_latency_measurement_to_dict():
    """Test latency measurement dictionary conversion."""
    measurement = LatencyMeasurement(
        operation="test_op",
        start_time=1.0,
        end_time=2.0,
        duration_ms=1000.0,
        success=True
    )

    data = measurement.to_dict()

    assert data["operation"] == "test_op"
    assert data["duration_ms"] == 1000.0
    assert data["success"] is True


def test_latency_tracker_track_success():
    """Test successful operation tracking."""
    tracker = LatencyTracker()

    with tracker.track("test_op", {"key": "value"}):
        time.sleep(0.01)

    assert len(tracker.measurements) == 1
    measurement = tracker.measurements[0]
    assert measurement.operation == "test_op"
    assert measurement.success is True
    assert measurement.duration_ms >= 10.0
    assert measurement.metadata["key"] == "value"


def test_latency_tracker_track_failure():
    """Test failed operation tracking."""
    tracker = LatencyTracker()

    with pytest.raises(ValueError):
        with tracker.track("test_op"):
            raise ValueError("Test error")

    assert len(tracker.measurements) == 1
    measurement = tracker.measurements[0]
    assert measurement.operation == "test_op"
    assert measurement.success is False
    assert "Test error" in measurement.error


def test_latency_tracker_get_stats_empty():
    """Test statistics for empty tracker."""
    tracker = LatencyTracker()
    stats = tracker.get_stats()

    assert stats["count"] == 0
    assert stats["mean_ms"] == 0.0
    assert stats["success_rate"] == 0.0


def test_latency_tracker_get_stats_with_data():
    """Test statistics calculation."""
    tracker = LatencyTracker()

    for i in range(10):
        with tracker.track("test_op"):
            time.sleep(0.001 * (i + 1))

    stats = tracker.get_stats("test_op")

    assert stats["count"] == 10
    assert stats["mean_ms"] > 0
    assert stats["min_ms"] > 0
    assert stats["max_ms"] > stats["min_ms"]
    assert stats["p50_ms"] > 0
    assert stats["p95_ms"] > 0
    assert stats["p99_ms"] > 0
    assert stats["success_rate"] == 100.0


def test_latency_tracker_get_stats_filtered():
    """Test filtered statistics."""
    tracker = LatencyTracker()

    with tracker.track("op1"):
        pass
    with tracker.track("op2"):
        pass
    with tracker.track("op1"):
        pass

    stats_op1 = tracker.get_stats("op1")
    stats_op2 = tracker.get_stats("op2")

    assert stats_op1["count"] == 2
    assert stats_op2["count"] == 1


def test_latency_tracker_clear():
    """Test tracker clearing."""
    tracker = LatencyTracker()

    with tracker.track("test_op"):
        pass

    assert len(tracker.measurements) == 1

    tracker.clear()

    assert len(tracker.measurements) == 0


def test_integration_metrics_initialization():
    """Test integration metrics initialization."""
    metrics = IntegrationMetrics()

    assert "fastapi_to_python" in metrics.trackers
    assert "python_to_db" in metrics.trackers
    assert "python_to_redis" in metrics.trackers
    assert "ui_to_fastapi" in metrics.trackers
    assert "end_to_end" in metrics.trackers


def test_integration_metrics_get_tracker():
    """Test getting tracker by component."""
    metrics = IntegrationMetrics()

    tracker = metrics.get_tracker("fastapi_to_python")

    assert isinstance(tracker, LatencyTracker)
    assert tracker is metrics.trackers["fastapi_to_python"]


def test_integration_metrics_get_tracker_new():
    """Test getting new tracker."""
    metrics = IntegrationMetrics()

    tracker = metrics.get_tracker("new_component")

    assert isinstance(tracker, LatencyTracker)
    assert "new_component" in metrics.trackers


def test_integration_metrics_generate_report_empty():
    """Test report generation with no data."""
    metrics = IntegrationMetrics()
    report = metrics.generate_report()

    assert "timestamp" in report
    assert "components" in report
    assert "summary" in report
    assert report["summary"]["total_operations"] == 0
    assert report["summary"]["latency_target_met"] is False


def test_integration_metrics_generate_report_with_data():
    """Test report generation with data."""
    metrics = IntegrationMetrics()

    tracker = metrics.get_tracker("fastapi_to_python")
    with tracker.track("test_op"):
        time.sleep(0.01)

    report = metrics.generate_report()

    assert report["summary"]["total_operations"] == 1
    assert report["summary"]["overall_success_rate"] == 100.0
    assert report["summary"]["max_latency_ms"] >= 10.0
    assert "fastapi_to_python" in report["components"]


def test_integration_metrics_latency_target():
    """Test latency target checking."""
    metrics = IntegrationMetrics()

    tracker = metrics.get_tracker("test_component")
    with tracker.track("fast_op"):
        time.sleep(0.001)

    report = metrics.generate_report()

    assert report["summary"]["max_latency_ms"] < 200.0
    assert report["summary"]["latency_target_met"] is True


def test_integration_metrics_save_report(tmp_path):
    """Test saving report to file."""
    metrics = IntegrationMetrics()

    tracker = metrics.get_tracker("test_component")
    with tracker.track("test_op"):
        pass

    report_path = tmp_path / "report.json"
    metrics.save_report(str(report_path))

    assert report_path.exists()

    import json
    with open(report_path) as f:
        report = json.load(f)

    assert "timestamp" in report
    assert "components" in report


def test_integration_metrics_clear_all():
    """Test clearing all trackers."""
    metrics = IntegrationMetrics()

    for component in ["fastapi_to_python", "python_to_db"]:
        tracker = metrics.get_tracker(component)
        with tracker.track("test_op"):
            pass

    metrics.clear_all()

    for tracker in metrics.trackers.values():
        assert len(tracker.measurements) == 0


def test_percentile_calculation():
    """Test percentile calculation in statistics."""
    tracker = LatencyTracker()

    for i in range(100):
        measurement = LatencyMeasurement(
            operation="test",
            start_time=0,
            end_time=0,
            duration_ms=float(i),
            success=True
        )
        tracker.measurements.append(measurement)

    stats = tracker.get_stats("test")

    assert stats["p50_ms"] >= 49.0
    assert stats["p50_ms"] <= 51.0
    assert stats["p95_ms"] >= 94.0
    assert stats["p95_ms"] <= 96.0
    assert stats["p99_ms"] >= 98.0
    assert stats["p99_ms"] <= 100.0
