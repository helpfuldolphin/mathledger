# PHASE II — NOT RUN IN PHASE I
"""
Tests for trace health snapshot, trends, and global health integration.

STATUS: PHASE II — NOT RUN IN PHASE I

Tests:
- build_trace_health_snapshot contract
- build_trace_trend cross-run analysis
- summarize_trace_for_global_health integration
- analyze_trace_health one-shot helper
"""

import json
import pytest
from typing import Any, Dict, List

from experiments.u2.trace_health import (
    TRACE_HEALTH_SCHEMA_VERSION,
    TraceHealthSnapshot,
    TraceTrend,
    GlobalTraceHealth,
    build_trace_health_snapshot,
    build_trace_trend,
    summarize_trace_for_global_health,
    analyze_trace_health,
    _compute_trend_direction,
    _compute_change_pct,
    _compute_hotspot_stability,
)
from experiments.u2.trace_correlator import CorrelationSummary
from experiments.u2.inspector import HotspotReport, HotspotEntry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def sample_correlation_summary() -> CorrelationSummary:
    """Create a sample CorrelationSummary for testing."""
    return CorrelationSummary(
        trace_path="/path/to/trace.jsonl",
        trace_cycles={0, 1, 2, 3, 4},
        trace_total_records=100,
        trace_error_cycles={2, 4},
        coverage_status="FULL",
        coverage_percentage=100.0,
        status="OK",
    )


@pytest.fixture
def sample_hotspot_report() -> HotspotReport:
    """Create a sample HotspotReport for testing."""
    return HotspotReport(
        top_n=5,
        total_cycles=5,
        entries=[
            HotspotEntry(cycle=4, duration_ms=200.0, error_count=2, notes="⚠ long + 2 errs"),
            HotspotEntry(cycle=2, duration_ms=150.0, error_count=1, notes="long + 1 errs"),
            HotspotEntry(cycle=1, duration_ms=100.0, error_count=0, notes=""),
            HotspotEntry(cycle=3, duration_ms=80.0, error_count=0, notes=""),
            HotspotEntry(cycle=0, duration_ms=50.0, error_count=0, notes=""),
        ],
        avg_duration_ms=116.0,
        max_duration_ms=200.0,
        total_error_count=3,
        cycles_with_errors=2,
    )


def make_snapshot(
    status: str = "OK",
    coverage_pct: float = 100.0,
    max_duration: float = 100.0,
    error_count: int = 0,
    hotspot_cycles: List[int] = None,
) -> Dict[str, Any]:
    """Create a sample trace health snapshot for testing."""
    return {
        "schema_version": TRACE_HEALTH_SCHEMA_VERSION,
        "trace_status": status,
        "coverage_percentage": coverage_pct,
        "max_cycle_duration_ms": max_duration,
        "cycles_with_errors": 1 if error_count > 0 else 0,
        "total_error_count": error_count,
        "top_hotspot_cycles": hotspot_cycles or [0, 1, 2],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 1 — Trace Health Snapshot Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildTraceHealthSnapshot:
    """Tests for build_trace_health_snapshot."""
    
    def test_basic_snapshot(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test basic snapshot creation."""
        snapshot = build_trace_health_snapshot(
            sample_correlation_summary,
            sample_hotspot_report,
        )
        
        assert snapshot["schema_version"] == TRACE_HEALTH_SCHEMA_VERSION
        assert snapshot["trace_status"] == "OK"
        assert snapshot["coverage_percentage"] == 100.0
        assert snapshot["max_cycle_duration_ms"] == 200.0
        assert snapshot["cycles_with_errors"] == 2
        assert snapshot["total_error_count"] == 3
    
    def test_top_hotspot_cycles(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test that top hotspot cycles are extracted correctly."""
        snapshot = build_trace_health_snapshot(
            sample_correlation_summary,
            sample_hotspot_report,
            top_n_hotspots=3,
        )
        
        assert snapshot["top_hotspot_cycles"] == [4, 2, 1]
    
    def test_custom_top_n(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test custom top_n_hotspots parameter."""
        snapshot = build_trace_health_snapshot(
            sample_correlation_summary,
            sample_hotspot_report,
            top_n_hotspots=2,
        )
        
        assert len(snapshot["top_hotspot_cycles"]) == 2
        assert snapshot["top_hotspot_cycles"] == [4, 2]
    
    def test_with_run_id_and_timestamp(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test snapshot with run_id and timestamp."""
        snapshot = build_trace_health_snapshot(
            sample_correlation_summary,
            sample_hotspot_report,
            run_id="test-run-123",
            timestamp=1699123456.789,
        )
        
        assert snapshot["run_id"] == "test-run-123"
        assert snapshot["timestamp"] == 1699123456.789
    
    def test_json_serializable(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test that snapshot is JSON serializable."""
        snapshot = build_trace_health_snapshot(
            sample_correlation_summary,
            sample_hotspot_report,
        )
        
        # Should not raise
        json_str = json.dumps(snapshot)
        assert "schema_version" in json_str
    
    def test_warn_status_propagated(self, sample_hotspot_report: HotspotReport):
        """Test that WARN status is propagated from correlation summary."""
        warn_summary = CorrelationSummary(
            trace_path="/path/to/trace.jsonl",
            status="WARN",
            coverage_percentage=80.0,
        )
        
        snapshot = build_trace_health_snapshot(warn_summary, sample_hotspot_report)
        
        assert snapshot["trace_status"] == "WARN"
    
    def test_error_status_propagated(self, sample_hotspot_report: HotspotReport):
        """Test that ERROR status is propagated from correlation summary."""
        error_summary = CorrelationSummary(
            trace_path="/path/to/trace.jsonl",
            status="ERROR",
            coverage_percentage=50.0,
        )
        
        snapshot = build_trace_health_snapshot(error_summary, sample_hotspot_report)
        
        assert snapshot["trace_status"] == "ERROR"


class TestTraceHealthSnapshotDataclass:
    """Tests for TraceHealthSnapshot dataclass."""
    
    def test_to_dict(self):
        """Test TraceHealthSnapshot.to_dict()."""
        snapshot = TraceHealthSnapshot(
            schema_version=TRACE_HEALTH_SCHEMA_VERSION,
            trace_status="OK",
            coverage_percentage=100.0,
            max_cycle_duration_ms=200.0,
            cycles_with_errors=2,
            total_error_count=3,
            top_hotspot_cycles=[4, 2, 1],
            run_id="test-run",
            timestamp=1699123456.789,
        )
        
        data = snapshot.to_dict()
        
        assert data["schema_version"] == TRACE_HEALTH_SCHEMA_VERSION
        assert data["trace_status"] == "OK"
        assert data["top_hotspot_cycles"] == [4, 2, 1]
    
    def test_immutability(self):
        """Test that TraceHealthSnapshot is frozen."""
        snapshot = TraceHealthSnapshot(
            schema_version=TRACE_HEALTH_SCHEMA_VERSION,
            trace_status="OK",
            coverage_percentage=100.0,
            max_cycle_duration_ms=200.0,
            cycles_with_errors=0,
            total_error_count=0,
            top_hotspot_cycles=[],
        )
        
        with pytest.raises(AttributeError):
            snapshot.trace_status = "ERROR"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 2 — Cross-Run Trace Trends Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestComputeTrendDirection:
    """Tests for _compute_trend_direction helper."""
    
    def test_improving_trend(self):
        """Test detection of improving trend (values decreasing)."""
        values = [100.0, 80.0, 60.0, 40.0]
        assert _compute_trend_direction(values) == "IMPROVING"
    
    def test_degrading_trend(self):
        """Test detection of degrading trend (values increasing)."""
        values = [40.0, 60.0, 80.0, 100.0]
        assert _compute_trend_direction(values) == "DEGRADING"
    
    def test_stable_trend(self):
        """Test detection of stable trend (values within threshold)."""
        values = [100.0, 102.0, 98.0, 101.0]
        assert _compute_trend_direction(values, threshold_pct=10.0) == "STABLE"
    
    def test_unknown_single_value(self):
        """Test UNKNOWN for single value."""
        assert _compute_trend_direction([100.0]) == "UNKNOWN"
    
    def test_unknown_empty(self):
        """Test UNKNOWN for empty list."""
        assert _compute_trend_direction([]) == "UNKNOWN"
    
    def test_zero_to_positive_is_degrading(self):
        """Test that going from 0 to positive is DEGRADING."""
        values = [0.0, 5.0, 10.0]
        assert _compute_trend_direction(values) == "DEGRADING"
    
    def test_zero_to_zero_is_stable(self):
        """Test that staying at 0 is STABLE."""
        values = [0.0, 0.0, 0.0]
        assert _compute_trend_direction(values) == "STABLE"


class TestComputeChangePct:
    """Tests for _compute_change_pct helper."""
    
    def test_positive_change(self):
        """Test positive percentage change."""
        assert _compute_change_pct([100.0, 150.0]) == 50.0
    
    def test_negative_change(self):
        """Test negative percentage change."""
        assert _compute_change_pct([100.0, 50.0]) == -50.0
    
    def test_no_change(self):
        """Test zero percentage change."""
        assert _compute_change_pct([100.0, 100.0]) == 0.0
    
    def test_from_zero(self):
        """Test change from zero is None."""
        assert _compute_change_pct([0.0, 50.0]) is None
    
    def test_zero_to_zero(self):
        """Test zero to zero is 0%."""
        assert _compute_change_pct([0.0, 0.0]) == 0.0
    
    def test_single_value(self):
        """Test single value returns None."""
        assert _compute_change_pct([100.0]) is None


class TestComputeHotspotStability:
    """Tests for _compute_hotspot_stability helper."""
    
    def test_identical_hotspots(self):
        """Test stability = 1.0 for identical hotspots."""
        hotspot_lists = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        stability, persistent, frequency = _compute_hotspot_stability(hotspot_lists)
        
        assert stability == 1.0
        assert set(persistent) == {1, 2, 3}
    
    def test_completely_different_hotspots(self):
        """Test stability = 0.0 for completely different hotspots."""
        hotspot_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        stability, persistent, frequency = _compute_hotspot_stability(hotspot_lists)
        
        assert stability == 0.0
        assert persistent == []  # No cycle appears in >50% of runs
    
    def test_partially_overlapping(self):
        """Test partial overlap."""
        hotspot_lists = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        stability, persistent, frequency = _compute_hotspot_stability(hotspot_lists)
        
        assert 0.0 < stability < 1.0
        assert 3 in persistent  # Appears in all 3 runs
    
    def test_persistent_hotspots_threshold(self):
        """Test persistent hotspots require >50% presence."""
        hotspot_lists = [[1, 2], [1, 3], [1, 4], [5, 6]]
        stability, persistent, frequency = _compute_hotspot_stability(hotspot_lists)
        
        # Cycle 1 appears in 3/4 = 75% of runs
        assert 1 in persistent
    
    def test_empty_list(self):
        """Test empty hotspot lists."""
        stability, persistent, frequency = _compute_hotspot_stability([])
        
        assert stability == 0.0
        assert persistent == []
        assert frequency == {}
    
    def test_single_run(self):
        """Test single run returns stability 1.0."""
        hotspot_lists = [[1, 2, 3]]
        stability, persistent, frequency = _compute_hotspot_stability(hotspot_lists)
        
        assert stability == 1.0
        assert persistent == [1, 2, 3]


class TestBuildTraceTrend:
    """Tests for build_trace_trend."""
    
    def test_empty_history(self):
        """Test with empty history."""
        trend = build_trace_trend([])
        
        assert trend["run_count"] == 0
        assert trend["duration_trend"] == "UNKNOWN"
        assert trend["error_trend"] == "UNKNOWN"
    
    def test_single_run(self):
        """Test with single run (no trend)."""
        history = [make_snapshot(max_duration=100.0, error_count=5)]
        trend = build_trace_trend(history)
        
        assert trend["run_count"] == 1
        assert trend["duration_values"] == [100.0]
        assert trend["error_counts"] == [5]
    
    def test_improving_duration_trend(self):
        """Test detection of improving duration trend."""
        history = [
            make_snapshot(max_duration=200.0),
            make_snapshot(max_duration=150.0),
            make_snapshot(max_duration=100.0),
        ]
        trend = build_trace_trend(history)
        
        assert trend["duration_trend"] == "IMPROVING"
        assert trend["duration_change_pct"] == -50.0
    
    def test_degrading_error_trend(self):
        """Test detection of degrading error trend."""
        history = [
            make_snapshot(error_count=0),
            make_snapshot(error_count=5),
            make_snapshot(error_count=10),
        ]
        trend = build_trace_trend(history)
        
        assert trend["error_trend"] == "DEGRADING"
    
    def test_stable_trends(self):
        """Test detection of stable trends."""
        history = [
            make_snapshot(max_duration=100.0, error_count=2),
            make_snapshot(max_duration=102.0, error_count=2),
            make_snapshot(max_duration=99.0, error_count=2),
        ]
        trend = build_trace_trend(history, trend_threshold_pct=10.0)
        
        assert trend["duration_trend"] == "STABLE"
        assert trend["error_trend"] == "STABLE"
    
    def test_hotspot_stability_tracked(self):
        """Test that hotspot stability is tracked."""
        history = [
            make_snapshot(hotspot_cycles=[1, 2, 3]),
            make_snapshot(hotspot_cycles=[2, 3, 4]),
            make_snapshot(hotspot_cycles=[3, 4, 5]),
        ]
        trend = build_trace_trend(history)
        
        assert 0.0 < trend["hotspot_stability"] < 1.0
        assert 3 in trend["persistent_hotspots"]
    
    def test_hotspot_frequency_tracked(self):
        """Test that hotspot frequency is tracked."""
        history = [
            make_snapshot(hotspot_cycles=[1, 2]),
            make_snapshot(hotspot_cycles=[1, 3]),
            make_snapshot(hotspot_cycles=[1, 4]),
        ]
        trend = build_trace_trend(history)
        
        assert trend["hotspot_frequency"][1] == 3
        assert trend["hotspot_frequency"].get(2, 0) == 1
    
    def test_json_serializable(self):
        """Test that trend is JSON serializable."""
        history = [make_snapshot(), make_snapshot(), make_snapshot()]
        trend = build_trace_trend(history)
        
        # Should not raise
        json_str = json.dumps(trend)
        assert "duration_trend" in json_str


class TestTraceTrendDataclass:
    """Tests for TraceTrend dataclass."""
    
    def test_to_dict(self):
        """Test TraceTrend.to_dict()."""
        trend = TraceTrend(
            schema_version=TRACE_HEALTH_SCHEMA_VERSION,
            run_count=3,
            duration_trend="IMPROVING",
            duration_values=[200.0, 150.0, 100.0],
            duration_change_pct=-50.0,
            error_trend="STABLE",
            error_counts=[2, 2, 2],
            error_change_pct=0.0,
            hotspot_stability=0.8,
            persistent_hotspots=[1, 2],
            hotspot_frequency={1: 3, 2: 3, 3: 2},
        )
        
        data = trend.to_dict()
        
        assert data["duration_trend"] == "IMPROVING"
        assert data["persistent_hotspots"] == [1, 2]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK 3 — Global Health Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSummarizeTraceForGlobalHealth:
    """Tests for summarize_trace_for_global_health."""
    
    def test_trace_ok_when_status_ok(self):
        """Test trace_ok is True when status is OK."""
        snapshot = make_snapshot(status="OK")
        summary = summarize_trace_for_global_health(snapshot)
        
        assert summary["trace_ok"] is True
        assert summary["trace_status"] == "OK"
    
    def test_trace_not_ok_when_warn(self):
        """Test trace_ok is False when status is WARN."""
        snapshot = make_snapshot(status="WARN")
        summary = summarize_trace_for_global_health(snapshot)
        
        assert summary["trace_ok"] is False
        assert summary["trace_status"] == "WARN"
    
    def test_trace_not_ok_when_error(self):
        """Test trace_ok is False when status is ERROR."""
        snapshot = make_snapshot(status="ERROR")
        summary = summarize_trace_for_global_health(snapshot)
        
        assert summary["trace_ok"] is False
        assert summary["trace_status"] == "ERROR"
    
    def test_without_trend(self):
        """Test summary without trend data."""
        snapshot = make_snapshot(hotspot_cycles=[1, 2, 3])
        summary = summarize_trace_for_global_health(snapshot)
        
        assert summary["persistent_hotspots"] == [1, 2, 3]
        assert summary["hotspot_stability"] == 1.0
        assert summary["error_trend"] == "UNKNOWN"
        assert summary["duration_trend"] == "UNKNOWN"
        assert summary["run_count"] == 1
    
    def test_with_trend(self):
        """Test summary with trend data."""
        snapshot = make_snapshot()
        trend = {
            "persistent_hotspots": [42, 100],
            "hotspot_stability": 0.75,
            "error_trend": "IMPROVING",
            "duration_trend": "STABLE",
            "run_count": 5,
        }
        
        summary = summarize_trace_for_global_health(snapshot, trend)
        
        assert summary["persistent_hotspots"] == [42, 100]
        assert summary["hotspot_stability"] == 0.75
        assert summary["error_trend"] == "IMPROVING"
        assert summary["duration_trend"] == "STABLE"
        assert summary["run_count"] == 5
    
    def test_metrics_from_snapshot(self):
        """Test that metrics come from snapshot."""
        snapshot = make_snapshot(
            coverage_pct=95.5,
            max_duration=250.0,
            error_count=5,
        )
        summary = summarize_trace_for_global_health(snapshot)
        
        assert summary["coverage_percentage"] == 95.5
        assert summary["max_cycle_duration_ms"] == 250.0
    
    def test_json_serializable(self):
        """Test that summary is JSON serializable."""
        snapshot = make_snapshot()
        summary = summarize_trace_for_global_health(snapshot)
        
        # Should not raise
        json_str = json.dumps(summary)
        assert "trace_ok" in json_str


class TestGlobalTraceHealthDataclass:
    """Tests for GlobalTraceHealth dataclass."""
    
    def test_to_dict(self):
        """Test GlobalTraceHealth.to_dict()."""
        summary = GlobalTraceHealth(
            schema_version=TRACE_HEALTH_SCHEMA_VERSION,
            trace_ok=True,
            trace_status="OK",
            persistent_hotspots=[42],
            hotspot_stability=0.9,
            error_trend="IMPROVING",
            duration_trend="STABLE",
            coverage_percentage=100.0,
            max_cycle_duration_ms=150.0,
            cycles_with_errors=1,
            run_count=3,
        )
        
        data = summary.to_dict()
        
        assert data["trace_ok"] is True
        assert data["persistent_hotspots"] == [42]
        assert data["error_trend"] == "IMPROVING"
    
    def test_immutability(self):
        """Test that GlobalTraceHealth is frozen."""
        summary = GlobalTraceHealth(
            schema_version=TRACE_HEALTH_SCHEMA_VERSION,
            trace_ok=True,
            trace_status="OK",
            persistent_hotspots=[],
            hotspot_stability=1.0,
            error_trend="STABLE",
            duration_trend="STABLE",
            coverage_percentage=100.0,
            max_cycle_duration_ms=100.0,
            cycles_with_errors=0,
            run_count=1,
        )
        
        with pytest.raises(AttributeError):
            summary.trace_ok = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAnalyzeTraceHealth:
    """Tests for analyze_trace_health one-shot helper."""
    
    def test_returns_all_components(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test that analyze_trace_health returns all components."""
        result = analyze_trace_health(
            sample_correlation_summary,
            sample_hotspot_report,
        )
        
        assert "snapshot" in result
        assert "trend" in result
        assert "global_health" in result
    
    def test_without_history(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test without history (trend is None)."""
        result = analyze_trace_health(
            sample_correlation_summary,
            sample_hotspot_report,
        )
        
        assert result["trend"] is None
        assert result["global_health"]["run_count"] == 1
    
    def test_with_history(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test with history (includes trend)."""
        history = [
            make_snapshot(max_duration=150.0),
            make_snapshot(max_duration=180.0),
        ]
        
        result = analyze_trace_health(
            sample_correlation_summary,
            sample_hotspot_report,
            history=history,
        )
        
        assert result["trend"] is not None
        assert result["trend"]["run_count"] == 3  # 2 historical + 1 current
    
    def test_run_id_and_timestamp_passed(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test that run_id and timestamp are passed to snapshot."""
        result = analyze_trace_health(
            sample_correlation_summary,
            sample_hotspot_report,
            run_id="test-run-456",
            timestamp=1699123456.0,
        )
        
        assert result["snapshot"]["run_id"] == "test-run-456"
        assert result["snapshot"]["timestamp"] == 1699123456.0


class TestEndToEndScenarios:
    """End-to-end tests for realistic scenarios."""
    
    def test_healthy_experiment_multi_run(self):
        """Test a healthy experiment over multiple runs."""
        # Simulate 3 runs with stable, good performance
        history = [
            make_snapshot(status="OK", max_duration=100.0, error_count=0, hotspot_cycles=[1, 2, 3]),
            make_snapshot(status="OK", max_duration=105.0, error_count=0, hotspot_cycles=[1, 2, 3]),
            make_snapshot(status="OK", max_duration=98.0, error_count=0, hotspot_cycles=[1, 2, 3]),
        ]
        
        trend = build_trace_trend(history)
        global_health = summarize_trace_for_global_health(history[-1], trend)
        
        assert global_health["trace_ok"] is True
        assert global_health["error_trend"] == "STABLE"
        assert global_health["duration_trend"] == "STABLE"
        assert global_health["hotspot_stability"] == 1.0
    
    def test_degrading_experiment_multi_run(self):
        """Test an experiment with degrading performance."""
        # Simulate 3 runs with increasing errors and duration
        history = [
            make_snapshot(status="OK", max_duration=100.0, error_count=0),
            make_snapshot(status="WARN", max_duration=150.0, error_count=3),
            make_snapshot(status="ERROR", max_duration=200.0, error_count=10),
        ]
        
        trend = build_trace_trend(history)
        global_health = summarize_trace_for_global_health(history[-1], trend)
        
        assert global_health["trace_ok"] is False
        assert global_health["error_trend"] == "DEGRADING"
        assert global_health["duration_trend"] == "DEGRADING"
    
    def test_improving_experiment_multi_run(self):
        """Test an experiment showing improvement."""
        # Simulate 3 runs with decreasing errors
        history = [
            make_snapshot(status="ERROR", max_duration=200.0, error_count=10),
            make_snapshot(status="WARN", max_duration=150.0, error_count=5),
            make_snapshot(status="OK", max_duration=100.0, error_count=0),
        ]
        
        trend = build_trace_trend(history)
        global_health = summarize_trace_for_global_health(history[-1], trend)
        
        assert global_health["trace_ok"] is True
        assert global_health["error_trend"] == "IMPROVING"
        assert global_health["duration_trend"] == "IMPROVING"
    
    def test_unstable_hotspots(self):
        """Test detection of unstable hotspots."""
        # Simulate 4 runs with completely different hotspots each time
        history = [
            make_snapshot(hotspot_cycles=[1, 2, 3]),
            make_snapshot(hotspot_cycles=[4, 5, 6]),
            make_snapshot(hotspot_cycles=[7, 8, 9]),
            make_snapshot(hotspot_cycles=[10, 11, 12]),
        ]
        
        trend = build_trace_trend(history)
        global_health = summarize_trace_for_global_health(history[-1], trend)
        
        assert trend["hotspot_stability"] == 0.0
        assert len(trend["persistent_hotspots"]) == 0


class TestDeterminism:
    """Tests verifying deterministic behavior."""
    
    def test_snapshot_determinism(
        self,
        sample_correlation_summary: CorrelationSummary,
        sample_hotspot_report: HotspotReport,
    ):
        """Test that snapshot building is deterministic."""
        results = []
        for _ in range(5):
            snapshot = build_trace_health_snapshot(
                sample_correlation_summary,
                sample_hotspot_report,
            )
            results.append(json.dumps(snapshot, sort_keys=True))
        
        assert all(r == results[0] for r in results)
    
    def test_trend_determinism(self):
        """Test that trend building is deterministic."""
        history = [
            make_snapshot(max_duration=100.0, hotspot_cycles=[1, 2, 3]),
            make_snapshot(max_duration=150.0, hotspot_cycles=[2, 3, 4]),
            make_snapshot(max_duration=200.0, hotspot_cycles=[3, 4, 5]),
        ]
        
        results = []
        for _ in range(5):
            trend = build_trace_trend(history)
            results.append(json.dumps(trend, sort_keys=True))
        
        assert all(r == results[0] for r in results)
    
    def test_global_health_determinism(self):
        """Test that global health summary is deterministic."""
        snapshot = make_snapshot()
        trend = {
            "persistent_hotspots": [42],
            "hotspot_stability": 0.8,
            "error_trend": "STABLE",
            "duration_trend": "STABLE",
            "run_count": 3,
        }
        
        results = []
        for _ in range(5):
            summary = summarize_trace_for_global_health(snapshot, trend)
            results.append(json.dumps(summary, sort_keys=True))
        
        assert all(r == results[0] for r in results)

