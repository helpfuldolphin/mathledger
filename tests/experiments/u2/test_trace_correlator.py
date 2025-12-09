# PHASE II — NOT RUN IN PHASE I
"""
Tests for TraceCorrelator

STATUS: PHASE II — NOT RUN IN PHASE I

Tests correlation functionality between trace logs, manifests, and budget health.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict

from experiments.u2.trace_correlator import TraceCorrelator, CorrelationSummary
from experiments.u2 import schema


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    """Create a temp directory for trace files."""
    return tmp_path


def write_trace(path: Path, events: list) -> None:
    """Write events to a trace log file."""
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")


def make_event(event_type: str, cycle: int = None, **extra) -> Dict[str, Any]:
    """Create a trace event dict."""
    payload = dict(extra)
    if cycle is not None:
        payload["cycle"] = cycle
    return {
        "ts": 1699123456.789 + (cycle or 0),
        "event_type": event_type,
        "schema_version": schema.TRACE_SCHEMA_VERSION,
        "payload": payload,
    }


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    """Write manifest JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def write_budget_health(path: Path, data: Dict[str, Any]) -> None:
    """Write budget health JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestTraceCorrelator:
    """Tests for TraceCorrelator class."""
    
    def test_basic_correlation(self, trace_dir: Path):
        """Test basic trace correlation with manifest."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        
        # Write trace with cycles 0-4
        events = [
            make_event("SessionStartEvent", run_id="test-run", slice_name="test"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": True}),
            make_event("CycleDurationEvent", cycle=0, duration_ms=50.0),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": True}),
            make_event("CycleDurationEvent", cycle=1, duration_ms=55.0),
            make_event("CycleTelemetryEvent", cycle=2, raw_record={"success": True}),
            make_event("CycleDurationEvent", cycle=2, duration_ms=60.0),
            make_event("CycleTelemetryEvent", cycle=3, raw_record={"success": True}),
            make_event("CycleDurationEvent", cycle=3, duration_ms=45.0),
            make_event("CycleTelemetryEvent", cycle=4, raw_record={"success": True}),
            make_event("CycleDurationEvent", cycle=4, duration_ms=52.0),
            make_event("SessionEndEvent", run_id="test-run", completed_cycles=5),
        ]
        write_trace(trace_path, events)
        
        # Write manifest expecting 5 cycles
        write_manifest(manifest_path, {
            "cycles": 5,
            "ht_series_length": 5,
            "mode": "baseline",
            "slice": "test_slice",
        })
        
        correlator = TraceCorrelator(trace_path, manifest_path)
        summary = correlator.correlate()
        
        assert summary.status == "OK"
        assert summary.coverage_status == "FULL"
        assert summary.coverage_percentage == 100.0
        assert len(summary.trace_cycles) == 5
        assert len(summary.missing_cycles) == 0
    
    def test_missing_cycles_detected(self, trace_dir: Path):
        """Test detection of missing cycles."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        
        # Write trace missing cycles 2 and 3
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("CycleTelemetryEvent", cycle=1),
            make_event("CycleTelemetryEvent", cycle=4),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        # Manifest expects 5 cycles
        write_manifest(manifest_path, {"cycles": 5})
        
        correlator = TraceCorrelator(trace_path, manifest_path)
        summary = correlator.correlate()
        
        assert summary.coverage_status == "PARTIAL"
        assert 2 in summary.missing_cycles
        assert 3 in summary.missing_cycles
        assert len(summary.missing_cycles) == 2
    
    def test_error_cycle_detection(self, trace_dir: Path):
        """Test detection of error cycles."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": True}),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": False}),  # Error
            make_event("CycleTelemetryEvent", cycle=2, raw_record={"success": True}),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        correlator = TraceCorrelator(trace_path)
        summary = correlator.correlate()
        
        assert 1 in summary.trace_error_cycles
        assert len(summary.trace_error_cycles) == 1
    
    def test_budget_health_correlation(self, trace_dir: Path):
        """Test correlation with budget health data."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        budget_path = trace_dir / "budget_health.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": True}),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": False}),  # Error
            make_event("CycleTelemetryEvent", cycle=2, raw_record={"success": True}),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        write_manifest(manifest_path, {"cycles": 3})
        
        # Budget exhausted on cycle 1
        write_budget_health(budget_path, {
            "exhausted_cycles": [1],
            "warnings": ["Cycle 1 budget exhausted"],
        })
        
        correlator = TraceCorrelator(trace_path, manifest_path, budget_path)
        summary = correlator.correlate()
        
        assert 1 in summary.budget_exhausted_cycles
        assert 1 in summary.errors_with_budget_exhaustion
        assert len(summary.errors_without_budget_exhaustion) == 0
    
    def test_errors_without_budget_exhaustion(self, trace_dir: Path):
        """Test error cycles without budget exhaustion."""
        trace_path = trace_dir / "trace.jsonl"
        budget_path = trace_dir / "budget_health.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": False}),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": True}),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        # No budget exhaustion
        write_budget_health(budget_path, {"exhausted_cycles": []})
        
        correlator = TraceCorrelator(trace_path, budget_health_path=budget_path)
        summary = correlator.correlate()
        
        assert 0 in summary.errors_without_budget_exhaustion
        assert len(summary.errors_with_budget_exhaustion) == 0
    
    def test_budget_exhaustion_without_errors(self, trace_dir: Path):
        """Test budget exhaustion without errors."""
        trace_path = trace_dir / "trace.jsonl"
        budget_path = trace_dir / "budget_health.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": True}),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": True}),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        # Budget exhausted but no errors
        write_budget_health(budget_path, {"exhausted_cycles": [0]})
        
        correlator = TraceCorrelator(trace_path, budget_health_path=budget_path)
        summary = correlator.correlate()
        
        assert 0 in summary.budget_exhaustion_without_errors
    
    def test_json_serialization(self, trace_dir: Path):
        """Test CorrelationSummary JSON serialization."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        correlator = TraceCorrelator(trace_path)
        summary = correlator.correlate()
        
        # Should serialize without error
        data = summary.to_dict()
        json_str = json.dumps(data)
        assert "trace_path" in json_str
        assert "coverage_status" in json_str
    
    def test_human_format(self, trace_dir: Path):
        """Test CorrelationSummary human-readable format."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        write_manifest(manifest_path, {"cycles": 1, "mode": "baseline"})
        
        correlator = TraceCorrelator(trace_path, manifest_path)
        summary = correlator.correlate()
        
        output = summary.format_human()
        assert "TRACE CORRELATION SUMMARY" in output
        assert "COVERAGE:" in output
        assert "Status:" in output
    
    def test_file_not_found_raises(self, trace_dir: Path):
        """Test FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            TraceCorrelator(trace_dir / "nonexistent.jsonl")
    
    def test_extra_cycles_detected(self, trace_dir: Path):
        """Test detection of extra cycles beyond manifest."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        
        # Trace has cycles 0-4, but manifest only expects 3
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("CycleTelemetryEvent", cycle=1),
            make_event("CycleTelemetryEvent", cycle=2),
            make_event("CycleTelemetryEvent", cycle=3),
            make_event("CycleTelemetryEvent", cycle=4),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        write_manifest(manifest_path, {"cycles": 3})
        
        correlator = TraceCorrelator(trace_path, manifest_path)
        summary = correlator.correlate()
        
        assert 3 in summary.extra_cycles
        assert 4 in summary.extra_cycles


class TestCorrelationCounts:
    """Tests verifying correlation count accuracy."""
    
    def test_correct_error_count(self, trace_dir: Path):
        """Test that error counts are accurate."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": False}),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": False}),
            make_event("CycleTelemetryEvent", cycle=2, raw_record={"success": False}),
            make_event("CycleTelemetryEvent", cycle=3, raw_record={"success": True}),
            make_event("CycleTelemetryEvent", cycle=4, raw_record={"success": True}),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        correlator = TraceCorrelator(trace_path)
        summary = correlator.correlate()
        
        assert len(summary.trace_error_cycles) == 3
        assert {0, 1, 2} == summary.trace_error_cycles
    
    def test_correct_budget_exhaustion_count(self, trace_dir: Path):
        """Test that budget exhaustion counts are accurate."""
        trace_path = trace_dir / "trace.jsonl"
        budget_path = trace_dir / "budget_health.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("CycleTelemetryEvent", cycle=1),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        write_budget_health(budget_path, {
            "exhausted_cycles": [0, 1, 5, 10],  # Some cycles not in trace
        })
        
        correlator = TraceCorrelator(trace_path, budget_health_path=budget_path)
        summary = correlator.correlate()
        
        # Should include all listed exhausted cycles
        assert len(summary.budget_exhausted_cycles) == 4
    
    def test_per_cycle_budget_status(self, trace_dir: Path):
        """Test extraction of per-cycle budget status."""
        trace_path = trace_dir / "trace.jsonl"
        budget_path = trace_dir / "budget_health.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("CycleTelemetryEvent", cycle=1),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        
        write_budget_health(budget_path, {
            "per_cycle": {
                "0": {"exhausted": False},
                "1": {"exhausted": True},
            },
        })
        
        correlator = TraceCorrelator(trace_path, budget_health_path=budget_path)
        summary = correlator.correlate()
        
        assert 1 in summary.budget_exhausted_cycles
        assert 0 not in summary.budget_exhausted_cycles


class TestStatusLabels:
    """Tests verifying status labels are correct."""
    
    def test_status_ok(self, trace_dir: Path):
        """Test OK status when everything matches."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": True}),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": True}),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        write_manifest(manifest_path, {"cycles": 2})
        
        correlator = TraceCorrelator(trace_path, manifest_path)
        summary = correlator.correlate()
        
        assert summary.status == "OK"
        assert len(summary.warnings) == 0
        assert len(summary.errors) == 0
    
    def test_status_warn_with_errors(self, trace_dir: Path):
        """Test WARN status when there are error cycles."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0, raw_record={"success": False}),
            make_event("CycleTelemetryEvent", cycle=1, raw_record={"success": True}),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        write_manifest(manifest_path, {"cycles": 2})
        
        correlator = TraceCorrelator(trace_path, manifest_path)
        summary = correlator.correlate()
        
        assert summary.status == "WARN"
        assert len(summary.warnings) > 0
    
    def test_status_error_missing_cycles(self, trace_dir: Path):
        """Test ERROR status when cycles are missing."""
        trace_path = trace_dir / "trace.jsonl"
        manifest_path = trace_dir / "manifest.json"
        
        events = [
            make_event("SessionStartEvent", run_id="test-run"),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("SessionEndEvent", run_id="test-run"),
        ]
        write_trace(trace_path, events)
        write_manifest(manifest_path, {"cycles": 3})
        
        correlator = TraceCorrelator(trace_path, manifest_path)
        summary = correlator.correlate()
        
        assert summary.status == "ERROR"
        assert len(summary.errors) > 0

