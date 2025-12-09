# PHASE II — NOT RUN IN PHASE I
"""
Tests for hotspot detection in TraceLogInspector

STATUS: PHASE II — NOT RUN IN PHASE I

Tests hotspot ranking logic, tie-breaking behavior, and report formatting.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict

from experiments.u2.inspector import (
    TraceLogInspector,
    HotspotEntry,
    HotspotReport,
)
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


def make_duration_event(cycle: int, duration_ms: float) -> Dict[str, Any]:
    """Create a CycleDurationEvent."""
    return {
        "ts": 1699123456.789 + cycle,
        "event_type": "CycleDurationEvent",
        "schema_version": schema.TRACE_SCHEMA_VERSION,
        "payload": {
            "cycle": cycle,
            "duration_ms": duration_ms,
            "mode": "baseline",
            "slice_name": "test",
        },
    }


def make_error_event(cycle: int) -> Dict[str, Any]:
    """Create a CycleTelemetryEvent with an error."""
    return {
        "ts": 1699123456.789 + cycle,
        "event_type": "CycleTelemetryEvent",
        "schema_version": schema.TRACE_SCHEMA_VERSION,
        "payload": {
            "cycle": cycle,
            "raw_record": {"success": False, "error": "test error"},
        },
    }


def make_success_event(cycle: int) -> Dict[str, Any]:
    """Create a CycleTelemetryEvent with success."""
    return {
        "ts": 1699123456.789 + cycle,
        "event_type": "CycleTelemetryEvent",
        "schema_version": schema.TRACE_SCHEMA_VERSION,
        "payload": {
            "cycle": cycle,
            "raw_record": {"success": True},
        },
    }


class TestHotspotRanking:
    """Tests for hotspot ranking logic."""
    
    def test_ranks_by_duration_descending(self, trace_dir: Path):
        """Test that hotspots are ranked by duration (descending)."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 50.0),
            make_duration_event(1, 100.0),  # Longest
            make_duration_event(2, 30.0),
            make_duration_event(3, 80.0),
            make_duration_event(4, 20.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=5)
        
        assert len(report.entries) == 5
        durations = [e.duration_ms for e in report.entries]
        assert durations == sorted(durations, reverse=True)
        assert report.entries[0].cycle == 1
        assert report.entries[0].duration_ms == 100.0
    
    def test_top_n_limits_results(self, trace_dir: Path):
        """Test that top_n limits the number of results."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [make_duration_event(i, 50.0 + i) for i in range(20)]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=5)
        
        assert len(report.entries) == 5
    
    def test_fewer_cycles_than_top_n(self, trace_dir: Path):
        """Test when there are fewer cycles than top_n."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 50.0),
            make_duration_event(1, 100.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert len(report.entries) == 2


class TestTieBreaking:
    """Tests for tie-breaking behavior."""
    
    def test_tie_break_by_cycle_number_ascending(self, trace_dir: Path):
        """Test that ties are broken by cycle number (ascending)."""
        trace_path = trace_dir / "trace.jsonl"
        
        # All have same duration
        events = [
            make_duration_event(5, 100.0),
            make_duration_event(2, 100.0),
            make_duration_event(8, 100.0),
            make_duration_event(1, 100.0),
            make_duration_event(3, 100.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=5)
        
        cycles = [e.cycle for e in report.entries]
        assert cycles == [1, 2, 3, 5, 8]
    
    def test_tie_break_preserves_duration_priority(self, trace_dir: Path):
        """Test that duration priority is preserved even with ties."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 50.0),   # Low duration
            make_duration_event(1, 100.0),  # High duration (tied)
            make_duration_event(2, 100.0),  # High duration (tied)
            make_duration_event(3, 75.0),   # Medium duration
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=4)
        
        # Should be: cycle 1, cycle 2 (both 100ms), cycle 3 (75ms), cycle 0 (50ms)
        assert report.entries[0].cycle == 1
        assert report.entries[1].cycle == 2
        assert report.entries[2].cycle == 3
        assert report.entries[3].cycle == 0


class TestErrorCounting:
    """Tests for error counting in hotspots."""
    
    def test_counts_errors_per_cycle(self, trace_dir: Path):
        """Test that errors are counted per cycle."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 100.0),
            make_error_event(0),
            make_error_event(0),  # Two errors
            make_duration_event(1, 80.0),
            make_success_event(1),  # No errors
            make_duration_event(2, 60.0),
            make_error_event(2),  # One error
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=3)
        
        # Find entries by cycle
        entry_map = {e.cycle: e for e in report.entries}
        
        assert entry_map[0].error_count >= 2
        assert entry_map[1].error_count == 0
        assert entry_map[2].error_count >= 1
    
    def test_cycles_with_errors_count(self, trace_dir: Path):
        """Test total count of cycles with errors."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 100.0),
            make_error_event(0),
            make_duration_event(1, 80.0),
            make_success_event(1),
            make_duration_event(2, 60.0),
            make_error_event(2),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert report.cycles_with_errors == 2
    
    def test_total_error_count(self, trace_dir: Path):
        """Test total error count across all cycles."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 100.0),
            make_error_event(0),
            make_error_event(0),
            make_duration_event(1, 80.0),
            make_error_event(1),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert report.total_error_count >= 3


class TestStatistics:
    """Tests for aggregate statistics."""
    
    def test_average_duration(self, trace_dir: Path):
        """Test average duration calculation."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 50.0),
            make_duration_event(1, 100.0),
            make_duration_event(2, 150.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert report.avg_duration_ms == 100.0
    
    def test_max_duration(self, trace_dir: Path):
        """Test max duration tracking."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 50.0),
            make_duration_event(1, 200.0),
            make_duration_event(2, 100.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert report.max_duration_ms == 200.0
    
    def test_total_cycles(self, trace_dir: Path):
        """Test total cycle count."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [make_duration_event(i, 50.0 + i) for i in range(7)]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert report.total_cycles == 7


class TestNotes:
    """Tests for hotspot notes generation."""
    
    def test_long_duration_note(self, trace_dir: Path):
        """Test 'long' note for durations > 1.5x average."""
        trace_path = trace_dir / "trace.jsonl"
        
        # Average will be 100, so 160 > 150 (1.5x avg)
        events = [
            make_duration_event(0, 160.0),  # > 1.5x avg
            make_duration_event(1, 80.0),
            make_duration_event(2, 60.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=3)
        
        # First entry should have 'long' in notes
        assert "long" in report.entries[0].notes.lower()
    
    def test_very_long_duration_note(self, trace_dir: Path):
        """Test warning note for durations > 2x average."""
        trace_path = trace_dir / "trace.jsonl"
        
        # Average will be 100, so 250 > 200 (2x avg)
        events = [
            make_duration_event(0, 250.0),  # > 2x avg
            make_duration_event(1, 50.0),
            make_duration_event(2, 50.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=3)
        
        # First entry should have warning indicator
        assert "⚠" in report.entries[0].notes or "long" in report.entries[0].notes.lower()
    
    def test_error_count_in_notes(self, trace_dir: Path):
        """Test that error count appears in notes."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 100.0),
            make_error_event(0),
            make_error_event(0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=1)
        
        assert "err" in report.entries[0].notes.lower()
    
    def test_combined_notes(self, trace_dir: Path):
        """Test combined notes for long duration + errors."""
        trace_path = trace_dir / "trace.jsonl"
        
        # Average = 100, so 250 > 2x avg
        events = [
            make_duration_event(0, 250.0),
            make_error_event(0),
            make_duration_event(1, 50.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=1)
        
        notes = report.entries[0].notes.lower()
        assert "long" in notes or "⚠" in notes
        assert "err" in notes


class TestHotspotReportFormatting:
    """Tests for HotspotReport formatting."""
    
    def test_to_dict_structure(self, trace_dir: Path):
        """Test HotspotReport.to_dict() structure."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 100.0),
            make_duration_event(1, 50.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=5)
        
        data = report.to_dict()
        
        assert "top_n" in data
        assert "total_cycles" in data
        assert "entries" in data
        assert "avg_duration_ms" in data
        assert "max_duration_ms" in data
        assert "total_error_count" in data
        assert "cycles_with_errors" in data
    
    def test_json_serialization(self, trace_dir: Path):
        """Test that report can be serialized to JSON."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [make_duration_event(0, 100.0)]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=5)
        
        # Should not raise
        json_str = json.dumps(report.to_dict())
        assert "top_n" in json_str
    
    def test_human_format_contains_table(self, trace_dir: Path):
        """Test that human format contains table structure."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(0, 100.0),
            make_duration_event(1, 50.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=5)
        
        output = report.format_human()
        
        assert "HOTSPOT ANALYSIS" in output
        assert "Cycle" in output
        assert "duration_ms" in output
        assert "error_count" in output
    
    def test_human_format_shows_statistics(self, trace_dir: Path):
        """Test that human format shows aggregate statistics."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [make_duration_event(0, 100.0)]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=5)
        
        output = report.format_human()
        
        assert "Total cycles" in output
        assert "Average duration" in output
        assert "Max duration" in output


class TestHotspotEntry:
    """Tests for HotspotEntry dataclass."""
    
    def test_entry_to_dict(self):
        """Test HotspotEntry.to_dict()."""
        entry = HotspotEntry(
            cycle=42,
            duration_ms=123.456,
            error_count=2,
            notes="⚠ long + 2 errs",
        )
        
        data = entry.to_dict()
        
        assert data["cycle"] == 42
        assert data["duration_ms"] == 123.456
        assert data["error_count"] == 2
        assert data["notes"] == "⚠ long + 2 errs"


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_trace(self, trace_dir: Path):
        """Test hotspots with empty trace."""
        trace_path = trace_dir / "trace.jsonl"
        write_trace(trace_path, [])
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert len(report.entries) == 0
        assert report.total_cycles == 0
        assert report.avg_duration_ms == 0.0
    
    def test_no_duration_events(self, trace_dir: Path):
        """Test hotspots when there are no duration events."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            {
                "ts": 1699123456.789,
                "event_type": "SessionStartEvent",
                "schema_version": schema.TRACE_SCHEMA_VERSION,
                "payload": {"run_id": "test"},
            },
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=10)
        
        assert len(report.entries) == 0
        assert report.total_cycles == 0
    
    def test_top_n_zero(self, trace_dir: Path):
        """Test hotspots with top_n=0."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [make_duration_event(0, 100.0)]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        report = inspector.hotspots(top_n=0)
        
        assert len(report.entries) == 0
        assert report.total_cycles == 1  # Still counts total


class TestDeterminism:
    """Tests verifying deterministic behavior."""
    
    def test_consistent_ordering(self, trace_dir: Path):
        """Test that ordering is consistent across multiple runs."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(5, 100.0),
            make_duration_event(2, 100.0),
            make_duration_event(8, 100.0),
            make_duration_event(1, 100.0),
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        
        # Run multiple times
        results = []
        for _ in range(5):
            report = inspector.hotspots(top_n=10)
            results.append([e.cycle for e in report.entries])
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_consistent_statistics(self, trace_dir: Path):
        """Test that statistics are consistent across multiple runs."""
        trace_path = trace_dir / "trace.jsonl"
        
        events = [
            make_duration_event(i, 50.0 + i * 10)
            for i in range(5)
        ]
        write_trace(trace_path, events)
        
        inspector = TraceLogInspector(trace_path)
        
        # Run multiple times
        results = []
        for _ in range(5):
            report = inspector.hotspots(top_n=10)
            results.append({
                "avg": report.avg_duration_ms,
                "max": report.max_duration_ms,
                "total": report.total_cycles,
            })
        
        # All results should be identical
        assert all(r == results[0] for r in results)

