# PHASE II — NOT RUN IN PHASE I
"""
Tests for trace_to_csv.py

STATUS: PHASE II — NOT RUN IN PHASE I

Tests CSV export functionality for trace logs.
"""

import csv
import json
import pytest
from pathlib import Path
from typing import Any, Dict

# Import directly from the script (add project root to path)
import sys
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.trace_to_csv import (
    export_to_csv,
    extract_row,
    format_value,
    detect_error_kind,
    STANDARD_HEADERS,
    EXTENDED_HEADERS,
    CSV_SCHEMA_VERSION,
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


class TestCSVHeaders:
    """Tests for CSV header consistency."""
    
    def test_standard_headers_defined(self):
        """Test that standard headers are properly defined."""
        assert "cycle" in STANDARD_HEADERS
        assert "event_type" in STANDARD_HEADERS
        assert "timestamp" in STANDARD_HEADERS
        assert "duration_ms" in STANDARD_HEADERS
        assert "error_kind" in STANDARD_HEADERS
    
    def test_extended_headers_include_standard(self):
        """Test that extended headers include all standard headers."""
        for header in STANDARD_HEADERS:
            assert header in EXTENDED_HEADERS
    
    def test_extended_headers_have_extra(self):
        """Test that extended headers have additional fields."""
        assert "schema_version" in EXTENDED_HEADERS
        assert "run_id" in EXTENDED_HEADERS
        assert len(EXTENDED_HEADERS) > len(STANDARD_HEADERS)
    
    def test_csv_has_correct_headers(self, trace_dir: Path):
        """Test that exported CSV has correct headers."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [
            make_event("CycleDurationEvent", cycle=0, duration_ms=50.0),
        ]
        write_trace(trace_path, events)
        
        export_to_csv(trace_path, csv_path)
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        assert headers == STANDARD_HEADERS
    
    def test_verbose_csv_has_extended_headers(self, trace_dir: Path):
        """Test that verbose CSV has extended headers."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [
            make_event("CycleDurationEvent", cycle=0, duration_ms=50.0),
        ]
        write_trace(trace_path, events)
        
        export_to_csv(trace_path, csv_path, verbose=True)
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        assert headers == EXTENDED_HEADERS


class TestNumericSerialization:
    """Tests for numeric value serialization."""
    
    def test_format_integer_float(self):
        """Test that integer floats are formatted without decimal."""
        assert format_value(50.0) == "50"
        assert format_value(100.0) == "100"
    
    def test_format_decimal_float(self):
        """Test that decimal floats have limited precision."""
        result = format_value(50.123456789)
        assert result == "50.123"
    
    def test_format_none(self):
        """Test that None formats as empty string."""
        assert format_value(None) == ""
    
    def test_format_boolean(self):
        """Test boolean formatting."""
        assert format_value(True) == "true"
        assert format_value(False) == "false"
    
    def test_duration_ms_serialization(self, trace_dir: Path):
        """Test that duration_ms is correctly serialized."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [
            make_event("CycleDurationEvent", cycle=0, duration_ms=123.456),
            make_event("CycleDurationEvent", cycle=1, duration_ms=200.0),
        ]
        write_trace(trace_path, events)
        
        export_to_csv(trace_path, csv_path)
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert rows[0]["duration_ms"] == "123.456"
        assert rows[1]["duration_ms"] == "200"
    
    def test_timestamp_serialization(self, trace_dir: Path):
        """Test that timestamps are correctly serialized."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [
            {
                "ts": 1699123456.789,
                "event_type": "CycleDurationEvent",
                "schema_version": schema.TRACE_SCHEMA_VERSION,
                "payload": {"cycle": 0, "duration_ms": 50.0},
            },
        ]
        write_trace(trace_path, events)
        
        export_to_csv(trace_path, csv_path)
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert rows[0]["timestamp"] == "1699123456.789"


class TestErrorKindDetection:
    """Tests for error kind detection."""
    
    def test_detect_timeout(self):
        """Test timeout error detection."""
        record = {"payload": {"error": "timeout occurred"}}
        assert detect_error_kind(record) == "timeout"
    
    def test_detect_budget_exhausted(self):
        """Test budget exhausted detection."""
        record = {"payload": {"message": "budget exhausted"}}
        assert detect_error_kind(record) == "budget_exhausted"
    
    def test_detect_parse_error(self):
        """Test parse error detection."""
        record = {"payload": {"error": "JSON decode error"}}
        assert detect_error_kind(record) == "parse_error"
    
    def test_detect_success_false(self):
        """Test error detection from success=False."""
        record = {"payload": {"raw_record": {"success": False}}}
        assert detect_error_kind(record) is not None
    
    def test_no_error(self):
        """Test no error when record is clean."""
        record = {"payload": {"raw_record": {"success": True, "result": "OK"}}}
        assert detect_error_kind(record) is None


class TestRowExtraction:
    """Tests for row extraction from records."""
    
    def test_extract_basic_fields(self):
        """Test extraction of basic fields."""
        record = make_event("CycleDurationEvent", cycle=5, duration_ms=42.5, mode="baseline")
        row = extract_row(record)
        
        assert row["cycle"] == 5
        assert row["event_type"] == "CycleDurationEvent"
        assert row["duration_ms"] == 42.5
        assert row["mode"] == "baseline"
    
    def test_extract_with_verbose(self):
        """Test extraction with verbose mode."""
        record = make_event(
            "CycleTelemetryEvent",
            cycle=3,
            run_id="test-run-123",
            raw_record={"item": "test_item", "result": "VERIFIED"},
        )
        row = extract_row(record, verbose=True)
        
        assert row["run_id"] == "test-run-123"
        assert row["item"] == "test_item"
        assert row["result"] == "VERIFIED"
    
    def test_missing_fields_are_none(self):
        """Test that missing fields are None."""
        record = make_event("SessionStartEvent")
        row = extract_row(record)
        
        assert row["cycle"] is None
        assert row["duration_ms"] is None
        assert row["substrate_duration_ms"] is None


class TestCSVExport:
    """Tests for full CSV export."""
    
    def test_export_creates_file(self, trace_dir: Path):
        """Test that export creates output file."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [make_event("CycleDurationEvent", cycle=0, duration_ms=50.0)]
        write_trace(trace_path, events)
        
        export_to_csv(trace_path, csv_path)
        
        assert csv_path.exists()
    
    def test_export_returns_row_count(self, trace_dir: Path):
        """Test that export returns correct row count."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [
            make_event("CycleDurationEvent", cycle=0, duration_ms=50.0),
            make_event("CycleDurationEvent", cycle=1, duration_ms=55.0),
            make_event("CycleDurationEvent", cycle=2, duration_ms=60.0),
        ]
        write_trace(trace_path, events)
        
        count = export_to_csv(trace_path, csv_path)
        
        assert count == 3
    
    def test_export_with_event_filter(self, trace_dir: Path):
        """Test export with event type filter."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [
            make_event("SessionStartEvent", run_id="test"),
            make_event("CycleDurationEvent", cycle=0, duration_ms=50.0),
            make_event("CycleTelemetryEvent", cycle=0),
            make_event("CycleDurationEvent", cycle=1, duration_ms=55.0),
            make_event("SessionEndEvent", run_id="test"),
        ]
        write_trace(trace_path, events)
        
        count = export_to_csv(trace_path, csv_path, event_types={"cycle_duration"})
        
        assert count == 2
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert all(r["event_type"] == "CycleDurationEvent" for r in rows)
    
    def test_export_with_cycle_range(self, trace_dir: Path):
        """Test export with cycle range filter."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        events = [
            make_event("CycleDurationEvent", cycle=0, duration_ms=50.0),
            make_event("CycleDurationEvent", cycle=1, duration_ms=55.0),
            make_event("CycleDurationEvent", cycle=2, duration_ms=60.0),
            make_event("CycleDurationEvent", cycle=3, duration_ms=65.0),
            make_event("CycleDurationEvent", cycle=4, duration_ms=70.0),
        ]
        write_trace(trace_path, events)
        
        count = export_to_csv(trace_path, csv_path, cycle_range=(1, 3))
        
        assert count == 3
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        cycles = [int(r["cycle"]) for r in rows]
        assert cycles == [1, 2, 3]
    
    def test_creates_parent_directories(self, trace_dir: Path):
        """Test that export creates parent directories."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "nested" / "dir" / "output.csv"
        
        events = [make_event("CycleDurationEvent", cycle=0, duration_ms=50.0)]
        write_trace(trace_path, events)
        
        export_to_csv(trace_path, csv_path)
        
        assert csv_path.exists()


class TestSchemaVersion:
    """Tests for schema version stability."""
    
    def test_schema_version_defined(self):
        """Test that schema version is defined."""
        assert CSV_SCHEMA_VERSION is not None
        assert isinstance(CSV_SCHEMA_VERSION, str)
    
    def test_schema_version_format(self):
        """Test schema version format."""
        parts = CSV_SCHEMA_VERSION.split(".")
        assert len(parts) >= 1
        # Should be numeric or semver-like
        assert all(p.isdigit() for p in parts)


class TestDocstringExample:
    """Tests verifying docstring example output format."""
    
    def test_example_csv_format(self, trace_dir: Path):
        """Test that output matches docstring example format."""
        trace_path = trace_dir / "trace.jsonl"
        csv_path = trace_dir / "output.csv"
        
        # Match the docstring example data
        events = [
            {
                "ts": 1699123456.789,
                "event_type": "CycleDurationEvent",
                "schema_version": schema.TRACE_SCHEMA_VERSION,
                "payload": {
                    "cycle": 0,
                    "duration_ms": 45.2,
                    "substrate_duration_ms": 12.3,
                    "mode": "baseline",
                    "slice_name": "test_slice",
                },
            },
        ]
        write_trace(trace_path, events)
        
        export_to_csv(trace_path, csv_path)
        
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            content = f.read()
        
        # Check header line
        assert content.startswith("cycle,event_type,timestamp,")
        
        # Check data format
        lines = content.strip().split("\n")
        assert len(lines) == 2  # Header + 1 data row
        
        # Parse and verify
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        
        assert row["cycle"] == "0"
        assert row["event_type"] == "CycleDurationEvent"
        # Float formatting uses 3 decimal places
        assert float(row["duration_ms"]) == pytest.approx(45.2)
        assert float(row["substrate_duration_ms"]) == pytest.approx(12.3)
        assert row["mode"] == "baseline"
        assert row["slice_name"] == "test_slice"

