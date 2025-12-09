# PHASE II — NOT RUN IN PHASE I
"""
Tests for U2 Trace Log Inspector

STATUS: PHASE II — NOT RUN IN PHASE I

Tests:
    - Inspector summary generation
    - Event histogram accuracy
    - Schema validation
    - Event filtering
    - Zero-semantics (read-only) invariant
"""

import json
import tempfile
from pathlib import Path

import pytest

from experiments.u2.inspector import (
    TraceLogInspector,
    TraceLogSummary,
    EventHistogram,
    ValidationError,
    LastMileReport,
    iter_events,
    parse_cycle_range,
)
from experiments.u2.logging import U2TraceLogger, CORE_EVENTS, ALL_EVENT_TYPES
from experiments.u2 import schema


@pytest.fixture
def sample_trace_log(tmp_path: Path) -> Path:
    """Create a sample trace log for testing."""
    log_path = tmp_path / "sample_trace.jsonl"
    
    with U2TraceLogger(log_path) as logger:
        # Session start
        logger.log_session_start(schema.SessionStartEvent(
            run_id="test_run_001",
            slice_name="test_slice",
            mode="baseline",
            schema_version=schema.TRACE_SCHEMA_VERSION,
            config_hash="abc123",
            total_cycles=10,
            initial_seed=42,
        ))
        
        # Several cycle events
        for i in range(5):
            logger.log_cycle_duration(schema.CycleDurationEvent(
                cycle=i,
                slice_name="test_slice",
                mode="baseline",
                duration_ms=float(i * 10 + 50),
                substrate_duration_ms=float(i * 5 + 20),
            ))
            logger.log_cycle_telemetry(schema.CycleTelemetryEvent(
                cycle=i,
                slice_name="test_slice",
                mode="baseline",
                raw_record={"cycle": i, "success": i % 2 == 0},
            ))
        
        # Session end
        logger.log_session_end(schema.SessionEndEvent(
            run_id="test_run_001",
            slice_name="test_slice",
            mode="baseline",
            schema_version=schema.TRACE_SCHEMA_VERSION,
            manifest_hash="manifest_xyz",
            ht_series_hash="ht_hash_123",
            total_cycles=10,
            completed_cycles=5,
        ))
    
    return log_path


class TestTraceLogSummary:
    """Test summary generation."""
    
    def test_summarize_basic(self, sample_trace_log: Path):
        """Test basic summary generation."""
        inspector = TraceLogInspector(sample_trace_log)
        summary = inspector.summarize()
        
        assert summary.total_records == 12  # 1 start + 5 duration + 5 telemetry + 1 end
        assert summary.run_id == "test_run_001"
        assert summary.slice_name == "test_slice"
        assert summary.mode == "baseline"
        assert summary.total_cycles == 10
        assert summary.completed_cycles == 5
        assert summary.parse_errors == 0
    
    def test_summarize_event_counts(self, sample_trace_log: Path):
        """Test event count breakdown."""
        inspector = TraceLogInspector(sample_trace_log)
        summary = inspector.summarize()
        
        assert summary.event_counts["SessionStartEvent"] == 1
        assert summary.event_counts["SessionEndEvent"] == 1
        assert summary.event_counts["CycleDurationEvent"] == 5
        assert summary.event_counts["CycleTelemetryEvent"] == 5
    
    def test_summarize_timestamps(self, sample_trace_log: Path):
        """Test timestamp tracking."""
        inspector = TraceLogInspector(sample_trace_log)
        summary = inspector.summarize()
        
        assert summary.first_timestamp is not None
        assert summary.last_timestamp is not None
        assert summary.last_timestamp >= summary.first_timestamp
        assert summary.duration_seconds is not None
        assert summary.duration_seconds >= 0
    
    def test_summary_format_human(self, sample_trace_log: Path):
        """Test human-readable formatting."""
        inspector = TraceLogInspector(sample_trace_log)
        summary = inspector.summarize()
        
        output = summary.format_human()
        assert "TRACE LOG SUMMARY" in output
        assert "test_run_001" in output
        assert "test_slice" in output


class TestEventHistogram:
    """Test histogram generation."""
    
    def test_histogram_counts(self, sample_trace_log: Path):
        """Test histogram count accuracy."""
        inspector = TraceLogInspector(sample_trace_log)
        histogram = inspector.event_histogram()
        
        assert histogram.total == 12
        assert histogram.counts["SessionStartEvent"] == 1
        assert histogram.counts["CycleDurationEvent"] == 5
    
    def test_histogram_percentages(self, sample_trace_log: Path):
        """Test percentage calculation."""
        inspector = TraceLogInspector(sample_trace_log)
        histogram = inspector.event_histogram()
        
        pcts = histogram.percentages()
        assert abs(pcts["SessionStartEvent"] - (1/12 * 100)) < 0.1
        assert abs(pcts["CycleDurationEvent"] - (5/12 * 100)) < 0.1
    
    def test_histogram_format_human(self, sample_trace_log: Path):
        """Test human-readable histogram formatting."""
        inspector = TraceLogInspector(sample_trace_log)
        histogram = inspector.event_histogram()
        
        output = histogram.format_human()
        assert "EVENT HISTOGRAM" in output
        assert "TOTAL" in output


class TestSchemaValidation:
    """Test schema validation."""
    
    def test_validate_valid_log(self, sample_trace_log: Path):
        """Valid log should have no errors."""
        inspector = TraceLogInspector(sample_trace_log)
        errors = inspector.validate_schema()
        
        assert len(errors) == 0
    
    def test_validate_invalid_json(self, tmp_path: Path):
        """Detect invalid JSON lines."""
        log_path = tmp_path / "invalid.jsonl"
        log_path.write_text("not valid json\n{}\n")
        
        inspector = TraceLogInspector(log_path)
        errors = inspector.validate_schema()
        
        assert len(errors) >= 1
        assert any("Invalid JSON" in e.error_message for e in errors)
    
    def test_validate_missing_fields(self, tmp_path: Path):
        """Detect missing required fields."""
        log_path = tmp_path / "missing_fields.jsonl"
        log_path.write_text('{"event_type": "Test"}\n')
        
        inspector = TraceLogInspector(log_path)
        errors = inspector.validate_schema()
        
        assert len(errors) >= 1
        assert any("Missing required fields" in e.error_message for e in errors)
    
    def test_validate_unknown_event_type(self, tmp_path: Path):
        """Detect unknown event types."""
        log_path = tmp_path / "unknown_event.jsonl"
        record = {
            "ts": 12345.0,
            "event_type": "UnknownEventType",
            "schema_version": "u2-trace-1.0.0",
            "payload": {},
        }
        log_path.write_text(json.dumps(record) + "\n")
        
        inspector = TraceLogInspector(log_path)
        errors = inspector.validate_schema()
        
        assert len(errors) >= 1
        assert any("Unknown event type" in e.error_message for e in errors)


class TestEventFiltering:
    """Test event filtering."""
    
    def test_filter_by_event_type(self, sample_trace_log: Path):
        """Filter events by type."""
        inspector = TraceLogInspector(sample_trace_log)
        
        duration_events = list(inspector.filter_events(
            event_types={"cycle_duration"}
        ))
        
        assert len(duration_events) == 5
        assert all(e["event_type"] == "CycleDurationEvent" for e in duration_events)
    
    def test_filter_by_cycle_range(self, sample_trace_log: Path):
        """Filter events by cycle range."""
        inspector = TraceLogInspector(sample_trace_log)
        
        filtered = list(inspector.filter_events(
            min_cycle=2,
            max_cycle=3,
        ))
        
        # Should get events for cycles 2 and 3 (both duration and telemetry)
        cycles_found = set()
        for event in filtered:
            cycle = event.get("payload", {}).get("cycle")
            if cycle is not None:
                cycles_found.add(cycle)
        
        assert 2 in cycles_found
        assert 3 in cycles_found
        assert 0 not in cycles_found
        assert 4 not in cycles_found
    
    def test_filter_combined(self, sample_trace_log: Path):
        """Filter with multiple criteria."""
        inspector = TraceLogInspector(sample_trace_log)
        
        filtered = list(inspector.filter_events(
            event_types={"cycle_telemetry"},
            min_cycle=1,
            max_cycle=2,
        ))
        
        assert len(filtered) == 2
        assert all(e["event_type"] == "CycleTelemetryEvent" for e in filtered)


class TestEventFilteringInLogger:
    """Test event filtering in U2TraceLogger."""
    
    def test_filter_limits_events(self, tmp_path: Path):
        """Only enabled events should be logged."""
        log_path = tmp_path / "filtered.jsonl"
        
        with U2TraceLogger(
            log_path,
            enabled_events={"cycle_duration"}  # Only duration events
        ) as logger:
            # Try to log multiple event types
            logger.log_session_start(schema.SessionStartEvent(
                run_id="test",
                slice_name="test",
                mode="baseline",
                schema_version=schema.TRACE_SCHEMA_VERSION,
                config_hash="test",
                total_cycles=1,
                initial_seed=1,
            ))
            logger.log_cycle_duration(schema.CycleDurationEvent(
                cycle=0,
                slice_name="test",
                mode="baseline",
                duration_ms=100.0,
                substrate_duration_ms=50.0,
            ))
            logger.log_cycle_telemetry(schema.CycleTelemetryEvent(
                cycle=0,
                slice_name="test",
                mode="baseline",
                raw_record={},
            ))
        
        # Only duration event should be logged
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        
        record = json.loads(lines[0])
        assert record["event_type"] == "CycleDurationEvent"
    
    def test_core_events_preset(self, tmp_path: Path):
        """Test CORE_EVENTS preset."""
        log_path = tmp_path / "core_events.jsonl"
        
        with U2TraceLogger(
            log_path,
            enabled_events=CORE_EVENTS
        ) as logger:
            # Log various events
            logger.log_session_start(schema.SessionStartEvent(
                run_id="test", slice_name="test", mode="baseline",
                schema_version=schema.TRACE_SCHEMA_VERSION,
                config_hash="test", total_cycles=1, initial_seed=1,
            ))
            logger.log_cycle_duration(schema.CycleDurationEvent(
                cycle=0, slice_name="test", mode="baseline",
                duration_ms=100.0, substrate_duration_ms=None,
            ))
            logger.log_cycle_telemetry(schema.CycleTelemetryEvent(
                cycle=0, slice_name="test", mode="baseline", raw_record={},
            ))
            logger.log_candidate_ordering(schema.CandidateOrderingEvent(
                cycle=0, slice_name="test", mode="baseline", ordering=(),
            ))  # This should NOT be logged (not in CORE_EVENTS)
            logger.log_session_end(schema.SessionEndEvent(
                run_id="test", slice_name="test", mode="baseline",
                schema_version=schema.TRACE_SCHEMA_VERSION,
                manifest_hash=None, ht_series_hash=None,
                total_cycles=1, completed_cycles=1,
            ))
        
        lines = log_path.read_text().strip().split("\n")
        event_types = [json.loads(line)["event_type"] for line in lines]
        
        # Should have start, duration, telemetry, end (all core events)
        # Should NOT have CandidateOrderingEvent
        assert "SessionStartEvent" in event_types
        assert "CycleDurationEvent" in event_types
        assert "CycleTelemetryEvent" in event_types
        assert "SessionEndEvent" in event_types
        assert "CandidateOrderingEvent" not in event_types
    
    def test_all_events_available(self):
        """Verify ALL_EVENT_TYPES contains expected events."""
        expected = {
            "session_start",
            "session_end",
            "cycle_telemetry",
            "cycle_duration",
            "candidate_ordering",
            "scoring_features",
            "policy_weight_update",
            "budget_consumption",
            "substrate_result",
            "hash_chain_entry",
        }
        assert ALL_EVENT_TYPES == expected


class TestZeroSemanticsInvariant:
    """Test that inspector is read-only and deterministic."""
    
    def test_inspector_does_not_modify_file(self, sample_trace_log: Path):
        """Inspector must not modify the trace log."""
        # Get file state before
        content_before = sample_trace_log.read_bytes()
        mtime_before = sample_trace_log.stat().st_mtime
        
        # Run inspector operations
        inspector = TraceLogInspector(sample_trace_log)
        _ = inspector.summarize()
        _ = inspector.event_histogram()
        _ = inspector.validate_schema()
        _ = list(inspector.filter_events())
        
        # Verify file unchanged
        content_after = sample_trace_log.read_bytes()
        mtime_after = sample_trace_log.stat().st_mtime
        
        assert content_before == content_after
        assert mtime_before == mtime_after
    
    def test_inspector_deterministic_output(self, sample_trace_log: Path):
        """Same input should produce same output."""
        inspector1 = TraceLogInspector(sample_trace_log)
        inspector2 = TraceLogInspector(sample_trace_log)
        
        summary1 = inspector1.summarize()
        summary2 = inspector2.summarize()
        
        # Summaries should be identical (except timestamps which we don't control)
        assert summary1.total_records == summary2.total_records
        assert summary1.event_counts == summary2.event_counts
        assert summary1.run_id == summary2.run_id


class TestIterEvents:
    """Test the iter_events convenience function."""
    
    def test_iter_events_basic(self, sample_trace_log: Path):
        """Basic iteration over all events."""
        events = list(iter_events(sample_trace_log))
        assert len(events) == 12  # All events in sample
    
    def test_iter_events_with_type_filter(self, sample_trace_log: Path):
        """Filter by event type."""
        events = list(iter_events(
            sample_trace_log,
            event_type="cycle_duration",
        ))
        assert len(events) == 5
        assert all(e["event_type"] == "CycleDurationEvent" for e in events)
    
    def test_iter_events_with_cycle_range(self, sample_trace_log: Path):
        """Filter by cycle range."""
        events = list(iter_events(
            sample_trace_log,
            cycle_range=(1, 3),
        ))
        
        # Should get cycles 1, 2, 3
        cycles = {e["payload"]["cycle"] for e in events if "cycle" in e.get("payload", {})}
        assert cycles == {1, 2, 3}
    
    def test_iter_events_combined(self, sample_trace_log: Path):
        """Combined event type and cycle range filters."""
        events = list(iter_events(
            sample_trace_log,
            event_type="cycle_telemetry",
            cycle_range=(0, 2),
        ))
        
        assert len(events) == 3
        assert all(e["event_type"] == "CycleTelemetryEvent" for e in events)


class TestParseCycleRange:
    """Test the parse_cycle_range function."""
    
    def test_parse_single_cycle(self):
        """Single number becomes range of one."""
        min_c, max_c = parse_cycle_range("42")
        assert min_c == 42
        assert max_c == 42
    
    def test_parse_full_range(self):
        """Full range with both bounds."""
        min_c, max_c = parse_cycle_range("10:20")
        assert min_c == 10
        assert max_c == 20
    
    def test_parse_open_start(self):
        """Range with open start (:20)."""
        min_c, max_c = parse_cycle_range(":20")
        assert min_c is None
        assert max_c == 20
    
    def test_parse_open_end(self):
        """Range with open end (10:)."""
        min_c, max_c = parse_cycle_range("10:")
        assert min_c == 10
        assert max_c is None
    
    def test_parse_invalid_raises(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError):
            parse_cycle_range("not_a_number")


class TestLastMileCheck:
    """Test last-mile readiness checking."""
    
    def test_last_mile_ok(self, sample_trace_log: Path):
        """Complete trace should pass last-mile check."""
        inspector = TraceLogInspector(sample_trace_log)
        report = inspector.last_mile_check()
        
        assert report.status == "OK"
        assert report.schema_version_ok
        assert len(report.violations) == 0
    
    def test_last_mile_incomplete_missing_telemetry(self, tmp_path: Path):
        """Incomplete trace should be flagged."""
        log_path = tmp_path / "incomplete.jsonl"
        
        with U2TraceLogger(log_path) as logger:
            # Start session with 5 cycles expected
            logger.log_session_start(schema.SessionStartEvent(
                run_id="test",
                slice_name="test",
                mode="baseline",
                schema_version=schema.TRACE_SCHEMA_VERSION,
                config_hash="test",
                total_cycles=5,
                initial_seed=1,
            ))
            
            # Only log duration for cycles 0-2 (missing telemetry)
            for i in range(3):
                logger.log_cycle_duration(schema.CycleDurationEvent(
                    cycle=i,
                    slice_name="test",
                    mode="baseline",
                    duration_ms=100.0,
                    substrate_duration_ms=None,
                ))
            
            # End session with 3 completed
            logger.log_session_end(schema.SessionEndEvent(
                run_id="test",
                slice_name="test",
                mode="baseline",
                schema_version=schema.TRACE_SCHEMA_VERSION,
                manifest_hash=None,
                ht_series_hash=None,
                total_cycles=5,
                completed_cycles=3,
            ))
        
        inspector = TraceLogInspector(log_path)
        report = inspector.last_mile_check()
        
        assert report.status == "INCOMPLETE"
        assert len(report.missing_telemetry_cycles) == 3  # All cycles missing telemetry
        assert len(report.violations) > 0
    
    def test_last_mile_report_json_serialization(self, sample_trace_log: Path):
        """Report should be JSON serializable."""
        inspector = TraceLogInspector(sample_trace_log)
        report = inspector.last_mile_check()
        
        # Should not raise
        json_str = json.dumps(report.to_dict())
        parsed = json.loads(json_str)
        
        assert parsed["status"] == report.status
        assert parsed["schema_version_ok"] == report.schema_version_ok
    
    def test_last_mile_report_human_format(self, sample_trace_log: Path):
        """Report should have human-readable format."""
        inspector = TraceLogInspector(sample_trace_log)
        report = inspector.last_mile_check()
        
        output = report.format_human()
        assert "LAST-MILE READINESS CHECK" in output
        assert "Status:" in output

