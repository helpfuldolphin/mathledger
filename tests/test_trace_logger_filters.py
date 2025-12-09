"""
Tests for U2TraceLogger event filtering.

PHASE II — NOT USED IN PHASE I
"""

import json
import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.u2.logging import (
    U2TraceLogger,
    CORE_EVENTS,
    ALL_EVENT_TYPES,
)
from experiments.u2 import schema as trace_schema


class TestTraceLoggerFiltering:
    """Tests for U2TraceLogger event filtering."""

    def test_all_event_types_defined(self):
        """Verify ALL_EVENT_TYPES matches available log_* methods."""
        # All core events should be in ALL_EVENT_TYPES
        assert CORE_EVENTS.issubset(ALL_EVENT_TYPES)
        
        # Expected event types
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

    def test_core_events_default(self):
        """Verify CORE_EVENTS contains minimal set."""
        expected_core = {
            "session_start",
            "session_end",
            "cycle_telemetry",
            "cycle_duration",
        }
        assert CORE_EVENTS == expected_core

    def test_no_filter_logs_all_events(self):
        """When enabled_events=None, all events should be logged."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            
            with U2TraceLogger(path, enabled_events=None) as logger:
                # Log various events
                logger.log_session_start(trace_schema.SessionStartEvent(
                    run_id="test",
                    slice_name="test_slice",
                    mode="baseline",
                    schema_version="1.0.0",
                    config_hash="abc",
                    total_cycles=10,
                    initial_seed=42,
                ))
                logger.log_cycle_duration(trace_schema.CycleDurationEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="baseline",
                    duration_ms=100.0,
                    substrate_duration_ms=50.0,
                ))
                logger.log_budget_consumption(trace_schema.BudgetConsumptionEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="baseline",
                    candidates_considered=10,
                    candidates_limit=40,
                    budget_exhausted=False,
                ))
            
            # Read back and verify
            events = []
            with open(path, "r") as f:
                for line in f:
                    events.append(json.loads(line))
            
            event_types = [e["event_type"] for e in events]
            assert "SessionStartEvent" in event_types
            assert "CycleDurationEvent" in event_types
            assert "BudgetConsumptionEvent" in event_types

    def test_filter_only_core_events(self):
        """When enabled_events=CORE_EVENTS, only core events should be logged."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            
            with U2TraceLogger(path, enabled_events=set(CORE_EVENTS)) as logger:
                # Log various events - some should be filtered
                logger.log_session_start(trace_schema.SessionStartEvent(
                    run_id="test",
                    slice_name="test_slice",
                    mode="baseline",
                    schema_version="1.0.0",
                    config_hash="abc",
                    total_cycles=10,
                    initial_seed=42,
                ))
                logger.log_cycle_duration(trace_schema.CycleDurationEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="baseline",
                    duration_ms=100.0,
                    substrate_duration_ms=50.0,
                ))
                # This should be filtered out (not in CORE_EVENTS)
                logger.log_budget_consumption(trace_schema.BudgetConsumptionEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="baseline",
                    candidates_considered=10,
                    candidates_limit=40,
                    budget_exhausted=False,
                ))
                # This should be filtered out (not in CORE_EVENTS)
                logger.log_policy_weight_update(trace_schema.PolicyWeightUpdateEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="rfl",
                    weights_before={"len": 0.0},
                    weights_after={"len": 0.1},
                    reward=1.0,
                    verified_count=1,
                    target=5,
                ))
            
            # Read back and verify
            events = []
            with open(path, "r") as f:
                for line in f:
                    events.append(json.loads(line))
            
            event_types = [e["event_type"] for e in events]
            
            # Core events should be present
            assert "SessionStartEvent" in event_types
            assert "CycleDurationEvent" in event_types
            
            # Non-core events should be filtered
            assert "BudgetConsumptionEvent" not in event_types
            assert "PolicyWeightUpdateEvent" not in event_types

    def test_filter_specific_events(self):
        """Test filtering with a specific set of enabled events."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            
            # Only enable policy_weight_update and budget_consumption
            with U2TraceLogger(
                path, 
                enabled_events={"policy_weight_update", "budget_consumption"}
            ) as logger:
                # These should be filtered
                logger.log_session_start(trace_schema.SessionStartEvent(
                    run_id="test",
                    slice_name="test_slice",
                    mode="baseline",
                    schema_version="1.0.0",
                    config_hash="abc",
                    total_cycles=10,
                    initial_seed=42,
                ))
                logger.log_cycle_duration(trace_schema.CycleDurationEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="baseline",
                    duration_ms=100.0,
                    substrate_duration_ms=50.0,
                ))
                
                # These should be logged
                logger.log_budget_consumption(trace_schema.BudgetConsumptionEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="baseline",
                    candidates_considered=10,
                    candidates_limit=40,
                    budget_exhausted=False,
                ))
                logger.log_policy_weight_update(trace_schema.PolicyWeightUpdateEvent(
                    cycle=0,
                    slice_name="test_slice",
                    mode="rfl",
                    weights_before={"len": 0.0},
                    weights_after={"len": 0.1},
                    reward=1.0,
                    verified_count=1,
                    target=5,
                ))
            
            # Read back and verify
            events = []
            with open(path, "r") as f:
                for line in f:
                    events.append(json.loads(line))
            
            event_types = [e["event_type"] for e in events]
            
            # Only specified events should be present
            assert "SessionStartEvent" not in event_types
            assert "CycleDurationEvent" not in event_types
            assert "BudgetConsumptionEvent" in event_types
            assert "PolicyWeightUpdateEvent" in event_types

    def test_invalid_event_type_warning(self):
        """Test that invalid event types are ignored with warning in fail_soft mode."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            
            # Should not raise, just warn
            logger = U2TraceLogger(
                path, 
                fail_soft=True,
                enabled_events={"cycle_duration", "invalid_event_type"}
            )
            
            # Should have filtered out invalid event type
            assert "invalid_event_type" not in logger.enabled_events
            assert "cycle_duration" in logger.enabled_events

    def test_invalid_event_type_raises_in_strict_mode(self):
        """Test that invalid event types raise ValueError in strict mode."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            
            with pytest.raises(ValueError, match="Invalid event types"):
                U2TraceLogger(
                    path, 
                    fail_soft=False,
                    enabled_events={"cycle_duration", "invalid_event_type"}
                )

    def test_enabled_events_property(self):
        """Test that enabled_events property returns correct value."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            
            # No filter
            logger1 = U2TraceLogger(path, enabled_events=None)
            assert logger1.enabled_events is None
            
            # With filter
            enabled = {"cycle_duration", "session_start"}
            logger2 = U2TraceLogger(path, enabled_events=enabled)
            assert logger2.enabled_events == frozenset(enabled)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

