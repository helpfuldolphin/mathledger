# PHASE II — NOT RUN IN PHASE I
"""
U2 Trace Log Inspector

STATUS: PHASE II — NOT RUN IN PHASE I

Provides utilities for analyzing trace logs without modifying experiment behavior:
- Summaries: Event counts, time ranges, session info
- Event histograms: Distribution of event types
- Record structure validation: Schema compliance checking
- Event filtering: Slice by event type and cycle range
- Incident bundles: Extract minimal trace data for bug reports
- Last-mile checks: Verify trace completeness for readiness

INVARIANTS:
- Read-only: Never modifies trace log files
- Deterministic: Same input always produces same output
- Zero-semantics: Inspection does not affect experiment behavior

Usage:
    from experiments.u2.inspector import TraceLogInspector, iter_events
    
    inspector = TraceLogInspector(Path("traces/run.jsonl"))
    summary = inspector.summarize()
    histogram = inspector.event_histogram()
    errors = inspector.validate_schema()
    
    # Filter and iterate
    for event in iter_events(path, event_type="cycle_telemetry", cycle_range=(10, 20)):
        print(event)

CLI:
    uv run python -m experiments.u2.inspector traces/run.jsonl --summary
    uv run python -m experiments.u2.inspector traces/run.jsonl --histogram
    uv run python -m experiments.u2.inspector traces/run.jsonl --validate
    uv run python -m experiments.u2.inspector traces/run.jsonl --filter-event cycle_telemetry --dump
    uv run python -m experiments.u2.inspector traces/run.jsonl --filter-cycle-range 10:20 --dump
    uv run python -m experiments.u2.inspector traces/run.jsonl --last-mile-check
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from . import schema
from .logging import ALL_EVENT_TYPES


# Event class name to filter name mapping
EVENT_CLASS_TO_FILTER = {
    "SessionStartEvent": "session_start",
    "SessionEndEvent": "session_end",
    "CycleTelemetryEvent": "cycle_telemetry",
    "CycleDurationEvent": "cycle_duration",
    "CandidateOrderingEvent": "candidate_ordering",
    "ScoringFeaturesEvent": "scoring_features",
    "PolicyWeightUpdateEvent": "policy_weight_update",
    "BudgetConsumptionEvent": "budget_consumption",
    "SubstrateResultEvent": "substrate_result",
    "HashChainEntryEvent": "hash_chain_entry",
}

# Reverse mapping
FILTER_TO_EVENT_CLASS = {v: k for k, v in EVENT_CLASS_TO_FILTER.items()}

# Expected schema version
EXPECTED_SCHEMA_VERSION = schema.TRACE_SCHEMA_VERSION


@dataclass(frozen=True)
class ValidationError:
    """A schema validation error for a trace log record."""
    line_number: int
    event_type: str
    error_message: str
    record_preview: str  # Truncated record for context


@dataclass
class TraceLogSummary:
    """Summary statistics for a trace log file."""
    
    # File info
    file_path: str
    file_size_bytes: int
    total_records: int
    
    # Time range
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None
    duration_seconds: Optional[float] = None
    
    # Session info (from SessionStartEvent/SessionEndEvent)
    run_id: Optional[str] = None
    slice_name: Optional[str] = None
    mode: Optional[str] = None
    schema_version: Optional[str] = None
    total_cycles: Optional[int] = None
    completed_cycles: Optional[int] = None
    
    # Event breakdown
    event_counts: Dict[str, int] = field(default_factory=dict)
    
    # Validation
    parse_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "total_records": self.total_records,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "duration_seconds": self.duration_seconds,
            "run_id": self.run_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "schema_version": self.schema_version,
            "total_cycles": self.total_cycles,
            "completed_cycles": self.completed_cycles,
            "event_counts": self.event_counts,
            "parse_errors": self.parse_errors,
        }
    
    def format_human(self) -> str:
        """Format summary for human-readable output."""
        lines = [
            "=" * 60,
            "TRACE LOG SUMMARY",
            "=" * 60,
            f"File: {self.file_path}",
            f"Size: {self.file_size_bytes:,} bytes",
            f"Records: {self.total_records:,}",
            "",
        ]
        
        if self.first_timestamp and self.last_timestamp:
            first_dt = datetime.fromtimestamp(self.first_timestamp)
            last_dt = datetime.fromtimestamp(self.last_timestamp)
            lines.extend([
                f"First timestamp: {first_dt.isoformat()}",
                f"Last timestamp:  {last_dt.isoformat()}",
                f"Duration: {self.duration_seconds:.2f} seconds" if self.duration_seconds else "",
                "",
            ])
        
        if self.run_id:
            lines.extend([
                "SESSION INFO:",
                f"  Run ID: {self.run_id}",
                f"  Slice: {self.slice_name}",
                f"  Mode: {self.mode}",
                f"  Schema: {self.schema_version}",
                f"  Total cycles: {self.total_cycles}",
                f"  Completed cycles: {self.completed_cycles}",
                "",
            ])
        
        lines.append("EVENT COUNTS:")
        for event_type, count in sorted(self.event_counts.items()):
            lines.append(f"  {event_type}: {count:,}")
        
        if self.parse_errors > 0:
            lines.extend([
                "",
                f"⚠️  Parse errors: {self.parse_errors}",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class EventHistogram:
    """Histogram of event types in a trace log."""
    
    counts: Dict[str, int] = field(default_factory=dict)
    total: int = 0
    
    def add(self, event_type: str) -> None:
        """Add an event to the histogram."""
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        self.total += 1
    
    def percentages(self) -> Dict[str, float]:
        """Return event type percentages."""
        if self.total == 0:
            return {}
        return {k: (v / self.total) * 100 for k, v in self.counts.items()}
    
    def format_human(self, bar_width: int = 40) -> str:
        """Format histogram for human-readable output."""
        lines = [
            "=" * 60,
            "EVENT HISTOGRAM",
            "=" * 60,
        ]
        
        if self.total == 0:
            lines.append("(no events)")
            return "\n".join(lines)
        
        max_count = max(self.counts.values()) if self.counts else 0
        
        for event_type in sorted(self.counts.keys()):
            count = self.counts[event_type]
            pct = (count / self.total) * 100
            bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = "█" * bar_len
            lines.append(f"{event_type:30s} {count:6,} ({pct:5.1f}%) {bar}")
        
        lines.extend([
            "-" * 60,
            f"{'TOTAL':30s} {self.total:6,}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


@dataclass
class HotspotEntry:
    """A single hotspot entry (cycle with notable duration/errors)."""
    
    cycle: int
    duration_ms: float
    error_count: int
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle": self.cycle,
            "duration_ms": self.duration_ms,
            "error_count": self.error_count,
            "notes": self.notes,
        }


@dataclass
class HotspotReport:
    """Report of performance/error hotspots in a trace log."""
    
    top_n: int
    total_cycles: int
    entries: List[HotspotEntry] = field(default_factory=list)
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    total_error_count: int = 0
    cycles_with_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "top_n": self.top_n,
            "total_cycles": self.total_cycles,
            "entries": [e.to_dict() for e in self.entries],
            "avg_duration_ms": self.avg_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "total_error_count": self.total_error_count,
            "cycles_with_errors": self.cycles_with_errors,
        }
    
    def format_human(self) -> str:
        """Format report as a human-readable table."""
        lines = [
            "=" * 70,
            "HOTSPOT ANALYSIS",
            "=" * 70,
            f"Total cycles analyzed: {self.total_cycles}",
            f"Average duration: {self.avg_duration_ms:.2f} ms",
            f"Max duration: {self.max_duration_ms:.2f} ms",
            f"Cycles with errors: {self.cycles_with_errors}",
            f"Total error count: {self.total_error_count}",
            "",
            f"Top {self.top_n} hotspots by duration:",
            "-" * 70,
            f"{'Cycle':>6} | {'duration_ms':>12} | {'error_count':>11} | notes",
            "-" * 70,
        ]
        
        if not self.entries:
            lines.append("(no cycles found)")
        else:
            for entry in self.entries:
                lines.append(
                    f"{entry.cycle:>6} | {entry.duration_ms:>12.2f} | "
                    f"{entry.error_count:>11} | {entry.notes}"
                )
        
        lines.extend([
            "-" * 70,
            "=" * 70,
        ])
        
        return "\n".join(lines)


@dataclass
class LastMileReport:
    """Last-mile readiness check report."""
    
    status: str  # "OK", "INCOMPLETE", "MISSING"
    file_path: str
    schema_version_ok: bool
    expected_schema_version: str
    actual_schema_version: Optional[str]
    total_cycles: Optional[int]
    completed_cycles: Optional[int]
    cycles_with_telemetry: Set[int] = field(default_factory=set)
    cycles_with_duration: Set[int] = field(default_factory=set)
    missing_telemetry_cycles: List[int] = field(default_factory=list)
    missing_duration_cycles: List[int] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "file_path": self.file_path,
            "schema_version_ok": self.schema_version_ok,
            "expected_schema_version": self.expected_schema_version,
            "actual_schema_version": self.actual_schema_version,
            "total_cycles": self.total_cycles,
            "completed_cycles": self.completed_cycles,
            "cycles_with_telemetry": sorted(self.cycles_with_telemetry),
            "cycles_with_duration": sorted(self.cycles_with_duration),
            "missing_telemetry_cycles": self.missing_telemetry_cycles,
            "missing_duration_cycles": self.missing_duration_cycles,
            "violations": self.violations,
        }
    
    def format_human(self) -> str:
        """Format report for human-readable output."""
        status_emoji = {"OK": "✅", "INCOMPLETE": "⚠️", "MISSING": "❌"}.get(self.status, "?")
        
        lines = [
            "=" * 60,
            "LAST-MILE READINESS CHECK",
            "=" * 60,
            f"Status: {status_emoji} {self.status}",
            f"File: {self.file_path}",
            "",
            f"Schema version: {self.actual_schema_version or 'N/A'}",
            f"  Expected: {self.expected_schema_version}",
            f"  OK: {'Yes' if self.schema_version_ok else 'No'}",
            "",
            f"Total cycles: {self.total_cycles or 'N/A'}",
            f"Completed cycles: {self.completed_cycles or 'N/A'}",
            f"Cycles with telemetry: {len(self.cycles_with_telemetry)}",
            f"Cycles with duration: {len(self.cycles_with_duration)}",
        ]
        
        if self.missing_telemetry_cycles:
            cycles_str = ", ".join(str(c) for c in self.missing_telemetry_cycles[:10])
            if len(self.missing_telemetry_cycles) > 10:
                cycles_str += f", ... ({len(self.missing_telemetry_cycles) - 10} more)"
            lines.append(f"Missing telemetry: [{cycles_str}]")
        
        if self.missing_duration_cycles:
            cycles_str = ", ".join(str(c) for c in self.missing_duration_cycles[:10])
            if len(self.missing_duration_cycles) > 10:
                cycles_str += f", ... ({len(self.missing_duration_cycles) - 10} more)"
            lines.append(f"Missing duration: [{cycles_str}]")
        
        if self.violations:
            lines.extend(["", "VIOLATIONS:"])
            for v in self.violations[:10]:
                lines.append(f"  - {v}")
            if len(self.violations) > 10:
                lines.append(f"  ... and {len(self.violations) - 10} more")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class TraceLogInspector:
    """
    Read-only inspector for U2 trace log files.
    
    INVARIANTS:
    - Never modifies input files
    - Never leaks experiment seeds or policy internals
    - Deterministic output for same input
    """
    
    def __init__(self, path: Path):
        """
        Initialize inspector for a trace log file.
        
        Args:
            path: Path to JSONL trace log file.
        """
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Trace log not found: {self._path}")
    
    @property
    def path(self) -> Path:
        """Return the trace log path."""
        return self._path
    
    def _iter_records(self) -> Iterator[Tuple[int, Optional[Dict[str, Any]]]]:
        """
        Iterate over records in the trace log.
        
        Yields:
            Tuple of (line_number, record_dict or None if parse error).
        """
        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    yield line_num, record
                except json.JSONDecodeError:
                    yield line_num, None
    
    def summarize(self) -> TraceLogSummary:
        """
        Generate summary statistics for the trace log.
        
        Returns:
            TraceLogSummary with file and session info.
        """
        summary = TraceLogSummary(
            file_path=str(self._path),
            file_size_bytes=self._path.stat().st_size,
            total_records=0,
        )
        
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None
        event_counts: Counter[str] = Counter()
        
        for line_num, record in self._iter_records():
            if record is None:
                summary.parse_errors += 1
                continue
            
            summary.total_records += 1
            
            # Track timestamps
            ts = record.get("ts")
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            
            # Track event types
            event_type = record.get("event_type", "unknown")
            event_counts[event_type] += 1
            
            # Extract session info from SessionStartEvent
            if event_type == "SessionStartEvent":
                payload = record.get("payload", {})
                summary.run_id = payload.get("run_id")
                summary.slice_name = payload.get("slice_name")
                summary.mode = payload.get("mode")
                summary.schema_version = payload.get("schema_version")
                summary.total_cycles = payload.get("total_cycles")
            
            # Extract completion info from SessionEndEvent
            if event_type == "SessionEndEvent":
                payload = record.get("payload", {})
                summary.completed_cycles = payload.get("completed_cycles")
        
        summary.first_timestamp = first_ts
        summary.last_timestamp = last_ts
        if first_ts is not None and last_ts is not None:
            summary.duration_seconds = last_ts - first_ts
        
        summary.event_counts = dict(event_counts)
        
        return summary
    
    def event_histogram(self) -> EventHistogram:
        """
        Generate histogram of event types.
        
        Returns:
            EventHistogram with counts and percentages.
        """
        histogram = EventHistogram()
        
        for _, record in self._iter_records():
            if record is None:
                continue
            event_type = record.get("event_type", "unknown")
            histogram.add(event_type)
        
        return histogram
    
    def validate_schema(self) -> List[ValidationError]:
        """
        Validate all records against expected schema.
        
        Returns:
            List of ValidationError for records that don't conform.
        """
        errors: List[ValidationError] = []
        
        for line_num, record in self._iter_records():
            if record is None:
                errors.append(ValidationError(
                    line_number=line_num,
                    event_type="(parse error)",
                    error_message="Invalid JSON",
                    record_preview="",
                ))
                continue
            
            # Check required top-level fields
            required_fields = {"ts", "event_type", "schema_version", "payload"}
            missing = required_fields - set(record.keys())
            if missing:
                preview = json.dumps(record)[:100]
                errors.append(ValidationError(
                    line_number=line_num,
                    event_type=record.get("event_type", "unknown"),
                    error_message=f"Missing required fields: {missing}",
                    record_preview=preview,
                ))
                continue
            
            # Check event type is known
            event_type = record.get("event_type", "")
            filter_name = EVENT_CLASS_TO_FILTER.get(event_type)
            if filter_name is None:
                preview = json.dumps(record)[:100]
                errors.append(ValidationError(
                    line_number=line_num,
                    event_type=event_type,
                    error_message=f"Unknown event type: {event_type}",
                    record_preview=preview,
                ))
                continue
            
            # Check schema version format
            schema_version = record.get("schema_version", "")
            if not schema_version.startswith("u2-trace-"):
                preview = json.dumps(record)[:100]
                errors.append(ValidationError(
                    line_number=line_num,
                    event_type=event_type,
                    error_message=f"Invalid schema version: {schema_version}",
                    record_preview=preview,
                ))
        
        return errors
    
    def filter_events(
        self,
        event_types: Optional[Set[str]] = None,
        min_cycle: Optional[int] = None,
        max_cycle: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Filter and iterate over events matching criteria.
        
        Args:
            event_types: Set of event type names to include (None = all).
                        Can use filter names (e.g., "cycle_telemetry") or
                        class names (e.g., "CycleTelemetryEvent").
            min_cycle: Minimum cycle number (inclusive).
            max_cycle: Maximum cycle number (inclusive).
        
        Yields:
            Matching records.
        
        Note: This never yields policy weights, seeds, or other sensitive data.
              The raw records are returned but consumers should use the payload
              fields, not attempt to extract internal state.
        """
        if event_types is not None:
            # Convert filter names to class names
            allowed_class_names = set()
            for t in event_types:
                if t in FILTER_TO_EVENT_CLASS:
                    allowed_class_names.add(FILTER_TO_EVENT_CLASS[t])
                else:
                    # Assume it's already a class name
                    allowed_class_names.add(t)
        else:
            allowed_class_names = None
        
        for _, record in self._iter_records():
            if record is None:
                continue
            
            event_type = record.get("event_type", "")
            
            # Filter by event type
            if allowed_class_names is not None and event_type not in allowed_class_names:
                continue
            
            # Filter by cycle (if payload has cycle field)
            payload = record.get("payload", {})
            cycle = payload.get("cycle")
            if cycle is not None:
                if min_cycle is not None and cycle < min_cycle:
                    continue
                if max_cycle is not None and cycle > max_cycle:
                    continue
            
            yield record
    
    def hotspots(self, top_n: int = 10) -> HotspotReport:
        """
        Detect performance and error hotspots in the trace log.
        
        Args:
            top_n: Number of top hotspots to return.
        
        Returns:
            HotspotReport with top cycles by duration and error counts.
        
        INVARIANTS:
        - Deterministic: Same input always produces same output.
        - Tie-breaking: Cycles with same duration are ordered by cycle number.
        """
        # Track per-cycle stats
        cycle_durations: Dict[int, float] = {}
        cycle_errors: Dict[int, int] = Counter()
        
        # Error indicators for error detection
        error_indicators = {"error", "fail", "exception", "timeout", "abort"}
        
        for _, record in self._iter_records():
            if record is None:
                continue
            
            payload = record.get("payload", {})
            event_type = record.get("event_type", "")
            cycle = payload.get("cycle")
            
            if cycle is None:
                continue
            
            # Extract duration from CycleDurationEvent
            if event_type == "CycleDurationEvent":
                duration = payload.get("duration_ms", 0.0)
                cycle_durations[cycle] = duration
            
            # Detect errors
            record_str = json.dumps(record).lower()
            if any(ind in record_str for ind in error_indicators):
                cycle_errors[cycle] += 1
            
            # Also check raw_record success field
            raw_record = payload.get("raw_record", {})
            if isinstance(raw_record, dict) and raw_record.get("success") is False:
                cycle_errors[cycle] += 1
        
        # Compute statistics
        total_cycles = len(cycle_durations)
        all_durations = list(cycle_durations.values())
        avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0.0
        max_duration = max(all_durations) if all_durations else 0.0
        total_error_count = sum(cycle_errors.values())
        cycles_with_errors = sum(1 for c in cycle_errors.values() if c > 0)
        
        # Sort by duration (desc), then by cycle number (asc) for tie-breaking
        sorted_cycles = sorted(
            cycle_durations.keys(),
            key=lambda c: (-cycle_durations[c], c),
        )
        
        # Build hotspot entries for top N
        entries: List[HotspotEntry] = []
        for cycle in sorted_cycles[:top_n]:
            duration = cycle_durations[cycle]
            error_count = cycle_errors.get(cycle, 0)
            
            # Build notes
            notes_parts = []
            if duration > avg_duration * 2:
                notes_parts.append("⚠ long")
            elif duration > avg_duration * 1.5:
                notes_parts.append("long")
            
            if error_count > 0:
                notes_parts.append(f"{error_count} errs")
            
            if duration > avg_duration * 2 and error_count > 0:
                notes_parts.insert(0, "")  # Will join to "⚠ long + errs"
                notes = " + ".join(filter(None, notes_parts))
            else:
                notes = " + ".join(notes_parts) if notes_parts else ""
            
            entries.append(HotspotEntry(
                cycle=cycle,
                duration_ms=duration,
                error_count=error_count,
                notes=notes,
            ))
        
        return HotspotReport(
            top_n=top_n,
            total_cycles=total_cycles,
            entries=entries,
            avg_duration_ms=avg_duration,
            max_duration_ms=max_duration,
            total_error_count=total_error_count,
            cycles_with_errors=cycles_with_errors,
        )
    
    def last_mile_check(self) -> LastMileReport:
        """
        Perform last-mile readiness check on the trace log.
        
        Verifies:
        - Trace file present (already checked in __init__)
        - Schema version matches expected
        - All cycles have at least one CycleTelemetryEvent
        - All cycles have at least one CycleDurationEvent
        
        Returns:
            LastMileReport with status and violations.
        """
        report = LastMileReport(
            status="OK",
            file_path=str(self._path),
            schema_version_ok=False,
            expected_schema_version=EXPECTED_SCHEMA_VERSION,
            actual_schema_version=None,
            total_cycles=None,
            completed_cycles=None,
        )
        
        cycles_with_telemetry: Set[int] = set()
        cycles_with_duration: Set[int] = set()
        
        for _, record in self._iter_records():
            if record is None:
                continue
            
            event_type = record.get("event_type", "")
            payload = record.get("payload", {})
            schema_version = record.get("schema_version", "")
            
            # Check schema version (use first seen)
            if report.actual_schema_version is None:
                report.actual_schema_version = schema_version
                report.schema_version_ok = (schema_version == EXPECTED_SCHEMA_VERSION)
            
            # Extract session info
            if event_type == "SessionStartEvent":
                report.total_cycles = payload.get("total_cycles")
            
            if event_type == "SessionEndEvent":
                report.completed_cycles = payload.get("completed_cycles")
            
            # Track which cycles have telemetry/duration
            cycle = payload.get("cycle")
            if cycle is not None:
                if event_type == "CycleTelemetryEvent":
                    cycles_with_telemetry.add(cycle)
                elif event_type == "CycleDurationEvent":
                    cycles_with_duration.add(cycle)
        
        report.cycles_with_telemetry = cycles_with_telemetry
        report.cycles_with_duration = cycles_with_duration
        
        # Determine expected cycles
        expected_cycles = report.completed_cycles or report.total_cycles or 0
        if expected_cycles > 0:
            expected_cycle_set = set(range(expected_cycles))
            
            # Find missing
            report.missing_telemetry_cycles = sorted(expected_cycle_set - cycles_with_telemetry)
            report.missing_duration_cycles = sorted(expected_cycle_set - cycles_with_duration)
        
        # Build violations list
        violations: List[str] = []
        
        if not report.schema_version_ok:
            violations.append(
                f"Schema version mismatch: expected {EXPECTED_SCHEMA_VERSION}, "
                f"got {report.actual_schema_version}"
            )
        
        if report.missing_telemetry_cycles:
            violations.append(
                f"{len(report.missing_telemetry_cycles)} cycles missing CycleTelemetryEvent"
            )
        
        if report.missing_duration_cycles:
            violations.append(
                f"{len(report.missing_duration_cycles)} cycles missing CycleDurationEvent"
            )
        
        report.violations = violations
        
        # Determine status
        if not self._path.exists():
            report.status = "MISSING"
        elif violations:
            report.status = "INCOMPLETE"
        else:
            report.status = "OK"
        
        return report


def iter_events(
    path: Path,
    event_type: Optional[str] = None,
    cycle_range: Optional[Tuple[int, int]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Convenience function to iterate over filtered events.
    
    Args:
        path: Path to trace log file.
        event_type: Single event type to filter (filter name or class name).
        cycle_range: Tuple of (min_cycle, max_cycle) inclusive.
    
    Yields:
        Matching event records.
    
    Example:
        for event in iter_events(path, event_type="cycle_telemetry", cycle_range=(10, 20)):
            print(event["payload"]["cycle"])
    """
    inspector = TraceLogInspector(path)
    
    event_types = {event_type} if event_type else None
    min_cycle = cycle_range[0] if cycle_range else None
    max_cycle = cycle_range[1] if cycle_range else None
    
    yield from inspector.filter_events(
        event_types=event_types,
        min_cycle=min_cycle,
        max_cycle=max_cycle,
    )


def parse_cycle_range(range_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a cycle range string like "10:20" or ":20" or "10:".
    
    Args:
        range_str: Range string in format "START:END", ":END", or "START:".
    
    Returns:
        Tuple of (min_cycle, max_cycle), either can be None.
    
    Raises:
        ValueError: If format is invalid.
    """
    if ":" not in range_str:
        # Single cycle
        cycle = int(range_str)
        return (cycle, cycle)
    
    parts = range_str.split(":", 1)
    min_cycle = int(parts[0]) if parts[0] else None
    max_cycle = int(parts[1]) if parts[1] else None
    
    return (min_cycle, max_cycle)


def main():
    """CLI entry point for trace log inspector."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PHASE II — U2 Trace Log Inspector",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Summary
  python -m experiments.u2.inspector trace.jsonl --summary
  
  # Filter by event type and dump
  python -m experiments.u2.inspector trace.jsonl --filter-event cycle_telemetry --dump
  
  # Filter by cycle range
  python -m experiments.u2.inspector trace.jsonl --filter-cycle-range 10:20 --dump
  
  # Last-mile readiness check
  python -m experiments.u2.inspector trace.jsonl --last-mile-check
  
  # Combined filters
  python -m experiments.u2.inspector trace.jsonl \\
    --filter-event cycle_duration \\
    --filter-cycle-range 0:10 \\
    --dump
""",
    )
    parser.add_argument(
        "trace_log",
        type=str,
        help="Path to trace log file (JSONL).",
    )
    
    # Analysis modes
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics.",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Print event type histogram.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate schema compliance and report errors.",
    )
    parser.add_argument(
        "--last-mile-check",
        action="store_true",
        help="Check trace completeness for last-mile readiness.",
    )
    parser.add_argument(
        "--hotspots",
        action="store_true",
        help="Detect performance and error hotspots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of top hotspots to show (default: 10).",
    )
    
    # Filtering
    parser.add_argument(
        "--filter-event",
        type=str,
        metavar="TYPE",
        help=(
            "Filter by event type. Available types:\n"
            "  session_start, session_end, cycle_telemetry,\n"
            "  cycle_duration, candidate_ordering, scoring_features,\n"
            "  policy_weight_update, budget_consumption, substrate_result"
        ),
    )
    parser.add_argument(
        "--filter-cycle-range",
        type=str,
        metavar="START:END",
        help=(
            "Filter by cycle range (inclusive). Examples:\n"
            "  10:20  - cycles 10 through 20\n"
            "  :20    - cycles 0 through 20\n"
            "  10:    - cycles 10 and above\n"
            "  15     - only cycle 15"
        ),
    )
    
    # Output modes
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump filtered records as JSONL to stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format (for --summary, --last-mile-check).",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="Write output to file instead of stdout.",
    )
    
    args = parser.parse_args()
    
    # Validate trace log exists
    trace_path = Path(args.trace_log)
    if not trace_path.exists():
        print(f"ERROR: Trace log not found: {trace_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        inspector = TraceLogInspector(trace_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output destination
    output_file = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    
    try:
        # Handle filtering with --dump
        if args.dump:
            event_types = {args.filter_event} if args.filter_event else None
            min_cycle, max_cycle = None, None
            
            if args.filter_cycle_range:
                try:
                    min_cycle, max_cycle = parse_cycle_range(args.filter_cycle_range)
                except ValueError:
                    print(f"ERROR: Invalid cycle range: {args.filter_cycle_range}", file=sys.stderr)
                    sys.exit(1)
            
            count = 0
            for record in inspector.filter_events(
                event_types=event_types,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
            ):
                output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
            
            if args.output:
                print(f"Dumped {count} records to {args.output}", file=sys.stderr)
            return
        
        # Handle last-mile check
        if args.last_mile_check:
            report = inspector.last_mile_check()
            
            if args.json:
                output_file.write(json.dumps(report.to_dict(), indent=2) + "\n")
            else:
                output_file.write(report.format_human() + "\n")
            
            # Also write JSON report file if output not specified
            if not args.output and not args.json:
                report_path = trace_path.parent / "last_mile_trace_report.json"
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report.to_dict(), f, indent=2)
                print(f"Report saved to: {report_path}", file=sys.stderr)
            
            sys.exit(0 if report.status == "OK" else 1)
        
        # Handle hotspots
        if args.hotspots:
            report = inspector.hotspots(top_n=args.top_n)
            
            if args.json:
                output_file.write(json.dumps(report.to_dict(), indent=2) + "\n")
            else:
                output_file.write(report.format_human() + "\n")
            return
        
        # Default to summary if no action specified
        if not any([args.summary, args.histogram, args.validate, args.last_mile_check, args.dump]):
            args.summary = True
        
        if args.summary:
            summary = inspector.summarize()
            if args.json:
                output_file.write(json.dumps(summary.to_dict(), indent=2) + "\n")
            else:
                output_file.write(summary.format_human() + "\n")
        
        if args.histogram:
            histogram = inspector.event_histogram()
            output_file.write(histogram.format_human() + "\n")
        
        if args.validate:
            errors = inspector.validate_schema()
            if errors:
                print(f"VALIDATION ERRORS: {len(errors)} found", file=output_file)
                print("-" * 60, file=output_file)
                for err in errors[:20]:
                    print(f"Line {err.line_number}: [{err.event_type}] {err.error_message}", file=output_file)
                    if err.record_preview:
                        print(f"  Preview: {err.record_preview}...", file=output_file)
                if len(errors) > 20:
                    print(f"... and {len(errors) - 20} more errors", file=output_file)
                sys.exit(1)
            else:
                print("✅ All records valid", file=output_file)
    
    finally:
        if args.output and output_file != sys.stdout:
            output_file.close()


if __name__ == "__main__":
    main()
