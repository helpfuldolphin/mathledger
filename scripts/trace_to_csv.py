#!/usr/bin/env python3
# PHASE II — NOT RUN IN PHASE I
"""
Trace to CSV Exporter

STATUS: PHASE II — NOT RUN IN PHASE I

Exports trace log events to CSV format for analysis in Excel, pandas, etc.

Features:
- Extracts minimal but useful subset of fields
- Stable CSV headers across versions
- Pure Python stdlib (no pandas dependency)
- Handles missing fields gracefully

INVARIANTS:
- Read-only: Never modifies input files
- Deterministic: Same input always produces same output
- Stable schema: Headers don't change between runs

CSV Schema (v1):
    cycle,event_type,timestamp,duration_ms,substrate_duration_ms,
    mode,slice_name,success,error_kind

Example Output:
    cycle,event_type,timestamp,duration_ms,substrate_duration_ms,mode,slice_name,success,error_kind
    0,CycleDurationEvent,1699123456.789,45.2,12.3,baseline,test_slice,,
    0,CycleTelemetryEvent,1699123456.790,,,baseline,test_slice,true,
    1,CycleDurationEvent,1699123456.891,52.1,15.2,baseline,test_slice,,

Usage:
    uv run python scripts/trace_to_csv.py --trace trace.jsonl --output trace_metrics.csv
    
    # Filter by event type
    uv run python scripts/trace_to_csv.py --trace trace.jsonl --output durations.csv \\
        --event-type cycle_duration
    
    # Include all fields (verbose mode)
    uv run python scripts/trace_to_csv.py --trace trace.jsonl --output full.csv --verbose
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.u2.inspector import TraceLogInspector, FILTER_TO_EVENT_CLASS

# CSV Schema version
CSV_SCHEMA_VERSION = "1.0"

# Standard CSV headers (stable across versions)
STANDARD_HEADERS = [
    "cycle",
    "event_type",
    "timestamp",
    "duration_ms",
    "substrate_duration_ms",
    "mode",
    "slice_name",
    "success",
    "error_kind",
]

# Extended headers for verbose mode
EXTENDED_HEADERS = STANDARD_HEADERS + [
    "schema_version",
    "run_id",
    "item",
    "result",
    "candidates_considered",
    "budget_exhausted",
]

# Error indicators for error_kind detection
ERROR_INDICATORS = {
    "timeout": ["timeout"],
    "parse_error": ["parse", "json", "decode"],
    "budget_exhausted": ["budget", "exhaust"],
    "substrate_error": ["substrate", "execution"],
    "validation_error": ["validation", "invalid"],
    "unknown_error": ["error", "fail", "exception"],
}


def detect_error_kind(record: Dict[str, Any]) -> Optional[str]:
    """
    Detect the kind of error from a trace record.
    
    Args:
        record: Trace record dict.
    
    Returns:
        Error kind string or None if no error detected.
    """
    record_str = json.dumps(record).lower()
    
    # Check payload for explicit success=False
    payload = record.get("payload", {})
    raw_record = payload.get("raw_record", {})
    
    if isinstance(raw_record, dict) and raw_record.get("success") is False:
        # Try to determine specific error kind
        for kind, indicators in ERROR_INDICATORS.items():
            if any(ind in record_str for ind in indicators):
                return kind
        return "unknown_error"
    
    # Check for error indicators in the record
    for kind, indicators in ERROR_INDICATORS.items():
        if kind == "unknown_error":
            continue  # Check this last
        if any(ind in record_str for ind in indicators):
            return kind
    
    return None


def extract_row(record: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Extract CSV row data from a trace record.
    
    Args:
        record: Trace record dict.
        verbose: If True, extract extended fields.
    
    Returns:
        Dict with field names as keys.
    """
    payload = record.get("payload", {})
    raw_record = payload.get("raw_record", {})
    
    row = {
        "cycle": payload.get("cycle"),
        "event_type": record.get("event_type"),
        "timestamp": record.get("ts"),
        "duration_ms": payload.get("duration_ms"),
        "substrate_duration_ms": payload.get("substrate_duration_ms"),
        "mode": payload.get("mode"),
        "slice_name": payload.get("slice_name"),
        "success": None,
        "error_kind": None,
    }
    
    # Extract success from various locations
    if isinstance(raw_record, dict) and "success" in raw_record:
        row["success"] = raw_record["success"]
    elif "success" in payload:
        row["success"] = payload["success"]
    
    # Detect error kind
    row["error_kind"] = detect_error_kind(record)
    
    if verbose:
        row["schema_version"] = record.get("schema_version")
        row["run_id"] = payload.get("run_id")
        row["item"] = payload.get("item") or (raw_record.get("item") if isinstance(raw_record, dict) else None)
        row["result"] = raw_record.get("result") if isinstance(raw_record, dict) else None
        row["candidates_considered"] = payload.get("candidates_considered")
        row["budget_exhausted"] = payload.get("budget_exhausted")
    
    return row


def format_value(value: Any) -> str:
    """
    Format a value for CSV output.
    
    Args:
        value: Any Python value.
    
    Returns:
        String representation for CSV.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Limit precision for readability
        if value == int(value):
            return str(int(value))
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        # Compact JSON for complex values
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def export_to_csv(
    trace_path: Path,
    output_path: Path,
    event_types: Optional[Set[str]] = None,
    cycle_range: Optional[tuple] = None,
    verbose: bool = False,
) -> int:
    """
    Export trace to CSV file.
    
    Args:
        trace_path: Path to trace JSONL file.
        output_path: Path to output CSV file.
        event_types: Optional set of event types to include.
        cycle_range: Optional (min, max) cycle range.
        verbose: If True, include extended headers.
    
    Returns:
        Number of rows written.
    """
    inspector = TraceLogInspector(trace_path)
    headers = EXTENDED_HEADERS if verbose else STANDARD_HEADERS
    
    # Convert filter names to class names
    if event_types:
        class_names = set()
        for t in event_types:
            if t in FILTER_TO_EVENT_CLASS:
                class_names.add(FILTER_TO_EVENT_CLASS[t])
            else:
                class_names.add(t)
        event_types = class_names
    
    min_cycle = cycle_range[0] if cycle_range else None
    max_cycle = cycle_range[1] if cycle_range else None
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    row_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        
        for record in inspector.filter_events(
            event_types=event_types,
            min_cycle=min_cycle,
            max_cycle=max_cycle,
        ):
            row = extract_row(record, verbose=verbose)
            # Format all values
            formatted_row = {k: format_value(v) for k, v in row.items()}
            writer.writerow(formatted_row)
            row_count += 1
    
    return row_count


def main():
    parser = argparse.ArgumentParser(
        description="PHASE II — Trace to CSV Exporter",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
CSV Schema (v{CSV_SCHEMA_VERSION}):
  Standard headers:
    {', '.join(STANDARD_HEADERS)}
  
  Extended headers (with --verbose):
    {', '.join(EXTENDED_HEADERS)}

Examples:
  # Basic export
  python scripts/trace_to_csv.py --trace trace.jsonl --output metrics.csv
  
  # Filter by event type
  python scripts/trace_to_csv.py --trace trace.jsonl --output durations.csv \\
    --event-type cycle_duration
  
  # Filter by cycle range
  python scripts/trace_to_csv.py --trace trace.jsonl --output subset.csv \\
    --cycle-range 10:20
  
  # Verbose mode with all fields
  python scripts/trace_to_csv.py --trace trace.jsonl --output full.csv --verbose
""",
    )
    
    parser.add_argument(
        "--trace",
        type=str,
        required=True,
        help="Path to trace log file (JSONL).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--event-type",
        type=str,
        action="append",
        dest="event_types",
        help="Filter by event type (can be specified multiple times).",
    )
    parser.add_argument(
        "--cycle-range",
        type=str,
        help="Filter by cycle range (e.g., '10:20', ':20', '10:').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include extended fields in output.",
    )
    
    args = parser.parse_args()
    
    trace_path = Path(args.trace)
    output_path = Path(args.output)
    
    if not trace_path.exists():
        print(f"ERROR: Trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)
    
    # Parse cycle range
    cycle_range = None
    if args.cycle_range:
        from experiments.u2.inspector import parse_cycle_range
        try:
            cycle_range = parse_cycle_range(args.cycle_range)
        except ValueError:
            print(f"ERROR: Invalid cycle range: {args.cycle_range}", file=sys.stderr)
            sys.exit(1)
    
    # Parse event types
    event_types = set(args.event_types) if args.event_types else None
    
    try:
        row_count = export_to_csv(
            trace_path=trace_path,
            output_path=output_path,
            event_types=event_types,
            cycle_range=cycle_range,
            verbose=args.verbose,
        )
        
        print(f"Exported {row_count:,} rows to {output_path}")
        print(f"Schema version: {CSV_SCHEMA_VERSION}")
        print(f"Headers: {', '.join(EXTENDED_HEADERS if args.verbose else STANDARD_HEADERS)}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

