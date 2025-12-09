#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Trace Log Inspector
═══════════════════════════════════════════════════════════════════════════════

Inspect and summarize Phase II trace JSONL logs.
Read-only tool for debugging and analysis.

Usage:
    # Basic summary
    uv run python experiments/inspect_trace_log.py --trace results/u2_test/trace.jsonl
    
    # Verbose output with samples
    uv run python experiments/inspect_trace_log.py --trace results/trace.jsonl --verbose
    
    # JSON output
    uv run python experiments/inspect_trace_log.py --trace results/trace.jsonl --json

Output:
    events:
      session_start: 1
      session_end: 1
      cycle_telemetry: 100
      policy_weight_update: 10
    
    cycles: 0..99
    slice: slice_uplift_goal
    mode: rfl

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TraceLogSummary:
    """Summary of a trace log file."""
    path: str = ""
    total_events: int = 0
    event_counts: Dict[str, int] = field(default_factory=dict)
    schema_version: Optional[str] = None
    slice_name: Optional[str] = None
    mode: Optional[str] = None
    first_cycle: Optional[int] = None
    last_cycle: Optional[int] = None
    run_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "path": self.path,
            "total_events": self.total_events,
            "event_counts": dict(sorted(self.event_counts.items())),
            "schema_version": self.schema_version,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "cycles": {
                "first": self.first_cycle,
                "last": self.last_cycle,
                "count": (self.last_cycle - self.first_cycle + 1) if self.first_cycle is not None and self.last_cycle is not None else 0,
            },
            "run_id": self.run_id,
            "errors": self.errors if self.errors else None,
        }


def parse_trace_log(path: Path, max_errors: int = 10) -> TraceLogSummary:
    """
    Parse a trace log file and extract summary statistics.
    
    Args:
        path: Path to the JSONL trace log.
        max_errors: Maximum parse errors to record.
        
    Returns:
        TraceLogSummary with statistics.
    """
    summary = TraceLogSummary(path=str(path))
    event_counter: Counter = Counter()
    cycles_seen: List[int] = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if len(summary.errors) < max_errors:
                    summary.errors.append(f"Line {line_num}: JSON parse error: {e}")
                continue
            
            summary.total_events += 1
            
            # Extract event type
            event_type = record.get("event_type", "unknown")
            event_counter[event_type] += 1
            
            # Extract schema version (first seen)
            if summary.schema_version is None:
                summary.schema_version = record.get("schema_version")
            
            # Extract payload
            payload = record.get("payload", {})
            
            # Extract slice/mode/run_id from various event types
            if "slice_name" in payload:
                if summary.slice_name is None:
                    summary.slice_name = payload["slice_name"]
            if "mode" in payload:
                if summary.mode is None:
                    summary.mode = payload["mode"]
            if "run_id" in payload:
                if summary.run_id is None:
                    summary.run_id = payload["run_id"]
            
            # Extract cycle numbers
            if "cycle" in payload:
                cycle = payload["cycle"]
                if isinstance(cycle, int) and cycle >= 0:
                    cycles_seen.append(cycle)
    
    # Compute cycle range
    if cycles_seen:
        summary.first_cycle = min(cycles_seen)
        summary.last_cycle = max(cycles_seen)
    
    summary.event_counts = dict(event_counter)
    
    return summary


def format_summary(summary: TraceLogSummary, verbose: bool = False) -> str:
    """Format a summary for human-readable output."""
    lines = []
    
    lines.append(f"Trace Log: {summary.path}")
    lines.append("─" * 60)
    
    # Metadata
    if summary.schema_version:
        lines.append(f"schema_version: {summary.schema_version}")
    if summary.run_id:
        lines.append(f"run_id: {summary.run_id}")
    if summary.slice_name:
        lines.append(f"slice: {summary.slice_name}")
    if summary.mode:
        lines.append(f"mode: {summary.mode}")
    
    # Cycle range
    if summary.first_cycle is not None and summary.last_cycle is not None:
        cycle_count = summary.last_cycle - summary.first_cycle + 1
        lines.append(f"cycles: {summary.first_cycle}..{summary.last_cycle} ({cycle_count} cycles)")
    
    lines.append("")
    lines.append(f"Total events: {summary.total_events}")
    lines.append("")
    
    # Event counts
    lines.append("Event Counts:")
    for event_type in sorted(summary.event_counts.keys()):
        count = summary.event_counts[event_type]
        pct = 100.0 * count / summary.total_events if summary.total_events > 0 else 0.0
        lines.append(f"  {event_type:<30} {count:>8} ({pct:>5.1f}%)")
    
    # Errors
    if summary.errors:
        lines.append("")
        lines.append(f"Parse Errors ({len(summary.errors)}):")
        for err in summary.errors:
            lines.append(f"  {err}")
    
    return "\n".join(lines)


def format_yaml_like(summary: TraceLogSummary) -> str:
    """Format as YAML-like structure for simple readability."""
    lines = []
    
    lines.append("events:")
    for event_type in sorted(summary.event_counts.keys()):
        lines.append(f"  {event_type}: {summary.event_counts[event_type]}")
    
    lines.append("")
    
    if summary.first_cycle is not None and summary.last_cycle is not None:
        lines.append(f"cycles: {summary.first_cycle}..{summary.last_cycle}")
    
    if summary.slice_name:
        lines.append(f"slice: {summary.slice_name}")
    if summary.mode:
        lines.append(f"mode: {summary.mode}")
    
    return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect Phase II trace JSONL logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--trace",
        required=True,
        type=str,
        help="Path to the trace JSONL log file."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with additional details."
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact YAML-like output (default for simple inspection)."
    )
    
    args = parser.parse_args()
    
    print("PHASE II — NOT USED IN PHASE I", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Check file exists
    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"ERROR: Trace log not found: {trace_path}", file=sys.stderr)
        sys.exit(1)
    
    # Parse log
    try:
        summary = parse_trace_log(trace_path)
    except Exception as e:
        print(f"ERROR: Failed to parse trace log: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output
    if args.json:
        output = {
            "phase": "PHASE II — NOT USED IN PHASE I",
            "summary": summary.to_dict(),
        }
        print(json.dumps(output, indent=2))
    elif args.compact or not args.verbose:
        print(format_yaml_like(summary))
    else:
        print(format_summary(summary, verbose=args.verbose))


if __name__ == "__main__":
    main()

