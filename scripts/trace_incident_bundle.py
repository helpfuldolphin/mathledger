#!/usr/bin/env python3
# PHASE II — NOT RUN IN PHASE I
"""
Trace Incident Bundle Extractor

STATUS: PHASE II — NOT RUN IN PHASE I

Extracts minimal trace bundles for bug reports and incident investigations.

Given a trace log and cycle(s), produces a compact JSONL file containing:
- SessionStartEvent (for context)
- All events for the specified cycle(s)
- Any error-related events
- SessionEndEvent (if present)

The output is suitable for attaching to bug reports / incident tickets.

INVARIANTS:
- Read-only: Never modifies input files
- Deterministic: Same input always produces same output
- Minimal: Only includes relevant events, not full trace

Usage:
    # Single cycle
    uv run python scripts/trace_incident_bundle.py trace.jsonl --cycle 42
    
    # Cycle range
    uv run python scripts/trace_incident_bundle.py trace.jsonl --cycle-range 40:45
    
    # Custom output
    uv run python scripts/trace_incident_bundle.py trace.jsonl --cycle 42 --output incident_42.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, Set, List

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.u2.inspector import TraceLogInspector, parse_cycle_range


# Event types to always include in bundles (for context)
CONTEXT_EVENTS = {"SessionStartEvent", "SessionEndEvent"}

# Event types that indicate errors or issues
ERROR_INDICATORS = {"error", "fail", "exception", "timeout"}


def is_error_event(record: Dict[str, Any]) -> bool:
    """
    Check if an event indicates an error condition.
    
    Looks for error indicators in:
    - Event type name
    - Payload values (recursive string search)
    """
    event_type = record.get("event_type", "").lower()
    if any(ind in event_type for ind in ERROR_INDICATORS):
        return True
    
    # Check payload for error indicators
    payload = record.get("payload", {})
    payload_str = json.dumps(payload).lower()
    return any(ind in payload_str for ind in ERROR_INDICATORS)


def extract_incident_bundle(
    trace_path: Path,
    cycles: Set[int],
    include_context: bool = True,
    include_errors: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Extract events for incident bundle.
    
    Args:
        trace_path: Path to trace log file.
        cycles: Set of cycle numbers to include.
        include_context: Include session start/end events.
        include_errors: Include events that indicate errors.
    
    Yields:
        Records to include in the bundle.
    """
    inspector = TraceLogInspector(trace_path)
    
    for record in inspector.filter_events():
        event_type = record.get("event_type", "")
        payload = record.get("payload", {})
        cycle = payload.get("cycle")
        
        # Always include context events
        if include_context and event_type in CONTEXT_EVENTS:
            yield record
            continue
        
        # Include events for specified cycles
        if cycle is not None and cycle in cycles:
            yield record
            continue
        
        # Include error events
        if include_errors and is_error_event(record):
            yield record
            continue


def generate_bundle_filename(
    trace_path: Path,
    run_id: Optional[str],
    cycles: Set[int],
) -> str:
    """Generate a descriptive filename for the incident bundle."""
    # Use run_id if available, otherwise trace filename
    base = run_id or trace_path.stem
    
    # Describe cycles
    if len(cycles) == 1:
        cycle_desc = f"cycle_{min(cycles)}"
    else:
        cycle_desc = f"cycles_{min(cycles)}-{max(cycles)}"
    
    return f"trace_incident_{base}_{cycle_desc}.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="PHASE II — Trace Incident Bundle Extractor",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Extract bundle for single cycle
  python scripts/trace_incident_bundle.py trace.jsonl --cycle 42
  
  # Extract bundle for cycle range
  python scripts/trace_incident_bundle.py trace.jsonl --cycle-range 40:45
  
  # Custom output path
  python scripts/trace_incident_bundle.py trace.jsonl --cycle 42 --output incident.jsonl
  
  # Exclude context events
  python scripts/trace_incident_bundle.py trace.jsonl --cycle 42 --no-context
  
  # Include all events (not just for specified cycles)
  python scripts/trace_incident_bundle.py trace.jsonl --cycle 42 --include-all
""",
    )
    
    parser.add_argument(
        "trace_log",
        type=str,
        help="Path to trace log file (JSONL).",
    )
    
    parser.add_argument(
        "--cycle",
        type=int,
        metavar="N",
        help="Single cycle to extract.",
    )
    
    parser.add_argument(
        "--cycle-range",
        type=str,
        metavar="START:END",
        help="Cycle range to extract (inclusive). Examples: 10:20, :20, 10:",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="Output file path. Default: auto-generated based on run_id and cycles.",
    )
    
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Exclude session start/end events.",
    )
    
    parser.add_argument(
        "--no-errors",
        action="store_true",
        help="Exclude error events outside specified cycles.",
    )
    
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all events, not just for specified cycles.",
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of extracted bundle.",
    )
    
    args = parser.parse_args()
    
    # Validate input
    trace_path = Path(args.trace_log)
    if not trace_path.exists():
        print(f"ERROR: Trace log not found: {trace_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine cycles to extract
    cycles: Set[int] = set()
    
    if args.cycle is not None:
        cycles.add(args.cycle)
    
    if args.cycle_range:
        try:
            min_cycle, max_cycle = parse_cycle_range(args.cycle_range)
            if min_cycle is not None and max_cycle is not None:
                cycles.update(range(min_cycle, max_cycle + 1))
            elif min_cycle is not None:
                # Need to determine max from trace
                inspector = TraceLogInspector(trace_path)
                summary = inspector.summarize()
                max_cycle = summary.completed_cycles or summary.total_cycles or 0
                cycles.update(range(min_cycle, max_cycle))
            elif max_cycle is not None:
                cycles.update(range(0, max_cycle + 1))
        except ValueError:
            print(f"ERROR: Invalid cycle range: {args.cycle_range}", file=sys.stderr)
            sys.exit(1)
    
    if not cycles and not args.include_all:
        print("ERROR: Must specify --cycle, --cycle-range, or --include-all", file=sys.stderr)
        sys.exit(1)
    
    # Get run_id for filename generation
    run_id: Optional[str] = None
    try:
        inspector = TraceLogInspector(trace_path)
        summary = inspector.summarize()
        run_id = summary.run_id
    except Exception:
        pass
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        filename = generate_bundle_filename(trace_path, run_id, cycles)
        output_path = trace_path.parent / filename
    
    # Extract bundle
    bundle_records: List[Dict[str, Any]] = []
    
    if args.include_all:
        # Include all events
        inspector = TraceLogInspector(trace_path)
        for record in inspector.filter_events():
            bundle_records.append(record)
    else:
        # Extract for specified cycles
        for record in extract_incident_bundle(
            trace_path,
            cycles,
            include_context=not args.no_context,
            include_errors=not args.no_errors,
        ):
            bundle_records.append(record)
    
    # Write bundle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in bundle_records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    
    # Summary
    print(f"Incident bundle extracted: {output_path}")
    print(f"  Records: {len(bundle_records)}")
    print(f"  Cycles: {sorted(cycles) if cycles else 'all'}")
    
    if args.summary:
        event_counts: Dict[str, int] = {}
        for record in bundle_records:
            event_type = record.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        print("\nEvent breakdown:")
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")
    
    # Calculate bundle size
    bundle_size = output_path.stat().st_size
    trace_size = trace_path.stat().st_size
    reduction = (1 - bundle_size / trace_size) * 100 if trace_size > 0 else 0
    
    print(f"\nSize: {bundle_size:,} bytes ({reduction:.1f}% reduction from original)")


if __name__ == "__main__":
    main()

