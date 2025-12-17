#!/usr/bin/env python3
"""
Compute CTRPK (Curriculum Transition Requests Per 1K Cycles) from Runtime Guard Signals.

Reads curriculum runtime guard JSONL logs and computes CTRPK metrics for evidence packs.

SHADOW MODE CONTRACT:
- This script is purely observational
- It reads governance signals and computes metrics
- It does not execute enforcement decisions
- Output is advisory only

Usage:
    python scripts/compute_ctrpk_from_signals.py \
        --signals-jsonl results/curriculum/runtime_guard_signals.jsonl \
        --output results/curriculum/ctrpk_compact.json

    # With trend computation (requires historical data)
    python scripts/compute_ctrpk_from_signals.py \
        --signals-jsonl results/curriculum/runtime_guard_signals.jsonl \
        --signals-jsonl-24h results/curriculum/runtime_guard_signals_24h.jsonl \
        --output results/curriculum/ctrpk_compact.json

Output:
    CTRPK compact block suitable for evidence["governance"]["curriculum"]["ctrpk"]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Import CTRPK functions from curriculum integration
from curriculum.integration import (
    compute_ctrpk,
    ctrpk_to_status_light,
    compute_ctrpk_trend,
    council_classify_ctrpk,
    build_ctrpk_compact,
    TrendDirection,
)


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file, returning list of records."""
    records = []
    if not file_path.exists():
        return records
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def count_transition_requests(signals: List[Dict[str, Any]]) -> int:
    """
    Count TRANSITION_REQUESTED governance signals.

    A signal is counted as a transition request if:
    - signal_type == "TRANSITION_REQUESTED" OR
    - signal_type == "DRIFT_DETECTED" with severity == "SEMANTIC" OR
    - governance_action indicates transition was requested

    Args:
        signals: List of governance signal records from JSONL

    Returns:
        Count of transition requests
    """
    count = 0
    for signal in signals:
        signal_type = signal.get("signal_type", "")
        severity = signal.get("severity", "")
        governance_action = signal.get("governance_action", "")

        # Direct transition request
        if signal_type == "TRANSITION_REQUESTED":
            count += 1
            continue

        # Drift detected with semantic severity implies transition consideration
        if signal_type == "DRIFT_DETECTED" and severity == "SEMANTIC":
            count += 1
            continue

        # Check governance action for transition indicators
        if "transition" in governance_action.lower():
            count += 1
            continue

    return count


def count_total_cycles(signals: List[Dict[str, Any]]) -> int:
    """
    Extract total cycle count from signals.

    Looks for:
    - Max cycle number in signals
    - Explicit total_cycles field in summary signals
    - Falls back to signal count as proxy

    Args:
        signals: List of governance signal records

    Returns:
        Total cycle count
    """
    max_cycle = 0
    explicit_total = None

    for signal in signals:
        # Check for explicit total_cycles
        if "total_cycles" in signal:
            explicit_total = signal["total_cycles"]

        # Check for cycle number
        cycle = signal.get("cycle", signal.get("cycle_number", 0))
        if isinstance(cycle, int) and cycle > max_cycle:
            max_cycle = cycle

        # Check run context for cycle info
        run_context = signal.get("run_context", {})
        if isinstance(run_context, dict):
            ctx_cycle = run_context.get("cycle", 0)
            if isinstance(ctx_cycle, int) and ctx_cycle > max_cycle:
                max_cycle = ctx_cycle

    # Prefer explicit total if available
    if explicit_total is not None and isinstance(explicit_total, int):
        return explicit_total

    # Use max cycle + 1 (cycles are 0-indexed)
    if max_cycle > 0:
        return max_cycle + 1

    # Fallback: use signal count as rough proxy
    return len(signals)


def count_semantic_violations(signals: List[Dict[str, Any]]) -> int:
    """Count signals with SEMANTIC severity."""
    return sum(
        1 for s in signals
        if s.get("severity") == "SEMANTIC"
        or s.get("drift_severity") == "SEMANTIC"
    )


def count_blocked_requests(signals: List[Dict[str, Any]]) -> int:
    """Count signals that would have been blocked."""
    return sum(
        1 for s in signals
        if s.get("status") == "BLOCK"
        or s.get("drift_status") == "BLOCK"
        or s.get("hypothetical", {}).get("would_allow_transition") is False
    )


def compute_ctrpk_from_signals(
    signals: List[Dict[str, Any]],
    signals_24h: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute CTRPK metrics from governance signal records.

    Args:
        signals: Recent signals (1h window by default)
        signals_24h: Optional 24h signals for trend computation

    Returns:
        CTRPK compact block for evidence packs
    """
    # Count metrics from recent signals
    transition_requests = count_transition_requests(signals)
    total_cycles = count_total_cycles(signals)
    semantic_violations = count_semantic_violations(signals)
    blocked_requests = count_blocked_requests(signals)

    # Compute current CTRPK
    ctrpk_value = compute_ctrpk(transition_requests, total_cycles)

    # Compute trend if 24h data available
    trend_direction: TrendDirection = "STABLE"
    if signals_24h:
        trans_24h = count_transition_requests(signals_24h)
        cycles_24h = count_total_cycles(signals_24h)
        ctrpk_24h = compute_ctrpk(trans_24h, cycles_24h)
        trend_direction = compute_ctrpk_trend(ctrpk_value, ctrpk_24h)

    # Build compact block
    compact = build_ctrpk_compact(
        transition_requests=transition_requests,
        total_cycles=total_cycles,
        trend_direction=trend_direction,
        semantic_violations=semantic_violations,
        blocked_requests=blocked_requests,
    )

    # Add metadata for traceability
    compact["computed_at"] = datetime.now(timezone.utc).isoformat()
    compact["signal_count"] = len(signals)
    if signals_24h:
        compact["signal_count_24h"] = len(signals_24h)

    return compact


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute CTRPK from curriculum runtime guard signals"
    )
    parser.add_argument(
        "--signals-jsonl",
        type=str,
        required=True,
        help="Path to runtime guard signals JSONL (recent window, typically 1h)",
    )
    parser.add_argument(
        "--signals-jsonl-24h",
        type=str,
        help="Optional path to 24h signals JSONL for trend computation",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for CTRPK compact JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    signals_path = Path(args.signals_jsonl)
    output_path = Path(args.output)

    # Load signals
    if not signals_path.exists():
        print(f"ERROR: Signals file not found: {signals_path}", file=sys.stderr)
        return 1

    signals = load_jsonl(signals_path)
    print(f"Loaded {len(signals)} signals from {signals_path}")

    # Load 24h signals if provided
    signals_24h = None
    if args.signals_jsonl_24h:
        signals_24h_path = Path(args.signals_jsonl_24h)
        if signals_24h_path.exists():
            signals_24h = load_jsonl(signals_24h_path)
            print(f"Loaded {len(signals_24h)} 24h signals from {signals_24h_path}")
        else:
            print(f"WARNING: 24h signals file not found: {signals_24h_path}", file=sys.stderr)

    # Compute CTRPK
    ctrpk_compact = compute_ctrpk_from_signals(signals, signals_24h)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ctrpk_compact, f, indent=indent)

    # Print summary
    print()
    print("=" * 60)
    print("CTRPK Computation Summary")
    print("=" * 60)
    print(f"  CTRPK Value:          {ctrpk_compact['value']:.2f}")
    print(f"  Status:               {ctrpk_compact['status']}")
    print(f"  Trend:                {ctrpk_compact['trend']}")
    print(f"  Window Cycles:        {ctrpk_compact['window_cycles']}")
    print(f"  Transition Requests:  {ctrpk_compact['transition_requests']}")
    print(f"  Signal Count:         {ctrpk_compact['signal_count']}")
    print()
    print(f"Output written to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
