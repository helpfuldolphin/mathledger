#!/usr/bin/env python3
"""
PRNG Drift Ledger Comparison Tool (CAL-EXP-1/2)

Compares PRNG drift ledgers across P3 mock, P4 mock, and P5 real runs to identify
drift patterns and calibration behavior during warm-start experiments.

This tool generates a JSON report showing:
- Delta counts (volatile_runs, drifting_runs, stable_runs) between phases
- Frequent rules comparison across phases
- Overall drift status transitions
- Deterministic ordering for reproducible reports

Expected PRNG Behavior During Warm-Start Calibration:

During CAL-EXP-1 and CAL-EXP-2 warm-start calibration experiments:

1. **P3 Mock Phase (Synthetic)**:
   - PRNG governance should be STABLE or DRIFTING with minimal violations
   - Synthetic telemetry provides controlled conditions for baseline establishment
   - Expected: 0-10% volatile_runs, 0-20% drifting_runs, 70-100% stable_runs
   - Frequent rules should be minimal or absent

2. **P4 Mock Phase (Shadow Mock)**:
   - PRNG governance should remain similar to P3, as mock telemetry should preserve
     deterministic behavior
   - Small increases in drift are acceptable if they reflect realistic mock conditions
   - Expected: P3→P4 delta should show minimal change (volatile_runs_delta ≤ 2,
     drifting_runs_delta ≤ 3)
   - If P4 shows significant drift increase, investigate mock telemetry fidelity

3. **P5 Real Phase (Real Telemetry)**:
   - PRNG governance may show increased drift compared to P3/P4, as real telemetry
     may introduce:
     * New PRNG usage patterns not present in synthetic/mock runs
     * Environmental factors affecting seed derivation
     * Runtime conditions that trigger different code paths
   - Expected: P4→P5 delta may show moderate increase (volatile_runs_delta ≤ 5,
     drifting_runs_delta ≤ 10)
   - If P5 shows excessive drift (volatile_runs > 30% of total_runs), investigate
     real telemetry PRNG usage patterns

Interpretation Guidelines:
- STABLE → STABLE: Expected baseline behavior, no action needed
- STABLE → DRIFTING: Acceptable during calibration, monitor for trends
- DRIFTING → VOLATILE: Investigate root cause, may indicate calibration issues
- Any phase showing >30% volatile_runs: Requires investigation regardless of transitions

Usage:
    python scripts/prng_compare_runs.py \
        --p3-ledger artifacts/p3_prng_drift_ledger.json \
        --p4-ledger artifacts/p4_prng_drift_ledger.json \
        --p5-ledger artifacts/p5_prng_drift_ledger.json \
        --output artifacts/prng_drift_delta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from rfl.prng.governance import DriftStatus


def load_ledger(path: Path) -> Dict[str, Any]:
    """Load a PRNG drift ledger from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Validate schema
    required_fields = ["schema_version", "total_runs", "volatile_runs", "drifting_runs", "stable_runs", "frequent_rules"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in ledger: {path}")
    
    return data


def compute_drift_delta(
    baseline: Dict[str, Any],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute delta between two PRNG drift ledgers.
    
    Returns a dict with:
    - total_runs_delta: difference in total_runs
    - volatile_runs_delta: difference in volatile_runs
    - drifting_runs_delta: difference in drifting_runs
    - stable_runs_delta: difference in stable_runs
    - frequent_rules_delta: dict of rule_id -> count difference
    """
    return {
        "total_runs_delta": comparison["total_runs"] - baseline["total_runs"],
        "volatile_runs_delta": comparison["volatile_runs"] - baseline["volatile_runs"],
        "drifting_runs_delta": comparison["drifting_runs"] - baseline["drifting_runs"],
        "stable_runs_delta": comparison["stable_runs"] - baseline["stable_runs"],
        "frequent_rules_delta": _compute_rules_delta(
            baseline.get("frequent_rules", {}),
            comparison.get("frequent_rules", {}),
        ),
    }


def _compute_rules_delta(
    baseline_rules: Dict[str, int],
    comparison_rules: Dict[str, int],
) -> Dict[str, int]:
    """Compute delta for frequent_rules, ensuring deterministic ordering."""
    all_rule_ids = sorted(set(baseline_rules.keys()) | set(comparison_rules.keys()))
    delta = {}
    for rule_id in all_rule_ids:
        baseline_count = baseline_rules.get(rule_id, 0)
        comparison_count = comparison_rules.get(rule_id, 0)
        diff = comparison_count - baseline_count
        if diff != 0:  # Only include non-zero deltas
            delta[rule_id] = diff
    
    return dict(sorted(delta.items()))


def classify_drift_status(runs: Dict[str, Any]) -> str:
    """
    Classify overall drift status from ledger counts.
    
    Uses same logic as build_prng_drift_radar:
    - VOLATILE: ≥30% BLOCK runs or ≥3 frequent violations
    - DRIFTING: 1-2 frequent violations, <30% BLOCK
    - STABLE: no frequent violations, <10% BLOCK
    """
    total = runs["total_runs"]
    if total == 0:
        return DriftStatus.STABLE.value
    
    volatile_pct = runs["volatile_runs"] / total
    drifting_pct = runs["drifting_runs"] / total
    
    # VOLATILE if ≥30% volatile runs
    if volatile_pct >= 0.30:
        return DriftStatus.VOLATILE.value
    
    # DRIFTING if any drifting runs (but <30% volatile)
    if runs["drifting_runs"] > 0:
        return DriftStatus.DRIFTING.value
    
    # STABLE otherwise
    return DriftStatus.STABLE.value


def build_comparison_report(
    p3_ledger: Optional[Dict[str, Any]],
    p4_ledger: Optional[Dict[str, Any]],
    p5_ledger: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build comprehensive PRNG drift delta report.
    
    Args:
        p3_ledger: P3 mock run ledger (optional)
        p4_ledger: P4 mock run ledger (optional)
        p5_ledger: P5 real run ledger (optional)
    
    Returns:
        JSON-serializable report with:
        - schema_version
        - phase_ledgers: individual ledger snapshots
        - deltas: P3→P4, P4→P5, P3→P5 comparisons
        - drift_status_transitions: status changes across phases
        - summary: human-readable summary
    """
    report = {
        "schema_version": "1.0.0",
        "phase_ledgers": {},
        "deltas": {},
        "drift_status_transitions": [],
        "summary": {},
    }
    
    # Store individual ledgers (deterministic ordering)
    if p3_ledger:
        report["phase_ledgers"]["p3_mock"] = p3_ledger
    if p4_ledger:
        report["phase_ledgers"]["p4_mock"] = p4_ledger
    if p5_ledger:
        report["phase_ledgers"]["p5_real"] = p5_ledger
    
    # Compute deltas
    if p3_ledger and p4_ledger:
        report["deltas"]["p3_to_p4"] = compute_drift_delta(p3_ledger, p4_ledger)
    
    if p4_ledger and p5_ledger:
        report["deltas"]["p4_to_p5"] = compute_drift_delta(p4_ledger, p5_ledger)
    
    if p3_ledger and p5_ledger:
        report["deltas"]["p3_to_p5"] = compute_drift_delta(p3_ledger, p5_ledger)
    
    # Track drift status transitions
    statuses = {}
    if p3_ledger:
        statuses["p3_mock"] = classify_drift_status(p3_ledger)
    if p4_ledger:
        statuses["p4_mock"] = classify_drift_status(p4_ledger)
    if p5_ledger:
        statuses["p5_real"] = classify_drift_status(p5_ledger)
    
    # Build transition list (deterministic ordering)
    phases = ["p3_mock", "p4_mock", "p5_real"]
    for i in range(len(phases) - 1):
        phase_a = phases[i]
        phase_b = phases[i + 1]
        if phase_a in statuses and phase_b in statuses:
            report["drift_status_transitions"].append({
                "from_phase": phase_a,
                "to_phase": phase_b,
                "from_status": statuses[phase_a],
                "to_status": statuses[phase_b],
                "status_changed": statuses[phase_a] != statuses[phase_b],
            })
    
    # Build summary
    report["summary"] = {
        "phases_analyzed": list(report["phase_ledgers"].keys()),
        "total_transitions": len(report["drift_status_transitions"]),
        "status_changes": sum(1 for t in report["drift_status_transitions"] if t["status_changed"]),
        "current_status": statuses.get("p5_real") or statuses.get("p4_mock") or statuses.get("p3_mock"),
    }
    
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PRNG drift ledgers across P3/P4/P5 runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--p3-ledger",
        type=Path,
        help="Path to P3 mock run PRNG drift ledger JSON",
    )
    
    parser.add_argument(
        "--p4-ledger",
        type=Path,
        help="Path to P4 mock run PRNG drift ledger JSON",
    )
    
    parser.add_argument(
        "--p5-ledger",
        type=Path,
        help="Path to P5 real run PRNG drift ledger JSON",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for PRNG drift delta report JSON",
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Load ledgers (optional)
    p3_ledger = None
    p4_ledger = None
    p5_ledger = None
    
    if args.p3_ledger:
        if not args.p3_ledger.exists():
            print(f"Error: P3 ledger not found: {args.p3_ledger}", file=sys.stderr)
            return 1
        p3_ledger = load_ledger(args.p3_ledger)
    
    if args.p4_ledger:
        if not args.p4_ledger.exists():
            print(f"Error: P4 ledger not found: {args.p4_ledger}", file=sys.stderr)
            return 1
        p4_ledger = load_ledger(args.p4_ledger)
    
    if args.p5_ledger:
        if not args.p5_ledger.exists():
            print(f"Error: P5 ledger not found: {args.p5_ledger}", file=sys.stderr)
            return 1
        p5_ledger = load_ledger(args.p5_ledger)
    
    # Require at least one ledger
    if not any([p3_ledger, p4_ledger, p5_ledger]):
        print("Error: At least one ledger (--p3-ledger, --p4-ledger, or --p5-ledger) must be provided", file=sys.stderr)
        return 1
    
    # Build comparison report
    report = build_comparison_report(p3_ledger, p4_ledger, p5_ledger)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(report, f, indent=2, sort_keys=True)
        else:
            json.dump(report, f, sort_keys=True)
    
    print(f"PRNG drift delta report written to: {args.output}")
    print(f"Phases analyzed: {', '.join(report['summary']['phases_analyzed'])}")
    print(f"Current status: {report['summary']['current_status']}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

