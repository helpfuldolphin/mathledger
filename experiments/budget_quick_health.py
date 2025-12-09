#!/usr/bin/env python3
"""
Phase III Budget Quick Health — One-Command Diagnostic + Snapshot JSON
═══════════════════════════════════════════════════════════════════════

A fast, human-friendly budget health snapshot for developers.

Usage:
    uv run python experiments/budget_quick_health.py           # Human-readable output
    uv run python experiments/budget_quick_health.py --snapshot-json  # JSON snapshot only

Exit Codes:
    0 = All budget invariants satisfied (summary_status == "OK")
    1 = Invariant failure or exception (summary_status in ["WARN", "FAIL"])

Modes:
    Default: Human-readable output with budget summary and invariant status
    --snapshot-json: Outputs only the invariant snapshot JSON (stable schema)

This tool:
    1. Runs a short synthetic derivation (micro slice, tiny budget)
    2. Collects PipelineStats
    3. Builds invariant snapshot (Phase III contract)
    4. Prints summarize_budget() or JSON snapshot
    5. Returns exit code based on summary_status

Runtime: <1 second on typical hardware.
Phase: III (budget invariant governance layer)

Author: Agent B1 (verifier-ops-1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Quick health configuration
MICRO_BUDGET_CYCLE_S = 0.05      # 50ms cycle budget
MICRO_TAUT_TIMEOUT_S = 0.01     # 10ms per-candidate
MICRO_MAX_CANDIDATES = 10        # Small candidate cap


def run_quick_health(snapshot_json: bool = False) -> int:
    """
    Run a quick budget health check.
    
    Args:
        snapshot_json: If True, output only JSON snapshot and exit.
    
    Returns:
        0 if summary_status == "OK", 1 otherwise.
    """
    if not snapshot_json:
        print("=" * 60)
        print("Phase III Budget Quick Health Check")
        print("=" * 60)
        print()
    
    start = time.perf_counter()
    
    try:
        # Import pipeline components
        from derivation.pipeline import (
            run_slice_for_test,
            summarize_budget,
            PipelineStats,
        )
        from derivation.bounds import SliceBounds
        from curriculum.gates import (
            CurriculumSlice,
            SliceGates,
            CoverageGateSpec,
            AbstentionGateSpec,
            VelocityGateSpec,
            CapsGateSpec,
        )
        from backend.verification.budget_loader import VerifierBudget
        from derivation.budget_invariants import build_budget_invariant_snapshot
        
    except ImportError as e:
        if snapshot_json:
            print(json.dumps({"error": str(e), "summary_status": "FAIL"}))
        else:
            print(f"✗ Import error: {e}", file=sys.stderr)
        return 1
    
    # Create micro slice for fast execution
    test_gates = SliceGates(
        coverage=CoverageGateSpec(ci_lower_min=0.5, sample_min=10),
        abstention=AbstentionGateSpec(max_rate_pct=25.0, max_mass=100),
        velocity=VelocityGateSpec(min_pph=1.0, stability_cv_max=0.5, window_minutes=5),
        caps=CapsGateSpec(min_attempt_mass=5, min_runtime_minutes=0.1, backlog_max=1.0),
    )
    
    micro_slice = CurriculumSlice(
        name="quick_health_micro",
        params={
            "atoms": 2,
            "depth_max": 2,
            "mp_depth": 1,
            "breadth_max": 5,
            "total_max": 5,
        },
        gates=test_gates,
        metadata={"quick_health": True},
    )
    
    # Create micro budget
    budget = VerifierBudget(
        cycle_budget_s=MICRO_BUDGET_CYCLE_S,
        taut_timeout_s=MICRO_TAUT_TIMEOUT_S,
        max_candidates_per_cycle=MICRO_MAX_CANDIDATES,
    )
    
    if not snapshot_json:
        print(f"Config: cycle_budget={MICRO_BUDGET_CYCLE_S}s, "
              f"taut_timeout={MICRO_TAUT_TIMEOUT_S}s, "
              f"max_candidates={MICRO_MAX_CANDIDATES}")
        print()
    
    # Run derivation
    try:
        result = run_slice_for_test(
            micro_slice,
            limit=1,
            budget=budget,
            emit_log=False,
        )
    except Exception as e:
        if snapshot_json:
            print(json.dumps({"error": str(e), "summary_status": "FAIL"}))
        else:
            print(f"✗ Derivation failed: {e}", file=sys.stderr)
        return 1
    
    stats = result.stats
    summary_dict = result.summary.to_dict()
    budget_section = summary_dict.get("budget", {})
    
    # Build invariant snapshot using Phase III contract
    snapshot = build_budget_invariant_snapshot(
        stats=stats,
        max_candidates_limit=MICRO_MAX_CANDIDATES,
        budget_section=budget_section,
    )
    
    # --snapshot-json mode: output JSON and exit
    if snapshot_json:
        print(json.dumps(snapshot, indent=2))
        return 0 if snapshot["summary_status"] == "OK" else 1
    
    # Human-readable mode
    print("BUDGET SUMMARY")
    print("-" * 60)
    print(summarize_budget(stats))
    print("-" * 60)
    print()
    
    # Print invariant results from snapshot
    print("INVARIANT STATUS")
    print("-" * 60)
    
    inv_checks = [
        ("INV-BUD-1", snapshot["inv_bud_1_ok"], f"post_exhaustion_candidates={stats.post_exhaustion_candidates}"),
        ("INV-BUD-2", snapshot["inv_bud_2_ok"], f"candidates={stats.candidates_considered}, max_hit={stats.max_candidates_hit}"),
        ("INV-BUD-3", snapshot["inv_bud_3_ok"], f"remaining={stats.budget_remaining_s:.4f}s"),
        ("INV-BUD-4", snapshot["inv_bud_4_ok"], f"all budget fields present"),
        ("INV-BUD-5", snapshot["inv_bud_5_ok"], f"to_dict() deterministic"),
    ]
    
    for inv_id, ok, detail in inv_checks:
        marker = "✓" if ok else "✗"
        status = "PASS" if ok else "FAIL"
        print(f"  {marker} {inv_id}: {status} — {detail}")
    
    print("-" * 60)
    print()
    
    # Final verdict
    elapsed_ms = (time.perf_counter() - start) * 1000
    summary_status = snapshot["summary_status"]
    
    if summary_status == "OK":
        print(f"✓ Budget invariants: PASS ({elapsed_ms:.1f}ms)")
        return 0
    elif summary_status == "WARN":
        print(f"⚠ Budget invariants: WARN ({elapsed_ms:.1f}ms)")
        print(f"  timeout_abstentions={snapshot['timeout_abstentions']}")
        return 1
    else:
        print(f"✗ Budget invariants: FAIL ({elapsed_ms:.1f}ms)")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase III Budget Quick Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0 = summary_status == "OK" (all invariants pass)
  1 = summary_status in ["WARN", "FAIL"] or error

Examples:
  uv run python experiments/budget_quick_health.py
  uv run python experiments/budget_quick_health.py --snapshot-json
  uv run python experiments/budget_quick_health.py --snapshot-json | jq .summary_status
""",
    )
    parser.add_argument(
        "--snapshot-json",
        action="store_true",
        help="Output only JSON snapshot (stable schema) and exit",
    )
    args = parser.parse_args()
    
    # Enable debug budget assertions
    if os.getenv("MATHLEDGER_DEBUG_BUDGET") is None:
        os.environ["MATHLEDGER_DEBUG_BUDGET"] = "1"
    
    try:
        return run_quick_health(snapshot_json=args.snapshot_json)
    except Exception as e:
        if args.snapshot_json:
            print(json.dumps({"error": str(e), "summary_status": "FAIL"}))
        else:
            print(f"✗ Script error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

