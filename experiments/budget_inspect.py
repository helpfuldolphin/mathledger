#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Budget Inspection CLI
═══════════════════════════════════════════════════════════════════════════════

Inspect Phase II verifier budget configuration for uplift slices.
Read-only tool for debugging and documentation.

Usage:
    # Inspect single slice
    uv run python experiments/budget_inspect.py --slice slice_uplift_goal
    
    # Inspect all Phase II slices
    uv run python experiments/budget_inspect.py --all
    
    # JSON output for scripts
    uv run python experiments/budget_inspect.py --all --json

Budget Parameters (from lean_control_sandbox_plan.md §14):
    
    cycle_budget_s:
        Wall-clock time limit per derivation cycle (seconds).
        When elapsed time exceeds this limit, remaining candidates are
        skipped with outcome "budget_skip". This bounds total cycle
        runtime to prevent runaway experiments.
        
    taut_timeout_s:
        Per-statement truth-table verification timeout (seconds).
        Tier 1 verification (is_tautology check) times out → abstain_timeout.
        Prevents infinite loops in pathological formulas.
        
    max_candidates_per_cycle:
        Maximum candidates to consider per cycle.
        Enforces deterministic bound on verification work.
        Makes candidate ordering matter for uplift measurement.

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.verification.budget_loader import (
    VerifierBudget,
    load_budget_for_slice,
    load_default_budget,
    is_phase2_slice,
    DEFAULT_CONFIG_PATH,
)

import yaml


# Parameter documentation from lean_control_sandbox_plan.md §14
# with quantitative interpretation guidance
PARAM_DOCS = {
    "cycle_budget_s": (
        "Wall-clock time limit per derivation cycle (seconds).\n"
        "When elapsed time exceeds this, remaining candidates receive 'budget_skip'.\n"
        "Bounds total cycle runtime to prevent runaway experiments.\n"
        "\n"
        "Quantitative guidance:\n"
        "  - Typical range: 4-10s for uplift slices\n"
        "  - If >5% cycles hit budget_exhausted → increase this value\n"
        "  - Measured via time.monotonic() (immune to clock skew)"
    ),
    "taut_timeout_s": (
        "Per-statement truth-table verification timeout (seconds).\n"
        "Tier 1 verification times out → 'abstain_timeout' outcome.\n"
        "Prevents infinite loops in pathological formulas.\n"
        "\n"
        "Quantitative guidance:\n"
        "  - Typical range: 0.05-0.20s\n"
        "  - Truth-table complexity is O(2^n) where n = atom count\n"
        "  - 2-atom: ~0.01s, 3-atom: ~0.02s, 4-atom: ~0.05s, 5-atom: ~0.12s\n"
        "  - If avg timeout_abstentions >1/cycle → increase this value"
    ),
    "max_candidates_per_cycle": (
        "Maximum candidates to consider per cycle.\n"
        "Enforces deterministic bound on verification work.\n"
        "Makes candidate ordering matter for uplift measurement.\n"
        "\n"
        "Quantitative guidance:\n"
        "  - Typical range: 30-50 for uplift experiments\n"
        "  - Lower values (20-30) → stronger ordering signal\n"
        "  - Higher values (50-100) → more exploration per cycle\n"
        "  - If <50% cycles hit max_candidates_hit → slice naturally small"
    ),
}

# Budget state explanations for --explain flag
BUDGET_STATE_DOCS = {
    "budget_exhausted": (
        "CONDITION: time.monotonic() - cycle_start > cycle_budget_s\n"
        "OUTCOME:   Remaining candidates skipped ('budget_skip')\n"
        "SEVERITY:  High (>5%) = budget too tight, cycle incomplete\n"
        "ACTION:    Increase cycle_budget_s or reduce slice complexity"
    ),
    "max_candidates_hit": (
        "CONDITION: candidates_considered >= max_candidates_per_cycle\n"
        "OUTCOME:   Additional candidates not explored\n"
        "SEVERITY:  Expected for RFL uplift (ordering should matter)\n"
        "ACTION:    No action needed if >90% cycles hit this"
    ),
    "timeout_abstentions": (
        "CONDITION: verification_time > taut_timeout_s (per candidate)\n"
        "OUTCOME:   Candidate outcome = 'abstain_timeout' (excluded)\n"
        "SEVERITY:  High (>1.0 avg/cycle) = timeout too tight\n"
        "ACTION:    Increase taut_timeout_s or constrain atom count"
    ),
}


def get_all_phase2_slices(config_path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    """Get all Phase II slice names from the budget config."""
    path = Path(config_path)
    if not path.exists():
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    slices = config.get("slices", {})
    return sorted(slices.keys())


def format_single_slice(slice_name: str, budget: VerifierBudget, verbose: bool = True) -> str:
    """Format budget info for a single slice."""
    lines = []
    lines.append(f"Slice: {slice_name}")
    lines.append("─" * 60)
    lines.append(f"  cycle_budget_s:          {budget.cycle_budget_s:>8.2f} s")
    lines.append(f"  taut_timeout_s:          {budget.taut_timeout_s:>8.2f} s")
    lines.append(f"  max_candidates_per_cycle:{budget.max_candidates_per_cycle:>8d}")
    
    if verbose:
        lines.append("")
        lines.append("Parameter Descriptions:")
        for param, doc in PARAM_DOCS.items():
            lines.append(f"\n  {param}:")
            for doc_line in doc.split("\n"):
                lines.append(f"    {doc_line}")
    
    return "\n".join(lines)


def format_table(budgets: Dict[str, VerifierBudget]) -> str:
    """Format budget info as a table."""
    lines = []
    
    # Header
    header = f"{'Slice':<30} {'cycle_budget_s':>15} {'taut_timeout_s':>15} {'max_candidates':>15}"
    lines.append(header)
    lines.append("─" * len(header))
    
    # Rows
    for slice_name in sorted(budgets.keys()):
        budget = budgets[slice_name]
        row = (
            f"{slice_name:<30} "
            f"{budget.cycle_budget_s:>15.2f} "
            f"{budget.taut_timeout_s:>15.2f} "
            f"{budget.max_candidates_per_cycle:>15d}"
        )
        lines.append(row)
    
    lines.append("─" * len(header))
    
    return "\n".join(lines)


def format_json(budgets: Dict[str, VerifierBudget]) -> str:
    """Format budget info as JSON."""
    output = {
        "phase": "PHASE II — NOT USED IN PHASE I",
        "config_path": DEFAULT_CONFIG_PATH,
        "slices": {}
    }
    
    for slice_name, budget in sorted(budgets.items()):
        output["slices"][slice_name] = {
            "cycle_budget_s": budget.cycle_budget_s,
            "taut_timeout_s": budget.taut_timeout_s,
            "max_candidates_per_cycle": budget.max_candidates_per_cycle,
        }
    
    output["parameter_docs"] = PARAM_DOCS
    
    return json.dumps(output, indent=2)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect Phase II verifier budget configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--slice",
        type=str,
        help="Name of the Phase II slice to inspect (e.g., 'slice_uplift_goal')."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Inspect all Phase II slices."
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to budget config file (default: {DEFAULT_CONFIG_PATH})."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include parameter descriptions."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show quantitative explanation of budget enforcement states."
    )
    
    args = parser.parse_args()
    
    print("PHASE II — NOT USED IN PHASE I", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Show budget state explanations if requested
    if args.explain:
        print("BUDGET ENFORCEMENT STATES (Quantitative)")
        print("═" * 60)
        for state_name, doc in BUDGET_STATE_DOCS.items():
            print(f"\n{state_name}:")
            for line in doc.split("\n"):
                print(f"  {line}")
        print("")
        print("─" * 60)
        print("")
    
    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Budget config not found: {config_path}", file=sys.stderr)
        print(f"Phase II budget enforcement requires: {DEFAULT_CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    
    if args.slice:
        # Single slice inspection
        try:
            budget = load_budget_for_slice(args.slice, args.config)
        except KeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            available = get_all_phase2_slices(args.config)
            print(f"Available slices: {available}", file=sys.stderr)
            sys.exit(1)
        
        if args.json:
            print(format_json({args.slice: budget}))
        else:
            print(format_single_slice(args.slice, budget, verbose=args.verbose))
    
    else:
        # All slices
        slice_names = get_all_phase2_slices(args.config)
        if not slice_names:
            print("No Phase II slices found in config.", file=sys.stderr)
            sys.exit(1)
        
        budgets = {}
        for name in slice_names:
            try:
                budgets[name] = load_budget_for_slice(name, args.config)
            except Exception as e:
                print(f"WARNING: Failed to load budget for {name}: {e}", file=sys.stderr)
        
        if args.json:
            print(format_json(budgets))
        else:
            print(f"Phase II Budget Configuration ({len(budgets)} slices)")
            print(f"Config: {args.config}")
            print("")
            print(format_table(budgets))
            
            if args.verbose:
                print("\nParameter Descriptions:")
                for param, doc in PARAM_DOCS.items():
                    print(f"\n  {param}:")
                    for doc_line in doc.split("\n"):
                        print(f"    {doc_line}")


if __name__ == "__main__":
    main()

