#!/usr/bin/env python3
"""
Emit Budget Confounding Truth Table (Deterministic Output).

Emits the canonical truth table for budget confounding semantics as JSON.
This provides an auditor-proof reference for confounding logic in both
strict (AND-rule) and legacy (OR-rule) modes.

SHADOW MODE CONTRACT:
- This script is observational only
- Output is for reference and validation
- No gating decisions based on this output
- Phase X/P5 POC only

Usage:
    python scripts/emit_budget_confounding_truth_table.py
    python scripts/emit_budget_confounding_truth_table.py --threshold 0.90
    python scripts/emit_budget_confounding_truth_table.py --json
"""

import argparse
import json
import sys

from derivation.budget_cal_exp_integration import build_budget_confounding_truth_table


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Emit Budget Confounding Truth Table (Deterministic Output)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confound stability threshold (default: 0.95)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only (no header, suitable for piping)",
    )
    
    args = parser.parse_args()
    
    # Build truth table
    table = build_budget_confounding_truth_table(
        confound_stability_threshold=args.threshold
    )
    
    # Output JSON
    if args.json:
        # JSON only, no header
        print(json.dumps(table, indent=2))
    else:
        # Human-readable output with header
        print("# Budget Confounding Truth Table", file=sys.stderr)
        print(f"# Threshold: {args.threshold}", file=sys.stderr)
        print(json.dumps(table, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

