#!/usr/bin/env python3
"""
Budget Invariants CI Check Script (Advisory-Only, Shadow Mode)

Reads a budget invariant timeline JSON, computes governance view and CI capsule,
prints a one-line neutral reason, and writes the CI capsule to artifacts.

This script is advisory-only (always exits with code 0). It does not block CI;
it provides observability for budget governance signals.

Budget Invariants = "Energy Law" of First-Light runs
Storyline + BNH-Φ = temporal coherence evidence
These appear in P3 stability reports and P4 calibration bundles

Usage:
    python scripts/budget_invariants_ci_check.py --timeline timeline.json [--artifacts artifacts/]

Exit Code:
    Always 0 (advisory-only, shadow mode)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from derivation.budget_invariants import (
        build_budget_invariants_governance_view,
        evaluate_budget_release_readiness,
        explain_budget_release_decision,
    )
except ImportError as e:
    print(f"ERROR: Failed to import budget invariants module: {e}", file=sys.stderr)
    print("       This script requires derivation.budget_invariants to be available", file=sys.stderr)
    sys.exit(0)  # Advisory-only: don't fail CI


def load_timeline(path: Path) -> Dict[str, Any]:
    """Load budget invariant timeline from JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Timeline file not found: {path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"WARNING: Invalid JSON in timeline file: {e}", file=sys.stderr)
        return {}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Budget Invariants CI Check (Advisory-Only, Shadow Mode)"
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        required=True,
        help="Path to budget invariant timeline JSON file",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write CI capsule artifact (default: artifacts/)",
    )
    parser.add_argument(
        "--budget-health",
        type=Path,
        help="Optional path to budget health JSON (for governance view)",
    )
    
    args = parser.parse_args()
    
    # Load timeline
    timeline = load_timeline(args.timeline)
    
    if not timeline:
        print("Budget invariants: status unavailable (no timeline data)")
        return 0  # Advisory-only: don't fail CI
    
    # Load budget health if provided (for governance view)
    budget_health: Dict[str, Any] = {
        "health_score": 100.0,
        "trend_status": "STABLE",
    }
    
    if args.budget_health and args.budget_health.exists():
        try:
            with open(args.budget_health, "r") as f:
                budget_health = json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load budget health: {e}", file=sys.stderr)
    
    # Build governance view
    try:
        governance_view = build_budget_invariants_governance_view(
            invariant_timeline=timeline,
            budget_health=budget_health,
        )
        
        # Evaluate release readiness
        readiness = evaluate_budget_release_readiness(governance_view)
        
        # Explain decision (CI capsule)
        ci_capsule = explain_budget_release_decision(governance_view, readiness)
        
        # Print one-line neutral reason (≤160 chars)
        summary = ci_capsule.get("ci_summary", "Budget invariants: status unknown")
        print(summary)
        
        # Write CI capsule to artifacts
        artifacts_dir = Path(args.artifacts)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        capsule_path = artifacts_dir / "budget_invariants_ci_capsule.json"
        try:
            with open(capsule_path, "w") as f:
                json.dump(ci_capsule, f, indent=2)
            print(f"CI capsule written to: {capsule_path}", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Failed to write CI capsule: {e}", file=sys.stderr)
        
        # Also write governance view for reference
        gov_path = artifacts_dir / "budget_invariants_governance_view.json"
        try:
            with open(gov_path, "w") as f:
                json.dump(governance_view, f, indent=2)
        except Exception:
            pass  # Non-critical
        
        return 0  # Advisory-only: always succeed
    
    except Exception as e:
        print(f"WARNING: Budget invariants check failed: {e}", file=sys.stderr)
        print("Budget invariants: status unavailable (check error)", file=sys.stderr)
        return 0  # Advisory-only: don't fail CI even on errors


if __name__ == "__main__":
    sys.exit(main())

