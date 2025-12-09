#!/usr/bin/env python3
"""
Phase Migration Orchestrator — Cross-Agent Migration Script Skeleton

Loads phase migration contract and emits a machine-readable execution plan
that other agents (A/B/C/D/E groups, Manus) can consume.

Author: Agent E4 (doc-ops-4) — Phase Migration Architect
Date: 2025-12-06

ABSOLUTE SAFEGUARDS:
- Read-only — no mutations to production state
- No execution — just emits a plan structure
- Deterministic output — same contract → same plan
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.phase_migration_check import (
    build_phase_migration_contract,
    build_migration_governance_snapshot,
    build_phase_impact_map,
    build_migration_posture,
    build_summary_result,
    build_author_check_result,
    run_migration_check,
)


def build_migration_execution_plan(
    migration_contract: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a machine-readable execution plan from migration contract.
    
    This plan can be consumed by other agents to understand what checks
    need to be run and what phases are involved.
    
    Args:
        migration_contract: The contract from build_phase_migration_contract
        
    Returns:
        Dictionary with execution plan structure
    """
    expected_checks = migration_contract.get("expected_downstream_checks", [])
    
    # Build checks list with PENDING status
    checks = [
        {"id": check_id, "status": "PENDING"}
        for check_id in expected_checks
    ]
    
    return {
        "phases_involved": migration_contract.get("phases_involved", []),
        "strict_mode_required": migration_contract.get("strict_mode_required", False),
        "checks": checks,
    }


def main():
    """Main entry point for phase migration orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase Migration Orchestrator — Cross-Agent Migration Plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script loads the phase migration contract and emits a machine-readable
execution plan that other agents can consume.

The plan includes:
- phases_involved: List of phases affected by the migration
- strict_mode_required: Whether strict mode enforcement is needed
- checks: List of downstream checks that need to be run (all PENDING initially)

Example:
  python scripts/phase_migration_orchestrator.py --base main --head HEAD
        """
    )
    parser.add_argument(
        "--base", "-b",
        default="main",
        help="Base git ref (default: main)",
    )
    parser.add_argument(
        "--head", "-H",
        default="HEAD",
        help="Head git ref (default: HEAD)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Run migration check to get all data
    result = run_migration_check(
        base_ref=args.base,
        head_ref=args.head,
        project_root=args.project_root,
        output_dir=None,
        verbose=False,
    )
    
    # Build contract
    summary_result = build_summary_result(result)
    author_result = build_author_check_result(result)
    impact_map = build_phase_impact_map(summary_result)
    posture = build_migration_posture(summary_result, author_result)
    governance = build_migration_governance_snapshot(impact_map, posture)
    contract = build_phase_migration_contract(governance)
    
    # Build execution plan
    plan = build_migration_execution_plan(contract)
    
    # Output
    output_json = json.dumps(plan, indent=2)
    
    if args.output:
        args.output.write_text(output_json)
        print(f"Execution plan written to: {args.output}")
    else:
        print(output_json)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

