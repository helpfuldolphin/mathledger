#!/usr/bin/env python3
"""
PRNG Drift Monitor (Shadow Only).

Reads a directory of PRNG governance run summaries, computes drift radar + tile,
and writes prng_drift.json. Prints WARN/BLOCK conditions but always exits 0.

This is a SHADOW mode script: it does not block CI, only provides observability.

Usage:
    python scripts/prng_drift_monitor.py --input-dir artifacts/prng_runs --output artifacts/prng_drift.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rfl.prng.governance import (
    build_prng_drift_radar,
    build_prng_governance_history,
    build_prng_governance_tile,
    DriftStatus,
    GovernanceStatus,
    PRNGGovernanceSnapshot,
    PolicyEvaluation,
)


def load_run_summaries(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Load PRNG governance run summaries from a directory.

    Expected format: Each JSON file should contain a run summary with:
    - governance_status: "OK" | "WARN" | "BLOCK"
    - violations: List of violation dicts with rule_id, severity, etc.

    Args:
        input_dir: Directory containing JSON run summary files.

    Returns:
        List of run summary dicts.
    """
    summaries = []
    
    if not input_dir.exists():
        print(f"WARNING: Input directory does not exist: {input_dir}", file=sys.stderr)
        return summaries
    
    for json_file in sorted(input_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
                summaries.append(summary)
        except Exception as e:
            print(f"WARNING: Failed to load {json_file}: {e}", file=sys.stderr)
            continue
    
    return summaries


def convert_summaries_to_history(
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convert run summaries to PRNG governance history format.

    Args:
        summaries: List of run summary dicts.

    Returns:
        PRNG governance history dict.
    """
    if not summaries:
        return {
            "schema_version": "1.0",
            "total_runs": 0,
            "runs": [],
            "status_counts": {"OK": 0, "WARN": 0, "BLOCK": 0},
            "history_hash": "",
        }
    
    # Convert summaries to snapshots and evaluations
    snapshots = []
    policy_evals = []
    run_ids = []
    
    for i, summary in enumerate(summaries):
        run_id = summary.get("run_id", f"run_{i:04d}")
        run_ids.append(run_id)
        
        # Extract governance status
        status_str = summary.get("governance_status", "OK")
        try:
            from rfl.prng.governance import GovernanceStatus, ManifestStatus
            governance_status = GovernanceStatus(status_str)
        except ValueError:
            governance_status = GovernanceStatus.OK
        
        # Extract manifest status
        manifest_status_str = summary.get("manifest_status", "EQUIVALENT")
        try:
            manifest_status = ManifestStatus(manifest_status_str)
        except ValueError:
            manifest_status = ManifestStatus.EQUIVALENT
        
        # Extract namespace issues
        namespace_issues_dict = summary.get("namespace_issues", {})
        from rfl.prng.governance import NamespaceIssues
        namespace_issues = NamespaceIssues(
            duplicate_count=namespace_issues_dict.get("duplicate_count", 0),
            duplicate_files=namespace_issues_dict.get("duplicate_files", []),
            hardcoded_seed_count=namespace_issues_dict.get("hardcoded_seed_count", 0),
            hardcoded_seed_files=namespace_issues_dict.get("hardcoded_seed_files", []),
            dynamic_path_count=namespace_issues_dict.get("dynamic_path_count", 0),
            suppressed_count=namespace_issues_dict.get("suppressed_count", 0),
        )
        
        # Build snapshot (minimal required fields)
        snapshot = PRNGGovernanceSnapshot(
            governance_status=governance_status,
            manifest_status=manifest_status,
            namespace_issues=namespace_issues,
        )
        snapshots.append(snapshot)
        
        # Build policy evaluation from violations
        violations = summary.get("violations", [])
        policy_violations = []
        for v in violations:
            from rfl.prng.governance import PolicyViolation, GovernanceStatus as GS
            try:
                severity = GS(v.get("severity", "WARN"))
            except ValueError:
                severity = GS.WARN
            
            policy_violations.append(
                PolicyViolation(
                    rule_id=v.get("rule_id", "UNKNOWN"),
                    rule_name=v.get("kind", v.get("rule_name", "unknown")),
                    severity=severity,
                    message=v.get("message", ""),
                    context=v.get("context", {}),
                )
            )
        
        policy_eval = PolicyEvaluation(
            violations=policy_violations,
            status=governance_status,
            policy_ok=governance_status == GovernanceStatus.OK,
        )
        policy_evals.append(policy_eval)
    
    # Build history
    history = build_prng_governance_history(
        snapshots=snapshots,
        run_ids=run_ids,
        policy_evaluations=policy_evals,
    )
    
    return history


def print_warn_block_conditions(radar: Dict[str, Any], tile: Dict[str, Any]) -> None:
    """
    Print WARN/BLOCK conditions to stdout.

    Args:
        radar: PRNG drift radar.
        tile: PRNG governance tile.
    """
    drift_status = radar.get("drift_status", DriftStatus.STABLE.value)
    status = tile.get("status", GovernanceStatus.OK.value)
    frequent_violations = radar.get("frequent_violations", {})
    blocking_rules = tile.get("blocking_rules", [])
    
    if drift_status == DriftStatus.VOLATILE.value:
        print(f"PRNG DRIFT: VOLATILE - {len(frequent_violations)} frequent violation(s)")
        for rule_id, count in sorted(frequent_violations.items()):
            print(f"  - {rule_id}: {count} occurrence(s)")
    
    elif drift_status == DriftStatus.DRIFTING.value:
        print(f"PRNG DRIFT: DRIFTING - {len(frequent_violations)} frequent violation(s)")
        for rule_id, count in sorted(frequent_violations.items()):
            print(f"  - {rule_id}: {count} occurrence(s)")
    
    if status == GovernanceStatus.BLOCK.value:
        print(f"PRNG BLOCK: {len(blocking_rules)} blocking rule(s) detected")
        for rule_id in sorted(blocking_rules):
            print(f"  - {rule_id}")
    
    elif status == GovernanceStatus.WARN.value:
        print(f"PRNG WARN: Drift detected but no blocking rules")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PRNG Drift Monitor (Shadow Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PRNG governance run summary JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for prng_drift.json",
    )
    
    args = parser.parse_args()
    
    # Load run summaries
    summaries = load_run_summaries(args.input_dir)
    
    if not summaries:
        print("INFO: No run summaries found. Creating empty drift report.", file=sys.stderr)
        radar = {
            "schema_version": "1.0.0",
            "drift_status": DriftStatus.STABLE.value,
            "frequent_violations": {},
            "total_runs": 0,
        }
        tile = {
            "schema_version": "1.0.0",
            "status": GovernanceStatus.OK.value,
            "drift_status": DriftStatus.STABLE.value,
            "blocking_rules": [],
            "headline": "PRNG governance: no history available",
        }
    else:
        # Convert to history format
        history = convert_summaries_to_history(summaries)
        
        # Build radar and tile
        radar = build_prng_drift_radar(history)
        tile = build_prng_governance_tile(history, radar=radar)
    
    # Print WARN/BLOCK conditions
    print_warn_block_conditions(radar, tile)
    
    # Write output
    output_data = {
        "schema_version": "1.0.0",
        "radar": radar,
        "tile": tile,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)
    
    print(f"INFO: Wrote PRNG drift report to {args.output}", file=sys.stderr)
    
    # Always exit 0 (shadow mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())

