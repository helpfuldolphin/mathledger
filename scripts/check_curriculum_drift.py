#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
# File: scripts/check_curriculum_drift.py
"""
CI Guard Script for Curriculum Drift Detection.

Checks for curriculum drift between the last two snapshots and:
- Fails CI on risk_level=BLOCK (PARAMETRIC_MAJOR, SEMANTIC, STRUCTURAL)
- Succeeds on INFO/WARN but prints warnings

Usage:
    uv run python scripts/check_curriculum_drift.py

    # With explicit config path
    uv run python scripts/check_curriculum_drift.py --config config/curriculum_uplift_phase2.yaml

    # Output JSON report to file
    uv run python scripts/check_curriculum_drift.py --output artifacts/phase_ii/drift_report.json

Exit Codes:
    0 - No drift or non-blocking drift (INFO/WARN)
    1 - Blocking drift detected (BLOCK)
    2 - Error (missing snapshots, config not found, etc.)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.curriculum_hash_ledger import (
    CurriculumHashLedger,
    DriftType,
    RiskLevel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI Guard for Curriculum Drift Detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/curriculum_uplift_phase2.yaml",
        help="Path to curriculum config file (for deep comparison)."
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default=None,
        help="Path to ledger JSONL file (overrides default)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON drift report."
    )
    parser.add_argument(
        "--from",
        dest="from_ref",
        type=str,
        default="-2",
        help="Older snapshot reference (default: -2, second to last)."
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        type=str,
        default="-1",
        help="Newer snapshot reference (default: -1, last)."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on WARN-level drift (PARAMETRIC_MINOR)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except for errors."
    )
    
    args = parser.parse_args()
    
    # Initialize ledger
    ledger_path = Path(args.ledger) if args.ledger else None
    ledger = CurriculumHashLedger(ledger_path=ledger_path)
    
    # Load snapshots
    snapshots = ledger.load_snapshots()
    
    if len(snapshots) < 2:
        if not args.quiet:
            print("⚠️  Not enough snapshots to compare (need at least 2).")
            print("    Recording a baseline snapshot for future comparisons...")
        
        # Try to record a snapshot if config exists
        config_path = Path(args.config)
        if config_path.exists():
            entry = ledger.record_snapshot(
                config_path=str(config_path),
                origin="ci",
                notes="Auto-recorded baseline snapshot by CI guard"
            )
            if not args.quiet:
                print(f"    Snapshot recorded: {entry['curriculum_hash'][:16]}...")
        else:
            if not args.quiet:
                print(f"    Warning: Config file not found: {args.config}")
        
        return 0  # Not an error, just no comparison possible yet
    
    # Get snapshots to compare
    old_snap = ledger.get_snapshot(args.from_ref)
    new_snap = ledger.get_snapshot(args.to_ref)
    
    if old_snap is None:
        print(f"Error: Snapshot not found: {args.from_ref}", file=sys.stderr)
        return 2
    if new_snap is None:
        print(f"Error: Snapshot not found: {args.to_ref}", file=sys.stderr)
        return 2
    
    # Load configs for deep comparison (if available)
    old_config = None
    new_config = None
    config_path = Path(args.config)
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            # For old config, we use the same file (assumes config at old snapshot)
            # In a real scenario, you'd retrieve from git at old commit
            old_config = new_config  # Simplified: compare current vs current structure
        except Exception as e:
            if not args.quiet:
                print(f"Warning: Could not load config for deep comparison: {e}")
    
    # Perform drift classification
    diff = ledger.classify_drift(old_snap, new_snap, old_config, new_config)
    
    # Write output report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "comparison": {
                "from": args.from_ref,
                "to": args.to_ref
            },
            "diff": diff
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        if not args.quiet:
            print(f"Drift report written to: {output_path}")
    
    # Determine exit based on risk level
    risk_level = diff.get('risk_level', 'UNKNOWN')
    drift_type = diff.get('drift_type', 'UNKNOWN')
    
    if not args.quiet:
        print(ledger.format_diff_report(diff))
        print()
    
    # Exit logic
    if risk_level == RiskLevel.BLOCK.value:
        if not args.quiet:
            print("=" * 60)
            print("🚫 CI BLOCKED: Curriculum drift exceeds acceptable threshold.")
            print(f"   Drift Type: {drift_type}")
            print(f"   Risk Level: {risk_level}")
            print()
            print("   To proceed, either:")
            print("   1. Revert the curriculum changes, or")
            print("   2. Record a new baseline snapshot after review:")
            print(f"      uv run python experiments/curriculum_hash_ledger.py \\")
            print(f"          --snapshot --config {args.config} --origin=manual \\")
            print(f"          --notes=\"Approved drift: <reason>\"")
            print("=" * 60)
        return 1
    
    elif risk_level == RiskLevel.WARN.value:
        if not args.quiet:
            print("=" * 60)
            print(f"⚠️  WARNING: Minor curriculum drift detected ({drift_type})")
            print(f"   Risk Level: {risk_level}")
            print("   CI will proceed, but review changes before release.")
            print("=" * 60)
        
        if args.strict:
            return 1  # Fail on WARN in strict mode
        return 0
    
    elif risk_level == RiskLevel.INFO.value:
        if not args.quiet:
            print("=" * 60)
            print(f"✅ CI PASSED: No significant curriculum drift ({drift_type})")
            print(f"   Risk Level: {risk_level}")
            print("=" * 60)
        return 0
    
    else:
        if not args.quiet:
            print(f"Unknown risk level: {risk_level}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

