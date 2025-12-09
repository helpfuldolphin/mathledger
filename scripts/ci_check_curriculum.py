#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
# File: scripts/ci_check_curriculum.py
"""
CI Gate Wrapper for Curriculum Change Control.

Comprehensive CI integration that:
1. Loads the last recorded snapshot
2. Computes current curriculum state
3. Classifies drift between them
4. Validates against drift contract
5. Emits machine-readable JSON report
6. Optionally generates migration guide
7. Provides per-slice drift timeline views

Usage:
    # Standard CI check
    uv run python scripts/ci_check_curriculum.py
    
    # With explicit config
    uv run python scripts/ci_check_curriculum.py --config config/curriculum.yaml
    
    # Generate migration guide on drift
    uv run python scripts/ci_check_curriculum.py --migration-guide
    
    # Output JSON report to file
    uv run python scripts/ci_check_curriculum.py --output report.json
    
    # Strict mode (WARN becomes BLOCK)
    uv run python scripts/ci_check_curriculum.py --strict
    
    # View drift timeline for a specific slice
    uv run python scripts/ci_check_curriculum.py --timeline slice_uplift_goal
    
    # View all slice timelines
    uv run python scripts/ci_check_curriculum.py --timeline-all

Exit Codes:
    0 - PASS or WARN (non-blocking)
    1 - BLOCK (contract violation)
    2 - Error (missing config, ledger issues, etc.)
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.curriculum_hash_ledger import (
    CurriculumHashLedger,
    DriftType,
    RiskLevel,
    LedgerSigner,
    DriftEvent,
)
from experiments.curriculum_drift_contract import (
    DriftContract,
    ContractVerdict,
    validate_drift,
)


def handle_timeline_mode(
    ledger: CurriculumHashLedger,
    slice_name: str,
    json_mode: bool = False
) -> int:
    """
    Handle --timeline mode: show drift history for a specific slice.
    
    Args:
        ledger: The CurriculumHashLedger instance.
        slice_name: Name of the slice to show timeline for.
        json_mode: If True, output JSON instead of formatted text.
    
    Returns:
        Exit code (0 for success, 2 for error).
    """
    timeline = ledger.build_drift_timeline(slice_name)
    
    if json_mode:
        output = {
            "slice": slice_name,
            "events": [e.to_dict() for e in timeline],
            "total_events": len(timeline),
            "block_count": sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK),
            "warn_count": sum(1 for e in timeline if e.risk_level == RiskLevel.WARN),
            "info_count": sum(1 for e in timeline if e.risk_level == RiskLevel.INFO)
        }
        print(json.dumps(output, indent=2))
    else:
        print(ledger.format_timeline(timeline, slice_name))
    
    return 0


def handle_timeline_all_mode(
    ledger: CurriculumHashLedger,
    json_mode: bool = False
) -> int:
    """
    Handle --timeline-all mode: show drift history for all slices.
    
    Args:
        ledger: The CurriculumHashLedger instance.
        json_mode: If True, output JSON instead of formatted text.
    
    Returns:
        Exit code (0 for success).
    """
    all_timelines = ledger.build_all_slices_timeline()
    
    if json_mode:
        output = {
            "slices": {},
            "summary": {
                "total_slices": len(all_timelines),
                "slices_with_drift": sum(1 for t in all_timelines.values() if t)
            }
        }
        for slice_name, timeline in all_timelines.items():
            output["slices"][slice_name] = {
                "events": [e.to_dict() for e in timeline],
                "total_events": len(timeline),
                "block_count": sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK)
            }
        print(json.dumps(output, indent=2))
    else:
        print("# All Slice Drift Timelines")
        print("")
        
        if not all_timelines:
            print("No slices found in ledger.")
            return 0
        
        for slice_name, timeline in sorted(all_timelines.items()):
            if timeline:
                print(f"\n{'='*60}")
                print(ledger.format_timeline(timeline, slice_name))
            else:
                print(f"\n## {slice_name}: No drift events")
    
    return 0


def format_block_violation(
    diff: Dict[str, Any],
    contract_result: Any,
    snapshots: list,
    config_path: str
) -> str:
    """
    Format a detailed BLOCK-level contract violation message.
    
    Includes:
    - Violation details
    - Ledger entry IDs involved
    - Clear remediation steps
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("🚫 DRIFT CONTRACT VIOLATION")
    lines.append("=" * 70)
    lines.append("")
    
    # Violation details
    affected_slices = diff.get("affected_slices", {})
    for slice_name, slice_info in affected_slices.items():
        drift_type = slice_info.get("drift_type", "UNKNOWN")
        risk = "BLOCK" if drift_type in ["STRUCTURAL", "SEMANTIC", "PARAMETRIC_MAJOR"] else "WARN"
        
        if risk == "BLOCK":
            lines.append(f"  DRIFT CONTRACT VIOLATION (slice={slice_name}, drift_type={drift_type}, risk=BLOCK)")
    
    lines.append("")
    
    # Ledger entry IDs
    lines.append("📋 Ledger Entries Involved:")
    old_idx = len(snapshots) - 2 if len(snapshots) >= 2 else 0
    new_idx = len(snapshots) - 1
    
    old_ts = diff.get("old_timestamp", "N/A")
    new_ts = diff.get("new_timestamp", "N/A")
    old_hash = diff.get("old_curriculum_hash", "N/A")[:16]
    new_hash = diff.get("new_curriculum_hash", "N/A")[:16]
    
    lines.append(f"  • Entry [{old_idx}]: {old_ts} (hash: {old_hash}...)")
    lines.append(f"  • Entry [{new_idx}]: {new_ts} (hash: {new_hash}...)")
    lines.append("")
    
    # Contract violations
    if contract_result.violations:
        lines.append("❌ Contract Violations:")
        for v in contract_result.violations:
            lines.append(f"  • [{v.rule.value}] {v.message}")
        lines.append("")
    
    # Remediation
    lines.append("🔧 To Resolve:")
    lines.append(f"  1. Review the curriculum changes carefully")
    lines.append(f"  2. If changes are intentional, approve with:")
    lines.append(f"")
    lines.append(f"     uv run python experiments/curriculum_hash_ledger.py \\")
    lines.append(f"         --snapshot --config {config_path} --origin=manual \\")
    lines.append(f"         --notes=\"Approved: <your justification>\"")
    lines.append(f"")
    lines.append(f"  3. Re-run CI after approval")
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_migration_guide(
    diff: Dict[str, Any],
    contract_result: Any,
    config_path: str,
    output_dir: Path = Path("docs")
) -> Path:
    """
    Generate a migration guide document for curriculum drift.
    
    Args:
        diff: The drift classification dict.
        contract_result: The contract validation result.
        config_path: Path to the curriculum config file.
        output_dir: Directory to write the guide.
    
    Returns:
        Path to the generated guide file.
    """
    # Generate filename from hash
    new_hash = diff.get("new_curriculum_hash", "unknown")[:12]
    old_hash = diff.get("old_curriculum_hash", "unknown")[:12]
    filename = f"CURRICULUM_MIGRATION_{old_hash}_to_{new_hash}.md"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    lines = []
    lines.append(f"# Curriculum Migration Guide")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}")
    lines.append(f"**Config**: `{config_path}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| **Old Hash** | `{diff.get('old_curriculum_hash', 'N/A')}` |")
    lines.append(f"| **New Hash** | `{diff.get('new_curriculum_hash', 'N/A')}` |")
    lines.append(f"| **Drift Type** | `{diff.get('drift_type', 'UNKNOWN')}` |")
    lines.append(f"| **Risk Level** | `{diff.get('risk_level', 'UNKNOWN')}` |")
    lines.append(f"| **Contract Verdict** | `{contract_result.verdict.value}` |")
    lines.append(f"| **Old Timestamp** | {diff.get('old_timestamp', 'N/A')} |")
    lines.append(f"| **New Timestamp** | {diff.get('new_timestamp', 'N/A')} |")
    lines.append(f"| **Git Commits** | `{diff.get('old_git_commit', 'N/A')[:8]}...` → `{diff.get('new_git_commit', 'N/A')[:8]}...` |")
    lines.append("")
    
    # Severity explanation
    lines.append("## Severity Classification")
    lines.append("")
    drift_type = diff.get('drift_type', 'UNKNOWN')
    severity_explanations = {
        "NONE": "No changes detected. Curriculum is identical to the previous snapshot.",
        "COSMETIC": "Only cosmetic changes (whitespace, key ordering). No functional impact.",
        "PARAMETRIC_MINOR": "Minor parameter adjustments (<10% change). Review recommended but not blocking.",
        "PARAMETRIC_MAJOR": "Major parameter changes (>50% or removed parameters). Requires explicit approval.",
        "SEMANTIC": "Experiment definition changed (formula pool, targets, success metrics). Critical review required.",
        "STRUCTURAL": "Slice topology changed (slices added or removed). Full review and approval required.",
    }
    lines.append(f"**{drift_type}**: {severity_explanations.get(drift_type, 'Unknown drift type.')}")
    lines.append("")
    
    # What changed section
    lines.append("## What Changed")
    lines.append("")
    
    affected_slices = diff.get('affected_slices', {})
    added_slices = diff.get('added_slices', [])
    removed_slices = diff.get('removed_slices', [])
    changed_slices = diff.get('changed_slices', [])
    
    if added_slices:
        lines.append("### Added Slices")
        lines.append("")
        for name in added_slices:
            lines.append(f"- `{name}` — **NEW**")
        lines.append("")
    
    if removed_slices:
        lines.append("### Removed Slices")
        lines.append("")
        for name in removed_slices:
            lines.append(f"- `{name}` — **REMOVED**")
        lines.append("")
    
    if changed_slices:
        lines.append("### Modified Slices")
        lines.append("")
        for name in changed_slices:
            slice_info = affected_slices.get(name, {})
            slice_drift = slice_info.get('drift_type', 'UNKNOWN')
            changed_keys = slice_info.get('changed_keys', [])
            
            lines.append(f"#### `{name}` ({slice_drift})")
            lines.append("")
            if changed_keys:
                lines.append("Changed keys:")
                lines.append("")
                for key in changed_keys[:15]:
                    lines.append(f"- `{key}`")
                if len(changed_keys) > 15:
                    lines.append(f"- ... and {len(changed_keys) - 15} more")
                lines.append("")
    
    if not (added_slices or removed_slices or changed_slices):
        lines.append("No slice-level changes detected.")
        lines.append("")
    
    # Contract violations section
    lines.append("## Contract Validation")
    lines.append("")
    
    if contract_result.violations:
        lines.append("### ❌ Violations (BLOCKING)")
        lines.append("")
        for v in contract_result.violations:
            lines.append(f"- **{v.rule.value}**: {v.message}")
        lines.append("")
    
    if contract_result.warnings:
        lines.append("### ⚠️ Warnings")
        lines.append("")
        for w in contract_result.warnings:
            lines.append(f"- **{w.rule.value}**: {w.message}")
        lines.append("")
    
    if contract_result.passes and not (contract_result.violations or contract_result.warnings):
        lines.append("✅ All contract rules passed.")
        lines.append("")
    
    # Recommended actions section
    lines.append("## Recommended Actions")
    lines.append("")
    
    if contract_result.verdict == ContractVerdict.BLOCK:
        lines.append("This change **requires explicit approval** before proceeding.")
        lines.append("")
        lines.append("1. **Review** all changes listed above carefully")
        lines.append("2. **Verify** that changes are intentional and documented")
        lines.append("3. **Approve** by recording a new baseline snapshot:")
        lines.append("   ```bash")
        lines.append(f"   uv run python experiments/curriculum_hash_ledger.py \\")
        lines.append(f"       --snapshot --config {config_path} --origin=manual \\")
        lines.append(f"       --notes=\"Approved: <reason for change>\"")
        lines.append("   ```")
        lines.append("4. **Update** any dependent experiments or documentation")
    elif contract_result.verdict == ContractVerdict.WARN:
        lines.append("This change is **non-blocking** but should be reviewed.")
        lines.append("")
        lines.append("1. **Review** the minor parameter changes")
        lines.append("2. **Document** the reason for adjustments")
        lines.append("3. **Monitor** experiment results for any impact")
    else:
        lines.append("No action required. Changes are cosmetic or non-existent.")
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*This migration guide was automatically generated by the Curriculum Hash Ledger system.*")
    lines.append(f"*Generated at: {datetime.now(timezone.utc).isoformat()}*")
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


def build_ci_report(
    diff: Dict[str, Any],
    contract_result: Any,
    config_path: str,
    ledger_path: str,
    migration_guide_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Build a comprehensive CI report as machine-readable JSON.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "version": "2.0",
        "config_path": config_path,
        "ledger_path": ledger_path,
        "drift": {
            "has_drift": diff.get("has_drift", False),
            "drift_type": diff.get("drift_type", "NONE"),
            "risk_level": diff.get("risk_level", "INFO"),
            "global_hash_changed": diff.get("global_hash_changed", False),
            "old_hash": diff.get("old_curriculum_hash"),
            "new_hash": diff.get("new_curriculum_hash"),
            "old_timestamp": diff.get("old_timestamp"),
            "new_timestamp": diff.get("new_timestamp"),
            "old_git_commit": diff.get("old_git_commit"),
            "new_git_commit": diff.get("new_git_commit"),
            "added_slices": diff.get("added_slices", []),
            "removed_slices": diff.get("removed_slices", []),
            "changed_slices": diff.get("changed_slices", []),
            "affected_slices": diff.get("affected_slices", {})
        },
        "contract": contract_result.to_dict(),
        "ci": {
            "verdict": contract_result.verdict.value,
            "exit_code": 1 if contract_result.verdict == ContractVerdict.BLOCK else 0,
            "blocking": contract_result.verdict == ContractVerdict.BLOCK,
            "has_warnings": len(contract_result.warnings) > 0
        },
        "migration_guide": str(migration_guide_path) if migration_guide_path else None
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI Gate for Curriculum Change Control"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/curriculum_uplift_phase2.yaml",
        help="Path to curriculum config file."
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default=None,
        help="Path to ledger JSONL file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON report."
    )
    parser.add_argument(
        "--migration-guide",
        action="store_true",
        help="Generate migration guide document on drift."
    )
    parser.add_argument(
        "--migration-dir",
        type=str,
        default="docs",
        help="Directory for migration guide output."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: WARN-level issues become BLOCK."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except for errors."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output only JSON to stdout."
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Don't record a new snapshot, just compare."
    )
    # Timeline mode arguments
    parser.add_argument(
        "--timeline",
        type=str,
        metavar="SLICE_NAME",
        help="Show drift timeline for a specific slice."
    )
    parser.add_argument(
        "--timeline-all",
        action="store_true",
        help="Show drift timelines for all slices."
    )
    
    args = parser.parse_args()
    
    # Handle timeline modes first (they don't need config file)
    if args.timeline or args.timeline_all:
        ledger_path = Path(args.ledger) if args.ledger else None
        ledger = CurriculumHashLedger(ledger_path=ledger_path)
        
        if args.timeline:
            return handle_timeline_mode(ledger, args.timeline, args.json)
        else:
            return handle_timeline_all_mode(ledger, args.json)
    
    # Initialize ledger
    ledger_path = Path(args.ledger) if args.ledger else None
    ledger = CurriculumHashLedger(ledger_path=ledger_path)
    
    config_path = Path(args.config)
    if not config_path.exists():
        if not args.quiet and not args.json:
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 2
    
    # Load snapshots
    snapshots = ledger.load_snapshots()
    
    # Get or create baseline snapshot
    if len(snapshots) == 0:
        if not args.quiet and not args.json:
            print("No baseline snapshot found. Recording initial snapshot...")
        
        entry = ledger.record_snapshot(
            config_path=str(config_path),
            origin="ci",
            notes="Initial baseline snapshot by CI gate"
        )
        
        if not args.quiet and not args.json:
            print(f"Baseline recorded: {entry['curriculum_hash'][:16]}...")
            print("✅ CI PASSED: Initial baseline established.")
        
        # No drift possible on first run
        if args.json:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "version": "2.0",
                "config_path": str(config_path),
                "ci": {
                    "verdict": "PASS",
                    "exit_code": 0,
                    "blocking": False,
                    "message": "Initial baseline established"
                }
            }
            print(json.dumps(report, indent=2))
        return 0
    
    # Get last snapshot as baseline
    old_snap = snapshots[-1]
    
    # Compute current state
    current_hash, current_slices = ledger.compute_curriculum_hash(str(config_path))
    
    # Check for quick no-change case
    if current_hash == old_snap.get("curriculum_hash"):
        if not args.quiet and not args.json:
            print("✅ CI PASSED: No curriculum changes detected.")
            print(f"   Hash: {current_hash[:16]}...")
        
        if args.json:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "version": "2.0",
                "config_path": str(config_path),
                "drift": {
                    "has_drift": False,
                    "drift_type": "NONE",
                    "hash": current_hash
                },
                "ci": {
                    "verdict": "PASS",
                    "exit_code": 0,
                    "blocking": False
                }
            }
            print(json.dumps(report, indent=2))
        return 0
    
    # Changes detected - record new snapshot and classify
    if not args.no_snapshot:
        new_snap = ledger.record_snapshot(
            config_path=str(config_path),
            origin="ci",
            notes="CI gate snapshot for comparison"
        )
    else:
        # Create virtual snapshot for comparison
        new_snap = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "config_path": str(config_path),
            "curriculum_hash": current_hash,
            "git_commit": "HEAD",
            "slice_hashes": current_slices,
            "origin": "ci",
            "notes": "Virtual snapshot (not recorded)"
        }
    
    # Load configs for deep comparison
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        new_config = yaml.safe_load(f)
    
    # Classify drift
    diff = ledger.classify_drift(old_snap, new_snap, new_config, new_config)
    
    # Validate against contract
    contract_result = validate_drift(diff, strict_mode=args.strict)
    
    # Generate migration guide if requested and there's drift
    migration_guide_path = None
    if args.migration_guide and diff.get("has_drift"):
        migration_guide_path = generate_migration_guide(
            diff=diff,
            contract_result=contract_result,
            config_path=str(config_path),
            output_dir=Path(args.migration_dir)
        )
        if not args.quiet and not args.json:
            print(f"Migration guide generated: {migration_guide_path}")
    
    # Build full report
    report = build_ci_report(
        diff=diff,
        contract_result=contract_result,
        config_path=str(config_path),
        ledger_path=str(ledger.ledger_path),
        migration_guide_path=migration_guide_path
    )
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        if not args.quiet and not args.json:
            print(f"Report written to: {output_path}")
    
    if args.json:
        print(json.dumps(report, indent=2))
    elif not args.quiet:
        # Print human-readable summary
        print(ledger.format_diff_report(diff))
        print()
        print(contract_result.summary)
        
        if contract_result.verdict == ContractVerdict.BLOCK:
            # Enhanced BLOCK violation reporting
            print(format_block_violation(
                diff=diff,
                contract_result=contract_result,
                snapshots=snapshots + [new_snap],
                config_path=str(config_path)
            ))
        elif contract_result.verdict == ContractVerdict.WARN:
            print()
            print("=" * 60)
            print("⚠️  CI PASSED with warnings.")
            print("    Review changes before deployment.")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print("✅ CI PASSED: No blocking issues.")
            print("=" * 60)
    
    # Return exit code
    return 1 if contract_result.verdict == ContractVerdict.BLOCK else 0


if __name__ == "__main__":
    sys.exit(main())

