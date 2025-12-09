#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

Snapshot Pruning Tool
=====================

Safely prunes old snapshots while guaranteeing at least one valid resume point.

Usage:
    # Dry-run to see what would be deleted
    python scripts/snapshot_prune.py --run-dir results/u2 --keep-latest 5 --dry-run

    # Actually prune, keeping the 5 most recent
    python scripts/snapshot_prune.py --run-dir results/u2 --keep-latest 5

    # Keep every 10th cycle snapshot
    python scripts/snapshot_prune.py --run-dir results/u2 --keep-interval 10 --dry-run

    # Combined: keep latest 5 + every 10th cycle
    python scripts/snapshot_prune.py --run-dir results/u2 --keep-latest 5 --keep-interval 10

Constraints:
    - NEVER deletes the last VALID snapshot (safety guarantee)
    - Prefers to keep VALID over CORRUPTED/INVALID snapshots
    - Only deletes snapshot files, never logs/manifests
    - Idempotent: safe to run multiple times
    - Read-only in --dry-run mode

Exit Codes:
    0 - Success (pruning completed or dry-run finished)
    1 - Some snapshots have issues
    2 - No snapshots found or configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.u2.snapshots import (
    SnapshotData,
    SnapshotValidationError,
    SnapshotCorruptionError,
    load_snapshot,
    list_snapshots,
)


class SnapshotHealth(str, Enum):
    """Snapshot health status for pruning decisions."""
    VALID = "VALID"
    CORRUPTED = "CORRUPTED"
    INVALID = "INVALID"
    UNREADABLE = "UNREADABLE"


@dataclass
class SnapshotInfo:
    """Information about a snapshot for pruning decisions."""
    path: Path
    health: SnapshotHealth
    cycle_index: Optional[int] = None
    total_cycles: Optional[int] = None
    is_final: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class PruneDecision:
    """Decision for a single snapshot."""
    path: Path
    action: str  # "KEEP" or "DELETE"
    reason: str
    health: SnapshotHealth
    cycle_index: Optional[int] = None


@dataclass
class PruneReport:
    """Report of pruning operation."""
    run_dir: str
    dry_run: bool
    keep_latest: int
    keep_interval: Optional[int]
    total_snapshots: int
    valid_count: int
    to_keep: List[PruneDecision]
    to_delete: List[PruneDecision]
    actually_deleted: List[str]
    errors: List[str]
    
    @property
    def delete_count(self) -> int:
        return len(self.to_delete)
    
    @property
    def keep_count(self) -> int:
        return len(self.to_keep)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "dry_run": self.dry_run,
            "policy": {
                "keep_latest": self.keep_latest,
                "keep_interval": self.keep_interval,
            },
            "summary": {
                "total_snapshots": self.total_snapshots,
                "valid_snapshots": self.valid_count,
                "to_keep": self.keep_count,
                "to_delete": self.delete_count,
                "actually_deleted": len(self.actually_deleted),
            },
            "to_keep": [
                {"path": str(d.path), "reason": d.reason, "health": d.health.value, "cycle": d.cycle_index}
                for d in self.to_keep
            ],
            "to_delete": [
                {"path": str(d.path), "reason": d.reason, "health": d.health.value, "cycle": d.cycle_index}
                for d in self.to_delete
            ],
            "actually_deleted": self.actually_deleted,
            "errors": self.errors,
        }


def assess_snapshot_health(path: Path) -> SnapshotInfo:
    """
    Assess the health of a snapshot file.
    
    Args:
        path: Path to snapshot file
        
    Returns:
        SnapshotInfo with health status and metadata
    """
    info = SnapshotInfo(
        path=path,
        health=SnapshotHealth.UNREADABLE,
        is_final="final" in path.stem.lower(),
    )
    
    if not path.exists():
        info.errors.append(f"File not found: {path}")
        return info
    
    if path.stat().st_size == 0:
        info.health = SnapshotHealth.INVALID
        info.errors.append("File is empty")
        return info
    
    try:
        snapshot = load_snapshot(path, verify_hash=True)
        info.health = SnapshotHealth.VALID
        info.cycle_index = snapshot.cycle_index
        info.total_cycles = snapshot.total_cycles
    except SnapshotCorruptionError as e:
        info.health = SnapshotHealth.CORRUPTED
        info.errors.append(f"Corruption: {e}")
        
        # Try to extract metadata without hash verification
        try:
            snapshot = load_snapshot(path, verify_hash=False)
            info.cycle_index = snapshot.cycle_index
            info.total_cycles = snapshot.total_cycles
        except Exception:
            pass
    except SnapshotValidationError as e:
        info.health = SnapshotHealth.INVALID
        info.errors.append(f"Validation: {e}")
    except Exception as e:
        info.health = SnapshotHealth.UNREADABLE
        info.errors.append(f"Load error: {e}")
    
    return info


def compute_prune_plan(
    snapshot_dir: Path,
    keep_latest: int = 5,
    keep_interval: Optional[int] = None,
) -> PruneReport:
    """
    Compute a pruning plan based on retention policy.
    
    Policy:
    1. Always keep at least one VALID snapshot (safety invariant)
    2. Keep the N most recent snapshots by cycle index
    3. Optionally keep every Kth cycle snapshot
    4. Always keep "final" snapshots
    5. Prefer keeping VALID over CORRUPTED/INVALID
    
    Args:
        snapshot_dir: Directory containing snapshots
        keep_latest: Number of latest snapshots to keep
        keep_interval: If set, also keep every Kth cycle
        
    Returns:
        PruneReport with decisions for each snapshot
    """
    # Discover all snapshots
    paths = list_snapshots(snapshot_dir)
    
    # Assess health of each
    infos: List[SnapshotInfo] = []
    for path in paths:
        info = assess_snapshot_health(path)
        infos.append(info)
    
    # Count valid snapshots
    valid_count = sum(1 for info in infos if info.health == SnapshotHealth.VALID)
    
    # Determine which to keep
    to_keep: Set[Path] = set()
    to_delete: Set[Path] = set()
    keep_reasons: Dict[Path, str] = {}
    delete_reasons: Dict[Path, str] = {}
    
    # Rule 1: Always keep "final" snapshots
    for info in infos:
        if info.is_final:
            to_keep.add(info.path)
            keep_reasons[info.path] = "Final snapshot (never deleted)"
    
    # Rule 2: Keep the N most recent snapshots by cycle
    # Sort by cycle index descending (newest first), but only VALID ones first
    valid_infos = sorted(
        [i for i in infos if i.health == SnapshotHealth.VALID and not i.is_final],
        key=lambda i: i.cycle_index or 0,
        reverse=True,
    )
    
    for i, info in enumerate(valid_infos[:keep_latest]):
        if info.path not in to_keep:
            to_keep.add(info.path)
            keep_reasons[info.path] = f"Latest {i+1} of {keep_latest}"
    
    # If we haven't kept enough, fill with non-valid snapshots
    remaining_to_keep = keep_latest - len([p for p in to_keep if "Final" not in keep_reasons.get(p, "")])
    if remaining_to_keep > 0:
        other_infos = sorted(
            [i for i in infos if i.path not in to_keep and not i.is_final],
            key=lambda i: i.cycle_index or 0,
            reverse=True,
        )
        for info in other_infos[:remaining_to_keep]:
            to_keep.add(info.path)
            keep_reasons[info.path] = f"Latest (non-valid) in top {keep_latest}"
    
    # Rule 3: Keep every Kth cycle snapshot if interval specified
    if keep_interval and keep_interval > 0:
        for info in infos:
            if info.cycle_index is not None and info.cycle_index > 0:
                if info.cycle_index % keep_interval == 0:
                    if info.path not in to_keep:
                        to_keep.add(info.path)
                        keep_reasons[info.path] = f"Interval checkpoint (every {keep_interval} cycles)"
    
    # Rule 4: SAFETY INVARIANT - Always keep at least one VALID snapshot
    valid_kept = sum(
        1 for path in to_keep 
        if any(i.path == path and i.health == SnapshotHealth.VALID for i in infos)
    )
    
    if valid_kept == 0 and valid_count > 0:
        # Force keep the latest valid snapshot
        for info in valid_infos:
            if info.path not in to_keep:
                to_keep.add(info.path)
                keep_reasons[info.path] = "SAFETY: Last valid snapshot (never deleted)"
                break
    
    # Everything else gets deleted
    for info in infos:
        if info.path not in to_keep:
            to_delete.add(info.path)
            if info.health != SnapshotHealth.VALID:
                delete_reasons[info.path] = f"Non-valid snapshot ({info.health.value}) outside retention"
            else:
                delete_reasons[info.path] = "Outside retention policy"
    
    # Build decisions
    keep_decisions = []
    delete_decisions = []
    
    for info in infos:
        if info.path in to_keep:
            keep_decisions.append(PruneDecision(
                path=info.path,
                action="KEEP",
                reason=keep_reasons.get(info.path, "Unknown"),
                health=info.health,
                cycle_index=info.cycle_index,
            ))
        else:
            delete_decisions.append(PruneDecision(
                path=info.path,
                action="DELETE",
                reason=delete_reasons.get(info.path, "Outside retention policy"),
                health=info.health,
                cycle_index=info.cycle_index,
            ))
    
    # Sort decisions by cycle index
    keep_decisions.sort(key=lambda d: d.cycle_index or 0, reverse=True)
    delete_decisions.sort(key=lambda d: d.cycle_index or 0, reverse=True)
    
    return PruneReport(
        run_dir=str(snapshot_dir),
        dry_run=True,  # Will be updated by caller
        keep_latest=keep_latest,
        keep_interval=keep_interval,
        total_snapshots=len(infos),
        valid_count=valid_count,
        to_keep=keep_decisions,
        to_delete=delete_decisions,
        actually_deleted=[],
        errors=[],
    )


def execute_prune(report: PruneReport, dry_run: bool = True) -> PruneReport:
    """
    Execute the pruning plan.
    
    Args:
        report: PruneReport from compute_prune_plan
        dry_run: If True, don't actually delete files
        
    Returns:
        Updated PruneReport with actually_deleted list
    """
    report.dry_run = dry_run
    
    if dry_run:
        return report
    
    for decision in report.to_delete:
        try:
            decision.path.unlink()
            report.actually_deleted.append(str(decision.path))
        except OSError as e:
            report.errors.append(f"Failed to delete {decision.path}: {e}")
    
    return report


def print_report(report: PruneReport) -> None:
    """Print pruning report in human-readable format."""
    print()
    print("=" * 70)
    print(f"Snapshot Pruning Report {'(DRY RUN)' if report.dry_run else ''}")
    print("=" * 70)
    print()
    
    print(f"  Directory:        {report.run_dir}")
    print(f"  Total Snapshots:  {report.total_snapshots}")
    print(f"  Valid Snapshots:  {report.valid_count}")
    print()
    
    print("  Retention Policy:")
    print(f"    Keep Latest:    {report.keep_latest}")
    if report.keep_interval:
        print(f"    Keep Interval:  Every {report.keep_interval} cycles")
    print()
    
    print("  Decision Summary:")
    print(f"    \033[32mKEEP:    {report.keep_count}\033[0m")
    print(f"    \033[31mDELETE:  {report.delete_count}\033[0m")
    print()
    
    if report.to_keep:
        print("  Snapshots to KEEP:")
        for d in report.to_keep[:10]:  # Show at most 10
            health_color = "\033[32m" if d.health == SnapshotHealth.VALID else "\033[33m"
            cycle_str = f"cycle {d.cycle_index}" if d.cycle_index is not None else "unknown"
            print(f"    {health_color}✓\033[0m {d.path.name} ({cycle_str})")
            print(f"      └─ {d.reason}")
        if len(report.to_keep) > 10:
            print(f"    ... and {len(report.to_keep) - 10} more")
        print()
    
    if report.to_delete:
        print("  Snapshots to DELETE:")
        for d in report.to_delete[:10]:  # Show at most 10
            health_color = "\033[31m" if d.health == SnapshotHealth.CORRUPTED else "\033[33m"
            cycle_str = f"cycle {d.cycle_index}" if d.cycle_index is not None else "unknown"
            print(f"    {health_color}✗\033[0m {d.path.name} ({cycle_str})")
            print(f"      └─ {d.reason}")
        if len(report.to_delete) > 10:
            print(f"    ... and {len(report.to_delete) - 10} more")
        print()
    
    if report.dry_run:
        print("  \033[33m⚠ DRY RUN - No files were deleted.\033[0m")
        print("  Re-run without --dry-run to actually delete files.")
    else:
        print(f"  Actually deleted: {len(report.actually_deleted)} files")
        if report.errors:
            print()
            print("  Errors:")
            for err in report.errors:
                print(f"    \033[31m✗\033[0m {err}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PHASE II Snapshot Pruning Tool - Safely prune old snapshots",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Dry-run to see what would be deleted
  python scripts/snapshot_prune.py --run-dir results/u2 --keep-latest 5 --dry-run

  # Actually prune, keeping the 5 most recent
  python scripts/snapshot_prune.py --run-dir results/u2 --keep-latest 5

  # Keep every 10th cycle snapshot
  python scripts/snapshot_prune.py --run-dir results/u2 --keep-interval 10 --dry-run

Safety:
  - NEVER deletes the last valid snapshot
  - Prefers keeping VALID over CORRUPTED/INVALID
  - Always preserves "final" snapshots
        """
    )
    
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to experiment run directory containing snapshots"
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=5,
        help="Number of latest snapshots to keep (default: 5)"
    )
    parser.add_argument(
        "--keep-interval",
        type=int,
        default=None,
        help="Keep every Nth cycle snapshot (e.g., 10 = keep cycles 10, 20, 30...)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)
    
    # Find snapshot directory
    snapshot_dir = run_dir / "snapshots"
    if not snapshot_dir.exists():
        snapshot_dir = run_dir  # Snapshots might be directly in run_dir
    
    if args.keep_latest < 1:
        print("ERROR: --keep-latest must be at least 1", file=sys.stderr)
        sys.exit(2)
    
    # Compute pruning plan
    report = compute_prune_plan(
        snapshot_dir,
        keep_latest=args.keep_latest,
        keep_interval=args.keep_interval,
    )
    
    # Execute (or dry-run)
    report = execute_prune(report, dry_run=args.dry_run)
    
    # Output
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)
    
    # Exit code
    if report.total_snapshots == 0:
        sys.exit(2)
    elif report.errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

