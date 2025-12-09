#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

Snapshot Doctor Tool
====================

Diagnoses snapshot health, validates integrity, and suggests recovery strategies.

Usage:
    # Scan all snapshots in a run directory
    python scripts/snapshot_doctor.py --run-dir results/u2

    # Verify a specific snapshot
    python scripts/snapshot_doctor.py --snapshot results/u2/snapshots/snapshot_0100.snap

    # Get recovery recommendation
    python scripts/snapshot_doctor.py --run-dir results/u2 --suggest-recovery

    # Deep verification (slower but thorough)
    python scripts/snapshot_doctor.py --run-dir results/u2 --deep

    # JSON output for scripting
    python scripts/snapshot_doctor.py --run-dir results/u2 --json

Absolute Safeguards:
    - Read-only: Never modifies or "fixes" snapshots.
    - Advisory: Reports findings and recommendations only.
    - Deterministic: Same inputs produce same outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    find_latest_snapshot,
    validate_snapshot,
    compute_snapshot_hash,
)
from experiments.u2.snapshot_guard import (
    ValidationStatus,
    ValidationResult,
    check_resume_compatibility,
    validate_snapshot_file_against_manifest,
)
from experiments.u2.snapshot_history import (
    build_snapshot_history,
    advise_resume_strategy,
    summarize_snapshots_for_global_health,
    HISTORY_SCHEMA_VERSION,
    HistoryStatus,
    ResumeStatus,
)


class SnapshotStatus(str, Enum):
    """Snapshot health status."""
    VALID = "VALID"
    CORRUPTED = "CORRUPTED"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"
    UNREADABLE = "UNREADABLE"


@dataclass
class VerificationResult:
    """Result of snapshot verification."""
    path: str
    status: SnapshotStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata (populated if readable)
    cycle_index: Optional[int] = None
    total_cycles: Optional[int] = None
    mode: Optional[str] = None
    slice_name: Optional[str] = None
    experiment_id: Optional[str] = None
    
    # Integrity checks
    hash_verified: bool = False
    schema_valid: bool = False
    fields_missing: List[str] = field(default_factory=list)
    
    # File info
    size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "status": self.status.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "cycle_index": self.cycle_index,
            "total_cycles": self.total_cycles,
            "mode": self.mode,
            "slice_name": self.slice_name,
            "experiment_id": self.experiment_id,
            "hash_verified": self.hash_verified,
            "schema_valid": self.schema_valid,
            "fields_missing": self.fields_missing,
            "size_bytes": self.size_bytes,
        }


@dataclass
class ScanReport:
    """Result of scanning a run directory."""
    run_dir: str
    snapshot_dir: str
    total_snapshots: int
    valid_count: int
    corrupted_count: int
    incomplete_count: int
    invalid_count: int
    unreadable_count: int
    snapshots: List[VerificationResult]
    
    # Recovery suggestion
    recommended_resume_point: Optional[str] = None
    recommended_cycle: Optional[int] = None
    recovery_notes: List[str] = field(default_factory=list)
    
    # Manifest validation (for --suggest-recovery)
    manifest_status: Optional[str] = None
    manifest_compatible: bool = True
    manifest_warnings: List[str] = field(default_factory=list)
    
    # Resume strategy (from snapshot_history module)
    resume_strategy: Optional[Dict[str, Any]] = None
    global_health: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_dir": self.run_dir,
            "snapshot_dir": self.snapshot_dir,
            "summary": {
                "total": self.total_snapshots,
                "valid": self.valid_count,
                "corrupted": self.corrupted_count,
                "incomplete": self.incomplete_count,
                "invalid": self.invalid_count,
                "unreadable": self.unreadable_count,
            },
            "snapshots": [s.to_dict() for s in self.snapshots],
            "recovery": {
                "recommended_resume_point": self.recommended_resume_point,
                "recommended_cycle": self.recommended_cycle,
                "notes": self.recovery_notes,
                "manifest_status": self.manifest_status,
                "manifest_compatible": self.manifest_compatible,
                "manifest_warnings": self.manifest_warnings,
            },
            "resume_strategy": self.resume_strategy,
            "global_health": self.global_health,
        }


def verify_snapshot(path: Path, deep: bool = False) -> VerificationResult:
    """
    Verify a snapshot file and return detailed results.
    
    Args:
        path: Path to snapshot file
        deep: If True, perform additional deep verification
        
    Returns:
        VerificationResult with status and details
    """
    result = VerificationResult(
        path=str(path),
        status=SnapshotStatus.UNREADABLE,
    )
    
    # Check file exists and get size
    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result
    
    result.size_bytes = path.stat().st_size
    
    if result.size_bytes == 0:
        result.status = SnapshotStatus.INCOMPLETE
        result.errors.append("File is empty (0 bytes)")
        return result
    
    # Try to load with hash verification
    try:
        snapshot = load_snapshot(path, verify_hash=True)
        result.hash_verified = True
    except SnapshotCorruptionError as e:
        result.status = SnapshotStatus.CORRUPTED
        result.errors.append(f"Hash verification failed: {e}")
        
        # Try to load without hash verification for diagnostics
        try:
            snapshot = load_snapshot(path, verify_hash=False)
            result.warnings.append("Snapshot readable but corrupted (hash mismatch)")
        except Exception:
            return result
    except Exception as e:
        result.status = SnapshotStatus.UNREADABLE
        result.errors.append(f"Failed to load: {e}")
        return result
    
    # Extract metadata
    result.cycle_index = snapshot.cycle_index
    result.total_cycles = snapshot.total_cycles
    result.mode = snapshot.mode
    result.slice_name = snapshot.slice_name
    result.experiment_id = snapshot.experiment_id
    
    # Check for missing/empty fields
    critical_fields = [
        ("schema_version", snapshot.schema_version),
        ("cycle_index", snapshot.cycle_index),
        ("master_seed", snapshot.master_seed),
    ]
    
    for field_name, value in critical_fields:
        if value is None or (isinstance(value, str) and not value):
            result.fields_missing.append(field_name)
    
    # Validate schema
    try:
        validate_snapshot(snapshot)
        result.schema_valid = True
    except SnapshotValidationError as e:
        result.schema_valid = False
        result.errors.append(f"Validation failed: {e}")
        result.status = SnapshotStatus.INVALID
        return result
    
    # Deep verification
    if deep:
        # Check PRNG state consistency
        if snapshot.python_rng_state is not None:
            if not isinstance(snapshot.python_rng_state, tuple):
                result.warnings.append("python_rng_state is not a tuple")
        
        if snapshot.numpy_rng_state is not None:
            if not isinstance(snapshot.numpy_rng_state, (tuple, list)):
                result.warnings.append("numpy_rng_state has unexpected type")
        
        # Check seed schedule consistency
        if snapshot.seed_schedule:
            if len(snapshot.seed_schedule) != snapshot.total_cycles:
                result.warnings.append(
                    f"seed_schedule length ({len(snapshot.seed_schedule)}) "
                    f"!= total_cycles ({snapshot.total_cycles})"
                )
        
        # Check cycle bounds
        if snapshot.cycle_index < 0:
            result.errors.append(f"Negative cycle_index: {snapshot.cycle_index}")
            result.status = SnapshotStatus.INVALID
            return result
        
        if snapshot.total_cycles > 0 and snapshot.cycle_index > snapshot.total_cycles:
            result.warnings.append(
                f"cycle_index ({snapshot.cycle_index}) > total_cycles ({snapshot.total_cycles})"
            )
    
    # Determine final status
    if result.status == SnapshotStatus.UNREADABLE:
        if result.hash_verified and result.schema_valid and not result.fields_missing:
            result.status = SnapshotStatus.VALID
        elif result.fields_missing:
            result.status = SnapshotStatus.INCOMPLETE
        else:
            result.status = SnapshotStatus.INVALID
    
    return result


def scan_run_dir(run_dir: Path, deep: bool = False) -> ScanReport:
    """
    Scan all snapshots in a run directory.
    
    Args:
        run_dir: Path to run directory
        deep: If True, perform deep verification on each snapshot
        
    Returns:
        ScanReport with all snapshot statuses
    """
    # Find snapshot directory
    snapshot_dir = run_dir / "snapshots"
    if not snapshot_dir.exists():
        snapshot_dir = run_dir
    
    # List all snapshots
    paths = list_snapshots(snapshot_dir)
    
    # Verify each
    results: List[VerificationResult] = []
    for path in paths:
        result = verify_snapshot(path, deep=deep)
        results.append(result)
    
    # Count by status
    valid_count = sum(1 for r in results if r.status == SnapshotStatus.VALID)
    corrupted_count = sum(1 for r in results if r.status == SnapshotStatus.CORRUPTED)
    incomplete_count = sum(1 for r in results if r.status == SnapshotStatus.INCOMPLETE)
    invalid_count = sum(1 for r in results if r.status == SnapshotStatus.INVALID)
    unreadable_count = sum(1 for r in results if r.status == SnapshotStatus.UNREADABLE)
    
    report = ScanReport(
        run_dir=str(run_dir),
        snapshot_dir=str(snapshot_dir),
        total_snapshots=len(results),
        valid_count=valid_count,
        corrupted_count=corrupted_count,
        incomplete_count=incomplete_count,
        invalid_count=invalid_count,
        unreadable_count=unreadable_count,
        snapshots=results,
    )
    
    return report


def suggest_recovery(run_dir: Path, deep: bool = False, strict: bool = False) -> ScanReport:
    """
    Analyze snapshots and suggest the best recovery strategy.
    
    Uses the snapshot_history module for comprehensive analysis and
    resume strategy advice.
    
    Args:
        run_dir: Path to run directory
        deep: If True, perform deep verification
        strict: If True, only recommend snapshots that pass manifest validation
        
    Returns:
        ScanReport with recovery recommendations and resume strategy
    """
    report = scan_run_dir(run_dir, deep=deep)
    
    # Build comprehensive history using snapshot_history module
    history = build_snapshot_history(run_dir, include_manifest_validation=True)
    
    # Get resume strategy advice
    strategy = advise_resume_strategy(history, strict=strict)
    report.resume_strategy = strategy
    
    # Get global health signal
    global_health = summarize_snapshots_for_global_health(history)
    report.global_health = global_health
    
    # Find valid snapshots sorted by cycle (newest first, excluding final)
    valid_snapshots = [
        r for r in report.snapshots 
        if r.status == SnapshotStatus.VALID and "final" not in r.path.lower()
    ]
    
    if not valid_snapshots:
        report.recovery_notes.append("No valid snapshots found for recovery.")
        
        # Check for corrupted snapshots
        if report.corrupted_count > 0:
            report.recovery_notes.append(
                f"Found {report.corrupted_count} corrupted snapshot(s). "
                "These cannot be used for recovery."
            )
        
        # Check for incomplete snapshots
        if report.incomplete_count > 0:
            report.recovery_notes.append(
                f"Found {report.incomplete_count} incomplete snapshot(s). "
                "Consider re-running from the beginning."
            )
        
        report.recovery_notes.append(
            "Recommendation: Start a fresh run with --snapshot-interval to enable recovery."
        )
        return report
    
    # Sort by cycle index descending
    valid_snapshots.sort(key=lambda r: r.cycle_index or 0, reverse=True)
    
    # Best candidate is the one with highest cycle index
    best = valid_snapshots[0]
    best_path = Path(best.path)
    
    # Validate against manifest (if available)
    manifest_validation = check_resume_compatibility(best_path, run_dir, strict=strict)
    report.manifest_status = manifest_validation.status.value
    report.manifest_compatible = manifest_validation.is_compatible
    report.manifest_warnings = list(manifest_validation.warnings)
    
    # In strict mode, find first manifest-compatible snapshot
    if strict and not manifest_validation.is_compatible:
        report.recovery_notes.append(
            f"Latest snapshot (cycle {best.cycle_index}) failed manifest validation: "
            f"{manifest_validation.status.value}"
        )
        
        # Try to find an earlier compatible snapshot
        for candidate in valid_snapshots[1:]:
            candidate_path = Path(candidate.path)
            candidate_validation = check_resume_compatibility(candidate_path, run_dir, strict=strict)
            if candidate_validation.is_compatible:
                best = candidate
                manifest_validation = candidate_validation
                report.manifest_status = manifest_validation.status.value
                report.manifest_compatible = manifest_validation.is_compatible
                report.manifest_warnings = list(manifest_validation.warnings)
                report.recovery_notes.append(
                    f"Falling back to cycle {best.cycle_index} (passed strict validation)"
                )
                break
        else:
            report.recovery_notes.append(
                "No snapshots pass strict manifest validation."
            )
            if not strict:
                report.recovery_notes.append(
                    "Consider running without --strict to allow recovery from older snapshots."
                )
    
    report.recommended_resume_point = best.path
    report.recommended_cycle = best.cycle_index
    
    # Add notes
    report.recovery_notes.append(
        f"Found {len(valid_snapshots)} valid snapshot(s) for recovery."
    )
    
    if best.cycle_index and best.total_cycles:
        progress = 100 * best.cycle_index / best.total_cycles
        report.recovery_notes.append(
            f"Recommended resume point: cycle {best.cycle_index}/{best.total_cycles} "
            f"({progress:.1f}% complete)"
        )
    
    # Add manifest status note
    if manifest_validation.status == ValidationStatus.OK:
        report.recovery_notes.append("✓ Manifest OK - snapshot matches experiment configuration")
    elif manifest_validation.status == ValidationStatus.UNKNOWN:
        report.recovery_notes.append(
            "⚠ Manifest unknown - snapshot predates manifest_hash field"
        )
    elif manifest_validation.status == ValidationStatus.MISMATCH:
        report.recovery_notes.append(
            "⚠ Manifest mismatch - snapshot may not match current configuration"
        )
    elif manifest_validation.status == ValidationStatus.MANIFEST_MISSING:
        report.recovery_notes.append(
            "ℹ No manifest found - cannot verify configuration compatibility"
        )
    
    # Add any manifest warnings
    for warning in manifest_validation.warnings:
        report.recovery_notes.append(f"  ↳ {warning}")
    
    report.recovery_notes.append(
        f"To resume: python experiments/run_uplift_u2.py --restore-from {best.path} ..."
    )
    
    # Check for gaps
    cycle_indices = sorted([r.cycle_index for r in valid_snapshots if r.cycle_index is not None])
    if len(cycle_indices) > 1:
        gaps = []
        for i in range(1, len(cycle_indices)):
            gap = cycle_indices[i] - cycle_indices[i-1]
            if gap > 1:
                gaps.append((cycle_indices[i-1], cycle_indices[i], gap))
        
        if gaps:
            report.recovery_notes.append(
                f"Warning: Found {len(gaps)} gap(s) in snapshot coverage. "
                "Some cycles may not have checkpoints."
            )
    
    return report


def print_verification_result(result: VerificationResult) -> None:
    """Print verification result in human-readable format."""
    print()
    print("=" * 60)
    print(f"Snapshot Verification: {Path(result.path).name}")
    print("=" * 60)
    print()
    
    # Status with color
    status_color = {
        SnapshotStatus.VALID: "\033[32m",      # Green
        SnapshotStatus.CORRUPTED: "\033[31m",  # Red
        SnapshotStatus.INCOMPLETE: "\033[33m", # Yellow
        SnapshotStatus.INVALID: "\033[33m",    # Yellow
        SnapshotStatus.UNREADABLE: "\033[31m", # Red
    }
    color = status_color.get(result.status, "")
    reset = "\033[0m" if color else ""
    
    print(f"  Status:         {color}{result.status.value}{reset}")
    print(f"  Path:           {result.path}")
    print(f"  Size:           {result.size_bytes:,} bytes")
    print()
    
    print("  Integrity Checks:")
    hash_status = "\033[32m✓\033[0m" if result.hash_verified else "\033[31m✗\033[0m"
    schema_status = "\033[32m✓\033[0m" if result.schema_valid else "\033[31m✗\033[0m"
    print(f"    Hash Verified:  {hash_status}")
    print(f"    Schema Valid:   {schema_status}")
    
    if result.fields_missing:
        print(f"    Missing Fields: {', '.join(result.fields_missing)}")
    print()
    
    if result.cycle_index is not None:
        print("  Snapshot Metadata:")
        print(f"    Cycle:          {result.cycle_index}/{result.total_cycles or '?'}")
        print(f"    Mode:           {result.mode or 'unknown'}")
        print(f"    Slice:          {result.slice_name or 'unknown'}")
        print(f"    Experiment:     {result.experiment_id or 'unknown'}")
        print()
    
    if result.errors:
        print("  Errors:")
        for err in result.errors:
            print(f"    \033[31m✗\033[0m {err}")
        print()
    
    if result.warnings:
        print("  Warnings:")
        for warn in result.warnings:
            print(f"    \033[33m⚠\033[0m {warn}")
        print()


def print_scan_report(report: ScanReport) -> None:
    """Print scan report in human-readable format."""
    print()
    print("=" * 60)
    print("Snapshot Doctor Report")
    print("=" * 60)
    print()
    
    print(f"  Run Directory:    {report.run_dir}")
    print(f"  Snapshot Dir:     {report.snapshot_dir}")
    print(f"  Total Snapshots:  {report.total_snapshots}")
    print()
    
    print("  Status Summary:")
    print(f"    \033[32m✓ Valid:      {report.valid_count}\033[0m")
    if report.corrupted_count > 0:
        print(f"    \033[31m✗ Corrupted:  {report.corrupted_count}\033[0m")
    if report.incomplete_count > 0:
        print(f"    \033[33m⚠ Incomplete: {report.incomplete_count}\033[0m")
    if report.invalid_count > 0:
        print(f"    \033[33m⚠ Invalid:    {report.invalid_count}\033[0m")
    if report.unreadable_count > 0:
        print(f"    \033[31m✗ Unreadable: {report.unreadable_count}\033[0m")
    print()
    
    # Show individual snapshot issues
    problem_snapshots = [s for s in report.snapshots if s.status != SnapshotStatus.VALID]
    if problem_snapshots:
        print("  Problem Snapshots:")
        for snap in problem_snapshots[:5]:  # Show at most 5
            print(f"    {Path(snap.path).name}: {snap.status.value}")
            for err in snap.errors[:2]:
                print(f"      └─ {err}")
        if len(problem_snapshots) > 5:
            print(f"    ... and {len(problem_snapshots) - 5} more")
        print()
    
    # Recovery recommendation
    if report.recovery_notes:
        print("  Recovery Recommendation:")
        for note in report.recovery_notes:
            print(f"    → {note}")
        print()
    
    if report.recommended_resume_point:
        # Show manifest status
        if report.manifest_status:
            manifest_color = {
                "OK": "\033[32m",           # Green
                "UNKNOWN": "\033[33m",      # Yellow
                "MISMATCH": "\033[31m",     # Red
                "MANIFEST_MISSING": "\033[34m",  # Blue
            }
            color = manifest_color.get(report.manifest_status, "")
            reset = "\033[0m" if color else ""
            
            compat_str = "✓" if report.manifest_compatible else "✗"
            print(f"  Manifest Status: {color}{compat_str} {report.manifest_status}{reset}")
            
            if report.manifest_warnings:
                for warning in report.manifest_warnings:
                    print(f"    └─ {warning}")
            print()
        
        print(f"  Recommended Resume Command:")
        print(f"    python experiments/run_uplift_u2.py \\")
        print(f"      --restore-from {report.recommended_resume_point} \\")
        print(f"      --snapshot-interval <N> \\")
        print(f"      [other args...]")
        print()
    
    # Show resume strategy (from snapshot_history module)
    if report.resume_strategy:
        strategy = report.resume_strategy
        status_color = {
            "ALLOWED": "\033[32m",       # Green
            "CONDITIONAL": "\033[33m",   # Yellow
            "BLOCKED": "\033[31m",       # Red
        }
        status = strategy.get("resume_status", "BLOCKED")
        color = status_color.get(status, "")
        reset = "\033[0m" if color else ""
        
        print(f"  Resume Strategy:")
        print(f"    Status:  {color}{status}{reset}")
        print(f"    Allowed: {'Yes' if strategy.get('resume_allowed') else 'No'}")
        
        if strategy.get("warnings"):
            print("    Warnings:")
            for warning in strategy["warnings"][:5]:
                print(f"      \033[33m⚠\033[0m {warning}")
        
        if strategy.get("notes"):
            print("    Notes:")
            for note in strategy["notes"][:3]:
                print(f"      • {note}")
        print()
    
    # Show global health signal
    if report.global_health:
        health = report.global_health
        health_color = {
            "OK": "\033[32m",
            "WARN": "\033[33m",
            "BLOCK": "\033[31m",
        }
        status = health.get("status", "WARN")
        color = health_color.get(status, "")
        reset = "\033[0m" if color else ""
        
        print(f"  Global Health Signal:")
        print(f"    Status:           {color}{status}{reset}")
        print(f"    Coverage OK:      {'Yes' if health.get('snapshot_coverage_ok') else 'No'}")
        print(f"    Max Gap:          {health.get('max_gap', 0)} cycles")
        print(f"    Manifest Issue:   {'Yes' if health.get('has_manifest_mismatch') else 'No'}")
        print(f"    Message:          {health.get('message', 'N/A')}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="PHASE II Snapshot Doctor - Diagnose and verify U2 snapshots",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scripts/snapshot_doctor.py --run-dir results/u2
  python scripts/snapshot_doctor.py --snapshot results/u2/snapshots/snapshot_0100.snap
  python scripts/snapshot_doctor.py --run-dir results/u2 --suggest-recovery
  python scripts/snapshot_doctor.py --run-dir results/u2 --deep --json

Status Codes:
  VALID      - Snapshot is healthy and usable
  CORRUPTED  - Hash mismatch detected (file tampered or damaged)
  INCOMPLETE - File exists but missing critical data
  INVALID    - Readable but fails validation
  UNREADABLE - Cannot parse file format

Exit Codes:
  0 - All snapshots valid (or recommended recovery found)
  1 - Some snapshots have issues
  2 - No snapshots found or no recovery possible
        """
    )
    
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to experiment run directory to scan"
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Path to a specific snapshot to verify"
    )
    parser.add_argument(
        "--suggest-recovery",
        action="store_true",
        help="Analyze snapshots and suggest best recovery strategy"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Perform deep verification (slower but more thorough)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Only recommend snapshots that pass manifest validation (for --suggest-recovery)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.run_dir and not args.snapshot:
        parser.error("Either --run-dir or --snapshot is required")
    
    if args.run_dir and args.snapshot:
        parser.error("Cannot specify both --run-dir and --snapshot")
    
    # Single snapshot verification
    if args.snapshot:
        path = Path(args.snapshot)
        if not path.exists():
            print(f"ERROR: Snapshot not found: {path}", file=sys.stderr)
            sys.exit(2)
        
        result = verify_snapshot(path, deep=args.deep)
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_verification_result(result)
        
        sys.exit(0 if result.status == SnapshotStatus.VALID else 1)
    
    # Run directory scan
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)
    
    if args.suggest_recovery:
        report = suggest_recovery(run_dir, deep=args.deep, strict=args.strict)
    else:
        report = scan_run_dir(run_dir, deep=args.deep)
    
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_scan_report(report)
    
    # Exit code based on health
    if report.total_snapshots == 0:
        sys.exit(2)
    elif report.valid_count == 0 and report.recommended_resume_point is None:
        sys.exit(2)
    elif report.corrupted_count > 0 or report.unreadable_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

