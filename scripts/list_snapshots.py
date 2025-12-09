#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

Snapshot Browser CLI
====================

Lists and inspects U2 experiment snapshots for developer workflows.

Usage:
    # List all snapshots in a run directory
    python scripts/list_snapshots.py --run-dir results/u2

    # JSON output for scripting
    python scripts/list_snapshots.py --run-dir results/u2 --json

    # Show details for a specific snapshot
    python scripts/list_snapshots.py --snapshot results/u2/snapshots/snapshot_0100.snap

    # List only valid snapshots
    python scripts/list_snapshots.py --run-dir results/u2 --valid-only

    # Check snapshot coverage and gaps
    python scripts/list_snapshots.py --run-dir results/u2 --policy-summary

Examples:
    # Find the latest resumable snapshot
    python scripts/list_snapshots.py --run-dir results/u2 --latest

    # Check snapshot health
    python scripts/list_snapshots.py --run-dir results/u2 --health
    
    # Check coverage and gap analysis
    python scripts/list_snapshots.py --run-dir results/u2 --policy-summary

Absolute Safeguards:
    - Read-only: No modifications to snapshots.
    - Advisory: Reports status, does not "fix" anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return ts[:19] if len(ts) >= 19 else ts


def get_snapshot_info(path: Path, verify: bool = True) -> Dict[str, Any]:
    """
    Get detailed information about a snapshot.
    
    Returns:
        Dict with snapshot metadata and status
    """
    info: Dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "status": "UNKNOWN",
        "errors": [],
    }
    
    try:
        # Load snapshot
        snapshot = load_snapshot(path, verify_hash=verify)
        
        # Extract metadata
        info.update({
            "status": "VALID",
            "cycle_index": snapshot.cycle_index,
            "total_cycles": snapshot.total_cycles,
            "progress": f"{snapshot.cycle_index}/{snapshot.total_cycles}" if snapshot.total_cycles > 0 else str(snapshot.cycle_index),
            "progress_pct": round(100 * snapshot.cycle_index / snapshot.total_cycles, 1) if snapshot.total_cycles > 0 else 0,
            "mode": snapshot.mode or "unknown",
            "slice_name": snapshot.slice_name or "unknown",
            "experiment_id": snapshot.experiment_id or "unknown",
            "master_seed": snapshot.master_seed,
            "schema_version": snapshot.schema_version,
            "config_hash": snapshot.config_hash[:12] + "..." if snapshot.config_hash else "N/A",
            "ht_series_hash": snapshot.ht_series_hash[:12] + "..." if snapshot.ht_series_hash else "N/A",
            "ht_series_length": snapshot.ht_series_length,
            "manifest_hash": snapshot.manifest_hash[:12] + "..." if snapshot.manifest_hash else "N/A",
            "snapshot_timestamp": format_timestamp(snapshot.snapshot_timestamp),
            "policy_update_count": snapshot.policy_update_count,
        })
        
        # Determine flags
        flags = []
        if "final" in path.stem.lower():
            flags.append("final")
        
        # Check if this is the latest
        parent_dir = path.parent
        latest = find_latest_snapshot(parent_dir)
        if latest and latest.resolve() == path.resolve():
            flags.append("latest")
        
        if snapshot.cycle_index == 0:
            flags.append("initial")
        
        if snapshot.total_cycles > 0 and snapshot.cycle_index >= snapshot.total_cycles - 1:
            flags.append("complete")
        
        info["flags"] = flags
        
        # Validate
        try:
            validate_snapshot(snapshot)
            info["validation"] = "PASSED"
        except SnapshotValidationError as e:
            info["validation"] = "FAILED"
            info["errors"].append(str(e))
        
    except SnapshotCorruptionError as e:
        info["status"] = "CORRUPTED"
        info["errors"].append(f"Corruption: {e}")
    except SnapshotValidationError as e:
        info["status"] = "INVALID"
        info["errors"].append(f"Validation: {e}")
    except Exception as e:
        info["status"] = "ERROR"
        info["errors"].append(f"Load error: {e}")
    
    return info


def list_run_snapshots(run_dir: Path, valid_only: bool = False) -> List[Dict[str, Any]]:
    """
    List all snapshots in a run directory.
    
    Args:
        run_dir: Path to run directory
        valid_only: If True, only return valid snapshots
        
    Returns:
        List of snapshot info dicts
    """
    # Find snapshot directory
    snapshot_dir = run_dir / "snapshots"
    if not snapshot_dir.exists():
        snapshot_dir = run_dir  # Maybe snapshots are directly in run_dir
    
    # List snapshots
    paths = list_snapshots(snapshot_dir)
    
    if not paths:
        return []
    
    # Get info for each
    results = []
    for path in paths:
        info = get_snapshot_info(path, verify=True)
        
        if valid_only and info["status"] != "VALID":
            continue
        
        results.append(info)
    
    return results


def print_snapshot_table(snapshots: List[Dict[str, Any]]) -> None:
    """Print snapshots in a formatted table."""
    if not snapshots:
        print("No snapshots found.")
        return
    
    # Header
    print()
    print(f"{'Cycle':<12} {'Progress':<10} {'Mode':<10} {'Slice':<20} {'Status':<10} {'Flags':<15} {'File'}")
    print("-" * 100)
    
    for s in snapshots:
        cycle = str(s.get("cycle_index", "?"))
        progress = s.get("progress", "?")
        mode = s.get("mode", "?")[:10]
        slice_name = (s.get("slice_name", "?") or "?")[:20]
        status = s.get("status", "?")
        flags = ",".join(s.get("flags", []))[:15] or "-"
        filename = s.get("filename", "?")
        
        # Color status
        status_display = status
        if status == "VALID":
            status_display = "\033[32mVALID\033[0m"  # Green
        elif status == "CORRUPTED":
            status_display = "\033[31mCORRUPT\033[0m"  # Red
        elif status == "INVALID":
            status_display = "\033[33mINVALID\033[0m"  # Yellow
        
        print(f"{cycle:<12} {progress:<10} {mode:<10} {slice_name:<20} {status_display:<10} {flags:<15} {filename}")
    
    print()


def print_snapshot_detail(info: Dict[str, Any]) -> None:
    """Print detailed information about a single snapshot."""
    print()
    print("=" * 60)
    print(f"Snapshot: {info.get('filename', 'unknown')}")
    print("=" * 60)
    print()
    
    print(f"  Status:           {info.get('status', 'unknown')}")
    print(f"  Path:             {info.get('path', 'unknown')}")
    print(f"  Size:             {info.get('size_bytes', 0):,} bytes")
    print()
    
    print("  Cycle Information:")
    print(f"    Cycle Index:    {info.get('cycle_index', '?')}")
    print(f"    Total Cycles:   {info.get('total_cycles', '?')}")
    print(f"    Progress:       {info.get('progress', '?')} ({info.get('progress_pct', 0)}%)")
    print()
    
    print("  Experiment:")
    print(f"    Experiment ID:  {info.get('experiment_id', 'unknown')}")
    print(f"    Mode:           {info.get('mode', 'unknown')}")
    print(f"    Slice:          {info.get('slice_name', 'unknown')}")
    print(f"    Master Seed:    {info.get('master_seed', '?')}")
    print()
    
    print("  Integrity:")
    print(f"    Schema Version: {info.get('schema_version', '?')}")
    print(f"    Config Hash:    {info.get('config_hash', 'N/A')}")
    print(f"    Manifest Hash:  {info.get('manifest_hash', 'N/A')}")
    print(f"    HT Series Hash: {info.get('ht_series_hash', 'N/A')}")
    print(f"    HT Series Len:  {info.get('ht_series_length', 0)}")
    print(f"    Validation:     {info.get('validation', 'N/A')}")
    print()
    
    print("  State:")
    print(f"    Policy Updates: {info.get('policy_update_count', 0)}")
    print(f"    Timestamp:      {info.get('snapshot_timestamp', 'N/A')}")
    print()
    
    flags = info.get("flags", [])
    if flags:
        print(f"  Flags:            {', '.join(flags)}")
    
    errors = info.get("errors", [])
    if errors:
        print()
        print("  Errors:")
        for err in errors:
            print(f"    - {err}")
    
    print()


def print_health_summary(snapshots: List[Dict[str, Any]]) -> None:
    """Print health summary of all snapshots."""
    if not snapshots:
        print("No snapshots found.")
        return
    
    total = len(snapshots)
    valid = sum(1 for s in snapshots if s.get("status") == "VALID")
    corrupted = sum(1 for s in snapshots if s.get("status") == "CORRUPTED")
    invalid = sum(1 for s in snapshots if s.get("status") == "INVALID")
    errors = sum(1 for s in snapshots if s.get("status") == "ERROR")
    
    print()
    print("=" * 40)
    print("Snapshot Health Summary")
    print("=" * 40)
    print(f"  Total:     {total}")
    print(f"  Valid:     {valid} ({100*valid/total:.1f}%)")
    print(f"  Corrupted: {corrupted}")
    print(f"  Invalid:   {invalid}")
    print(f"  Errors:    {errors}")
    print()
    
    if valid > 0:
        valid_snapshots = [s for s in snapshots if s.get("status") == "VALID"]
        latest_valid = valid_snapshots[0] if valid_snapshots else None
        if latest_valid:
            print(f"  Latest Valid: {latest_valid.get('filename')} (cycle {latest_valid.get('cycle_index')})")
    
    if corrupted > 0 or invalid > 0 or errors > 0:
        print()
        print("  ⚠️  Some snapshots have issues. Run with --json for details.")
    
    print()


def compute_policy_summary(
    snapshots: List[Dict[str, Any]],
    manifest_total_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute coverage and gap metrics for a set of snapshots.
    
    Metrics computed:
    - Total cycles (from manifest or snapshots)
    - Number of snapshots
    - Percent of cycles covered by snapshots
    - Longest gap between valid snapshots
    - Gap distribution
    
    Args:
        snapshots: List of snapshot info dicts
        manifest_total_cycles: Total cycle count from manifest (if available)
        
    Returns:
        Dict with computed metrics
    """
    if not snapshots:
        return {
            "total_cycles": manifest_total_cycles or 0,
            "total_cycles_source": "manifest" if manifest_total_cycles else "unknown",
            "snapshot_count": 0,
            "valid_snapshot_count": 0,
            "coverage_percent": 0.0,
            "coverage_label": "unknown",
            "gaps": [],
            "longest_gap": 0,
            "average_gap": 0.0,
            "has_final_snapshot": False,
            "checkpoint_cycles": [],
            "notes": ["No snapshots found"],
        }
    
    # Extract valid snapshots with cycle indices
    valid_snapshots = [
        s for s in snapshots 
        if s.get("status") == "VALID" and s.get("cycle_index") is not None
    ]
    
    # Sort by cycle index
    valid_snapshots.sort(key=lambda s: s.get("cycle_index", 0))
    
    checkpoint_cycles = [s.get("cycle_index", 0) for s in valid_snapshots]
    
    # Determine total cycles
    total_cycles = manifest_total_cycles
    total_cycles_source = "manifest" if manifest_total_cycles else "unknown"
    
    if total_cycles is None:
        # Try to infer from snapshots
        if valid_snapshots:
            # Use max of total_cycles from snapshots
            max_total = max(s.get("total_cycles", 0) for s in snapshots)
            if max_total > 0:
                total_cycles = max_total
                total_cycles_source = "snapshot_metadata"
            else:
                # Fall back to max cycle_index seen
                max_cycle = max(s.get("cycle_index", 0) for s in valid_snapshots)
                total_cycles = max_cycle + 1
                total_cycles_source = "inferred_approx"
    
    # Compute gaps between consecutive checkpoints
    gaps = []
    if len(checkpoint_cycles) > 1:
        for i in range(1, len(checkpoint_cycles)):
            gap = checkpoint_cycles[i] - checkpoint_cycles[i - 1]
            gaps.append({
                "from_cycle": checkpoint_cycles[i - 1],
                "to_cycle": checkpoint_cycles[i],
                "gap_size": gap,
            })
    
    longest_gap = max((g["gap_size"] for g in gaps), default=0)
    average_gap = sum(g["gap_size"] for g in gaps) / len(gaps) if gaps else 0.0
    
    # Compute coverage
    # Coverage = cycles that could be resumed from / total cycles
    if total_cycles and total_cycles > 0:
        # Each snapshot covers the cycles from its index onwards
        coverage_percent = (len(checkpoint_cycles) / total_cycles) * 100.0
        # This is a rough measure - more sophisticated would be "how many cycles have a nearby checkpoint"
    else:
        coverage_percent = 0.0
    
    # Label the coverage
    if total_cycles_source == "inferred_approx":
        coverage_label = "approx"
    elif total_cycles_source == "manifest":
        coverage_label = "exact"
    else:
        coverage_label = "unknown"
    
    # Check for final snapshot
    has_final = any("final" in s.get("filename", "").lower() for s in snapshots)
    
    # Generate notes
    notes = []
    if total_cycles_source == "inferred_approx":
        notes.append("Total cycles inferred from snapshots (approx)")
    if longest_gap > 20:
        notes.append(f"Warning: Large gap of {longest_gap} cycles between checkpoints")
    if len(valid_snapshots) == 1:
        notes.append("Only one valid checkpoint - limited recovery options")
    if has_final:
        notes.append("Run completed (final snapshot present)")
    
    return {
        "total_cycles": total_cycles or 0,
        "total_cycles_source": total_cycles_source,
        "snapshot_count": len(snapshots),
        "valid_snapshot_count": len(valid_snapshots),
        "coverage_percent": round(coverage_percent, 1),
        "coverage_label": coverage_label,
        "gaps": gaps,
        "longest_gap": longest_gap,
        "average_gap": round(average_gap, 1),
        "has_final_snapshot": has_final,
        "checkpoint_cycles": checkpoint_cycles,
        "notes": notes,
    }


def try_load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Try to load manifest from run directory.
    
    Looks for manifest.json in common locations.
    
    Returns:
        Manifest dict or None if not found
    """
    # Try common locations
    candidates = [
        run_dir / "manifest.json",
        run_dir / "results" / "manifest.json",
        run_dir.parent / "manifest.json",
    ]
    
    for path in candidates:
        if path.exists():
            try:
                import json
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    
    return None


def print_policy_summary(
    summary: Dict[str, Any],
    snapshots: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print policy summary in human-readable format."""
    print()
    print("=" * 60)
    print("Snapshot Policy Summary")
    print("=" * 60)
    print()
    
    # Total cycles
    total = summary.get("total_cycles", 0)
    source = summary.get("total_cycles_source", "unknown")
    source_label = f" ({source})" if source != "manifest" else ""
    print(f"  Total Cycles:         {total}{source_label}")
    
    # Snapshot counts
    print(f"  Snapshot Count:       {summary.get('snapshot_count', 0)}")
    print(f"  Valid Snapshots:      {summary.get('valid_snapshot_count', 0)}")
    print()
    
    # Coverage
    coverage = summary.get("coverage_percent", 0)
    coverage_label = summary.get("coverage_label", "unknown")
    label_suffix = " (approx)" if coverage_label == "approx" else ""
    
    # Coverage bar
    bar_width = 40
    filled = int(bar_width * min(coverage, 100) / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    print(f"  Coverage:             {coverage:.1f}%{label_suffix}")
    print(f"  [{bar}]")
    print()
    
    # Gaps
    longest = summary.get("longest_gap", 0)
    average = summary.get("average_gap", 0)
    
    print(f"  Longest Gap:          {longest} cycles")
    print(f"  Average Gap:          {average:.1f} cycles")
    print()
    
    # Checkpoint locations
    checkpoints = summary.get("checkpoint_cycles", [])
    if checkpoints:
        if len(checkpoints) <= 10:
            checkpoint_str = ", ".join(str(c) for c in checkpoints)
        else:
            # Show first 5, ..., last 5
            first_five = ", ".join(str(c) for c in checkpoints[:5])
            last_five = ", ".join(str(c) for c in checkpoints[-5:])
            checkpoint_str = f"{first_five} ... {last_five}"
        print(f"  Checkpoints:          {checkpoint_str}")
    
    # Gap details (if significant gaps)
    gaps = summary.get("gaps", [])
    significant_gaps = [g for g in gaps if g.get("gap_size", 0) > 10]
    if significant_gaps:
        print()
        print("  Significant Gaps (>10 cycles):")
        for g in significant_gaps[:5]:
            print(f"    Cycle {g['from_cycle']} → {g['to_cycle']}: {g['gap_size']} cycles")
        if len(significant_gaps) > 5:
            print(f"    ... and {len(significant_gaps) - 5} more")
    
    # Final snapshot
    if summary.get("has_final_snapshot"):
        print()
        print("  \033[32m✓ Final snapshot present (run completed)\033[0m")
    
    # Notes
    notes = summary.get("notes", [])
    if notes:
        print()
        print("  Notes:")
        for note in notes:
            if "Warning" in note:
                print(f"    \033[33m⚠ {note}\033[0m")
            else:
                print(f"    • {note}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PHASE II Snapshot Browser - List and inspect U2 experiment snapshots",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scripts/list_snapshots.py --run-dir results/u2
  python scripts/list_snapshots.py --run-dir results/u2 --json
  python scripts/list_snapshots.py --snapshot results/u2/snapshots/snapshot_0100.snap
  python scripts/list_snapshots.py --run-dir results/u2 --latest
  python scripts/list_snapshots.py --run-dir results/u2 --health
        """
    )
    
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to experiment run directory containing snapshots"
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Path to a specific snapshot file to inspect"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format for scripting"
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only show valid snapshots"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show only the latest resumable snapshot"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Show health summary of all snapshots"
    )
    parser.add_argument(
        "--policy-summary",
        action="store_true",
        help="Show coverage and gap analysis for snapshots"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.json (optional, for accurate total_cycles)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.run_dir and not args.snapshot:
        parser.error("Either --run-dir or --snapshot is required")
    
    if args.run_dir and args.snapshot:
        parser.error("Cannot specify both --run-dir and --snapshot")
    
    # Handle single snapshot
    if args.snapshot:
        path = Path(args.snapshot)
        if not path.exists():
            print(f"ERROR: Snapshot not found: {path}", file=sys.stderr)
            sys.exit(1)
        
        info = get_snapshot_info(path, verify=True)
        
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print_snapshot_detail(info)
        
        sys.exit(0 if info["status"] == "VALID" else 1)
    
    # Handle run directory
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Handle --latest
    if args.latest:
        snapshot_dir = run_dir / "snapshots"
        if not snapshot_dir.exists():
            snapshot_dir = run_dir
        
        latest = find_latest_snapshot(snapshot_dir)
        if latest is None:
            print("No resumable snapshot found.", file=sys.stderr)
            sys.exit(2)
        
        info = get_snapshot_info(latest, verify=True)
        
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print_snapshot_detail(info)
        
        sys.exit(0 if info["status"] == "VALID" else 1)
    
    # List all snapshots
    snapshots = list_run_snapshots(run_dir, valid_only=args.valid_only)
    
    # Handle --policy-summary
    if args.policy_summary:
        # Try to load manifest for accurate total_cycles
        manifest_total = None
        if args.manifest:
            manifest_path = Path(args.manifest)
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        manifest_total = manifest.get("cycles") or manifest.get("total_cycles")
                except (json.JSONDecodeError, OSError):
                    pass
        else:
            # Try auto-discovery
            manifest = try_load_manifest(run_dir)
            if manifest:
                manifest_total = manifest.get("cycles") or manifest.get("total_cycles")
        
        summary = compute_policy_summary(snapshots, manifest_total)
        
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print_policy_summary(summary, snapshots)
        
        sys.exit(0 if summary.get("valid_snapshot_count", 0) > 0 else 2)
    
    if args.json:
        print(json.dumps(snapshots, indent=2))
    elif args.health:
        print_health_summary(snapshots)
    else:
        print_snapshot_table(snapshots)
    
    # Exit with status based on snapshot health
    if not snapshots:
        sys.exit(2)
    
    has_valid = any(s.get("status") == "VALID" for s in snapshots)
    sys.exit(0 if has_valid else 1)


if __name__ == "__main__":
    main()

