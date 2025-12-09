"""
PHASE II — NOT USED IN PHASE I

Snapshot History & Resume Intelligence Module
=============================================

Provides a comprehensive history ledger and intelligent resume strategy
advisory for U2 experiment snapshots.

This module is the central intelligence layer for snapshot management,
synthesizing data from:
- list_snapshots.py (snapshot enumeration)
- snapshot_guard.py (manifest validation)
- snapshot_doctor.py (health verification)

Usage:
    from experiments.u2.snapshot_history import (
        build_snapshot_history,
        advise_resume_strategy,
        summarize_snapshots_for_global_health,
        HistoryStatus,
    )
    
    # Build comprehensive history
    history = build_snapshot_history(run_dir)
    
    # Get resume advice
    advice = advise_resume_strategy(history, strict=False)
    
    # Get global health signal
    health = summarize_snapshots_for_global_health(history)

Absolute Safeguards:
    - Read-only: Never modifies snapshots or manifests
    - Advisory: Provides recommendations, does not enforce
    - Deterministic: Same inputs produce same outputs
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .snapshots import (
    SnapshotData,
    SnapshotValidationError,
    SnapshotCorruptionError,
    load_snapshot,
    list_snapshots,
    find_latest_snapshot,
    validate_snapshot,
)
from .snapshot_guard import (
    ValidationStatus,
    ValidationResult,
    check_resume_compatibility,
    validate_snapshot_file_against_manifest,
)


# Schema version for history format
HISTORY_SCHEMA_VERSION = "1.0"


class HistoryStatus(str, Enum):
    """Overall status of snapshot history."""
    OK = "OK"           # Valid snapshots available, good coverage
    WARN = "WARN"       # Valid snapshots but issues (gaps, manifest mismatch)
    BLOCK = "BLOCK"     # No valid snapshots, cannot resume
    EMPTY = "EMPTY"     # No snapshots found


class ResumeStatus(str, Enum):
    """Status of resume capability."""
    ALLOWED = "ALLOWED"       # Resume is safe
    CONDITIONAL = "CONDITIONAL"  # Resume possible with warnings
    BLOCKED = "BLOCKED"       # Resume not recommended


@dataclass
class SnapshotRecord:
    """Record of a single snapshot in the history."""
    path: str
    cycle_index: int
    total_cycles: int
    status: str  # VALID, CORRUPTED, INVALID, UNKNOWN
    mode: str
    slice_name: str
    experiment_id: str
    manifest_hash: str
    is_final: bool
    size_bytes: int
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "cycle_index": self.cycle_index,
            "total_cycles": self.total_cycles,
            "status": self.status,
            "mode": self.mode,
            "slice_name": self.slice_name,
            "experiment_id": self.experiment_id,
            "manifest_hash": self.manifest_hash,
            "is_final": self.is_final,
            "size_bytes": self.size_bytes,
            "errors": self.errors,
        }


def _get_snapshot_record(path: Path) -> SnapshotRecord:
    """
    Load a snapshot and create a record of its state.
    
    Args:
        path: Path to snapshot file
        
    Returns:
        SnapshotRecord with snapshot metadata and status
    """
    record = SnapshotRecord(
        path=str(path),
        cycle_index=-1,
        total_cycles=0,
        status="UNKNOWN",
        mode="",
        slice_name="",
        experiment_id="",
        manifest_hash="",
        is_final="final" in path.stem.lower(),
        size_bytes=path.stat().st_size if path.exists() else 0,
        errors=[],
    )
    
    if not path.exists():
        record.status = "UNKNOWN"
        record.errors.append(f"File not found: {path}")
        return record
    
    if record.size_bytes == 0:
        record.status = "INVALID"
        record.errors.append("File is empty")
        return record
    
    try:
        snapshot = load_snapshot(path, verify_hash=True)
        record.status = "VALID"
        record.cycle_index = snapshot.cycle_index
        record.total_cycles = snapshot.total_cycles
        record.mode = snapshot.mode or ""
        record.slice_name = snapshot.slice_name or ""
        record.experiment_id = snapshot.experiment_id or ""
        record.manifest_hash = snapshot.manifest_hash or ""
    except SnapshotCorruptionError as e:
        record.status = "CORRUPTED"
        record.errors.append(f"Corruption: {e}")
        
        # Try to extract metadata without hash verification
        try:
            snapshot = load_snapshot(path, verify_hash=False)
            record.cycle_index = snapshot.cycle_index
            record.total_cycles = snapshot.total_cycles
            record.mode = snapshot.mode or ""
            record.slice_name = snapshot.slice_name or ""
            record.experiment_id = snapshot.experiment_id or ""
            record.manifest_hash = snapshot.manifest_hash or ""
        except Exception:
            pass
    except SnapshotValidationError as e:
        record.status = "INVALID"
        record.errors.append(f"Validation: {e}")
    except Exception as e:
        record.status = "UNKNOWN"
        record.errors.append(f"Load error: {e}")
    
    return record


def _try_load_manifest(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """
    Try to load manifest from run directory.
    
    Returns:
        Tuple of (manifest_dict, manifest_path) or (None, None) if not found
    """
    candidates = [
        run_dir / "manifest.json",
        run_dir / "results" / "manifest.json",
        run_dir.parent / "manifest.json",
    ]
    
    for path in candidates:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f), path
            except (json.JSONDecodeError, OSError):
                continue
    
    return None, None


def build_snapshot_history(
    run_dir: Path,
    include_manifest_validation: bool = True,
) -> Dict[str, Any]:
    """
    Build a comprehensive snapshot history ledger for a run directory.
    
    This is the central function that synthesizes all snapshot information
    into a single, queryable history structure.
    
    Args:
        run_dir: Path to experiment run directory
        include_manifest_validation: If True, validate against manifest
        
    Returns:
        Dict with:
            - schema_version: History format version
            - run_dir: Source directory
            - manifest_found: Whether manifest was located
            - manifest_total_cycles: From manifest if available
            - snapshots: List of all discovered snapshots sorted by cycle
            - valid_snapshots: List of valid snapshot records
            - corrupted_snapshots: List of corrupted snapshot records
            - unknown_snapshots: List of unknown/invalid snapshot records
            - coverage_pct: Percentage of cycles with checkpoints
            - max_gap: Largest gap between consecutive checkpoints
            - avg_gap: Average gap between checkpoints
            - recommended_resume_point: Dict with cycle and path
            - status: Overall history status (OK/WARN/BLOCK/EMPTY)
    """
    run_dir = Path(run_dir)
    
    # Find snapshot directory
    snapshot_dir = run_dir / "snapshots"
    if not snapshot_dir.exists():
        snapshot_dir = run_dir
    
    # Try to load manifest
    manifest, manifest_path = _try_load_manifest(run_dir)
    manifest_total_cycles = None
    if manifest:
        manifest_total_cycles = manifest.get("cycles") or manifest.get("total_cycles")
    
    # Discover and analyze all snapshots
    paths = list_snapshots(snapshot_dir)
    records: List[SnapshotRecord] = []
    
    for path in paths:
        record = _get_snapshot_record(path)
        records.append(record)
    
    # Sort by cycle index (ascending)
    records.sort(key=lambda r: r.cycle_index if r.cycle_index >= 0 else float('inf'))
    
    # Categorize snapshots
    valid_snapshots = [r for r in records if r.status == "VALID"]
    corrupted_snapshots = [r for r in records if r.status == "CORRUPTED"]
    unknown_snapshots = [r for r in records if r.status in ("UNKNOWN", "INVALID")]
    
    # Get valid cycle indices
    valid_cycles = sorted([r.cycle_index for r in valid_snapshots if r.cycle_index >= 0])
    
    # Determine total cycles
    total_cycles = manifest_total_cycles
    total_cycles_source = "manifest" if manifest_total_cycles else "unknown"
    
    if total_cycles is None and valid_snapshots:
        # Try from snapshot metadata
        max_total = max((r.total_cycles for r in valid_snapshots), default=0)
        if max_total > 0:
            total_cycles = max_total
            total_cycles_source = "snapshot_metadata"
        elif valid_cycles:
            total_cycles = max(valid_cycles) + 1
            total_cycles_source = "inferred_approx"
    
    # Compute gaps
    gaps = []
    if len(valid_cycles) > 1:
        for i in range(1, len(valid_cycles)):
            gap = valid_cycles[i] - valid_cycles[i - 1]
            gaps.append({
                "from_cycle": valid_cycles[i - 1],
                "to_cycle": valid_cycles[i],
                "gap_size": gap,
            })
    
    max_gap = max((g["gap_size"] for g in gaps), default=0)
    avg_gap = sum(g["gap_size"] for g in gaps) / len(gaps) if gaps else 0.0
    
    # Compute coverage
    if total_cycles and total_cycles > 0:
        coverage_pct = round((len(valid_cycles) / total_cycles) * 100.0, 1)
    else:
        coverage_pct = 0.0
    
    # Find recommended resume point (latest valid, non-final)
    recommended_resume_point: Dict[str, Any] = {
        "cycle": None,
        "path": None,
        "manifest_status": None,
    }
    
    non_final_valid = [r for r in valid_snapshots if not r.is_final]
    if non_final_valid:
        # Sort by cycle descending and take the latest
        non_final_valid.sort(key=lambda r: r.cycle_index, reverse=True)
        best = non_final_valid[0]
        recommended_resume_point["cycle"] = best.cycle_index
        recommended_resume_point["path"] = best.path
        
        # Validate against manifest if requested
        if include_manifest_validation and manifest:
            validation = validate_snapshot_file_against_manifest(
                Path(best.path),
                manifest=manifest,
            )
            recommended_resume_point["manifest_status"] = validation.status.value
    
    # Determine overall status
    if not records:
        status = HistoryStatus.EMPTY
    elif not valid_snapshots:
        status = HistoryStatus.BLOCK
    elif (
        corrupted_snapshots or 
        max_gap > 20 or 
        recommended_resume_point.get("manifest_status") in ("MISMATCH", "CYCLE_INVALID")
    ):
        status = HistoryStatus.WARN
    else:
        status = HistoryStatus.OK
    
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "snapshot_dir": str(snapshot_dir),
        "manifest_found": manifest is not None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_total_cycles": manifest_total_cycles,
        "total_cycles": total_cycles,
        "total_cycles_source": total_cycles_source,
        "snapshots": [r.to_dict() for r in records],
        "valid_snapshots": [r.to_dict() for r in valid_snapshots],
        "corrupted_snapshots": [r.to_dict() for r in corrupted_snapshots],
        "unknown_snapshots": [r.to_dict() for r in unknown_snapshots],
        "valid_count": len(valid_snapshots),
        "corrupted_count": len(corrupted_snapshots),
        "unknown_count": len(unknown_snapshots),
        "coverage_pct": coverage_pct,
        "max_gap": max_gap,
        "avg_gap": round(avg_gap, 1),
        "gaps": gaps,
        "checkpoint_cycles": valid_cycles,
        "recommended_resume_point": recommended_resume_point,
        "status": status.value,
    }


def advise_resume_strategy(
    history: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Provide intelligent resume strategy advice based on snapshot history.
    
    This function analyzes the history ledger and provides actionable
    advice on whether and how to resume an experiment.
    
    Args:
        history: History dict from build_snapshot_history()
        strict: If True, require manifest validation to pass
        
    Returns:
        Dict with:
            - resume_allowed: Whether resume is recommended (bool)
            - resume_status: Status enum value (ALLOWED/CONDITIONAL/BLOCKED)
            - resume_from: Recommended snapshot path (or None)
            - resume_cycle: Recommended cycle index (or None)
            - warnings: List of text hints about the resume
            - notes: Additional context information
    """
    warnings: List[str] = []
    notes: List[str] = []
    
    status = history.get("status", "EMPTY")
    valid_count = history.get("valid_count", 0)
    corrupted_count = history.get("corrupted_count", 0)
    unknown_count = history.get("unknown_count", 0)
    max_gap = history.get("max_gap", 0)
    coverage_pct = history.get("coverage_pct", 0)
    recommended = history.get("recommended_resume_point", {})
    manifest_found = history.get("manifest_found", False)
    
    # Default response
    result: Dict[str, Any] = {
        "resume_allowed": False,
        "resume_status": ResumeStatus.BLOCKED.value,
        "resume_from": None,
        "resume_cycle": None,
        "warnings": warnings,
        "notes": notes,
    }
    
    # Case: No snapshots
    if status == "EMPTY":
        result["resume_status"] = ResumeStatus.BLOCKED.value
        warnings.append("No snapshots found - cannot resume")
        notes.append("Start a fresh run with --snapshot-interval to enable future recovery")
        return result
    
    # Case: No valid snapshots
    if status == "BLOCK" or valid_count == 0:
        result["resume_status"] = ResumeStatus.BLOCKED.value
        warnings.append("No valid snapshots available")
        if corrupted_count > 0:
            warnings.append(f"Found {corrupted_count} corrupted snapshot(s)")
        notes.append("Consider starting fresh or investigating corrupted files")
        return result
    
    # We have valid snapshots - analyze quality
    resume_path = recommended.get("path")
    resume_cycle = recommended.get("cycle")
    manifest_status = recommended.get("manifest_status")
    
    if resume_path is None:
        result["resume_status"] = ResumeStatus.BLOCKED.value
        warnings.append("No suitable resume point found")
        return result
    
    # Check manifest validation in strict mode
    if strict and manifest_status not in (None, "OK", "MANIFEST_MISSING"):
        result["resume_status"] = ResumeStatus.BLOCKED.value
        warnings.append(f"Strict mode: manifest validation failed ({manifest_status})")
        
        # Try to find an alternative
        valid_snapshots = history.get("valid_snapshots", [])
        for snap in reversed(valid_snapshots):  # Try from newest
            if snap.get("is_final"):
                continue
            # Would need to re-validate each one
            warnings.append("Consider re-running without --strict to allow resume")
            break
        
        return result
    
    # Resume is allowed - determine if conditional
    result["resume_allowed"] = True
    result["resume_from"] = resume_path
    result["resume_cycle"] = resume_cycle
    
    # Build warnings based on conditions
    conditional = False
    
    if corrupted_count > 0:
        conditional = True
        warnings.append(f"Found {corrupted_count} corrupted snapshot(s) in history")
    
    if max_gap > 20:
        conditional = True
        warnings.append(f"Large gap detected: {max_gap} cycles between checkpoints")
    
    if manifest_status == "MISMATCH":
        conditional = True
        warnings.append("Manifest hash mismatch - configuration may have changed")
    elif manifest_status == "UNKNOWN":
        warnings.append("Snapshot predates manifest_hash field - compatibility unverified")
    elif manifest_status == "MANIFEST_MISSING":
        warnings.append("No manifest found - cannot verify configuration compatibility")
    
    if not manifest_found:
        notes.append("Consider adding manifest.json for better validation")
    
    if coverage_pct < 10 and valid_count < 5:
        conditional = True
        warnings.append(f"Low snapshot coverage ({coverage_pct}%) - limited recovery options")
    
    # Set final status
    if conditional:
        result["resume_status"] = ResumeStatus.CONDITIONAL.value
    else:
        result["resume_status"] = ResumeStatus.ALLOWED.value
    
    # Add helpful notes
    total_cycles = history.get("total_cycles")
    if resume_cycle and total_cycles:
        progress = round(100 * resume_cycle / total_cycles, 1)
        notes.append(f"Resume point: cycle {resume_cycle}/{total_cycles} ({progress}% complete)")
    
    notes.append(f"Command: python experiments/run_uplift_u2.py --restore-from {resume_path}")
    
    return result


def summarize_snapshots_for_global_health(
    history: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a simple health signal for the global health dashboard.
    
    This provides a quick summary suitable for integration with
    broader health monitoring systems.
    
    Args:
        history: History dict from build_snapshot_history()
        
    Returns:
        Dict with:
            - snapshot_coverage_ok: Whether coverage is acceptable (bool)
            - max_gap: Largest gap between checkpoints
            - has_manifest_mismatch: Whether manifest validation failed (bool)
            - status: OK|WARN|BLOCK
            - message: Human-readable status message
    """
    status = history.get("status", "EMPTY")
    valid_count = history.get("valid_count", 0)
    corrupted_count = history.get("corrupted_count", 0)
    max_gap = history.get("max_gap", 0)
    coverage_pct = history.get("coverage_pct", 0)
    recommended = history.get("recommended_resume_point", {})
    manifest_status = recommended.get("manifest_status")
    
    # Determine coverage acceptability
    # Coverage is OK if we have valid snapshots with reasonable coverage
    snapshot_coverage_ok = (
        valid_count > 0 and 
        (coverage_pct >= 5 or valid_count >= 3)
    )
    
    # Check for manifest issues
    has_manifest_mismatch = manifest_status in ("MISMATCH", "CYCLE_INVALID")
    
    # Build message
    if status == "EMPTY":
        message = "No snapshots found"
        health_status = "BLOCK"
    elif status == "BLOCK":
        message = f"No valid snapshots ({corrupted_count} corrupted)"
        health_status = "BLOCK"
    elif has_manifest_mismatch:
        message = f"Manifest mismatch detected ({valid_count} valid snapshots)"
        health_status = "WARN"
    elif max_gap > 50:
        message = f"Large checkpoint gap ({max_gap} cycles)"
        health_status = "WARN"
    elif corrupted_count > 0:
        message = f"{valid_count} valid, {corrupted_count} corrupted snapshots"
        health_status = "WARN"
    elif valid_count > 0:
        message = f"{valid_count} valid snapshots, {coverage_pct}% coverage"
        health_status = "OK"
    else:
        message = "Unknown snapshot state"
        health_status = "WARN"
    
    return {
        "snapshot_coverage_ok": snapshot_coverage_ok,
        "valid_count": valid_count,
        "corrupted_count": corrupted_count,
        "max_gap": max_gap,
        "coverage_pct": coverage_pct,
        "has_manifest_mismatch": has_manifest_mismatch,
        "status": health_status,
        "message": message,
    }


def build_multi_run_snapshot_history(
    run_dirs: Sequence[str],
    include_manifest_validation: bool = True,
) -> Dict[str, Any]:
    """
    Build a comprehensive snapshot history across multiple experiment runs.
    
    This aggregates snapshot history from multiple run directories to provide
    a global view of snapshot health and resume capabilities across all runs.
    
    Args:
        run_dirs: Sequence of run directory paths (strings or Paths)
        include_manifest_validation: If True, validate against manifests
        
    Returns:
        Dict with:
            - schema_version: Multi-run history format version
            - run_count: Number of runs analyzed
            - runs: List of per-run history summaries
            - runs_with_block_status: Count of runs with BLOCK status
            - global_max_gap: Maximum gap across all runs
            - overall_status: OK|WARN|BLOCK
            - summary: Aggregated statistics
    """
    MULTI_RUN_SCHEMA_VERSION = "1.0"
    
    run_dirs_list = [Path(d) for d in run_dirs]
    runs: List[Dict[str, Any]] = []
    
    status_counts = {
        "OK": 0,
        "WARN": 0,
        "BLOCK": 0,
        "EMPTY": 0,
    }
    
    global_max_gap = 0
    runs_with_block_status = 0
    
    for run_dir in run_dirs_list:
        if not run_dir.exists():
            continue
        
        # Build history for this run
        try:
            history = build_snapshot_history(run_dir, include_manifest_validation)
        except Exception:
            # Skip runs that fail to analyze
            continue
        
        # Extract run identifier (use directory name or experiment_id)
        run_id = run_dir.name
        if history.get("snapshots"):
            # Try to get experiment_id from first snapshot
            first_snap = history["snapshots"][0] if history["snapshots"] else {}
            run_id = first_snap.get("experiment_id", run_id)
        
        # Create run summary
        run_summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": history.get("status", "EMPTY"),
            "valid_count": history.get("valid_count", 0),
            "corrupted_count": history.get("corrupted_count", 0),
            "coverage_pct": history.get("coverage_pct", 0.0),
            "max_gap": history.get("max_gap", 0),
            "avg_gap": history.get("avg_gap", 0.0),
            "recommended_resume_point": history.get("recommended_resume_point", {}),
            "checkpoint_cycles": history.get("checkpoint_cycles", []),
        }
        
        runs.append(run_summary)
        
        # Update aggregates
        status = history.get("status", "EMPTY")
        status_counts[status] = status_counts.get(status, 0) + 1
        
        if status == "BLOCK":
            runs_with_block_status += 1
        
        run_max_gap = history.get("max_gap", 0)
        if run_max_gap > global_max_gap:
            global_max_gap = run_max_gap
    
    # Determine overall status
    if runs_with_block_status > 0:
        overall_status = "BLOCK"
    elif status_counts.get("WARN", 0) > 0 or global_max_gap > 50:
        overall_status = "WARN"
    elif len(runs) == 0:
        overall_status = "BLOCK"
    else:
        overall_status = "OK"
    
    # Build summary statistics
    total_valid = sum(r.get("valid_count", 0) for r in runs)
    total_corrupted = sum(r.get("corrupted_count", 0) for r in runs)
    avg_coverage = sum(r.get("coverage_pct", 0.0) for r in runs) / len(runs) if runs else 0.0
    
    return {
        "schema_version": MULTI_RUN_SCHEMA_VERSION,
        "run_count": len(runs),
        "runs": runs,
        "runs_with_block_status": runs_with_block_status,
        "global_max_gap": global_max_gap,
        "overall_status": overall_status,
        "status_counts": status_counts,
        "summary": {
            "total_valid_snapshots": total_valid,
            "total_corrupted_snapshots": total_corrupted,
            "average_coverage_pct": round(avg_coverage, 1),
            "runs_with_resume_points": sum(
                1 for r in runs 
                if r.get("recommended_resume_point", {}).get("path")
            ),
        },
    }


def plan_future_runs(
    multi_history: Dict[str, Any],
    target_coverage: float = 10.0,
) -> Dict[str, Any]:
    """
    Analyze multi-run snapshot history and provide run planning advice.
    
    This function identifies which runs should be extended (resumed) and
    whether new runs should be created based on coverage targets.
    
    Args:
        multi_history: Multi-run history from build_multi_run_snapshot_history()
        target_coverage: Target coverage percentage (default: 10.0)
        
    Returns:
        Dict with:
            - runs_to_extend: List of runs recommended for resuming
            - suggested_new_runs: Count or hints for new runs
            - message: Human-readable summary
            - priority: List of runs sorted by priority
    """
    runs = multi_history.get("runs", [])
    overall_status = multi_history.get("overall_status", "BLOCK")
    
    runs_to_extend: List[Dict[str, Any]] = []
    suggested_new_runs = 0
    message_parts: List[str] = []
    
    # Analyze each run for extension potential
    for run in runs:
        status = run.get("status", "EMPTY")
        coverage_pct = run.get("coverage_pct", 0.0)
        max_gap = run.get("max_gap", 0)
        resume_point = run.get("recommended_resume_point", {})
        resume_path = resume_point.get("path")
        
        # Criteria for extending a run:
        # 1. Has valid snapshots (not BLOCK or EMPTY)
        # 2. Coverage below target OR has large gaps
        # 3. Has a valid resume point
        
        if status in ("BLOCK", "EMPTY"):
            continue
        
        if not resume_path:
            continue
        
        # Calculate priority score (higher = more beneficial to extend)
        priority_score = 0
        
        # Higher priority if coverage is low
        if coverage_pct < target_coverage:
            priority_score += (target_coverage - coverage_pct) * 10
        
        # Higher priority if there are large gaps
        if max_gap > 20:
            priority_score += max_gap
        
        # Higher priority if close to completion (but not final)
        resume_cycle = resume_point.get("cycle")
        total_cycles = resume_point.get("total_cycles")
        if resume_cycle and total_cycles:
            progress = resume_cycle / total_cycles
            if 0.3 < progress < 0.9:  # Sweet spot: not too early, not too late
                priority_score += 20
        
        if priority_score > 0:
            runs_to_extend.append({
                "run_id": run.get("run_id", "unknown"),
                "run_dir": run.get("run_dir", ""),
                "priority_score": round(priority_score, 1),
                "current_coverage": coverage_pct,
                "max_gap": max_gap,
                "resume_point": resume_point,
                "reason": _generate_extension_reason(run, target_coverage),
            })
    
    # Sort by priority (highest first)
    runs_to_extend.sort(key=lambda r: r["priority_score"], reverse=True)
    
    # Determine if new runs are needed
    avg_coverage = multi_history.get("summary", {}).get("average_coverage_pct", 0.0)
    runs_with_resume = multi_history.get("summary", {}).get("runs_with_resume_points", 0)
    
    if overall_status == "BLOCK" and runs_with_resume == 0:
        suggested_new_runs = 3  # Start fresh if nothing is resumable
        message_parts.append("No resumable runs found - suggest starting 3 new runs")
    elif avg_coverage < target_coverage / 2:
        suggested_new_runs = 2
        message_parts.append(f"Low average coverage ({avg_coverage:.1f}%) - suggest 2 new runs")
    elif len(runs_to_extend) == 0:
        suggested_new_runs = 1
        message_parts.append("No runs identified for extension - suggest 1 new run")
    else:
        message_parts.append(f"Found {len(runs_to_extend)} run(s) suitable for extension")
    
    # Build final message
    if not message_parts:
        message = "Snapshot coverage appears adequate"
    else:
        message = ". ".join(message_parts)
    
    return {
        "runs_to_extend": runs_to_extend,
        "suggested_new_runs": suggested_new_runs,
        "message": message,
        "priority": runs_to_extend[:5],  # Top 5 by priority
        "target_coverage": target_coverage,
        "current_avg_coverage": round(avg_coverage, 1),
    }


def _generate_extension_reason(run: Dict[str, Any], target_coverage: float) -> str:
    """Generate a human-readable reason for extending a run."""
    coverage = run.get("coverage_pct", 0.0)
    max_gap = run.get("max_gap", 0)
    
    reasons = []
    
    if coverage < target_coverage:
        reasons.append(f"low coverage ({coverage:.1f}%)")
    
    if max_gap > 20:
        reasons.append(f"large gap ({max_gap} cycles)")
    
    if not reasons:
        reasons.append("resumable checkpoint available")
    
    return ", ".join(reasons)


def summarize_snapshot_plans_for_u2_orchestrator(
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert run planning output into a format consumable by U2 orchestrator.
    
    This provides a simplified interface for orchestrator agents (A4/A5/A6)
    to determine what action to take next.
    
    Args:
        plan: Plan dict from plan_future_runs()
        
    Returns:
        Dict with:
            - has_resume_targets: Whether any runs can be resumed
            - preferred_run_id: ID of highest-priority run to resume
            - preferred_snapshot_path: Path to recommended snapshot
            - status: NO_ACTION | RESUME | NEW_RUN
            - details: Additional context
    """
    runs_to_extend = plan.get("runs_to_extend", [])
    suggested_new_runs = plan.get("suggested_new_runs", 0)
    
    has_resume_targets = len(runs_to_extend) > 0
    
    preferred_run_id = None
    preferred_snapshot_path = None
    
    if runs_to_extend:
        top_priority = runs_to_extend[0]
        preferred_run_id = top_priority.get("run_id")
        resume_point = top_priority.get("resume_point", {})
        preferred_snapshot_path = resume_point.get("path")
    
    # Determine status
    if has_resume_targets and preferred_snapshot_path:
        status = "RESUME"
    elif suggested_new_runs > 0:
        status = "NEW_RUN"
    else:
        status = "NO_ACTION"
    
    return {
        "has_resume_targets": has_resume_targets,
        "preferred_run_id": preferred_run_id,
        "preferred_snapshot_path": preferred_snapshot_path,
        "status": status,
        "details": {
            "runs_available": len(runs_to_extend),
            "suggested_new_runs": suggested_new_runs,
            "message": plan.get("message", ""),
        },
    }


# Export public API
__all__ = [
    "HISTORY_SCHEMA_VERSION",
    "HistoryStatus",
    "ResumeStatus",
    "SnapshotRecord",
    "build_snapshot_history",
    "advise_resume_strategy",
    "summarize_snapshots_for_global_health",
    "build_multi_run_snapshot_history",
    "plan_future_runs",
    "summarize_snapshot_plans_for_u2_orchestrator",
]

