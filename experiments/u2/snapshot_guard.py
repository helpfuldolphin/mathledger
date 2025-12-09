"""
PHASE II — NOT USED IN PHASE I

Snapshot Guard Module
=====================

Provides validation utilities to prevent resuming from a snapshot that
doesn't match the current manifest or experiment configuration.

This is a **guardrail only** - it never auto-mutates or "fixes" snapshots.

Usage:
    from experiments.u2.snapshot_guard import (
        validate_snapshot_against_manifest,
        ValidationResult,
        ValidationStatus,
    )
    
    result = validate_snapshot_against_manifest(snapshot, manifest)
    if result.status == ValidationStatus.OK:
        # Safe to resume
        ...
    elif result.status == ValidationStatus.MISMATCH:
        # Manifest hash doesn't match - may not be compatible
        ...

Constraints:
    - Read-only: Never modifies snapshots or manifests
    - Backwards compatible: Snapshots without manifest_hash treated as "unknown"
    - Advisory: Reports findings, does not block (unless --strict mode)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ValidationStatus(str, Enum):
    """Result status for snapshot validation against manifest."""
    OK = "OK"                       # All checks passed
    MISMATCH = "MISMATCH"           # manifest_hash doesn't match
    UNKNOWN = "UNKNOWN"             # Snapshot predates manifest_hash field
    CYCLE_INVALID = "CYCLE_INVALID" # created_at_cycle exceeds manifest cycles
    MANIFEST_MISSING = "MANIFEST_MISSING"  # No manifest provided for comparison
    ERROR = "ERROR"                 # Validation error occurred


@dataclass
class ValidationResult:
    """
    Result of validating a snapshot against a manifest.
    
    Attributes:
        status: Overall validation status
        snapshot_manifest_hash: manifest_hash from snapshot (may be empty)
        expected_manifest_hash: computed/provided manifest hash for comparison
        snapshot_cycle: cycle index from snapshot
        manifest_total_cycles: total cycles from manifest (if available)
        is_compatible: True if snapshot is safe to resume from
        warnings: List of non-fatal issues
        errors: List of fatal issues
        details: Additional context
    """
    status: ValidationStatus
    snapshot_manifest_hash: str = ""
    expected_manifest_hash: str = ""
    snapshot_cycle: Optional[int] = None
    manifest_total_cycles: Optional[int] = None
    is_compatible: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "expected_manifest_hash": self.expected_manifest_hash,
            "snapshot_cycle": self.snapshot_cycle,
            "manifest_total_cycles": self.manifest_total_cycles,
            "is_compatible": self.is_compatible,
            "warnings": self.warnings,
            "errors": self.errors,
            "details": self.details,
        }
    
    @property
    def status_label(self) -> str:
        """Human-readable status label."""
        labels = {
            ValidationStatus.OK: "Manifest OK",
            ValidationStatus.MISMATCH: "Manifest mismatch",
            ValidationStatus.UNKNOWN: "Manifest unknown (snapshot predates field)",
            ValidationStatus.CYCLE_INVALID: "Cycle exceeds manifest",
            ValidationStatus.MANIFEST_MISSING: "No manifest for comparison",
            ValidationStatus.ERROR: "Validation error",
        }
        return labels.get(self.status, "Unknown status")


def compute_manifest_hash(manifest: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of manifest for comparison.
    
    Args:
        manifest: Manifest dictionary
        
    Returns:
        SHA256 hash (hex) of canonical manifest
    """
    import hashlib
    
    # Canonical JSON serialization
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def validate_snapshot_against_manifest(
    snapshot: "SnapshotData",
    manifest: Optional[Dict[str, Any]] = None,
    manifest_hash: Optional[str] = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate that a snapshot is compatible with a manifest.
    
    Checks performed:
    1. manifest_hash match (if present in snapshot)
    2. created_at_cycle <= manifest total_cycles (if present)
    3. Mode consistency (if present)
    
    Args:
        snapshot: SnapshotData to validate
        manifest: Manifest dict for comparison (optional)
        manifest_hash: Pre-computed manifest hash (optional, overrides manifest)
        strict: If True, treat UNKNOWN as incompatible
        
    Returns:
        ValidationResult with status and details
    """
    from .snapshots import SnapshotData
    
    result = ValidationResult(
        status=ValidationStatus.OK,
        snapshot_cycle=snapshot.cycle_index,
        is_compatible=True,
    )
    
    # Extract snapshot's manifest_hash
    snapshot_hash = getattr(snapshot, 'manifest_hash', '') or ''
    result.snapshot_manifest_hash = snapshot_hash
    
    # If no manifest provided, we can't validate
    if manifest is None and manifest_hash is None:
        result.status = ValidationStatus.MANIFEST_MISSING
        result.is_compatible = True  # Allow resume, but can't verify
        result.warnings.append("No manifest provided - cannot verify compatibility")
        return result
    
    # Compute expected hash if manifest provided
    if manifest is not None:
        expected_hash = compute_manifest_hash(manifest)
        result.expected_manifest_hash = expected_hash
        
        # Extract total_cycles from manifest
        total_cycles = manifest.get('cycles') or manifest.get('total_cycles')
        result.manifest_total_cycles = total_cycles
        
        # Check mode consistency
        manifest_mode = manifest.get('mode', '')
        if snapshot.mode and manifest_mode and snapshot.mode != manifest_mode:
            result.warnings.append(
                f"Mode mismatch: snapshot={snapshot.mode}, manifest={manifest_mode}"
            )
    else:
        expected_hash = manifest_hash or ''
        result.expected_manifest_hash = expected_hash
    
    # Check 1: manifest_hash match
    if not snapshot_hash:
        # Snapshot predates manifest_hash field
        result.status = ValidationStatus.UNKNOWN
        result.is_compatible = not strict
        result.warnings.append(
            "Snapshot has no manifest_hash (predates this field) - "
            "compatibility cannot be verified"
        )
        if strict:
            result.errors.append("Strict mode: rejecting snapshot without manifest_hash")
    elif snapshot_hash != expected_hash:
        # Hash mismatch
        result.status = ValidationStatus.MISMATCH
        result.is_compatible = False
        result.errors.append(
            f"manifest_hash mismatch: "
            f"snapshot={snapshot_hash[:12]}..., "
            f"expected={expected_hash[:12]}..."
        )
        result.details["snapshot_manifest_hash_full"] = snapshot_hash
        result.details["expected_manifest_hash_full"] = expected_hash
    else:
        # Hash match
        result.status = ValidationStatus.OK
        result.is_compatible = True
        result.details["manifest_hash_verified"] = True
    
    # Check 2: cycle bounds (only if we have total_cycles)
    if result.manifest_total_cycles is not None:
        created_at = getattr(snapshot, 'created_at_cycle', None) or snapshot.cycle_index
        
        if created_at > result.manifest_total_cycles:
            # Snapshot cycle exceeds manifest - very suspicious
            result.status = ValidationStatus.CYCLE_INVALID
            result.is_compatible = False
            result.errors.append(
                f"Snapshot cycle ({created_at}) exceeds manifest total ({result.manifest_total_cycles})"
            )
    
    return result


def validate_snapshot_file_against_manifest(
    snapshot_path: Path,
    manifest_path: Optional[Path] = None,
    manifest: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> ValidationResult:
    """
    Convenience function to validate snapshot file against manifest.
    
    Args:
        snapshot_path: Path to snapshot file
        manifest_path: Path to manifest.json (optional)
        manifest: Manifest dict (optional, alternative to path)
        strict: If True, treat UNKNOWN as incompatible
        
    Returns:
        ValidationResult with status and details
    """
    from .snapshots import load_snapshot, SnapshotCorruptionError, SnapshotValidationError
    
    result = ValidationResult(
        status=ValidationStatus.ERROR,
        is_compatible=False,
    )
    
    # Load snapshot
    try:
        snapshot = load_snapshot(snapshot_path, verify_hash=True)
    except (SnapshotCorruptionError, SnapshotValidationError) as e:
        result.errors.append(f"Failed to load snapshot: {e}")
        return result
    except Exception as e:
        result.errors.append(f"Error loading snapshot: {e}")
        return result
    
    result.snapshot_cycle = snapshot.cycle_index
    
    # Load manifest if path provided
    if manifest is None and manifest_path is not None:
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            result.errors.append(f"Failed to load manifest: {e}")
            return result
    
    # Delegate to main validation function
    return validate_snapshot_against_manifest(snapshot, manifest, strict=strict)


def check_resume_compatibility(
    snapshot_path: Path,
    run_dir: Path,
    strict: bool = False,
) -> ValidationResult:
    """
    Check if a snapshot is compatible for resuming an experiment.
    
    This is the recommended entry point for resume workflows.
    Automatically finds and loads manifest from common locations.
    
    Args:
        snapshot_path: Path to snapshot to check
        run_dir: Run directory (used to find manifest)
        strict: If True, reject snapshots without manifest_hash
        
    Returns:
        ValidationResult with status and recommendations
    """
    from .snapshots import load_snapshot
    
    # Find manifest
    manifest = None
    manifest_path = None
    
    for candidate in [
        run_dir / "manifest.json",
        run_dir / "results" / "manifest.json",
        run_dir.parent / "manifest.json",
    ]:
        if candidate.exists():
            try:
                with open(candidate, 'r') as f:
                    manifest = json.load(f)
                manifest_path = candidate
                break
            except (json.JSONDecodeError, OSError):
                continue
    
    result = validate_snapshot_file_against_manifest(
        snapshot_path,
        manifest=manifest,
        strict=strict,
    )
    
    # Add path info to details
    result.details["snapshot_path"] = str(snapshot_path)
    if manifest_path:
        result.details["manifest_path"] = str(manifest_path)
    
    return result


# Export types and functions
__all__ = [
    "ValidationStatus",
    "ValidationResult",
    "compute_manifest_hash",
    "validate_snapshot_against_manifest",
    "validate_snapshot_file_against_manifest",
    "check_resume_compatibility",
]

