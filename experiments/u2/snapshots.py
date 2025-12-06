"""
U2 Snapshots - Deterministic State Capture

Provides snapshot save/load/validation for deterministic pause/resume
of U2 experiments.
"""

import json
import hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


class SnapshotValidationError(Exception):
    """Raised when snapshot validation fails."""
    pass


class SnapshotCorruptionError(Exception):
    """Raised when snapshot data is corrupted."""
    pass


class NoSnapshotFoundError(Exception):
    """Raised when no snapshot is found."""
    pass


@dataclass
class SnapshotData:
    """
    Container for U2 runner state snapshot.
    
    All fields must be JSON-serializable for persistence.
    """
    cycle_index: int
    ht_series: List[str]
    policy_update_count: int
    success_count: Dict[str, int]
    attempt_count: Dict[str, int]
    weights: Dict[str, float]
    config: Dict[str, Any]
    
    # Metadata
    snapshot_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotData":
        """Create from dictionary."""
        # Extract known fields
        return cls(
            cycle_index=data["cycle_index"],
            ht_series=data["ht_series"],
            policy_update_count=data["policy_update_count"],
            success_count=data["success_count"],
            attempt_count=data["attempt_count"],
            weights=data["weights"],
            config=data["config"],
            snapshot_version=data.get("snapshot_version", "1.0.0"),
        )


def compute_snapshot_hash(snapshot_data: Dict[str, Any]) -> str:
    """
    Compute SHA256 hash of snapshot data.
    
    Args:
        snapshot_data: Dictionary representation of snapshot
        
    Returns:
        Hex string of SHA256 hash
    """
    # Sort keys for deterministic hashing
    canonical_json = json.dumps(snapshot_data, sort_keys=True)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def save_snapshot(snapshot: SnapshotData, path: Path, compute_hash: bool = True) -> str:
    """
    Save snapshot to file with optional hash.
    
    Args:
        snapshot: SnapshotData to save
        path: Path to save to
        compute_hash: Whether to compute and store hash
        
    Returns:
        Hash of snapshot data
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    snapshot_dict = snapshot.to_dict()
    
    # Compute hash if requested
    snapshot_hash = ""
    if compute_hash:
        snapshot_hash = compute_snapshot_hash(snapshot_dict)
        snapshot_dict["_hash"] = snapshot_hash
    
    # Write to file
    with open(path, "w") as f:
        json.dump(snapshot_dict, f, indent=2)
    
    return snapshot_hash


def load_snapshot(path: Path, verify_hash: bool = True) -> SnapshotData:
    """
    Load snapshot from file with optional hash verification.
    
    Args:
        path: Path to snapshot file
        verify_hash: Whether to verify stored hash
        
    Returns:
        Loaded SnapshotData
        
    Raises:
        SnapshotValidationError: If hash verification fails
        SnapshotCorruptionError: If file is corrupted
        NoSnapshotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise NoSnapshotFoundError(f"Snapshot not found: {path}")
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise SnapshotCorruptionError(f"Failed to parse snapshot: {e}")
    except Exception as e:
        raise SnapshotCorruptionError(f"Failed to read snapshot: {e}")
    
    # Verify hash if present and requested
    if verify_hash and "_hash" in data:
        stored_hash = data.pop("_hash")
        computed_hash = compute_snapshot_hash(data)
        
        if stored_hash != computed_hash:
            raise SnapshotValidationError(
                f"Hash mismatch: stored={stored_hash}, computed={computed_hash}"
            )
    
    # Create SnapshotData from dict
    try:
        snapshot = SnapshotData.from_dict(data)
    except Exception as e:
        raise SnapshotCorruptionError(f"Failed to create snapshot from data: {e}")
    
    return snapshot


def find_latest_snapshot(snapshot_dir: Path) -> Optional[Path]:
    """
    Find the most recent snapshot in a directory.
    
    Args:
        snapshot_dir: Directory to search
        
    Returns:
        Path to latest snapshot, or None if no snapshots found
    """
    if not snapshot_dir.exists():
        return None
    
    # Find all .snap files
    snapshots = list(snapshot_dir.glob("*.snap"))
    
    if not snapshots:
        return None
    
    # Sort by modification time (most recent first)
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return snapshots[0]


def rotate_snapshots(snapshot_dir: Path, keep_count: int = 5) -> List[Path]:
    """
    Rotate snapshots, keeping only the most recent N.
    
    Args:
        snapshot_dir: Directory containing snapshots
        keep_count: Number of snapshots to keep (0 = no rotation)
        
    Returns:
        List of deleted snapshot paths
    """
    if keep_count <= 0:
        return []
    
    if not snapshot_dir.exists():
        return []
    
    # Find all .snap files
    snapshots = list(snapshot_dir.glob("*.snap"))
    
    if len(snapshots) <= keep_count:
        return []
    
    # Sort by modification time (oldest first for deletion)
    snapshots.sort(key=lambda p: p.stat().st_mtime)
    
    # Delete oldest snapshots
    to_delete = snapshots[:len(snapshots) - keep_count]
    deleted = []
    
    for snapshot_path in to_delete:
        try:
            snapshot_path.unlink()
            deleted.append(snapshot_path)
        except Exception:
            # Ignore errors during deletion
            pass
    
    return deleted
