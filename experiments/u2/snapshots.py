# PHASE II — NOT USED IN PHASE I
# Snapshot management for U2 experiments

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    """Container for snapshot state."""
    cycle_index: int
    ht_series: List[str]
    policy_update_count: int
    success_count: Dict[str, int]
    attempt_count: Dict[str, int]
    experiment_id: str
    slice_name: str
    mode: str
    master_seed: int
    checksum: Optional[str] = None


def _compute_checksum(data: dict) -> str:
    """Compute SHA256 checksum of snapshot data."""
    # Exclude checksum field itself
    data_copy = {k: v for k, v in data.items() if k != "checksum"}
    serialized = json.dumps(data_copy, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def save_snapshot(snapshot: SnapshotData, path: Path) -> None:
    """
    Save snapshot to file with checksum.
    
    Args:
        snapshot: Snapshot data to save
        path: Path to save snapshot file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and compute checksum
    data = asdict(snapshot)
    data["checksum"] = _compute_checksum(data)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_snapshot(path: Path, verify_hash: bool = True) -> SnapshotData:
    """
    Load snapshot from file.
    
    Args:
        path: Path to snapshot file
        verify_hash: If True, verify checksum
        
    Returns:
        Loaded snapshot data
        
    Raises:
        SnapshotValidationError: If checksum verification fails
        SnapshotCorruptionError: If snapshot file is corrupted
    """
    if not path.exists():
        raise NoSnapshotFoundError(f"Snapshot not found: {path}")
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise SnapshotCorruptionError(f"Failed to load snapshot: {e}")
    
    # Verify checksum if requested
    if verify_hash and "checksum" in data:
        stored_checksum = data["checksum"]
        computed_checksum = _compute_checksum(data)
        if stored_checksum != computed_checksum:
            raise SnapshotValidationError(
                f"Checksum mismatch: stored={stored_checksum}, computed={computed_checksum}"
            )
    
    # Remove checksum before creating dataclass
    checksum = data.pop("checksum", None)
    
    try:
        snapshot = SnapshotData(**data)
        snapshot.checksum = checksum
        return snapshot
    except TypeError as e:
        raise SnapshotCorruptionError(f"Invalid snapshot format: {e}")


def find_latest_snapshot(snapshot_dir: Path) -> Optional[Path]:
    """
    Find the most recent snapshot in directory.
    
    Args:
        snapshot_dir: Directory containing snapshots
        
    Returns:
        Path to latest snapshot, or None if no snapshots found
    """
    if not snapshot_dir.exists():
        return None
    
    snapshots = list(snapshot_dir.glob("*.snap"))
    if not snapshots:
        return None
    
    # Sort by modification time
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]


def rotate_snapshots(snapshot_dir: Path, keep_count: int = 5) -> List[Path]:
    """
    Rotate snapshots, keeping only the N most recent.
    
    Args:
        snapshot_dir: Directory containing snapshots
        keep_count: Number of snapshots to keep
        
    Returns:
        List of deleted snapshot paths
    """
    if not snapshot_dir.exists() or keep_count <= 0:
        return []
    
    snapshots = list(snapshot_dir.glob("*.snap"))
    if len(snapshots) <= keep_count:
        return []
    
    # Sort by modification time (newest first)
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Delete old snapshots
    to_delete = snapshots[keep_count:]
    deleted = []
    for path in to_delete:
        try:
            path.unlink()
            deleted.append(path)
        except OSError:
            pass  # Ignore deletion errors
    
    return deleted
