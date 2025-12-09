"""
U2 Planner Snapshot System

Enables:
- Checkpointing during long experiments
- Resume from failure
- Deterministic replay from any cycle
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class SnapshotError(Exception):
    """Base exception for snapshot operations."""
    pass


class SnapshotValidationError(SnapshotError):
    """Snapshot failed validation."""
    pass


class SnapshotCorruptionError(SnapshotError):
    """Snapshot file is corrupted."""
    pass


class NoSnapshotFoundError(SnapshotError):
    """No snapshot found at specified location."""
    pass


@dataclass
class SnapshotData:
    """
    Complete snapshot of U2 planner state.
    
    INVARIANTS:
    - Contains all state needed to resume execution
    - Includes hash for integrity verification
    - Serializable to JSON
    """
    
    # Experiment metadata
    experiment_id: str
    slice_name: str
    mode: str
    master_seed: str
    
    # Execution state
    current_cycle: int
    total_cycles: int
    
    # Planner state
    frontier_state: Dict[str, Any] = field(default_factory=dict)
    prng_state: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Safety context (Neural Link)
    safety_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    snapshot_cycle: int = 0
    snapshot_timestamp: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "master_seed": self.master_seed,
            "current_cycle": self.current_cycle,
            "total_cycles": self.total_cycles,
            "frontier_state": self.frontier_state,
            "prng_state": self.prng_state,
            "stats": self.stats,
            "safety_context": self.safety_context,
            "snapshot_cycle": self.snapshot_cycle,
            "snapshot_timestamp": self.snapshot_timestamp,
        }
    
    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert to canonical dict for hashing (excludes timestamp)."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "master_seed": self.master_seed,
            "current_cycle": self.current_cycle,
            "total_cycles": self.total_cycles,
            "frontier_state": self.frontier_state,
            "prng_state": self.prng_state,
            "stats": self.stats,
            "safety_context": self.safety_context,
            "snapshot_cycle": self.snapshot_cycle,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnapshotData':
        """Reconstruct from dict."""
        return cls(
            experiment_id=data["experiment_id"],
            slice_name=data["slice_name"],
            mode=data["mode"],
            master_seed=data["master_seed"],
            current_cycle=data["current_cycle"],
            total_cycles=data["total_cycles"],
            frontier_state=data.get("frontier_state", {}),
            prng_state=data.get("prng_state", {}),
            stats=data.get("stats", {}),
            safety_context=data.get("safety_context", {}),
            snapshot_cycle=data.get("snapshot_cycle", 0),
            snapshot_timestamp=data.get("snapshot_timestamp", 0),
        )
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of snapshot data.
        
        Returns:
            Hex digest of snapshot hash
        """
        # Create canonical JSON representation (excludes timestamp for stability)
        canonical = json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def save_snapshot(
    snapshot: SnapshotData,
    output_path: Path,
    verify: bool = False,
) -> str:
    """
    Save snapshot to disk with integrity hash.
    
    Args:
        snapshot: Snapshot data to save
        output_path: Path to save snapshot
        verify: Verify snapshot after writing
        
    Returns:
        SHA-256 hash of snapshot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute hash
    snapshot_hash = snapshot.compute_hash()
    
    # Create snapshot file with hash
    snapshot_file = {
        "version": "1.0",
        "hash": snapshot_hash,
        "data": snapshot.to_dict(),
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot_file, f, indent=2, sort_keys=True)
    
    # Verify if requested (disabled by default to avoid circular hash issues)
    if verify:
        loaded = load_snapshot(output_path, verify_hash=False)
        if loaded.compute_hash() != snapshot_hash:
            raise SnapshotCorruptionError(f"Snapshot verification failed: {output_path}")
    
    return snapshot_hash


def load_snapshot(
    snapshot_path: Path,
    verify_hash: bool = True,
) -> SnapshotData:
    """
    Load snapshot from disk.
    
    Args:
        snapshot_path: Path to snapshot file
        verify_hash: Verify hash integrity
        
    Returns:
        SnapshotData object
        
    Raises:
        NoSnapshotFoundError: Snapshot file not found
        SnapshotCorruptionError: Hash verification failed
    """
    if not snapshot_path.exists():
        raise NoSnapshotFoundError(f"Snapshot not found: {snapshot_path}")
    
    try:
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            snapshot_file = json.load(f)
    except json.JSONDecodeError as e:
        raise SnapshotCorruptionError(f"Invalid JSON in snapshot: {e}")
    
    # Extract data
    stored_hash = snapshot_file.get("hash")
    data = snapshot_file.get("data")
    
    if not data:
        raise SnapshotCorruptionError("Snapshot missing data field")
    
    # Reconstruct snapshot
    snapshot = SnapshotData.from_dict(data)
    
    # Verify hash if requested
    if verify_hash:
        computed_hash = snapshot.compute_hash()
        if stored_hash != computed_hash:
            raise SnapshotCorruptionError(
                f"Hash mismatch: stored={stored_hash}, computed={computed_hash}"
            )
    
    return snapshot


def find_latest_snapshot(snapshot_dir: Path, experiment_id: str) -> Optional[Path]:
    """
    Find the most recent snapshot for an experiment.
    
    Args:
        snapshot_dir: Directory containing snapshots
        experiment_id: Experiment identifier
        
    Returns:
        Path to latest snapshot, or None if not found
    """
    if not snapshot_dir.exists():
        return None
    
    # Find all snapshots for this experiment
    pattern = f"{experiment_id}_cycle_*.json"
    snapshots = list(snapshot_dir.glob(pattern))
    
    if not snapshots:
        return None
    
    # Sort by modification time (most recent first)
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return snapshots[0]


def rotate_snapshots(
    snapshot_dir: Path,
    experiment_id: str,
    keep: int = 5,
) -> List[Path]:
    """
    Rotate snapshots, keeping only the most recent N.
    
    Args:
        snapshot_dir: Directory containing snapshots
        experiment_id: Experiment identifier
        keep: Number of snapshots to keep
        
    Returns:
        List of deleted snapshot paths
    """
    if not snapshot_dir.exists():
        return []
    
    # Find all snapshots for this experiment
    pattern = f"{experiment_id}_cycle_*.json"
    snapshots = list(snapshot_dir.glob(pattern))
    
    if len(snapshots) <= keep:
        return []
    
    # Sort by modification time (oldest first)
    snapshots.sort(key=lambda p: p.stat().st_mtime)
    
    # Delete oldest snapshots
    to_delete = snapshots[:len(snapshots) - keep]
    deleted = []
    
    for snapshot_path in to_delete:
        try:
            snapshot_path.unlink()
            deleted.append(snapshot_path)
        except OSError:
            pass  # Ignore deletion errors
    
    return deleted


def create_snapshot_name(experiment_id: str, cycle: int) -> str:
    """
    Create standardized snapshot filename.
    
    Args:
        experiment_id: Experiment identifier
        cycle: Cycle number
        
    Returns:
        Snapshot filename
    """
    return f"{experiment_id}_cycle_{cycle:06d}.json"
