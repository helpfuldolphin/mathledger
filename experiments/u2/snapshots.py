"""
Snapshot Management for U2 Uplift Experiments

Provides deterministic snapshot save/restore functionality for long-running experiments.

PHASE II — NOT USED IN PHASE I
"""

import hashlib
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class SnapshotError(Exception):
    """Base class for snapshot errors."""
    pass


class SnapshotValidationError(SnapshotError):
    """Raised when snapshot validation fails."""
    pass


class SnapshotCorruptionError(SnapshotError):
    """Raised when snapshot appears corrupted."""
    pass


class NoSnapshotFoundError(SnapshotError):
    """Raised when no snapshot found for resume."""
    pass


@dataclass
class SnapshotData:
    """
    Snapshot of experiment state.
    
    Attributes:
        cycle_index: Current cycle index
        ht_series: Series of H_t values
        policy_update_count: Number of policy updates (RFL mode)
        success_count: Success counts per item (RFL mode)
        attempt_count: Attempt counts per item (RFL mode)
        config: Experiment configuration
        metadata: Additional metadata
    """
    cycle_index: int
    ht_series: List[str]
    policy_update_count: int
    success_count: Dict[str, int]
    attempt_count: Dict[str, int]
    config: Any  # U2Config, but avoid circular import
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert config to dict if needed
        if hasattr(data['config'], '__dict__'):
            data['config'] = {
                'experiment_id': data['config'].experiment_id,
                'slice_name': data['config'].slice_name,
                'mode': data['config'].mode,
                'total_cycles': data['config'].total_cycles,
                'master_seed': data['config'].master_seed,
                'snapshot_interval': data['config'].snapshot_interval,
                'snapshot_dir': str(data['config'].snapshot_dir),
                'output_dir': str(data['config'].output_dir),
                'slice_config': data['config'].slice_config,
            }
        return data
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SnapshotData':
        """Create from dictionary."""
        # Import here to avoid circular dependency
        from experiments.u2.runner import U2Config
        
        config_dict = data['config']
        config = U2Config(
            experiment_id=config_dict['experiment_id'],
            slice_name=config_dict['slice_name'],
            mode=config_dict['mode'],
            total_cycles=config_dict['total_cycles'],
            master_seed=config_dict['master_seed'],
            snapshot_interval=config_dict['snapshot_interval'],
            snapshot_dir=Path(config_dict['snapshot_dir']),
            output_dir=Path(config_dict['output_dir']),
            slice_config=config_dict['slice_config'],
        )
        
        return SnapshotData(
            cycle_index=data['cycle_index'],
            ht_series=data['ht_series'],
            policy_update_count=data['policy_update_count'],
            success_count=data['success_count'],
            attempt_count=data['attempt_count'],
            config=config,
            metadata=data.get('metadata'),
        )


def _compute_snapshot_hash(data: Dict[str, Any]) -> str:
    """Compute SHA256 hash of snapshot data."""
    # Sort keys for deterministic hashing
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def save_snapshot(snapshot: SnapshotData, path: Path) -> None:
    """
    Save snapshot to disk with integrity hash.
    
    Args:
        snapshot: SnapshotData to save
        path: Path to save snapshot file
    """
    # Convert to dict
    data = snapshot.to_dict()
    
    # Compute hash
    data_hash = _compute_snapshot_hash(data)
    
    # Create snapshot package with hash
    package = {
        'version': 1,
        'hash': data_hash,
        'data': data,
    }
    
    # Write to disk
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(package, f)


def load_snapshot(path: Path, verify_hash: bool = True) -> SnapshotData:
    """
    Load snapshot from disk with optional hash verification.
    
    Args:
        path: Path to snapshot file
        verify_hash: Whether to verify integrity hash
        
    Returns:
        SnapshotData
        
    Raises:
        SnapshotValidationError: If hash verification fails
        SnapshotCorruptionError: If snapshot is corrupted
    """
    if not path.exists():
        raise NoSnapshotFoundError(f"Snapshot not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            package = pickle.load(f)
    except Exception as e:
        raise SnapshotCorruptionError(f"Failed to load snapshot: {e}")
    
    # Validate structure
    if not isinstance(package, dict) or 'data' not in package:
        raise SnapshotCorruptionError("Invalid snapshot structure")
    
    data = package['data']
    
    # Verify hash if requested
    if verify_hash:
        stored_hash = package.get('hash')
        if stored_hash is None:
            raise SnapshotValidationError("Snapshot missing integrity hash")
        
        computed_hash = _compute_snapshot_hash(data)
        if stored_hash != computed_hash:
            raise SnapshotValidationError(
                f"Hash mismatch: stored={stored_hash[:16]}..., computed={computed_hash[:16]}..."
            )
    
    # Reconstruct SnapshotData
    try:
        return SnapshotData.from_dict(data)
    except Exception as e:
        raise SnapshotCorruptionError(f"Failed to reconstruct snapshot: {e}")


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


def rotate_snapshots(snapshot_dir: Path, keep_count: int) -> List[Path]:
    """
    Rotate snapshots to keep only the most recent N.
    
    Args:
        snapshot_dir: Directory containing snapshots
        keep_count: Number of snapshots to keep
        
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
    
    # Sort by modification time (oldest first)
    snapshots.sort(key=lambda p: p.stat().st_mtime)
    
    # Delete oldest snapshots
    to_delete = snapshots[:-keep_count]
    deleted = []
    for path in to_delete:
        try:
            path.unlink()
            deleted.append(path)
        except Exception:
            # Ignore errors during cleanup
            pass
    
    return deleted
