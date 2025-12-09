"""
PHASE II — NOT USED IN PHASE I

U2 Snapshot & Restore Module
============================

Provides deterministic save/restore for U2 uplift experiments, enabling:
- Long experiments to be paused and resumed
- Deterministic replay from any checkpoint
- Audit trails for reproducibility verification

Snapshot Format:
    - Serialization: msgpack (compact, deterministic)
    - Compression: zstd (fast, high ratio)
    - Integrity: SHA256 hash embedded in snapshot

Usage:
    from experiments.u2.snapshots import (
        SnapshotData,
        save_snapshot,
        load_snapshot,
        compute_snapshot_hash,
    )

Absolute Safeguards:
    - Do NOT reinterpret Phase I logs as uplift evidence.
    - All Phase II artifacts must be clearly labeled.
    - Snapshot/restore must preserve exact determinism.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies - graceful degradation
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


__all__ = [
    "SnapshotData",
    "SnapshotValidationError",
    "SnapshotCorruptionError",
    "NoSnapshotFoundError",
    "compute_snapshot_hash",
    "save_snapshot",
    "load_snapshot",
    "validate_snapshot",
    "find_latest_snapshot",
    "list_snapshots",
    "rotate_snapshots",
]


class SnapshotValidationError(Exception):
    """Raised when snapshot validation fails (e.g., missing fields, invalid values)."""
    pass


class SnapshotCorruptionError(SnapshotValidationError):
    """Raised when snapshot file is corrupted (hash mismatch, decompression failure)."""
    pass


class NoSnapshotFoundError(Exception):
    """Raised when no snapshot is found for resume operation."""
    pass


@dataclass
class SnapshotData:
    """
    Deterministic snapshot for U2 experiments.
    
    This dataclass captures all runtime state required to restore a U2
    experiment to an exact point and continue with identical behavior.
    
    Fields are grouped by category:
        - Schema: Version for forward compatibility
        - Cycle: Current position in experiment
        - PRNG: All random number generator states
        - Policy: RFL policy weights and history
        - Accumulators: Running statistics
        - Config: Hashes for integrity verification
        - Metadata: Experiment identification
    """
    
    # --- Schema Version ---
    schema_version: str = "1.0"
    
    # --- Cycle Tracking ---
    cycle_index: int = 0                    # Current cycle (0-indexed, snapshot taken AFTER this cycle)
    total_cycles: int = 0                   # Total cycles in experiment
    
    # --- PRNG State ---
    # numpy.random state: 5-tuple (str, ndarray[624], int, int, float)
    # We store as: (algorithm, list[int], pos, has_gauss, cached_gaussian)
    numpy_rng_state: Optional[Tuple] = None
    
    # Python stdlib random state
    python_rng_state: Optional[Tuple] = None
    
    # Master seed for experiment
    master_seed: int = 0
    
    # Seed schedule (list of per-cycle seeds)
    seed_schedule: List[int] = field(default_factory=list)
    
    # --- Policy State (RFL Mode) ---
    policy_scores: Dict[str, float] = field(default_factory=dict)
    policy_rng_state: Optional[Tuple] = None    # State of policy's internal RNG
    
    # Extended RFL runner state (from rfl/runner.py)
    policy_weights: Dict[str, float] = field(default_factory=dict)
    success_count: Dict[str, int] = field(default_factory=dict)
    attempt_count: Dict[str, int] = field(default_factory=dict)
    policy_update_count: int = 0
    first_organism_runs_total: int = 0
    
    # --- Accumulators ---
    abstention_histogram: Dict[str, int] = field(default_factory=dict)
    abstention_fraction: float = 0.0
    previous_coverage_rate: Optional[float] = None
    throughput_reference: float = 1.0
    has_throughput_reference: bool = False
    lean_failure_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # --- Configuration Integrity ---
    config_hash: str = ""                   # SHA256 of full config
    slice_config_hash: str = ""             # SHA256 of current slice params
    curriculum_hash: str = ""               # SHA256 of curriculum.yaml
    
    # --- Experiment Metadata ---
    experiment_id: str = ""
    mode: str = ""                          # "baseline" or "rfl"
    slice_name: str = ""
    label: str = "PHASE II — NOT USED IN PHASE I"
    
    # --- Telemetry History (optional, can be large) ---
    # By default, we exclude ht_series to keep snapshots small.
    # Set include_history=True in capture_state() to include it.
    ht_series_hash: str = ""                # Hash of telemetry series (for verification)
    ht_series_length: int = 0               # Length of telemetry series
    
    # --- Replay & Governance Metadata (advisory, no new semantics) ---
    # These fields are used by replay tools and governance code to determine
    # snapshot eligibility for deterministic replay.
    manifest_hash: str = ""                 # SHA256 of experiment manifest at snapshot time
    created_at_cycle: int = 0               # Cycle index when snapshot was created (alias for cycle_index)
    snapshot_timestamp: str = ""            # ISO timestamp when snapshot was taken (deterministic from cycle)


def _serialize_numpy_state(state: Any) -> Optional[Tuple]:
    """
    Convert numpy RandomState to serializable tuple.
    
    numpy.random.get_state() returns:
        ('MT19937', array([...], dtype=uint32), pos, has_gauss, cached_gaussian)
    
    We convert the array to a list for msgpack serialization.
    """
    if state is None:
        return None
    
    if not HAS_NUMPY:
        return None
    
    try:
        alg, keys, pos, has_gauss, cached = state
        # Convert numpy array to list of ints
        keys_list = keys.tolist() if hasattr(keys, 'tolist') else list(keys)
        return (alg, keys_list, int(pos), int(has_gauss), float(cached))
    except (ValueError, TypeError, AttributeError):
        return None


def _deserialize_numpy_state(state: Optional[Tuple]) -> Optional[Tuple]:
    """
    Convert serialized tuple back to numpy RandomState format.
    """
    if state is None:
        return None
    
    if not HAS_NUMPY:
        return None
    
    try:
        alg, keys_list, pos, has_gauss, cached = state
        # Convert list back to numpy array with correct dtype
        keys_array = np.array(keys_list, dtype=np.uint32)
        return (alg, keys_array, pos, has_gauss, cached)
    except (ValueError, TypeError, AttributeError):
        return None


def _to_plain_dict(data: SnapshotData) -> Dict[str, Any]:
    """
    Convert SnapshotData to a plain dict suitable for serialization.
    
    Handles special cases like numpy arrays.
    """
    d = asdict(data)
    
    # Ensure numpy state is serializable
    if d.get('numpy_rng_state') is not None:
        d['numpy_rng_state'] = _serialize_numpy_state(d['numpy_rng_state'])
    
    return d


def _convert_lists_to_tuples(obj: Any) -> Any:
    """
    Recursively convert lists to tuples.
    
    msgpack deserializes tuples as lists, but random.setstate() requires tuples.
    """
    if isinstance(obj, list):
        return tuple(_convert_lists_to_tuples(item) for item in obj)
    return obj


def _from_plain_dict(d: Dict[str, Any]) -> SnapshotData:
    """
    Reconstruct SnapshotData from a plain dict.
    """
    # Handle numpy state deserialization
    if d.get('numpy_rng_state') is not None:
        d['numpy_rng_state'] = _deserialize_numpy_state(d['numpy_rng_state'])
    
    # Handle python RNG state - must be tuple, not list
    if d.get('python_rng_state') is not None:
        d['python_rng_state'] = _convert_lists_to_tuples(d['python_rng_state'])
    
    # Handle policy RNG state - must be tuple, not list
    if d.get('policy_rng_state') is not None:
        d['policy_rng_state'] = _convert_lists_to_tuples(d['policy_rng_state'])
    
    # Filter to only known fields (forward compatibility)
    known_fields = {f.name for f in SnapshotData.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in known_fields}
    
    return SnapshotData(**filtered)


def compute_snapshot_hash(data: SnapshotData) -> str:
    """
    Compute deterministic SHA256 hash of snapshot data.
    
    Uses canonical JSON serialization (sorted keys, no whitespace)
    to ensure identical data always produces identical hash.
    
    Args:
        data: SnapshotData instance to hash
        
    Returns:
        64-character hexadecimal SHA256 hash
    """
    # Convert to plain dict
    plain = _to_plain_dict(data)
    
    # Canonical JSON: sorted keys, compact separators, ASCII only
    canonical = json.dumps(
        plain,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        default=str,  # Handle any non-serializable types
    )
    
    # SHA256 hash
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def save_snapshot(data: SnapshotData, path: Path, compression_level: int = 3) -> str:
    """
    Save snapshot to file with compression and integrity hash.
    
    File format:
        zstd.compress(msgpack.packb({
            "schema_version": "1.0",
            "snapshot": {...},
            "hash": "sha256hex"
        }))
    
    Args:
        data: SnapshotData instance to save
        path: Output file path
        compression_level: zstd compression level (1-22, default 3)
        
    Returns:
        SHA256 hash of the snapshot data
        
    Raises:
        ValueError: If computed hash doesn't match (defensive check)
        ImportError: If msgpack or zstd not available
    """
    if not HAS_MSGPACK:
        raise ImportError(
            "msgpack is required for snapshot serialization. "
            "Install with: pip install msgpack"
        )
    
    if not HAS_ZSTD:
        raise ImportError(
            "zstandard is required for snapshot compression. "
            "Install with: pip install zstandard"
        )
    
    # Convert to plain dict
    plain = _to_plain_dict(data)
    
    # Compute hash
    snapshot_hash = compute_snapshot_hash(data)
    
    # Wrapper with hash for integrity verification
    wrapper = {
        "schema_version": data.schema_version,
        "snapshot": plain,
        "hash": snapshot_hash,
        "label": "PHASE II — NOT USED IN PHASE I",
    }
    
    # Serialize with msgpack
    packed = msgpack.packb(wrapper, use_bin_type=True)
    
    # Compress with zstd
    compressor = zstd.ZstdCompressor(level=compression_level)
    compressed = compressor.compress(packed)
    
    # Ensure parent directory exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write atomically (write to temp, then rename)
    temp_path = path.with_suffix('.tmp')
    try:
        with open(temp_path, 'wb') as f:
            f.write(compressed)
        temp_path.replace(path)
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise
    
    return snapshot_hash


def load_snapshot(path: Path, verify_hash: bool = True) -> SnapshotData:
    """
    Load snapshot from file with integrity verification.
    
    Args:
        path: Path to snapshot file
        verify_hash: If True, verify SHA256 hash matches (default True)
        
    Returns:
        SnapshotData instance
        
    Raises:
        FileNotFoundError: If snapshot file doesn't exist
        SnapshotValidationError: If hash verification fails
        ImportError: If msgpack or zstd not available
    """
    if not HAS_MSGPACK:
        raise ImportError(
            "msgpack is required for snapshot deserialization. "
            "Install with: pip install msgpack"
        )
    
    if not HAS_ZSTD:
        raise ImportError(
            "zstandard is required for snapshot decompression. "
            "Install with: pip install zstandard"
        )
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    
    # Read compressed data
    with open(path, 'rb') as f:
        compressed = f.read()
    
    # Decompress
    decompressor = zstd.ZstdDecompressor()
    packed = decompressor.decompress(compressed)
    
    # Deserialize
    wrapper = msgpack.unpackb(packed, raw=False)
    
    # Extract components
    stored_hash = wrapper.get('hash', '')
    plain = wrapper.get('snapshot', {})
    
    # Reconstruct SnapshotData
    data = _from_plain_dict(plain)
    
    # Verify hash if requested
    if verify_hash:
        computed_hash = compute_snapshot_hash(data)
        if computed_hash != stored_hash:
            raise SnapshotCorruptionError(
                f"Snapshot hash mismatch!\n"
                f"  Stored:   {stored_hash}\n"
                f"  Computed: {computed_hash}\n"
                f"  File may be corrupted or tampered with."
            )
    
    return data


def list_snapshots(snapshot_dir: Path, pattern: str = "snapshot_*.snap") -> List[Path]:
    """
    List all snapshot files in a directory, sorted by cycle index (newest first).
    
    Args:
        snapshot_dir: Directory to search
        pattern: Glob pattern for snapshot files (default: snapshot_*.snap)
        
    Returns:
        List of snapshot paths, sorted by cycle index descending (newest first)
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []
    
    snapshots = list(snapshot_dir.glob(pattern))
    
    # Sort by cycle index extracted from filename
    # Expected format: snapshot_<experiment_id>_<cycle_index>.snap
    # or: snapshot_cycle_<cycle_index>.snap
    def extract_cycle_index(path: Path) -> int:
        """Extract cycle index from snapshot filename."""
        name = path.stem  # Remove .snap extension
        parts = name.split('_')
        
        # Try to find a numeric part that looks like a cycle index
        for part in reversed(parts):
            # Skip 'final' suffix
            if part == 'final':
                return float('inf')  # Sort final snapshots last
            try:
                return int(part)
            except ValueError:
                continue
        return 0
    
    # Sort by cycle index descending (newest first)
    return sorted(snapshots, key=extract_cycle_index, reverse=True)


def find_latest_snapshot(snapshot_dir: Path, pattern: str = "snapshot_*.snap") -> Optional[Path]:
    """
    Find the most recent snapshot file in a directory.
    
    Discovery logic:
    1. Lists all files matching the pattern
    2. Extracts cycle index from filename
    3. Returns the one with highest cycle index (excluding 'final')
    
    Args:
        snapshot_dir: Directory to search
        pattern: Glob pattern for snapshot files (default: snapshot_*.snap)
        
    Returns:
        Path to the latest snapshot, or None if no snapshots found
    """
    snapshots = list_snapshots(snapshot_dir, pattern)
    
    if not snapshots:
        return None
    
    # Filter out 'final' snapshots for resume (they represent completed runs)
    non_final = [s for s in snapshots if 'final' not in s.stem]
    
    if non_final:
        return non_final[0]  # Already sorted newest first
    
    # Fall back to any snapshot if only finals exist
    return snapshots[0] if snapshots else None


def rotate_snapshots(
    snapshot_dir: Path,
    keep_count: int = 5,
    pattern: str = "snapshot_*.snap",
    exclude_final: bool = True,
) -> List[Path]:
    """
    Delete old snapshots, keeping only the most recent N.
    
    Rotation policy:
    - Keeps the newest `keep_count` snapshots based on cycle index
    - Optionally preserves 'final' snapshots (completed run markers)
    - Returns list of deleted files
    
    Args:
        snapshot_dir: Directory containing snapshots
        keep_count: Number of snapshots to keep (default: 5)
        pattern: Glob pattern for snapshot files
        exclude_final: If True, never delete 'final' snapshots (default: True)
        
    Returns:
        List of paths that were deleted
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []
    
    snapshots = list_snapshots(snapshot_dir, pattern)
    
    if exclude_final:
        # Separate final from regular snapshots
        final_snapshots = [s for s in snapshots if 'final' in s.stem]
        regular_snapshots = [s for s in snapshots if 'final' not in s.stem]
    else:
        final_snapshots = []
        regular_snapshots = snapshots
    
    # Keep the newest N regular snapshots
    to_keep = set(regular_snapshots[:keep_count])
    to_keep.update(final_snapshots)  # Always keep finals
    
    deleted = []
    for snapshot in regular_snapshots:
        if snapshot not in to_keep:
            try:
                snapshot.unlink()
                deleted.append(snapshot)
            except OSError:
                pass  # Ignore deletion errors
    
    return deleted


def validate_snapshot(
    data: SnapshotData,
    expected_config_hash: Optional[str] = None,
    expected_slice_hash: Optional[str] = None,
) -> bool:
    """
    Validate snapshot integrity and configuration consistency.
    
    Args:
        data: SnapshotData to validate
        expected_config_hash: Expected config hash (optional)
        expected_slice_hash: Expected slice config hash (optional)
        
    Returns:
        True if valid
        
    Raises:
        SnapshotValidationError: If validation fails
    """
    errors = []
    
    # Schema version check
    if not data.schema_version.startswith("1."):
        errors.append(f"Unsupported schema version: {data.schema_version}")
    
    # Cycle index sanity
    if data.cycle_index < 0:
        errors.append(f"Invalid cycle_index: {data.cycle_index}")
    
    if data.total_cycles > 0 and data.cycle_index >= data.total_cycles:
        errors.append(
            f"cycle_index ({data.cycle_index}) >= total_cycles ({data.total_cycles})"
        )
    
    # Config hash verification
    if expected_config_hash and data.config_hash != expected_config_hash:
        errors.append(
            f"Config hash mismatch: expected {expected_config_hash}, "
            f"got {data.config_hash}"
        )
    
    if expected_slice_hash and data.slice_config_hash != expected_slice_hash:
        errors.append(
            f"Slice config hash mismatch: expected {expected_slice_hash}, "
            f"got {data.slice_config_hash}"
        )
    
    # Mode validation
    if data.mode and data.mode not in ("baseline", "rfl"):
        errors.append(f"Invalid mode: {data.mode}")
    
    if errors:
        raise SnapshotValidationError(
            "Snapshot validation failed:\n" + 
            "\n".join(f"  - {e}" for e in errors)
        )
    
    return True


# --- Convenience Functions for State Capture/Restore ---

def capture_prng_states() -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Capture current PRNG states from numpy and stdlib random.
    
    Returns:
        Tuple of (numpy_state, python_state)
    """
    numpy_state = None
    if HAS_NUMPY:
        numpy_state = np.random.get_state()
    
    python_state = random.getstate()
    
    return numpy_state, python_state


def restore_prng_states(
    numpy_state: Optional[Tuple],
    python_state: Optional[Tuple],
) -> None:
    """
    Restore PRNG states to numpy and stdlib random.
    
    Args:
        numpy_state: State tuple from np.random.get_state()
        python_state: State tuple from random.getstate()
    """
    if HAS_NUMPY and numpy_state is not None:
        # Deserialize if needed
        restored = _deserialize_numpy_state(numpy_state)
        if restored is not None:
            np.random.set_state(restored)
    
    if python_state is not None:
        random.setstate(python_state)


def capture_random_instance_state(rng: random.Random) -> Tuple:
    """
    Capture state from a random.Random instance.
    
    Args:
        rng: random.Random instance
        
    Returns:
        State tuple suitable for setstate()
    """
    return rng.getstate()


def restore_random_instance_state(rng: random.Random, state: Tuple) -> None:
    """
    Restore state to a random.Random instance.
    
    Args:
        rng: random.Random instance
        state: State tuple from getstate()
    """
    rng.setstate(state)

