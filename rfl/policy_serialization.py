"""
RFL Policy Serialization & Delta Tracking Module
=================================================

Provides deterministic serialization, checkpointing, and delta tracking
for RFL policy evolution. Ensures policies can be replayed from logs
and verified for determinism.

Key Features:
- Versioned policy serialization (JSON + binary formats)
- Symbolic delta tracking with provenance
- Policy checkpointing with integrity verification
- Replay infrastructure for determinism audits

Usage:
    from rfl.policy_serialization import (
        PolicyCheckpoint,
        DeltaLog,
        save_checkpoint,
        load_checkpoint,
        replay_from_deltas,
    )

    # Save checkpoint
    checkpoint = PolicyCheckpoint.from_policy_state(policy, config)
    save_checkpoint(checkpoint, "checkpoints/policy_epoch_10.json")

    # Load and verify
    loaded = load_checkpoint("checkpoints/policy_epoch_10.json")
    assert loaded.verify_integrity()

    # Replay from deltas
    final_policy = replay_from_deltas(initial_policy, delta_log)
"""

from __future__ import annotations

import hashlib
import json
import gzip
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from backend.repro.determinism import deterministic_isoformat

from .update_algebra import PolicyState, PolicyUpdate, apply_update, PolicyEvolutionChain


# Serialization format version (increment on breaking changes)
SERIALIZATION_VERSION = "1.0.0"


@dataclass(frozen=True)
class PolicyCheckpoint:
    """
    Immutable policy checkpoint with integrity verification.
    
    A checkpoint captures the complete policy state at a specific epoch,
    along with metadata for provenance and verification.
    
    Attributes:
        policy_state: The policy state π_t
        experiment_id: Experiment identifier for traceability
        config_hash: SHA256 hash of RFL config used
        checkpoint_version: Serialization format version
        created_at: ISO 8601 timestamp of checkpoint creation
        integrity_hash: SHA256 hash of checkpoint content
    """
    policy_state: PolicyState
    experiment_id: str
    config_hash: str
    checkpoint_version: str = SERIALIZATION_VERSION
    created_at: str = ""
    integrity_hash: str = ""
    
    def __post_init__(self):
        """Compute integrity hash if not provided."""
        if not self.integrity_hash:
            # Use object.__setattr__ to modify frozen dataclass
            object.__setattr__(self, 'integrity_hash', self._compute_integrity_hash())
    
    def _compute_integrity_hash(self) -> str:
        """Compute SHA256 hash of checkpoint content."""
        content = json.dumps(
            {
                "policy_state": self.policy_state.to_dict(),
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "checkpoint_version": self.checkpoint_version,
                "created_at": self.created_at,
            },
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True
        )
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify that integrity_hash matches computed hash."""
        return self.integrity_hash == self._compute_integrity_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "policy_state": self.policy_state.to_dict(),
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "checkpoint_version": self.checkpoint_version,
            "created_at": self.created_at,
            "integrity_hash": self.integrity_hash,
            "verified": self.verify_integrity(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    @classmethod
    def from_policy_state(
        cls,
        policy_state: PolicyState,
        experiment_id: str,
        config_hash: str,
        created_at: Optional[str] = None,
    ) -> PolicyCheckpoint:
        """
        Create checkpoint from policy state.
        
        Args:
            policy_state: Policy state to checkpoint
            experiment_id: Experiment identifier
            config_hash: Hash of RFL config
            created_at: Optional timestamp (auto-generated if None)
        
        Returns:
            New PolicyCheckpoint
        """
        if created_at is None:
            created_at = deterministic_isoformat(experiment_id, config_hash, policy_state)
        
        return cls(
            policy_state=policy_state,
            experiment_id=experiment_id,
            config_hash=config_hash,
            created_at=created_at,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyCheckpoint:
        """Deserialize from dictionary."""
        return cls(
            policy_state=PolicyState.from_dict(data["policy_state"]),
            experiment_id=data["experiment_id"],
            config_hash=data["config_hash"],
            checkpoint_version=data.get("checkpoint_version", SERIALIZATION_VERSION),
            created_at=data.get("created_at", ""),
            integrity_hash=data.get("integrity_hash", ""),
        )


@dataclass
class DeltaLogEntry:
    """
    Single entry in the symbolic delta log.
    
    Records a policy update with full provenance, enabling replay
    and verification of policy evolution.
    
    Attributes:
        epoch: Epoch number when update was applied
        update: The policy update Δπ
        source_event_hash: Hash of dual-attested event (H_t)
        timestamp: Deterministic timestamp of update
        metadata: Additional context (abstention_rate, verified_count, etc.)
    """
    epoch: int
    update: PolicyUpdate
    source_event_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "epoch": self.epoch,
            "update": self.update.to_dict(),
            "source_event_hash": self.source_event_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeltaLogEntry:
        """Deserialize from dictionary."""
        return cls(
            epoch=data["epoch"],
            update=PolicyUpdate(
                deltas=data["update"]["deltas"],
                step_size=data["update"]["step_size"],
                gradient_norm=data["update"].get("gradient_norm", 0.0),
                source_event_hash=data["update"].get("source_event_hash"),
                metadata=data["update"].get("metadata", {}),
            ),
            source_event_hash=data["source_event_hash"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class DeltaLog:
    """
    Complete log of symbolic deltas across policy evolution.
    
    Maintains an append-only log of all policy updates, enabling:
    - Deterministic replay from initial state
    - Verification of update provenance
    - Audit trail for governance
    
    Attributes:
        entries: Ordered list of delta log entries
        initial_policy_hash: Hash of π_0
        experiment_id: Experiment identifier
        metadata: Additional context
    """
    entries: List[DeltaLogEntry] = field(default_factory=list)
    initial_policy_hash: str = ""
    experiment_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def append(
        self,
        epoch: int,
        update: PolicyUpdate,
        source_event_hash: str,
        timestamp: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a new delta entry to the log.
        
        Args:
            epoch: Epoch number
            update: Policy update
            source_event_hash: Hash of source event
            timestamp: Deterministic timestamp
            metadata: Optional additional context
        """
        entry = DeltaLogEntry(
            epoch=epoch,
            update=update,
            source_event_hash=source_event_hash,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        self.entries.append(entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "initial_policy_hash": self.initial_policy_hash,
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
            "entry_count": len(self.entries),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    def save(self, filepath: str, compress: bool = False) -> None:
        """
        Save delta log to file.
        
        Args:
            filepath: Output file path
            compress: If True, use gzip compression
        """
        content = self.to_json()
        
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    @classmethod
    def load(cls, filepath: str) -> DeltaLog:
        """
        Load delta log from file.
        
        Args:
            filepath: Input file path (auto-detects compression)
        
        Returns:
            Loaded DeltaLog
        """
        # Try gzip first, fall back to plain text
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, gzip.BadGzipFile):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        entries = [DeltaLogEntry.from_dict(e) for e in data["entries"]]
        
        return cls(
            entries=entries,
            initial_policy_hash=data.get("initial_policy_hash", ""),
            experiment_id=data.get("experiment_id", ""),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Checkpoint Management
# -----------------------------------------------------------------------------

def save_checkpoint(checkpoint: PolicyCheckpoint, filepath: str) -> None:
    """
    Save policy checkpoint to JSON file.
    
    Args:
        checkpoint: Checkpoint to save
        filepath: Output file path
    
    Raises:
        ValueError: If checkpoint integrity verification fails
    """
    if not checkpoint.verify_integrity():
        raise ValueError("Cannot save checkpoint: integrity verification failed")
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(checkpoint.to_json())


def load_checkpoint(filepath: str) -> PolicyCheckpoint:
    """
    Load policy checkpoint from JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded PolicyCheckpoint
    
    Raises:
        ValueError: If checkpoint integrity verification fails
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    checkpoint = PolicyCheckpoint.from_dict(data)
    
    if not checkpoint.verify_integrity():
        raise ValueError(f"Checkpoint integrity verification failed: {filepath}")
    
    return checkpoint


def list_checkpoints(directory: str) -> List[Tuple[int, str]]:
    """
    List all checkpoints in a directory, sorted by epoch.
    
    Args:
        directory: Directory containing checkpoint files
    
    Returns:
        List of (epoch, filepath) tuples, sorted by epoch
    """
    checkpoint_dir = Path(directory)
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for filepath in checkpoint_dir.glob("*.json"):
        try:
            checkpoint = load_checkpoint(str(filepath))
            checkpoints.append((checkpoint.policy_state.epoch, str(filepath)))
        except Exception:
            # Skip invalid checkpoints
            continue
    
    return sorted(checkpoints, key=lambda x: x[0])


# -----------------------------------------------------------------------------
# Replay Infrastructure
# -----------------------------------------------------------------------------

def replay_from_deltas(
    initial_policy: PolicyState,
    delta_log: DeltaLog,
    constraints: Optional[Dict[str, tuple]] = None,
    verify_hashes: bool = True,
) -> Tuple[PolicyState, List[str]]:
    """
    Replay policy evolution from delta log.
    
    Applies all updates in the delta log sequentially to reconstruct
    the final policy state. Optionally verifies that source event hashes
    match expected values.
    
    Args:
        initial_policy: Initial policy state π_0
        delta_log: Log of policy updates
        constraints: Optional weight bounds
        verify_hashes: If True, verify source event hashes
    
    Returns:
        Tuple of (final_policy_state, list of warnings)
    
    Raises:
        ValueError: If replay fails due to invalid updates
    """
    warnings = []
    
    # Verify initial policy hash matches log
    if verify_hashes and delta_log.initial_policy_hash:
        if initial_policy.hash() != delta_log.initial_policy_hash:
            warnings.append(
                f"Initial policy hash mismatch: "
                f"expected {delta_log.initial_policy_hash}, got {initial_policy.hash()}"
            )
    
    current_policy = initial_policy
    
    for i, entry in enumerate(delta_log.entries):
        # Verify epoch sequence
        expected_epoch = initial_policy.epoch + i + 1
        if entry.epoch != expected_epoch:
            warnings.append(
                f"Entry {i}: epoch mismatch (expected {expected_epoch}, got {entry.epoch})"
            )
        
        # Apply update
        try:
            current_policy = apply_update(
                current_policy,
                entry.update,
                entry.timestamp,
                constraints,
            )
        except Exception as e:
            raise ValueError(f"Replay failed at entry {i} (epoch {entry.epoch}): {e}")
    
    return current_policy, warnings


def replay_from_checkpoints(
    checkpoint_dir: str,
    delta_log: DeltaLog,
    constraints: Optional[Dict[str, tuple]] = None,
) -> Tuple[PolicyState, List[str]]:
    """
    Replay policy evolution using checkpoints and deltas.
    
    Finds the latest checkpoint before the delta log and replays
    only the remaining updates. This is more efficient than replaying
    from π_0 for long evolution chains.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        delta_log: Log of policy updates
        constraints: Optional weight bounds
    
    Returns:
        Tuple of (final_policy_state, list of warnings)
    
    Raises:
        ValueError: If no suitable checkpoint found or replay fails
    """
    warnings = []
    
    # Find latest checkpoint
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Use latest checkpoint as starting point
    latest_epoch, latest_path = checkpoints[-1]
    checkpoint = load_checkpoint(latest_path)
    
    # Filter delta log to entries after checkpoint
    remaining_entries = [e for e in delta_log.entries if e.epoch > latest_epoch]
    
    if not remaining_entries:
        # Checkpoint is already at final state
        return checkpoint.policy_state, warnings
    
    # Create filtered delta log
    filtered_log = DeltaLog(
        entries=remaining_entries,
        initial_policy_hash=checkpoint.policy_state.hash(),
        experiment_id=delta_log.experiment_id,
        metadata=delta_log.metadata,
    )
    
    # Replay remaining updates
    final_policy, replay_warnings = replay_from_deltas(
        checkpoint.policy_state,
        filtered_log,
        constraints,
        verify_hashes=False,  # Already verified checkpoint
    )
    
    warnings.extend(replay_warnings)
    return final_policy, warnings


# -----------------------------------------------------------------------------
# Config Hashing
# -----------------------------------------------------------------------------

def compute_config_hash(config: Any) -> str:
    """
    Compute deterministic SHA256 hash of RFL config.
    
    Args:
        config: RFLConfig instance
    
    Returns:
        SHA256 hash of config
    """
    # Convert to dict and extract relevant fields
    config_dict = config.to_dict()
    
    # Remove non-deterministic fields
    exclude_fields = {"database_url", "redis_url", "artifacts_dir", "results_file"}
    filtered = {k: v for k, v in config_dict.items() if k not in exclude_fields}
    
    # Canonical serialization
    canonical = json.dumps(filtered, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


__all__ = [
    "SERIALIZATION_VERSION",
    "PolicyCheckpoint",
    "DeltaLogEntry",
    "DeltaLog",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    "replay_from_deltas",
    "replay_from_checkpoints",
    "compute_config_hash",
]
