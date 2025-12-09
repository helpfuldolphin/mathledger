"""
Epoch Management Module

Manages hash algorithm epochs and provides epoch resolution for blocks.

Author: Manus-H
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from backend.consensus_pq.rules import ConsensusRuleVersion


@dataclass(frozen=True)
class HashEpoch:
    """
    Represents a hash algorithm epoch.
    
    An epoch is a contiguous range of blocks using the same canonical
    hash algorithm and consensus rules.
    
    Attributes:
        start_block: First block number in this epoch (inclusive)
        end_block: Last block number in this epoch (inclusive), None for current epoch
        algorithm_id: Hash algorithm ID for this epoch
        algorithm_name: Human-readable algorithm name
        rule_version: Consensus rule version for this epoch
        activation_timestamp: Unix timestamp when epoch was activated
        governance_hash: Hash of governance proposal that activated this epoch
    """
    
    start_block: int
    end_block: Optional[int]
    algorithm_id: int
    algorithm_name: str
    rule_version: str
    activation_timestamp: float
    governance_hash: str


# Global epoch registry
# In production, this would be persisted to database
_EPOCH_REGISTRY: List[HashEpoch] = []


def register_epoch(epoch: HashEpoch) -> None:
    """
    Register a new hash epoch.
    
    Args:
        epoch: The epoch to register
        
    Raises:
        ValueError: If epoch violates invariants (overlap, non-monotonic, etc.)
    """
    global _EPOCH_REGISTRY
    
    # Validate epoch
    if epoch.start_block < 0:
        raise ValueError(f"Epoch start_block must be non-negative: {epoch.start_block}")
    
    if epoch.end_block is not None and epoch.end_block < epoch.start_block:
        raise ValueError(
            f"Epoch end_block ({epoch.end_block}) must be >= start_block ({epoch.start_block})"
        )
    
    # Check for overlaps with existing epochs
    for existing_epoch in _EPOCH_REGISTRY:
        # Check if new epoch overlaps with existing
        if existing_epoch.end_block is None:
            # Existing epoch is open-ended, must not overlap
            if epoch.start_block <= existing_epoch.start_block:
                raise ValueError(
                    f"Epoch starting at {epoch.start_block} overlaps with "
                    f"open-ended epoch starting at {existing_epoch.start_block}"
                )
        else:
            # Check for range overlap
            if not (
                epoch.start_block > existing_epoch.end_block or
                (epoch.end_block is not None and epoch.end_block < existing_epoch.start_block)
            ):
                raise ValueError(
                    f"Epoch [{epoch.start_block}, {epoch.end_block}] overlaps with "
                    f"existing epoch [{existing_epoch.start_block}, {existing_epoch.end_block}]"
                )
    
    # Check monotonicity: new epoch should start after all existing epochs
    if _EPOCH_REGISTRY:
        max_start = max(e.start_block for e in _EPOCH_REGISTRY)
        if epoch.start_block <= max_start:
            # Close the previous open-ended epoch
            for i, existing_epoch in enumerate(_EPOCH_REGISTRY):
                if existing_epoch.end_block is None and existing_epoch.start_block < epoch.start_block:
                    # Close this epoch at block before new epoch starts
                    closed_epoch = HashEpoch(
                        start_block=existing_epoch.start_block,
                        end_block=epoch.start_block - 1,
                        algorithm_id=existing_epoch.algorithm_id,
                        algorithm_name=existing_epoch.algorithm_name,
                        rule_version=existing_epoch.rule_version,
                        activation_timestamp=existing_epoch.activation_timestamp,
                        governance_hash=existing_epoch.governance_hash,
                    )
                    _EPOCH_REGISTRY[i] = closed_epoch
    
    # Register epoch
    _EPOCH_REGISTRY.append(epoch)
    
    # Sort by start_block
    _EPOCH_REGISTRY.sort(key=lambda e: e.start_block)


def get_epoch_for_block(block_number: int) -> Optional[HashEpoch]:
    """
    Get the hash epoch for a specific block number.
    
    Args:
        block_number: The block number to query
        
    Returns:
        The HashEpoch containing this block, or None if no epoch registered
    """
    for epoch in _EPOCH_REGISTRY:
        if epoch.start_block <= block_number:
            if epoch.end_block is None or block_number <= epoch.end_block:
                return epoch
    
    return None


def get_current_epoch() -> Optional[HashEpoch]:
    """
    Get the current (most recent) hash epoch.
    
    Returns:
        The current HashEpoch, or None if no epochs registered
    """
    if not _EPOCH_REGISTRY:
        return None
    
    # Return the epoch with the highest start_block
    return max(_EPOCH_REGISTRY, key=lambda e: e.start_block)


def list_epochs() -> List[HashEpoch]:
    """
    List all registered epochs in chronological order.
    
    Returns:
        List of all HashEpoch objects, sorted by start_block
    """
    return sorted(_EPOCH_REGISTRY, key=lambda e: e.start_block)


def get_epoch_by_governance_hash(governance_hash: str) -> Optional[HashEpoch]:
    """
    Get epoch by its governance proposal hash.
    
    Args:
        governance_hash: The governance proposal hash
        
    Returns:
        The HashEpoch with matching governance_hash, or None if not found
    """
    for epoch in _EPOCH_REGISTRY:
        if epoch.governance_hash == governance_hash:
            return epoch
    
    return None


def is_epoch_transition(block_number: int) -> bool:
    """
    Check if a block number is an epoch transition point.
    
    Args:
        block_number: The block number to check
        
    Returns:
        True if this block starts a new epoch, False otherwise
    """
    for epoch in _EPOCH_REGISTRY:
        if epoch.start_block == block_number:
            return True
    
    return False


def get_next_epoch_transition(current_block: int) -> Optional[int]:
    """
    Get the block number of the next epoch transition.
    
    Args:
        current_block: Current block number
        
    Returns:
        Block number of next epoch transition, or None if no future transitions
    """
    future_epochs = [
        e for e in _EPOCH_REGISTRY
        if e.start_block > current_block
    ]
    
    if not future_epochs:
        return None
    
    return min(e.start_block for e in future_epochs)


def initialize_genesis_epoch() -> None:
    """
    Initialize the genesis epoch (SHA-256, v1-legacy).
    
    This should be called once at system initialization.
    """
    global _EPOCH_REGISTRY
    
    # Clear registry
    _EPOCH_REGISTRY = []
    
    # Register genesis epoch
    genesis_epoch = HashEpoch(
        start_block=0,
        end_block=None,  # Open-ended until first transition
        algorithm_id=0x00,  # SHA-256
        algorithm_name="SHA-256",
        rule_version=ConsensusRuleVersion.V1_LEGACY.value,
        activation_timestamp=0.0,  # Genesis
        governance_hash="0x" + "00" * 32,  # Genesis has no governance proposal
    )
    
    register_epoch(genesis_epoch)


# Initialize genesis epoch on module load
initialize_genesis_epoch()
