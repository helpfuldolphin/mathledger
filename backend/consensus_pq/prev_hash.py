"""
Prev-Hash Validation Module

Validates prev_hash linkage for both legacy and PQ hash chains.

Author: Manus-H
"""

from typing import Optional, Tuple

from basis.ledger.block_pq import BlockHeaderPQ
from basis.crypto.hash_versioned import hash_block_versioned
from backend.consensus_pq.epoch import get_epoch_for_block
from backend.consensus_pq.rules import ConsensusRuleVersion, get_consensus_rules_for_phase


def compute_prev_hash(
    prev_block: BlockHeaderPQ,
    algorithm_id: int,
) -> str:
    """
    Compute the prev_hash for a block using a specific algorithm.
    
    Args:
        prev_block: The previous block header
        algorithm_id: The hash algorithm ID to use
        
    Returns:
        The computed prev_hash as a hex string
    """
    return hash_block_versioned(prev_block, algorithm_id=algorithm_id)


def validate_prev_hash_linkage(
    block: BlockHeaderPQ,
    prev_block: BlockHeaderPQ,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a block correctly links to its predecessor.
    
    This validates both legacy and PQ prev_hash chains based on the
    consensus rules for the block's epoch.
    
    Args:
        block: The current block header
        prev_block: The previous block header
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
        If invalid, error_message describes the violation
    """
    # Get epochs for both blocks
    block_epoch = get_epoch_for_block(block.block_number)
    prev_epoch = get_epoch_for_block(prev_block.block_number)
    
    if block_epoch is None:
        return False, f"No epoch registered for block {block.block_number}"
    
    if prev_epoch is None:
        return False, f"No epoch registered for prev block {prev_block.block_number}"
    
    # Get consensus rules for current block
    rule_version = ConsensusRuleVersion(block_epoch.rule_version)
    rules = get_consensus_rules_for_phase(rule_version)
    
    # Validate legacy prev_hash
    if rules.legacy_fields_required:
        expected_legacy_prev = compute_prev_hash(
            prev_block,
            algorithm_id=prev_epoch.algorithm_id
        )
        
        if block.prev_hash != expected_legacy_prev:
            return False, (
                f"Legacy prev_hash mismatch: "
                f"expected {expected_legacy_prev}, got {block.prev_hash}"
            )
    
    # Validate PQ prev_hash
    if rules.pq_fields_required:
        if block.pq_prev_hash is None:
            return False, "PQ prev_hash required but missing"
        
        # Compute expected PQ prev_hash
        expected_pq_prev = compute_prev_hash(
            prev_block,
            algorithm_id=block.pq_algorithm
        )
        
        if block.pq_prev_hash != expected_pq_prev:
            return False, (
                f"PQ prev_hash mismatch: "
                f"expected {expected_pq_prev}, got {block.pq_prev_hash}"
            )
    
    # All checks passed
    return True, None


def validate_dual_prev_hash(
    block: BlockHeaderPQ,
    prev_block: BlockHeaderPQ,
) -> Tuple[bool, bool, Optional[str], Optional[str]]:
    """
    Validate both legacy and PQ prev_hash chains independently.
    
    This is useful for drift detection and monitoring.
    
    Args:
        block: The current block header
        prev_block: The previous block header
        
    Returns:
        Tuple of (legacy_valid, pq_valid, legacy_error, pq_error)
    """
    # Get epochs
    block_epoch = get_epoch_for_block(block.block_number)
    prev_epoch = get_epoch_for_block(prev_block.block_number)
    
    if block_epoch is None or prev_epoch is None:
        return False, False, "Epoch not found", "Epoch not found"
    
    # Validate legacy chain
    legacy_valid = True
    legacy_error = None
    
    if block.prev_hash is not None:
        expected_legacy_prev = compute_prev_hash(
            prev_block,
            algorithm_id=prev_epoch.algorithm_id
        )
        
        if block.prev_hash != expected_legacy_prev:
            legacy_valid = False
            legacy_error = f"Legacy prev_hash mismatch"
    
    # Validate PQ chain
    pq_valid = True
    pq_error = None
    
    if block.pq_prev_hash is not None:
        if block.pq_algorithm is None:
            pq_valid = False
            pq_error = "PQ algorithm not specified"
        else:
            expected_pq_prev = compute_prev_hash(
                prev_block,
                algorithm_id=block.pq_algorithm
            )
            
            if block.pq_prev_hash != expected_pq_prev:
                pq_valid = False
                pq_error = f"PQ prev_hash mismatch"
    
    return legacy_valid, pq_valid, legacy_error, pq_error


def validate_genesis_block(block: BlockHeaderPQ) -> Tuple[bool, Optional[str]]:
    """
    Validate that a block is a valid genesis block.
    
    Genesis blocks have special prev_hash rules (typically all zeros).
    
    Args:
        block: The block header to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if block.block_number != 0:
        return False, f"Genesis block must have block_number=0, got {block.block_number}"
    
    # Genesis prev_hash should be all zeros
    expected_genesis_prev = "0x" + "00" * 32
    
    if block.prev_hash != expected_genesis_prev:
        return False, (
            f"Genesis prev_hash must be {expected_genesis_prev}, "
            f"got {block.prev_hash}"
        )
    
    return True, None
