"""
Block Validation Module

Implements comprehensive block validation logic for PQ migration.

Author: Manus-H
"""

from typing import Optional, Tuple

from basis.ledger.block_pq import BlockHeaderPQ
from basis.crypto.hash_versioned import (
    merkle_root_versioned,
    compute_dual_commitment,
)
from backend.consensus_pq.epoch import get_epoch_for_block
from backend.consensus_pq.rules import (
    ConsensusRuleVersion,
    get_consensus_rules_for_phase,
    validate_consensus_rules,
)
from backend.consensus_pq.prev_hash import validate_prev_hash_linkage, validate_genesis_block


def validate_merkle_root(
    block: BlockHeaderPQ,
    algorithm_id: int,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a block's Merkle root is correct for its statements.
    
    Args:
        block: The block header to validate
        algorithm_id: The hash algorithm ID to use
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Recompute Merkle root
    computed_root = merkle_root_versioned(
        block.statements,
        algorithm_id=algorithm_id
    )
    
    # Determine which root to check
    if algorithm_id == 0x00:  # SHA-256 (legacy)
        claimed_root = block.merkle_root
    else:  # PQ algorithm
        claimed_root = block.pq_merkle_root
    
    if computed_root != claimed_root:
        return False, (
            f"Merkle root mismatch for algorithm {algorithm_id:02x}: "
            f"expected {computed_root}, got {claimed_root}"
        )
    
    return True, None


def validate_dual_commitment(block: BlockHeaderPQ) -> Tuple[bool, Optional[str]]:
    """
    Validate that a block's dual commitment correctly binds legacy and PQ hashes.
    
    Args:
        block: The block header to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if block.dual_commitment is None:
        return False, "Dual commitment is None"
    
    if block.pq_merkle_root is None:
        return False, "PQ merkle root is None"
    
    if block.pq_algorithm is None:
        return False, "PQ algorithm is None"
    
    # Recompute dual commitment
    computed_commitment = compute_dual_commitment(
        legacy_hash=block.merkle_root,
        pq_hash=block.pq_merkle_root,
        pq_algorithm_id=block.pq_algorithm,
    )
    
    if computed_commitment != block.dual_commitment:
        return False, (
            f"Dual commitment mismatch: "
            f"expected {computed_commitment}, got {block.dual_commitment}"
        )
    
    return True, None


def validate_block_header(
    block: BlockHeaderPQ,
    prev_block: Optional[BlockHeaderPQ] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a block header according to consensus rules.
    
    This performs structural validation (fields present, types correct)
    but does not validate Merkle roots or prev_hash linkage.
    
    Args:
        block: The block header to validate
        prev_block: The previous block header (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Get epoch for this block
    epoch = get_epoch_for_block(block.block_number)
    if epoch is None:
        return False, f"No epoch registered for block {block.block_number}"
    
    # Get consensus rules
    rule_version = ConsensusRuleVersion(epoch.rule_version)
    rules = get_consensus_rules_for_phase(rule_version)
    
    # Validate consensus rules
    rules_valid, rules_error = validate_consensus_rules(block, rules)
    if not rules_valid:
        return False, rules_error
    
    # Validate block number continuity
    if prev_block is not None:
        if block.block_number != prev_block.block_number + 1:
            return False, (
                f"Block number discontinuity: "
                f"prev={prev_block.block_number}, current={block.block_number}"
            )
    
    # Validate timestamp (must be after previous block)
    if prev_block is not None:
        if block.timestamp <= prev_block.timestamp:
            return False, (
                f"Timestamp must increase: "
                f"prev={prev_block.timestamp}, current={block.timestamp}"
            )
    
    # Validate genesis block if block_number == 0
    if block.block_number == 0:
        genesis_valid, genesis_error = validate_genesis_block(block)
        if not genesis_valid:
            return False, genesis_error
    
    return True, None


def validate_block_full(
    block: BlockHeaderPQ,
    prev_block: Optional[BlockHeaderPQ] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Perform full validation of a block including Merkle roots and prev_hash.
    
    This is the main entry point for block validation.
    
    Args:
        block: The block header to validate
        prev_block: The previous block header (required for non-genesis blocks)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate header structure
    header_valid, header_error = validate_block_header(block, prev_block)
    if not header_valid:
        return False, header_error
    
    # Get epoch and rules
    epoch = get_epoch_for_block(block.block_number)
    rule_version = ConsensusRuleVersion(epoch.rule_version)
    rules = get_consensus_rules_for_phase(rule_version)
    
    # Validate legacy Merkle root
    if rules.legacy_fields_required:
        legacy_valid, legacy_error = validate_merkle_root(block, algorithm_id=0x00)
        if not legacy_valid:
            return False, f"Legacy Merkle root validation failed: {legacy_error}"
    
    # Validate PQ Merkle root
    if rules.pq_fields_required:
        pq_valid, pq_error = validate_merkle_root(block, algorithm_id=block.pq_algorithm)
        if not pq_valid:
            return False, f"PQ Merkle root validation failed: {pq_error}"
    
    # Validate dual commitment
    if rules.dual_commitment_required:
        commitment_valid, commitment_error = validate_dual_commitment(block)
        if not commitment_valid:
            return False, f"Dual commitment validation failed: {commitment_error}"
    
    # Validate prev_hash linkage
    if prev_block is not None and block.block_number > 0:
        linkage_valid, linkage_error = validate_prev_hash_linkage(block, prev_block)
        if not linkage_valid:
            return False, f"Prev-hash linkage validation failed: {linkage_error}"
    
    # All validations passed
    return True, None


def validate_block_batch(
    blocks: list[BlockHeaderPQ],
) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Validate a batch of consecutive blocks.
    
    Args:
        blocks: List of blocks to validate (must be consecutive)
        
    Returns:
        Tuple of (all_valid, error_message, failed_block_index)
        If all valid, error_message and failed_block_index are None
        If invalid, returns the index of the first failing block
    """
    if not blocks:
        return True, None, None
    
    # Validate first block
    first_valid, first_error = validate_block_full(blocks[0], prev_block=None)
    if not first_valid:
        return False, first_error, 0
    
    # Validate remaining blocks
    for i in range(1, len(blocks)):
        block = blocks[i]
        prev_block = blocks[i - 1]
        
        valid, error = validate_block_full(block, prev_block)
        if not valid:
            return False, error, i
    
    return True, None, None
