"""
Historical Verification Compatibility Layer.

This module provides epoch-aware verification functions that automatically
use the correct hash algorithm based on block numbers. This ensures that
historical blocks remain verifiable even after hash algorithm migrations.

Key Features:
- Epoch-based algorithm resolution
- Cross-epoch chain verification
- Backward-compatible with legacy blocks
- Merkle proof verification with versioned algorithms

Security Invariants:
- Historical roots must be verifiable under their original algorithm IDs
- Epoch boundaries are immutable
- Algorithm selection is deterministic based on block number
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from basis.core import Block, BlockHeader, HexDigest
from basis.crypto.hash_registry import (
    get_canonical_algorithm,
    get_epoch_for_block,
)
from basis.crypto.hash_versioned import (
    hash_block_versioned,
    merkle_root_versioned,
    verify_merkle_proof_versioned,
)
from basis.ledger.block import block_json
from basis.ledger.block_pq import BlockPQ, block_pq_json


def verify_merkle_root_historical(
    block_number: int,
    leaves: Sequence[str],
    expected_root: HexDigest,
) -> bool:
    """
    Verify Merkle root using the algorithm that was canonical at the given block.
    
    This function automatically selects the correct hash algorithm based on
    the block number's epoch, ensuring historical blocks remain verifiable.
    
    Args:
        block_number: Block number to determine epoch
        leaves: Statement leaves to hash
        expected_root: Expected Merkle root
        
    Returns:
        True if computed root matches expected root
        
    Examples:
        >>> # Block 100 uses SHA-256 (epoch 0)
        >>> verify_merkle_root_historical(100, ["p->p"], "abc...123")
        True
        
        >>> # Block 1000000 might use PQ1 (epoch 1)
        >>> verify_merkle_root_historical(1000000, ["p->p"], "def...456")
        True
    """
    # Determine canonical algorithm for this block
    epoch = get_epoch_for_block(block_number)
    algorithm = get_canonical_algorithm(block_number)
    
    # Compute Merkle root with correct algorithm
    computed_root = merkle_root_versioned(
        leaves,
        algorithm_id=algorithm.algorithm_id,
    )
    
    return computed_root == expected_root


def verify_merkle_proof_historical(
    block_number: int,
    leaf: str,
    proof: Sequence[Tuple[HexDigest, bool]],
    expected_root: HexDigest,
) -> bool:
    """
    Verify Merkle proof using the algorithm that was canonical at the given block.
    
    Args:
        block_number: Block number to determine epoch
        leaf: Leaf value to verify
        proof: Merkle proof (sibling_hash, sibling_is_left) tuples
        expected_root: Expected Merkle root
        
    Returns:
        True if proof is valid
    """
    # Determine canonical algorithm for this block
    algorithm = get_canonical_algorithm(block_number)
    
    # Verify proof with correct algorithm
    return verify_merkle_proof_versioned(
        leaf,
        proof,
        expected_root,
        algorithm_id=algorithm.algorithm_id,
    )


def hash_block_header_historical(
    header: Union[BlockHeader, BlockPQ],
) -> HexDigest:
    """
    Hash a block header using the algorithm that was canonical at its block number.
    
    Args:
        header: Block header to hash
        
    Returns:
        Block header hash
    """
    # Determine canonical algorithm for this block
    algorithm = get_canonical_algorithm(header.block_number)
    
    # Serialize header (use appropriate method based on type)
    if isinstance(header, BlockPQ):
        from basis.ledger.block_pq import block_pq_to_dict
        import json
        serialized = json.dumps(
            block_pq_to_dict(BlockPQ(header=header, statements=tuple())),
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        from basis.ledger.block import block_to_dict
        import json
        serialized = json.dumps(
            block_to_dict(Block(header=header, statements=tuple())),
            sort_keys=True,
            separators=(",", ":"),
        )
    
    # Hash with correct algorithm
    return hash_block_versioned(
        serialized,
        algorithm_id=algorithm.algorithm_id,
    )


def verify_block_chain(
    blocks: Sequence[Union[Block, BlockPQ]],
) -> Tuple[bool, Optional[str]]:
    """
    Verify a chain of blocks that may span multiple hash epochs.
    
    This function verifies:
    1. Each block's Merkle root is valid
    2. Each block's prev_hash correctly links to previous block
    3. Correct hash algorithms are used based on epochs
    
    Args:
        blocks: Sequence of blocks to verify (must be in order)
        
    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if chain is valid
        - (False, error_message) if chain is invalid
        
    Examples:
        >>> blocks = [block0, block1, block2]
        >>> is_valid, error = verify_block_chain(blocks)
        >>> if not is_valid:
        ...     print(f"Chain invalid: {error}")
    """
    if not blocks:
        return True, None
    
    for i, block in enumerate(blocks):
        header = block.header
        statements = block.statements
        
        # Verify Merkle root
        if not verify_merkle_root_historical(
            header.block_number,
            statements,
            header.merkle_root,
        ):
            return False, f"Block {header.block_number}: Invalid Merkle root"
        
        # Verify chain linkage (skip genesis block)
        if i > 0:
            prev_block = blocks[i - 1]
            prev_header = prev_block.header
            
            # Hash previous block header with its canonical algorithm
            computed_prev_hash = hash_block_header_historical(prev_header)
            
            if computed_prev_hash != header.prev_hash:
                return False, (
                    f"Block {header.block_number}: Invalid prev_hash. "
                    f"Expected {computed_prev_hash}, got {header.prev_hash}"
                )
        
        # Verify dual commitment if present (for PQ blocks)
        if isinstance(header, BlockPQ) and header.has_dual_commitment():
            if not header.verify_dual_commitment():
                return False, f"Block {header.block_number}: Invalid dual commitment"
    
    return True, None


def verify_block_chain_segment(
    blocks: Sequence[Union[Block, BlockPQ]],
    expected_first_prev_hash: Optional[HexDigest] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Verify a segment of the block chain (not necessarily from genesis).
    
    This is useful for verifying a portion of the chain without needing
    the entire history.
    
    Args:
        blocks: Sequence of blocks to verify (must be in order)
        expected_first_prev_hash: Expected prev_hash of first block (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not blocks:
        return True, None
    
    # Verify first block's prev_hash if provided
    if expected_first_prev_hash is not None:
        first_header = blocks[0].header
        if first_header.prev_hash != expected_first_prev_hash:
            return False, (
                f"Block {first_header.block_number}: Invalid prev_hash. "
                f"Expected {expected_first_prev_hash}, got {first_header.prev_hash}"
            )
    
    # Verify the rest of the chain
    return verify_block_chain(blocks)


def get_block_hash(block: Union[Block, BlockPQ]) -> HexDigest:
    """
    Compute the hash of a block using the canonical algorithm for its epoch.
    
    Args:
        block: Block to hash
        
    Returns:
        Block hash
    """
    return hash_block_header_historical(block.header)


def verify_epoch_transition(
    last_legacy_block: Union[Block, BlockPQ],
    first_pq_block: BlockPQ,
) -> Tuple[bool, Optional[str]]:
    """
    Verify a hash algorithm epoch transition.
    
    During epoch transitions, the first block of the new epoch must:
    1. Have a valid dual commitment
    2. Correctly link to the last block of the previous epoch
    3. Use the new canonical algorithm
    
    Args:
        last_legacy_block: Last block before epoch transition
        first_pq_block: First block after epoch transition
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Verify first PQ block has dual commitment
    if not first_pq_block.header.has_dual_commitment():
        return False, "First PQ block must have dual commitment"
    
    # Verify dual commitment is valid
    if not first_pq_block.header.verify_dual_commitment():
        return False, "First PQ block has invalid dual commitment"
    
    # Verify linkage
    legacy_hash = hash_block_header_historical(last_legacy_block.header)
    if first_pq_block.header.prev_hash != legacy_hash:
        return False, (
            f"PQ block prev_hash mismatch. "
            f"Expected {legacy_hash}, got {first_pq_block.header.prev_hash}"
        )
    
    # Verify block numbers are consecutive
    if first_pq_block.header.block_number != last_legacy_block.header.block_number + 1:
        return False, "Block numbers must be consecutive at epoch transition"
    
    return True, None


def verify_block_statements(
    block: Union[Block, BlockPQ],
) -> Tuple[bool, Optional[str]]:
    """
    Verify that a block's statements match its Merkle root.
    
    Args:
        block: Block to verify
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not verify_merkle_root_historical(
        block.header.block_number,
        block.statements,
        block.header.merkle_root,
    ):
        return False, f"Block {block.header.block_number}: Statements do not match Merkle root"
    
    return True, None


__all__ = [
    "verify_merkle_root_historical",
    "verify_merkle_proof_historical",
    "hash_block_header_historical",
    "verify_block_chain",
    "verify_block_chain_segment",
    "get_block_hash",
    "verify_epoch_transition",
    "verify_block_statements",
]
