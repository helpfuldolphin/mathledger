"""
Replay engine for block verification.

This module orchestrates the replay verification process, coordinating
block fetching, root recomputation, and integrity checking.
"""

import json
from typing import Any, Dict, List, Optional

from .recompute import recompute_attestation_roots
from .checker import verify_block_integrity, IntegrityResult


def replay_block(block_data: Dict[str, Any]) -> IntegrityResult:
    """
    Replay a single block and verify integrity.
    
    This function reconstructs attestation roots from a historical block's
    canonical payloads and verifies they match the stored roots. This is
    the core invariant enforcement mechanism for ledger integrity.
    
    Args:
        block_data: Block data from database (must include canonical payloads)
            Required fields:
            - id: Block database ID
            - block_number: Block number in chain
            - reasoning_merkle_root: Stored R_t
            - ui_merkle_root: Stored U_t
            - composite_attestation_root: Stored H_t
            - canonical_proofs: Canonical proof payloads
            - attestation_metadata: Metadata with UI leaves
            
    Returns:
        IntegrityResult with verification status
        
    Raises:
        ValueError: If block data is missing required fields
        
    Example:
        >>> block = {
        ...     "id": 1,
        ...     "block_number": 1,
        ...     "reasoning_merkle_root": "a" * 64,
        ...     "ui_merkle_root": "b" * 64,
        ...     "composite_attestation_root": "c" * 64,
        ...     "canonical_proofs": [{"statement": "p -> p"}],
        ...     "attestation_metadata": {"ui_leaves": []},
        ... }
        >>> result = replay_block(block)
        >>> assert isinstance(result, IntegrityResult)
    """
    # Extract block metadata
    block_id = block_data.get("id")
    block_number = block_data.get("block_number")
    
    if not block_id or block_number is None:
        raise ValueError("Block data missing id or block_number")
    
    # Extract stored roots
    stored_r_t = block_data.get("reasoning_merkle_root")
    stored_u_t = block_data.get("ui_merkle_root")
    stored_h_t = block_data.get("composite_attestation_root")
    
    if not stored_r_t or not stored_u_t or not stored_h_t:
        raise ValueError(
            f"Block {block_id} missing attestation roots. "
            f"R_t={stored_r_t}, U_t={stored_u_t}, H_t={stored_h_t}"
        )
    
    # Extract canonical payloads
    canonical_statements = block_data.get("canonical_statements", [])
    canonical_proofs = block_data.get("canonical_proofs", [])
    
    # Handle both dict and list formats for canonical_proofs
    if isinstance(canonical_proofs, dict):
        canonical_proofs = canonical_proofs.get("proofs", [])
    
    # Extract UI events from attestation metadata
    attestation_metadata = block_data.get("attestation_metadata", {})
    if isinstance(attestation_metadata, str):
        try:
            attestation_metadata = json.loads(attestation_metadata)
        except json.JSONDecodeError:
            attestation_metadata = {}
    
    ui_leaves = attestation_metadata.get("ui_leaves", [])
    ui_events = [leaf.get("canonical_value", leaf) for leaf in ui_leaves]
    
    # Recompute roots from canonical payloads
    try:
        recomputed_r_t, recomputed_u_t, recomputed_h_t = recompute_attestation_roots(
            canonical_statements=canonical_statements,
            canonical_proofs=canonical_proofs,
            ui_events=ui_events,
        )
    except Exception as e:
        raise ValueError(f"Failed to recompute roots for block {block_id}: {e}")
    
    # Verify integrity
    result = verify_block_integrity(
        block_id=block_id,
        block_number=block_number,
        stored_r_t=stored_r_t,
        stored_u_t=stored_u_t,
        stored_h_t=stored_h_t,
        recomputed_r_t=recomputed_r_t,
        recomputed_u_t=recomputed_u_t,
        recomputed_h_t=recomputed_h_t,
    )
    
    return result


def replay_chain(
    blocks: List[Dict[str, Any]],
    stop_on_failure: bool = False,
) -> Dict[str, Any]:
    """
    Replay multiple blocks and aggregate results.
    
    This function replays a sequence of blocks (e.g., entire chain or range)
    and aggregates the verification results. It's used for bulk integrity
    verification and CI checks.
    
    Args:
        blocks: List of block data dictionaries
        stop_on_failure: If True, stop on first failure
        
    Returns:
        Dictionary with aggregated results:
        - total_blocks: Total blocks replayed
        - valid_blocks: Number of valid blocks
        - invalid_blocks: Number of invalid blocks
        - success_rate: Fraction of valid blocks
        - results: List of IntegrityResult dictionaries
        - first_failure: First failure details (if any)
        
    Example:
        >>> blocks = [block1, block2, block3]
        >>> result = replay_chain(blocks)
        >>> assert result["total_blocks"] == 3
        >>> assert result["success_rate"] >= 0.0
    """
    results = []
    valid_count = 0
    invalid_count = 0
    first_failure = None
    
    for block in blocks:
        try:
            result = replay_block(block)
            results.append(result.to_dict())
            
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                if first_failure is None:
                    first_failure = result.to_dict()
                
                if stop_on_failure:
                    break
                    
        except Exception as e:
            error_result = {
                "block_id": block.get("id"),
                "block_number": block.get("block_number"),
                "is_valid": False,
                "error": str(e),
            }
            results.append(error_result)
            invalid_count += 1
            
            if first_failure is None:
                first_failure = error_result
            
            if stop_on_failure:
                break
    
    total = len(blocks)
    success_rate = valid_count / total if total > 0 else 0.0
    
    return {
        "total_blocks": total,
        "valid_blocks": valid_count,
        "invalid_blocks": invalid_count,
        "success_rate": success_rate,
        "results": results,
        "first_failure": first_failure,
        "is_chain_valid": invalid_count == 0,
    }


def replay_block_range(
    blocks: List[Dict[str, Any]],
    start_number: int,
    end_number: int,
    stop_on_failure: bool = False,
) -> Dict[str, Any]:
    """
    Replay a range of blocks by block number.
    
    Args:
        blocks: List of all blocks
        start_number: Start block number (inclusive)
        end_number: End block number (inclusive)
        stop_on_failure: If True, stop on first failure
        
    Returns:
        Aggregated replay results for the range
    """
    filtered_blocks = [
        b for b in blocks
        if start_number <= b.get("block_number", 0) <= end_number
    ]
    
    return replay_chain(filtered_blocks, stop_on_failure=stop_on_failure)


__all__ = [
    "replay_block",
    "replay_chain",
    "replay_block_range",
]
