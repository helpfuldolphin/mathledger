"""
Epoch sealing logic - aggregates block attestation roots into epoch commitments.
"""

from typing import List, Dict, Any
from substrate.crypto.hashing import merkle_root

DEFAULT_EPOCH_SIZE = 100

def compute_epoch_root(composite_roots: List[str]) -> str:
    """Compute epoch root from composite attestation roots."""
    if not composite_roots:
        raise ValueError("Cannot compute epoch root from empty list")
    return merkle_root(composite_roots)

def seal_epoch(epoch_number: int, blocks: List[Dict[str, Any]], system_id: str) -> Dict[str, Any]:
    """Seal an epoch with aggregated block roots."""
    if not blocks:
        raise ValueError(f"Cannot seal empty epoch {epoch_number}")
    
    composite_roots = [block["composite_attestation_root"] for block in blocks]
    epoch_root = compute_epoch_root(composite_roots)
    
    block_numbers = [block["block_number"] for block in blocks]
    return {
        "epoch_number": epoch_number,
        "epoch_root": epoch_root,
        "start_block_number": min(block_numbers),
        "end_block_number": max(block_numbers) + 1,
        "block_count": len(blocks),
        "system_id": system_id,
    }
