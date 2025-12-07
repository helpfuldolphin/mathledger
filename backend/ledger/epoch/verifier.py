"""
Epoch verification logic - verifies epoch root integrity.
"""

from typing import Dict, Any, List
from .sealer import compute_epoch_root

def verify_epoch_integrity(epoch_data: Dict[str, Any], blocks: List[Dict[str, Any]]) -> bool:
    """Verify epoch root matches recomputed root from blocks."""
    stored_epoch_root = epoch_data["epoch_root"]
    composite_roots = [block["composite_attestation_root"] for block in blocks]
    recomputed_epoch_root = compute_epoch_root(composite_roots)
    return stored_epoch_root == recomputed_epoch_root

def replay_epoch(epoch_data: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay epoch verification with detailed results."""
    is_valid = verify_epoch_integrity(epoch_data, blocks)
    return {
        "epoch_number": epoch_data["epoch_number"],
        "is_valid": is_valid,
        "block_count": len(blocks),
        "stored_epoch_root": epoch_data["epoch_root"],
    }
