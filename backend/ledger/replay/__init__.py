"""
Replay verification system for ledger integrity.

This module provides deterministic replay of historical blocks,
recomputing attestation roots and verifying they match stored values.

Core Invariant:
    For any historical block, replaying from canonical payloads MUST
    produce identical R_t, U_t, H_t as originally sealed.

Usage:
    from backend.ledger.replay import replay_block, replay_chain
    
    # Replay single block
    result = replay_block(block_data)
    if not result.is_valid:
        print(f"Block {result.block_number} failed: {result.error}")
    
    # Replay chain
    chain_result = replay_chain(blocks)
    print(f"Valid: {chain_result['valid_blocks']}/{chain_result['total_blocks']}")
"""

from .engine import replay_block, replay_chain, replay_block_range
from .recompute import (
    recompute_attestation_roots,
    recompute_reasoning_root,
    recompute_ui_root,
)
from .checker import (
    IntegrityResult,
    verify_block_integrity,
    verify_composite_consistency,
)

__all__ = [
    # Engine
    "replay_block",
    "replay_chain",
    "replay_block_range",
    # Recomputation
    "recompute_attestation_roots",
    "recompute_reasoning_root",
    "recompute_ui_root",
    # Verification
    "IntegrityResult",
    "verify_block_integrity",
    "verify_composite_consistency",
]
