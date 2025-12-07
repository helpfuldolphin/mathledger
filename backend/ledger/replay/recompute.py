"""
Root recomputation from canonical block payloads.

This module provides deterministic recomputation of attestation roots
from historical block data, enabling replay verification.

Invariant:
    For any historical block, recomputing roots from canonical payloads
    MUST produce identical R_t, U_t, H_t values as originally sealed.
"""

from typing import Any, Dict, List, Sequence, Tuple

from attestation.dual_root import (
    build_reasoning_attestation,
    build_ui_attestation,
    compute_composite_root,
)


def recompute_attestation_roots(
    canonical_statements: Sequence[Dict[str, Any]],
    canonical_proofs: Sequence[Dict[str, Any]],
    ui_events: Sequence[Any],
) -> Tuple[str, str, str]:
    """
    Recompute attestation roots from canonical block payloads.
    
    This function reconstructs the dual-root attestation (R_t, U_t, H_t)
    from the canonical payloads stored in a block. It uses the same
    attestation primitives as block sealing to ensure deterministic
    recomputation.
    
    Args:
        canonical_statements: Canonical statement payloads (currently unused,
            reserved for future statement-level attestation)
        canonical_proofs: Canonical proof payloads (reasoning events)
        ui_events: UI event payloads
        
    Returns:
        Tuple of (R_t, U_t, H_t):
        - R_t: Reasoning Merkle root
        - U_t: UI Merkle root
        - H_t: Composite attestation root SHA256(R_t || U_t)
        
    Raises:
        ValueError: If inputs are invalid or recomputation fails
        
    Example:
        >>> proofs = [{"statement": "p -> p", "method": "axiom"}]
        >>> ui_events = ["click_1"]
        >>> r_t, u_t, h_t = recompute_attestation_roots([], proofs, ui_events)
        >>> assert len(r_t) == 64  # SHA-256 hex
        >>> assert len(u_t) == 64
        >>> assert len(h_t) == 64
    """
    # Build reasoning tree from proofs (reasoning events)
    # This uses the same canonicalization and hashing as seal_block_with_dual_roots()
    reasoning_tree = build_reasoning_attestation(canonical_proofs)
    r_t = reasoning_tree.root
    
    # Build UI tree from UI events
    # Empty UI events produce deterministic empty tree root
    ui_tree = build_ui_attestation(ui_events)
    u_t = ui_tree.root
    
    # Compute composite root
    # H_t = SHA256(R_t || U_t) binds both streams cryptographically
    h_t = compute_composite_root(r_t, u_t)
    
    return r_t, u_t, h_t


def recompute_reasoning_root(canonical_proofs: Sequence[Dict[str, Any]]) -> str:
    """
    Recompute only the reasoning Merkle root (R_t).
    
    Args:
        canonical_proofs: Canonical proof payloads
        
    Returns:
        R_t: Reasoning Merkle root (64-char hex)
    """
    reasoning_tree = build_reasoning_attestation(canonical_proofs)
    return reasoning_tree.root


def recompute_ui_root(ui_events: Sequence[Any]) -> str:
    """
    Recompute only the UI Merkle root (U_t).
    
    Args:
        ui_events: UI event payloads
        
    Returns:
        U_t: UI Merkle root (64-char hex)
    """
    ui_tree = build_ui_attestation(ui_events)
    return ui_tree.root


__all__ = [
    "recompute_attestation_roots",
    "recompute_reasoning_root",
    "recompute_ui_root",
]
