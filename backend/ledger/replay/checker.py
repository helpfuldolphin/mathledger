"""
Integrity verification for replayed blocks.

This module provides verification logic to compare recomputed attestation
roots against stored values, detecting any discrepancies that would indicate
ledger corruption or non-deterministic sealing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IntegrityResult:
    """
    Result of block integrity verification.
    
    Attributes:
        block_id: Database block ID
        block_number: Block number in chain
        is_valid: True if all roots match
        stored_r_t: Stored reasoning root
        stored_u_t: Stored UI root
        stored_h_t: Stored composite root
        recomputed_r_t: Recomputed reasoning root
        recomputed_u_t: Recomputed UI root
        recomputed_h_t: Recomputed composite root
        r_t_match: True if R_t matches
        u_t_match: True if U_t matches
        h_t_match: True if H_t matches
        error: Error message if verification failed
    """
    
    block_id: int
    block_number: int
    is_valid: bool
    
    stored_r_t: str
    stored_u_t: str
    stored_h_t: str
    
    recomputed_r_t: str
    recomputed_u_t: str
    recomputed_h_t: str
    
    r_t_match: bool
    u_t_match: bool
    h_t_match: bool
    
    error: Optional[str] = None
    
    def to_dict(self):
        """Serialize result to dictionary for JSON export."""
        return {
            "block_id": self.block_id,
            "block_number": self.block_number,
            "is_valid": self.is_valid,
            "stored_roots": {
                "r_t": self.stored_r_t,
                "u_t": self.stored_u_t,
                "h_t": self.stored_h_t,
            },
            "recomputed_roots": {
                "r_t": self.recomputed_r_t,
                "u_t": self.recomputed_u_t,
                "h_t": self.recomputed_h_t,
            },
            "matches": {
                "r_t": self.r_t_match,
                "u_t": self.u_t_match,
                "h_t": self.h_t_match,
            },
            "error": self.error,
        }
    
    def __repr__(self):
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return f"<IntegrityResult block={self.block_number} {status}>"


def verify_block_integrity(
    block_id: int,
    block_number: int,
    stored_r_t: str,
    stored_u_t: str,
    stored_h_t: str,
    recomputed_r_t: str,
    recomputed_u_t: str,
    recomputed_h_t: str,
) -> IntegrityResult:
    """
    Verify that recomputed roots match stored roots.
    
    This function performs byte-level comparison of attestation roots,
    detecting any discrepancies that would indicate:
    - Non-deterministic sealing
    - Ledger corruption
    - Canonicalization bugs
    - Hash algorithm changes
    
    Args:
        block_id: Database block ID
        block_number: Block number in chain
        stored_r_t: Stored reasoning root (from blocks table)
        stored_u_t: Stored UI root (from blocks table)
        stored_h_t: Stored composite root (from blocks table)
        recomputed_r_t: Recomputed reasoning root
        recomputed_u_t: Recomputed UI root
        recomputed_h_t: Recomputed composite root
        
    Returns:
        IntegrityResult with detailed comparison
        
    Example:
        >>> result = verify_block_integrity(
        ...     block_id=1,
        ...     block_number=1,
        ...     stored_r_t="a" * 64,
        ...     stored_u_t="b" * 64,
        ...     stored_h_t="c" * 64,
        ...     recomputed_r_t="a" * 64,
        ...     recomputed_u_t="b" * 64,
        ...     recomputed_h_t="c" * 64,
        ... )
        >>> assert result.is_valid is True
    """
    # Perform exact string comparison (case-sensitive hex)
    r_t_match = stored_r_t == recomputed_r_t
    u_t_match = stored_u_t == recomputed_u_t
    h_t_match = stored_h_t == recomputed_h_t
    
    # Block is valid only if ALL roots match
    is_valid = r_t_match and u_t_match and h_t_match
    
    # Generate error message if invalid
    error = None
    if not is_valid:
        mismatches = []
        if not r_t_match:
            mismatches.append("R_t (reasoning root)")
        if not u_t_match:
            mismatches.append("U_t (UI root)")
        if not h_t_match:
            mismatches.append("H_t (composite root)")
        error = f"Root mismatch: {', '.join(mismatches)}"
    
    return IntegrityResult(
        block_id=block_id,
        block_number=block_number,
        is_valid=is_valid,
        stored_r_t=stored_r_t,
        stored_u_t=stored_u_t,
        stored_h_t=stored_h_t,
        recomputed_r_t=recomputed_r_t,
        recomputed_u_t=recomputed_u_t,
        recomputed_h_t=recomputed_h_t,
        r_t_match=r_t_match,
        u_t_match=u_t_match,
        h_t_match=h_t_match,
        error=error,
    )


def verify_composite_consistency(r_t: str, u_t: str, h_t: str) -> bool:
    """
    Verify that composite root H_t is consistent with R_t and U_t.
    
    This checks that H_t = SHA256(R_t || U_t), ensuring the composite
    attestation properly binds both reasoning and UI streams.
    
    Args:
        r_t: Reasoning root
        u_t: UI root
        h_t: Composite root
        
    Returns:
        True if H_t matches SHA256(R_t || U_t)
    """
    from attestation.dual_root import compute_composite_root
    
    try:
        expected_h_t = compute_composite_root(r_t, u_t)
        return h_t == expected_h_t
    except Exception:
        return False


__all__ = [
    "IntegrityResult",
    "verify_block_integrity",
    "verify_composite_consistency",
]
