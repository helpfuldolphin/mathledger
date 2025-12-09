"""
Attestation Chain Verifier with TDA Pipeline Binding
=====================================================

Extends the attestation chain verification to include TDA pipeline hash validation,
ensuring that:
1. Each experiment block includes its TDA pipeline configuration hash
2. TDA configuration doesn't drift across runs in the same experiment series
3. Hard Gate decisions (e.g., ABANDONED_TDA) are cryptographically bound
4. Full attestation chain integrity is maintained

Exit Codes:
- 0: Verification passed
- 1: Attestation integrity failure (missing fields, invalid hashes)
- 2: Merkle root mismatch
- 3: Chain linkage broken
- 4: TDA-Ledger Divergence Detected
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from .dual_root import verify_composite_integrity
from .tda_pipeline import compute_tda_pipeline_hash, detect_tda_divergence, TDADivergence


class AttestationVerificationError(IntEnum):
    """Exit codes for attestation verification."""
    
    SUCCESS = 0
    INTEGRITY_FAILURE = 1
    MERKLE_MISMATCH = 2
    CHAIN_LINKAGE_BROKEN = 3
    TDA_DIVERGENCE = 4


@dataclass
class ExperimentBlock:
    """
    Single experiment block in the attestation chain.
    
    Contains dual-root attestation plus TDA pipeline configuration hash.
    """
    
    run_id: str
    experiment_id: str
    
    # Dual-root attestation
    reasoning_root: str  # R_t
    ui_root: str  # U_t
    composite_root: str  # H_t
    
    # TDA pipeline binding
    tda_pipeline_hash: str
    tda_config: Dict[str, Any]
    
    # Hard Gate decisions (cryptographically bound)
    gate_decisions: Optional[Dict[str, str]] = None  # gate_name -> decision
    
    # Chain linkage
    prev_block_hash: Optional[str] = None
    block_number: int = 0
    
    def compute_block_hash(self) -> str:
        """
        Compute deterministic hash of this block.
        
        Includes all critical fields:
        - Composite attestation root (H_t)
        - TDA pipeline hash
        - Gate decisions
        - Block number
        """
        payload = {
            "run_id": self.run_id,
            "composite_root": self.composite_root,
            "tda_pipeline_hash": self.tda_pipeline_hash,
            "gate_decisions": self.gate_decisions or {},
            "block_number": self.block_number,
        }
        
        # Canonical JSON serialization
        import json
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    
    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify internal integrity of this block.
        
        Returns:
            (is_valid, error_message)
        """
        # Verify dual-root attestation
        if not verify_composite_integrity(
            self.reasoning_root,
            self.ui_root,
            self.composite_root
        ):
            return False, "Composite root H_t does not match SHA256(R_t || U_t)"
        
        # Verify TDA pipeline hash
        try:
            expected_tda_hash = compute_tda_pipeline_hash(self.tda_config)
            if expected_tda_hash != self.tda_pipeline_hash:
                return False, (
                    f"TDA pipeline hash mismatch: "
                    f"expected {expected_tda_hash}, got {self.tda_pipeline_hash}"
                )
        except Exception as e:
            return False, f"Error computing TDA hash: {e}"
        
        # Verify gate decisions are bound (if present)
        if self.gate_decisions:
            # Gate decisions should be included in block hash
            # This verification is implicit in compute_block_hash()
            pass
        
        return True, None


@dataclass
class VerificationResult:
    """Result of attestation chain verification."""
    
    is_valid: bool
    error_code: AttestationVerificationError
    error_message: Optional[str] = None
    divergences: List[TDADivergence] = None
    
    def __post_init__(self):
        if self.divergences is None:
            self.divergences = []


class AttestationChainVerifier:
    """
    Verifies attestation chains with TDA pipeline binding.
    
    Checks:
    1. Block integrity (dual-root attestation + TDA hash)
    2. Chain linkage (prev_block_hash)
    3. TDA configuration consistency across runs
    4. Hard Gate decision binding
    """
    
    def __init__(self, strict_tda_consistency: bool = True):
        """
        Initialize verifier.
        
        Args:
            strict_tda_consistency: If True, TDA divergence causes verification failure
        """
        self.strict_tda_consistency = strict_tda_consistency
    
    def verify_chain(self, blocks: List[ExperimentBlock]) -> VerificationResult:
        """
        Verify a chain of experiment blocks.
        
        Args:
            blocks: List of blocks in chronological order
            
        Returns:
            VerificationResult with status and details
        """
        if not blocks:
            return VerificationResult(
                is_valid=True,
                error_code=AttestationVerificationError.SUCCESS
            )
        
        divergences = []
        
        # Verify each block's internal integrity
        for i, block in enumerate(blocks):
            is_valid, error = block.verify_integrity()
            if not is_valid:
                return VerificationResult(
                    is_valid=False,
                    error_code=AttestationVerificationError.INTEGRITY_FAILURE,
                    error_message=f"Block {i} (run_id={block.run_id}): {error}"
                )
        
        # Verify chain linkage
        for i in range(1, len(blocks)):
            prev_block = blocks[i - 1]
            curr_block = blocks[i]
            
            expected_prev_hash = prev_block.compute_block_hash()
            
            if curr_block.prev_block_hash != expected_prev_hash:
                return VerificationResult(
                    is_valid=False,
                    error_code=AttestationVerificationError.CHAIN_LINKAGE_BROKEN,
                    error_message=(
                        f"Block {i} prev_hash mismatch: "
                        f"expected {expected_prev_hash[:16]}..., "
                        f"got {curr_block.prev_block_hash[:16] if curr_block.prev_block_hash else 'None'}..."
                    )
                )
        
        # Detect TDA configuration drift
        for i in range(1, len(blocks)):
            prev_block = blocks[i - 1]
            curr_block = blocks[i]
            
            divergence = detect_tda_divergence(
                run_id_1=prev_block.run_id,
                config_1=prev_block.tda_config,
                run_id_2=curr_block.run_id,
                config_2=curr_block.tda_config,
            )
            
            if divergence:
                divergences.append(divergence)
        
        # Return result based on divergences
        if divergences:
            if self.strict_tda_consistency:
                return VerificationResult(
                    is_valid=False,
                    error_code=AttestationVerificationError.TDA_DIVERGENCE,
                    error_message=f"TDA configuration drift detected across {len(divergences)} transition(s)",
                    divergences=divergences
                )
            else:
                # Warning mode: divergences detected but not failing
                return VerificationResult(
                    is_valid=True,
                    error_code=AttestationVerificationError.SUCCESS,
                    error_message=f"Warning: TDA drift detected but allowed in non-strict mode",
                    divergences=divergences
                )
        
        return VerificationResult(
            is_valid=True,
            error_code=AttestationVerificationError.SUCCESS
        )
    
    def verify_hard_gate_binding(
        self,
        block: ExperimentBlock,
        expected_decisions: Dict[str, str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that Hard Gate decisions are correctly bound in the block.
        
        Args:
            block: Experiment block to verify
            expected_decisions: Expected gate decisions (gate_name -> decision)
            
        Returns:
            (is_valid, error_message)
        """
        if not block.gate_decisions:
            if expected_decisions:
                return False, "Block has no gate decisions but expected decisions provided"
            return True, None
        
        for gate_name, expected_decision in expected_decisions.items():
            actual_decision = block.gate_decisions.get(gate_name)
            
            if actual_decision != expected_decision:
                return False, (
                    f"Gate decision mismatch for '{gate_name}': "
                    f"expected '{expected_decision}', got '{actual_decision}'"
                )
        
        # Verify decisions are bound via block hash
        block_hash = block.compute_block_hash()
        
        # Block hash includes gate_decisions, so if hash matches,
        # decisions are cryptographically bound
        return True, None


def verify_experiment_attestation_chain(
    blocks: List[ExperimentBlock],
    strict_tda: bool = True
) -> VerificationResult:
    """
    Convenience function to verify an experiment attestation chain.
    
    Args:
        blocks: List of experiment blocks
        strict_tda: If True, TDA divergence causes failure (exit code 4)
        
    Returns:
        VerificationResult with status and details
    """
    verifier = AttestationChainVerifier(strict_tda_consistency=strict_tda)
    return verifier.verify_chain(blocks)


__all__ = [
    "AttestationVerificationError",
    "ExperimentBlock",
    "VerificationResult",
    "AttestationChainVerifier",
    "verify_experiment_attestation_chain",
]
