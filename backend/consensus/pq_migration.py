"""
Post-Quantum (PQ) Migration Module

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Consensus Runtime Activation
Date: 2025-12-06

Purpose:
    Orchestrate post-quantum cryptographic migration from SHA-256 to SHA-3.
    
    Migration Phases:
    1. Pre-Migration: All blocks use SHA-256
    2. Dual-Commitment: New blocks commit to both SHA-256 and SHA-3
    3. Pure SHA-3: New blocks use only SHA-3 (legacy blocks remain SHA-256)
    
    This module manages phase transitions, validates migration invariants,
    and ensures replay determinism across heterogeneous hash chains.

Design Principles:
    1. Backward Compatible: Legacy blocks remain verifiable
    2. Forward Compatible: New blocks support PQ algorithms
    3. Deterministic: Same migration state → same behavior
    4. Auditable: All migration decisions logged

Integration Points:
    - Manus-H: Hash algorithm abstraction layer
    - Consensus Rules: Hash version transition validation
    - Replay Verification: Cross-algorithm replay support
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class MigrationPhase(Enum):
    """PQ migration phases."""
    PRE_MIGRATION = "pre_migration"          # All SHA-256
    DUAL_COMMITMENT = "dual_commitment"      # SHA-256 + SHA-3
    PURE_SHA3 = "pure_sha3"                  # All SHA-3 (new blocks)


class HashVersion(Enum):
    """Hash algorithm versions."""
    SHA256_V1 = "sha256-v1"    # Legacy SHA-256
    DUAL_V1 = "dual-v1"        # Dual-commitment (SHA-256 + SHA-3)
    SHA3_V1 = "sha3-v1"        # Pure SHA-3-256


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MigrationState:
    """
    Represents current PQ migration state.
    
    Attributes:
        phase: Current migration phase
        activation_block: Block number where current phase started
        hash_version_distribution: Count of blocks by hash_version
        dual_commitment_start: Block number where dual-commitment started (None if not started)
        sha3_cutover: Block number where SHA-3 cutover happened (None if not happened)
        metadata: Additional metadata
    """
    phase: MigrationPhase
    activation_block: int
    hash_version_distribution: Dict[str, int]
    dual_commitment_start: Optional[int]
    sha3_cutover: Optional[int]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase.value,
            "activation_block": self.activation_block,
            "hash_version_distribution": self.hash_version_distribution,
            "dual_commitment_start": self.dual_commitment_start,
            "sha3_cutover": self.sha3_cutover,
            "metadata": self.metadata,
        }


@dataclass
class ActivationBlock:
    """
    Represents a migration activation block.
    
    Attributes:
        block_number: Block number where activation occurs
        from_phase: Previous migration phase
        to_phase: New migration phase
        activation_root: Composite attestation root of activation block
        activation_metadata: Additional activation metadata
    """
    block_number: int
    from_phase: MigrationPhase
    to_phase: MigrationPhase
    activation_root: str
    activation_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "block_number": self.block_number,
            "from_phase": self.from_phase.value,
            "to_phase": self.to_phase.value,
            "activation_root": self.activation_root,
            "activation_metadata": self.activation_metadata,
        }


# ============================================================================
# MIGRATION STATE DETECTION
# ============================================================================

def detect_migration_state(blocks: List[Dict[str, Any]]) -> MigrationState:
    """
    Detect current PQ migration state from block list.
    
    Args:
        blocks: List of blocks (sorted by block_number)
    
    Returns:
        MigrationState with current phase and metadata
    
    Detection Algorithm:
        1. Count blocks by hash_version
        2. Detect phase transitions (sha256-v1 → dual-v1 → sha3-v1)
        3. Determine current phase based on latest blocks
    
    Deterministic Ordering:
        - Blocks must be pre-sorted by block_number
        - Phase detection in sequential order
    
    Input Schema:
        blocks: [
            {
                "id": int,
                "block_number": int,
                "attestation_metadata": {
                    "hash_version": str,
                    ...
                },
                ...
            },
            ...
        ]
    
    Output Schema:
        MigrationState(
            phase=MigrationPhase,
            activation_block=int,
            hash_version_distribution={"sha256-v1": int, "dual-v1": int, "sha3-v1": int},
            dual_commitment_start=int or None,
            sha3_cutover=int or None,
            metadata=dict,
        )
    """
    if not blocks:
        return MigrationState(
            phase=MigrationPhase.PRE_MIGRATION,
            activation_block=0,
            hash_version_distribution={},
            dual_commitment_start=None,
            sha3_cutover=None,
            metadata={},
        )
    
    # Count blocks by hash_version
    hash_version_distribution = {}
    dual_commitment_start = None
    sha3_cutover = None
    
    for block in blocks:
        hash_version = block.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
        hash_version_distribution[hash_version] = hash_version_distribution.get(hash_version, 0) + 1
        
        # Detect dual-commitment start
        if hash_version == "dual-v1" and dual_commitment_start is None:
            dual_commitment_start = block["block_number"]
        
        # Detect SHA-3 cutover
        if hash_version == "sha3-v1" and sha3_cutover is None:
            sha3_cutover = block["block_number"]
    
    # Determine current phase based on latest block
    latest_block = blocks[-1]
    latest_hash_version = latest_block.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    
    if latest_hash_version == "sha256-v1":
        phase = MigrationPhase.PRE_MIGRATION
        activation_block = 0
    elif latest_hash_version == "dual-v1":
        phase = MigrationPhase.DUAL_COMMITMENT
        activation_block = dual_commitment_start or 0
    elif latest_hash_version == "sha3-v1":
        phase = MigrationPhase.PURE_SHA3
        activation_block = sha3_cutover or 0
    else:
        phase = MigrationPhase.PRE_MIGRATION
        activation_block = 0
    
    return MigrationState(
        phase=phase,
        activation_block=activation_block,
        hash_version_distribution=hash_version_distribution,
        dual_commitment_start=dual_commitment_start,
        sha3_cutover=sha3_cutover,
        metadata={
            "total_blocks": len(blocks),
            "latest_block_number": latest_block["block_number"],
            "latest_hash_version": latest_hash_version,
        },
    )


# ============================================================================
# ACTIVATION BLOCK VALIDATION
# ============================================================================

def validate_activation_block(
    block: Dict[str, Any],
    expected_phase: MigrationPhase,
) -> Tuple[bool, Optional[str]]:
    """
    Validate activation block for phase transition.
    
    Args:
        block: Activation block
        expected_phase: Expected migration phase after activation
    
    Returns:
        (is_valid, error_message)
    
    Activation Block Invariants:
        1. Hash version matches expected phase
        2. Dual-commitment blocks have both SHA-256 and SHA-3 roots
        3. Activation metadata present and valid
    
    Input Schema:
        block: {
            "id": int,
            "block_number": int,
            "composite_attestation_root": str (hex),
            "composite_attestation_root_sha3": str (hex) or None,
            "attestation_metadata": {
                "hash_version": str,
                "activation_phase": str or None,
                ...
            },
            ...
        }
    
    Output Schema:
        (bool, str or None)
    """
    hash_version = block.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    
    # Validate hash version matches expected phase
    expected_hash_version = {
        MigrationPhase.PRE_MIGRATION: "sha256-v1",
        MigrationPhase.DUAL_COMMITMENT: "dual-v1",
        MigrationPhase.PURE_SHA3: "sha3-v1",
    }.get(expected_phase)
    
    if hash_version != expected_hash_version:
        return False, f"Hash version mismatch: expected {expected_hash_version}, got {hash_version}"
    
    # Validate dual-commitment blocks have both roots
    if expected_phase == MigrationPhase.DUAL_COMMITMENT:
        composite_root_sha256 = block.get("composite_attestation_root")
        composite_root_sha3 = block.get("composite_attestation_root_sha3")
        
        if not composite_root_sha256:
            return False, "Missing composite_attestation_root (SHA-256)"
        
        if not composite_root_sha3:
            return False, "Missing composite_attestation_root_sha3 (SHA-3)"
        
        # Validate both roots are valid hex strings
        if not (len(composite_root_sha256) == 64 and len(composite_root_sha3) == 64):
            return False, "Invalid composite root length"
    
    # Validate activation metadata
    activation_phase = block.get("attestation_metadata", {}).get("activation_phase")
    if activation_phase != expected_phase.value:
        return False, f"Activation phase mismatch: expected {expected_phase.value}, got {activation_phase}"
    
    return True, None


# ============================================================================
# DUAL-COMMITMENT VERIFICATION
# ============================================================================

def verify_dual_commitment(block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify dual-commitment block has matching SHA-256 and SHA-3 roots.
    
    Args:
        block: Dual-commitment block
    
    Returns:
        (is_valid, error_message)
    
    Dual-Commitment Invariant:
        composite_attestation_root = SHA256(R_t || U_t)
        composite_attestation_root_sha3 = SHA3_256(R_t || U_t)
        where R_t, U_t are the same for both
    
    Input Schema:
        block: {
            "reasoning_attestation_root": str (hex),
            "ui_attestation_root": str (hex),
            "composite_attestation_root": str (hex),
            "composite_attestation_root_sha3": str (hex),
            ...
        }
    
    Output Schema:
        (bool, str or None)
    """
    r_t = block.get("reasoning_attestation_root")
    u_t = block.get("ui_attestation_root")
    h_t_sha256 = block.get("composite_attestation_root")
    h_t_sha3 = block.get("composite_attestation_root_sha3")
    
    if not all([r_t, u_t, h_t_sha256, h_t_sha3]):
        return False, "Missing attestation roots"
    
    # Compute expected roots
    expected_h_t_sha256 = hashlib.sha256((r_t + u_t).encode()).hexdigest()
    expected_h_t_sha3 = hashlib.sha3_256((r_t + u_t).encode()).hexdigest()
    
    # Verify SHA-256 root
    if h_t_sha256 != expected_h_t_sha256:
        return False, f"SHA-256 composite root mismatch: expected {expected_h_t_sha256}, got {h_t_sha256}"
    
    # Verify SHA-3 root
    if h_t_sha3 != expected_h_t_sha3:
        return False, f"SHA-3 composite root mismatch: expected {expected_h_t_sha3}, got {h_t_sha3}"
    
    return True, None


# ============================================================================
# CROSS-ALGORITHM PREV-HASH VALIDATION
# ============================================================================

def validate_cross_algorithm_prev_hash(
    block: Dict[str, Any],
    predecessor: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Validate prev_hash across hash algorithm boundaries.
    
    Args:
        block: Current block
        predecessor: Predecessor block
    
    Returns:
        (is_valid, error_message)
    
    Cross-Algorithm Prev-Hash Rules:
        - Block N (SHA-256) → Block N+1 (DUAL): prev_hash uses SHA-256
        - Block N (DUAL) → Block N+1 (SHA-3): prev_hash uses SHA-256 (primary)
        - Block N (SHA-3) → Block N+1 (SHA-3): prev_hash uses SHA-3
    
    Input Schema:
        block: {
            "prev_hash": str (hex),
            "attestation_metadata": {"hash_version": str},
            ...
        }
        predecessor: {
            "id": int,
            "composite_attestation_root": str (hex),
            "attestation_metadata": {"hash_version": str},
            ...
        }
    
    Output Schema:
        (bool, str or None)
    """
    prev_hash = block.get("prev_hash")
    predecessor_hash_version = predecessor.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    
    # Compute expected prev_hash using predecessor's hash algorithm
    predecessor_identity = str(predecessor.get("id"))  # Simplified: use block ID
    
    if predecessor_hash_version in ["sha256-v1", "dual-v1"]:
        # Use SHA-256 for prev_hash
        expected_prev_hash = hashlib.sha256(predecessor_identity.encode()).hexdigest()
    elif predecessor_hash_version == "sha3-v1":
        # Use SHA-3 for prev_hash
        expected_prev_hash = hashlib.sha3_256(predecessor_identity.encode()).hexdigest()
    else:
        return False, f"Unsupported predecessor hash_version: {predecessor_hash_version}"
    
    # Verify prev_hash matches
    if prev_hash != expected_prev_hash:
        return False, f"prev_hash mismatch: expected {expected_prev_hash}, got {prev_hash}"
    
    return True, None


# ============================================================================
# MIGRATION ORCHESTRATOR
# ============================================================================

class PQMigrationOrchestrator:
    """
    Orchestrates PQ migration phase transitions.
    
    Usage:
        orchestrator = PQMigrationOrchestrator()
        state = orchestrator.detect_state(blocks)
        is_valid = orchestrator.validate_activation_block(block, MigrationPhase.DUAL_COMMITMENT)
        is_valid = orchestrator.verify_dual_commitment(block)
    """
    
    def __init__(self):
        """Initialize PQ migration orchestrator."""
        self.migration_history: List[MigrationState] = []
        self.activation_blocks: List[ActivationBlock] = []
    
    def detect_state(self, blocks: List[Dict[str, Any]]) -> MigrationState:
        """Detect current migration state."""
        state = detect_migration_state(blocks)
        self.migration_history.append(state)
        return state
    
    def validate_activation_block(
        self,
        block: Dict[str, Any],
        expected_phase: MigrationPhase,
    ) -> Tuple[bool, Optional[str]]:
        """Validate activation block."""
        is_valid, error = validate_activation_block(block, expected_phase)
        
        if is_valid:
            # Record activation block
            activation_block = ActivationBlock(
                block_number=block["block_number"],
                from_phase=MigrationPhase.PRE_MIGRATION,  # Simplified
                to_phase=expected_phase,
                activation_root=block.get("composite_attestation_root", ""),
                activation_metadata=block.get("attestation_metadata", {}),
            )
            self.activation_blocks.append(activation_block)
        
        return is_valid, error
    
    def verify_dual_commitment(self, block: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Verify dual-commitment block."""
        return verify_dual_commitment(block)
    
    def validate_cross_algorithm_prev_hash(
        self,
        block: Dict[str, Any],
        predecessor: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Validate cross-algorithm prev_hash."""
        return validate_cross_algorithm_prev_hash(block, predecessor)
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration state history."""
        return [s.to_dict() for s in self.migration_history]
    
    def get_activation_blocks(self) -> List[Dict[str, Any]]:
        """Get activation block history."""
        return [a.to_dict() for a in self.activation_blocks]


# ============================================================================
# MANUS-H INTEGRATION INTERFACE
# ============================================================================

class HashAlgorithmInterface:
    """
    Interface for Manus-H hash algorithm abstraction.
    
    This is a placeholder for the actual Manus-H implementation.
    Manus-H will provide concrete implementations of hash algorithms.
    """
    
    def hash(self, data: bytes) -> str:
        """
        Compute hash of data.
        
        Args:
            data: Data to hash
        
        Returns:
            Hash (hex string)
        """
        raise NotImplementedError("Manus-H must implement this method")
    
    def merkle_root(self, leaves: List[str]) -> str:
        """
        Compute Merkle root of leaves.
        
        Args:
            leaves: List of leaf hashes (hex strings)
        
        Returns:
            Merkle root (hex string)
        """
        raise NotImplementedError("Manus-H must implement this method")
    
    def version(self) -> str:
        """
        Get hash algorithm version.
        
        Returns:
            Version string ("sha256-v1" | "sha3-v1")
        """
        raise NotImplementedError("Manus-H must implement this method")


def get_hash_algorithm(version: str) -> HashAlgorithmInterface:
    """
    Get hash algorithm implementation for version.
    
    Args:
        version: Hash algorithm version
    
    Returns:
        HashAlgorithmInterface implementation
    
    Note:
        This is a placeholder. Manus-H will provide the actual implementation.
    """
    raise NotImplementedError("Manus-H must implement this function")
