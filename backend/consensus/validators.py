"""
Consensus Validators

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Consensus Runtime Activation
Date: 2025-12-06

Purpose:
    Validator orchestration for consensus rule enforcement.
    
    Validators:
    - BlockValidator: Validates individual blocks
    - ChainValidator: Validates entire chains
    - EpochValidator: Validates epochs
    - PQMigrationValidator: Validates PQ migration transitions

Design Principles:
    1. Composable: Validators can be combined
    2. Cacheable: Validation results can be cached
    3. Parallelizable: Validators can run in parallel
    4. Auditable: All validation decisions are logged
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from .rules import (
    RuleResult,
    RuleViolation,
    RuleViolationType,
    RuleSeverity,
    validate_block,
    validate_chain,
    validate_epoch_boundary,
    validate_hash_version_transition,
)


# ============================================================================
# VALIDATOR BASE CLASS
# ============================================================================

class Validator:
    """
    Base class for all validators.
    
    Attributes:
        name: Validator name
        cache: Validation result cache (block_id → RuleResult)
        stats: Validation statistics
    """
    
    def __init__(self, name: str, enable_cache: bool = True):
        """
        Initialize validator.
        
        Args:
            name: Validator name
            enable_cache: Whether to enable result caching
        """
        self.name = name
        self.enable_cache = enable_cache
        self.cache: Dict[int, RuleResult] = {}
        self.stats = {
            "total_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "warnings": 0,
        }
    
    def validate(self, *args, **kwargs) -> RuleResult:
        """
        Validate (abstract method).
        
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    def get_cache_key(self, *args, **kwargs) -> Optional[int]:
        """
        Get cache key for validation result.
        
        Returns None if caching not applicable.
        """
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.stats = {
            "total_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "warnings": 0,
        }
    
    def clear_cache(self):
        """Clear validation result cache."""
        self.cache.clear()


# ============================================================================
# BLOCK VALIDATOR
# ============================================================================

class BlockValidator(Validator):
    """
    Validates individual blocks.
    
    Validation Steps:
        1. Block structure validation
        2. Attestation consistency validation
        3. Prev-hash linkage validation
        4. Hash version transition validation (if predecessor provided)
    
    Input Schema:
        block: {
            "id": int,
            "block_number": int,
            "system_id": str (UUID),
            "prev_hash": str (hex) or None,
            "composite_attestation_root": str (hex),
            "reasoning_attestation_root": str (hex),
            "ui_attestation_root": str (hex),
            "canonical_proofs": list or dict,
            "attestation_metadata": {
                "hash_version": str,
                ...
            },
            "sealed_at": str (ISO timestamp),
        }
    
    Output Schema:
        RuleResult: {
            "is_valid": bool,
            "violations": [...]
        }
    
    Deterministic Ordering:
        - Validation rules applied in fixed order
        - All rules executed (no short-circuit)
        - Violations aggregated in order
    
    Error Taxonomy:
        See rules.py for detailed error taxonomy
    """
    
    def __init__(self, enable_cache: bool = True):
        """Initialize block validator."""
        super().__init__(name="BlockValidator", enable_cache=enable_cache)
    
    def validate(
        self,
        block: Dict[str, Any],
        predecessor: Optional[Dict[str, Any]] = None,
        validate_pq_transition: bool = True,
    ) -> RuleResult:
        """
        Validate block.
        
        Args:
            block: Block to validate
            predecessor: Predecessor block (None for genesis)
            validate_pq_transition: Whether to validate PQ hash version transition
        
        Returns:
            RuleResult with validation outcome
        """
        self.stats["total_validations"] += 1
        
        # Check cache
        cache_key = self.get_cache_key(block)
        if self.enable_cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Validate block
        result = validate_block(block, predecessor)
        
        # Validate PQ transition (if predecessor provided)
        if validate_pq_transition and predecessor is not None:
            pq_result = validate_hash_version_transition(block, predecessor)
            result.violations.extend(pq_result.violations)
            result.is_valid = result.is_valid and pq_result.is_valid
        
        # Update stats
        for violation in result.violations:
            if violation.severity == RuleSeverity.ERROR or violation.severity == RuleSeverity.CRITICAL:
                self.stats["errors"] += 1
            elif violation.severity == RuleSeverity.WARNING:
                self.stats["warnings"] += 1
        
        # Cache result
        if self.enable_cache and cache_key is not None:
            self.cache[cache_key] = result
        
        return result
    
    def get_cache_key(self, block: Dict[str, Any]) -> Optional[int]:
        """Get cache key (block ID)."""
        return block.get("id")


# ============================================================================
# CHAIN VALIDATOR
# ============================================================================

class ChainValidator(Validator):
    """
    Validates entire chains.
    
    Validation Steps:
        1. Chain monotonicity validation
        2. Chain lineage validation
        3. Individual block validation (for each block)
    
    Input Schema:
        blocks: [
            {
                "id": int,
                "block_number": int,
                ...
            },
            ...
        ]
    
    Output Schema:
        RuleResult: {
            "is_valid": bool,
            "violations": [...]
        }
    
    Deterministic Ordering:
        - Blocks must be pre-sorted by block_number
        - Validation rules applied in fixed order
        - All rules executed (no short-circuit)
    """
    
    def __init__(self, enable_cache: bool = False):
        """Initialize chain validator."""
        super().__init__(name="ChainValidator", enable_cache=enable_cache)
        self.block_validator = BlockValidator(enable_cache=True)
    
    def validate(self, blocks: List[Dict[str, Any]]) -> RuleResult:
        """
        Validate chain.
        
        Args:
            blocks: List of blocks (must be sorted by block_number)
        
        Returns:
            RuleResult with validation outcome
        """
        self.stats["total_validations"] += 1
        
        # Validate chain
        result = validate_chain(blocks)
        
        # Update stats
        for violation in result.violations:
            if violation.severity == RuleSeverity.ERROR or violation.severity == RuleSeverity.CRITICAL:
                self.stats["errors"] += 1
            elif violation.severity == RuleSeverity.WARNING:
                self.stats["warnings"] += 1
        
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics (including block validator stats)."""
        stats = super().get_stats()
        block_stats = self.block_validator.get_stats()
        stats["block_validator"] = block_stats
        return stats


# ============================================================================
# EPOCH VALIDATOR
# ============================================================================

class EpochValidator(Validator):
    """
    Validates epochs.
    
    Validation Steps:
        1. Epoch boundary validation
        2. Epoch root computation validation
        3. Block membership validation
    
    Input Schema:
        epoch: {
            "id": int,
            "epoch_number": int,
            "start_block_number": int,
            "end_block_number": int,
            "block_count": int,
            "epoch_root": str (hex),
            "epoch_metadata": {
                "epoch_size": int,
                "composite_roots": [str, ...],
                "hash_version": str,
                ...
            },
            ...
        }
        blocks: [
            {
                "id": int,
                "block_number": int,
                "composite_attestation_root": str (hex),
                ...
            },
            ...
        ]
    
    Output Schema:
        RuleResult: {
            "is_valid": bool,
            "violations": [...]
        }
    """
    
    def __init__(self, enable_cache: bool = True):
        """Initialize epoch validator."""
        super().__init__(name="EpochValidator", enable_cache=enable_cache)
    
    def validate(
        self,
        epoch: Dict[str, Any],
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        """
        Validate epoch.
        
        Args:
            epoch: Epoch to validate
            blocks: Blocks in epoch (optional, for root verification)
        
        Returns:
            RuleResult with validation outcome
        """
        self.stats["total_validations"] += 1
        
        # Check cache
        cache_key = self.get_cache_key(epoch)
        if self.enable_cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        all_violations = []
        
        # Validate epoch boundary
        result = validate_epoch_boundary(epoch)
        all_violations.extend(result.violations)
        
        # Validate epoch root (if blocks provided)
        if blocks is not None:
            root_result = self.validate_epoch_root(epoch, blocks)
            all_violations.extend(root_result.violations)
        
        # Create result
        result = RuleResult(is_valid=len(all_violations) == 0, violations=all_violations)
        
        # Update stats
        for violation in result.violations:
            if violation.severity == RuleSeverity.ERROR or violation.severity == RuleSeverity.CRITICAL:
                self.stats["errors"] += 1
            elif violation.severity == RuleSeverity.WARNING:
                self.stats["warnings"] += 1
        
        # Cache result
        if self.enable_cache and cache_key is not None:
            self.cache[cache_key] = result
        
        return result
    
    def validate_epoch_root(
        self,
        epoch: Dict[str, Any],
        blocks: List[Dict[str, Any]],
    ) -> RuleResult:
        """
        Validate epoch root computation.
        
        Args:
            epoch: Epoch dictionary
            blocks: Blocks in epoch
        
        Returns:
            RuleResult with validation outcome
        
        Invariant:
            epoch_root = MerkleRoot([H_0, H_1, ..., H_99])
            where H_i = block_i.composite_attestation_root
        """
        violations = []
        
        # Extract composite roots from blocks
        composite_roots = [b["composite_attestation_root"] for b in blocks]
        
        # Detect hash algorithm
        hash_version = epoch.get("epoch_metadata", {}).get("hash_version", "sha256-v1")
        
        # Compute expected epoch root
        expected_root = self.compute_merkle_root(composite_roots, hash_version)
        
        # Verify epoch root matches
        actual_root = epoch.get("epoch_root")
        if actual_root != expected_root:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.INVALID_EPOCH_ROOT,
                severity=RuleSeverity.ERROR,
                block_number=None,
                block_id=None,
                message=f"Epoch root mismatch: expected {expected_root}, got {actual_root}",
                context={
                    "epoch_number": epoch.get("epoch_number"),
                    "expected": expected_root,
                    "actual": actual_root,
                    "hash_version": hash_version,
                    "block_count": len(blocks),
                    "error": "INVALID_EPOCH_ROOT",
                },
            ))
        
        return RuleResult(is_valid=len(violations) == 0, violations=violations)
    
    def compute_merkle_root(self, leaves: List[str], hash_version: str) -> str:
        """
        Compute Merkle root of leaves.
        
        Args:
            leaves: List of leaf hashes (hex strings)
            hash_version: Hash algorithm version
        
        Returns:
            Merkle root (hex string)
        
        Algorithm:
            1. Hash each leaf: H(leaf)
            2. Pair adjacent hashes and hash: H(H1 || H2)
            3. Repeat until single root remains
            4. If odd number of nodes, duplicate last node
        """
        if not leaves:
            return ""
        
        # Select hash function
        if hash_version in ["sha256-v1", "dual-v1"]:
            hash_func = hashlib.sha256
        elif hash_version == "sha3-v1":
            hash_func = hashlib.sha3_256
        else:
            raise ValueError(f"Unsupported hash_version: {hash_version}")
        
        # Build Merkle tree
        current_level = [hash_func(leaf.encode()).digest() for leaf in leaves]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = hash_func(left + right).digest()
                next_level.append(parent)
            current_level = next_level
        
        return current_level[0].hex()
    
    def get_cache_key(self, epoch: Dict[str, Any]) -> Optional[int]:
        """Get cache key (epoch ID)."""
        return epoch.get("id")


# ============================================================================
# PQ MIGRATION VALIDATOR
# ============================================================================

class PQMigrationValidator(Validator):
    """
    Validates PQ migration transitions.
    
    Validation Steps:
        1. Hash version transition validation
        2. Dual-commitment validation
        3. Migration boundary validation
    
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
        RuleResult: {
            "is_valid": bool,
            "violations": [...]
        }
    """
    
    def __init__(self, enable_cache: bool = False):
        """Initialize PQ migration validator."""
        super().__init__(name="PQMigrationValidator", enable_cache=enable_cache)
    
    def validate(self, blocks: List[Dict[str, Any]]) -> RuleResult:
        """
        Validate PQ migration across blocks.
        
        Args:
            blocks: List of blocks (must be sorted by block_number)
        
        Returns:
            RuleResult with validation outcome
        """
        self.stats["total_validations"] += 1
        
        all_violations = []
        
        # Validate hash version transitions
        for i in range(1, len(blocks)):
            predecessor = blocks[i - 1]
            current = blocks[i]
            
            result = validate_hash_version_transition(current, predecessor)
            all_violations.extend(result.violations)
        
        # Update stats
        for violation in all_violations:
            if violation.severity == RuleSeverity.ERROR or violation.severity == RuleSeverity.CRITICAL:
                self.stats["errors"] += 1
            elif violation.severity == RuleSeverity.WARNING:
                self.stats["warnings"] += 1
        
        return RuleResult(is_valid=len(all_violations) == 0, violations=all_violations)


# ============================================================================
# VALIDATOR ORCHESTRATOR
# ============================================================================

class ValidatorOrchestrator:
    """
    Orchestrates multiple validators.
    
    Validators:
        - BlockValidator
        - ChainValidator
        - EpochValidator
        - PQMigrationValidator
    
    Usage:
        orchestrator = ValidatorOrchestrator()
        result = orchestrator.validate_block(block, predecessor)
        result = orchestrator.validate_chain(blocks)
        result = orchestrator.validate_epoch(epoch, blocks)
    """
    
    def __init__(self):
        """Initialize validator orchestrator."""
        self.block_validator = BlockValidator()
        self.chain_validator = ChainValidator()
        self.epoch_validator = EpochValidator()
        self.pq_migration_validator = PQMigrationValidator()
    
    def validate_block(
        self,
        block: Dict[str, Any],
        predecessor: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        """Validate block."""
        return self.block_validator.validate(block, predecessor)
    
    def validate_chain(self, blocks: List[Dict[str, Any]]) -> RuleResult:
        """Validate chain."""
        return self.chain_validator.validate(blocks)
    
    def validate_epoch(
        self,
        epoch: Dict[str, Any],
        blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> RuleResult:
        """Validate epoch."""
        return self.epoch_validator.validate(epoch, blocks)
    
    def validate_pq_migration(self, blocks: List[Dict[str, Any]]) -> RuleResult:
        """Validate PQ migration."""
        return self.pq_migration_validator.validate(blocks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all validator statistics."""
        return {
            "block_validator": self.block_validator.get_stats(),
            "chain_validator": self.chain_validator.get_stats(),
            "epoch_validator": self.epoch_validator.get_stats(),
            "pq_migration_validator": self.pq_migration_validator.get_stats(),
        }
    
    def reset_stats(self):
        """Reset all validator statistics."""
        self.block_validator.reset_stats()
        self.chain_validator.reset_stats()
        self.epoch_validator.reset_stats()
        self.pq_migration_validator.reset_stats()
    
    def clear_caches(self):
        """Clear all validator caches."""
        self.block_validator.clear_cache()
        self.chain_validator.clear_cache()
        self.epoch_validator.clear_cache()
        self.pq_migration_validator.clear_cache()
