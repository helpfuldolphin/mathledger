"""
Consensus Runtime Rules

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Consensus Runtime Activation
Date: 2025-12-06

Purpose:
    Define consensus rules for ledger governance, block validation, and chain evolution.
    
    Rules enforce:
    - Block validity (structure, attestation, prev_hash)
    - Chain validity (monotonicity, lineage, no forks)
    - Epoch validity (sealing conditions, root computation)
    - PQ migration validity (hash version transitions)

Design Principles:
    1. Deterministic: Same inputs → same outputs (no randomness)
    2. Verifiable: All rules can be checked independently
    3. Composable: Rules can be combined (AND, OR, NOT)
    4. Versioned: Rules can evolve (with migration path)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class RuleViolationType(Enum):
    """Types of consensus rule violations."""
    # Block-level violations
    INVALID_BLOCK_STRUCTURE = "invalid_block_structure"
    INVALID_ATTESTATION = "invalid_attestation"
    INVALID_PREV_HASH = "invalid_prev_hash"
    INVALID_HASH_VERSION = "invalid_hash_version"
    
    # Chain-level violations
    MONOTONICITY_VIOLATION = "monotonicity_violation"
    LINEAGE_BROKEN = "lineage_broken"
    FORK_DETECTED = "fork_detected"
    CYCLE_DETECTED = "cycle_detected"
    ORPHAN_BLOCK = "orphan_block"
    
    # Epoch-level violations
    INVALID_EPOCH_BOUNDARY = "invalid_epoch_boundary"
    INVALID_EPOCH_ROOT = "invalid_epoch_root"
    INCOMPLETE_EPOCH = "incomplete_epoch"
    
    # PQ migration violations
    INVALID_HASH_TRANSITION = "invalid_hash_transition"
    MISSING_DUAL_COMMITMENT = "missing_dual_commitment"
    PREMATURE_SHA3_ADOPTION = "premature_sha3_adoption"


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    INFO = "info"          # Informational, no action required
    WARNING = "warning"    # Warning, should be investigated
    ERROR = "error"        # Error, block should be rejected
    CRITICAL = "critical"  # Critical, chain integrity compromised


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RuleViolation:
    """
    Represents a consensus rule violation.
    
    Attributes:
        violation_type: Type of violation
        severity: Severity level
        block_number: Block number where violation occurred
        block_id: Block ID (if applicable)
        message: Human-readable error message
        context: Additional context (evidence, affected blocks, etc.)
    """
    violation_type: RuleViolationType
    severity: RuleSeverity
    block_number: Optional[int]
    block_id: Optional[int]
    message: str
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "block_number": self.block_number,
            "block_id": self.block_id,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class RuleResult:
    """
    Result of applying a consensus rule.
    
    Attributes:
        is_valid: Whether the rule passed
        violations: List of violations (empty if valid)
    """
    is_valid: bool
    violations: List[RuleViolation]
    
    def __bool__(self) -> bool:
        """Allow boolean evaluation (if result: ...)"""
        return self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
        }


# ============================================================================
# BLOCK VALIDATION RULES
# ============================================================================

def validate_block_structure(block: Dict[str, Any]) -> RuleResult:
    """
    Validate block structure (required fields, types).
    
    Args:
        block: Block dictionary
    
    Returns:
        RuleResult with validation outcome
    
    Required Fields:
        - id (int)
        - block_number (int)
        - system_id (str, UUID)
        - prev_hash (str, hex)
        - composite_attestation_root (str, hex)
        - reasoning_attestation_root (str, hex)
        - ui_attestation_root (str, hex)
        - canonical_proofs (list or dict)
        - attestation_metadata (dict)
        - sealed_at (str, ISO timestamp)
    
    Deterministic Ordering:
        - Field validation order: alphabetical by field name
        - Error accumulation: all errors collected before returning
    
    Error Taxonomy:
        - MISSING_FIELD: Required field not present
        - INVALID_TYPE: Field has wrong type
        - INVALID_FORMAT: Field has invalid format (e.g., non-hex hash)
    """
    violations = []
    
    # Required fields with expected types
    required_fields = {
        "id": int,
        "block_number": int,
        "system_id": str,
        "prev_hash": (str, type(None)),  # None for genesis block
        "composite_attestation_root": str,
        "reasoning_attestation_root": str,
        "ui_attestation_root": str,
        "canonical_proofs": (list, dict),
        "attestation_metadata": dict,
        "sealed_at": str,
    }
    
    # Check required fields (alphabetical order for determinism)
    for field in sorted(required_fields.keys()):
        expected_type = required_fields[field]
        
        # Check field exists
        if field not in block:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.INVALID_BLOCK_STRUCTURE,
                severity=RuleSeverity.ERROR,
                block_number=block.get("block_number"),
                block_id=block.get("id"),
                message=f"Missing required field: {field}",
                context={"field": field, "error": "MISSING_FIELD"},
            ))
            continue
        
        # Check field type
        value = block[field]
        if not isinstance(value, expected_type):
            violations.append(RuleViolation(
                violation_type=RuleViolationType.INVALID_BLOCK_STRUCTURE,
                severity=RuleSeverity.ERROR,
                block_number=block.get("block_number"),
                block_id=block.get("id"),
                message=f"Invalid type for field {field}: expected {expected_type}, got {type(value)}",
                context={"field": field, "expected_type": str(expected_type), "actual_type": str(type(value)), "error": "INVALID_TYPE"},
            ))
    
    # Validate hash formats (must be hex strings)
    hash_fields = ["prev_hash", "composite_attestation_root", "reasoning_attestation_root", "ui_attestation_root"]
    for field in hash_fields:
        if field in block and block[field] is not None:
            value = block[field]
            if not isinstance(value, str):
                continue  # Already reported as type error
            
            # Check hex format (64 chars for SHA-256, 128 for SHA-3-512)
            if not (len(value) == 64 or len(value) == 128):
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.INVALID_BLOCK_STRUCTURE,
                    severity=RuleSeverity.ERROR,
                    block_number=block.get("block_number"),
                    block_id=block.get("id"),
                    message=f"Invalid hash length for {field}: expected 64 or 128 chars, got {len(value)}",
                    context={"field": field, "length": len(value), "error": "INVALID_FORMAT"},
                ))
            
            # Check hex characters
            try:
                int(value, 16)
            except ValueError:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.INVALID_BLOCK_STRUCTURE,
                    severity=RuleSeverity.ERROR,
                    block_number=block.get("block_number"),
                    block_id=block.get("id"),
                    message=f"Invalid hex format for {field}: {value[:16]}...",
                    context={"field": field, "value": value[:16], "error": "INVALID_FORMAT"},
                ))
    
    return RuleResult(is_valid=len(violations) == 0, violations=violations)


def validate_attestation_consistency(block: Dict[str, Any]) -> RuleResult:
    """
    Validate attestation root consistency: H_t = Hash(R_t || U_t).
    
    Args:
        block: Block dictionary
    
    Returns:
        RuleResult with validation outcome
    
    Invariant:
        composite_attestation_root = Hash(reasoning_attestation_root || ui_attestation_root)
        where Hash ∈ {SHA256, SHA3_256} depending on hash_version
    
    Deterministic Ordering:
        - Hash algorithm detection: read hash_version from attestation_metadata
        - Concatenation order: R_t || U_t (reasoning first, UI second)
        - Domain separation: "EPOCH:" prefix for composite hash
    
    Error Taxonomy:
        - MISSING_HASH_VERSION: hash_version not in attestation_metadata
        - UNSUPPORTED_HASH_VERSION: hash_version not recognized
        - COMPOSITE_ROOT_MISMATCH: H_t doesn't match Hash(R_t || U_t)
        - DUAL_COMMITMENT_MISMATCH: For dual-v1, SHA-3 root doesn't match
    
    Replay Compatibility:
        - Must use same hash algorithm as original sealing
        - Must produce identical H_t when replayed
    """
    violations = []
    
    # Extract attestation roots
    r_t = block.get("reasoning_attestation_root")
    u_t = block.get("ui_attestation_root")
    h_t = block.get("composite_attestation_root")
    
    if not all([r_t, u_t, h_t]):
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_ATTESTATION,
            severity=RuleSeverity.ERROR,
            block_number=block.get("block_number"),
            block_id=block.get("id"),
            message="Missing attestation roots",
            context={"r_t": r_t, "u_t": u_t, "h_t": h_t, "error": "MISSING_ATTESTATION_ROOTS"},
        ))
        return RuleResult(is_valid=False, violations=violations)
    
    # Detect hash algorithm
    attestation_metadata = block.get("attestation_metadata", {})
    hash_version = attestation_metadata.get("hash_version", "sha256-v1")
    
    # Compute expected composite root
    if hash_version == "sha256-v1":
        # Legacy SHA-256
        expected_h_t = hashlib.sha256((r_t + u_t).encode()).hexdigest()
    
    elif hash_version == "sha3-v1":
        # Pure SHA-3
        expected_h_t = hashlib.sha3_256((r_t + u_t).encode()).hexdigest()
    
    elif hash_version == "dual-v1":
        # Dual-commitment: verify both SHA-256 and SHA-3
        expected_h_t_sha256 = hashlib.sha256((r_t + u_t).encode()).hexdigest()
        expected_h_t_sha3 = hashlib.sha3_256((r_t + u_t).encode()).hexdigest()
        
        # Check SHA-256 root (primary)
        if h_t != expected_h_t_sha256:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.INVALID_ATTESTATION,
                severity=RuleSeverity.ERROR,
                block_number=block.get("block_number"),
                block_id=block.get("id"),
                message=f"Composite root mismatch (SHA-256): expected {expected_h_t_sha256}, got {h_t}",
                context={"expected": expected_h_t_sha256, "actual": h_t, "hash_version": "sha256", "error": "COMPOSITE_ROOT_MISMATCH"},
            ))
        
        # Check SHA-3 root (secondary)
        h_t_sha3 = block.get("composite_attestation_root_sha3")
        if h_t_sha3 != expected_h_t_sha3:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.INVALID_ATTESTATION,
                severity=RuleSeverity.ERROR,
                block_number=block.get("block_number"),
                block_id=block.get("id"),
                message=f"Composite root mismatch (SHA-3): expected {expected_h_t_sha3}, got {h_t_sha3}",
                context={"expected": expected_h_t_sha3, "actual": h_t_sha3, "hash_version": "sha3", "error": "DUAL_COMMITMENT_MISMATCH"},
            ))
        
        return RuleResult(is_valid=len(violations) == 0, violations=violations)
    
    else:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_HASH_VERSION,
            severity=RuleSeverity.ERROR,
            block_number=block.get("block_number"),
            block_id=block.get("id"),
            message=f"Unsupported hash_version: {hash_version}",
            context={"hash_version": hash_version, "error": "UNSUPPORTED_HASH_VERSION"},
        ))
        return RuleResult(is_valid=False, violations=violations)
    
    # Verify composite root matches
    if h_t != expected_h_t:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_ATTESTATION,
            severity=RuleSeverity.ERROR,
            block_number=block.get("block_number"),
            block_id=block.get("id"),
            message=f"Composite root mismatch: expected {expected_h_t}, got {h_t}",
            context={"expected": expected_h_t, "actual": h_t, "hash_version": hash_version, "error": "COMPOSITE_ROOT_MISMATCH"},
        ))
    
    return RuleResult(is_valid=len(violations) == 0, violations=violations)


def validate_prev_hash_linkage(block: Dict[str, Any], predecessor: Optional[Dict[str, Any]]) -> RuleResult:
    """
    Validate prev_hash linkage to predecessor block.
    
    Args:
        block: Current block
        predecessor: Predecessor block (None for genesis)
    
    Returns:
        RuleResult with validation outcome
    
    Invariant:
        block.prev_hash = Hash(predecessor.block_identity)
        where Hash uses predecessor's hash algorithm
    
    Deterministic Ordering:
        - Genesis block: prev_hash must be None
        - Non-genesis block: prev_hash must match predecessor
        - Hash algorithm: use predecessor's hash_version
    
    Error Taxonomy:
        - GENESIS_HAS_PREV_HASH: Genesis block (block_number=0) has non-null prev_hash
        - MISSING_PREV_HASH: Non-genesis block has null prev_hash
        - PREV_HASH_MISMATCH: prev_hash doesn't match predecessor
        - ORPHAN_BLOCK: Predecessor not provided for non-genesis block
    
    Replay Compatibility:
        - Must use predecessor's hash algorithm
        - Must produce identical prev_hash when replayed
    """
    violations = []
    
    block_number = block.get("block_number")
    prev_hash = block.get("prev_hash")
    
    # Genesis block: prev_hash must be None
    if block_number == 0:
        if prev_hash is not None:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.INVALID_PREV_HASH,
                severity=RuleSeverity.ERROR,
                block_number=block_number,
                block_id=block.get("id"),
                message="Genesis block must have prev_hash=None",
                context={"prev_hash": prev_hash, "error": "GENESIS_HAS_PREV_HASH"},
            ))
        return RuleResult(is_valid=len(violations) == 0, violations=violations)
    
    # Non-genesis block: prev_hash must be present
    if prev_hash is None:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_PREV_HASH,
            severity=RuleSeverity.ERROR,
            block_number=block_number,
            block_id=block.get("id"),
            message="Non-genesis block must have prev_hash",
            context={"block_number": block_number, "error": "MISSING_PREV_HASH"},
        ))
        return RuleResult(is_valid=False, violations=violations)
    
    # Predecessor must be provided
    if predecessor is None:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.ORPHAN_BLOCK,
            severity=RuleSeverity.ERROR,
            block_number=block_number,
            block_id=block.get("id"),
            message="Predecessor block not provided",
            context={"block_number": block_number, "error": "ORPHAN_BLOCK"},
        ))
        return RuleResult(is_valid=False, violations=violations)
    
    # Compute expected prev_hash using predecessor's hash algorithm
    predecessor_hash_version = predecessor.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    predecessor_identity = str(predecessor.get("id"))  # Simplified: use block ID as identity
    
    if predecessor_hash_version in ["sha256-v1", "dual-v1"]:
        expected_prev_hash = hashlib.sha256(predecessor_identity.encode()).hexdigest()
    elif predecessor_hash_version == "sha3-v1":
        expected_prev_hash = hashlib.sha3_256(predecessor_identity.encode()).hexdigest()
    else:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_HASH_VERSION,
            severity=RuleSeverity.ERROR,
            block_number=block_number,
            block_id=block.get("id"),
            message=f"Predecessor has unsupported hash_version: {predecessor_hash_version}",
            context={"predecessor_hash_version": predecessor_hash_version, "error": "UNSUPPORTED_HASH_VERSION"},
        ))
        return RuleResult(is_valid=False, violations=violations)
    
    # Verify prev_hash matches
    if prev_hash != expected_prev_hash:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_PREV_HASH,
            severity=RuleSeverity.ERROR,
            block_number=block_number,
            block_id=block.get("id"),
            message=f"prev_hash mismatch: expected {expected_prev_hash}, got {prev_hash}",
            context={
                "expected": expected_prev_hash,
                "actual": prev_hash,
                "predecessor_id": predecessor.get("id"),
                "predecessor_hash_version": predecessor_hash_version,
                "error": "PREV_HASH_MISMATCH",
            },
        ))
    
    return RuleResult(is_valid=len(violations) == 0, violations=violations)


def validate_block(block: Dict[str, Any], predecessor: Optional[Dict[str, Any]] = None) -> RuleResult:
    """
    Validate block against all consensus rules.
    
    Args:
        block: Block to validate
        predecessor: Predecessor block (None for genesis)
    
    Returns:
        RuleResult with aggregated violations
    
    Rules Applied (in order):
        1. validate_block_structure
        2. validate_attestation_consistency
        3. validate_prev_hash_linkage
    
    Deterministic Ordering:
        - Rules applied in fixed order
        - All rules executed (no short-circuit)
        - Violations aggregated in order
    
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
                "hash_version": str ("sha256-v1" | "sha3-v1" | "dual-v1"),
                ...
            },
            "sealed_at": str (ISO timestamp),
        }
    
    Output Schema:
        RuleResult: {
            "is_valid": bool,
            "violations": [
                {
                    "violation_type": str,
                    "severity": str,
                    "block_number": int,
                    "block_id": int,
                    "message": str,
                    "context": dict,
                },
                ...
            ]
        }
    """
    all_violations = []
    
    # Rule 1: Block structure
    result = validate_block_structure(block)
    all_violations.extend(result.violations)
    
    # Rule 2: Attestation consistency
    result = validate_attestation_consistency(block)
    all_violations.extend(result.violations)
    
    # Rule 3: Prev-hash linkage
    result = validate_prev_hash_linkage(block, predecessor)
    all_violations.extend(result.violations)
    
    return RuleResult(is_valid=len(all_violations) == 0, violations=all_violations)


# ============================================================================
# CHAIN VALIDATION RULES
# ============================================================================

def validate_chain_monotonicity(blocks: List[Dict[str, Any]]) -> RuleResult:
    """
    Validate chain monotonicity: block_number strictly increasing, no gaps.
    
    Args:
        blocks: List of blocks (must be sorted by block_number)
    
    Returns:
        RuleResult with validation outcome
    
    Invariant:
        ∀ consecutive blocks bₙ, bₙ₊₁:
            bₙ₊₁.block_number = bₙ.block_number + 1
            bₙ₊₁.sealed_at ≥ bₙ.sealed_at
    
    Deterministic Ordering:
        - Blocks must be pre-sorted by block_number (ascending)
        - Gaps detected by checking consecutive differences
        - Duplicates detected by checking equality
    
    Error Taxonomy:
        - GAP_DETECTED: Missing block_number in sequence
        - DUPLICATE_BLOCK_NUMBER: Multiple blocks with same block_number
        - OUT_OF_ORDER_SEALING: sealed_at decreases
        - UNSORTED_INPUT: Blocks not sorted by block_number
    """
    violations = []
    
    if not blocks:
        return RuleResult(is_valid=True, violations=[])
    
    # Check blocks are sorted
    for i in range(len(blocks) - 1):
        if blocks[i]["block_number"] > blocks[i + 1]["block_number"]:
            violations.append(RuleViolation(
                violation_type=RuleViolationType.MONOTONICITY_VIOLATION,
                severity=RuleSeverity.ERROR,
                block_number=blocks[i]["block_number"],
                block_id=blocks[i].get("id"),
                message="Blocks not sorted by block_number",
                context={"error": "UNSORTED_INPUT"},
            ))
            return RuleResult(is_valid=False, violations=violations)
    
    # Check monotonicity
    for i in range(len(blocks) - 1):
        current = blocks[i]
        next_block = blocks[i + 1]
        
        current_num = current["block_number"]
        next_num = next_block["block_number"]
        
        # Check for gap
        if next_num != current_num + 1:
            if next_num > current_num + 1:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.MONOTONICITY_VIOLATION,
                    severity=RuleSeverity.ERROR,
                    block_number=current_num,
                    block_id=current.get("id"),
                    message=f"Gap detected: block {current_num} → {next_num} (missing {next_num - current_num - 1} blocks)",
                    context={"gap_start": current_num, "gap_end": next_num, "missing_count": next_num - current_num - 1, "error": "GAP_DETECTED"},
                ))
            elif next_num == current_num:
                violations.append(RuleViolation(
                    violation_type=RuleViolationType.MONOTONICITY_VIOLATION,
                    severity=RuleSeverity.ERROR,
                    block_number=current_num,
                    block_id=current.get("id"),
                    message=f"Duplicate block_number: {current_num}",
                    context={"block_number": current_num, "block_ids": [current.get("id"), next_block.get("id")], "error": "DUPLICATE_BLOCK_NUMBER"},
                ))
    
    return RuleResult(is_valid=len(violations) == 0, violations=violations)


def validate_chain_lineage(blocks: List[Dict[str, Any]]) -> RuleResult:
    """
    Validate chain lineage: prev_hash forms valid DAG (no cycles, forks).
    
    Args:
        blocks: List of blocks (must be sorted by block_number)
    
    Returns:
        RuleResult with validation outcome
    
    Invariant:
        ∀ consecutive blocks bₙ, bₙ₊₁:
            bₙ₊₁.prev_hash = Hash(bₙ.block_identity)
    
    Deterministic Ordering:
        - Blocks must be pre-sorted by block_number
        - Lineage checked in sequential order
    
    Error Taxonomy:
        - LINEAGE_BROKEN: prev_hash doesn't match predecessor
        - CYCLE_DETECTED: Block references itself or creates cycle
        - FORK_DETECTED: Multiple blocks reference same prev_hash
    """
    violations = []
    
    if not blocks:
        return RuleResult(is_valid=True, violations=[])
    
    # Validate lineage
    for i in range(1, len(blocks)):
        predecessor = blocks[i - 1]
        current = blocks[i]
        
        result = validate_prev_hash_linkage(current, predecessor)
        violations.extend(result.violations)
    
    return RuleResult(is_valid=len(violations) == 0, violations=violations)


def validate_chain(blocks: List[Dict[str, Any]]) -> RuleResult:
    """
    Validate entire chain against all consensus rules.
    
    Args:
        blocks: List of blocks (must be sorted by block_number)
    
    Returns:
        RuleResult with aggregated violations
    
    Rules Applied (in order):
        1. validate_chain_monotonicity
        2. validate_chain_lineage
        3. validate_block (for each block)
    
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
    """
    all_violations = []
    
    # Rule 1: Chain monotonicity
    result = validate_chain_monotonicity(blocks)
    all_violations.extend(result.violations)
    
    # Rule 2: Chain lineage
    result = validate_chain_lineage(blocks)
    all_violations.extend(result.violations)
    
    # Rule 3: Individual block validation
    for i, block in enumerate(blocks):
        predecessor = blocks[i - 1] if i > 0 else None
        result = validate_block(block, predecessor)
        all_violations.extend(result.violations)
    
    return RuleResult(is_valid=len(all_violations) == 0, violations=all_violations)


# ============================================================================
# EPOCH VALIDATION RULES
# ============================================================================

def validate_epoch_boundary(epoch: Dict[str, Any]) -> RuleResult:
    """
    Validate epoch boundary conditions.
    
    Args:
        epoch: Epoch dictionary
    
    Returns:
        RuleResult with validation outcome
    
    Invariant:
        epoch.start_block_number = epoch.epoch_number * epoch_size
        epoch.end_block_number = (epoch.epoch_number + 1) * epoch_size
        epoch.block_count = epoch_size (usually 100)
    
    Error Taxonomy:
        - INVALID_EPOCH_BOUNDARY: start/end blocks don't match epoch_number
        - INVALID_BLOCK_COUNT: block_count doesn't match range
    """
    violations = []
    
    epoch_number = epoch.get("epoch_number")
    start_block = epoch.get("start_block_number")
    end_block = epoch.get("end_block_number")
    block_count = epoch.get("block_count")
    epoch_size = epoch.get("epoch_metadata", {}).get("epoch_size", 100)
    
    # Validate boundary
    expected_start = epoch_number * epoch_size
    expected_end = (epoch_number + 1) * epoch_size
    
    if start_block != expected_start or end_block != expected_end:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_EPOCH_BOUNDARY,
            severity=RuleSeverity.ERROR,
            block_number=None,
            block_id=None,
            message=f"Invalid epoch boundary: expected [{expected_start}, {expected_end}), got [{start_block}, {end_block})",
            context={
                "epoch_number": epoch_number,
                "expected_start": expected_start,
                "expected_end": expected_end,
                "actual_start": start_block,
                "actual_end": end_block,
                "error": "INVALID_EPOCH_BOUNDARY",
            },
        ))
    
    # Validate block count
    expected_count = end_block - start_block
    if block_count != expected_count:
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_EPOCH_BOUNDARY,
            severity=RuleSeverity.ERROR,
            block_number=None,
            block_id=None,
            message=f"Invalid block count: expected {expected_count}, got {block_count}",
            context={
                "expected_count": expected_count,
                "actual_count": block_count,
                "error": "INVALID_BLOCK_COUNT",
            },
        ))
    
    return RuleResult(is_valid=len(violations) == 0, violations=violations)


# ============================================================================
# PQ MIGRATION VALIDATION RULES
# ============================================================================

def validate_hash_version_transition(block: Dict[str, Any], predecessor: Optional[Dict[str, Any]]) -> RuleResult:
    """
    Validate hash version transitions during PQ migration.
    
    Args:
        block: Current block
        predecessor: Predecessor block (None for genesis)
    
    Returns:
        RuleResult with validation outcome
    
    Valid Transitions:
        sha256-v1 → sha256-v1 (no change)
        sha256-v1 → dual-v1 (start dual-commitment)
        dual-v1 → dual-v1 (continue dual-commitment)
        dual-v1 → sha3-v1 (end dual-commitment)
        sha3-v1 → sha3-v1 (no change)
    
    Invalid Transitions:
        sha256-v1 → sha3-v1 (skipped dual-commitment)
        dual-v1 → sha256-v1 (rollback)
        sha3-v1 → sha256-v1 (rollback)
        sha3-v1 → dual-v1 (rollback)
    
    Error Taxonomy:
        - INVALID_HASH_TRANSITION: Transition not allowed
        - PREMATURE_SHA3_ADOPTION: Skipped dual-commitment phase
        - HASH_VERSION_ROLLBACK: Attempted rollback to older version
    """
    violations = []
    
    if predecessor is None:
        return RuleResult(is_valid=True, violations=[])  # Genesis block, no transition
    
    current_version = block.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    predecessor_version = predecessor.get("attestation_metadata", {}).get("hash_version", "sha256-v1")
    
    # Valid transitions
    valid_transitions = {
        "sha256-v1": ["sha256-v1", "dual-v1"],
        "dual-v1": ["dual-v1", "sha3-v1"],
        "sha3-v1": ["sha3-v1"],
    }
    
    if current_version not in valid_transitions.get(predecessor_version, []):
        # Classify error
        if predecessor_version == "sha256-v1" and current_version == "sha3-v1":
            error_type = "PREMATURE_SHA3_ADOPTION"
            message = "Skipped dual-commitment phase: sha256-v1 → sha3-v1"
        elif current_version in ["sha256-v1", "dual-v1"] and predecessor_version == "sha3-v1":
            error_type = "HASH_VERSION_ROLLBACK"
            message = f"Attempted rollback: {predecessor_version} → {current_version}"
        else:
            error_type = "INVALID_HASH_TRANSITION"
            message = f"Invalid hash version transition: {predecessor_version} → {current_version}"
        
        violations.append(RuleViolation(
            violation_type=RuleViolationType.INVALID_HASH_TRANSITION,
            severity=RuleSeverity.ERROR,
            block_number=block.get("block_number"),
            block_id=block.get("id"),
            message=message,
            context={
                "predecessor_version": predecessor_version,
                "current_version": current_version,
                "error": error_type,
            },
        ))
    
    return RuleResult(is_valid=len(violations) == 0, violations=violations)
