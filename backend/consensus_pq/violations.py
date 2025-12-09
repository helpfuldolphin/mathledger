"""
Consensus Violations Module

Detects and classifies consensus violations during block validation.

Author: Manus-H
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from basis.ledger.block_pq import BlockHeaderPQ
from backend.consensus_pq.validation import (
    validate_block_full,
    validate_merkle_root,
    validate_dual_commitment,
)
from backend.consensus_pq.prev_hash import validate_prev_hash_linkage
from backend.consensus_pq.epoch import get_epoch_for_block
from backend.consensus_pq.rules import ConsensusRuleVersion, get_consensus_rules_for_phase


class ViolationType(Enum):
    """Types of consensus violations."""
    
    INVALID_MERKLE_ROOT = "invalid_merkle_root"
    INVALID_PQ_MERKLE_ROOT = "invalid_pq_merkle_root"
    INVALID_PREV_HASH = "invalid_prev_hash"
    INVALID_PQ_PREV_HASH = "invalid_pq_prev_hash"
    INVALID_DUAL_COMMITMENT = "invalid_dual_commitment"
    MISSING_PQ_FIELDS = "missing_pq_fields"
    MISSING_LEGACY_FIELDS = "missing_legacy_fields"
    ALGORITHM_MISMATCH = "algorithm_mismatch"
    EPOCH_CUTOVER_VIOLATION = "epoch_cutover_violation"
    TIMESTAMP_VIOLATION = "timestamp_violation"
    BLOCK_NUMBER_DISCONTINUITY = "block_number_discontinuity"
    UNKNOWN = "unknown"


class ViolationSeverity(Enum):
    """Severity levels for consensus violations."""
    
    CRITICAL = "critical"  # Consensus-breaking, reject block immediately
    HIGH = "high"  # Serious issue, reject block
    MEDIUM = "medium"  # Warning, may reject block
    LOW = "low"  # Informational


@dataclass
class ConsensusViolation:
    """
    Represents a consensus violation detected during validation.
    
    Attributes:
        violation_type: The type of violation
        severity: The severity level
        block_number: The block number where violation occurred
        block_hash: The block hash (if computable)
        message: Human-readable description of the violation
        details: Additional details (optional)
    """
    
    violation_type: ViolationType
    severity: ViolationSeverity
    block_number: int
    block_hash: Optional[str]
    message: str
    details: Optional[dict] = None


def classify_violation(violation_type: ViolationType) -> ViolationSeverity:
    """
    Classify the severity of a violation type.
    
    Args:
        violation_type: The type of violation
        
    Returns:
        The severity level for this violation type
    """
    # Critical violations (consensus-breaking)
    critical_violations = {
        ViolationType.INVALID_MERKLE_ROOT,
        ViolationType.INVALID_PQ_MERKLE_ROOT,
        ViolationType.INVALID_PREV_HASH,
        ViolationType.INVALID_PQ_PREV_HASH,
        ViolationType.INVALID_DUAL_COMMITMENT,
        ViolationType.ALGORITHM_MISMATCH,
        ViolationType.EPOCH_CUTOVER_VIOLATION,
    }
    
    # High severity violations
    high_violations = {
        ViolationType.MISSING_PQ_FIELDS,
        ViolationType.MISSING_LEGACY_FIELDS,
        ViolationType.BLOCK_NUMBER_DISCONTINUITY,
    }
    
    # Medium severity violations
    medium_violations = {
        ViolationType.TIMESTAMP_VIOLATION,
    }
    
    if violation_type in critical_violations:
        return ViolationSeverity.CRITICAL
    elif violation_type in high_violations:
        return ViolationSeverity.HIGH
    elif violation_type in medium_violations:
        return ViolationSeverity.MEDIUM
    else:
        return ViolationSeverity.LOW


def detect_violations(
    block: BlockHeaderPQ,
    prev_block: Optional[BlockHeaderPQ] = None,
) -> List[ConsensusViolation]:
    """
    Detect all consensus violations in a block.
    
    This performs comprehensive validation and returns a list of all
    violations found. If the list is empty, the block is valid.
    
    Args:
        block: The block header to validate
        prev_block: The previous block header (optional)
        
    Returns:
        List of ConsensusViolation objects (empty if block is valid)
    """
    violations = []
    
    # Get epoch and rules
    epoch = get_epoch_for_block(block.block_number)
    if epoch is None:
        violations.append(ConsensusViolation(
            violation_type=ViolationType.UNKNOWN,
            severity=ViolationSeverity.CRITICAL,
            block_number=block.block_number,
            block_hash=None,
            message=f"No epoch registered for block {block.block_number}",
        ))
        return violations
    
    rule_version = ConsensusRuleVersion(epoch.rule_version)
    rules = get_consensus_rules_for_phase(rule_version)
    
    # Check for missing PQ fields
    if rules.pq_fields_required and not block.has_dual_commitment():
        violations.append(ConsensusViolation(
            violation_type=ViolationType.MISSING_PQ_FIELDS,
            severity=classify_violation(ViolationType.MISSING_PQ_FIELDS),
            block_number=block.block_number,
            block_hash=None,
            message=f"PQ fields required by {rule_version.value} but missing",
        ))
    
    # Check for missing legacy fields
    if rules.legacy_fields_required:
        if block.merkle_root is None or block.prev_hash is None:
            violations.append(ConsensusViolation(
                violation_type=ViolationType.MISSING_LEGACY_FIELDS,
                severity=classify_violation(ViolationType.MISSING_LEGACY_FIELDS),
                block_number=block.block_number,
                block_hash=None,
                message=f"Legacy fields required by {rule_version.value} but missing",
            ))
    
    # Validate legacy Merkle root
    if rules.legacy_fields_required and block.merkle_root is not None:
        legacy_valid, legacy_error = validate_merkle_root(block, algorithm_id=0x00)
        if not legacy_valid:
            violations.append(ConsensusViolation(
                violation_type=ViolationType.INVALID_MERKLE_ROOT,
                severity=classify_violation(ViolationType.INVALID_MERKLE_ROOT),
                block_number=block.block_number,
                block_hash=None,
                message=f"Invalid legacy Merkle root: {legacy_error}",
            ))
    
    # Validate PQ Merkle root
    if rules.pq_fields_required and block.pq_merkle_root is not None:
        pq_valid, pq_error = validate_merkle_root(block, algorithm_id=block.pq_algorithm)
        if not pq_valid:
            violations.append(ConsensusViolation(
                violation_type=ViolationType.INVALID_PQ_MERKLE_ROOT,
                severity=classify_violation(ViolationType.INVALID_PQ_MERKLE_ROOT),
                block_number=block.block_number,
                block_hash=None,
                message=f"Invalid PQ Merkle root: {pq_error}",
            ))
    
    # Validate dual commitment
    if rules.dual_commitment_required and block.dual_commitment is not None:
        commitment_valid, commitment_error = validate_dual_commitment(block)
        if not commitment_valid:
            violations.append(ConsensusViolation(
                violation_type=ViolationType.INVALID_DUAL_COMMITMENT,
                severity=classify_violation(ViolationType.INVALID_DUAL_COMMITMENT),
                block_number=block.block_number,
                block_hash=None,
                message=f"Invalid dual commitment: {commitment_error}",
            ))
    
    # Validate prev_hash linkage
    if prev_block is not None and block.block_number > 0:
        linkage_valid, linkage_error = validate_prev_hash_linkage(block, prev_block)
        if not linkage_valid:
            # Determine if it's legacy or PQ prev_hash issue
            if "Legacy" in linkage_error or "legacy" in linkage_error:
                violation_type = ViolationType.INVALID_PREV_HASH
            elif "PQ" in linkage_error or "pq" in linkage_error:
                violation_type = ViolationType.INVALID_PQ_PREV_HASH
            else:
                violation_type = ViolationType.UNKNOWN
            
            violations.append(ConsensusViolation(
                violation_type=violation_type,
                severity=classify_violation(violation_type),
                block_number=block.block_number,
                block_hash=None,
                message=f"Invalid prev_hash linkage: {linkage_error}",
            ))
    
    # Check algorithm mismatch
    if block.pq_algorithm is not None:
        if block.pq_algorithm != epoch.algorithm_id:
            violations.append(ConsensusViolation(
                violation_type=ViolationType.ALGORITHM_MISMATCH,
                severity=classify_violation(ViolationType.ALGORITHM_MISMATCH),
                block_number=block.block_number,
                block_hash=None,
                message=(
                    f"Algorithm mismatch: epoch requires {epoch.algorithm_id:02x}, "
                    f"block uses {block.pq_algorithm:02x}"
                ),
                details={
                    "expected_algorithm": epoch.algorithm_id,
                    "actual_algorithm": block.pq_algorithm,
                },
            ))
    
    # Check timestamp ordering
    if prev_block is not None:
        if block.timestamp <= prev_block.timestamp:
            violations.append(ConsensusViolation(
                violation_type=ViolationType.TIMESTAMP_VIOLATION,
                severity=classify_violation(ViolationType.TIMESTAMP_VIOLATION),
                block_number=block.block_number,
                block_hash=None,
                message=(
                    f"Timestamp must increase: "
                    f"prev={prev_block.timestamp}, current={block.timestamp}"
                ),
            ))
    
    # Check block number continuity
    if prev_block is not None:
        if block.block_number != prev_block.block_number + 1:
            violations.append(ConsensusViolation(
                violation_type=ViolationType.BLOCK_NUMBER_DISCONTINUITY,
                severity=classify_violation(ViolationType.BLOCK_NUMBER_DISCONTINUITY),
                block_number=block.block_number,
                block_hash=None,
                message=(
                    f"Block number discontinuity: "
                    f"prev={prev_block.block_number}, current={block.block_number}"
                ),
            ))
    
    return violations


def has_critical_violations(violations: List[ConsensusViolation]) -> bool:
    """
    Check if a list of violations contains any critical violations.
    
    Args:
        violations: List of violations to check
        
    Returns:
        True if any critical violations present, False otherwise
    """
    return any(v.severity == ViolationSeverity.CRITICAL for v in violations)


def format_violation_report(violations: List[ConsensusViolation]) -> str:
    """
    Format a list of violations into a human-readable report.
    
    Args:
        violations: List of violations to format
        
    Returns:
        Formatted report string
    """
    if not violations:
        return "No violations detected"
    
    lines = [f"Detected {len(violations)} violation(s):\n"]
    
    for i, violation in enumerate(violations, 1):
        lines.append(
            f"{i}. [{violation.severity.value.upper()}] "
            f"{violation.violation_type.value} at block {violation.block_number}"
        )
        lines.append(f"   {violation.message}")
        if violation.details:
            lines.append(f"   Details: {violation.details}")
        lines.append("")
    
    return "\n".join(lines)
