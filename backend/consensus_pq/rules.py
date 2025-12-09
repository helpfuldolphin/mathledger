"""
Consensus Rules Module

Implements phase-specific consensus rules for post-quantum migration.
Each migration phase has different validation requirements.

Author: Manus-H
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from basis.ledger.block_pq import BlockHeaderPQ


class ConsensusRuleVersion(Enum):
    """Consensus rule versions corresponding to migration phases."""
    
    V1_LEGACY = "v1-legacy"  # Phase 0-1: SHA-256 only
    V2_DUAL_OPTIONAL = "v2-dual-optional"  # Phase 2: Dual optional
    V2_DUAL_REQUIRED = "v2-dual-required"  # Phase 3: Dual required
    V2_PQ_PRIMARY = "v2-pq-primary"  # Phase 4: PQ primary
    V3_PQ_ONLY = "v3-pq-only"  # Phase 5: PQ only


@dataclass(frozen=True)
class ConsensusRules:
    """
    Consensus rules for a specific migration phase.
    
    Attributes:
        version: Rule version identifier
        pq_fields_required: Whether PQ fields must be present
        pq_fields_validated: Whether PQ fields are validated if present
        legacy_fields_required: Whether legacy fields must be present
        legacy_fields_canonical: Whether legacy fields are canonical for consensus
        pq_fields_canonical: Whether PQ fields are canonical for consensus
        dual_commitment_required: Whether dual commitment must be present
        dual_commitment_validated: Whether dual commitment is validated
    """
    
    version: ConsensusRuleVersion
    pq_fields_required: bool
    pq_fields_validated: bool
    legacy_fields_required: bool
    legacy_fields_canonical: bool
    pq_fields_canonical: bool
    dual_commitment_required: bool
    dual_commitment_validated: bool


# Define rules for each phase
CONSENSUS_RULES_MAP = {
    ConsensusRuleVersion.V1_LEGACY: ConsensusRules(
        version=ConsensusRuleVersion.V1_LEGACY,
        pq_fields_required=False,
        pq_fields_validated=False,
        legacy_fields_required=True,
        legacy_fields_canonical=True,
        pq_fields_canonical=False,
        dual_commitment_required=False,
        dual_commitment_validated=False,
    ),
    ConsensusRuleVersion.V2_DUAL_OPTIONAL: ConsensusRules(
        version=ConsensusRuleVersion.V2_DUAL_OPTIONAL,
        pq_fields_required=False,
        pq_fields_validated=True,  # Validate if present
        legacy_fields_required=True,
        legacy_fields_canonical=True,
        pq_fields_canonical=False,
        dual_commitment_required=False,
        dual_commitment_validated=True,  # Validate if present
    ),
    ConsensusRuleVersion.V2_DUAL_REQUIRED: ConsensusRules(
        version=ConsensusRuleVersion.V2_DUAL_REQUIRED,
        pq_fields_required=True,
        pq_fields_validated=True,
        legacy_fields_required=True,
        legacy_fields_canonical=True,  # Legacy still canonical
        pq_fields_canonical=False,
        dual_commitment_required=True,
        dual_commitment_validated=True,
    ),
    ConsensusRuleVersion.V2_PQ_PRIMARY: ConsensusRules(
        version=ConsensusRuleVersion.V2_PQ_PRIMARY,
        pq_fields_required=True,
        pq_fields_validated=True,
        legacy_fields_required=True,  # Still required for compatibility
        legacy_fields_canonical=False,  # PQ now canonical
        pq_fields_canonical=True,
        dual_commitment_required=True,
        dual_commitment_validated=True,
    ),
    ConsensusRuleVersion.V3_PQ_ONLY: ConsensusRules(
        version=ConsensusRuleVersion.V3_PQ_ONLY,
        pq_fields_required=True,
        pq_fields_validated=True,
        legacy_fields_required=False,  # Legacy now optional
        legacy_fields_canonical=False,
        pq_fields_canonical=True,
        dual_commitment_required=False,  # No longer needed
        dual_commitment_validated=False,
    ),
}


def get_consensus_rules_for_phase(rule_version: ConsensusRuleVersion) -> ConsensusRules:
    """
    Get consensus rules for a specific phase.
    
    Args:
        rule_version: The consensus rule version
        
    Returns:
        ConsensusRules object for the specified phase
        
    Raises:
        ValueError: If rule_version is not recognized
    """
    if rule_version not in CONSENSUS_RULES_MAP:
        raise ValueError(f"Unknown consensus rule version: {rule_version}")
    
    return CONSENSUS_RULES_MAP[rule_version]


def validate_consensus_rules(
    block: BlockHeaderPQ,
    rules: ConsensusRules,
) -> tuple[bool, Optional[str]]:
    """
    Validate that a block conforms to consensus rules for its phase.
    
    Args:
        block: The block header to validate
        rules: The consensus rules to apply
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
        If invalid, error_message describes the violation
    """
    # Check PQ fields presence
    has_pq_fields = block.has_dual_commitment()
    
    if rules.pq_fields_required and not has_pq_fields:
        return False, f"PQ fields required by {rules.version.value} but missing"
    
    # Check legacy fields presence
    has_legacy_fields = (
        block.merkle_root is not None and
        block.prev_hash is not None
    )
    
    if rules.legacy_fields_required and not has_legacy_fields:
        return False, f"Legacy fields required by {rules.version.value} but missing"
    
    # Check dual commitment presence
    has_dual_commitment = block.dual_commitment is not None
    
    if rules.dual_commitment_required and not has_dual_commitment:
        return False, f"Dual commitment required by {rules.version.value} but missing"
    
    # All structural requirements met
    return True, None


def get_canonical_hash_fields(
    block: BlockHeaderPQ,
    rules: ConsensusRules,
) -> tuple[str, Optional[str]]:
    """
    Get the canonical hash fields for a block based on consensus rules.
    
    Args:
        block: The block header
        rules: The consensus rules
        
    Returns:
        Tuple of (canonical_merkle_root, canonical_prev_hash)
        Returns the fields that are canonical for consensus decisions
    """
    if rules.pq_fields_canonical:
        # PQ fields are canonical
        return block.pq_merkle_root, block.pq_prev_hash
    else:
        # Legacy fields are canonical
        return block.merkle_root, block.prev_hash


def requires_dual_validation(rules: ConsensusRules) -> bool:
    """
    Check if consensus rules require validating both hash chains.
    
    Args:
        rules: The consensus rules
        
    Returns:
        True if both chains must be validated, False otherwise
    """
    return (
        rules.pq_fields_validated and
        rules.legacy_fields_required and
        rules.pq_fields_required
    )
