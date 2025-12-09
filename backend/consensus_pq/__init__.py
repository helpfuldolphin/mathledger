"""
Post-Quantum Consensus Module

This module implements consensus validation rules for MathLedger's post-quantum
migration. It provides phase-aware block validation, epoch management, prev_hash
verification, reorganization handling, and consensus violation detection.

Module Structure:
- rules.py: Phase-specific consensus rules
- validation.py: Block validation logic
- epoch.py: Epoch management and resolution
- reorg.py: Reorganization handling
- violations.py: Consensus violation detection and classification
- prev_hash.py: Prev-hash chain validation

Author: Manus-H
Version: 1.0
"""

from backend.consensus_pq.rules import (
    ConsensusRuleVersion,
    get_consensus_rules_for_phase,
    validate_consensus_rules,
)
from backend.consensus_pq.validation import (
    validate_block_full,
    validate_block_header,
    validate_merkle_root,
    validate_dual_commitment,
)
from backend.consensus_pq.epoch import (
    HashEpoch,
    get_epoch_for_block,
    get_current_epoch,
    register_epoch,
    list_epochs,
)
from backend.consensus_pq.reorg import (
    evaluate_reorg,
    can_reorg_to_chain,
    find_fork_point,
    validate_reorg_constraints,
)
from backend.consensus_pq.violations import (
    ConsensusViolation,
    ViolationType,
    detect_violations,
    classify_violation,
)
from backend.consensus_pq.prev_hash import (
    validate_prev_hash_linkage,
    compute_prev_hash,
    validate_dual_prev_hash,
)

__all__ = [
    # Rules
    "ConsensusRuleVersion",
    "get_consensus_rules_for_phase",
    "validate_consensus_rules",
    # Validation
    "validate_block_full",
    "validate_block_header",
    "validate_merkle_root",
    "validate_dual_commitment",
    # Epoch
    "HashEpoch",
    "get_epoch_for_block",
    "get_current_epoch",
    "register_epoch",
    "list_epochs",
    # Reorg
    "evaluate_reorg",
    "can_reorg_to_chain",
    "find_fork_point",
    "validate_reorg_constraints",
    # Violations
    "ConsensusViolation",
    "ViolationType",
    "detect_violations",
    "classify_violation",
    # Prev-hash
    "validate_prev_hash_linkage",
    "compute_prev_hash",
    "validate_dual_prev_hash",
]

__version__ = "1.0.0"
__author__ = "Manus-H"
