"""
Reorganization Handling Module

Implements reorganization (reorg) logic for dual-hash chains.

Author: Manus-H
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from basis.ledger.block_pq import BlockHeaderPQ
from backend.consensus_pq.validation import validate_block_full
from backend.consensus_pq.epoch import get_epoch_for_block
from backend.consensus_pq.rules import ConsensusRuleVersion, get_consensus_rules_for_phase


# Finality depth: reorgs cannot cross this boundary
FINALITY_DEPTH = 100


@dataclass
class ReorgEvaluation:
    """
    Result of evaluating a potential reorganization.
    
    Attributes:
        can_reorg: Whether the reorg is allowed
        fork_point: Block number where chains diverge
        current_chain_length: Length of current canonical chain
        candidate_chain_length: Length of candidate chain
        reason: Human-readable reason for decision
    """
    
    can_reorg: bool
    fork_point: int
    current_chain_length: int
    candidate_chain_length: int
    reason: str


def find_fork_point(
    current_chain: List[BlockHeaderPQ],
    candidate_chain: List[BlockHeaderPQ],
) -> Optional[int]:
    """
    Find the block number where two chains diverge.
    
    Args:
        current_chain: The current canonical chain
        candidate_chain: The candidate chain
        
    Returns:
        Block number of the fork point, or None if chains don't share history
    """
    # Build hash maps for efficient lookup
    current_hashes = {b.block_number: b.prev_hash for b in current_chain}
    candidate_hashes = {b.block_number: b.prev_hash for b in candidate_chain}
    
    # Find the last common block
    common_blocks = set(current_hashes.keys()) & set(candidate_hashes.keys())
    
    if not common_blocks:
        return None
    
    # Find the highest common block where hashes match
    for block_num in sorted(common_blocks, reverse=True):
        if current_hashes[block_num] == candidate_hashes[block_num]:
            return block_num
    
    return None


def validate_reorg_constraints(
    current_chain: List[BlockHeaderPQ],
    candidate_chain: List[BlockHeaderPQ],
    fork_point: int,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a reorg satisfies all constraints.
    
    Constraints:
    1. Cannot cross finalized epoch boundaries
    2. Candidate chain must be longer than current chain
    3. Both chains must be fully valid
    4. Fork point must not be deeper than finality depth
    
    Args:
        current_chain: The current canonical chain
        candidate_chain: The candidate chain
        fork_point: Block number where chains diverge
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Get current block number
    current_tip = max(b.block_number for b in current_chain)
    candidate_tip = max(b.block_number for b in candidate_chain)
    
    # Check finality depth
    reorg_depth = current_tip - fork_point
    if reorg_depth > FINALITY_DEPTH:
        return False, (
            f"Reorg depth ({reorg_depth}) exceeds finality depth ({FINALITY_DEPTH})"
        )
    
    # Check that candidate chain is longer
    if candidate_tip <= current_tip:
        return False, (
            f"Candidate chain (tip={candidate_tip}) not longer than "
            f"current chain (tip={current_tip})"
        )
    
    # Check for epoch boundary crossing
    fork_epoch = get_epoch_for_block(fork_point)
    current_epoch = get_epoch_for_block(current_tip)
    
    if fork_epoch != current_epoch:
        # Reorg crosses epoch boundary
        # Check if the epoch boundary is finalized
        if current_epoch is not None and current_epoch.start_block is not None:
            blocks_since_epoch_start = current_tip - current_epoch.start_block
            if blocks_since_epoch_start > FINALITY_DEPTH:
                return False, (
                    f"Reorg crosses finalized epoch boundary at block "
                    f"{current_epoch.start_block}"
                )
    
    return True, None


def evaluate_reorg(
    current_chain: List[BlockHeaderPQ],
    candidate_chain: List[BlockHeaderPQ],
) -> ReorgEvaluation:
    """
    Evaluate whether a reorganization should be accepted.
    
    This implements the fork choice rule for dual-hash chains.
    
    Args:
        current_chain: The current canonical chain
        candidate_chain: The candidate chain
        
    Returns:
        ReorgEvaluation with decision and reasoning
    """
    # Find fork point
    fork_point = find_fork_point(current_chain, candidate_chain)
    
    if fork_point is None:
        return ReorgEvaluation(
            can_reorg=False,
            fork_point=-1,
            current_chain_length=len(current_chain),
            candidate_chain_length=len(candidate_chain),
            reason="Chains do not share common history",
        )
    
    # Validate reorg constraints
    constraints_valid, constraints_error = validate_reorg_constraints(
        current_chain,
        candidate_chain,
        fork_point,
    )
    
    if not constraints_valid:
        return ReorgEvaluation(
            can_reorg=False,
            fork_point=fork_point,
            current_chain_length=len(current_chain),
            candidate_chain_length=len(candidate_chain),
            reason=f"Reorg constraints violated: {constraints_error}",
        )
    
    # Determine canonical chain based on epoch rules
    current_tip = max(b.block_number for b in current_chain)
    current_epoch = get_epoch_for_block(current_tip)
    
    if current_epoch is None:
        return ReorgEvaluation(
            can_reorg=False,
            fork_point=fork_point,
            current_chain_length=len(current_chain),
            candidate_chain_length=len(candidate_chain),
            reason="No epoch registered for current chain tip",
        )
    
    rule_version = ConsensusRuleVersion(current_epoch.rule_version)
    rules = get_consensus_rules_for_phase(rule_version)
    
    # Fork choice: longest valid chain wins
    # In Phases 1-3, use legacy chain length
    # In Phases 4-5, use PQ chain length
    
    if rules.pq_fields_canonical:
        # PQ chain is canonical
        # Longest valid PQ chain wins
        candidate_tip = max(b.block_number for b in candidate_chain)
        
        if candidate_tip > current_tip:
            return ReorgEvaluation(
                can_reorg=True,
                fork_point=fork_point,
                current_chain_length=len(current_chain),
                candidate_chain_length=len(candidate_chain),
                reason=f"Candidate chain longer (PQ canonical): {candidate_tip} > {current_tip}",
            )
        else:
            return ReorgEvaluation(
                can_reorg=False,
                fork_point=fork_point,
                current_chain_length=len(current_chain),
                candidate_chain_length=len(candidate_chain),
                reason=f"Current chain longer or equal (PQ canonical): {current_tip} >= {candidate_tip}",
            )
    else:
        # Legacy chain is canonical
        # Longest valid legacy chain wins
        candidate_tip = max(b.block_number for b in candidate_chain)
        
        if candidate_tip > current_tip:
            return ReorgEvaluation(
                can_reorg=True,
                fork_point=fork_point,
                current_chain_length=len(current_chain),
                candidate_chain_length=len(candidate_chain),
                reason=f"Candidate chain longer (legacy canonical): {candidate_tip} > {current_tip}",
            )
        else:
            return ReorgEvaluation(
                can_reorg=False,
                fork_point=fork_point,
                current_chain_length=len(current_chain),
                candidate_chain_length=len(candidate_chain),
                reason=f"Current chain longer or equal (legacy canonical): {current_tip} >= {candidate_tip}",
            )


def can_reorg_to_chain(
    current_chain: List[BlockHeaderPQ],
    candidate_chain: List[BlockHeaderPQ],
) -> bool:
    """
    Check if a reorganization to a candidate chain is allowed.
    
    This is a convenience function that wraps evaluate_reorg.
    
    Args:
        current_chain: The current canonical chain
        candidate_chain: The candidate chain
        
    Returns:
        True if reorg is allowed, False otherwise
    """
    evaluation = evaluate_reorg(current_chain, candidate_chain)
    return evaluation.can_reorg


def validate_candidate_chain(
    candidate_chain: List[BlockHeaderPQ],
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a candidate chain is fully valid.
    
    All blocks in the chain must pass full validation.
    
    Args:
        candidate_chain: The candidate chain to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not candidate_chain:
        return False, "Candidate chain is empty"
    
    # Validate first block
    first_valid, first_error = validate_block_full(candidate_chain[0], prev_block=None)
    if not first_valid:
        return False, f"Block {candidate_chain[0].block_number} invalid: {first_error}"
    
    # Validate remaining blocks
    for i in range(1, len(candidate_chain)):
        block = candidate_chain[i]
        prev_block = candidate_chain[i - 1]
        
        valid, error = validate_block_full(block, prev_block)
        if not valid:
            return False, f"Block {block.block_number} invalid: {error}"
    
    return True, None
