"""
Consensus Reorganization (Reorg) Handler

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Consensus Runtime Activation
Date: 2025-12-06

Purpose:
    Handle chain reorganizations while preserving ledger monotonicity.
    
    MathLedger is append-only, so traditional blockchain reorgs (replacing blocks)
    are NOT ALLOWED. Instead, we handle "soft forks" by marking blocks as:
    - CANONICAL: Part of the main chain
    - ORPHANED: Valid but not part of main chain
    
    This module detects forks, resolves conflicts, and maintains chain integrity.

Design Principles:
    1. Monotonicity Preserved: No block deletion, only status changes
    2. Deterministic: Same fork → same resolution
    3. Auditable: All reorg decisions logged
    4. Reversible: Reorg decisions can be audited and verified

Reorg Policy:
    - Longest chain wins (most blocks)
    - Tie-breaker: Highest composite attestation root (lexicographic)
    - All reorg decisions are recorded in ledger
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class BlockStatus(Enum):
    """Block status in chain."""
    CANONICAL = "canonical"      # Part of main chain
    ORPHANED = "orphaned"        # Valid but not on main chain
    PENDING = "pending"          # Not yet confirmed
    INVALID = "invalid"          # Failed validation


class ReorgReason(Enum):
    """Reason for chain reorganization."""
    FORK_DETECTED = "fork_detected"              # Multiple blocks with same prev_hash
    LONGER_CHAIN = "longer_chain"                # Alternative chain is longer
    HIGHER_ATTESTATION = "higher_attestation"    # Tie-breaker: higher composite root
    MANUAL_OVERRIDE = "manual_override"          # Manual intervention


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Fork:
    """
    Represents a chain fork.
    
    Attributes:
        fork_point: Block number where fork occurred
        fork_block_id: Block ID at fork point
        branches: List of branch chains (each is list of blocks)
        detected_at: Timestamp when fork was detected
    """
    fork_point: int
    fork_block_id: int
    branches: List[List[Dict[str, Any]]]
    detected_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fork_point": self.fork_point,
            "fork_block_id": self.fork_block_id,
            "branch_count": len(self.branches),
            "branch_lengths": [len(b) for b in self.branches],
            "detected_at": self.detected_at,
        }


@dataclass
class ReorgDecision:
    """
    Represents a chain reorganization decision.
    
    Attributes:
        fork: Fork being resolved
        canonical_branch: Index of canonical branch
        orphaned_branches: Indices of orphaned branches
        reason: Reason for decision
        decided_at: Timestamp when decision was made
        metadata: Additional metadata
    """
    fork: Fork
    canonical_branch: int
    orphaned_branches: List[int]
    reason: ReorgReason
    decided_at: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fork": self.fork.to_dict(),
            "canonical_branch": self.canonical_branch,
            "orphaned_branches": self.orphaned_branches,
            "reason": self.reason.value,
            "decided_at": self.decided_at,
            "metadata": self.metadata,
        }


# ============================================================================
# FORK DETECTION
# ============================================================================

def detect_forks(blocks: List[Dict[str, Any]]) -> List[Fork]:
    """
    Detect forks in block list.
    
    Args:
        blocks: List of blocks (sorted by block_number)
    
    Returns:
        List of detected forks
    
    Fork Detection Algorithm:
        1. Group blocks by prev_hash
        2. If multiple blocks have same prev_hash → fork detected
        3. Build fork branches by following prev_hash chain
    
    Deterministic Ordering:
        - Blocks must be pre-sorted by block_number
        - Forks detected in sequential order
        - Branches sorted by first block ID
    
    Input Schema:
        blocks: [
            {
                "id": int,
                "block_number": int,
                "prev_hash": str (hex),
                ...
            },
            ...
        ]
    
    Output Schema:
        [
            Fork(
                fork_point=int,
                fork_block_id=int,
                branches=[[block, ...], [block, ...]],
                detected_at=str (ISO timestamp),
            ),
            ...
        ]
    """
    forks = []
    
    # Group blocks by prev_hash
    prev_hash_map: Dict[str, List[Dict[str, Any]]] = {}
    for block in blocks:
        prev_hash = block.get("prev_hash")
        if prev_hash is None:
            continue  # Genesis block
        
        if prev_hash not in prev_hash_map:
            prev_hash_map[prev_hash] = []
        prev_hash_map[prev_hash].append(block)
    
    # Detect forks (multiple blocks with same prev_hash)
    for prev_hash, fork_blocks in prev_hash_map.items():
        if len(fork_blocks) > 1:
            # Fork detected
            fork_point = fork_blocks[0]["block_number"] - 1
            
            # Find fork block (predecessor)
            fork_block = next((b for b in blocks if b["block_number"] == fork_point), None)
            fork_block_id = fork_block["id"] if fork_block else None
            
            # Build branches
            branches = []
            for fork_block in sorted(fork_blocks, key=lambda b: b["id"]):
                branch = build_branch(fork_block, blocks)
                branches.append(branch)
            
            # Create fork
            fork = Fork(
                fork_point=fork_point,
                fork_block_id=fork_block_id,
                branches=branches,
                detected_at=datetime.utcnow().isoformat() + "Z",
            )
            forks.append(fork)
    
    return forks


def build_branch(start_block: Dict[str, Any], all_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build branch chain starting from start_block.
    
    Args:
        start_block: Starting block
        all_blocks: All available blocks
    
    Returns:
        List of blocks in branch (sorted by block_number)
    
    Algorithm:
        1. Start with start_block
        2. Find next block (prev_hash = current block hash)
        3. Repeat until no next block found
    """
    branch = [start_block]
    current = start_block
    
    # Build block hash map for fast lookup
    block_hash_map = {b["id"]: compute_block_hash(b) for b in all_blocks}
    
    while True:
        current_hash = block_hash_map.get(current["id"])
        if current_hash is None:
            break
        
        # Find next block
        next_block = next((b for b in all_blocks if b.get("prev_hash") == current_hash), None)
        if next_block is None:
            break
        
        branch.append(next_block)
        current = next_block
    
    return branch


def compute_block_hash(block: Dict[str, Any]) -> str:
    """
    Compute block hash (simplified: use block ID).
    
    In production, this should use the actual block identity hash.
    """
    import hashlib
    return hashlib.sha256(str(block["id"]).encode()).hexdigest()


# ============================================================================
# FORK RESOLUTION
# ============================================================================

def resolve_fork(fork: Fork, policy: str = "longest_chain") -> ReorgDecision:
    """
    Resolve fork by selecting canonical branch.
    
    Args:
        fork: Fork to resolve
        policy: Resolution policy ("longest_chain" | "highest_attestation")
    
    Returns:
        ReorgDecision with canonical and orphaned branches
    
    Resolution Policies:
        - longest_chain: Longest branch wins
        - highest_attestation: Highest composite root wins (tie-breaker)
    
    Deterministic Ordering:
        - Branches compared in order
        - Tie-breaker: lexicographic comparison of composite roots
    
    Input Schema:
        fork: Fork(
            fork_point=int,
            fork_block_id=int,
            branches=[[block, ...], [block, ...]],
            detected_at=str,
        )
    
    Output Schema:
        ReorgDecision(
            fork=fork,
            canonical_branch=int (index),
            orphaned_branches=[int, ...] (indices),
            reason=ReorgReason,
            decided_at=str (ISO timestamp),
            metadata=dict,
        )
    """
    if policy == "longest_chain":
        return resolve_fork_longest_chain(fork)
    elif policy == "highest_attestation":
        return resolve_fork_highest_attestation(fork)
    else:
        raise ValueError(f"Unknown resolution policy: {policy}")


def resolve_fork_longest_chain(fork: Fork) -> ReorgDecision:
    """
    Resolve fork by selecting longest branch.
    
    Tie-breaker: Highest composite attestation root (lexicographic).
    """
    # Find longest branch
    branch_lengths = [len(b) for b in fork.branches]
    max_length = max(branch_lengths)
    
    # Find branches with max length
    longest_branches = [i for i, length in enumerate(branch_lengths) if length == max_length]
    
    if len(longest_branches) == 1:
        # Single longest branch
        canonical_branch = longest_branches[0]
        reason = ReorgReason.LONGER_CHAIN
    else:
        # Tie-breaker: Highest composite attestation root
        canonical_branch = max(
            longest_branches,
            key=lambda i: fork.branches[i][-1].get("composite_attestation_root", ""),
        )
        reason = ReorgReason.HIGHER_ATTESTATION
    
    # Orphaned branches
    orphaned_branches = [i for i in range(len(fork.branches)) if i != canonical_branch]
    
    # Create decision
    decision = ReorgDecision(
        fork=fork,
        canonical_branch=canonical_branch,
        orphaned_branches=orphaned_branches,
        reason=reason,
        decided_at=datetime.utcnow().isoformat() + "Z",
        metadata={
            "branch_lengths": branch_lengths,
            "max_length": max_length,
            "tie_breaker_used": len(longest_branches) > 1,
        },
    )
    
    return decision


def resolve_fork_highest_attestation(fork: Fork) -> ReorgDecision:
    """
    Resolve fork by selecting branch with highest composite attestation root.
    """
    # Find branch with highest composite root
    canonical_branch = max(
        range(len(fork.branches)),
        key=lambda i: fork.branches[i][-1].get("composite_attestation_root", ""),
    )
    
    # Orphaned branches
    orphaned_branches = [i for i in range(len(fork.branches)) if i != canonical_branch]
    
    # Create decision
    decision = ReorgDecision(
        fork=fork,
        canonical_branch=canonical_branch,
        orphaned_branches=orphaned_branches,
        reason=ReorgReason.HIGHER_ATTESTATION,
        decided_at=datetime.utcnow().isoformat() + "Z",
        metadata={
            "branch_lengths": [len(b) for b in fork.branches],
            "composite_roots": [b[-1].get("composite_attestation_root", "") for b in fork.branches],
        },
    )
    
    return decision


# ============================================================================
# REORG APPLICATION
# ============================================================================

def apply_reorg_decision(decision: ReorgDecision) -> Dict[str, Any]:
    """
    Apply reorg decision by updating block statuses.
    
    Args:
        decision: ReorgDecision to apply
    
    Returns:
        Dictionary with status updates
    
    Status Updates:
        - Canonical branch blocks → BlockStatus.CANONICAL
        - Orphaned branch blocks → BlockStatus.ORPHANED
    
    Monotonicity Preservation:
        - No blocks are deleted
        - Only status field is updated
        - All blocks remain in database
    
    Output Schema:
        {
            "canonical_blocks": [block_id, ...],
            "orphaned_blocks": [block_id, ...],
            "status_updates": [
                {"block_id": int, "old_status": str, "new_status": str},
                ...
            ],
        }
    """
    canonical_blocks = []
    orphaned_blocks = []
    status_updates = []
    
    # Mark canonical branch
    for block in decision.fork.branches[decision.canonical_branch]:
        canonical_blocks.append(block["id"])
        status_updates.append({
            "block_id": block["id"],
            "old_status": block.get("status", BlockStatus.PENDING.value),
            "new_status": BlockStatus.CANONICAL.value,
        })
    
    # Mark orphaned branches
    for branch_idx in decision.orphaned_branches:
        for block in decision.fork.branches[branch_idx]:
            orphaned_blocks.append(block["id"])
            status_updates.append({
                "block_id": block["id"],
                "old_status": block.get("status", BlockStatus.PENDING.value),
                "new_status": BlockStatus.ORPHANED.value,
            })
    
    return {
        "canonical_blocks": canonical_blocks,
        "orphaned_blocks": orphaned_blocks,
        "status_updates": status_updates,
    }


# ============================================================================
# REORG ORCHESTRATOR
# ============================================================================

class ReorgOrchestrator:
    """
    Orchestrates chain reorganization detection and resolution.
    
    Usage:
        orchestrator = ReorgOrchestrator(policy="longest_chain")
        forks = orchestrator.detect_forks(blocks)
        decisions = orchestrator.resolve_forks(forks)
        updates = orchestrator.apply_decisions(decisions)
    """
    
    def __init__(self, policy: str = "longest_chain"):
        """
        Initialize reorg orchestrator.
        
        Args:
            policy: Fork resolution policy
        """
        self.policy = policy
        self.fork_history: List[Fork] = []
        self.decision_history: List[ReorgDecision] = []
    
    def detect_forks(self, blocks: List[Dict[str, Any]]) -> List[Fork]:
        """Detect forks in block list."""
        forks = detect_forks(blocks)
        self.fork_history.extend(forks)
        return forks
    
    def resolve_forks(self, forks: List[Fork]) -> List[ReorgDecision]:
        """Resolve forks using configured policy."""
        decisions = [resolve_fork(fork, self.policy) for fork in forks]
        self.decision_history.extend(decisions)
        return decisions
    
    def apply_decisions(self, decisions: List[ReorgDecision]) -> List[Dict[str, Any]]:
        """Apply reorg decisions."""
        return [apply_reorg_decision(d) for d in decisions]
    
    def get_fork_history(self) -> List[Dict[str, Any]]:
        """Get fork detection history."""
        return [f.to_dict() for f in self.fork_history]
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get reorg decision history."""
        return [d.to_dict() for d in self.decision_history]
