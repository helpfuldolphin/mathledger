"""
PHASE II — NOT USED IN PHASE I
This module provides pure, deterministic functions for computing slice-specific success metrics.
"""

from typing import List, Set, Dict, Any, Tuple

# A statement is represented as a dictionary-like object with at least a 'hash' key.
Statement = Dict[str, Any]

def compute_goal_hit(
    verified_statements: List[Statement],
    target_hashes: Set[str],
    min_total_verified: int,
) -> Tuple[bool, float]:
    """
    Computes success based on hitting a minimum number of specific target goals.

    Args:
        verified_statements: A list of verified statement objects, each with a 'hash'.
        target_hashes: A set of goal statement hashes.
        min_total_verified: The minimum number of target hashes that must be verified.

    Returns:
        A tuple of (success, metric_value), where metric_value is the count of
        verified targets.
    """
    verified_hashes = {s['hash'] for s in verified_statements}
    hits = len(verified_hashes.intersection(target_hashes))
    success = hits >= min_total_verified
    return success, float(hits)


def compute_sparse_success(
    verified_count: int,
    attempted_count: int,
    min_verified: int,
) -> Tuple[bool, float]:
    """
    Computes success based on a simple minimum count of verified statements.

    Args:
        verified_count: The total number of verified statements.
        attempted_count: The total number of attempted statements (ignored, for API compatibility).
        min_verified: The minimum number of verified statements for success.

    Returns:
        A tuple of (success, metric_value), where metric_value is the verified_count.
    """
    # attempted_count is included for API consistency but not used in this metric's logic.
    _ = attempted_count
    success = verified_count >= min_verified
    return success, float(verified_count)


def compute_chain_success(
    verified_statements: List[Statement],
    dependency_graph: Dict[str, List[str]],
    chain_target_hash: str,
    min_chain_length: int,
) -> Tuple[bool, float]:
    """
    Computes success based on the length of a verified dependency chain ending at a target.

    Args:
        verified_statements: A list of verified statement objects, each with a 'hash'.
        dependency_graph: A dict mapping statement hashes to their dependency hashes.
        chain_target_hash: The hash of the final statement in the chain.
        min_chain_length: The minimum required length of the verified chain.

    Returns:
        A tuple of (success, metric_value), where metric_value is the longest
        verified chain length ending at the target.

    Notes:
        - Handles cycles in the dependency graph by tracking visited nodes.
        - Cycles are treated as terminating the chain (nodes in a cycle are counted once).
        - Deterministic: always produces the same output for the same inputs.
    """
    verified_hashes = {s['hash'] for s in verified_statements}
    memo: Dict[str, int] = {}
    # Track nodes currently being visited to detect cycles
    visiting: Set[str] = set()

    def get_max_chain_length(current_hash: str) -> int:
        if current_hash not in verified_hashes:
            return 0
        if current_hash in memo:
            return memo[current_hash]
        # Cycle detection: if we're already visiting this node, treat as terminal
        if current_hash in visiting:
            return 1  # Count this node but don't recurse further

        visiting.add(current_hash)

        dependencies = dependency_graph.get(current_hash, [])
        if not dependencies:
            max_prev_len = 0
        else:
            max_prev_len = max(
                [get_max_chain_length(dep) for dep in dependencies] + [0]
            )

        visiting.discard(current_hash)

        length = 1 + max_prev_len
        memo[current_hash] = length
        return length

    chain_len = get_max_chain_length(chain_target_hash)
    success = chain_len >= min_chain_length
    return success, float(chain_len)


def compute_multi_goal_success(
    verified_hashes: Set[str],
    required_goal_hashes: Set[str],
) -> Tuple[bool, float]:
    """
    Computes success based on verifying a set of required goals.

    Args:
        verified_hashes: A set of hashes that have been verified.
        required_goal_hashes: A set of hashes that must all be verified for success.

    Returns:
        A tuple of (success, metric_value), where metric_value is the number
        of required goals that were met.
    """
    if not required_goal_hashes:
        return True, 0.0

    met_goals = verified_hashes.intersection(required_goal_hashes)
    num_met = len(met_goals)
    success = num_met == len(required_goal_hashes)
    return success, float(num_met)