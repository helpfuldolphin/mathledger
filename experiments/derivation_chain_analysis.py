# experiments/derivation_chain_analysis.py
"""
PHASE II -- NOT USED IN PHASE I

Derivation Chain Analysis
=========================

Tools for analyzing derivation chains from proof logs. This module provides
utilities for constructing and analyzing dependency DAGs from derivation records.

Module Responsibilities:
  - Construct proof dependency DAGs from derivation lists
  - Compute chain depths (longest paths from axioms to targets)
  - Extract dependency relationships for slice analysis

All functions are:
  - Pure (no side effects)
  - Deterministic (same inputs always produce same outputs)
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

# Type aliases for clarity
Derivation = Dict[str, Any]
DAG = Dict[str, List[str]]


def construct_dependency_dag(derivations: List[Derivation]) -> DAG:
    """Construct a proof dependency DAG from a list of derivations.

    The DAG is represented as a dictionary where keys are conclusion hashes
    and values are lists of their premise hashes. Axioms (nodes with no premises)
    are represented as keys with empty lists.

    Args:
        derivations: List of derivation records, each containing:
            - 'hash': The conclusion hash (required)
            - 'premises': List of premise hashes (optional, defaults to [])

    Returns:
        DAG mapping conclusion hashes to their premise hash lists.
        All nodes (including axioms) are present as keys.

    Example:
        >>> derivations = [
        ...     {'hash': 'h1', 'premises': []},       # axiom
        ...     {'hash': 'h2', 'premises': ['h1']},   # derived from h1
        ... ]
        >>> dag = construct_dependency_dag(derivations)
        >>> dag['h2']
        ['h1']
    """
    dag: DAG = {}
    for derivation in derivations:
        conclusion_hash = derivation.get('hash')
        premise_hashes = derivation.get('premises', [])
        if conclusion_hash:
            if conclusion_hash not in dag:
                dag[conclusion_hash] = []
            dag[conclusion_hash].extend(premise_hashes)

    # Add premises that might not be conclusions of other derivations (axioms)
    all_premises: Set[str] = set()
    for premises in dag.values():
        all_premises.update(premises)

    for premise in all_premises:
        if premise not in dag:
            dag[premise] = []

    return dag


def compute_chain_depth(target_hash: str, dag: DAG) -> int:
    """Compute the maximum dependency chain length ending at target_hash.

    This is the longest path from an axiom (a node with no premises) to the target.
    Uses memoization for efficiency on shared sub-DAGs.

    Args:
        target_hash: The hash of the target node to compute depth for.
        dag: The dependency DAG mapping nodes to their premise lists.

    Returns:
        The maximum chain depth (1 for axioms, 0 if target not in DAG).

    Note:
        Depth is 1-indexed: axioms have depth 1, direct conclusions have depth 2, etc.
        If the target is not present in the DAG, returns 0.
    """
    memo: Dict[str, int] = {}

    def _get_depth(h: str) -> int:
        if h in memo:
            return memo[h]

        if h not in dag:
            return 1

        premises = dag[h]
        if not premises:
            memo[h] = 1
            return 1

        max_premise_depth = 0
        for p_hash in premises:
            max_premise_depth = max(max_premise_depth, _get_depth(p_hash))

        depth = 1 + max_premise_depth
        memo[h] = depth
        return depth

    if target_hash not in dag:
        return 0

    return _get_depth(target_hash)


def extract_dependency_graph(derivation: Derivation) -> Tuple[str, List[str]]:
    """Extract direct dependencies from a single derivation record.

    Utility for slice_uplift_tree. Returns the conclusion hash and its
    direct premise hashes.

    Args:
        derivation: A derivation record containing 'hash' and optional 'premises'.

    Returns:
        Tuple of (conclusion_hash, premise_hashes).
        conclusion_hash is empty string if 'hash' key is missing.
        premise_hashes is empty list if 'premises' key is missing.
    """
    conclusion_hash = derivation.get('hash', '')
    premise_hashes = derivation.get('premises', [])
    return (conclusion_hash, premise_hashes)
