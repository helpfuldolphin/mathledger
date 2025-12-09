# experiments/derivation_chain_analysis.py
"""
PHASE II - NOT USED IN PHASE I
Tools for analyzing derivation chains from proof logs.
"""
from typing import List, Dict, Any, Tuple, Set

# Assuming a derivation is represented as a dictionary, e.g.:
# {'hash': 'conclusion_hash', 'premises': ['premise_hash_1', 'premise_hash_2']}
Derivation = Dict[str, Any]
DAG = Dict[str, List[str]]


class ChainAnalyzer:
    """
    Analyzer for derivation chains from proof logs.

    Provides methods to construct dependency DAGs, compute chain depths,
    and extract dependency graphs from derivations.
    """

    def __init__(self, derivations: List[Derivation] = None):
        """
        Initialize ChainAnalyzer with optional derivations.

        Args:
            derivations: Optional list of derivation dictionaries.
        """
        self._derivations = derivations or []
        self._dag: DAG = {}
        if self._derivations:
            self._dag = construct_dependency_dag(self._derivations)

    def add_derivations(self, derivations: List[Derivation]) -> None:
        """Add derivations and rebuild the DAG."""
        self._derivations.extend(derivations)
        self._dag = construct_dependency_dag(self._derivations)

    def get_chain_depth(self, target_hash: str) -> int:
        """
        Get the maximum chain depth for a target statement.

        Args:
            target_hash: The hash of the target statement.

        Returns:
            Maximum chain depth (1 for axioms, more for derived statements).
        """
        return compute_chain_depth(target_hash, self._dag)

    def get_dependencies(self, statement_hash: str) -> List[str]:
        """
        Get direct dependencies (premises) for a statement.

        Args:
            statement_hash: Hash of the statement.

        Returns:
            List of premise hashes.
        """
        return self._dag.get(statement_hash, [])

    def get_dag(self) -> DAG:
        """Return the current dependency DAG."""
        return self._dag

    def extract_subgraph(self, target_hash: str) -> DAG:
        """
        Extract the subgraph of all dependencies for a target.

        Args:
            target_hash: Hash of the target statement.

        Returns:
            DAG containing only the target and its transitive dependencies.
        """
        subgraph: DAG = {}
        visited: Set[str] = set()

        def visit(h: str) -> None:
            if h in visited or h not in self._dag:
                return
            visited.add(h)
            subgraph[h] = self._dag.get(h, [])
            for premise in self._dag.get(h, []):
                visit(premise)

        visit(target_hash)
        return subgraph


def construct_dependency_dag(derivations: List[Derivation]) -> DAG:
    """
    Constructs a proof dependency DAG from a list of derivations.
    The DAG is represented as a dictionary where keys are conclusion hashes
    and values are lists of their premise hashes.
    """
    dag: DAG = {}
    for derivation in derivations:
        conclusion_hash = derivation.get('hash')
        premise_hashes = derivation.get('premises', [])
        if conclusion_hash:
            # Ensure even derivations with no premises are in the DAG
            if conclusion_hash not in dag:
                dag[conclusion_hash] = []
            dag[conclusion_hash].extend(premise_hashes)

    # Add premises that might not be conclusions of other derivations (axioms)
    all_premises: Set[str] = set()
    for premises in dag.values():
        all_premises.update(premises)

    for premise in all_premises:
        if premise not in dag:
            dag[premise] = [] # Axioms have no premises

    return dag


def compute_chain_depth(target_hash: str, dag: DAG) -> int:
    """
    Computes the maximum length of a dependency chain leading to target_hash.
    This is the longest path from an axiom (a node with no premises) to the target.
    Uses memoization to handle repeated computations on the same node.
    """
    memo: Dict[str, int] = {}

    def _get_depth(h: str) -> int:
        if h in memo:
            return memo[h]
        
        if h not in dag:
            # This case should ideally not be hit if the DAG is well-formed
            # and contains all premises. This indicates a missing part of the chain.
            # We treat it as an axiom of depth 1.
            return 1

        premises = dag[h]
        if not premises:
            # It's an axiom or a leaf node in the dependency graph.
            memo[h] = 1
            return 1
        
        max_premise_depth = 0
        for p_hash in premises:
            max_premise_depth = max(max_premise_depth, _get_depth(p_hash))

        depth = 1 + max_premise_depth
        memo[h] = depth
        return depth

    if target_hash not in dag:
        # If the target itself is not in the DAG, we can't determine its depth.
        # This could mean the derivation for it is missing.
        # Returning 0 or raising an error are options. Let's return 0.
        return 0

    return _get_depth(target_hash)


def extract_dependency_graph(derivation: Derivation) -> Tuple[str, List[str]]:
    """
    Utility for slice_uplift_tree.
    Extracts the direct dependencies (conclusion and premises) from a single derivation.
    """
    conclusion_hash = derivation.get('hash', '')
    premise_hashes = derivation.get('premises', [])
    return (conclusion_hash, premise_hashes)
