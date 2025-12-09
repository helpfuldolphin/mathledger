# backend/dag/global_dag_builder.py
"""
PHASE II - Global DAG Builder implementation.

This module provides the `GlobalDagBuilder` class, which is responsible for
constructing a cumulative, global Proof Dependency DAG from a sequence of
per-cycle derivation logs. It enforces consistency invariants and computes
evolution metrics.
"""
import json
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add project root for local imports
import sys
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.schema import CyclicDependencyError, Derivation, DanglingPremiseError

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading Global DAG Builder.", file=sys.stderr)


class DanglingPremiseWarning(UserWarning):
    """Warning for a premise that is not a known axiom or conclusion."""
    pass


class GlobalDagBuilder:
    """
    Builds, validates, and analyzes a global proof dependency DAG.
    """

    def __init__(self, axiom_registry: Set[str] = None, strict: bool = False):
        """
        Initializes the builder.

        Args:
            axiom_registry: A set of hashes for known axioms.
            strict: If True, raises DanglingPremiseError instead of a warning.
        """
        self._global_dag: Dict[str, Set[Tuple[str, ...]]] = {}
        self._nodes: Set[str] = set()
        self.evolution_metrics: List[Dict[str, Any]] = []
        self._depth_memo: Dict[str, int] = {}
        
        self._axioms: Set[str] = axiom_registry if axiom_registry is not None else set()
        self._strict_mode = strict
        self._nodes.update(self._axioms)

    def _is_ancestor(self, start_node: str, target_ancestor: str) -> Tuple[bool, Tuple[str, ...]]:
        """
        Checks if `target_ancestor` is an ancestor of `start_node`.
        Uses BFS to traverse upwards from `start_node` via premises.
        Returns (is_ancestor, path_from_start_node_to_ancestor)
        """
        if start_node not in self._global_dag:
            return False, tuple()
        
        q = deque([(start_node, (start_node,))])
        visited = {start_node}

        while q:
            current_node, path = q.popleft()

            if current_node == target_ancestor:
                return True, path

            if current_node not in self._global_dag:
                continue

            for proof in self._global_dag[current_node]:
                for premise in proof:
                    if premise not in visited:
                        visited.add(premise)
                        new_path = path + (premise,)
                        q.append((premise, new_path))
        return False, tuple()

    def _validate_and_add_derivation(self, derivation: Derivation) -> bool:
        """
        Validates a single derivation and adds it to the DAG.
        Returns True if the derivation was new, False if it was a duplicate.
        """
        conclusion = derivation.conclusion
        premises = derivation.premises

        # Invariant 2: Parent Existence Check
        for p in premises:
            if p not in self._nodes:
                message = f"Premise '{p}' for conclusion '{conclusion}' is not a known conclusion or axiom."
                if self._strict_mode:
                    raise DanglingPremiseError(message, conclusion=conclusion, premise=p)
                else:
                    warnings.warn(message, DanglingPremiseWarning)

        # Invariant 1: Acyclicity Check
        if conclusion in premises:
             raise CyclicDependencyError(f"Self-dependency detected for node {conclusion}", (conclusion, conclusion))

        for p in premises:
            is_ancestor, path_from_p = self._is_ancestor(p, conclusion)
            if is_ancestor:
                full_cycle_path = path_from_p + (p,)
                raise CyclicDependencyError(
                    f"Adding derivation ({conclusion} <- {premises}) would create a cycle.",
                    full_cycle_path
                )

        # Add nodes to master set
        self._nodes.add(conclusion)
        self._nodes.update(premises)

        # Add derivation to DAG
        is_new_derivation = False
        if conclusion not in self._global_dag:
            self._global_dag[conclusion] = set()
        
        if premises not in self._global_dag[conclusion]:
            self._global_dag[conclusion].add(premises)
            is_new_derivation = True

        return is_new_derivation

    def _compute_depths(self):
        """
        Computes depth for all nodes in the DAG using an iterative method.
        """
        self._depth_memo.clear()
        for node in self._nodes:
            if node not in self._depth_memo:
                self._get_depth_iterative(node)
                
    def _get_depth_iterative(self, start_node: str):
        """Iterative (non-recursive) depth calculation."""
        stack = [(start_node, False)] # node, children_visited
        
        while stack:
            node, children_visited = stack[-1]

            if node in self._depth_memo:
                stack.pop()
                continue
            
            if node not in self._global_dag or not self._global_dag[node]:
                self._depth_memo[node] = 1
                stack.pop()
                continue
            
            if children_visited:
                max_premise_depth = 0
                for proof in self._global_dag[node]:
                    if not proof: continue
                    max_premise_depth = max(max_premise_depth, max(self._depth_memo.get(p, 1) for p in proof))
                self._depth_memo[node] = 1 + max_premise_depth
                stack.pop()
            else:
                stack[-1] = (node, True)
                for proof in self._global_dag[node]:
                    for p in proof:
                        if p not in self._depth_memo:
                            stack.append((p, False))

    def add_cycle_derivations(self, cycle_index: int, cycle_derivations: List[Dict[str, Any]]):
        """
        Adds a new cycle's derivations to the global DAG and computes evolution metrics.
        """
        prev_nodes = len(self._nodes)
        prev_edges = sum(len(proofs) for proofs in self._global_dag.values())
        prev_max_depth = max(self._depth_memo.values()) if self._depth_memo else 0

        new_derivations_count = 0
        canonical_derivations = {Derivation(d['conclusion'], tuple(sorted(d['premises']))) for d in cycle_derivations}

        for d in canonical_derivations:
            if self._validate_and_add_derivation(d):
                new_derivations_count += 1
        
        self._compute_depths()
        
        total_nodes = len(self._nodes)
        total_edges = sum(len(proofs) for proofs in self._global_dag.values())
        max_depth = max(self._depth_memo.values()) if self._depth_memo else 0
        
        metrics = {
            "cycle": cycle_index,
            "Nodes(t)": total_nodes, "Edges(t)": total_edges,
            "ΔNodes(t)": total_nodes - prev_nodes, "ΔEdges(t)": total_edges - prev_edges,
            "MaxDepth(t)": max_depth, "ΔMaxDepth(t)": max_depth - prev_max_depth,
            "GlobalBranchingFactor(t)": total_edges / total_nodes if total_nodes > 0 else 0,
            "total_derivations_in_cycle": len(canonical_derivations),
            "new_derivations_in_cycle": new_derivations_count,
            "duplicate_derivations_in_cycle": len(canonical_derivations) - new_derivations_count,
        }
        self.evolution_metrics.append(metrics)

    def save_metrics(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.evolution_metrics, f, indent=2)
            
    def save_global_dag_structure(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable_dag = {
            conclusion: [list(premise_tuple) for premise_tuple in proofs]
            for conclusion, proofs in self._global_dag.items()
        }
        with open(path, "w") as f:
            json.dump(serializable_dag, f, indent=2)
