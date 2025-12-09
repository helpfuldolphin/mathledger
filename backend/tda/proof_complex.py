"""
Combinatorial (Flag/Clique) Complex Construction from Proof DAGs.

This module implements the mapping from proof DAGs to simplicial complexes
as specified in TDA_MIND_SCANNER_SPEC.md Section 3.1.

Given a local proof DAG G = (V, E):
1. Build undirected 1-skeleton G' = (V, E')
2. Construct flag complex K_comb where k-cliques become (k-1)-simplices
3. Compute Betti numbers β_0, β_1, ... for structural analysis

The resulting complex captures:
- 0-simplices: nodes (statements, proofs)
- 1-simplices: edges (dependencies)
- 2-simplices: triangles (tightly coupled triples)
- Higher: k-cliques for densely connected substructures

References:
    - TDA_MIND_SCANNER_SPEC.md Section 3.1
    - Wasserman (2016), Section 5 (Homology)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None  # type: ignore


@dataclass
class SimplicialComplex:
    """
    A simplicial complex built from a proof DAG.

    Stores simplices as frozensets of vertex indices, organized by dimension.
    Provides Betti number computation and structural metrics.

    Attributes:
        vertices: List of vertex identifiers (hashes or ids)
        vertex_to_index: Mapping from vertex id to integer index
        simplices_by_dim: Dict mapping dimension -> set of simplices
        max_dim: Maximum simplex dimension in the complex
        source_graph: Optional reference to the source NetworkX graph
    """

    vertices: List[str] = field(default_factory=list)
    vertex_to_index: Dict[str, int] = field(default_factory=dict)
    simplices_by_dim: Dict[int, Set[FrozenSet[int]]] = field(default_factory=dict)
    max_dim: int = 0
    source_graph: Optional[Any] = None  # nx.Graph when available

    @property
    def num_vertices(self) -> int:
        """Number of 0-simplices (vertices)."""
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        """Number of 1-simplices (edges)."""
        return len(self.simplices_by_dim.get(1, set()))

    @property
    def num_simplices(self) -> int:
        """Total number of simplices across all dimensions."""
        return sum(len(s) for s in self.simplices_by_dim.values())

    def simplices(self, dim: int) -> Set[FrozenSet[int]]:
        """Get all simplices of a given dimension."""
        return self.simplices_by_dim.get(dim, set())

    def add_simplex(self, vertices: Iterable[int]) -> None:
        """
        Add a simplex and all its faces to the complex.

        Args:
            vertices: Vertex indices forming the simplex
        """
        vertex_set = frozenset(vertices)
        dim = len(vertex_set) - 1

        if dim < 0:
            return

        if dim not in self.simplices_by_dim:
            self.simplices_by_dim[dim] = set()

        self.simplices_by_dim[dim].add(vertex_set)
        self.max_dim = max(self.max_dim, dim)

        # Add all faces (subsets)
        if dim > 0:
            for v in vertex_set:
                face = vertex_set - {v}
                self.add_simplex(face)

    def boundary_matrix(self, dim: int) -> List[List[int]]:
        """
        Compute the boundary matrix ∂_dim: C_dim -> C_{dim-1}.

        Returns a matrix where entry (i,j) is ±1 if (dim-1)-simplex i
        is a face of dim-simplex j, 0 otherwise.

        Used for homology computation.
        """
        if dim <= 0:
            return []

        high_simplices = sorted(self.simplices(dim), key=lambda s: tuple(sorted(s)))
        low_simplices = sorted(self.simplices(dim - 1), key=lambda s: tuple(sorted(s)))

        if not high_simplices or not low_simplices:
            return []

        low_index = {s: i for i, s in enumerate(low_simplices)}

        matrix: List[List[int]] = []
        for _ in low_simplices:
            matrix.append([0] * len(high_simplices))

        for j, sigma in enumerate(high_simplices):
            sorted_verts = sorted(sigma)
            for k, v in enumerate(sorted_verts):
                face = frozenset(sorted_verts[:k] + sorted_verts[k+1:])
                if face in low_index:
                    i = low_index[face]
                    sign = (-1) ** k
                    matrix[i][j] = sign

        return matrix

    def compute_betti_numbers(self, max_dim: Optional[int] = None) -> Dict[int, int]:
        """
        Compute Betti numbers β_0, β_1, ..., β_max_dim.

        Uses rank-nullity over Z_2 (mod 2 arithmetic) for simplicity.
        This is sufficient for detecting non-trivial topology.

        Args:
            max_dim: Maximum dimension to compute (default: self.max_dim)

        Returns:
            Dict mapping dimension k to Betti number β_k
        """
        if max_dim is None:
            max_dim = self.max_dim

        betti: Dict[int, int] = {}

        # β_0 = number of connected components
        # For a connected graph, β_0 = 1
        if self.source_graph is not None and HAS_NETWORKX:
            betti[0] = nx.number_connected_components(self.source_graph)
        else:
            # Fallback: compute from boundary matrix
            betti[0] = self._compute_betti_0()

        # Higher Betti numbers via boundary matrices
        for k in range(1, max_dim + 1):
            betti[k] = self._compute_betti_k(k)

        return betti

    def _compute_betti_0(self) -> int:
        """Compute β_0 (connected components) via union-find."""
        if not self.vertices:
            return 0

        parent = list(range(len(self.vertices)))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for edge in self.simplices(1):
            verts = list(edge)
            if len(verts) == 2:
                union(verts[0], verts[1])

        roots = len(set(find(i) for i in range(len(self.vertices))))
        return roots

    def _compute_betti_k(self, k: int) -> int:
        """
        Compute β_k using rank-nullity theorem over Z_2.

        β_k = dim(ker(∂_k)) - dim(im(∂_{k+1}))
            = nullity(∂_k) - rank(∂_{k+1})
        """
        # Boundary matrix ∂_k
        d_k = self.boundary_matrix(k)
        # Boundary matrix ∂_{k+1}
        d_k1 = self.boundary_matrix(k + 1)

        # Compute ranks over Z_2
        rank_dk = _matrix_rank_z2(d_k) if d_k else 0
        rank_dk1 = _matrix_rank_z2(d_k1) if d_k1 else 0

        # Number of k-simplices
        num_k = len(self.simplices(k))

        # nullity(∂_k) = num_k - rank(∂_k)
        nullity_dk = num_k - rank_dk

        # β_k = nullity(∂_k) - rank(∂_{k+1})
        betti_k = nullity_dk - rank_dk1

        return max(0, betti_k)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "num_vertices": self.num_vertices,
            "num_edges": self.num_edges,
            "num_simplices": self.num_simplices,
            "max_dim": self.max_dim,
            "simplices_count_by_dim": {
                k: len(v) for k, v in self.simplices_by_dim.items()
            },
        }


def _matrix_rank_z2(matrix: List[List[int]]) -> int:
    """
    Compute matrix rank over Z_2 (mod 2 arithmetic) via Gaussian elimination.

    Args:
        matrix: 2D list of integers (treated as mod 2)

    Returns:
        Rank of the matrix over Z_2
    """
    if not matrix or not matrix[0]:
        return 0

    # Copy matrix and reduce mod 2
    m = len(matrix)
    n = len(matrix[0])
    mat = [[matrix[i][j] % 2 for j in range(n)] for i in range(m)]

    rank = 0
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(rank, m):
            if mat[row][col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        # Swap rows
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]

        # Eliminate
        for row in range(m):
            if row != rank and mat[row][col] == 1:
                for c in range(n):
                    mat[row][c] = (mat[row][c] + mat[rank][c]) % 2

        rank += 1

    return rank


def build_combinatorial_complex(
    local_dag: Any,  # nx.DiGraph
    max_clique_size: int = 4,
) -> SimplicialComplex:
    """
    Build a flag (clique) complex from a local proof DAG.

    Per TDA_MIND_SCANNER_SPEC.md Section 3.1:
    1. Convert directed DAG to undirected 1-skeleton
    2. Find all cliques up to max_clique_size
    3. Each k-clique becomes a (k-1)-simplex

    Args:
        local_dag: NetworkX DiGraph representing local proof structure
        max_clique_size: Maximum clique size to enumerate (default 4)

    Returns:
        SimplicialComplex with flag complex structure
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for combinatorial complex construction")

    if not isinstance(local_dag, nx.DiGraph):
        raise TypeError(f"Expected nx.DiGraph, got {type(local_dag)}")

    # Step 1: Convert to undirected graph (1-skeleton)
    undirected = local_dag.to_undirected()

    # Build vertex index mapping
    vertices = sorted(undirected.nodes())
    vertex_to_index = {v: i for i, v in enumerate(vertices)}

    # Initialize complex
    complex_ = SimplicialComplex(
        vertices=vertices,
        vertex_to_index=vertex_to_index,
        simplices_by_dim={},
        max_dim=0,
        source_graph=undirected,
    )

    # Add 0-simplices (vertices)
    for v in vertices:
        complex_.add_simplex([vertex_to_index[v]])

    # Step 2: Find cliques up to max_clique_size
    # Note: find_cliques returns maximal cliques; we need all cliques
    # For small graphs, enumerate all cliques explicitly

    if len(vertices) <= 50:
        # Small graph: enumerate all cliques via BK algorithm variant
        for clique in nx.enumerate_all_cliques(undirected):
            if len(clique) > max_clique_size:
                continue
            indices = [vertex_to_index[v] for v in clique]
            complex_.add_simplex(indices)
    else:
        # Larger graph: use maximal cliques and extract sub-cliques
        for max_clique in nx.find_cliques(undirected):
            if len(max_clique) > max_clique_size:
                # Truncate to max_clique_size and add all subsets
                max_clique = max_clique[:max_clique_size]

            indices = [vertex_to_index[v] for v in max_clique]
            complex_.add_simplex(indices)

    return complex_


def extract_local_neighborhood(
    full_dag: Any,  # nx.DiGraph
    target_hash: str,
    max_depth: int = 3,
    include_descendants: bool = True,
) -> Any:  # nx.DiGraph
    """
    Extract a local subgraph around a target node.

    Used to bound the size of DAGs fed to TDA computation.

    Args:
        full_dag: The full proof DAG as nx.DiGraph
        target_hash: Hash of the target statement
        max_depth: Maximum depth to traverse (default 3)
        include_descendants: Also include descendants (default True)

    Returns:
        Subgraph as nx.DiGraph containing the local neighborhood
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for neighborhood extraction")

    if target_hash not in full_dag:
        # Return empty graph if target not found
        return nx.DiGraph()

    # Collect nodes via BFS
    nodes: Set[str] = {target_hash}

    # Ancestors (predecessors in DAG)
    frontier = {target_hash}
    for _ in range(max_depth):
        next_frontier: Set[str] = set()
        for node in frontier:
            for pred in full_dag.predecessors(node):
                if pred not in nodes:
                    nodes.add(pred)
                    next_frontier.add(pred)
        frontier = next_frontier
        if not frontier:
            break

    # Descendants (successors in DAG)
    if include_descendants:
        frontier = {target_hash}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for succ in full_dag.successors(node):
                    if succ not in nodes:
                        nodes.add(succ)
                        next_frontier.add(succ)
            frontier = next_frontier
            if not frontier:
                break

    return full_dag.subgraph(nodes).copy()


def dag_from_proof_dag(proof_dag: Any) -> Any:
    """
    Convert a ProofDag object to a NetworkX DiGraph.

    Args:
        proof_dag: ProofDag instance from backend.dag.proof_dag

    Returns:
        nx.DiGraph with edges from parent_hash to child_hash
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required")

    G = nx.DiGraph()

    for edge in proof_dag.edges:
        parent = edge.parent_hash
        child = edge.child_hash
        if parent is not None and child is not None:
            G.add_edge(parent, child, proof_id=edge.proof_id)

    return G
