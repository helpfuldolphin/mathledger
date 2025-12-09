"""
Tests for backend/tda/proof_complex.py.

Tests combinatorial (flag/clique) complex construction from proof DAGs.

Per TDA_MIND_SCANNER_SPEC.md Section 3.1:
- Conversion from DAG to undirected 1-skeleton
- Clique enumeration and simplex construction
- Betti number computation (β_0, β_1)

Determinism requirement: same inputs must produce identical outputs.
"""

from __future__ import annotations

import pytest

# Skip all tests if networkx not available
networkx = pytest.importorskip("networkx")
import networkx as nx


class TestSimplicialComplex:
    """Tests for SimplicialComplex dataclass."""

    def test_empty_complex(self) -> None:
        """Empty complex has zero vertices and simplices."""
        from backend.tda.proof_complex import SimplicialComplex

        complex_ = SimplicialComplex()
        assert complex_.num_vertices == 0
        assert complex_.num_edges == 0
        assert complex_.num_simplices == 0
        assert complex_.max_dim == 0

    def test_add_simplex_single_vertex(self) -> None:
        """Adding a 0-simplex (vertex) works correctly."""
        from backend.tda.proof_complex import SimplicialComplex

        complex_ = SimplicialComplex(vertices=["a"], vertex_to_index={"a": 0})
        complex_.add_simplex([0])

        assert complex_.num_simplices >= 1
        assert 0 in complex_.simplices_by_dim

    def test_add_simplex_edge(self) -> None:
        """Adding a 1-simplex (edge) also adds its faces."""
        from backend.tda.proof_complex import SimplicialComplex

        complex_ = SimplicialComplex(
            vertices=["a", "b"],
            vertex_to_index={"a": 0, "b": 1},
        )
        complex_.add_simplex([0, 1])

        assert complex_.num_edges == 1
        assert len(complex_.simplices(0)) == 2  # Two vertices
        assert len(complex_.simplices(1)) == 1  # One edge

    def test_add_simplex_triangle(self) -> None:
        """Adding a 2-simplex (triangle) adds all faces."""
        from backend.tda.proof_complex import SimplicialComplex

        complex_ = SimplicialComplex(
            vertices=["a", "b", "c"],
            vertex_to_index={"a": 0, "b": 1, "c": 2},
        )
        complex_.add_simplex([0, 1, 2])

        assert len(complex_.simplices(0)) == 3  # Three vertices
        assert len(complex_.simplices(1)) == 3  # Three edges
        assert len(complex_.simplices(2)) == 1  # One triangle
        assert complex_.max_dim == 2

    def test_to_dict_serialization(self) -> None:
        """Complex serializes to dictionary correctly."""
        from backend.tda.proof_complex import SimplicialComplex

        complex_ = SimplicialComplex(vertices=["a", "b"], vertex_to_index={"a": 0, "b": 1})
        complex_.add_simplex([0, 1])

        d = complex_.to_dict()
        assert "num_vertices" in d
        assert "num_edges" in d
        assert "num_simplices" in d
        assert d["num_vertices"] == 2


class TestBettiNumbers:
    """Tests for Betti number computation."""

    def test_betti_single_component(self) -> None:
        """Connected graph has β_0 = 1."""
        from backend.tda.proof_complex import SimplicialComplex

        # Line graph: a - b - c
        complex_ = SimplicialComplex(
            vertices=["a", "b", "c"],
            vertex_to_index={"a": 0, "b": 1, "c": 2},
        )
        complex_.add_simplex([0, 1])
        complex_.add_simplex([1, 2])

        betti = complex_.compute_betti_numbers(max_dim=1)
        assert betti[0] == 1  # One connected component

    def test_betti_two_components(self) -> None:
        """Disconnected graph has β_0 = 2."""
        from backend.tda.proof_complex import SimplicialComplex

        # Two disconnected edges: a-b and c-d
        complex_ = SimplicialComplex(
            vertices=["a", "b", "c", "d"],
            vertex_to_index={"a": 0, "b": 1, "c": 2, "d": 3},
        )
        complex_.add_simplex([0, 1])
        complex_.add_simplex([2, 3])

        betti = complex_.compute_betti_numbers(max_dim=1)
        assert betti[0] == 2  # Two connected components

    def test_betti_triangle_no_cycle(self) -> None:
        """Filled triangle has β_1 = 0 (no holes)."""
        from backend.tda.proof_complex import SimplicialComplex

        # Filled triangle
        complex_ = SimplicialComplex(
            vertices=["a", "b", "c"],
            vertex_to_index={"a": 0, "b": 1, "c": 2},
        )
        complex_.add_simplex([0, 1, 2])  # Adds triangle and all faces

        betti = complex_.compute_betti_numbers(max_dim=1)
        assert betti[0] == 1  # Connected
        assert betti[1] == 0  # No hole (filled)

    def test_betti_cycle_has_hole(self) -> None:
        """Unfilled cycle has β_1 = 1 (one hole)."""
        from backend.tda.proof_complex import SimplicialComplex

        # Triangle boundary only (no fill)
        complex_ = SimplicialComplex(
            vertices=["a", "b", "c"],
            vertex_to_index={"a": 0, "b": 1, "c": 2},
        )
        complex_.add_simplex([0, 1])
        complex_.add_simplex([1, 2])
        complex_.add_simplex([0, 2])
        # Note: NOT adding [0,1,2] so no 2-simplex

        betti = complex_.compute_betti_numbers(max_dim=1)
        assert betti[0] == 1  # Connected
        assert betti[1] == 1  # One hole


class TestBuildCombinatorialComplex:
    """Tests for build_combinatorial_complex function."""

    def test_simple_dag(self) -> None:
        """Build complex from simple DAG."""
        from backend.tda.proof_complex import build_combinatorial_complex

        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("b", "c")

        complex_ = build_combinatorial_complex(G)

        assert complex_.num_vertices == 3
        assert complex_.num_edges == 2

    def test_dag_with_triangle(self) -> None:
        """DAG with triangle structure creates 2-simplices."""
        from backend.tda.proof_complex import build_combinatorial_complex

        G = nx.DiGraph()
        G.add_edge("a", "b")
        G.add_edge("a", "c")
        G.add_edge("b", "c")

        complex_ = build_combinatorial_complex(G)

        assert complex_.num_vertices == 3
        # Should have edges and a triangle
        assert len(complex_.simplices(2)) == 1

    def test_empty_dag(self) -> None:
        """Empty DAG produces empty complex."""
        from backend.tda.proof_complex import build_combinatorial_complex

        G = nx.DiGraph()
        complex_ = build_combinatorial_complex(G)

        assert complex_.num_vertices == 0

    def test_max_clique_size_respected(self) -> None:
        """max_clique_size parameter limits simplex dimension."""
        from backend.tda.proof_complex import build_combinatorial_complex

        # Create a 4-clique
        G = nx.DiGraph()
        nodes = ["a", "b", "c", "d"]
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                G.add_edge(n1, n2)

        # With max_clique_size=3, should not have 3-simplices
        complex_ = build_combinatorial_complex(G, max_clique_size=3)
        assert complex_.max_dim <= 2

    def test_determinism(self) -> None:
        """Same input produces identical output."""
        from backend.tda.proof_complex import build_combinatorial_complex

        G = nx.DiGraph()
        G.add_edge("x", "y")
        G.add_edge("y", "z")
        G.add_edge("x", "z")

        c1 = build_combinatorial_complex(G)
        c2 = build_combinatorial_complex(G)

        assert c1.num_vertices == c2.num_vertices
        assert c1.num_edges == c2.num_edges
        assert c1.num_simplices == c2.num_simplices


class TestExtractLocalNeighborhood:
    """Tests for extract_local_neighborhood function."""

    def test_extract_neighborhood(self) -> None:
        """Extract bounded neighborhood from larger DAG."""
        from backend.tda.proof_complex import extract_local_neighborhood

        G = nx.DiGraph()
        # Chain: a -> b -> c -> d -> e
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        G.add_edge("c", "d")
        G.add_edge("d", "e")

        # Depth 1 from c
        subgraph = extract_local_neighborhood(G, "c", max_depth=1)

        assert "c" in subgraph.nodes()
        assert "b" in subgraph.nodes()  # ancestor
        assert "d" in subgraph.nodes()  # descendant
        # a and e should be excluded (depth > 1)

    def test_missing_target(self) -> None:
        """Missing target returns empty graph."""
        from backend.tda.proof_complex import extract_local_neighborhood

        G = nx.DiGraph()
        G.add_edge("a", "b")

        subgraph = extract_local_neighborhood(G, "missing", max_depth=2)
        assert len(subgraph.nodes()) == 0


class TestDagFromProofDag:
    """Tests for dag_from_proof_dag conversion."""

    def test_conversion(self) -> None:
        """Convert ProofDag to NetworkX DiGraph."""
        from backend.tda.proof_complex import dag_from_proof_dag
        from backend.dag.proof_dag import ProofDag, ProofEdge

        edges = [
            ProofEdge(
                proof_id=1,
                child_statement_id=2,
                child_hash="child_hash",
                parent_statement_id=1,
                parent_hash="parent_hash",
            ),
        ]
        proof_dag = ProofDag(edges)

        G = dag_from_proof_dag(proof_dag)

        assert "parent_hash" in G.nodes()
        assert "child_hash" in G.nodes()
        assert G.has_edge("parent_hash", "child_hash")
