"""
Unit tests for causal graph data structures.

Tests graph construction, validation, and topological ordering.
"""

import pytest
from backend.causal.graph import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    VariableType,
    build_rfl_graph,
    validate_graph
)


def test_node_creation():
    """Test creating causal nodes."""
    node = CausalNode(
        name="test_var",
        var_type=VariableType.POLICY,
        description="Test variable"
    )

    assert node.name == "test_var"
    assert node.var_type == VariableType.POLICY
    assert node.description == "Test variable"
    assert node.observed_values == []


def test_edge_creation():
    """Test creating causal edges."""
    source = CausalNode("X", VariableType.POLICY)
    target = CausalNode("Y", VariableType.THROUGHPUT)

    edge = CausalEdge(
        source=source,
        target=target,
        coefficient=0.5,
        mechanism="X causes Y"
    )

    assert edge.source == source
    assert edge.target == target
    assert edge.coefficient == 0.5
    assert edge.mechanism == "X causes Y"


def test_graph_add_node():
    """Test adding nodes to graph."""
    graph = CausalGraph()
    node = CausalNode("X", VariableType.POLICY)

    graph.add_node(node)

    assert "X" in graph.nodes
    assert graph.nodes["X"] == node


def test_graph_add_edge():
    """Test adding edges to graph."""
    graph = CausalGraph()

    X = CausalNode("X", VariableType.POLICY)
    Y = CausalNode("Y", VariableType.THROUGHPUT)

    edge = CausalEdge(source=X, target=Y)
    graph.add_edge(edge)

    assert edge in graph.edges
    assert "Y" in graph._adjacency["X"]
    assert "X" in graph._reverse_adj["Y"]


def test_graph_cycle_detection():
    """Test that cycles are detected and prevented."""
    graph = CausalGraph()

    X = CausalNode("X", VariableType.POLICY)
    Y = CausalNode("Y", VariableType.ABSTENTION)
    Z = CausalNode("Z", VariableType.THROUGHPUT)

    # Add X -> Y -> Z
    graph.add_edge(CausalEdge(source=X, target=Y))
    graph.add_edge(CausalEdge(source=Y, target=Z))

    # Try to add Z -> X (would create cycle)
    with pytest.raises(ValueError, match="cycle"):
        graph.add_edge(CausalEdge(source=Z, target=X))


def test_get_parents():
    """Test retrieving parent nodes."""
    graph = CausalGraph()

    X = CausalNode("X", VariableType.POLICY)
    Y = CausalNode("Y", VariableType.ABSTENTION)
    Z = CausalNode("Z", VariableType.THROUGHPUT)

    graph.add_edge(CausalEdge(source=X, target=Z))
    graph.add_edge(CausalEdge(source=Y, target=Z))

    parents = graph.get_parents("Z")
    parent_names = {p.name for p in parents}

    assert len(parents) == 2
    assert parent_names == {"X", "Y"}


def test_get_children():
    """Test retrieving child nodes."""
    graph = CausalGraph()

    X = CausalNode("X", VariableType.POLICY)
    Y = CausalNode("Y", VariableType.ABSTENTION)
    Z = CausalNode("Z", VariableType.THROUGHPUT)

    graph.add_edge(CausalEdge(source=X, target=Y))
    graph.add_edge(CausalEdge(source=X, target=Z))

    children = graph.get_children("X")
    child_names = {c.name for c in children}

    assert len(children) == 2
    assert child_names == {"Y", "Z"}


def test_get_ancestors():
    """Test retrieving all ancestors (transitive closure)."""
    graph = CausalGraph()

    A = CausalNode("A", VariableType.POLICY)
    B = CausalNode("B", VariableType.ABSTENTION)
    C = CausalNode("C", VariableType.VERIFICATION_TIME)
    D = CausalNode("D", VariableType.THROUGHPUT)

    # Chain: A -> B -> C -> D
    graph.add_edge(CausalEdge(source=A, target=B))
    graph.add_edge(CausalEdge(source=B, target=C))
    graph.add_edge(CausalEdge(source=C, target=D))

    ancestors = graph.get_ancestors("D")

    assert ancestors == {"A", "B", "C"}


def test_get_descendants():
    """Test retrieving all descendants."""
    graph = CausalGraph()

    A = CausalNode("A", VariableType.POLICY)
    B = CausalNode("B", VariableType.ABSTENTION)
    C = CausalNode("C", VariableType.VERIFICATION_TIME)
    D = CausalNode("D", VariableType.THROUGHPUT)

    # Chain: A -> B -> C -> D
    graph.add_edge(CausalEdge(source=A, target=B))
    graph.add_edge(CausalEdge(source=B, target=C))
    graph.add_edge(CausalEdge(source=C, target=D))

    descendants = graph.get_descendants("A")

    assert descendants == {"B", "C", "D"}


def test_topological_sort():
    """Test topological ordering of nodes."""
    graph = CausalGraph()

    X = CausalNode("X", VariableType.POLICY)
    Y = CausalNode("Y", VariableType.ABSTENTION)
    Z = CausalNode("Z", VariableType.THROUGHPUT)

    # X -> Y -> Z
    graph.add_edge(CausalEdge(source=X, target=Y))
    graph.add_edge(CausalEdge(source=Y, target=Z))

    order = graph.topological_sort()

    # X should come before Y, Y before Z
    assert order.index("X") < order.index("Y")
    assert order.index("Y") < order.index("Z")


def test_build_rfl_graph():
    """Test building the RFL causal graph."""
    graph = build_rfl_graph()

    # Check expected nodes
    assert "policy_hash" in graph.nodes
    assert "abstain_pct" in graph.nodes
    assert "proofs_per_sec" in graph.nodes
    assert "verify_ms_p50" in graph.nodes
    assert "depth_max" in graph.nodes

    # Check that it's acyclic
    order = graph.topological_sort()
    assert len(order) == len(graph.nodes)

    # Check key edges exist
    policy_to_abstain = graph.get_edge("policy_hash", "abstain_pct")
    assert policy_to_abstain is not None

    abstain_to_throughput = graph.get_edge("abstain_pct", "proofs_per_sec")
    assert abstain_to_throughput is not None


def test_validate_graph():
    """Test graph validation."""
    graph = build_rfl_graph()

    warnings = validate_graph(graph)

    # Should have no structural warnings
    assert "cycle" not in " ".join(warnings).lower()


def test_graph_serialization():
    """Test serializing graph to dict."""
    graph = CausalGraph()

    X = CausalNode("X", VariableType.POLICY)
    Y = CausalNode("Y", VariableType.THROUGHPUT)

    graph.add_edge(CausalEdge(source=X, target=Y, coefficient=0.5))

    data = graph.to_dict()

    assert "nodes" in data
    assert "edges" in data
    assert "X" in data["nodes"]
    assert "Y" in data["nodes"]
    assert len(data["edges"]) == 1
    assert data["edges"][0]["source"] == "X"
    assert data["edges"][0]["target"] == "Y"
    assert data["edges"][0]["coefficient"] == 0.5


def test_graph_hash_determinism():
    """Test that graph hash is deterministic."""
    graph1 = build_rfl_graph()
    graph2 = build_rfl_graph()

    # Same structure should have same hash
    assert graph1.hash() == graph2.hash()


def test_get_edge():
    """Test retrieving specific edge."""
    graph = CausalGraph()

    X = CausalNode("X", VariableType.POLICY)
    Y = CausalNode("Y", VariableType.THROUGHPUT)

    edge = CausalEdge(source=X, target=Y, coefficient=0.42)
    graph.add_edge(edge)

    retrieved = graph.get_edge("X", "Y")

    assert retrieved is not None
    assert retrieved.coefficient == 0.42

    # Non-existent edge
    assert graph.get_edge("Y", "X") is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
