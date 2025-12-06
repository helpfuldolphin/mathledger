"""
Unit tests for the ProofDag in-memory graph structure.

These cases underpin the organism lineage coverage exercised by
tests/integration/test_first_organism_dag.py.

Schema Assumptions (proof_parents):
    The DAG tests rely on proof_parents exposing at minimum:
    - proof_id UUID (nullable if schema only stores hashes)
    - child_statement_id UUID
    - child_hash TEXT
    - parent_statement_id UUID
    - parent_hash TEXT
    - edge_index INT DEFAULT 0
    - created_at TIMESTAMPTZ
    If any of these columns are missing, ProofDagRepository will emit
    *_unverified sentinel issues and the integration tests will fail.
"""

from backend.dag.proof_dag import ProofDag, ProofEdge


def _edge(
    proof_id=None,
    child_id=None,
    child_hash=None,
    parent_id=None,
    parent_hash=None,
    edge_index=0,
):
    return ProofEdge(
        proof_id=proof_id,
        child_statement_id=child_id,
        child_hash=child_hash,
        parent_statement_id=parent_id,
        parent_hash=parent_hash,
        edge_index=edge_index,
    )


def test_proof_dag_lineage_queries():
    edges = [
        _edge(proof_id=1, child_id=2, child_hash="h2", parent_id=1, parent_hash="h1"),
        _edge(proof_id=2, child_id=3, child_hash="h3", parent_id=2, parent_hash="h2"),
        _edge(proof_id=3, child_id=4, child_hash="h4", parent_id=2, parent_hash="h2"),
    ]
    dag = ProofDag(edges)

    assert {edge.parent_statement_id for edge in dag.parents_of(3)} == {2}
    assert {edge.child_statement_id for edge in dag.children_of(2)} == {3, 4}

    assert dag.ancestors("h3") == ["h2", "h1"]
    assert dag.descendants("h2") == ["h3", "h4"]


def test_proof_dag_detects_cycles_without_ids():
    edges = [
        _edge(child_hash="h1", parent_hash="h2"),
        _edge(child_hash="h2", parent_hash="h1"),
    ]
    report = ProofDag(edges).validate()
    assert "cycle_nodes" in report.issues
    assert sorted(report.issues["cycle_nodes"]) == ["hash:h1", "hash:h2"]


def test_proof_dag_duplicate_edge_detection():
    edges = [
        _edge(proof_id=1, child_id=2, child_hash="h2", parent_id=1, parent_hash="h1"),
        _edge(proof_id=1, child_id=2, child_hash="h2", parent_id=1, parent_hash="h1"),
        _edge(proof_id=1, child_id=2, child_hash="h2", parent_id=1, parent_hash="h1", edge_index=1),
        _edge(proof_id=1, child_id=3, child_hash="h3", parent_id=1, parent_hash="h1"),
    ]
    report = ProofDag(edges).validate()
    duplicates = report.issues.get("duplicate_edges")
    assert duplicates, "Expected duplicate edge detection"
    first = duplicates[0]
    assert first["proof_id"] == 1
    assert first["child"] == "id:2"
    assert first["parent"] == "id:1"
    assert first["count"] == 3


def test_proof_dag_detects_longer_cycles():
    """Cycle detection for chains longer than 2 nodes."""
    edges = [
        _edge(child_hash="h1", parent_hash="h2"),
        _edge(child_hash="h2", parent_hash="h3"),
        _edge(child_hash="h3", parent_hash="h4"),
        _edge(child_hash="h4", parent_hash="h1"),
    ]
    report = ProofDag(edges).validate()
    assert "cycle_nodes" in report.issues
    assert sorted(report.issues["cycle_nodes"]) == [
        "hash:h1",
        "hash:h2",
        "hash:h3",
        "hash:h4",
    ]


def test_proof_dag_detects_self_loop():
    """Self-referential edge should be flagged."""
    edges = [
        _edge(child_id=1, child_hash="h1", parent_id=1, parent_hash="h1"),
    ]
    report = ProofDag(edges).validate()
    assert "self_loops" in report.issues
    assert len(report.issues["self_loops"]) == 1


def test_proof_dag_no_cycles_in_valid_chain():
    """Linear chain should pass validation."""
    edges = [
        _edge(child_hash="h2", parent_hash="h1"),
        _edge(child_hash="h3", parent_hash="h2"),
        _edge(child_hash="h4", parent_hash="h3"),
    ]
    report = ProofDag(edges).validate()
    assert report.ok
    assert "cycle_nodes" not in report.issues


def test_proof_dag_diamond_structure():
    """Diamond DAG (common ancestor) should pass validation."""
    edges = [
        _edge(child_hash="h2", parent_hash="h1"),
        _edge(child_hash="h3", parent_hash="h1"),
        _edge(child_hash="h4", parent_hash="h2"),
        _edge(child_hash="h4", parent_hash="h3"),
    ]
    dag = ProofDag(edges)
    report = dag.validate()
    assert report.ok
    assert "cycle_nodes" not in report.issues
    # h4 has two parents
    assert len(dag.parents_of_hash("h4")) == 2
    # h1 has two children
    assert len(dag.children_of_hash("h1")) == 2
    # ancestors of h4 include h1, h2, h3
    assert set(dag.ancestors("h4")) == {"h1", "h2", "h3"}

