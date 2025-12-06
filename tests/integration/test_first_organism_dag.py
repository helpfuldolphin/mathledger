"""
Integration coverage for the First Organism proof lineage DAG.

This test ensures the actual proof_parents graph that backs the expedition
still respects the non-pathological invariants we enforce in ProofDag.

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
    *_unverified sentinel issues and the test will fail.

DAG Invariants Enforced:
    1. No cycles: Derivation chains cannot loop back to earlier statements
    2. No self-loops: A statement cannot be its own parent
    3. No duplicate edges: Each (proof_id, child, parent) triple is unique
    4. Hash/ID consistency: Statement IDs map consistently to hashes
    5. Complete edges: All edges have resolvable child and parent identifiers
    6. Edge index ordering: Indices are sequential within each proof
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple, Set

import pytest
from backend.crypto.core import hash_statement, sha256_hex
from backend.axiom_engine.derive_utils import record_proof_edge
from backend.dag.proof_dag import ProofDag, ProofDagRepository, ProofEdge
from tests.helpers.dag_assertions import (
    assert_no_cycles,
    assert_no_self_loops,
    assert_no_duplicate_edges,
    assert_hash_id_consistency,
    assert_complete_edges,
    assert_edge_index_ordering,
    assert_lineage_complete,
    assert_ancestor_chain,
    assert_descendant_chain,
    validate_organism_lineage,
    validate_organism_lineage_from_db,
    pytest_assert_dag_valid,
    pytest_assert_organism_lineage,
    OrganismLineageReport,
)


def _hex64(value: str) -> str:
    """Return canonical SHA-256 hex for the given text."""
    return hash_statement(value)


def _insert_statement(cur, system_id: str, text: str) -> Tuple[str, str]:
    """Insert a statement and return (id, hash)."""
    stmt_hash = _hex64(text)
    cur.execute(
        """
        INSERT INTO statements (system_id, hash, content_norm, normalized_text, status)
        VALUES (%s, %s, %s, %s, 'unknown')
        RETURNING id
        """,
        (system_id, stmt_hash, text, text),
    )
    stmt_id = cur.fetchone()[0]
    return stmt_id, stmt_hash


def _insert_proof(cur, statement_id: str, system_id: str) -> str:
    """Insert a proof and return its id."""
    proof_payload = f"{statement_id}:{system_id}"
    proof_hash = sha256_hex(proof_payload)
    cur.execute(
        """
        INSERT INTO proofs (statement_id, system_id, prover, status, proof_hash)
        VALUES (%s, %s, 'lean', 'success', %s)
        RETURNING id
        """,
        (statement_id, system_id, proof_hash),
    )
    return cur.fetchone()[0]


def _ensure_proof_parents_table(cur) -> None:
    """Create proof_parents table with full DAG schema."""
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS proof_parents (
            proof_id UUID,
            child_statement_id UUID,
            child_hash TEXT,
            parent_statement_id UUID,
            parent_hash TEXT,
            edge_index INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def _create_theory(cur, name: str = "First Organism Theory") -> str:
    """Create a theory and return its id."""
    cur.execute(
        """
        INSERT INTO theories (name, slug, version, logic)
        VALUES (%s, %s, 'v1', 'classical')
        RETURNING id
        """,
        (name, name.lower().replace(" ", "-")),
    )
    return cur.fetchone()[0]


# =============================================================================
# Basic DAG Integrity Tests
# =============================================================================

@pytest.mark.first_organism
@pytest.mark.integration
def test_first_organism_dag_integrity(first_organism_db) -> None:
    """
    Verifies that the First Organism proof chain is fully captured by the DAG.

    This is the canonical test for First Organism lineage:
    - Parents: "p", "q" (atomic propositions)
    - Child: "((p /\\ q) -> p)" (derived statement)

    Asserts all DAG invariants hold.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur)

        # Parent statements for the organism path
        parent_defs = ("p", "q")
        parent_ids: List[str] = []
        parent_hashes: List[str] = []
        for text in parent_defs:
            stmt_id, stmt_hash = _insert_statement(cur, system_id, text)
            parent_ids.append(stmt_id)
            parent_hashes.append(stmt_hash)

        # Child statement (the organism proof target)
        child_text = "((p /\\ q) -> p)"
        child_id, child_hash = _insert_statement(cur, system_id, child_text)

        # Insert proof referencing the child
        proof_id = _insert_proof(cur, child_id, system_id)

        # Record edges from parents to child
        for idx, (parent_id, parent_hash) in enumerate(zip(parent_ids, parent_hashes)):
            record_proof_edge(
                cur,
                proof_id=proof_id,
                child_statement_id=child_id,
                child_hash=child_hash,
                parent_statement_id=parent_id,
                parent_hash=parent_hash,
                edge_index=idx,
            )

        # Validate the DAG using repository
        repo = ProofDagRepository(cur)
        report = repo.validate()
        assert report.ok, f"DAG validation failed: {report.issues}"
        assert "cycle_nodes" not in report.issues
        assert "duplicate_edges" not in report.issues
        assert "duplicate_edges_db" not in report.issues

        # Ensure no *_unverified sentinel issues exist
        unverified_keys = [k for k in report.issues if k.endswith("_unverified")]
        assert not unverified_keys, f"Schema incomplete: {unverified_keys}"

        # Load and verify DAG structure
        dag = repo.load_dag()
        organism_edges = [
            edge for edge in dag.edges if edge.child_statement_id == child_id
        ]
        assert len(organism_edges) == len(parent_defs)

        seen_hashes = {edge.parent_hash for edge in organism_edges}
        assert seen_hashes == set(parent_hashes)

        ancestors = dag.ancestors(child_hash)
        assert set(ancestors) >= set(parent_hashes)

        descendants = dag.descendants(parent_hashes[0])
        assert child_hash in descendants


@pytest.mark.first_organism
@pytest.mark.integration
def test_first_organism_dag_with_assertion_helpers(first_organism_db) -> None:
    """
    Comprehensive DAG validation using the dag_assertions helper module.

    This test demonstrates the full validation pipeline using all
    assertion helpers to verify First Organism lineage integrity.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Assertion Helper Test Theory")

        # Create parent statements
        p_id, p_hash = _insert_statement(cur, system_id, "p")
        q_id, q_hash = _insert_statement(cur, system_id, "q")
        parent_hashes = {p_hash, q_hash}

        # Create organism (derived statement)
        organism_text = "((p /\\ q) -> p)"
        organism_id, organism_hash = _insert_statement(cur, system_id, organism_text)

        # Create proof and edges
        proof_id = _insert_proof(cur, organism_id, system_id)
        record_proof_edge(
            cur,
            proof_id=proof_id,
            child_statement_id=organism_id,
            child_hash=organism_hash,
            parent_statement_id=p_id,
            parent_hash=p_hash,
            edge_index=0,
        )
        record_proof_edge(
            cur,
            proof_id=proof_id,
            child_statement_id=organism_id,
            child_hash=organism_hash,
            parent_statement_id=q_id,
            parent_hash=q_hash,
            edge_index=1,
        )

        # Use validate_organism_lineage_from_db for comprehensive validation
        report = validate_organism_lineage_from_db(
            cur, organism_hash, parent_hashes, proof_id=proof_id
        )

        assert report.ok, f"Organism lineage validation failed: {report.summary}"

        # Verify all individual invariants passed
        for name, inv in report.invariants.items():
            assert inv.passed, f"Invariant '{name}' failed: {inv.message}"

        # Verify metrics
        assert report.metrics["total_edges"] == 2
        assert report.metrics["organism_parent_count"] == 2
        assert report.metrics["failed_invariants"] == 0


# =============================================================================
# Multi-Level Derivation Chain Tests
# =============================================================================

@pytest.mark.first_organism
@pytest.mark.integration
def test_multi_level_derivation_chain(first_organism_db) -> None:
    """
    Test DAG integrity for multi-level derivation chains.

    Creates a chain: axiom1, axiom2 -> derived1 -> derived2
    Verifies that transitive ancestor/descendant queries work correctly.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Multi-Level Chain Theory")

        # Level 0: Axioms
        axiom1_id, axiom1_hash = _insert_statement(cur, system_id, "A1")
        axiom2_id, axiom2_hash = _insert_statement(cur, system_id, "A2")

        # Level 1: First derivation (from axiom1 + axiom2)
        derived1_text = "(A1 -> A2)"
        derived1_id, derived1_hash = _insert_statement(cur, system_id, derived1_text)
        proof1_id = _insert_proof(cur, derived1_id, system_id)
        record_proof_edge(
            cur, proof_id=proof1_id, child_statement_id=derived1_id,
            child_hash=derived1_hash, parent_statement_id=axiom1_id,
            parent_hash=axiom1_hash, edge_index=0,
        )
        record_proof_edge(
            cur, proof_id=proof1_id, child_statement_id=derived1_id,
            child_hash=derived1_hash, parent_statement_id=axiom2_id,
            parent_hash=axiom2_hash, edge_index=1,
        )

        # Level 2: Second derivation (from derived1)
        derived2_text = "((A1 -> A2) -> A1)"
        derived2_id, derived2_hash = _insert_statement(cur, system_id, derived2_text)
        proof2_id = _insert_proof(cur, derived2_id, system_id)
        record_proof_edge(
            cur, proof_id=proof2_id, child_statement_id=derived2_id,
            child_hash=derived2_hash, parent_statement_id=derived1_id,
            parent_hash=derived1_hash, edge_index=0,
        )

        # Load DAG and validate
        repo = ProofDagRepository(cur)
        dag = repo.load_dag()

        # Validate no cycles or other issues
        pytest_assert_dag_valid(dag, "Multi-level chain DAG should be valid")

        # Verify transitive ancestors of derived2
        ancestors_of_d2 = dag.ancestors(derived2_hash)
        assert derived1_hash in ancestors_of_d2, "derived1 should be ancestor of derived2"
        assert axiom1_hash in ancestors_of_d2, "axiom1 should be transitive ancestor of derived2"
        assert axiom2_hash in ancestors_of_d2, "axiom2 should be transitive ancestor of derived2"

        # Verify transitive descendants of axiom1
        descendants_of_a1 = dag.descendants(axiom1_hash)
        assert derived1_hash in descendants_of_a1, "derived1 should be descendant of axiom1"
        assert derived2_hash in descendants_of_a1, "derived2 should be transitive descendant of axiom1"

        # Use assertion helper for multi-hop ancestor chain
        result = assert_ancestor_chain(
            dag, derived2_hash, {axiom1_hash, axiom2_hash, derived1_hash}
        )
        assert result.passed, f"Ancestor chain check failed: {result.message}"


@pytest.mark.first_organism
@pytest.mark.integration
def test_diamond_shaped_derivation(first_organism_db) -> None:
    """
    Test DAG integrity for diamond-shaped derivation patterns.

    Creates:
           axiom
          /     \\
      derived1  derived2
          \\     /
          merged

    Verifies DAG handles converging derivation paths correctly.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Diamond Pattern Theory")

        # Top of diamond: axiom
        axiom_id, axiom_hash = _insert_statement(cur, system_id, "X")

        # Middle: two derivations from same axiom
        derived1_id, derived1_hash = _insert_statement(cur, system_id, "(X -> X)")
        proof1_id = _insert_proof(cur, derived1_id, system_id)
        record_proof_edge(
            cur, proof_id=proof1_id, child_statement_id=derived1_id,
            child_hash=derived1_hash, parent_statement_id=axiom_id,
            parent_hash=axiom_hash, edge_index=0,
        )

        derived2_id, derived2_hash = _insert_statement(cur, system_id, "((X -> X) -> X)")
        proof2_id = _insert_proof(cur, derived2_id, system_id)
        record_proof_edge(
            cur, proof_id=proof2_id, child_statement_id=derived2_id,
            child_hash=derived2_hash, parent_statement_id=axiom_id,
            parent_hash=axiom_hash, edge_index=0,
        )

        # Bottom: merged from both derived statements
        merged_id, merged_hash = _insert_statement(
            cur, system_id, "(((X -> X) -> X) -> (X -> X))"
        )
        proof3_id = _insert_proof(cur, merged_id, system_id)
        record_proof_edge(
            cur, proof_id=proof3_id, child_statement_id=merged_id,
            child_hash=merged_hash, parent_statement_id=derived1_id,
            parent_hash=derived1_hash, edge_index=0,
        )
        record_proof_edge(
            cur, proof_id=proof3_id, child_statement_id=merged_id,
            child_hash=merged_hash, parent_statement_id=derived2_id,
            parent_hash=derived2_hash, edge_index=1,
        )

        # Load and validate DAG
        repo = ProofDagRepository(cur)
        dag = repo.load_dag()

        # Verify DAG is acyclic and valid
        pytest_assert_dag_valid(dag, "Diamond DAG should be valid")

        # Verify merged statement has both derived statements as parents
        result = assert_lineage_complete(dag, merged_hash, {derived1_hash, derived2_hash})
        assert result.passed, f"Lineage check failed: {result.message}"

        # Verify axiom appears once in ancestors (not duplicated)
        ancestors = dag.ancestors(merged_hash)
        assert axiom_hash in ancestors, "Axiom should be transitive ancestor of merged"

        # Verify all paths lead to the same axiom
        result = assert_ancestor_chain(
            dag, merged_hash, {axiom_hash, derived1_hash, derived2_hash}
        )
        assert result.passed, f"Ancestor chain check failed: {result.message}"


# =============================================================================
# Edge Index Ordering Tests
# =============================================================================

@pytest.mark.first_organism
@pytest.mark.integration
def test_edge_index_sequential_ordering(first_organism_db) -> None:
    """
    Verify that edge indices are sequential (0, 1, 2, ...) for each proof.

    This is critical for deterministic proof reconstruction.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Edge Index Test Theory")

        # Create 4 parent statements
        parents = []
        for i in range(4):
            stmt_id, stmt_hash = _insert_statement(cur, system_id, f"P{i}")
            parents.append((stmt_id, stmt_hash))

        # Create child with 4 parents
        child_id, child_hash = _insert_statement(cur, system_id, "((P0 /\\ P1) /\\ (P2 /\\ P3))")
        proof_id = _insert_proof(cur, child_id, system_id)

        # Record edges with sequential indices
        for idx, (parent_id, parent_hash) in enumerate(parents):
            record_proof_edge(
                cur, proof_id=proof_id, child_statement_id=child_id,
                child_hash=child_hash, parent_statement_id=parent_id,
                parent_hash=parent_hash, edge_index=idx,
            )

        # Load DAG and check edge index ordering
        repo = ProofDagRepository(cur)
        dag = repo.load_dag()

        result = assert_edge_index_ordering(dag, proof_id)
        assert result.passed, f"Edge index ordering check failed: {result.message}"

        # Verify we can recover the exact parent order
        edges_for_proof = sorted(
            [e for e in dag.edges if e.proof_id == proof_id],
            key=lambda e: e.edge_index
        )
        assert len(edges_for_proof) == 4
        for i, edge in enumerate(edges_for_proof):
            assert edge.edge_index == i, f"Edge {i} has wrong index: {edge.edge_index}"
            assert edge.parent_hash == parents[i][1], f"Edge {i} has wrong parent hash"


# =============================================================================
# Hash/ID Consistency Tests
# =============================================================================

@pytest.mark.first_organism
@pytest.mark.integration
def test_hash_id_mapping_consistency(first_organism_db) -> None:
    """
    Verify that the same statement ID always maps to the same hash across all edges.

    Hash/ID consistency is critical for data integrity.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Hash Consistency Test Theory")

        # Create a statement that appears as parent in multiple proofs
        shared_parent_id, shared_parent_hash = _insert_statement(cur, system_id, "SHARED")

        # Create two different children both using the shared parent
        child1_id, child1_hash = _insert_statement(cur, system_id, "(SHARED -> A)")
        proof1_id = _insert_proof(cur, child1_id, system_id)
        record_proof_edge(
            cur, proof_id=proof1_id, child_statement_id=child1_id,
            child_hash=child1_hash, parent_statement_id=shared_parent_id,
            parent_hash=shared_parent_hash, edge_index=0,
        )

        child2_id, child2_hash = _insert_statement(cur, system_id, "(SHARED -> B)")
        proof2_id = _insert_proof(cur, child2_id, system_id)
        record_proof_edge(
            cur, proof_id=proof2_id, child_statement_id=child2_id,
            child_hash=child2_hash, parent_statement_id=shared_parent_id,
            parent_hash=shared_parent_hash, edge_index=0,
        )

        # Load DAG and verify hash/ID consistency
        repo = ProofDagRepository(cur)
        dag = repo.load_dag()

        result = assert_hash_id_consistency(dag)
        assert result.passed, f"Hash/ID consistency check failed: {result.message}"

        # Manually verify the shared parent appears with consistent hash
        edges_with_shared_parent = [
            e for e in dag.edges if e.parent_statement_id == shared_parent_id
        ]
        assert len(edges_with_shared_parent) == 2
        for edge in edges_with_shared_parent:
            assert edge.parent_hash == shared_parent_hash, \
                f"Hash mismatch for statement {shared_parent_id}"


# =============================================================================
# Error Detection Tests (Negative Tests)
# =============================================================================

@pytest.mark.first_organism
def test_cycle_detection_unit() -> None:
    """
    Unit test verifying cycle detection works correctly.

    Cycles in the proof DAG are logically invalid.
    """
    def _edge(child_hash, parent_hash):
        return ProofEdge(
            proof_id=None, child_statement_id=None, child_hash=child_hash,
            parent_statement_id=None, parent_hash=parent_hash, edge_index=0
        )

    # Create a cycle: A -> B -> C -> A
    edges = [
        _edge("hash_b", "hash_a"),  # B derived from A
        _edge("hash_c", "hash_b"),  # C derived from B
        _edge("hash_a", "hash_c"),  # A derived from C (creates cycle)
    ]
    dag = ProofDag(edges)

    result = assert_no_cycles(dag)
    assert not result.passed, "Cycle should have been detected"
    assert "cycle_nodes" in str(result.details)


@pytest.mark.first_organism
def test_self_loop_detection_unit() -> None:
    """
    Unit test verifying self-loop detection works correctly.

    A statement cannot prove itself.
    """
    def _edge(child_hash, parent_hash, child_id=None, parent_id=None):
        return ProofEdge(
            proof_id=1, child_statement_id=child_id, child_hash=child_hash,
            parent_statement_id=parent_id, parent_hash=parent_hash, edge_index=0
        )

    # Self-loop by hash
    edges = [_edge("same_hash", "same_hash")]
    dag = ProofDag(edges)
    result = assert_no_self_loops(dag)
    assert not result.passed, "Self-loop (by hash) should have been detected"

    # Self-loop by ID
    edges = [_edge("hash_a", "hash_b", child_id=1, parent_id=1)]
    dag = ProofDag(edges)
    result = assert_no_self_loops(dag)
    assert not result.passed, "Self-loop (by ID) should have been detected"


@pytest.mark.first_organism
def test_duplicate_edge_detection_unit() -> None:
    """
    Unit test verifying duplicate edge detection works correctly.

    Each (proof_id, child, parent) tuple should be unique.
    """
    def _edge(proof_id, child_id, parent_id, edge_index=0):
        return ProofEdge(
            proof_id=proof_id, child_statement_id=child_id, child_hash=f"h{child_id}",
            parent_statement_id=parent_id, parent_hash=f"h{parent_id}", edge_index=edge_index
        )

    # Duplicate edges: same (proof_id, child, parent) but different edge_index
    edges = [
        _edge(1, 2, 1, edge_index=0),
        _edge(1, 2, 1, edge_index=1),  # Duplicate of above
        _edge(1, 2, 1, edge_index=2),  # Another duplicate
    ]
    dag = ProofDag(edges)

    result = assert_no_duplicate_edges(dag)
    assert not result.passed, "Duplicate edges should have been detected"
    duplicates = result.details["duplicates"]
    assert len(duplicates) == 1
    assert duplicates[0]["count"] == 3


# =============================================================================
# Comprehensive Organism Lineage Report Test
# =============================================================================

@pytest.mark.first_organism
@pytest.mark.integration
def test_comprehensive_organism_lineage_report(first_organism_db) -> None:
    """
    Generate and validate a comprehensive OrganismLineageReport.

    This test demonstrates the full validation output structure.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Comprehensive Report Theory")

        # Create canonical First Organism structure
        p_id, p_hash = _insert_statement(cur, system_id, "p")
        q_id, q_hash = _insert_statement(cur, system_id, "q")

        organism_id, organism_hash = _insert_statement(cur, system_id, "((p /\\ q) -> p)")
        proof_id = _insert_proof(cur, organism_id, system_id)

        record_proof_edge(
            cur, proof_id=proof_id, child_statement_id=organism_id,
            child_hash=organism_hash, parent_statement_id=p_id,
            parent_hash=p_hash, edge_index=0,
        )
        record_proof_edge(
            cur, proof_id=proof_id, child_statement_id=organism_id,
            child_hash=organism_hash, parent_statement_id=q_id,
            parent_hash=q_hash, edge_index=1,
        )

        # Use pytest assertion helper (will raise on failure)
        repo = ProofDagRepository(cur)
        dag = repo.load_dag()

        report = pytest_assert_organism_lineage(
            dag, organism_hash, {p_hash, q_hash}, proof_id=proof_id
        )

        # Verify report structure
        assert report.ok
        assert report.metrics["total_edges"] == 2
        assert report.metrics["organism_parent_count"] == 2
        assert report.metrics["total_invariants"] >= 8  # All standard checks
        assert report.metrics["failed_invariants"] == 0

        # Verify all expected invariants are present
        expected_invariants = {
            "no_cycles", "no_self_loops", "no_duplicate_edges",
            "hash_id_consistency", "complete_edges", "lineage_complete",
            "ancestor_chain", "edge_index_ordering",
        }
        actual_invariants = set(report.invariants.keys())
        assert expected_invariants <= actual_invariants, \
            f"Missing invariants: {expected_invariants - actual_invariants}"


# =============================================================================
# Multi-Block DAG Tests
# =============================================================================

@pytest.mark.first_organism
@pytest.mark.integration
def test_dag_across_multiple_blocks(first_organism_db) -> None:
    """
    Test DAG integrity when proofs span multiple blocks/seals.

    Simulates a realistic scenario where:
    - Block 1: Axioms are sealed
    - Block 2: First derivation is sealed
    - Block 3: Second derivation (using Block 2 result) is sealed

    Verifies the DAG correctly tracks cross-block dependencies.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Multi-Block Theory")

        # Block 1: Axioms (no parents)
        axiom_a_id, axiom_a_hash = _insert_statement(cur, system_id, "A")
        axiom_b_id, axiom_b_hash = _insert_statement(cur, system_id, "B")
        # Axioms have no proof edges (they are roots)

        # Block 2: First derivation from axioms
        derived1_id, derived1_hash = _insert_statement(cur, system_id, "(A -> B)")
        proof1_id = _insert_proof(cur, derived1_id, system_id)
        record_proof_edge(
            cur, proof_id=proof1_id, child_statement_id=derived1_id,
            child_hash=derived1_hash, parent_statement_id=axiom_a_id,
            parent_hash=axiom_a_hash, edge_index=0,
        )
        record_proof_edge(
            cur, proof_id=proof1_id, child_statement_id=derived1_id,
            child_hash=derived1_hash, parent_statement_id=axiom_b_id,
            parent_hash=axiom_b_hash, edge_index=1,
        )

        # Block 3: Second derivation using Block 2 result
        derived2_id, derived2_hash = _insert_statement(cur, system_id, "((A -> B) -> A)")
        proof2_id = _insert_proof(cur, derived2_id, system_id)
        record_proof_edge(
            cur, proof_id=proof2_id, child_statement_id=derived2_id,
            child_hash=derived2_hash, parent_statement_id=derived1_id,
            parent_hash=derived1_hash, edge_index=0,
        )

        # Block 4: Third derivation using both Block 2 and Block 3 results
        derived3_id, derived3_hash = _insert_statement(
            cur, system_id, "(((A -> B) -> A) -> (A -> B))"
        )
        proof3_id = _insert_proof(cur, derived3_id, system_id)
        record_proof_edge(
            cur, proof_id=proof3_id, child_statement_id=derived3_id,
            child_hash=derived3_hash, parent_statement_id=derived1_id,
            parent_hash=derived1_hash, edge_index=0,
        )
        record_proof_edge(
            cur, proof_id=proof3_id, child_statement_id=derived3_id,
            child_hash=derived3_hash, parent_statement_id=derived2_id,
            parent_hash=derived2_hash, edge_index=1,
        )

        # Load and validate the complete DAG
        repo = ProofDagRepository(cur)
        report = repo.validate()

        assert report.ok, f"Multi-block DAG validation failed: {report.issues}"

        # Verify no sentinel issues
        unverified = [k for k in report.issues if k.endswith("_unverified")]
        assert not unverified, f"Schema incomplete: {unverified}"

        # Load DAG and verify structure
        dag = repo.load_dag()

        # Verify total edge count (2 + 1 + 2 = 5 edges)
        assert len(dag.edges) == 5, f"Expected 5 edges, got {len(dag.edges)}"

        # Verify axioms are roots (no parents)
        assert len(dag.parents_of_hash(axiom_a_hash)) == 0
        assert len(dag.parents_of_hash(axiom_b_hash)) == 0

        # Verify derived3 has the full ancestor chain
        ancestors_of_d3 = set(dag.ancestors(derived3_hash))
        expected_ancestors = {axiom_a_hash, axiom_b_hash, derived1_hash, derived2_hash}
        assert ancestors_of_d3 == expected_ancestors, \
            f"Ancestor mismatch: expected {expected_ancestors}, got {ancestors_of_d3}"

        # Verify axiom_a has the full descendant chain
        descendants_of_a = set(dag.descendants(axiom_a_hash))
        expected_descendants = {derived1_hash, derived2_hash, derived3_hash}
        assert descendants_of_a == expected_descendants, \
            f"Descendant mismatch: expected {expected_descendants}, got {descendants_of_a}"


@pytest.mark.first_organism
@pytest.mark.integration
def test_dag_with_isolated_blocks(first_organism_db) -> None:
    """
    Test DAG integrity with isolated derivation chains (no cross-block deps).

    Creates two independent derivation chains and verifies they don't
    interfere with each other.
    """
    connection = first_organism_db
    with connection.cursor() as cur:
        _ensure_proof_parents_table(cur)
        system_id = _create_theory(cur, "Isolated Blocks Theory")

        # Chain 1: X -> Y
        x_id, x_hash = _insert_statement(cur, system_id, "X")
        y_id, y_hash = _insert_statement(cur, system_id, "(X -> X)")
        proof_y = _insert_proof(cur, y_id, system_id)
        record_proof_edge(
            cur, proof_id=proof_y, child_statement_id=y_id,
            child_hash=y_hash, parent_statement_id=x_id,
            parent_hash=x_hash, edge_index=0,
        )

        # Chain 2: P -> Q (completely independent)
        p_id, p_hash = _insert_statement(cur, system_id, "P")
        q_id, q_hash = _insert_statement(cur, system_id, "(P -> P)")
        proof_q = _insert_proof(cur, q_id, system_id)
        record_proof_edge(
            cur, proof_id=proof_q, child_statement_id=q_id,
            child_hash=q_hash, parent_statement_id=p_id,
            parent_hash=p_hash, edge_index=0,
        )

        # Validate DAG
        repo = ProofDagRepository(cur)
        dag = repo.load_dag()

        pytest_assert_dag_valid(dag, "Isolated chains DAG should be valid")

        # Verify chains are truly isolated
        ancestors_of_y = set(dag.ancestors(y_hash))
        assert ancestors_of_y == {x_hash}, "Y should only have X as ancestor"
        assert p_hash not in ancestors_of_y, "P should not be ancestor of Y"

        ancestors_of_q = set(dag.ancestors(q_hash))
        assert ancestors_of_q == {p_hash}, "Q should only have P as ancestor"
        assert x_hash not in ancestors_of_q, "X should not be ancestor of Q"

        # Verify no cross-chain descendants
        descendants_of_x = set(dag.descendants(x_hash))
        assert descendants_of_x == {y_hash}, "X should only have Y as descendant"
        assert q_hash not in descendants_of_x, "Q should not be descendant of X"

