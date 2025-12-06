"""
DAG Integrity Assertion Helpers for First Organism Lineage.

Provides reusable assertion functions for verifying:
- No cycles in proof DAG
- No duplicate edges
- Correct hash/ID linkage consistency
- Complete lineage capture for derived statements
- Edge index ordering
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.dag import ProofDag, ProofDagRepository, ProofDagValidationReport, ProofEdge


@dataclass
class LineageAssertion:
    """Result of a lineage assertion check."""

    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class OrganismLineageReport:
    """Comprehensive report on First Organism DAG integrity."""

    ok: bool
    summary: str
    invariants: Dict[str, LineageAssertion]
    metrics: Dict[str, int]
    dag_report: Optional[ProofDagValidationReport] = None


def assert_no_cycles(dag: ProofDag) -> LineageAssertion:
    """
    Assert that the proof DAG contains no cycles.

    A cycle would indicate an impossible derivation chain where
    a statement depends on itself through some path.
    """
    report = dag.validate()
    cycle_nodes = report.issues.get("cycle_nodes", [])

    if cycle_nodes:
        return LineageAssertion(
            passed=False,
            message=f"DAG contains {len(cycle_nodes)} nodes in cycles",
            details={"cycle_nodes": cycle_nodes},
        )

    return LineageAssertion(
        passed=True,
        message="No cycles detected in proof DAG",
    )


def assert_no_self_loops(dag: ProofDag) -> LineageAssertion:
    """
    Assert that no statement is its own parent.

    Self-loops would indicate a statement proven from itself,
    which is logically invalid.
    """
    report = dag.validate()
    self_loops = report.issues.get("self_loops", [])

    if self_loops:
        return LineageAssertion(
            passed=False,
            message=f"DAG contains {len(self_loops)} self-loop edges",
            details={"self_loops": self_loops},
        )

    return LineageAssertion(
        passed=True,
        message="No self-loops detected",
    )


def assert_no_duplicate_edges(dag: ProofDag) -> LineageAssertion:
    """
    Assert that no duplicate parent-child edges exist for the same proof.

    Duplicate edges indicate recording errors or logic bugs in
    the derivation pipeline.
    """
    report = dag.validate()
    duplicates = report.issues.get("duplicate_edges", [])

    if duplicates:
        return LineageAssertion(
            passed=False,
            message=f"DAG contains {len(duplicates)} duplicate edge groups",
            details={"duplicates": duplicates},
        )

    return LineageAssertion(
        passed=True,
        message="No duplicate edges detected",
    )


def assert_hash_id_consistency(dag: ProofDag) -> LineageAssertion:
    """
    Assert that hash/ID mappings are consistent across all edges.

    If a statement ID appears with different hashes in different edges,
    this indicates data corruption or hash collision.
    """
    report = dag.validate()
    child_mismatches = report.issues.get("child_hash_mismatch", [])
    parent_mismatches = report.issues.get("parent_hash_mismatch", [])

    all_mismatches = child_mismatches + parent_mismatches

    if all_mismatches:
        return LineageAssertion(
            passed=False,
            message=f"Hash/ID inconsistency: {len(all_mismatches)} mismatches",
            details={
                "child_hash_mismatches": child_mismatches,
                "parent_hash_mismatches": parent_mismatches,
            },
        )

    return LineageAssertion(
        passed=True,
        message="Hash/ID mappings are consistent",
    )


def assert_complete_edges(dag: ProofDag) -> LineageAssertion:
    """
    Assert that all edges have complete identification.

    Incomplete edges (missing both child/parent ID and hash) cannot
    be properly validated for integrity.
    """
    report = dag.validate()
    incomplete = report.issues.get("incomplete_edge", [])

    if incomplete:
        return LineageAssertion(
            passed=False,
            message=f"DAG contains {len(incomplete)} incomplete edges",
            details={"incomplete_edges": incomplete},
        )

    return LineageAssertion(
        passed=True,
        message="All edges have complete identification",
    )


def assert_edge_index_ordering(dag: ProofDag, proof_id: int) -> LineageAssertion:
    """
    Assert that edge_index values for a proof are sequential and unique.

    Edge indices provide stable ordering of dependencies within a proof.
    """
    edges_for_proof = [e for e in dag.edges if e.proof_id == proof_id]

    if not edges_for_proof:
        return LineageAssertion(
            passed=True,
            message=f"No edges found for proof_id={proof_id}",
        )

    indices = sorted(e.edge_index for e in edges_for_proof)
    expected = list(range(len(indices)))

    if indices != expected:
        return LineageAssertion(
            passed=False,
            message=f"Edge indices not sequential for proof_id={proof_id}",
            details={"found": indices, "expected": expected},
        )

    # Check for duplicate indices
    if len(indices) != len(set(indices)):
        return LineageAssertion(
            passed=False,
            message=f"Duplicate edge indices for proof_id={proof_id}",
            details={"indices": indices},
        )

    return LineageAssertion(
        passed=True,
        message=f"Edge indices valid for proof_id={proof_id}",
    )


def assert_lineage_complete(
    dag: ProofDag,
    child_hash: str,
    expected_parent_hashes: Set[str],
) -> LineageAssertion:
    """
    Assert that a statement has exactly the expected parent lineage.

    This verifies that the derivation chain is correctly captured
    for the First Organism proof path.
    """
    parent_edges = dag.parents_of_hash(child_hash)
    actual_parent_hashes = {e.parent_hash for e in parent_edges if e.parent_hash}

    missing = expected_parent_hashes - actual_parent_hashes
    extra = actual_parent_hashes - expected_parent_hashes

    if missing or extra:
        return LineageAssertion(
            passed=False,
            message="Lineage mismatch for child statement",
            details={
                "child_hash": child_hash,
                "expected_parents": list(expected_parent_hashes),
                "actual_parents": list(actual_parent_hashes),
                "missing": list(missing),
                "extra": list(extra),
            },
        )

    return LineageAssertion(
        passed=True,
        message=f"Lineage complete: {len(actual_parent_hashes)} parents",
        details={"parent_hashes": list(actual_parent_hashes)},
    )


def assert_ancestor_chain(
    dag: ProofDag,
    child_hash: str,
    required_ancestors: Set[str],
    max_depth: Optional[int] = None,
) -> LineageAssertion:
    """
    Assert that all required ancestors appear in the transitive closure.

    This verifies that multi-hop derivation chains are correctly linked.
    """
    actual_ancestors = set(dag.ancestors(child_hash, max_depth=max_depth))
    missing = required_ancestors - actual_ancestors

    if missing:
        return LineageAssertion(
            passed=False,
            message=f"Missing {len(missing)} required ancestors",
            details={
                "child_hash": child_hash,
                "missing_ancestors": list(missing),
                "found_ancestors": list(actual_ancestors),
            },
        )

    return LineageAssertion(
        passed=True,
        message=f"All {len(required_ancestors)} required ancestors found",
        details={"ancestors": list(actual_ancestors)},
    )


def assert_descendant_chain(
    dag: ProofDag,
    parent_hash: str,
    required_descendants: Set[str],
    max_depth: Optional[int] = None,
) -> LineageAssertion:
    """
    Assert that all required descendants appear in the transitive closure.

    This verifies forward propagation of derivations from axioms.
    """
    actual_descendants = set(dag.descendants(parent_hash, max_depth=max_depth))
    missing = required_descendants - actual_descendants

    if missing:
        return LineageAssertion(
            passed=False,
            message=f"Missing {len(missing)} required descendants",
            details={
                "parent_hash": parent_hash,
                "missing_descendants": list(missing),
                "found_descendants": list(actual_descendants),
            },
        )

    return LineageAssertion(
        passed=True,
        message=f"All {len(required_descendants)} required descendants found",
        details={"descendants": list(actual_descendants)},
    )


def validate_organism_lineage(
    dag: ProofDag,
    organism_hash: str,
    expected_parent_hashes: Set[str],
    proof_id: Optional[int] = None,
) -> OrganismLineageReport:
    """
    Comprehensive validation of First Organism proof lineage.

    Runs all DAG invariant checks and lineage-specific assertions.

    Args:
        dag: ProofDag instance to validate
        organism_hash: Hash of the "organism" (derived) statement
        expected_parent_hashes: Set of expected direct parent hashes
        proof_id: Optional proof ID to validate edge ordering

    Returns:
        OrganismLineageReport with all invariant checks
    """
    dag_report = dag.validate()

    invariants: Dict[str, LineageAssertion] = {
        "no_cycles": assert_no_cycles(dag),
        "no_self_loops": assert_no_self_loops(dag),
        "no_duplicate_edges": assert_no_duplicate_edges(dag),
        "hash_id_consistency": assert_hash_id_consistency(dag),
        "complete_edges": assert_complete_edges(dag),
        "lineage_complete": assert_lineage_complete(
            dag, organism_hash, expected_parent_hashes
        ),
        "ancestor_chain": assert_ancestor_chain(
            dag, organism_hash, expected_parent_hashes
        ),
    }

    if proof_id is not None:
        invariants["edge_index_ordering"] = assert_edge_index_ordering(dag, proof_id)

    # Add descendant checks for each parent
    for idx, parent_hash in enumerate(expected_parent_hashes):
        key = f"descendant_from_parent_{idx}"
        invariants[key] = assert_descendant_chain(
            dag, parent_hash, {organism_hash}
        )

    all_passed = all(inv.passed for inv in invariants.values())
    failed_names = [name for name, inv in invariants.items() if not inv.passed]

    summary = "All organism lineage invariants passed"
    if not all_passed:
        summary = f"Failed invariants: {', '.join(failed_names)}"

    metrics = {
        "total_edges": len(dag.edges),
        "total_invariants": len(invariants),
        "passed_invariants": sum(1 for inv in invariants.values() if inv.passed),
        "failed_invariants": len(failed_names),
        "organism_parent_count": len(expected_parent_hashes),
    }

    return OrganismLineageReport(
        ok=all_passed,
        summary=summary,
        invariants=invariants,
        metrics=metrics,
        dag_report=dag_report,
    )


def validate_organism_lineage_from_db(
    cur,
    organism_hash: str,
    expected_parent_hashes: Set[str],
    proof_id: Optional[int] = None,
) -> OrganismLineageReport:
    """
    Load DAG from database and validate organism lineage.

    Convenience wrapper that handles DB loading.

    Args:
        cur: Database cursor
        organism_hash: Hash of the derived statement
        expected_parent_hashes: Set of expected direct parent hashes
        proof_id: Optional proof ID to validate edge ordering

    Returns:
        OrganismLineageReport with all invariant checks
    """
    repo = ProofDagRepository(cur)
    dag = repo.load_dag()

    # Also run DB-level validation
    db_report = repo.validate()

    report = validate_organism_lineage(
        dag, organism_hash, expected_parent_hashes, proof_id
    )

    # Merge DB-level issues
    if not db_report.ok:
        for key, value in db_report.issues.items():
            if key not in report.invariants:
                report.invariants[f"db_{key}"] = LineageAssertion(
                    passed=False,
                    message=f"DB validation issue: {key}",
                    details={"value": value},
                )
        report.ok = False
        report.summary = f"{report.summary}; DB issues: {list(db_report.issues.keys())}"

    return report


# Pytest assertion helpers
def pytest_assert_dag_valid(dag: ProofDag, message: str = "") -> None:
    """
    Pytest assertion that DAG is structurally valid.

    Raises AssertionError with detailed info on failure.
    """
    report = dag.validate()
    if not report.ok:
        details = "\n".join(
            f"  - {key}: {value}" for key, value in report.issues.items()
        )
        raise AssertionError(
            f"DAG validation failed{': ' + message if message else ''}\n"
            f"Issues:\n{details}\n"
            f"Metrics: {report.metrics}"
        )


def pytest_assert_organism_lineage(
    dag: ProofDag,
    organism_hash: str,
    expected_parent_hashes: Set[str],
    proof_id: Optional[int] = None,
) -> OrganismLineageReport:
    """
    Pytest assertion for complete organism lineage validation.

    Returns the report for further inspection on success.
    Raises AssertionError with detailed info on failure.
    """
    report = validate_organism_lineage(
        dag, organism_hash, expected_parent_hashes, proof_id
    )

    if not report.ok:
        failed = [
            f"  - {name}: {inv.message}"
            for name, inv in report.invariants.items()
            if not inv.passed
        ]
        details = "\n".join(failed)
        raise AssertionError(
            f"Organism lineage validation failed:\n"
            f"{report.summary}\n\n"
            f"Failed checks:\n{details}\n\n"
            f"Metrics: {report.metrics}"
        )

    return report


# =============================================================================
# Additional DAG Validation Utilities
# =============================================================================

def assert_proof_has_parents(
    dag: ProofDag,
    proof_id: int,
    expected_count: int,
) -> LineageAssertion:
    """
    Assert that a proof has the expected number of parent edges.

    Useful for verifying multi-parent derivations are correctly recorded.
    """
    edges = [e for e in dag.edges if e.proof_id == proof_id]
    actual_count = len(edges)

    if actual_count != expected_count:
        return LineageAssertion(
            passed=False,
            message=f"Proof {proof_id} has {actual_count} parents, expected {expected_count}",
            details={"proof_id": proof_id, "actual": actual_count, "expected": expected_count},
        )

    return LineageAssertion(
        passed=True,
        message=f"Proof {proof_id} has {expected_count} parent(s)",
    )


def assert_statement_is_root(
    dag: ProofDag,
    statement_hash: str,
) -> LineageAssertion:
    """
    Assert that a statement is a root node (has no parents).

    Root nodes are axioms or base statements in the derivation tree.
    """
    parents = dag.parents_of_hash(statement_hash)

    if parents:
        parent_hashes = [e.parent_hash for e in parents if e.parent_hash]
        return LineageAssertion(
            passed=False,
            message=f"Statement has {len(parents)} parent(s), expected to be root",
            details={"statement_hash": statement_hash, "parent_hashes": parent_hashes},
        )

    return LineageAssertion(
        passed=True,
        message="Statement is a root node (no parents)",
    )


def assert_statement_has_descendants(
    dag: ProofDag,
    statement_hash: str,
    min_count: int = 1,
) -> LineageAssertion:
    """
    Assert that a statement has at least the specified number of descendants.

    Useful for verifying axioms propagate through the derivation tree.
    """
    descendants = dag.descendants(statement_hash)

    if len(descendants) < min_count:
        return LineageAssertion(
            passed=False,
            message=f"Statement has {len(descendants)} descendants, expected at least {min_count}",
            details={"statement_hash": statement_hash, "descendants": descendants},
        )

    return LineageAssertion(
        passed=True,
        message=f"Statement has {len(descendants)} descendant(s)",
        details={"descendants": descendants},
    )


def get_all_root_hashes(dag: ProofDag) -> Set[str]:
    """
    Return all root node hashes (nodes with no parents).

    Roots represent axioms or base statements in the proof DAG.
    """
    # Collect all hashes that appear as children
    child_hashes: Set[str] = set()
    parent_hashes: Set[str] = set()

    for edge in dag.edges:
        if edge.child_hash:
            child_hashes.add(edge.child_hash)
        if edge.parent_hash:
            parent_hashes.add(edge.parent_hash)

    # Roots are parents that are never children
    return parent_hashes - child_hashes


def get_all_leaf_hashes(dag: ProofDag) -> Set[str]:
    """
    Return all leaf node hashes (nodes with no children).

    Leaves represent the most derived statements in the proof DAG.
    """
    child_hashes: Set[str] = set()
    parent_hashes: Set[str] = set()

    for edge in dag.edges:
        if edge.child_hash:
            child_hashes.add(edge.child_hash)
        if edge.parent_hash:
            parent_hashes.add(edge.parent_hash)

    # Leaves are children that are never parents
    return child_hashes - parent_hashes


def compute_dag_depth(dag: ProofDag, statement_hash: str) -> int:
    """
    Compute the derivation depth of a statement (longest path from any root).

    Returns 0 for root nodes, -1 if statement not found in DAG.
    """
    # Check if statement exists in DAG
    parents = dag.parents_of_hash(statement_hash)
    if not parents:
        # Could be a root or not in DAG
        children = dag.children_of_hash(statement_hash)
        if children:
            return 0  # Root with descendants
        # Check if it appears anywhere
        all_hashes: Set[str] = set()
        for edge in dag.edges:
            if edge.child_hash:
                all_hashes.add(edge.child_hash)
            if edge.parent_hash:
                all_hashes.add(edge.parent_hash)
        if statement_hash not in all_hashes:
            return -1  # Not found
        return 0  # Isolated node

    # BFS to find max depth from any root
    max_depth = 0
    for parent_edge in parents:
        if parent_edge.parent_hash:
            parent_depth = compute_dag_depth(dag, parent_edge.parent_hash)
            if parent_depth >= 0:
                max_depth = max(max_depth, parent_depth + 1)

    return max_depth


def validate_dag_metrics(dag: ProofDag) -> Dict[str, Any]:
    """
    Compute comprehensive metrics about the DAG structure.

    Returns a dictionary with:
    - node_count: Total unique nodes
    - edge_count: Total edges
    - root_count: Nodes with no parents
    - leaf_count: Nodes with no children
    - max_depth: Maximum derivation depth
    - avg_parents_per_derived: Average parent count for non-root nodes
    """
    roots = get_all_root_hashes(dag)
    leaves = get_all_leaf_hashes(dag)

    # Get all unique hashes
    all_hashes: Set[str] = set()
    for edge in dag.edges:
        if edge.child_hash:
            all_hashes.add(edge.child_hash)
        if edge.parent_hash:
            all_hashes.add(edge.parent_hash)

    # Compute max depth
    max_depth = 0
    for leaf_hash in leaves:
        depth = compute_dag_depth(dag, leaf_hash)
        if depth > max_depth:
            max_depth = depth

    # Compute average parents for derived nodes
    derived_nodes = all_hashes - roots
    total_parent_edges = sum(
        len(dag.parents_of_hash(h)) for h in derived_nodes
    )
    avg_parents = total_parent_edges / len(derived_nodes) if derived_nodes else 0.0

    return {
        "node_count": len(all_hashes),
        "edge_count": len(dag.edges),
        "root_count": len(roots),
        "leaf_count": len(leaves),
        "max_depth": max_depth,
        "avg_parents_per_derived": round(avg_parents, 2),
        "roots": list(roots),
        "leaves": list(leaves),
    }
