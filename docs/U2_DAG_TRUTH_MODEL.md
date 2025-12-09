# U2 DAG Truth Model

**PHASE II — NOT RUN IN PHASE I**

**Author:** CLAUDE G — DAG Analyst General
**Status:** Specification Document
**Version:** 1.0.0

---

## 1. Overview

This document defines the formal graph model for U2 derivation experiments, establishes structural invariants that must hold for valid proof DAGs, and classifies error conditions that indicate DAG pathologies.

The U2 DAG Truth Model serves as the authoritative reference for:
- Validating derivation chain structures
- Detecting anomalies in proof dependencies
- Ensuring deterministic and reproducible analysis

---

## 2. Formal Graph Model

### 2.1 Definition

A **U2 Derivation DAG** is a directed acyclic graph G = (V, E) where:

- **V** (Vertices): Set of statement hashes representing proven or candidate statements
- **E** (Edges): Set of directed edges (child, parent) representing derivation dependencies

Each vertex v ∈ V has associated metadata:
```
v = {
    hash: SHA256(normalized_formula),
    normalized: canonical_ascii_form,
    parents: [parent_hash_1, parent_hash_2, ...],
    verified: boolean,
    depth: integer,
    rule: derivation_rule_name
}
```

### 2.2 Edge Semantics

An edge (c, p) ∈ E indicates that statement c **depends on** statement p:
- c is the **child** (derived statement)
- p is the **parent** (premise used in derivation)

For Modus Ponens derivations:
```
    (p, p → q) ⊢ q

    Edges: (q, p), (q, p→q)
```

### 2.3 Vertex Classification

| Class | Definition | Depth |
|-------|------------|-------|
| **Axiom** | v where outdegree(v) = 0 (no parents) | 1 |
| **Derived** | v where outdegree(v) > 0 | > 1 |
| **Leaf** | v where indegree(v) = 0 (no children) | Any |
| **Internal** | v where indegree(v) > 0 AND outdegree(v) > 0 | > 1 |

### 2.4 Chain Depth Function

The **chain depth** d(v) of a vertex v is defined recursively:

```
d(v) = {
    1                           if parents(v) = ∅
    1 + max(d(p) for p in parents(v))   otherwise
}
```

Properties:
- d(v) ≥ 1 for all v ∈ V
- d(v) = 1 iff v is an axiom
- d(v) represents the longest path from any axiom to v

### 2.5 Verified Chain Depth

For a set S ⊆ V of verified statements, the **verified chain depth** d_S(v) is:

```
d_S(v) = {
    0                           if v ∉ S
    1                           if v ∈ S AND (parents(v) ∩ S = ∅)
    1 + max(d_S(p) for p in parents(v) ∩ S)   otherwise
}
```

This metric only counts chains through verified statements.

---

## 3. Structural Invariants

### 3.1 Acyclicity Invariant (INV-001)

**Statement:** The derivation DAG must be acyclic.

**Formal:** ∀ v ∈ V: v ∉ ancestors(v)

**Rationale:** A cycle would imply a statement is used to prove itself, violating logical soundness.

**Verification:** Kahn's algorithm (topological sort) or DFS cycle detection.

**Severity:** CRITICAL — Any cycle invalidates the entire proof structure.

### 3.2 Self-Loop Prohibition (INV-002)

**Statement:** No vertex may have itself as a parent.

**Formal:** ∀ v ∈ V: (v, v) ∉ E

**Rationale:** Self-reference is a degenerate form of cycle.

**Verification:** Direct edge inspection.

**Severity:** CRITICAL

### 3.3 Parent Existence (INV-003)

**Statement:** Every parent reference must correspond to an existing vertex.

**Formal:** ∀ (c, p) ∈ E: p ∈ V

**Rationale:** Dangling references indicate incomplete or corrupted data.

**Verification:** Set membership check during DAG construction.

**Severity:** ERROR — Invalidates affected derivation chains.

### 3.4 Hash Uniqueness (INV-004)

**Statement:** Each hash in V corresponds to exactly one normalized formula.

**Formal:** ∀ v1, v2 ∈ V: v1.hash = v2.hash ⟹ v1.normalized = v2.normalized

**Rationale:** Hash collisions would undermine provenance tracking.

**Verification:** Hash-to-formula mapping consistency check.

**Severity:** CRITICAL — Indicates hash function failure or data corruption.

### 3.5 Depth Boundedness (INV-005)

**Statement:** Chain depth must not exceed the configured maximum.

**Formal:** ∀ v ∈ V: d(v) ≤ MAX_CHAIN_DEPTH

**Rationale:** Unbounded depth indicates runaway derivation or misconfiguration.

**Verification:** Compute max depth and compare to configuration.

**Severity:** WARNING — May indicate slice bounds violation.

### 3.6 Axiom Foundation (INV-006)

**Statement:** Every derived statement must have a path to at least one axiom.

**Formal:** ∀ v ∈ V where d(v) > 1: ∃ path from v to some axiom a where d(a) = 1

**Rationale:** All proofs must ultimately rest on axioms.

**Verification:** BFS/DFS from each derived vertex.

**Severity:** ERROR — Orphaned derivations are logically unsound.

### 3.7 Deterministic Ordering (INV-007)

**Statement:** Parent lists must be stored in canonical (sorted) order.

**Formal:** ∀ v ∈ V: parents(v) = sorted(parents(v))

**Rationale:** Ensures deterministic hash computation and reproducible analysis.

**Verification:** Check sorted order of parent arrays.

**Severity:** WARNING — Non-determinism in serialization.

---

## 4. Error Classifications

### 4.1 Structural Pathologies

| Code | Name | Description | Severity | Recovery |
|------|------|-------------|----------|----------|
| **E-CYCLE** | Cyclic Dependency | DAG contains one or more cycles | CRITICAL | Reject entire DAG |
| **E-SELF** | Self-Loop | Vertex references itself as parent | CRITICAL | Remove self-edge |
| **E-DANGLE** | Dangling Reference | Parent hash not found in vertex set | ERROR | Mark chain invalid |
| **E-ORPHAN** | Orphaned Derivation | Derived vertex unreachable from axioms | ERROR | Mark vertex invalid |
| **E-DUPLICATE** | Duplicate Edge | Same (child, parent) edge appears multiple times | WARNING | Deduplicate |

### 4.2 Depth Pathologies

| Code | Name | Description | Severity | Recovery |
|------|------|-------------|----------|----------|
| **D-EXCEED** | Depth Exceeded | Chain depth exceeds configured maximum | WARNING | Flag for review |
| **D-FLAT** | Trivial Depth | All statements at depth 1 (no derivations) | WARNING | Check derivation engine |
| **D-SPARSE** | Sparse Distribution | Large gaps in depth distribution | INFO | May indicate slice issues |
| **D-UNBOUNDED** | Unbounded Growth | Depth increases without convergence | WARNING | Check termination |

### 4.3 Verification Pathologies

| Code | Name | Description | Severity | Recovery |
|------|------|-------------|----------|----------|
| **V-GAP** | Verification Gap | Unverified statement in otherwise verified chain | WARNING | Recompute verified depth |
| **V-EMPTY** | No Verifications | Zero verified statements in log | ERROR | Check verifier |
| **V-MISMATCH** | Status Mismatch | Verification status inconsistent across records | ERROR | Reconcile statuses |

### 4.4 Provenance Pathologies

| Code | Name | Description | Severity | Recovery |
|------|------|-------------|----------|----------|
| **P-HASH** | Hash Collision | Different formulas produce same hash | CRITICAL | Investigate hash function |
| **P-RULE** | Invalid Rule | Unknown or malformed derivation rule | WARNING | Flag derivation |
| **P-ORDER** | Non-Canonical Order | Parent list not in sorted order | WARNING | Re-sort parents |

---

## 5. DAG Metrics

### 5.1 Structural Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Node Count** | \|V\| | Total statements in DAG |
| **Edge Count** | \|E\| | Total dependency relationships |
| **Density** | \|E\| / (\|V\| × (\|V\| - 1)) | Sparsity measure |
| **Axiom Ratio** | \|{v : d(v) = 1}\| / \|V\| | Foundation breadth |
| **Branching Factor** | avg(indegree(v)) | Average children per node |

### 5.2 Depth Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Max Depth** | max(d(v) for v ∈ V) | Longest derivation chain |
| **Mean Depth** | avg(d(v) for v ∈ V) | Average chain complexity |
| **Depth Variance** | var(d(v) for v ∈ V) | Distribution spread |
| **Verified Max Depth** | max(d_S(v) for v ∈ S) | Longest verified chain |

### 5.3 Chain Success Metrics (for slice_uplift_tree)

| Metric | Definition | Success Criterion |
|--------|------------|-------------------|
| **Chain Length** | d_S(target) | ≥ min_chain_length |
| **Target Verified** | target ∈ S | TRUE |
| **Chain Complete** | All ancestors verified | TRUE |

---

## 6. Verification Protocol

### 6.1 Pre-Analysis Checks

Before computing metrics, verify:

1. **INV-002**: No self-loops
2. **INV-001**: No cycles (via Kahn's algorithm)
3. **INV-003**: All parent references resolve

If any CRITICAL invariant fails, abort analysis.

### 6.2 Depth Computation Order

Compute depths in topological order to ensure:
- All parent depths computed before child
- Memoization prevents redundant computation
- O(V + E) time complexity

### 6.3 Verified Chain Analysis

For chain_success metric:
1. Filter to verified statements: S = {v : v.verified = TRUE}
2. Compute d_S(target)
3. Compare to min_chain_length threshold

---

## 7. Implementation Reference

The formal model is implemented in:

- `experiments/derivation_chain_analysis.py` — Core DAG functions
- `experiments/analyze_chain_depth_u2.py` — CLI analyzer with sanity checks

Key functions mapping to this specification:

| Specification | Implementation |
|---------------|----------------|
| d(v) | `compute_chain_depth(v, dag)` |
| d_S(v) | `compute_chain_depth_filtered(v, dag, S)` |
| INV-001 check | `validate_dag(dag).has_cycles` |
| INV-002 check | `validate_dag(dag).self_loops` |
| Chain success | `verify_chain_success(S, dag, target, min_len)` |

---

## 8. References

- `PREREG_UPLIFT_U2.yaml` — Preregistration for U2 experiments
- `docs/whitepaper.md` — MathLedger system architecture
- `backend/dag/proof_dag.py` — Database DAG repository

---

*PHASE II — NOT RUN IN PHASE I*
