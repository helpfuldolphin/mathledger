# GLOBAL_DAG_CONSISTENCY_SPEC.md

**Version:** 1.0
**Status:** DRAFT
**Owner:** Gemini G (DAG Systems Engineer)

## 1. Overview

This document specifies the semantics, invariants, and error states related to the management of Proof Dependency Directed Acyclic Graphs (DAGs) within the MathLedger Phase II system. It defines the relationship between the per-cycle DAGs generated during an experiment run and the global, cumulative DAG that represents the total knowledge state of that run.

The purpose of this specification is to ensure that all DAG-related computations are consistent, deterministic, and robust against data anomalies.

## 2. DAG Definitions

### 2.1. Node

A **Node** represents a single, unique logical statement (e.g., a formula, theorem, or axiom). Each node is uniquely identified by a cryptographic hash of its canonical content.

### 2.2. Derivation

A **Derivation** is a directed edge in the graph, representing a single logical inference step. It is defined as a tuple `(C, P)` where:
- `C` is the hash of the conclusion Node.
- `P` is a set of hashes of the premise Nodes, `P = {p_1, p_2, ..., p_n}`.

An edge can be said to exist from each `p_i` to `C`.

### 2.3. Per-Cycle DAG (`DAG_cycle`)

A **Per-Cycle DAG** is the proof dependency graph constructed from the set of all derivations produced within a single execution cycle of the experiment runner (`run_uplift_u2.py`).

- **Scope:** Limited to the derivations generated in one cycle `t`.
- **Semantics:** Represents the new knowledge discovered or re-verified in that specific cycle.
- **Properties:** It is a valid DAG, but it may be disconnected and may contain "dangling" premises. A dangling premise is a node that is used as a premise within `DAG_cycle` but is not itself a conclusion within `DAG_cycle`. These premises are assumed to exist either as axioms or as conclusions from previous cycles.

### 2.4. Global DAG (`DAG_global`)

A **Global DAG** represents the cumulative, total knowledge graph aggregated over all cycles of a single experiment run, from `t=0` up to the current cycle `T`.

- **Scope:** Encompasses all unique derivations from all cycles `t <= T`.
- **Semantics:** It is the canonical source of truth for the total state of discovered knowledge for an entire experiment run (e.g., a full `baseline` or `rfl` run). The `dag_footprint` added to the manifest describes this DAG.
- **Construction:** `DAG_global(T) = U(DAG_cycle(t) for t in 0..T)`, where `U` is the graph union operation on the sets of derivations.

## 3. Consolidation Invariants

These invariants must hold true during the construction and evolution of the Global DAG. The runner and analysis tools must enforce or validate them.

- **INV-1 (Cumulative Growth):** The set of nodes and edges in `DAG_global(t)` must be a superset of the nodes and edges in `DAG_global(t-1)`. `Nodes(t-1) ⊆ Nodes(t)` and `Edges(t-1) ⊆ Edges(t)`. Knowledge is only ever added.

- **INV-2 (Hash Uniqueness):** A given node hash must always refer to the same logical statement across all cycles and all experiments. This is guaranteed by the content-based hashing function and is a foundational assumption of the entire system.

- **INV-3 (Derivation Immutability):** A derivation `(C, P)` must be immutable. If two cycles produce a derivation for the same conclusion `C`, the set of premises `P` must be identical. *Correction:* This is too strict. The system may discover multiple proofs for the same theorem. The DAG must therefore support multiple, distinct incoming edge sets for a single conclusion node. The `DAG` data structure (`Dict[str, List[str]]`) must be adjusted to `Dict[str, Set[Tuple[str,...]]]` or a similar structure if multiple proof paths are to be stored distinctly. For the current implementation (`Dict[str, List[str]]`), consolidation implies taking the union of premise lists.

- **INV-4 (Acyclicity):** `DAG_global` must not contain cyclic dependencies (e.g., A -> B -> A). Any consolidation step that would introduce a cycle is considered an error.

## 4. Error States

Violations of the invariants lead to defined error states.

- **ERROR-1 (Cyclic Dependency):** Detected when adding a new derivation `(C, P)` results in a path from `C` back to itself.
  - **Detection:** During consolidation, before adding an edge `p_i -> C`, verify that `C` is not an ancestor of `p_i` in the existing `DAG_global`.
  - **Handling:** Reject the new derivation. Log a critical error. This indicates a flaw in the upstream derivation logic.

- **ERROR-2 (Dangling Premise - Strict Mode):** A "strict" validation mode could define a dangling premise as an error. A premise `p` of a derivation `(C, P)` in `DAG_cycle(t)` is dangling if `p` is not a conclusion in `DAG_global(t-1)` and is not a defined Axiom.
  - **Detection:** For each premise `p`, check `p`'s existence in `Nodes(DAG_global(t-1))` or a predefined set of Axioms.
  - **Handling:** Quarantine the derivation for analysis. This may indicate a log transmission error or a non-deterministic substrate. The current system implicitly treats these as new axioms. A formal system requires explicit axiom definition.

- **ERROR-3 (Hash Collision):** While cryptographically unlikely, a hash collision would be a catastrophic, unrecoverable error.
  - **Detection:** External, manual process of auditing formula text against hashes. Not a runtime check.
  - **Handling:** Halt all processing. Requires manual intervention and re-hashing of the entire formula space.

---

## 5. DAG Evolution Metrics (Proposal)

To monitor the growth and evolution of the `DAG_global` over the course of an experiment, the following metrics should be computed at the end of each cycle `t`.

- **`Nodes(t)`:** Total number of unique nodes in `DAG_global(t)`.
- **`Edges(t)`:** Total number of unique derivations (edges) in `DAG_global(t)`.
- **`ΔNodes(t)`:** `Nodes(t) - Nodes(t-1)`. The number of new theorems/formulas discovered in cycle `t`.
- **`ΔEdges(t)`:** `Edges(t) - Edges(t-1)`. The number of new proof steps discovered in cycle `t`.
- **`MaxDepth(t)`:** The maximum depth of any node in `DAG_global(t)`, as computed by `ChainAnalyzer`.
- **`ΔMaxDepth(t)`:** `MaxDepth(t) - MaxDepth(t-1)`. The change in the longest proof chain.
- **`GlobalBranchingFactor(t)`:** `Edges(t) / Nodes(t)`. The average number of premises per node across the entire global DAG.
- **`CycleBranchingFactor(t)`:** The average number of premises for derivations discovered *only* in `DAG_cycle(t)`. Measures the complexity of "new" discoveries.
- **`NewAxioms(t)`:** The number of new dangling premises introduced in cycle `t` that are not defined as conclusions anywhere in `DAG_global(t)`.

## 6. Future Integration Plan: Graph Persistence in a Database

The current model of rebuilding the entire Global DAG in memory for each run is not scalable for cross-experiment analysis or large-scale production use. A persistent database is the required next step.

### Phase A: Technology Evaluation & Schema Design

1.  **Evaluate Database Technologies:**
    -   **Graph Databases (e.g., Neo4j, ArangoDB):** **(Recommended)** The most natural fit. The data model maps directly to nodes and edges. Optimized for deep traversal queries, pathfinding, and other graph algorithms.
    -   **Relational Databases (e.g., PostgreSQL):** Feasible, but requires representing the graph in tables (e.g., `nodes`, `edges`). Complex traversals require expensive recursive SQL queries (CTEs).
2.  **Define Schema:**
    -   **For Graph DB:**
        -   `(:Formula {hash: string, text: string})`
        -   `(:Derivation {id: string})`
        -   Relationships: `(:Formula)-[:IS_PREMISE_FOR]->(:Derivation)`, `(:Derivation)-[:PRODUCES]->(:Formula)`
    -   **For Relational DB:**
        -   `formulas` table: `hash (PK), text`
        -   `derivations` table: `derivation_id (PK)`
        -   `derivation_premises` table: `derivation_id (FK), premise_hash (FK)`
        -   `derivation_conclusions` table: `derivation_id (FK), conclusion_hash (FK)`

### Phase B: Runtime Integration

1.  **Modify Experiment Runner:** Update `run_uplift_u2.py` to write to the database at the end of each cycle.
2.  **Transactional Writes:** Each cycle's data (new nodes and derivations) should be written within a single database transaction to ensure atomicity and maintain the consistency invariants.
3.  **Error Handling:** Implement robust error handling for database connection issues and constraint violations (e.g., attempting to add a node with a hash that already exists but with different content).

### Phase C: Query Layer

1.  **Develop a `GraphDBProvider`:** Create a new Python module that provides a high-level API for interacting with the database.
2.  **API Methods:** This provider should expose methods like:
    -   `get_depth(hash)`
    -   `get_ancestors(hash)`
    -   `get_descendants(hash)`
    -   `get_longest_path(hash)`
3.  **Deprecate `ChainAnalyzer` for live queries:** The `ChainAnalyzer` would be relegated to offline analysis of static log files. The `GraphDBProvider` would become the source of truth for all runtime queries.

### Phase D: Data Migration

1.  **Create Migration Script:** Write a standalone script to parse all existing experiment manifests (`manifest_v2_*.json`).
2.  **Backfill the Database:** The script will iterate through all historical `ht_series` data, extract the derivations, and populate the new database, allowing for large-scale, cross-experiment analysis of all past results.
