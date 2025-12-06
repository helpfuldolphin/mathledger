# Proof DAG Invariants

> **STATUS: INTERNAL DIAGNOSTIC SPEC / TOOLING FOR PHASE II.**
> **Not part of Evidence Pack v1 claims.**
>
> This document describes invariants that the DAG *should* satisfy and tooling
> that *could* verify them. As of Evidence Pack v1, no full historical audit
> of the proof DAG has been performed. The First Organism closed-loop test
> validates lineage for a single derivation path; it does not constitute a
> comprehensive DAG audit.
>
> **RFL Logs Out of Scope:** The RFL (Reflective Feedback Loop) logs from
> 50-cycle and 330-cycle runs do not create any DAG entries. RFL operates
> as a standalone refinement harness that emits JSONL logs but does not
> write to `proof_parents` or any ledger tables. Therefore, RFL logs are
> not subject to DAG invariants and are outside the scope of this spec.

This document specifies the complete set of graph invariants, normalization rules,
canonical ordering, and classification logic for the MathLedger proof dependency graph.

## Table of Contents

1. [Overview](#overview)
2. [Graph Invariants](#graph-invariants)
3. [Normalization Rules](#normalization-rules)
4. [Canonical Ordering of Parents](#canonical-ordering-of-parents)
5. [Deterministic Visit Rules](#deterministic-visit-rules)
6. [Root/Leaf Classification Logic](#rootleaf-classification-logic)
7. [Validation Levels](#validation-levels)
8. [Sentinel Issues](#sentinel-issues)
9. [Phase II: RFL-Driven DAG Updates (Design Only)](#phase-ii-rfl-driven-dag-updates-design-only)
10. [Phase II RFL DAG Footprint](#phase-ii-rfl-dag-footprint)
    - [Chain Dependency Invariants](#chain-dependency-invariants)
    - [Multi-Goal Proof Set Invariants](#multi-goal-proof-set-invariants)
    - [DAG Evolution Constraints During U2 Experiments](#dag-evolution-constraints-during-u2-experiments)

---

## Overview

The Proof DAG is a directed acyclic graph where:
- **Nodes** represent mathematical statements (identified by `statement_id` or `hash`)
- **Edges** represent derivation dependencies (parent → child)
- Each edge belongs to a **proof** that justifies the derivation

The DAG must satisfy strict invariants to ensure logical soundness and data integrity.

---

## Graph Invariants

### INV-1: Acyclicity

**Statement**: The DAG must contain no directed cycles.

**Rationale**: A cycle would imply circular reasoning—a statement cannot depend
(transitively) on itself for its own proof.

**Detection Algorithm**: Kahn's algorithm (topological sort)
1. Compute in-degree for each node
2. Initialize queue with nodes having in-degree = 0
3. While queue is non-empty:
   - Dequeue node, decrement in-degree of its children
   - Enqueue children with in-degree = 0
4. If visited count < total nodes, cycle exists

**Cycle Nodes**: Nodes with non-zero in-degree after processing are part of cycles.

**Severity**: CRITICAL — blocks DAG validation

---

### INV-2: No Self-Loops

**Statement**: No edge may have identical child and parent identifiers.

**Formal**: For all edges `e`: `e.child_key != e.parent_key`

**Detection**:
```python
if (child_statement_id == parent_statement_id and child_statement_id is not None):
    # Self-loop by ID
if (child_hash == parent_hash and child_hash is not None):
    # Self-loop by hash
```

**Severity**: CRITICAL — a degenerate cycle

---

### INV-3: No Duplicate Edges

**Statement**: Each `(proof_id, child_key, parent_key)` triple must be unique.

**Rationale**: Recording the same dependency multiple times indicates a bug in the
derivation pipeline or data corruption.

**Detection**: Group edges by `(proof_id, child_key, parent_key)` and flag groups
with count > 1.

**Note**: Multiple edges with different `edge_index` values for the same triple
are still duplicates—`edge_index` is for ordering, not uniqueness.

**Severity**: ERROR — indicates data corruption

---

### INV-4: Hash/ID Consistency

**Statement**: If a `statement_id` appears in multiple edges, it must always map
to the same `hash`.

**Formal**: For all edges `e1, e2`:
```
e1.child_statement_id == e2.child_statement_id
  => e1.child_hash == e2.child_hash
```

**Detection**: Build a map `statement_id -> hash` during edge ingestion. Flag
conflicts when a new edge provides a different hash for an existing ID.

**Reported Issues**:
- `child_hash_mismatch`: Child ID maps to multiple hashes
- `parent_hash_mismatch`: Parent ID maps to multiple hashes

**Severity**: CRITICAL — indicates hash collision or data corruption

---

### INV-5: Complete Edges

**Statement**: Every edge must have resolvable child and parent identifiers.

**Requirements**:
- At least one of: `child_statement_id`, `child_hash`, or resolvable via `proof_id`
- At least one of: `parent_statement_id`, `parent_hash`

**Detection**:
```python
if child_statement_id is None and child_hash is None:
    # Incomplete edge (missing child)
if parent_statement_id is None and parent_hash is None:
    # Incomplete edge (missing parent)
```

**Severity**: ERROR — breaks lineage queries

---

### INV-6: Edge Index Ordering

**Statement**: For each proof, `edge_index` values should form a sequential
integer sequence starting from 0.

**Formal**: For proof `p` with edges `E_p`:
```
sorted([e.edge_index for e in E_p]) == [0, 1, 2, ..., len(E_p)-1]
```

**Detection**: Collect edge indices per proof and verify sequence.

**Rationale**: Provides deterministic parent ordering for proof reconstruction.

**Severity**: WARNING — may indicate incomplete edge recording

---

### INV-7: Referential Integrity (Statements)

**Statement**: All `parent_statement_id` and `child_statement_id` values must
reference existing statements in the `statements` table.

**Detection**:
```sql
SELECT COUNT(*)
FROM proof_parents pp
LEFT JOIN statements s ON s.id = pp.parent_statement_id
WHERE s.id IS NULL AND pp.parent_statement_id IS NOT NULL
```

**Severity**: ERROR — dangling references indicate incomplete transactions

---

### INV-8: Referential Integrity (Proofs)

**Statement**: All `proof_id` values must reference existing proofs in the
`proofs` table.

**Detection**:
```sql
SELECT COUNT(*)
FROM proof_parents pp
LEFT JOIN proofs p ON p.id = pp.proof_id
WHERE pp.proof_id IS NOT NULL AND p.id IS NULL
```

**Severity**: ERROR — orphaned edges without corresponding proofs

---

## Normalization Rules

### Node Key Normalization

Nodes are identified by a canonical key derived from available identifiers:

```python
def node_key(statement_id: Optional[int], statement_hash: Optional[str]) -> str:
    if statement_id is not None:
        return f"id:{statement_id}"
    if statement_hash:
        return f"hash:{statement_hash}"
    return None  # Invalid/incomplete
```

**Priority**: ID takes precedence over hash when both are available.

### Edge Normalization

Before storage or comparison, edges are normalized to a canonical form:

1. **Coerce types**: Convert memoryview/bytes to strings, ensure int for IDs
2. **Resolve hashes**: If ID is present but hash is missing, resolve via `statements` table
3. **Compute node keys**: Generate canonical keys for child and parent

### Hash Normalization

Statement hashes must be normalized before use:

1. **Encoding**: UTF-8 encoded, lowercase hexadecimal
2. **Algorithm**: SHA-256 of canonicalized formula text
3. **Length**: 64 hexadecimal characters (256 bits)

---

## Canonical Ordering of Parents

When a proof has multiple parent statements (e.g., Modus Ponens with major and
minor premises), parents are ordered canonically for deterministic reconstruction.

### Sort Key Definition

```python
def edge_sort_key(edge: ProofEdge) -> tuple:
    return (
        edge.proof_id if edge.proof_id is not None else -1,
        edge.child_statement_id if edge.child_statement_id is not None else -1,
        edge.child_hash or "",
        edge.parent_statement_id if edge.parent_statement_id is not None else -1,
        edge.parent_hash or "",
        edge.edge_index,
    )
```

### Ordering Priority

1. **proof_id**: Primary grouping by proof
2. **child_statement_id**: Secondary grouping by derived statement
3. **child_hash**: Fallback for hash-only schemas
4. **parent_statement_id**: Parent ordering within proof
5. **parent_hash**: Fallback for hash-only schemas
6. **edge_index**: Explicit ordering annotation

### Edge Index Semantics

The `edge_index` column provides explicit parent ordering:

| edge_index | Meaning (for Modus Ponens) |
|------------|----------------------------|
| 0          | Major premise (p → q)      |
| 1          | Minor premise (p)          |

For axiom instantiations with single parent (schema), `edge_index = 0`.

---

## Deterministic Visit Rules

Graph traversal algorithms must produce deterministic results for reproducibility.

### BFS Traversal (Ancestors/Descendants)

```python
def ancestors(child_hash: str, max_depth: Optional[int] = None) -> List[str]:
    seen: Dict[str, int] = {}  # hash -> depth
    queue: Deque[Tuple[str, int]] = deque([(child_hash, 0)])
    result: List[str] = []

    while queue:
        node_hash, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue

        for parent_hash in sorted(parents_of(node_hash)):  # Sorted for determinism
            next_depth = depth + 1
            if parent_hash not in seen or next_depth < seen[parent_hash]:
                seen[parent_hash] = next_depth
                result.append(parent_hash)
                queue.append((parent_hash, next_depth))

    return result
```

**Determinism Guarantees**:
1. Parents are processed in sorted order
2. BFS ensures level-by-level discovery
3. Result order reflects discovery order

### Topological Order

For cycle detection, nodes are processed by in-degree:

1. Initialize: all nodes with in-degree = 0
2. Process: lexicographically smallest node from queue
3. Update: decrement children's in-degrees

---

## Root/Leaf Classification Logic

### Root Nodes (Axioms)

**Definition**: A node with no incoming edges (in-degree = 0).

```python
def is_root(node_hash: str) -> bool:
    return len(parents_of_hash(node_hash)) == 0
```

**Characteristics**:
- Represents an axiom or axiom instantiation
- Has no proof dependencies
- May or may not have a proof record (axioms are self-evident)

**Detection Query**:
```sql
SELECT s.hash
FROM statements s
WHERE NOT EXISTS (
    SELECT 1 FROM proof_parents pp
    WHERE pp.child_hash = s.hash
)
```

### Leaf Nodes (Derived Theorems)

**Definition**: A node with no outgoing edges (out-degree = 0).

```python
def is_leaf(node_hash: str) -> bool:
    return len(children_of_hash(node_hash)) == 0
```

**Characteristics**:
- Not yet used as a premise in any derivation
- Represents the "frontier" of current derivations
- May become non-leaf when used in future proofs

**Detection Query**:
```sql
SELECT s.hash
FROM statements s
WHERE NOT EXISTS (
    SELECT 1 FROM proof_parents pp
    WHERE pp.parent_hash = s.hash
)
```

### Interior Nodes

**Definition**: A node with both incoming and outgoing edges.

```python
def is_interior(node_hash: str) -> bool:
    return not is_root(node_hash) and not is_leaf(node_hash)
```

### Node Statistics

| Metric | Description |
|--------|-------------|
| `root_count` | Number of axiom/root nodes |
| `leaf_count` | Number of frontier nodes |
| `interior_count` | Number of intermediate derived nodes |
| `max_depth` | Longest path from any root to any leaf |
| `avg_in_degree` | Average number of parents per node |
| `avg_out_degree` | Average number of children per node |

---

## Validation Levels

### Level 1: In-Memory Validation

Checks invariants using only loaded edge data:
- INV-1: Cycle detection
- INV-2: Self-loop detection
- INV-3: Duplicate edge detection
- INV-4: Hash/ID consistency
- INV-5: Edge completeness

**API**: `ProofDag.validate() -> ProofDagValidationReport`

### Level 2: Database Validation

Adds database-level referential integrity checks:
- INV-6: Edge index ordering (if column exists)
- INV-7: Missing parent statements
- INV-7: Missing child statements
- INV-8: Missing proofs

**API**: `ProofDagRepository.validate() -> ProofDagValidationReport`

### Level 3: Organism Lineage Validation

Extended checks for First Organism tests:
- Lineage completeness (expected parents present)
- Ancestor chain verification
- Descendant chain verification
- Proof method verification

**API**: `validate_organism_lineage_from_db() -> OrganismLineageReport`

---

## Sentinel Issues

When schema lacks columns needed for validation, sentinel issues are emitted:

| Sentinel | Meaning | Implication |
|----------|---------|-------------|
| `missing_parents_unverified` | Cannot verify parent references | INV-7 skipped |
| `missing_children_unverified` | Cannot verify child references | INV-7 skipped |
| `missing_proofs_unverified` | Cannot verify proof references | INV-8 skipped |
| `duplicate_edges_unverified` | Cannot detect duplicate edges | INV-3 skipped |

**Handling**: Sentinel issues indicate incomplete validation, not failures.
Production systems should ensure full schema to avoid sentinels.

---

## Phase II: RFL-Driven DAG Updates (Design Only)

> **STATUS: DESIGN ONLY — NOT IMPLEMENTED**
>
> This section describes how future RFL (Reflective Feedback Loop) uplift
> experiments *could* write into the proof DAG. **No current RFL run satisfies
> these conditions.** The 50-cycle and 330-cycle RFL logs remain JSONL-only
> and do not create DAG entries.

## Phase II RFL DAG Footprint

> **PHASE II — NOT RUN IN PHASE I**
>
> This section documents Phase II U2 experiment DAG footprint requirements.
> All invariants below are additive to Phase I and apply only to U2 runs.

Phase I DAG invariants remain unchanged; Phase II DAG checks are observational only.

---

### Chain Dependency Invariants

These invariants govern how U2 experiment proof chains must maintain structural integrity.

#### INV-P2-CD-1: Chain Completeness

**Statement**: Every U2-generated proof chain must have unbroken ancestry back to
a registered axiom or previously-verified statement within the same theory.

**Formal**: For all statements `s` in U2 experiment `E`:
```
ancestors(s) ∩ (axioms(E.theory) ∪ verified_statements(E.theory)) ≠ ∅
```

**Detection**:
1. For each U2-generated statement, trace full ancestry
2. Verify at least one ancestor is in `{axioms} ∪ {pre-experiment verified}`
3. Flag orphaned chains with no valid root

**Severity**: CRITICAL — orphaned chains invalidate experiment results

---

#### INV-P2-CD-2: Chain Dependency Ordering

**Statement**: Within a U2 cycle, derived statements must respect temporal ordering
of their dependencies—a statement cannot depend on another statement derived later
in the same cycle.

**Formal**: For statements `s1`, `s2` in cycle `C`:
```
s1 ∈ parents(s2) => derivation_order(s1, C) < derivation_order(s2, C)
```

**Detection**:
```python
def check_cycle_ordering(cycle_statements: List[Statement]) -> List[Violation]:
    order_map = {s.hash: i for i, s in enumerate(cycle_statements)}
    violations = []
    for s in cycle_statements:
        for parent_hash in parents_of(s.hash):
            if parent_hash in order_map and order_map[parent_hash] >= order_map[s.hash]:
                violations.append((s.hash, parent_hash, "temporal_violation"))
    return violations
```

**Severity**: CRITICAL — temporal violations indicate non-deterministic derivation

---

#### INV-P2-CD-3: Cross-Cycle Dependency Bounds

**Statement**: Dependencies spanning multiple U2 cycles must not exceed a maximum
cycle distance, preventing unbounded lookback that could mask stale reasoning.

**Configuration**: `MAX_CROSS_CYCLE_DISTANCE` (default: 10 cycles)

**Formal**: For statement `s` in cycle `C_n` with parent `p` from cycle `C_m`:
```
|n - m| ≤ MAX_CROSS_CYCLE_DISTANCE
```

**Detection**:
```sql
SELECT
    child.cycle_number AS child_cycle,
    parent.cycle_number AS parent_cycle,
    ABS(child.cycle_number - parent.cycle_number) AS distance
FROM u2_proof_edges e
JOIN u2_statements child ON e.child_hash = child.hash
JOIN u2_statements parent ON e.parent_hash = parent.hash
WHERE ABS(child.cycle_number - parent.cycle_number) > :max_distance
```

**Severity**: WARNING — excessive lookback may indicate suboptimal policy learning

---

### Multi-Goal Proof Set Invariants

These invariants apply to U2 experiments with multiple concurrent derivation goals.

#### INV-P2-MG-1: Goal Attribution Completeness

**Statement**: Every U2-derived statement in a multi-goal experiment must be
attributed to at least one goal from the experiment's goal set.

**Schema**:
```sql
CREATE TABLE u2_goal_attributions (
    id SERIAL PRIMARY KEY,
    statement_hash TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    attribution_weight FLOAT DEFAULT 1.0,
    UNIQUE (statement_hash, goal_id, experiment_id)
);
```

**Detection**:
```sql
SELECT s.hash FROM u2_statements s
WHERE s.experiment_id = :exp_id
AND NOT EXISTS (
    SELECT 1 FROM u2_goal_attributions ga
    WHERE ga.statement_hash = s.hash AND ga.experiment_id = :exp_id
)
```

**Severity**: ERROR — unattributed statements cannot contribute to goal metrics

---

#### INV-P2-MG-2: Goal Conflict Detection

**Statement**: When a statement contributes to multiple goals, any logical
conflicts between those goals must be flagged for manual review.

**Conflict Definition**: Goals `G1` and `G2` conflict if:
- Their target formulas are contradictory (one is negation of other)
- Their required axiom sets are mutually exclusive
- Their slice constraints are incompatible

**Detection**:
```python
@dataclass
class GoalConflict:
    statement_hash: str
    goal_ids: List[str]
    conflict_type: str  # 'logical' | 'axiom_exclusion' | 'slice_incompatible'

def detect_conflicts(experiment_id: str) -> List[GoalConflict]:
    conflicts = []
    multi_goal_stmts = get_multi_goal_statements(experiment_id)
    for stmt in multi_goal_stmts:
        goal_pairs = combinations(stmt.goal_ids, 2)
        for g1, g2 in goal_pairs:
            if goals_conflict(g1, g2):
                conflicts.append(GoalConflict(stmt.hash, [g1, g2], conflict_type(g1, g2)))
    return conflicts
```

**Severity**: WARNING — conflicts require human review but don't invalidate DAG

---

#### INV-P2-MG-3: Goal Progress Monotonicity

**Statement**: Within a U2 experiment, progress toward each goal (measured by
proof chain depth toward goal formula) must be non-decreasing across cycles.

**Rationale**: Ensures policy updates don't regress on previously achieved progress.

**Metrics**:
```python
@dataclass
class GoalProgress:
    goal_id: str
    cycle: int
    max_chain_depth: int
    statements_toward_goal: int
    closest_statement_hash: Optional[str]
```

**Detection**:
```python
def check_monotonicity(goal_id: str, experiment_id: str) -> List[Regression]:
    progress = get_goal_progress_by_cycle(goal_id, experiment_id)
    regressions = []
    for i in range(1, len(progress)):
        if progress[i].max_chain_depth < progress[i-1].max_chain_depth:
            regressions.append(Regression(
                goal_id=goal_id,
                cycle=progress[i].cycle,
                previous_depth=progress[i-1].max_chain_depth,
                current_depth=progress[i].max_chain_depth
            ))
    return regressions
```

**Severity**: OBSERVATIONAL — regressions indicate policy instability

---

### DAG Evolution Constraints During U2 Experiments

These constraints govern how the DAG may evolve during active U2 experiments.

#### INV-P2-EV-1: Experiment Isolation

**Statement**: U2 experiment DAG modifications must be isolated from the main
production DAG until experiment completion and validation.

**Implementation**:
```sql
-- U2 experiments write to shadow tables
CREATE TABLE u2_proof_parents (
    LIKE proof_parents INCLUDING ALL,
    experiment_id TEXT NOT NULL,
    merged_to_main BOOLEAN DEFAULT FALSE,
    merge_timestamp TIMESTAMPTZ
);

CREATE INDEX idx_u2_pp_experiment ON u2_proof_parents(experiment_id);
CREATE INDEX idx_u2_pp_unmerged ON u2_proof_parents(experiment_id) WHERE NOT merged_to_main;
```

**Merge Protocol**:
1. Experiment completes all cycles
2. Full DAG validation (INV-1 through INV-8) on shadow tables
3. Phase II invariant validation (INV-P2-*)
4. Atomic merge to main tables with `merged_to_main = TRUE`

**Severity**: CRITICAL — premature merges violate experiment integrity

---

#### INV-P2-EV-2: Concurrent Experiment Non-Interference

**Statement**: Multiple concurrent U2 experiments must not share intermediate
derivation state, preventing cross-contamination of policy learning.

**Formal**: For experiments `E1`, `E2` running concurrently:
```
(statements(E1) ∩ statements(E2)) ⊆ (shared_axioms ∪ pre_experiment_verified)
```

**Detection**:
```sql
SELECT s1.hash, s1.experiment_id, s2.experiment_id
FROM u2_statements s1
JOIN u2_statements s2 ON s1.hash = s2.hash
WHERE s1.experiment_id != s2.experiment_id
AND s1.is_derived = TRUE  -- Not an axiom or pre-existing
AND s2.is_derived = TRUE
```

**Severity**: CRITICAL — shared derived statements invalidate both experiments

---

#### INV-P2-EV-3: DAG Snapshot Consistency

**Statement**: At experiment start, a consistent snapshot of the relevant DAG
subgraph must be captured and remain immutable for the experiment duration.

**Snapshot Schema**:
```sql
CREATE TABLE u2_dag_snapshots (
    id SERIAL PRIMARY KEY,
    experiment_id TEXT NOT NULL UNIQUE,
    snapshot_timestamp TIMESTAMPTZ NOT NULL,
    theory_id INT NOT NULL,
    statement_count INT NOT NULL,
    edge_count INT NOT NULL,
    root_merkle_hash TEXT NOT NULL,
    snapshot_manifest JSONB NOT NULL
);
```

**Immutability Check**:
```python
def verify_snapshot_immutability(experiment_id: str) -> bool:
    snapshot = get_snapshot(experiment_id)
    current_merkle = compute_merkle(snapshot.theory_id, snapshot.snapshot_timestamp)
    return current_merkle == snapshot.root_merkle_hash
```

**Severity**: CRITICAL — snapshot mutation invalidates experiment baseline

---

#### INV-P2-EV-4: Rollback Capability

**Statement**: Any U2 experiment must be fully rollback-capable, removing all
experiment-generated DAG entries without affecting pre-experiment state.

**Rollback Procedure**:
```sql
-- Rollback experiment (transaction)
BEGIN;

-- Remove goal attributions
DELETE FROM u2_goal_attributions WHERE experiment_id = :exp_id;

-- Remove experiment edges
DELETE FROM u2_proof_parents WHERE experiment_id = :exp_id;

-- Remove experiment statements
DELETE FROM u2_statements WHERE experiment_id = :exp_id;

-- Mark experiment as rolled back
UPDATE u2_experiments
SET status = 'rolled_back', rollback_timestamp = NOW()
WHERE experiment_id = :exp_id;

COMMIT;
```

**Verification**:
```python
def verify_clean_rollback(experiment_id: str) -> bool:
    # No orphaned references should remain
    return (
        count_experiment_statements(experiment_id) == 0 and
        count_experiment_edges(experiment_id) == 0 and
        count_experiment_attributions(experiment_id) == 0
    )
```

**Severity**: CRITICAL — incomplete rollback corrupts DAG state

---

#### INV-P2-EV-5: Slice-Scoped Mutations

**Statement**: U2 experiment DAG mutations must respect slice boundaries defined
in the experiment configuration—no derivations outside the permitted slice.

**Slice Definition** (from PREREG_UPLIFT_U2.yaml):
```yaml
slice_constraints:
  max_atoms: 5
  max_depth: 6
  permitted_connectives: ["→", "∧", "∨", "¬"]
  theory_id: 1  # Propositional Logic
```

**Detection**:
```python
def check_slice_compliance(statement: Statement, slice_config: SliceConfig) -> bool:
    formula = parse(statement.text)
    return (
        count_atoms(formula) <= slice_config.max_atoms and
        formula_depth(formula) <= slice_config.max_depth and
        all(c in slice_config.permitted_connectives for c in connectives(formula))
    )
```

**Severity**: ERROR — out-of-slice derivations violate experiment preregistration

---

### INV-P2-1: Chain Success (Minimum Depth Requirement)

**Statement**: Any RFL-derived statement marked as "successful" within a Phase II
experiment must belong to a DAG chain of at least a specified minimum depth,
originating from a root node within the same experiment context.

**Rationale**: Ensures that successful RFL outputs are not isolated statements
but contribute to a meaningful derivation chain, preventing superficial
"successes" that lack foundational depth. This invariant measures the
compositional rigor of RFL-generated proofs.

**Detection**: For each RFL-originated leaf node flagged as "successful" in its
experiment manifest:
1. Trace its ancestry back to a root node.
2. Calculate the depth of the longest path from a root within the experiment
   context to the successful leaf.
3. Flag if `longest_path_depth < MIN_SUCCESS_CHAIN_DEPTH`.

**Severity**: OBSERVATIONAL — indicates RFL experiment may be generating shallow
proofs, but does not invalidate the DAG itself.

---

### INV-P2-2: Multi-Goal Provenance

**Statement**: RFL-originated statements contributing to Phase II multi-goal
experiments must explicitly link to the multiple goals they address.

**Rationale**: Multi-goal experiments require clear attribution of derived
statements to specific target goals. This invariant ensures that the DAG
reflects the complex provenance in such scenarios, facilitating analysis
of goal achievement and interdependencies.

**Schema Extension (Example)**:
```sql
ALTER TABLE proof_parents ADD COLUMN goal_ids JSONB; -- Array of goal IDs
```

**Detection**: For RFL-originated edges in multi-goal experiments:
1. Check if `goal_ids` column is present and non-empty.
2. Verify that all `goal_ids` correspond to goals defined in the experiment manifest.

**Severity**: OBSERVATIONAL — flags mis-attributed or un-attributed statements
in multi-goal contexts.

---

### INV-P2-3: Per-Cycle DAG Stability

**Statement**: Within a single RFL cycle of a Phase II experiment, the set of
newly introduced DAG edges must exhibit a high degree of internal consistency
and avoid rapid, unmotivated structural changes.

**Rationale**: Rapid, unpredictable shifts in the DAG structure within a single
RFL cycle could indicate instability or noise in the RFL process. This invariant
monitors the "churn" and coherence of the DAG footprint generated by RFL,
providing a diagnostic for the RFL's internal dynamics.

**Metrics (Examples)**:
- **New Edge Cohesion**: Ratio of new edges that connect to other new edges
  within the same cycle versus connecting to pre-existing DAG nodes.
- **Structural Overlap**: Jaccard index or similar metric comparing the graph
  structure generated by cycle `N` vs. cycle `N-1`.
- **Node Reuse Rate**: Frequency with which existing DAG nodes are re-used
  as parents by new RFL-generated edges within a cycle.

**Detection**: Compute and report metrics on `(new_edges_cycle_N)` and
`((new_edges_cycle_N) ∪ (new_edges_cycle_N-1))`.

**Severity**: OBSERVATIONAL — provides insights into RFL process stability;
anomalies suggest deeper RFL issues.

---

### Current State (Phase I)

- RFL operates as a standalone refinement harness
- RFL emits JSONL logs only; no writes to `proof_parents` or ledger tables
- RFL-proposed proofs are not verified by a trusted kernel
- RFL logs are **out of scope** for DAG invariants

### Future Integration Requirements

For an RFL-proposed proof to become a valid DAG entry, it must satisfy:

#### Requirement 1: Trusted Kernel Verification

The proposed proof must be verified by a trusted verification kernel before
DAG insertion:

- **Lean 4**: Full proof term checked by `lake build`
- **Truth Table**: For propositional logic, verified via `tautCheck`
- **Future**: Other verified kernels (Coq, Metamath, etc.)

Verification must complete successfully; RFL proposals that fail verification
are rejected and never enter the DAG.

#### Requirement 2: Dual-Attested Context

Each RFL-originated edge must be accompanied by attested context:

```python
@dataclass
class RflDagContext:
    experiment_id: str          # e.g., "rfl_uplift_2025_06_15"
    cycle_number: int           # Which RFL cycle produced this
    R_t: str                    # Reasoning trace hash
    U_t: str                    # Uncertainty estimate hash
    H_t: str                    # Heuristic context hash
    verifier_result: str        # "lean_success" | "tautcheck_success"
    timestamp: datetime
```

This context must be stored alongside the edge (in a `proof_parents_rfl_context`
table or as JSONB metadata).

#### Requirement 3: Experiment Manifest

RFL experiments that write to the DAG must have a sealed manifest:

```json
{
  "experiment_id": "rfl_uplift_2025_06_15",
  "start_time": "2025-06-15T02:00:00Z",
  "end_time": "2025-06-15T04:30:00Z",
  "total_cycles": 500,
  "proofs_proposed": 127,
  "proofs_verified": 98,
  "proofs_rejected": 29,
  "manifest_hash": "sha256:..."
}
```

### RFL-Specific Invariants

In addition to INV-1 through INV-8, RFL-originated edges must satisfy:

#### INV-RFL-1: Provenance Tag

**Statement**: Every RFL-originated edge must have an `origin` tag.

**Schema Extension**:
```sql
ALTER TABLE proof_parents ADD COLUMN origin TEXT DEFAULT 'derivation_engine';
-- RFL edges: origin = 'rfl_uplift_experiment_X'
```

**Detection**:
```sql
SELECT * FROM proof_parents
WHERE origin LIKE 'rfl_%' AND experiment_manifest_id IS NULL
```

#### INV-RFL-2: No Cycles Introduced

**Statement**: An RFL-proposed edge must not create a cycle when added to the
existing DAG.

**Detection**: Before insertion, run cycle detection on `DAG ∪ {new_edge}`.

**Severity**: CRITICAL — reject the proposed edge

#### INV-RFL-3: Verification Receipt

**Statement**: Every RFL-originated edge must have a linked verification receipt.

**Schema Extension**:
```sql
CREATE TABLE rfl_verification_receipts (
    id SERIAL PRIMARY KEY,
    proof_parent_id BIGINT REFERENCES proof_parents(id),
    verifier TEXT NOT NULL,           -- 'lean4' | 'tautcheck'
    result TEXT NOT NULL,             -- 'success' | 'failure'
    duration_ms INT,
    log_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### INV-RFL-4: Experiment Boundary

**Statement**: RFL edges must reference a valid, sealed experiment manifest.

**Detection**:
```sql
SELECT pp.* FROM proof_parents pp
LEFT JOIN rfl_experiments e ON pp.experiment_id = e.id
WHERE pp.origin LIKE 'rfl_%' AND e.id IS NULL
```

### Auditor Support (Future)

The `ProofDagAuditor` will support an `--include-rfl` mode that:

1. Filters for edges where `origin LIKE 'rfl_%'`
2. Applies INV-RFL-1 through INV-RFL-4 in addition to base invariants
3. Cross-references experiment manifests
4. Validates verification receipts

**This mode is not implemented.** See `tools/proof_dag_audit.py` for the
skeleton design.

### Migration Path

When RFL integration is implemented:

1. Add `origin` column to `proof_parents` (default: `'derivation_engine'`)
2. Create `rfl_experiments` and `rfl_verification_receipts` tables
3. Implement pre-insertion cycle check in RFL writer
4. Implement `--include-rfl` mode in auditor
5. Update this document to remove "Design Only" status

---

## Related Documentation

- [DAG_SPEC.md](./DAG_SPEC.md) — Schema requirements and overview
- [backend/dag/proof_dag.py](../backend/dag/proof_dag.py) — Core implementation
- [tests/test_proof_dag.py](../tests/test_proof_dag.py) — Unit tests
