# METRIC INTEGRATION CONSISTENCY SPECIFICATION

**Version**: 1.0  
**Author**: Gemini D, Metric Integration Analyst  
**Status**: DRAFT  

This document specifies the data contracts, structural invariants, and error conditions for the Phase II Metric Engine integration, specifically focusing on the flow from derivation logs to the `chain_length` success metric. Its purpose is to guarantee that the metric evaluation pipeline is robust, deterministic, and verifiable.

---

## 1. Derivations Log Data Contract (JSONL)

The canonical input for chain analysis is a JSONL log file where each line is a JSON object representing a single **Derivation**. Each derivation object must adhere to the following contract:

```json
{
  "hash": "string",
  "text": "string (optional)",
  "premises": ["string", ...]
}
```

| Key        | Type           | Description                                                                                                   | Required |
| :--------- | :------------- | :------------------------------------------------------------------------------------------------------------ | :------- |
| `hash`     | `string`       | The unique SHA-256 hash identifying this derivation.                                                            | Yes      |
| `text`     | `string`       | Optional. The human-readable text of the derived formula.                                                     | No       |
| `premises` | `Array<string>` | A list of hashes corresponding to the direct parent derivations used to produce this one. An empty list `[]` signifies an axiom or a starting point in the graph. | Yes      |

### Example JSONL line:
```json
{"hash": "h4", "text": "...", "premises": ["h2"]}
```

---

## 2. Structural Invariants

The consistency of the `chain_length` metric relies on a series of invariants that link the raw `derivations` data to the final metric output.

### Invariant 1: Graph Isomorphism (Derivations → ChainAnalyzer)
The `ChainAnalyzer` constructs an internal dependency graph that is isomorphic to the directed acyclic graph (DAG) defined by the `derivations` log.

- **Nodes**: Each unique `hash` in the `derivations` log corresponds to a single node in the `ChainAnalyzer`'s internal graph.
- **Edges**: For each derivation `D`, a directed edge exists from `p` to `D.hash` for every premise `p` in `D.premises`. Traversal for depth analysis proceeds "backwards" from a hash to its premises.

### Invariant 2: Recursive Depth Definition (`ChainAnalyzer` Logic)
The depth of any node `h` in the graph is defined by the recursive function:
`depth(h) = 1 + max(depth(p) for p in premises(h))`
with the base cases:
- `depth(h_axiom) = 1` for any `h` with empty `premises`.
- `depth(h_dangling) = 1` for any premise `h` not found in the set of derivation hashes (a "dangling edge").

The `ChainAnalyzer.get_depth(h)` method is a direct, memoized implementation of this function. This guarantees that the calculated depth is a deterministic function of the graph structure.

### Invariant 3: Data Flow Integrity (`ChainAnalyzer` → `compute_metric`)
The `compute_metric` function acts as the final link in the chain, ensuring data integrity.

- When `kind='chain_length'`, `compute_metric` **must** receive a `result` object in its `kwargs`. This object must contain a `derivations` key holding the list of derivation objects compliant with the data contract.
- `compute_metric` instantiates `ChainAnalyzer` with these `derivations`.
- The final `target_chain_length` in the metric's output dictionary is the direct, unmodified return value of `analyzer.get_depth(chain_target_hash)`.
- If `chain_target_hash` is not in the `verified_hashes` set, the metric **must** report `success=False` and `metric_value=0.0`, regardless of the chain's potential depth.

---

## 3. Metric Integration Error Classes (METINT)

| Code      | Severity | Description                                                                                              | Component Affected     |
| :-------- | :------- | :------------------------------------------------------------------------------------------------------- | :--------------------- |
| `METINT-1`  | Critical | `result` object or `result.derivations` key is missing when `kind='chain_length'`.                       | `compute_metric`       |
| `METINT-2`  | Critical | A derivation object in the log is missing the required `hash` key.                                     | `ChainAnalyzer`        |
| `METINT-3`  | Critical | A derivation object in the log is missing the required `premises` key.                                 | `ChainAnalyzer`        |
| `METINT-4`  | Warning  | `chain_target_hash` specified in the curriculum does not exist in the derivation log.                    | `compute_metric`       |
| `METINT-5`  | Critical | Dependency `ChainAnalyzer` could not be imported or instantiated.                                        | `compute_metric`       |
_METINT-6_ to _METINT-20_ are reserved for future use.

---

## 4. Derivation-to-Metric Truth Table

This table describes the expected metric output for specific structural conditions in the derivation data when calling `compute_metric(kind='chain_length', ...)` and assuming the target `h` is verified.

| Condition in Derivations                                   | `ChainAnalyzer.get_depth(h)` Result | `compute_metric` Output (`success`, `value`) | Notes                                                                                                    |
| :--------------------------------------------------------- | :---------------------------------- | :------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| `h` has `premises: []`                                     | `1`                                 | `(True, 1.0)` if `min_len<=1`                | Base case for an axiom or starting formula.                                                              |
| `h` has `premises: ["p1"]`, where `depth(p1)=3`            | `4`                                 | `(True, 4.0)` if `min_len<=4`                | Standard recursive step.                                                                                 |
| `h` is not in `verified_hashes` set                        | `get_depth` is not called by metric   | `(False, 0.0)`                               | Verification status is a hard prerequisite for success and a non-zero metric value.                    |
| `h` has `premises: ["p_dangle"]`, where `p_dangle` is not a known hash | `2`                                 | `(True, 2.0)` if `min_len<=2`                | A dangling premise is treated as an axiom of depth 1. `depth(h) = 1 + depth(p_dangle) = 1 + 1`.         |
| Derivations form a cycle `h1->h2->h1`, `h=h2`               | Terminates and returns `2`            | `(True, 2.0)` if `min_len<=2`                | The memoization in `get_depth` prevents infinite recursion and produces a stable, though arbitrary, depth. |
| `chain_target_hash` is not found in the derivation log     | `get_depth` on it returns `1`         | `(False, 0.0)`                               | The target is treated as a dangling node of depth 1, but since it's not verified, value is 0.      |

---

## 5. Proposed Integration Stress Test Suite

To ensure the robustness of the entire pipeline, a suite of stress tests using synthetically generated derivation logs is proposed.

### Test Harness
A Python script (`tests/experiments/test_metric_stress.py`) will be created. It will contain a generator function `generate_synthetic_derivations(**params)` capable of creating complex derivation structures.

### Generator Parameters:
- `num_nodes: int`: Total number of derivations.
- `max_depth: int`: The maximum depth of any chain.
- `max_fan_out: int`: The maximum number of premises a node can have.
- `cycle_injection_prob: float`: Probability of creating a cycle by linking a node to one of its ancestors.
- `dangling_edge_prob: float`: Probability of a premise being a random hash not corresponding to any node.

### Proposed Test Cases:

1.  **`test_max_recursion_depth`**:
    - **Generator Params**: `num_nodes=2000`, `max_depth=1999`, `max_fan_out=1`.
    - **Objective**: Generate a single, very deep linked list (`h2000->h1999...->h1`).
    - **Assert**: Ensure the `get_depth` call completes without hitting Python's recursion limit and correctly calculates the depth.

2.  **`test_graph_isomorphism`**:
    - **Generator Params**: `num_nodes=500`, `max_fan_out=5`.
    - **Objective**: Generate a complex but valid DAG.
    - **Assert**: Manually calculate the depth of a few key nodes and assert that `ChainAnalyzer.get_depth()` returns the same values.

3.  **`test_cycle_termination`**:
    - **Generator Params**: `num_nodes=100`, `cycle_injection_prob=0.1`.
    - **Objective**: Generate a graph guaranteed to contain cycles.
    - **Assert**: The `get_depth` function must terminate successfully (i.e., not hang or crash). The exact depth value is less important than termination.

4.  **`test_dangling_edge_behavior`**:
    - **Generator Params**: `num_nodes=100`, `dangling_edge_prob=0.5`.
    - **Objective**: Generate a graph with many dangling premises.
    - **Assert**: Verify that the calculated depths match the behavior specified in the Truth Table (i.e., dangling premises are treated as having depth 1).

5.  **`test_malformed_log_resilience`**:
    - **Objective**: Generate a JSONL file where lines are missing `hash` or `premises` keys.
    - **Assert**: Ensure that `ChainAnalyzer` or `compute_metric` raises the appropriate `METINT` error (or a `KeyError`/`TypeError`) gracefully, rather than crashing unpredictably.
