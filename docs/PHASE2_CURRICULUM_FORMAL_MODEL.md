# PHASE II — Curriculum Formal Model & Correctness Contract
**Version:** 1.0
**Author:** GEMINI B, Curriculum Axiomatization Engineer
**Status:** PROPOSED

## 1. Formal Model of the Phase II Curriculum

This document provides a formal definition of the structure and properties of the MathLedger Phase II curriculum (`curriculum_uplift_phase2.yaml`). Its purpose is to ensure all current and future curriculum slices are sound, verifiable, and consistent.

### 1.1. Slice Definition Grammar (Schema)

A curriculum file is a YAML document with a top-level `version` and a `slices` map. Each entry in the `slices` map is a `slice_name` mapped to a `Slice` object. The schema for a `Slice` is as follows:

```yaml
# Formal Schema for a Slice Object
description: string                       # Human-readable explanation of the slice's purpose.
uplift:
  phase: "II"                             # Must be "II" for this curriculum version.
  not_allowed_in_phase_I: true          # Must be true.
parameters:
  atoms: [string, ...]                    # List of propositional atoms available for formula generation.
  depth_min: int                          # Minimum derivation depth for formulas in the pool.
  depth_max: int                          # Maximum derivation depth for formulas in the pool.
  breadth_max: int                        # Maximum number of premises for any single derivation step.
  total_max: int                          # Maximum total number of formulas in the generated pool.
success_metric:
  kind: string                            # The type of metric used to evaluate success. Must be one of the registered kinds.
  parameters: { ... }                     # A dictionary of parameters specific to the 'kind' of metric.
budget:
  max_candidates_per_cycle: int           # The maximum number of formulas the agent can attempt to verify per cycle.
formula_pool_entries: [string, ...]       # A seed list of formulas or axioms.
```

### 1.2. Inference Rules for a Valid Asymmetric Slice

A slice is considered **valid and asymmetric** if it conforms to the schema above AND the following inference rules hold:

1.  **Rule (Asymmetry Intent):** A slice `S` is asymmetric iff its `success_metric` is designed to reward a specific, non-trivial behavior that is statistically unlikely to be achieved by a random-search baseline policy.
2.  **Rule (Goal-Conditioned Slice):**
    `IF S.success_metric.kind == "goal_hit" THEN S.success_metric.parameters.target_hashes MUST exist AND be a non-empty list of strings.`
3.  **Rule (Density Slice):**
    `IF S.success_metric.kind == "density" THEN S.success_metric.parameters.min_verified MUST exist AND be an integer > 0.`
4.  **Rule (Chain-Depth Slice):**
    `IF S.success_metric.kind == "chain_length" THEN S.success_metric.parameters.min_chain_length MUST exist AND be an integer > 1.`
5.  **Rule (Dependency Slice):**
    `IF S.success_metric.kind == "multi_goal" THEN S.success_metric.parameters.required_goal_hashes MUST exist AND be a non-empty list of strings.`

### 1.3. Monotonicity Proof Sketch

We must ensure that our curriculum can be ordered by difficulty. A curriculum exhibits monotonicity if increasing its complexity parameters leads to a provably harder or more expressive task.

**Claim:** A slice `S'` is monotonically harder than a slice `S` if its parameter space `P'` contains the parameter space `P` of `S`.

**Proof Sketch:**
Let `P = {atoms, depth_max, breadth_max}`. Let `SP(P)` be the search space of all possible valid formulas that can be generated within the constraints of `P`.

1.  **Atom Monotonicity:** If `S.atoms ⊂ S'.atoms`, then any formula constructible with `S.atoms` is also constructible with `S'.atoms`. Therefore, `SP(S) ⊂ SP(S')`.
2.  **Depth Monotonicity:** If `S.depth_max < S'.depth_max`, then any derivation tree of depth `d <= S.depth_max` is also a valid tree of depth `d <= S'.depth_max`. Therefore, `SP(S) ⊂ SP(S')`.
3.  **Breadth Monotonicity:** If `S.breadth_max < S'.breadth_max`, then any derivation step with `b <= S.breadth_max` premises is also a valid step with `b <= S'.breadth_max` premises. Therefore, `SP(S) ⊂ SP(S')`.

**Conclusion:** Since increasing any of these parameters results in a superset of the original search space, the task of finding a specific formula or structure within that space is guaranteed to be at least as hard, and generally harder, due to the larger search space. This allows for the creation of progressively more difficult slices.

### 1.4. `success_metric.kind` to Parameter Schema Mapping

| `kind` | `parameters` Schema | Description |
| :--- | :--- | :--- |
| `goal_hit` | `{ min_goal_hits: int, min_total_verified: int, target_hashes: [string, ...] }` | Success requires hitting a minimum number of specific target formulas. |
| `density` | `{ min_verified: int }` | Success requires verifying a minimum number of formulas within the cycle budget, measuring efficiency. |
| `chain_length` | `{ min_chain_length: int }` | Success requires proving a formula with a derivation chain of at least a minimum length. |
| `multi_goal` | `{ min_each_goal: int, required_goal_hashes: [string, ...] }` | Success requires proving *all* specified goals in the same cycle. |

---

## 2. Slice Correctness Contract

Any new slice added to `curriculum_uplift_phase2.yaml` or future versions must adhere to this contract.

### 2.1. Structure Invariants

These invariants must hold for any individual slice definition.

*   **INV-S-01:** `parameters.depth_min` MUST be less than or equal to `parameters.depth_max`.
*   **INV-S-02:** All integer parameters (`depth_min`, `depth_max`, `breadth_max`, `total_max`, `max_candidates_per_cycle`, etc.) MUST be non-negative.
*   **INV-S-03:** `success_metric.kind` MUST be one of the formally defined kinds in section 1.4.
*   **INV-S-04:** The `parameters` block within `success_metric` MUST conform to the schema defined for its `kind` in section 1.4.
*   **INV-S-05:** `formula_pool_entries` MUST be a non-empty list of strings.

### 2.2. Cross-Slice Invariants

These invariants must hold across the set of all slices in the curriculum file.

*   **INV-C-01:** All `slice_name` keys in the `slices` map MUST be unique.
*   **INV-C-02:** (Recommendation) Slices intended to be compared (e.g., for measuring uplift) should share identical parameters *except* for the specific dimension being varied to create asymmetry.

### 2.3. Forbidden Patterns

The following patterns are explicitly disallowed in any slice definition.

*   **FORBID-01:** A `success_metric` of kind `goal_hit` or `multi_goal` MUST NOT contain an empty or missing `target_hashes` / `required_goal_hashes` list.
*   **FORBID-02:** A `target_hash` or `required_goal_hash` MUST correspond to a formula that is theoretically derivable from the provided `formula_pool_entries` and `parameters`. (Note: This is a semantic requirement that may require an external verifier tool).
*   **FORBID-03:** The `uplift.phase` key MUST NOT be set to `"I"` or any value other than `"II"`.
*   **FORBID-04:** A `min_chain_length` in a `chain_length` metric MUST be `> 1`, as a length of 1 is just an axiom.

### 2.4. Compliance Checklist for New Slices

Before submitting a PR with a new slice, authors must verify the following:

- [ ] **Schema Conformance:** The slice definition is valid YAML and conforms to the grammar in section 1.1.
- [ ] **Asymmetry Rule:** The slice and its success metric adhere to at least one inference rule from section 1.2.
- [ ] **Structural Invariants:** All invariants from section 2.1 are met.
- [ ] **Cross-Slice Invariants:** The new slice does not violate the uniqueness invariant (INV-C-01).
- [ ] **Forbidden Patterns:** The slice contains no forbidden patterns from section 2.3.
- [ ] **Parameter Justification:** A justification for the chosen `parameters` and `budget` values is included in the PR description.
- [ ] **Determinism:** All aspects of the slice definition are deterministic. There are no random or time-dependent values.

---

## 3. Roadmap for Curriculum v3

Curriculum v2.1 is a robust, declarative system. A future v3 should introduce dynamic capabilities and formal verification tooling.

**Proposed Features for v3:**

1.  **Automated Curriculum Verifier (`curriculum-linter`):**
    *   **Concept:** A standalone CLI tool that ingests a curriculum YAML file and automatically verifies its compliance with the full Slice Correctness Contract (sections 2.1, 2.2, 2.3).
    *   **Roadmap:** This should be the first priority. It would be integrated into the CI/CD pipeline, automatically blocking PRs that propose invalid slices. It would check for schema validity, structural invariants, and forbidden patterns.

2.  **Formal Sequencing and Gating:**
    *   **Concept:** Evolve from a flat list of slices to a directed acyclic graph (DAG). Introduce new keys to the slice definition, such as `unlocks_on_completion: [slice_name, ...]` and a formal `completion_criteria` block (e.g., `{ metric: success_rate, threshold: 0.8, window: 100_cycles }`).
    *   **Benefit:** This would create a true, adaptive curriculum where the system only presents harder slices to an agent after it has demonstrated mastery over easier, prerequisite ones.

3.  **Procedural Slice Generation:**
    *   **Concept:** Instead of hand-authoring every slice, define "slice templates" or "generators." A generator could take high-level difficulty parameters (e.g., `difficulty: 0.7`, `type: 'goal_seeking'`) and procedurally generate a valid slice definition that meets the formal contract.
    *   **Benefit:** Allows for much finer-grained control over difficulty and enables the creation of a vast, smooth curriculum without manual effort.

4.  **Dynamic Difficulty Adjustment (DDA):**
    *   **Concept:** The ultimate goal. A closed-loop system where agent telemetry is fed back into the v3 curriculum system. If an agent's performance on a procedurally generated slice is too high or too low, the system automatically adjusts the parameters and generates a new, more appropriately difficult slice for the next block of cycles.
    *   **Benefit:** Creates a perfectly tailored learning environment that maximizes the agent's rate of improvement. This is the foundation for true automated curriculum design.
