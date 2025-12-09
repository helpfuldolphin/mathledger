# DAG Drift Auditor Specification

**PHASE II — NOT RUN IN PHASE I**

**Author:** CLAUDE G — DAG Analyst General
**Status:** Specification Document
**Version:** 1.0.0

---

## 1. Purpose

The DAG Drift Auditor detects and quantifies structural changes in derivation DAGs across experiment cycles. Drift detection is essential for:

- Identifying non-deterministic behavior in derivation engines
- Detecting data corruption or logging inconsistencies
- Validating that RFL policy updates don't corrupt proof structures
- Ensuring reproducibility across baseline and treatment runs

---

## 2. Drift Definition

### 2.1 Temporal DAG Sequence

A U2 experiment produces a sequence of DAG snapshots:

```
G₀ → G₁ → G₂ → ... → Gₙ
```

Where Gₜ = (Vₜ, Eₜ) is the DAG state after cycle t.

### 2.2 Drift Types

| Type | Symbol | Description |
|------|--------|-------------|
| **Structural Drift** | Δ_struct | Changes to graph topology (edges, vertices) |
| **Depth Drift** | Δ_depth | Changes to depth distribution |
| **Verification Drift** | Δ_verify | Changes to verification status |
| **Provenance Drift** | Δ_prov | Changes to parent assignments |

### 2.3 Drift Magnitude

For a drift metric Δ, magnitude is computed as:

```
|Δ(Gₜ, Gₜ₊₁)| = measure of change between consecutive snapshots
```

Cumulative drift over k cycles:

```
Δ_cum(G₀, Gₖ) = Σᵢ |Δ(Gᵢ, Gᵢ₊₁)| for i ∈ [0, k-1]
```

---

## 3. Structural Drift Metrics

### 3.1 Vertex Drift

**Definition:** Change in vertex set between cycles.

```
Δ_V(Gₜ, Gₜ₊₁) = |Vₜ₊₁ \ Vₜ| + |Vₜ \ Vₜ₊₁|
               = |added vertices| + |removed vertices|
```

**Expected Behavior:**
- Monotonic growth: Vertices should only be added, never removed
- Removal indicates data loss or corruption

**Alert Threshold:**
```
ALERT if |Vₜ \ Vₜ₊₁| > 0  (any vertex removal)
```

### 3.2 Edge Drift

**Definition:** Change in edge set between cycles.

```
Δ_E(Gₜ, Gₜ₊₁) = |Eₜ₊₁ \ Eₜ| + |Eₜ \ Eₜ₊₁|
               = |added edges| + |removed edges|
```

**Expected Behavior:**
- Edges should only be added with new derivations
- Edge removal indicates parent reassignment (pathological)

**Alert Threshold:**
```
ALERT if |Eₜ \ Eₜ₊₁| > 0  (any edge removal)
```

### 3.3 Topology Change Rate

**Definition:** Normalized structural change per cycle.

```
τ(Gₜ, Gₜ₊₁) = (Δ_V + Δ_E) / (|Vₜ| + |Eₜ|)
```

**Interpretation:**
- τ ≈ 0: Stable structure
- τ > 0.1: Significant restructuring
- τ > 0.5: Major topology change

---

## 4. Depth Drift Metrics

### 4.1 Depth Distribution Drift

**Definition:** Change in depth histogram.

Let H_t(d) = count of vertices at depth d in Gₜ.

```
Δ_hist(Gₜ, Gₜ₊₁) = Σ_d |Hₜ₊₁(d) - Hₜ(d)|
```

**Normalized Form (Earth Mover's Distance approximation):**
```
Δ_hist_norm = Δ_hist / |Vₜ₊₁|
```

### 4.2 Max Depth Drift

**Definition:** Change in maximum chain depth.

```
Δ_max(Gₜ, Gₜ₊₁) = max_depth(Gₜ₊₁) - max_depth(Gₜ)
```

**Expected Behavior:**
- Non-negative (depth should not decrease)
- Bounded by slice configuration

**Alert Threshold:**
```
ALERT if Δ_max < 0  (depth regression)
ALERT if max_depth(Gₜ) > MAX_CONFIGURED_DEPTH
```

### 4.3 Depth Stability Index

**Definition:** Variance of max depth over window.

```
DSI(G₀...Gₖ) = var([max_depth(Gᵢ) for i ∈ [0, k]])
```

**Interpretation:**
- DSI ≈ 0: Stable depth growth
- High DSI: Erratic depth changes

---

## 5. Verification Drift Metrics

### 5.1 Verification Status Drift

**Definition:** Change in verification statuses.

Let S_t = {v ∈ Vₜ : v.verified = TRUE}

```
Δ_verify(Gₜ, Gₜ₊₁) = |Sₜ₊₁ \ Sₜ| + |Sₜ \ Sₜ₊₁|
                    = |newly verified| + |de-verified|
```

**Expected Behavior:**
- De-verification should never occur
- Verification is monotonic

**Alert Threshold:**
```
ALERT if |Sₜ \ Sₜ₊₁| > 0  (any de-verification)
```

### 5.2 Verification Rate Drift

**Definition:** Change in verification success rate.

```
r_t = |Sₜ| / |Vₜ|
Δ_rate(Gₜ, Gₜ₊₁) = rₜ₊₁ - rₜ
```

**Interpretation:**
- Δ_rate > 0: Improving verification
- Δ_rate < 0: Degrading verification
- |Δ_rate| > 0.1: Significant rate change

### 5.3 Verified Depth Consistency

**Definition:** Stability of verified chain depths.

```
VDC(v, Gₜ, Gₜ₊₁) = |d_Sₜ₊₁(v) - d_Sₜ(v)|
```

**Expected Behavior:**
- Verified depth should only increase (as more parents verified)
- Decrease indicates verification regression

---

## 6. Provenance Drift Metrics

### 6.1 Parent Set Drift

**Definition:** Change in parent assignments for existing vertices.

For v ∈ Vₜ ∩ Vₜ₊₁:
```
Δ_parents(v) = |parentsₜ₊₁(v) Δ parentsₜ(v)|
             = symmetric difference of parent sets
```

**Aggregate:**
```
Δ_prov(Gₜ, Gₜ₊₁) = Σ_{v ∈ Vₜ ∩ Vₜ₊₁} Δ_parents(v)
```

**Expected Behavior:**
- Δ_prov = 0 for all cycles (parents never change)
- Any non-zero value indicates provenance corruption

**Alert Threshold:**
```
ALERT if Δ_prov > 0  (any parent change)
SEVERITY: CRITICAL
```

### 6.2 Rule Drift

**Definition:** Change in derivation rule assignments.

```
Δ_rule(v) = {
    0  if ruleₜ₊₁(v) = ruleₜ(v)
    1  otherwise
}
```

**Expected Behavior:**
- Rules should never change for existing vertices

---

## 7. Cross-Run Drift Analysis

### 7.1 Baseline vs RFL Structural Divergence

Compare DAGs from baseline and RFL runs at same cycle:

```
Divergence(G_base_t, G_rfl_t) = {
    V_divergence: |V_base_t Δ V_rfl_t| / max(|V_base_t|, |V_rfl_t|),
    E_divergence: |E_base_t Δ E_rfl_t| / max(|E_base_t|, |E_rfl_t|),
    depth_divergence: |max_depth(G_base_t) - max_depth(G_rfl_t)|
}
```

**Interpretation:**
- Low divergence: Similar exploration paths
- High divergence: RFL finding different derivations

### 7.2 Determinism Validation

For identical seeds, baseline runs should produce identical DAGs:

```
Determinism_check(G₁, G₂) = {
    PASS if V₁ = V₂ AND E₁ = E₂
    FAIL otherwise
}
```

### 7.3 Seed Sensitivity

Measure DAG variation across different seeds:

```
Sensitivity(seeds) = var([|V_seed| for seed in seeds])
```

High sensitivity indicates non-deterministic components.

---

## 8. Inconsistency Detection

### 8.1 Parent-Depth Inconsistency

**Definition:** Vertex depth doesn't match parent depths.

```
Inconsistent(v) = d(v) ≠ 1 + max(d(p) for p in parents(v))
```

**Action:** Recompute depths from scratch.

### 8.2 Orphan Detection

**Definition:** Derived vertex with no path to axioms.

```
Orphan(v) = d(v) > 1 AND ∄ path to any axiom
```

**Action:** Flag vertex as invalid.

### 8.3 Hash-Formula Mismatch

**Definition:** Same hash maps to different formulas across cycles.

```
Mismatch(h) = |{formula_t(h) for t ∈ cycles}| > 1
```

**Action:** CRITICAL alert, investigate hash function.

### 8.4 Temporal Ordering Violation

**Definition:** Child appears before parent in log.

```
Violation(c, p, t_c, t_p) = t_c < t_p where (c, p) ∈ E
```

**Action:** Flag log ordering issue.

---

## 9. Drift Auditor Output Format

### 9.1 Per-Cycle Report

```json
{
    "cycle": 42,
    "timestamp": "2025-01-15T12:00:00Z",
    "structural": {
        "vertices_added": 5,
        "vertices_removed": 0,
        "edges_added": 8,
        "edges_removed": 0,
        "topology_change_rate": 0.02
    },
    "depth": {
        "max_depth_before": 4,
        "max_depth_after": 5,
        "depth_drift": 1,
        "distribution_drift": 5
    },
    "verification": {
        "newly_verified": 3,
        "de_verified": 0,
        "rate_before": 0.75,
        "rate_after": 0.78
    },
    "provenance": {
        "parent_changes": 0,
        "rule_changes": 0
    },
    "alerts": []
}
```

### 9.2 Cumulative Summary

```json
{
    "experiment_id": "uplift_u2_tree_001",
    "total_cycles": 500,
    "cumulative_drift": {
        "structural": 1250,
        "depth": 45,
        "verification": 380,
        "provenance": 0
    },
    "alerts_summary": {
        "critical": 0,
        "error": 0,
        "warning": 12,
        "info": 45
    },
    "determinism_check": "PASS",
    "invariants_violated": []
}
```

---

## 10. Alert Severity Levels

| Level | Criteria | Action |
|-------|----------|--------|
| **CRITICAL** | Provenance change, hash collision, cycle detected | Halt experiment, investigate |
| **ERROR** | Vertex/edge removal, de-verification | Flag run, review data |
| **WARNING** | Depth regression, high topology change | Log for analysis |
| **INFO** | Normal growth metrics | Record for trends |

---

## 11. Drift Thresholds

### 11.1 Recommended Defaults

| Metric | Warning | Error | Critical |
|--------|---------|-------|----------|
| Topology change rate (τ) | > 0.1 | > 0.3 | > 0.5 |
| Vertices removed | 1 | 5 | Any |
| Edges removed | 1 | 5 | Any |
| Parent changes | — | — | 1 |
| De-verifications | 1 | 5 | Any |
| Max depth regression | 1 | 3 | 5 |

### 11.2 Slice-Specific Overrides

For `slice_uplift_tree`:
- Max depth warning: > MAX_CHAIN_LENGTH + 2
- Verified depth regression: ERROR at any occurrence

---

## 12. Integration Points

### 12.1 CLI Integration

```bash
# Run drift audit on experiment logs
uv run python experiments/audit_dag_drift.py \
    --baseline results/uplift_u2_tree_baseline.jsonl \
    --treatment results/uplift_u2_tree_rfl.jsonl \
    --out results/drift_audit_report.json
```

### 12.2 Programmatic API

```python
from experiments.dag_drift_auditor import DriftAuditor

auditor = DriftAuditor()
auditor.load_baseline("baseline.jsonl")
auditor.load_treatment("rfl.jsonl")

report = auditor.compute_drift()
alerts = auditor.get_alerts(severity="WARNING")
```

### 12.3 CI/CD Integration

```yaml
# In CI pipeline
- name: Audit DAG Drift
  run: |
    python experiments/audit_dag_drift.py \
      --baseline $BASELINE_LOG \
      --treatment $RFL_LOG \
      --fail-on-error
```

---

## 13. Appendix: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| G = (V, E) | Graph with vertices V and edges E |
| Gₜ | DAG state at cycle t |
| d(v) | Chain depth of vertex v |
| d_S(v) | Verified chain depth (through set S) |
| Δ | Symmetric difference |
| Δ_X | Drift metric for property X |
| τ | Topology change rate |
| Σ | Summation |
| var() | Variance |
| max() | Maximum |
| \| · \| | Cardinality (set size) |

---

*PHASE II — NOT RUN IN PHASE I*
