# Metric Correctness Contract — Phase II U2

**Version**: 1.0.0
**Status**: ENFORCED
**Effective**: Phase II — NOT RUN IN PHASE I
**Module**: `experiments/slice_success_metrics.py`

---

## 1. The Prime Directive

> "A metric that drifts is a metric that lies."

All slice success metrics in the U2 experiment framework MUST be:
1. **Deterministic**: Same inputs → identical outputs, always.
2. **Pure**: No side effects, no hidden state, no I/O.
3. **Isolated**: No cross-metric contamination or shared mutable state.
4. **Versioned**: Changes require explicit version bumps and governance approval.

---

## 2. Axioms of Metric Correctness

### Axiom D — Determinism

```
∀ inputs I, ∀ time t₁ < t₂:
  compute_metric(kind, I, t₁) ≡ compute_metric(kind, I, t₂)
```

A metric function MUST produce bit-identical results regardless of:
- Wall-clock time
- CPU load or execution speed
- Number of previous invocations
- Python interpreter version (within 3.9+)
- Operating system

**Enforcement**: 100-iteration replay tests in `test_slice_success_metrics.py`.

### Axiom P — Purity

```
∀ metric M:
  effect_set(M) = ∅
  dependency_set(M) ⊆ {explicit_parameters}
```

A metric function MUST NOT:
- Modify any global state
- Read from files, databases, or network
- Write to any external system
- Cache results between calls (internal memoization within a single call is permitted)
- Access `datetime.now()`, `time.time()`, `random.*`, or `uuid.uuid4()`

**Enforcement**: Static analysis grep guard + pure function contract tests.

### Axiom I — Isolation

```
∀ metrics M₁, M₂ where M₁ ≠ M₂:
  shared_state(M₁, M₂) = ∅
  result(M₁) ⊥ result(M₂)  // Independent
```

Metrics MUST be:
- Independently computable in any order
- Safe for parallel execution
- Free from cross-metric dependencies

**Enforcement**: Parallel execution tests with shuffled metric order.

### Axiom O — Order Independence

```
∀ list L, ∀ permutation π:
  compute_metric(kind, L) = compute_metric(kind, π(L))
```

For list-typed inputs (e.g., `verified_statements`), the metric result MUST be invariant to input ordering.

**Enforcement**: Permutation tests in `TestDeterminism` class.

---

## 3. Negative Control Requirements

### NC-1: Zero-Input Baseline

Every metric MUST produce a defined, deterministic result for empty inputs:

| Metric | Empty Input | Expected Result |
|--------|-------------|-----------------|
| `goal_hit` | `verified_statements=[]` | `(False, 0.0)` unless `min_total_verified=0` |
| `sparse_success` | `verified_count=0` | `(False, 0.0)` unless `min_verified=0` |
| `chain_success` | `verified_statements=[]` | `(False, 0.0)` |
| `multi_goal` | `verified_hashes={}` | `(True, 0.0)` if `required_goal_hashes={}`, else `(False, 0.0)` |

### NC-2: Impossibility Detection

Metrics MUST NOT claim success when success is impossible:

| Condition | Required Behavior |
|-----------|-------------------|
| `target_hashes` has 5 elements, only 3 verified | `value ≤ 3.0` |
| `min_chain_length=10`, only 5 nodes verified | `success=False` |
| `required_goal_hashes` has 4, only 2 verified | `success=False`, `value=2.0` |

### NC-3: Threshold Boundary

For any threshold parameter `T`:
- Input yielding metric value `T-1` → `success=False`
- Input yielding metric value `T` → `success=True`
- Input yielding metric value `T+1` → `success=True`

---

## 4. Monotonicity Expectations

### MON-1: Additive Monotonicity (Goal Hit, Sparse, Multi-Goal)

Adding more verified items MUST NOT decrease the metric value:

```
∀ V₁ ⊆ V₂:
  value(metric(V₁)) ≤ value(metric(V₂))
```

**Exception**: Chain success is graph-dependent and may not be monotonic in the general case.

### MON-2: Threshold Monotonicity

Lowering a threshold MUST NOT decrease the success probability:

```
∀ T₁ > T₂:
  P[success | threshold=T₂] ≥ P[success | threshold=T₁]
```

### MON-3: Subset Inclusion

For `multi_goal`:
```
∀ R₁ ⊆ R₂ (required goals):
  success(V, R₁) ← success(V, R₂)  // Success with R₂ implies success with R₁
```

---

## 5. Metric-Specific Invariant Tables

### 5.1 GOAL_HIT Invariants

| ID | Invariant | Formal Statement |
|----|-----------|------------------|
| GOAL-1 | Hit count ≤ target count | `value ≤ len(target_hashes)` |
| GOAL-2 | Hit count ≤ verified count | `value ≤ len(verified_statements)` |
| GOAL-3 | Empty targets → zero hits | `target_hashes = {} → value = 0.0` |
| GOAL-4 | Disjoint sets → zero hits | `verified ∩ targets = {} → value = 0.0` |
| GOAL-5 | Success threshold | `success ↔ value ≥ min_total_verified` |
| GOAL-6 | Deterministic | `compute(I) at t₁ = compute(I) at t₂` |
| GOAL-7 | Order independent | `compute([a,b]) = compute([b,a])` |

### 5.2 SPARSE_SUCCESS Invariants

| ID | Invariant | Formal Statement |
|----|-----------|------------------|
| SPARSE-1 | Value equals input | `value = float(verified_count)` |
| SPARSE-2 | Success threshold | `success ↔ verified_count ≥ min_verified` |
| SPARSE-3 | Attempted ignored | `compute(v, a₁, m) = compute(v, a₂, m)` for any `a₁, a₂` |
| SPARSE-4 | Zero verified | `verified_count = 0 → value = 0.0` |
| SPARSE-5 | Non-negative | `value ≥ 0.0` |
| SPARSE-6 | Deterministic | `compute(I) at t₁ = compute(I) at t₂` |

### 5.3 CHAIN_SUCCESS Invariants

| ID | Invariant | Formal Statement |
|----|-----------|------------------|
| CHAIN-1 | Value ≤ verified count | `value ≤ len(verified_statements)` |
| CHAIN-2 | Unverified target → zero | `target ∉ verified → value = 0.0` |
| CHAIN-3 | Isolated target → one | `target ∈ verified ∧ deps(target) = {} → value = 1.0` |
| CHAIN-4 | Cycle safety | Graph cycles do not cause infinite recursion |
| CHAIN-5 | Success threshold | `success ↔ value ≥ min_chain_length` |
| CHAIN-6 | Longest path | `value = max_chain_length(target, graph, verified)` |
| CHAIN-7 | Deterministic | `compute(I) at t₁ = compute(I) at t₂` |
| CHAIN-8 | Order independent | Statement list order does not affect result |

### 5.4 MULTI_GOAL_SUCCESS Invariants

| ID | Invariant | Formal Statement |
|----|-----------|------------------|
| MULTI-1 | Value ≤ required count | `value ≤ len(required_goal_hashes)` |
| MULTI-2 | All-or-nothing success | `success ↔ value = len(required_goal_hashes)` |
| MULTI-3 | Empty required → success | `required = {} → success = True, value = 0.0` |
| MULTI-4 | Subset monotonicity | `verified₁ ⊆ verified₂ → value₁ ≤ value₂` |
| MULTI-5 | Goal counting | `value = len(verified ∩ required)` |
| MULTI-6 | Deterministic | `compute(I) at t₁ = compute(I) at t₂` |

---

## 6. Metric Drift Classifier

### 6.1 Drift Taxonomy

| Class | Severity | Description | Example |
|-------|----------|-------------|---------|
| **D0 — Cosmetic** | INFO | Documentation, naming, logging changes | Fixing typo in docstring |
| **D1 — Additive** | LOW | New optional parameters with defaults | Adding `verbose` flag |
| **D2 — Behavioral-Compatible** | MEDIUM | Internal algorithm change, same outputs | Optimizing loop structure |
| **D3 — Behavioral-Breaking** | HIGH | Output changes for existing inputs | Changing threshold semantics |
| **D4 — Schema-Breaking** | CRITICAL | Parameter or return type changes | Changing `value` from float to int |
| **D5 — Semantic-Breaking** | CRITICAL | Fundamental metric meaning changes | Redefining what "success" means |

### 6.2 Drift Detection Triggers

A change is classified as drift if ANY of the following occur:

| Trigger | Detection Method | Classification |
|---------|------------------|----------------|
| Return value differs for any test case | Regression test failure | D3+ |
| New required parameter added | Type signature analysis | D4 |
| Parameter removed | Type signature analysis | D4 |
| Return type changed | Type signature analysis | D4 |
| Invariant violated | Contract test failure | D3+ |
| Determinism test fails | Replay test failure | D5 |
| New exception type raised | Exception audit | D2+ |

### 6.3 Governance Approval Matrix

| Drift Class | Approval Required | Minimum Review | Version Bump |
|-------------|-------------------|----------------|--------------|
| D0 | None | Self | None |
| D1 | Code owner | 1 reviewer | PATCH |
| D2 | Code owner + QA | 2 reviewers | MINOR |
| D3 | Governance board | 3 reviewers + tests | MAJOR |
| D4 | Governance board + migration plan | Full team | MAJOR |
| D5 | Full experiment abort + repreregistration | External audit | NEW EXPERIMENT |

---

## 7. Versioning Scheme

### 7.1 Semantic Versioning for Metrics

```
MAJOR.MINOR.PATCH[-PHASE.QUALIFIER]

Examples:
  1.0.0-phaseII.u2    # Initial Phase II U2 release
  1.0.1-phaseII.u2    # Patch (D0/D1 changes)
  1.1.0-phaseII.u2    # Minor (D2 changes, backward compatible)
  2.0.0-phaseII.u2    # Major (D3/D4 changes, breaking)
```

### 7.2 Version Compatibility Rules

| Consumer Version | Provider Version | Compatibility |
|------------------|------------------|---------------|
| 1.x.x | 1.y.z where y ≥ x | COMPATIBLE |
| 1.x.x | 2.y.z | INCOMPATIBLE (migration required) |
| 1.x.x | 1.y.z where y < x | DOWNGRADE (may work, not guaranteed) |

### 7.3 Version Attestation

Every metric computation SHOULD include version attestation:

```python
from experiments.slice_success_metrics import VERSION

result = compute_metric("goal_hit", ...)
attestation = {
    "metric_version": VERSION,
    "computation_hash": sha256(canonical_inputs),
    "result": result,
}
```

### 7.4 Forward Compatibility Guarantees

The following are guaranteed stable across MINOR version bumps:

| Element | Stability Guarantee |
|---------|---------------------|
| `compute_metric(kind, **kwargs)` signature | Stable |
| Return type `Tuple[bool, float]` | Stable |
| `METRIC_KINDS` membership | Additive only |
| Required kwargs per kind | Stable (new optional only) |
| Invariant contracts | Strengthening only |

---

## 8. Enforcement Mechanisms

### 8.1 Static Analysis

```bash
# Forbidden patterns in slice_success_metrics.py
grep -E "(datetime\.now|time\.time|random\.|uuid\.uuid4)" experiments/slice_success_metrics.py
# Expected: 0 matches
```

### 8.2 Contract Tests

All invariants from Section 5 MUST have corresponding test cases in:
- `tests/test_slice_success_metrics.py`

### 8.3 Regression Baseline

A golden baseline of (input, output) pairs MUST be maintained:
- Location: `tests/fixtures/metric_golden_baseline.json`
- Any change to golden baseline requires D3+ classification

### 8.4 CI Integration

```yaml
# .github/workflows/metric-contract.yml
- name: Metric Contract Verification
  run: |
    python -m pytest tests/test_slice_success_metrics.py -v
    python -m pytest tests/test_metric_invariants.py -v
    python scripts/verify_metric_golden_baseline.py
```

---

## 9. Exception Registry

The following are explicitly permitted exceptions to the purity axiom:

| Exception | Scope | Justification |
|-----------|-------|---------------|
| Internal memoization | Within single call | Performance optimization, no external effect |
| Type coercion logging | Development only | Debug output, stripped in production |

---

## 10. Change Log

| Version | Date | Changes | Drift Class |
|---------|------|---------|-------------|
| 1.0.0 | Phase II U2 | Initial contract | N/A |

---

## Appendix A: Contract Verification Checklist

Before any metric change is merged:

- [ ] All invariants from Section 5 verified via tests
- [ ] Determinism test passes (100 iterations)
- [ ] Order independence test passes (3+ permutations)
- [ ] Negative control tests pass
- [ ] Monotonicity tests pass
- [ ] Golden baseline unchanged OR drift classified
- [ ] Version bumped appropriately
- [ ] Governance approval obtained (if D2+)
- [ ] CHANGELOG updated

---

## Appendix B: Quick Reference Card

```
┌──────────────────────────────────────────────────────────────┐
│                 METRIC CORRECTNESS QUICK REF                  │
├──────────────────────────────────────────────────────────────┤
│ AXIOMS:        D(eterminism) P(urity) I(solation) O(rder)    │
├──────────────────────────────────────────────────────────────┤
│ DRIFT CLASSES: D0=cosmetic D1=additive D2=compatible         │
│                D3=breaking D4=schema D5=semantic             │
├──────────────────────────────────────────────────────────────┤
│ VERSION BUMP:  D0-D1=PATCH D2=MINOR D3-D4=MAJOR D5=ABORT     │
├──────────────────────────────────────────────────────────────┤
│ FORBIDDEN:     datetime.now() random.* uuid.uuid4() I/O      │
├──────────────────────────────────────────────────────────────┤
│ ENFORCEMENT:   100-iter replay + golden baseline + CI        │
└──────────────────────────────────────────────────────────────┘
```
