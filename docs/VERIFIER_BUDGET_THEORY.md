# Verifier Budget Theory

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This document provides theoretical justification for the verifier budget system
> used in Phase II uplift experiments. It explains the rationale for cycle budgets,
> timeout semantics, and governance constraints.

---

## 1. Introduction

The Verifier Budget Envelope is a resource governance mechanism that bounds the
computational work performed during each derivation cycle. It ensures:

1. **Deterministic execution bounds** — Experiments complete in predictable time
2. **Fair comparison** — Baseline and RFL modes operate under identical constraints
3. **Abstention integrity** — Timeouts produce structured outcomes, never false positives
4. **Reproducibility** — Same budget + seed → same verification sequence

This document provides the theoretical foundation for these properties.

---

## 2. Theoretical Justification for Cycle Budgets

### 2.1 The Bounded Computation Principle

**Theorem (Bounded Verification):** For any derivation cycle with budget `B_cycle`,
the total verification work `W` satisfies:

```
W ≤ min(N × T_stmt, B_cycle)
```

Where:
- `N` = number of candidates (`max_candidates_per_cycle`)
- `T_stmt` = per-statement timeout (`taut_timeout_s`)
- `B_cycle` = cycle wall-clock budget (`cycle_budget_s`)

**Proof:** Each candidate verification is bounded by `T_stmt`. The cycle processes
at most `N` candidates, giving upper bound `N × T_stmt`. The cycle-level budget
`B_cycle` provides a second, independent bound. The actual work is the minimum.

### 2.2 Why Cycle Budgets Are Necessary

Without cycle budgets, verification time becomes unbounded in pathological cases:

| Scenario | Without Budget | With Budget |
|----------|----------------|-------------|
| Many complex candidates | Unbounded | `≤ B_cycle` |
| Slow verification oracle | Unbounded | `≤ B_cycle` |
| Resource contention | Unbounded | `≤ B_cycle` |
| Adversarial inputs | Unbounded | `≤ B_cycle` |

**Key insight:** Per-statement timeouts alone are insufficient. If the oracle
is slow but never times out, `N` statements at `T_stmt - ε` each yields
`N × (T_stmt - ε)` total time — potentially hours for large `N`.

The cycle budget provides a **hard ceiling** independent of individual statement
behavior.

### 2.3 Budget Decomposition

The total cycle time decomposes into:

```
T_cycle = T_setup + Σᵢ T_verify(i) + T_policy + T_attestation
```

Where:
- `T_setup` — Candidate generation and ordering (~constant)
- `T_verify(i)` — Verification of candidate `i` (bounded by `T_stmt`)
- `T_policy` — Policy update computation (~constant for RFL)
- `T_attestation` — Hash computation and logging (~constant)

The budget primarily constrains `Σᵢ T_verify(i)`, which dominates for large `N`.

### 2.4 Optimal Budget Selection

**Goal:** Choose `B_cycle` to maximize throughput while ensuring completion.

For a cycle with `N` candidates where each verification takes expected time `E[T]`:

```
B_cycle = α × N × E[T] + β
```

Where:
- `α > 1` — Safety margin (typically 1.5–2.0)
- `β` — Overhead allowance for setup, policy, attestation

**Phase II slice budgets were derived empirically:**

| Slice | `N` | `E[T]` est. | `α` | `β` | `B_cycle` |
|-------|-----|-------------|-----|-----|-----------|
| `slice_uplift_goal` | 40 | 0.05s | 2.0 | 1.0 | 5.0s |
| `slice_uplift_sparse` | 40 | 0.06s | 2.0 | 1.2 | 6.0s |
| `slice_uplift_tree` | 30 | 0.05s | 2.0 | 1.0 | 4.0s |
| `slice_uplift_dependency` | 40 | 0.06s | 2.0 | 1.2 | 6.0s |

---

## 3. Verifier Timeout Effects on Abstention Dynamics

### 3.1 The Abstention Taxonomy

Verification produces one of four outcome classes:

| Outcome | Symbol | Information Content | Policy Signal |
|---------|--------|---------------------|---------------|
| Verified | `V` | High (true positive) | +reward |
| Refuted | `R` | High (true negative) | -reward |
| Abstain | `A` | Low (unknown) | neutral |
| Skip | `S` | Zero (not attempted) | none |

**Critical property:** Abstentions (`A`) and skips (`S`) are information-theoretically
distinct from verification outcomes (`V`, `R`).

### 3.2 Timeout-Induced Abstention

When verification exceeds `taut_timeout_s`:

```
Outcome(candidate) = A_timeout  (abstain due to timeout)
```

**Properties of timeout abstention:**

1. **Conservative:** Never claims `V` or `R` without evidence
2. **Deterministic:** Same candidate + seed + timeout → same abstention
3. **Distinguishable:** `A_timeout` is logged separately from `A_complexity`
4. **Policy-neutral:** Does not update policy weights

### 3.3 Abstention Rate Analysis

Let `p_timeout` be the probability a candidate times out. The expected abstention
rate per cycle is:

```
E[abstention_rate] = p_timeout × (1 - p_skip)
```

Where `p_skip` is the probability of budget-induced skipping.

**Implication:** Shorter timeouts increase `p_timeout`, which increases abstention
rate but decreases cycle time. This creates a tradeoff:

```
                    High timeout
                         │
                         ▼
          ┌──────────────────────────┐
          │ Low abstention rate      │
          │ High cycle time          │
          │ More information/cycle   │
          └──────────────────────────┘
                         │
         Timeout tradeoff curve
                         │
                         ▼
          ┌──────────────────────────┐
          │ High abstention rate     │
          │ Low cycle time           │
          │ Less information/cycle   │
          └──────────────────────────┘
                         ▲
                         │
                    Low timeout
```

### 3.4 Abstention Rate Bounds

For a well-configured budget:

```
abstention_rate ∈ [0, abstention_cap]
```

Where `abstention_cap` is typically set by the slice gate configuration.

**Phase II slices allow up to 100% abstention** (`max_rate_pct: 100.0`) because:
- Hermetic truth-table verification rarely times out
- High abstention would indicate configuration error, not expected behavior

### 3.5 The Abstention Invariant

**Theorem (Abstention Safety):** For any timeout-induced abstention:

```
Outcome = A_timeout  ⟹  Outcome ∉ {V, R}
```

**Proof:** The timeout handler is invoked *before* result interpretation.
When timeout fires:
1. Verification computation is interrupted
2. Result buffer is discarded (not interpreted)
3. Abstention outcome is set atomically
4. No path exists from timeout to `V` or `R`

This guarantees no false positives or false negatives from timeouts.

---

## 4. Governance Rationale for Hard Budget Caps

### 4.1 Why Hard Caps?

Soft caps (advisory limits) are insufficient for scientific reproducibility:

| Property | Soft Cap | Hard Cap |
|----------|----------|----------|
| Determinism | No guarantee | Guaranteed |
| Reproducibility | Variable | Exact |
| Comparison validity | Questionable | Valid |
| Audit trail | Incomplete | Complete |

**Hard caps ensure:** Given the same seed and configuration, two runs produce
identical verification sequences and outcomes.

### 4.2 The Three-Level Budget Hierarchy

Phase II enforces budgets at three levels:

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: Per-Statement Budget (taut_timeout_s)                  │
│ ─────────────────────────────────────────────────────────────── │
│ Scope: Single candidate verification                            │
│ Enforcement: Timer interrupt                                    │
│ Violation: Abstention (A_timeout)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 2: Per-Cycle Budget (cycle_budget_s)                      │
│ ─────────────────────────────────────────────────────────────── │
│ Scope: All verifications in one cycle                           │
│ Enforcement: Wall-clock check before each candidate             │
│ Violation: Skip remaining candidates (S_budget)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Level 3: Candidate Cap (max_candidates_per_cycle)               │
│ ─────────────────────────────────────────────────────────────── │
│ Scope: Candidate count per cycle                                │
│ Enforcement: Counter limit                                      │
│ Violation: Normal cycle termination                             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Governance Constraints

The budget system enforces governance rules from `PREREG_UPLIFT_U2.yaml`:

| Constraint | Mechanism | Violation Handling |
|------------|-----------|-------------------|
| Determinism | Fixed budgets per slice | N/A (by design) |
| Bounded resources | Hard caps at all levels | Abstain or skip |
| No silent failures | Explicit outcome codes | Logged + attested |
| Reproducibility | Budget in manifest | Audit-verifiable |

### 4.4 Why Not Adaptive Budgets?

Adaptive budgets (that adjust based on runtime behavior) are explicitly excluded:

**Problems with adaptive budgets:**
1. Non-deterministic — different runs may have different budgets
2. Feedback loops — budget affects outcomes, outcomes affect budget
3. Comparison invalidity — baseline vs RFL under different constraints
4. Audit complexity — harder to verify correct behavior

**Phase II mandate:** All budget parameters are fixed at experiment start and
recorded in the manifest.

### 4.5 Budget Exhaustion Policy

When cycle budget is exhausted:

```python
# Pseudocode for budget exhaustion
if elapsed >= cycle_budget_s:
    for remaining in candidates[verified_count:]:
        outcome[remaining] = "budget_skip"
        log_skip(remaining, reason="cycle_budget_exhausted")

    attestation.budget_exhausted = True
    attestation.skipped_count = len(candidates) - verified_count
```

**Policy signal for skipped candidates:** None. Skipped candidates are not
observed, so they cannot inform policy updates.

---

## 5. Formal Properties

### 5.1 Determinism Theorem

**Theorem:** For fixed `(seed, budget_config, candidate_pool)`:

```
run₁(seed, budget, pool) = run₂(seed, budget, pool)
```

Where equality includes:
- Verification sequence
- Outcome sequence
- Attestation hashes

**Proof sketch:**
1. Candidate ordering is deterministic (seeded RNG or policy scores)
2. Verification is deterministic (truth table evaluation)
3. Budget checks are deterministic (fixed thresholds)
4. Timeout handling is deterministic (abstain on expiry)
5. Hash computation is deterministic (SHA-256)

### 5.2 Fairness Theorem

**Theorem:** Baseline and RFL modes operate under identical budget constraints:

```
budget(baseline, slice) = budget(rfl, slice)
```

**Proof:** Budget parameters are slice-specific, not mode-specific.
Both modes load from the same `verifier_budget_phase2.yaml` section.

### 5.3 Safety Theorem

**Theorem:** No budget event produces a false verification outcome:

```
∀ candidate: timeout(candidate) ∨ skip(candidate) ⟹ outcome ∉ {V, R}
```

**Proof:** Timeout and skip handlers produce only `A_timeout` or `S_budget`,
which are disjoint from `{V, R}` by construction.

---

## 6. Practical Implications

### 6.1 For Experiment Design

1. **Choose budgets conservatively** — Allow 2× expected verification time
2. **Monitor abstention rates** — High rates indicate misconfigured budgets
3. **Record all parameters** — Budget config must be in manifest
4. **Test boundary conditions** — Verify behavior at budget limits

### 6.2 For Implementation

1. **Check budget before each verification** — Not after
2. **Use monotonic clocks** — Wall clock can jump
3. **Log all budget events** — Timeout, skip, exhaustion
4. **Include budget in attestation** — Required for reproducibility

### 6.3 For Analysis

1. **Exclude skipped candidates from metrics** — They weren't observed
2. **Report abstention rates separately** — By reason (timeout vs complexity)
3. **Verify budget consistency** — Compare manifest to config
4. **Account for budget effects** — When comparing slice performance

---

## 7. References

- `backend/lean_control_sandbox_plan.md` §12-16 (Verifier Budget Envelope)
- `config/verifier_budget_phase2.yaml` (Budget Configuration)
- `experiments/prereg/PREREG_UPLIFT_U2.yaml` (Preregistration)
- `backend/verification/budget_loader.py` (Implementation)

---

## Appendix A: Budget Parameter Reference

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `taut_timeout_s` | float | Per-statement truth-table timeout | 0.10 |
| `cycle_budget_s` | float | Total cycle wall-clock budget | 5.0 |
| `max_candidates_per_cycle` | int | Maximum candidates per cycle | 100 |

## Appendix B: Outcome Code Reference

| Code | Name | Meaning | Policy Effect |
|------|------|---------|---------------|
| `V` | Verified | Statement is a tautology | +reward |
| `R` | Refuted | Statement is not a tautology | -reward |
| `A_timeout` | Abstain (timeout) | Verification timed out | neutral |
| `A_complexity` | Abstain (complexity) | Formula too complex | neutral |
| `A_crash` | Abstain (crash) | Verifier crashed | neutral |
| `S_budget` | Skip (budget) | Cycle budget exhausted | none |
| `S_quota` | Skip (quota) | Candidate quota reached | none |

---

*End of Verifier Budget Theory document.*
