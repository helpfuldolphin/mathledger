# CAL-EXP-2 BINDING DEFINITIONS

> **Status:** CANONICAL
> **Date:** 2025-12-13
> **Scope:** CAL-EXP-2 only
> **Authority:** Claude Y (Release Captain / Boundary Coordinator)
>
> **SHADOW MODE:** All observations are for analysis only. No enforcement semantics.
> **NO NEW SCIENCE:** These definitions use existing metrics and thresholds. No new formulas, algorithms, or heuristics are introduced.

---

## Constraints

- No new capabilities
- No narrative changes
- No pilot execution
- No enforcement semantics
- No scope creep

---

## 1. DIVERGENCE REDUCTION

### Definition

**Divergence reduction** occurs when, for a given intervention X applied to the system:

```
mean_δp(post_intervention) < mean_δp(baseline)
```

Where:
- `δp = |Δp_real(t) - Δp_twin(t)|` (per Phase_X_Divergence_Metric.md §2.1)
- `baseline` = CAL-EXP-1 canonical metrics OR pre-intervention window
- `post_intervention` = measurement window after intervention applied
- Window size ≥ 200 cycles (to account for warm-up divergence phases 2-3)

### Required Metrics

| Metric | Description |
|--------|-------------|
| `baseline_mean_δp` | Numeric value from pre-intervention run |
| `post_mean_δp` | Numeric value from post-intervention run |
| `absolute_reduction` | `baseline_mean_δp - post_mean_δp` |
| `percentage_reduction` | `(absolute_reduction / baseline_mean_δp) * 100` |
| `window_size` | Integer ≥ 200 |

---

## 2. MONOTONE IMPROVEMENT

### Definition

**Monotone improvement** occurs when, for a sequence of N consecutive measurement windows W₁, W₂, ..., Wₙ:

```
∀i ∈ [2, N]: mean_δp(Wᵢ) ≤ mean_δp(Wᵢ₋₁)
```

**Strict monotone improvement** requires:

```
∀i ∈ [2, N]: mean_δp(Wᵢ) < mean_δp(Wᵢ₋₁)
```

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Minimum window size | 50 cycles | Per CAL-EXP-2 config |
| Minimum N | 4 windows | Avoid noise artifacts |
| Warm-up exclusion | First 400 cycles | Known diverging behavior (phases 2-3) |

### ε-Monotone Clause

**ε-monotone improvement** (relaxed definition) allows:

```
∀i ∈ [2, N]: mean_δp(Wᵢ) ≤ mean_δp(Wᵢ₋₁) + ε
```

Where `ε = 0.001` (numerical precision floor per Phase_X_Divergence_Metric.md §2.2)

**Usage requirement:** Must explicitly state if ε-monotone is used: "ε-monotone improvement (ε=0.001)"

---

## 3. NO NEW PATHOLOGY

### Definition

**No new pathology** is satisfied when an intervention introduces ZERO behaviors meeting ANY of the following criteria:

| Pathology Class | Detection Criterion |
|-----------------|---------------------|
| **Barrier Violation** | Any FORB-XX forbidden edge crossed |
| **SHADOW Mode Breach** | `action ≠ "LOGGED_ONLY"` for any observation |
| **New CRITICAL Streak** | ≥5 consecutive cycles with `severity="CRITICAL"` not present in baseline |
| **Validity Regression** | `validity_score(post) < validity_score(baseline)` |
| **Warm-up Amplification** | `max_δp(phases 2-3, post) > max_δp(phases 2-3, baseline) * 1.5` |
| **Floor Regression** | `convergence_floor(post) > convergence_floor(baseline)` |
| **Invariant Violation** | Any of FI-001 through FI-007 violated |

### Verification Checklist

A claim of "no new pathology" requires ALL of the following:

- [ ] Zero FORB-XX violations in logs
- [ ] All observations have `action="LOGGED_ONLY"`
- [ ] No new CRITICAL streaks (≥5 consecutive)
- [ ] `validity_score` stable or improved
- [ ] Warm-up peak not amplified beyond 1.5× baseline
- [ ] Convergence floor not raised
- [ ] All frozen invariants FI-001 through FI-007 satisfied

---

## 4. INVALID CLAIMS

The following claims are **INVALID** and must be rejected:

### Divergence Reduction

- "Divergence reduced" without numeric before/after values
- Comparing within warm-up phase (cycles < 400) to post-warmup
- Single-cycle improvements cited as "reduction"
- "Reduction" based on cherry-picked window excluding peaks
- Confusing severity band changes with metric reduction

### Monotone Improvement

- Claiming monotone improvement when ANY window shows δp increase (excluding ε tolerance)
- Claiming monotone improvement during warm-up phase (cycles 1-400)
- Using single-cycle comparisons instead of window aggregates
- "Eventually monotone" — either the full post-warmup sequence is monotone or it is not
- Ignoring phases 2-3 divergence when characterizing overall trajectory

### No New Pathology

- "No new pathology" without explicit verification of all checklist items
- "Pathology-free" when validity_score dropped (even if still > 0.8)
- Ignoring warm-up phase behavior when assessing pathology
- "Acceptable pathology" — pathology is binary (present/absent)

### General

- Claiming divergence reduction alone establishes broader system readiness

---

## 5. BINDING CONSTRAINTS

### CAL-EXP-2 MAY

| Action | Constraint |
|--------|------------|
| Adjust learning rates (LR_H, LR_ρ, LR_τ, LR_β) | Within existing LR framework |
| Extend observation horizon | Cycles only, no structural changes |
| Compare divergence across configurations | Read-only, no enforcement |
| Document convergence floor | Observation only |

### CAL-EXP-2 MAY NOT

| Forbidden Action | Reason |
|------------------|--------|
| Introduce new divergence metrics | No new science |
| Change severity thresholds | Frozen in Phase_X_Divergence_Metric.md |
| Add enforcement based on divergence | SHADOW mode invariant |
| Modify convergence floor algorithmically | Requires UPGRADE-2 spec (separate) |

---

## Authority

These definitions are issued by Claude Y (Release Captain / Boundary Coordinator).

Any claim of "divergence reduction," "monotone improvement," or "no new pathology" that does not conform to these definitions is **INVALID**.

---

## References

| Document | Binding To |
|----------|------------|
| Phase_X_Divergence_Metric.md | δp definition, severity thresholds, validity_score |
| RUN_SHADOW_AUDIT_V0_1_SPEC.md | Frozen invariants FI-001 through FI-007 |
| MATHLEDGER_RESPONSIBILITY_BOUNDARY.md | FORB-XX forbidden edges |
| CAL_EXP_2_Canonical_Record.md | Baseline metrics, warm-up behavior |

---

**Document Hash:** SHA256:28bf67e1b4e81784a582e2ca3ddce8c84fcfd01cef403817e60558afb2aaff6a
**Effective:** 2025-12-13
**Supersedes:** None (first issuance)
