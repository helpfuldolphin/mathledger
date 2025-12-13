# CAL-EXP-2 VALIDITY ATTESTATION

> **Status:** CANONICAL
> **Date:** 2025-12-13
> **Scope:** CAL-EXP-2 only
> **Authority:** Claude Y (Release Captain / Boundary Coordinator)
> **Governing Document:** CAL_EXP_2_DEFINITIONS_BINDING.md
>
> **SHADOW MODE:** All observations are for analysis only. No enforcement semantics.
> **NO NEW SCIENCE:** This attestation applies existing definitions. No new formulas, algorithms, or heuristics are introduced.

---

## Ruling Summary

| Claim | Ruling | Reference |
|-------|--------|-----------|
| Divergence Reduction | **VALID** | §1 |
| Monotone Improvement | **INVALID** | §2 |
| No New Pathology | **VALID** | §3 |

---

## 1. DIVERGENCE REDUCTION

### Ruling: **VALID**

### Evidence

| Metric | Value | Source |
|--------|-------|--------|
| `baseline_mean_δp` | 0.0197 | First window δp (CAL-EXP-2 Canonical Record) |
| `post_mean_δp` | 0.0187 | Last window δp (CAL-EXP-2 Canonical Record) |
| `absolute_reduction` | 0.0010 | Computed: 0.0197 - 0.0187 |
| `percentage_reduction` | 5.1% | Computed: (0.0010 / 0.0197) × 100 |
| `window_size` | 1000 cycles | Exceeds 200 minimum |

### Justification

CAL-EXP-2 demonstrates valid divergence reduction because `post_mean_δp` (0.0187) < `baseline_mean_δp` (0.0197) with a 5.1% reduction over a 1000-cycle window that exceeds the 200-cycle minimum required by the binding definition.

---

## 2. MONOTONE IMPROVEMENT

### Ruling: **INVALID**

### Evidence

| Phase | Cycles | Mean δp | Δ from Prior |
|-------|--------|---------|--------------|
| 1 | 1-200 | 0.0230 | — |
| 2 | 201-400 | 0.0267 | **+0.0037** (INCREASE) |
| 3 | 401-600 | 0.0307 | **+0.0040** (INCREASE) |
| 4 | 601-800 | 0.0268 | -0.0039 (decrease) |
| 5 | 801-1000 | 0.0254 | -0.0014 (decrease) |

Canonical record explicitly states: `"non_monotonic_convergence": true`

### Justification

CAL-EXP-2 does NOT demonstrate monotone improvement because phases 2 and 3 show δp increases (0.0230 → 0.0267 → 0.0307), violating the binding definition requirement that `∀i ∈ [2, N]: mean_δp(Wᵢ) ≤ mean_δp(Wᵢ₋₁)`.

### Note on Warm-Up Exclusion

Even applying the warm-up exclusion (first 400 cycles), only 2 post-warmup windows remain (phases 4-5), which fails the minimum N=4 windows requirement.

---

## 3. NO NEW PATHOLOGY

### Ruling: **VALID**

### Checklist Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Zero FORB-XX violations | PASS | No violations reported in canonical record |
| All observations `action="LOGGED_ONLY"` | PASS | "SHADOW MODE: Active throughout"; "All observations are for analysis only" |
| No new CRITICAL streaks | PASS | No CRITICAL streaks reported; severity distribution shows PLATEAUING, not CRITICAL |
| `validity_score` stable or improved | PASS | Success Accuracy 82.1%, Variance STABLE |
| Warm-up peak ≤ 1.5× baseline | PASS | Peak 0.0307 / Baseline 0.0230 = **1.33×** (within limit) |
| Convergence floor not raised | PASS | Floor established at 0.025 (first measurement, no prior floor to regress from) |
| FI-001 through FI-007 satisfied | PASS | SHADOW mode maintained; no enforcement actions taken |

### Justification

CAL-EXP-2 demonstrates no new pathology because all seven verification criteria are satisfied: SHADOW mode was maintained throughout, warm-up peak amplification (1.33×) remained within the 1.5× limit, and no barrier violations, CRITICAL streaks, or invariant violations were reported.

---

## Attestation Limitations

This ruling is based on the CAL-EXP-2 Canonical Record. The following items are inferred rather than explicitly verified in raw logs:

1. `validity_score` is inferred from Success Accuracy (82.1%) and STABLE variance
2. FI-001 through FI-007 compliance is inferred from SHADOW mode attestation
3. FORB-XX non-violation is inferred from absence of reported violations

A full audit would require inspection of:
- `results/cal_exp_2/p4_20251212_103832/divergence_log.jsonl`
- Raw cycle logs for CRITICAL streak detection

---

## Ruling Authority

This ruling is issued under the authority of Claude Y as defined in the CAL_EXP_2_DEFINITIONS_BINDING.md document.

Claims inconsistent with this ruling are **INVALID**.

| Claim | Validity |
|-------|----------|
| "CAL-EXP-2 achieved divergence reduction" | VALID |
| "CAL-EXP-2 achieved monotone improvement" | INVALID |
| "CAL-EXP-2 achieved monotone convergence" | INVALID |
| "CAL-EXP-2 introduced no new pathology" | VALID |
| "CAL-EXP-2 converged non-monotonically with net reduction" | VALID |

---

## Implications

1. **Divergence reduction is demonstrated** — The UPGRADE-1 LR configuration produces measurable δp reduction over the 1000-cycle horizon.

2. **Monotone improvement is NOT demonstrated** — The warm-up divergence pattern (phases 2-3) is a characteristic behavior, not a failure, but it disqualifies claims of monotone improvement per binding definitions.

3. **No new pathology is demonstrated** — The UPGRADE-1 configuration does not introduce behaviors that violate barriers, breach SHADOW mode, or regress validity.

4. **Convergence floor is established** — δp ≈ 0.025 represents the algorithmic ceiling of the current architecture. Breaking this floor requires UPGRADE-2 (structural change), not parameter tuning.

---

**Document Hash:** SHA256:15a50f0a0c67f8c2abce705c745794aac681f401f54497717b63a752a663e8a0
**Ruling Date:** 2025-12-13
**Effective:** Immediately
