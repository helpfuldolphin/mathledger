# CAL-EXP-3 Early Runs — Non-Canonical Record

---

## Status Block

| Field | Value |
|-------|-------|
| **Status** | NON-CANONICAL |
| **Authority** | None (observational record only) |
| **Date** | 2025-12-14 |
| **Spec Reference** | `CAL_EXP_3_UPLIFT_SPEC.md` |
| **Impl Reference** | `CAL_EXP_3_IMPLEMENTATION_PLAN.md` |

---

## Runs Recorded

### Run 1: seed=42

| Field | Value |
|-------|-------|
| **run_id** | `cal_exp_3_seed42_20251214_044612` |
| **seed** | 42 |
| **baseline_mean_delta_p** | 0.753962 |
| **treatment_mean_delta_p** | 0.786023 |
| **delta_delta_p** | +0.032061 |
| **evaluation_window** | cycles 201-1000 |
| **n_cycles** | 800 |
| **noise_floor** | 0.000870 |
| **exceeds_noise_floor** | true |
| **claim_level** | L4 |
| **validity_passed** | true |
| **isolation_audit_passed** | true |
| **verifier_checks** | 40 PASS, 0 FAIL, 0 WARN |

### Run 2: seed=43

| Field | Value |
|-------|-------|
| **run_id** | `cal_exp_3_seed43_20251214_044619` |
| **seed** | 43 |
| **baseline_mean_delta_p** | 0.742698 |
| **treatment_mean_delta_p** | 0.784877 |
| **delta_delta_p** | +0.042179 |
| **evaluation_window** | cycles 201-1000 |
| **n_cycles** | 800 |
| **noise_floor** | 0.001165 |
| **exceeds_noise_floor** | true |
| **claim_level** | L4 |
| **validity_passed** | true |
| **isolation_audit_passed** | true |
| **verifier_checks** | 40 PASS, 0 FAIL, 0 WARN |

### Run 3: seed=44

| Field | Value |
|-------|-------|
| **run_id** | `cal_exp_3_seed44_20251214_051658` |
| **seed** | 44 |
| **baseline_mean_delta_p** | 0.755755 |
| **treatment_mean_delta_p** | 0.786991 |
| **delta_delta_p** | +0.031235 |
| **evaluation_window** | cycles 201-1000 |
| **n_cycles** | 800 |
| **noise_floor** | 0.000853 |
| **exceeds_noise_floor** | true |
| **claim_level** | L4 |
| **validity_passed** | true |
| **isolation_audit_passed** | true |
| **verifier_checks** | 40 PASS, 0 FAIL, 0 WARN |

---

## Formula Verification

Per `CAL_EXP_3_UPLIFT_SPEC.md` Section "Formal Definition of Uplift":

```
delta_delta_p = treatment_mean_delta_p - baseline_mean_delta_p
```

| Run | Computation | Result |
|-----|-------------|--------|
| seed=42 | 0.786023 - 0.753962 | +0.032061 |
| seed=43 | 0.784877 - 0.742698 | +0.042179 |
| seed=44 | 0.786991 - 0.755755 | +0.031235 |

Positive delta_delta_p indicates treatment mean exceeds baseline mean.

---

## Contract Compliance

| Requirement | seed=42 | seed=43 | seed=44 |
|-------------|---------|---------|---------|
| Artifact layout per plan Section 4.1 | PASS | PASS | PASS |
| cycles.jsonl schema per plan Section 4.2 | PASS | PASS | PASS |
| validity/isolation_audit.json present | PASS | PASS | PASS |
| Window coverage (no missing cycles) | PASS | PASS | PASS |
| No external ingestion (fail-close) | PASS | PASS | PASS |
| Toolchain fingerprint identical | PASS | PASS | PASS |

---

## Claim Level Assignment

Per `CAL_EXP_3_UPLIFT_SPEC.md` Section "Claim Strength Ladder":

| Level | Requirement | seed=42 | seed=43 | seed=44 |
|-------|-------------|---------|---------|---------|
| L0 | Experiment completed | PASS | PASS | PASS |
| L1 | Measurements obtained | PASS | PASS | PASS |
| L2 | delta_delta_p computed | PASS | PASS | PASS |
| L3 | delta_delta_p exceeds noise floor | PASS | PASS | PASS |
| L4 | L3 + all validity conditions | PASS | PASS | PASS |
| L5 | L4 across >= 3 independent run-pairs | **THRESHOLD MET** | **THRESHOLD MET** | **THRESHOLD MET** |

**Note**: L5 requires >= 3 independent run-pairs per plan Section 5.2.1. With N=3 runs all at L4, the L5 threshold is met. However, this document remains NON-CANONICAL until formal review and sign-off.

---

## Observations

- In all three runs, delta_delta_p is positive, indicating treatment mean_delta_p exceeds baseline mean_delta_p.
- All three runs exceed their respective noise floors.
- Toolchain fingerprint is identical across all arms: `d173d4ddc637578b...`
- Cross-run delta_delta_p range: [+0.031235, +0.042179] (seed=44 lowest, seed=43 highest).

---

## Limitations

- Synthetic corpus; not representative of production workloads.
- No cross-run aggregation or statistical tests performed.
- No monotonicity claims: uplift measured in evaluation window only.
- L5 threshold met but document remains NON-CANONICAL pending formal review.

---

## Verifier Output Summary

```
seed=42: 40 checks, 0 FAIL, 0 WARN — VERDICT: PASS
seed=43: 40 checks, 0 FAIL, 0 WARN — VERDICT: PASS
seed=44: 40 checks, 0 FAIL, 0 WARN — VERDICT: PASS
```

---

## Git Status Proof

```
results/cal_exp_3/** is excluded via .gitignore
git status --short results/ returns empty (untracked)
```

---

**SHADOW MODE — observational only.**

---

## Execution Closure Note

| Field | Value |
|-------|-------|
| **Closure Date** | 2025-12-14 |
| **Authority** | STRATCOM CLOSURE directive |
| **Final Run Count** | 3 (seeds 42, 43, 44) |

- Execution complete.
- Replication threshold met (L5: ≥3 independent run-pairs at L4).
- Further runs require explicit STRATCOM authorization.

---

**Document Status**: NON-CANONICAL (observational record)
