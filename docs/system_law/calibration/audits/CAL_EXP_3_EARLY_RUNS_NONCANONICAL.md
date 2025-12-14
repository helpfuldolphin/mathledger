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

Positive delta_delta_p indicates treatment mean exceeds baseline mean.

---

## Contract Compliance

| Requirement | seed=42 | seed=43 |
|-------------|---------|---------|
| Artifact layout per plan Section 4.1 | PASS | PASS |
| cycles.jsonl schema per plan Section 4.2 | PASS | PASS |
| validity/isolation_audit.json present | PASS | PASS |
| Window coverage (no missing cycles) | PASS | PASS |
| No external ingestion (fail-close) | PASS | PASS |
| Toolchain fingerprint identical | PASS | PASS |

---

## Claim Level Assignment

Per `CAL_EXP_3_UPLIFT_SPEC.md` Section "Claim Strength Ladder":

| Level | Requirement | seed=42 | seed=43 |
|-------|-------------|---------|---------|
| L0 | Experiment completed | PASS | PASS |
| L1 | Measurements obtained | PASS | PASS |
| L2 | delta_delta_p computed | PASS | PASS |
| L3 | delta_delta_p exceeds noise floor | PASS | PASS |
| L4 | L3 + all validity conditions | PASS | PASS |
| L5 | L4 across >= 3 independent run-pairs | NOT MET | NOT MET |

**Note**: L5 requires >= 3 independent run-pairs per plan Section 5.2.1. With N=2 runs, L5 is not achievable. No replication claim is made.

---

## Observations

- In both runs, delta_delta_p is positive, indicating treatment mean_delta_p exceeds baseline mean_delta_p.
- Both runs exceed their respective noise floors.
- Toolchain fingerprint is identical across all arms: `d173d4ddc637578b...`

---

## Limitations

- N=2 runs are insufficient for L5 (replication) per spec.
- Synthetic corpus; not representative of production workloads.
- No cross-run aggregation or statistical tests performed.
- No monotonicity claims: uplift measured in evaluation window only.

---

## Verifier Output Summary

```
seed=42: 40 checks, 0 FAIL, 0 WARN — VERDICT: PASS
seed=43: 40 checks, 0 FAIL, 0 WARN — VERDICT: PASS
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

**Document Status**: NON-CANONICAL (observational record)
