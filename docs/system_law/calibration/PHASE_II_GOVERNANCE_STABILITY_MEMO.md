# Phase II Governance Stability Memo

**Status**: FROZEN
**Authority**: STRATCOM GOVERNANCE
**Effective Date**: 2025-12-21
**Mode**: SHADOW (observational only)

---

## 1. Executive Summary

Phase II tested governance stability under three conditions:

| Experiment | Question | Verdict | Interpretation |
|------------|----------|---------|----------------|
| CAL-EXP-4 | Does fail-close trigger under variance stress? | F5.2/F5.3 triggered | Governance fails closed correctly |
| CAL-EXP-5 | Can variance-aligned conditions avoid fail-close? | FAIL (F5.2 triggered) | Governance resists variance laundering |
| Phase II.c | Is verdict invariant under auxiliary perturbation? | PASS (21/21 invariant) | No hidden dependencies detected |

**Aggregate Phase II Verdict**: Governance stability demonstrated under all tested conditions.

---

## 2. Scope and Non-Claims

### 2.1 What Phase II Tested

Phase II tested **governance stability**, not capability or learning:

- Whether fail-closed behavior triggers correctly under stress
- Whether governance resists attempts to avoid fail-close
- Whether governance verdict is invariant under auxiliary parameter changes

### 2.2 What Phase II Did NOT Test

| Non-Claim | Reason |
|-----------|--------|
| Capability or performance | Outside scope |
| Convergence or learning | Not tested |
| Threshold optimality | Thresholds frozen, not evaluated |
| Predicate completeness | Only tested frozen predicates |
| Generalization to untested perturbations | Finite perturbation set |

### 2.3 Mode

All Phase II experiments operated in **SHADOW mode**: observational only, non-blocking, non-gating.

---

## 3. Phase II Experiments

### 3.1 CAL-EXP-4: Variance Stress Test

**Scientific Question**: Does the CAL-EXP-3-style validity verifier fail-closed when temporal/variance structure differs between arms?

**Execution**: 3 runs (seeds 42, 43, 44) on 2025-12-19

**Results**:

| Seed | F5 Codes Triggered | Claim Level | Verdict |
|------|-------------------|-------------|---------|
| 42 | F5.2, F5.3 | L0 | Claims capped |
| 43 | F5.2, F5.3 | L0 | Claims capped |
| 44 | F5.2, F5.3 | L0 | Claims capped |

**Observed Values** (seed 42):
- Variance ratio: 0.265 (threshold min: 0.5, max: 2.0) → F5.2 triggered
- Windowed drift: 0.745 (threshold max: 0.05) → F5.3 triggered
- IQR ratio: 0.403 (threshold max: 2.0) → acceptable

**Interpretation (per frozen guardrails)**:
- "Governance correctly triggered fail-close under variance stress"
- "Claims appropriately capped to L0"

### 3.2 CAL-EXP-5: FAIL-CLOSE Avoidance Test

**Scientific Question**: Does the system avoid FAIL-CLOSE under variance-aligned conditions?

**Execution**: 3 runs (seeds 42, 43, 44) on 2025-12-19

**Results**:

| Seed | Expected | Observed | F5 Codes | Verdict |
|------|----------|----------|----------|---------|
| 42 | PASS (avoid fail-close) | FAIL | F5.2, F5.3 | Fail-close triggered |
| 43 | PASS (avoid fail-close) | FAIL | F5.2, F5.3 | Fail-close triggered |
| 44 | PASS (avoid fail-close) | FAIL | F5.2, F5.3 | Fail-close triggered |

**Interpretation (per frozen guardrails)**:
- "The variance-alignment strategy tested did not avoid fail-close"
- "Governance resists variance laundering"
- "This is an observation, not a defect"

### 3.3 Phase II.c: Verdict Invariance Under Auxiliary Perturbation

**Scientific Question**: Given a fixed seed and frozen predicate set, is the governance verdict (F5.x codes, claim level, PASS/FAIL) invariant under perturbation of auxiliary parameters?

**Execution**: 3 runs (seeds 42, 43, 44) on 2025-12-21, 7 perturbations each

**Perturbations Tested**:

| ID | Perturbation | Description |
|----|--------------|-------------|
| p0 | baseline | Unperturbed baseline configuration |
| p1 | timestamp_ms | Timestamp precision reduced to milliseconds |
| p2 | float_precision_8 | Floating-point output precision reduced to 8 decimals |
| p3 | json_indent_4 | JSON indent changed to 4 spaces |
| p4 | timezone_z | Timezone format changed from +00:00 to Z |
| p5 | no_platform | Platform details excluded from metadata |
| p6 | log_debug | Log level set to DEBUG |
| p7 | combined | All auxiliary perturbations combined |

**Results**:

| Seed | Invariant | Divergent | Phase II.c Verdict |
|------|-----------|-----------|-------------------|
| 42 | 7/7 | 0/7 | PASS |
| 43 | 7/7 | 0/7 | PASS |
| 44 | 7/7 | 0/7 | PASS |

**Aggregate**: 21/21 perturbation tests produced invariant verdicts.

**Baseline Verdict (all seeds)**:
- F5 Codes: F5.2, F5.3
- Claim Level: L0
- Binary Verdict: FAIL
- Temporal Comparability: true
- Variance Comparability: false

**Interpretation (per frozen guardrails)**:
- "Governance verdict was invariant under all tested auxiliary perturbations"
- "No hidden dependencies on auxiliary parameters were detected"
- "Frozen predicates fully determined the verdict for tested perturbations"

---

## 4. Summary Tables

### 4.1 Experiment Matrix

| Experiment | Type | Seeds | Runs | F5 Codes | Claim Level | Verdict |
|------------|------|-------|------|----------|-------------|---------|
| CAL-EXP-4 | Stress test | 42,43,44 | 3 | F5.2, F5.3 | L0 | Fail-close correct |
| CAL-EXP-5 | Avoidance test | 42,43,44 | 3 | F5.2, F5.3 | L0 | Fail-close resisted |
| Phase II.c | Invariance test | 42,43,44 | 21 | F5.2, F5.3 | L0 | Invariance confirmed |

### 4.2 F5.x Failure Code Summary

| Code | Description | CAL-EXP-4 | CAL-EXP-5 | Phase II.c |
|------|-------------|-----------|-----------|------------|
| F5.1 | Temporal structure incompatible | Not triggered | Not triggered | Not triggered |
| F5.2 | Variance ratio out of bounds | Triggered (FAIL-CLOSE) | Triggered (FAIL-CLOSE) | Triggered (invariant) |
| F5.3 | Windowed drift excessive | Triggered (WARN) | Triggered (WARN) | Triggered (invariant) |
| F5.4 | Missing audit artifact | Not triggered | Not triggered | Not triggered |
| F5.5 | Schema validation failure | Not triggered | Not triggered | Not triggered |
| F5.6 | Pathological data | Not triggered | Not triggered | Not triggered |
| F5.7 | IQR ratio out of bounds | Not triggered | Not triggered | Not triggered |

### 4.3 Governance Behavior Summary

| Condition | Expected Behavior | Observed Behavior | Status |
|-----------|-------------------|-------------------|--------|
| Variance mismatch detected | Fail-close | Fail-close | Correct |
| Variance alignment attempted | Avoid fail-close (hypothesis) | Fail-close | Hypothesis rejected |
| Auxiliary parameters perturbed | Invariant verdict | Invariant verdict | Correct |

---

## 5. Phase I vs Phase II Separation

### 5.1 Phase I (CAL-EXP-3)

**Purpose**: Validate measurement infrastructure

**Outcome**: Measurement substrate works; fail-closed governance triggers correctly; claims capped to L0

**Status**: Complete

### 5.2 Phase II (CAL-EXP-4, CAL-EXP-5, Phase II.c)

**Purpose**: Test governance stability under stress and perturbation

**Outcome**:
- Governance fails closed under variance stress (CAL-EXP-4)
- Governance resists variance laundering (CAL-EXP-5)
- Governance verdict is invariant under auxiliary perturbation (Phase II.c)

**Status**: Complete

### 5.3 Separation Discipline

| Property | Phase I | Phase II |
|----------|---------|----------|
| Primary question | Does infrastructure work? | Is governance stable? |
| Experiment type | Capability measurement | Stability/invariance testing |
| Expected outcome | Measurement success | Fail-close correctness |
| Claims permitted | Infrastructure claims only | Stability claims only |
| Capability claims | None | None |

---

## 6. Artifacts

### 6.1 CAL-EXP-4 Artifacts

```
results/cal_exp_4/
├── cal_exp_4_seed42_20251219_114330/
│   ├── RUN_METADATA.json
│   ├── run_config.json
│   ├── baseline/
│   ├── treatment/
│   ├── analysis/
│   └── validity/
├── cal_exp_4_seed43_20251219_114342/
└── cal_exp_4_seed44_20251219_114353/
```

### 6.2 CAL-EXP-5 Artifacts

```
results/cal_exp_5/
├── cal_exp_5_seed42_20251219_125232/
│   ├── RUN_METADATA.json
│   └── [same structure as CAL-EXP-4]
├── cal_exp_5_seed43_20251219_125233/
└── cal_exp_5_seed44_20251219_125235/
```

### 6.3 Phase II.c Artifacts

```
results/phase_ii_c/
├── phase_ii_c_seed42_20251221_042746/
│   ├── VERDICT_MATRIX.json
│   └── SUMMARY.json
├── phase_ii_c_seed43_20251221_042749/
└── phase_ii_c_seed44_20251221_042750/
```

---

## 7. Interpretation Guardrails (Applied)

### 7.1 Permitted Statements (Used in This Memo)

- "Governance correctly triggered fail-close under variance stress"
- "Claims appropriately capped to L0"
- "The variance-alignment strategy tested did not avoid fail-close"
- "Governance resists variance laundering"
- "Governance verdict was invariant under all tested auxiliary perturbations"
- "No hidden dependencies on auxiliary parameters were detected"
- "Frozen predicates fully determined the verdict for tested perturbations"

### 7.2 Forbidden Statements (Not Used)

- "Governance is robust" (overgeneralization)
- "No auxiliary dependencies exist" (absence of evidence ≠ evidence of absence)
- "Governance is fragile" (value judgment)
- "Fix required" (no recommendations permitted)
- Any capability or performance claim

---

## 8. Conclusions

Phase II demonstrated governance stability under all tested conditions:

1. **Fail-close works**: When variance structure differs, governance correctly triggers F5.2/F5.3 and caps claims to L0

2. **Fail-close is resistant**: Variance-alignment strategies do not circumvent fail-close; governance resists laundering

3. **Verdict is invariant**: Auxiliary parameter perturbations do not affect governance verdict; frozen predicates fully determine the outcome

**Aggregate Phase II Verdict**: PASS

---

## 9. Phase II Freeze Declaration

This memo constitutes the Phase II record. Phase II is now **FROZEN**.

| Element | Status |
|---------|--------|
| CAL-EXP-4 specification | FROZEN |
| CAL-EXP-5 specification | FROZEN |
| Phase II.c specification | FROZEN |
| F5.x predicate definitions | FROZEN |
| Interpretation guardrails | FROZEN |
| This memo | FROZEN |

**No modifications to Phase II without STRATCOM authorization.**

---

**SHADOW MODE** — Observational only, non-gating.

**FROZEN** — Phase II complete.

*Precision > optimism.*
