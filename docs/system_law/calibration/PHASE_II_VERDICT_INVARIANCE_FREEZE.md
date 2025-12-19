# Phase II: Verdict Invariance — Semantic Freeze Declaration

**Status**: FROZEN
**Authority**: STRATCOM GOVERNANCE
**Effective Date**: 2025-12-19
**Freeze Version**: 1.0.0
**Mode**: SHADOW (observational only)

---

## Declaration

This document declares Phase II (Verdict Invariance Under Auxiliary Perturbation) semantics as **FROZEN**. All definitions, predicates, and interpretation guardrails specified in `PHASE_II_VERDICT_INVARIANCE_SPEC.md` are binding.

> **Any change to Phase II semantics requires explicit STRATCOM authorization.**

---

## 1. Scientific Question (Frozen)

**Given a fixed seed and frozen predicate set, is the governance verdict (F5.x codes, claim level, PASS/FAIL) invariant under perturbation of auxiliary parameters that are not part of the frozen governance contract?**

This question is frozen. No reformulation permitted without STRATCOM re-ratification.

---

## 2. Binary Verdict Predicate (Frozen)

### 2.1 PASS

```
VERDICT_INVARIANCE_PASS ≡
    ∀ perturbation p ∈ ADMISSIBLE_PERTURBATIONS:
        verdict(seed, p) = verdict(seed, p₀)
```

### 2.2 FAIL

```
VERDICT_INVARIANCE_FAIL ≡
    ∃ perturbation p ∈ ADMISSIBLE_PERTURBATIONS:
        verdict(seed, p) ≠ verdict(seed, p₀)
```

### 2.3 No PARTIAL Verdict

Phase II produces exactly one of two outcomes: **PASS** or **FAIL**.

---

## 3. Core Definitions (Frozen)

### 3.1 Governance Verdict

The tuple `(F5.x codes, claim level, binary verdict)` produced by the verification process.

| Component | Type | Example |
|-----------|------|---------|
| F5.x codes | Set of strings | `{"F5.2", "F5.3"}` |
| Claim level | String | `"L0"` |
| Binary verdict | String | `"FAIL"` |

### 3.2 Auxiliary Parameter

A parameter that:
1. Affects execution
2. Is not a frozen predicate input
3. Is not semantically load-bearing

### 3.3 Perturbation

A controlled modification to an auxiliary parameter that:
1. Modifies only auxiliary parameters
2. Does not violate artifact contracts
3. Does not modify frozen predicate inputs
4. Is deterministic

### 3.4 Invariance

Governance verdict is invariant when all components are identical:
- F5.x codes (set equality, order-independent)
- Claim level (string equality)
- Binary verdict (string equality)
- Temporal comparability (boolean equality)
- Variance comparability (boolean equality)

---

## 4. Auxiliary Parameter Categories (Frozen)

| Category | Auxiliary? | Rationale |
|----------|------------|-----------|
| Timestamp representation | YES | Not a predicate input |
| JSON serialization order | YES | Not a predicate input |
| Floating-point output precision | YES | Not a predicate input |
| Logging verbosity | YES | Not a predicate input |
| Filesystem metadata | YES | Not a predicate input |
| Environment metadata | YES | Not a predicate input |

### 4.1 Not Auxiliary (Frozen)

| Parameter | Reason |
|-----------|--------|
| Seed | Frozen predicate input |
| Threshold values | Frozen predicate definition |
| Cycle counts | Frozen evaluation window |
| Learning rate | Frozen arm differentiation |
| Variance profile | Registered in run_config |

---

## 5. Inherited Elements (Frozen)

Phase II inherits the following from Phase I without modification:

| Element | Source | Modification |
|---------|--------|--------------|
| F5.x predicate definitions | `CAL_EXP_4_FREEZE.md` §3 | NONE |
| Threshold values | `CAL_EXP_4_FREEZE.md` §2 | NONE |
| Schema versions | `CAL_EXP_4_FREEZE.md` §1 | NONE |
| Artifact contract | `CAL_EXP_4_IMPLEMENTATION_PLAN.md` | NONE |
| Verifier logic | `scripts/verify_cal_exp_4_run.py` | NONE |

---

## 6. Explicit Non-Claims (Frozen)

### 6.1 PASS Does Not Establish

| Non-Claim | Reason |
|-----------|--------|
| Governance is correct | Invariance ≠ correctness |
| Thresholds are optimal | Thresholds not evaluated |
| System has capability | Outside scope |
| Predicates are complete | Invariance ≠ completeness |
| All auxiliary dependencies tested | Finite perturbation set |

### 6.2 FAIL Does Not Establish

| Non-Claim | Reason |
|-----------|--------|
| Governance is broken | Sensitivity ≠ incorrectness |
| Changes are needed | No recommendations permitted |
| System lacks capability | Outside scope |
| Phase I was invalid | Phase I used fixed parameters |
| Specific fix required | No recommendations |

---

## 7. Interpretation Guardrails (Frozen)

### 7.1 If PASS

**Permitted Statements**:
- "Governance verdict was invariant under all tested auxiliary perturbations"
- "No hidden dependencies on auxiliary parameters were detected"
- "Frozen predicates fully determined the verdict for tested perturbations"

**Forbidden Statements**:
- "Governance is robust" (overgeneralization)
- "No auxiliary dependencies exist" (absence of evidence ≠ evidence of absence)
- Any capability or performance claim

### 7.2 If FAIL

**Permitted Statements**:
- "Governance verdict varied under auxiliary perturbation"
- "Specific perturbation(s) caused verdict divergence" (with details)
- "Hidden dependency on [parameter] detected"

**Forbidden Statements**:
- "Governance is fragile" (value judgment)
- "Fix required" (no recommendations)
- Any capability or performance claim

---

## 8. Change Control

### 8.1 Frozen Elements

The following elements are FROZEN and require STRATCOM authorization to modify:

- [ ] Scientific question (§1)
- [ ] Binary verdict predicate (§2)
- [ ] Core definitions (§3)
- [ ] Auxiliary parameter categories (§4)
- [ ] Inherited elements (§5)
- [ ] Explicit non-claims (§6)
- [ ] Interpretation guardrails (§7)

### 8.2 Non-Frozen Elements

The following may be modified without STRATCOM authorization:

- Harness implementation details (not affecting semantics)
- Specific perturbation selection (within frozen categories)
- Test fixtures (not affecting production)
- Documentation clarifications (not affecting semantics)

---

## 9. Attestation

This freeze declaration is binding for all Phase II operations.

| Role | Responsibility |
|------|----------------|
| Harness | Execute perturbations per frozen categories |
| Verifier | Produce verdict per frozen predicates |
| Analysts | Interpret per frozen guardrails |

---

**FROZEN** — No semantic changes without STRATCOM authorization.

**SHADOW MODE** — Observational only, non-gating.

*Precision > optimism.*
