# Phase II: Verdict Invariance Under Auxiliary Perturbation

**Status**: PRE-FREEZE (question crystallization)
**Authority**: STRATCOM selection (Candidate B)
**Date**: 2025-12-19
**Mode**: SHADOW (observational only)
**Mutability**: Draft pending ratification

---

## 1. Scientific Question

**Given a fixed seed and frozen predicate set, is the governance verdict (F5.x codes, claim level, PASS/FAIL) invariant under perturbation of auxiliary parameters that are not part of the frozen governance contract?**

---

## 2. Question Decomposition

### 2.1 Core Terms

| Term | Definition |
|------|------------|
| **Governance verdict** | The tuple (F5.x codes, claim level, binary verdict) produced by the verification process |
| **Frozen predicate set** | The F5.1–F5.7 checks as defined in `CAL_EXP_4_FREEZE.md`, with their frozen thresholds |
| **Auxiliary parameter** | Any parameter that affects execution but is not part of the frozen predicate inputs |
| **Perturbation** | A controlled modification to an auxiliary parameter |
| **Invariance** | Identical governance verdict across perturbation conditions |

### 2.2 What the Question Tests

This question tests whether the governance verdict is fully determined by the frozen predicates, or whether it has hidden dependencies on implementation details outside the governance contract.

---

## 3. Governance-Level Predicate (Binary)

### 3.1 PASS Definition

```
VERDICT_INVARIANCE_PASS ≡
    ∀ perturbation p ∈ ADMISSIBLE_PERTURBATIONS:
        verdict(seed, p) = verdict(seed, p₀)
```

Where:
- `p₀` is the baseline (unperturbed) configuration
- `verdict(seed, p)` is the governance verdict under perturbation `p`
- Equality means identical F5.x codes, identical claim level, identical binary verdict

**In prose**: PASS if and only if all admissible perturbations produce the same governance verdict as the baseline.

### 3.2 FAIL Definition

```
VERDICT_INVARIANCE_FAIL ≡
    ∃ perturbation p ∈ ADMISSIBLE_PERTURBATIONS:
        verdict(seed, p) ≠ verdict(seed, p₀)
```

**In prose**: FAIL if any admissible perturbation produces a different governance verdict than the baseline.

### 3.3 Binary Verdict Only

This question produces exactly one of two outcomes: **PASS** or **FAIL**.

No PARTIAL verdict exists.

---

## 4. Auxiliary Parameter Perturbation

### 4.1 Conceptual Definition

An **auxiliary parameter** is any parameter that:

1. **Affects execution** — changing it produces a different execution trace
2. **Is not a frozen predicate input** — it is not one of the values consumed by F5.1–F5.7 checks
3. **Is not semantically load-bearing** — changing it should not, by design, affect the governance verdict

### 4.2 Categories of Auxiliary Parameters

| Category | Examples | Frozen? |
|----------|----------|---------|
| **Timestamp representation** | ISO8601 precision, timezone offset format | NO |
| **Serialization order** | JSON key ordering (beyond `sort_keys`) | NO |
| **Floating-point representation** | Decimal precision in output, scientific notation | NO |
| **Logging verbosity** | Debug output presence, log level | NO |
| **Filesystem metadata** | File creation time, path separators | NO |
| **Environment metadata** | Python version string format, platform details | NO |

### 4.3 What Is NOT an Auxiliary Parameter

| Not Auxiliary | Reason |
|---------------|--------|
| Seed | Frozen; directly affects corpus and execution |
| Threshold values | Frozen; part of predicate definitions |
| Cycle counts | Frozen; defines evaluation window |
| Learning rate | Frozen; defines arm differentiation |
| Variance profile | Frozen; registered in run_config |

### 4.4 Perturbation Admissibility

A perturbation is **admissible** if:

1. It modifies only auxiliary parameters (per §4.2)
2. It does not violate any artifact contract (files still valid JSON, etc.)
3. It does not modify frozen predicate inputs
4. It is deterministic (same perturbation always produces same effect)

---

## 5. Verdict Invariance vs Violation

### 5.1 Invariance (PASS condition)

Verdict invariance holds when:

| Component | Invariance Criterion |
|-----------|---------------------|
| F5.x codes | Identical set (order-independent) |
| Claim level | Identical (e.g., L0 = L0) |
| Binary verdict | Identical (PASS = PASS, FAIL = FAIL) |
| Temporal comparability | Identical boolean |
| Variance comparability | Identical boolean |

All components must be identical for invariance to hold.

### 5.2 Violation (FAIL condition)

Verdict violation occurs when any of the following differ between baseline and perturbed execution:

| Violation Type | Description |
|----------------|-------------|
| **Code divergence** | Different F5.x codes triggered |
| **Level divergence** | Different claim level assigned |
| **Verdict divergence** | Different binary verdict (PASS vs FAIL) |
| **Comparability divergence** | Different temporal or variance comparability |

Any single divergence constitutes a violation.

### 5.3 Divergence Classification

If FAIL occurs, the divergence should be classified:

| Class | Description |
|-------|-------------|
| **Semantic divergence** | Perturbation affected predicate computation (governance bug) |
| **Representation divergence** | Perturbation affected artifact format (contract gap) |
| **Ordering divergence** | Perturbation affected comparison order (implementation sensitivity) |

Classification is observational; no corrective action implied.

---

## 6. Explicit Non-Claims

### 6.1 What This Question Does NOT Ask

| Non-Claim | Reason |
|-----------|--------|
| Whether invariance is desirable | Value judgment outside scope |
| Whether violations should be fixed | No recommendations permitted |
| What perturbations should be tested | Implementation detail |
| How many perturbations are sufficient | Experiment design detail |
| Whether frozen predicates are correct | Predicates are assumed frozen |

### 6.2 What PASS Does NOT Establish

| Not Established by PASS | Reason |
|-------------------------|--------|
| Governance is correct | Invariance ≠ correctness |
| Thresholds are optimal | Thresholds not evaluated |
| System has capability | Outside scope |
| Predicates are complete | Invariance ≠ completeness |

### 6.3 What FAIL Does NOT Establish

| Not Established by FAIL | Reason |
|-------------------------|--------|
| Governance is broken | Sensitivity ≠ incorrectness |
| Changes are needed | No recommendations permitted |
| System lacks capability | Outside scope |
| Phase I was invalid | Phase I used fixed parameters |

---

## 7. Relationship to Phase I

### 7.1 Independence from Phase I Outcomes

| Phase I Outcome | Effect on Phase II |
|-----------------|-------------------|
| CAL-EXP-4/5 FAIL | No effect; Phase II tests invariance, not PASS achievement |
| F5.2/F5.3 triggered | No effect; Phase II tests whether same codes trigger under perturbation |
| L0 claim cap | No effect; Phase II tests whether L0 is invariant |

### 7.2 What Phase II Reuses from Phase I

| Reused | Source |
|--------|--------|
| Frozen thresholds | `CAL_EXP_4_FREEZE.md` |
| F5.x predicate definitions | `CAL_EXP_4_FREEZE.md` §3 |
| Artifact contract | `CAL_EXP_4_IMPLEMENTATION_PLAN.md` |
| Verifier logic | `scripts/verify_cal_exp_4_run.py` |

### 7.3 What Phase II Does NOT Reuse

| Not Reused | Reason |
|------------|--------|
| Phase I scientific questions | Phase II asks a different question |
| Phase I expected outcomes | Phase II has its own PASS/FAIL semantics |
| Phase I interpretation guardrails | Phase II requires its own guardrails |

---

## 8. Interpretation Guardrails

### 8.1 If PASS

**Permitted Statements**:
- "Governance verdict was invariant under all tested auxiliary perturbations"
- "No hidden dependencies on auxiliary parameters were detected"
- "Frozen predicates fully determined the verdict for tested perturbations"

**Forbidden Statements**:
- "Governance is robust" (overgeneralization beyond tested perturbations)
- "No auxiliary dependencies exist" (absence of evidence ≠ evidence of absence)
- Any capability or performance claim

### 8.2 If FAIL

**Permitted Statements**:
- "Governance verdict varied under auxiliary perturbation"
- "Specific perturbation(s) caused verdict divergence" (with details)
- "Hidden dependency on [parameter] detected"

**Forbidden Statements**:
- "Governance is fragile" (value judgment)
- "Fix required" (no recommendations)
- Any capability or performance claim

---

## 9. Binding Constraints

### 9.1 What This Document Defines

- The scientific question (§1)
- Binary verdict semantics (§3)
- Auxiliary parameter conceptual definition (§4)
- Invariance vs violation criteria (§5)
- Explicit non-claims (§6)
- Interpretation guardrails (§8)

### 9.2 What This Document Does NOT Define

- Specific perturbations to test
- Number of perturbation conditions
- Harness implementation
- Execution schedule
- Experiment numbering (CAL-EXP-N)

### 9.3 Document Status

| Property | Value |
|----------|-------|
| Status | PRE-FREEZE (draft) |
| Mutability | Pending STRATCOM ratification |
| Additive changes | Permitted (clarifications only) |
| Semantic changes | Require STRATCOM re-ratification |

---

**SHADOW MODE** — Observational only.

**No execution authorized until freeze ratification.**

*Precision > optimism.*
