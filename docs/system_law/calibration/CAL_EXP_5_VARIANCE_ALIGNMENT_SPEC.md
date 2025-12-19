# CAL-EXP-5: Variance Alignment — FAIL-CLOSE Avoidance Test

**Status**: SPECIFICATION (BINDING CHARTER)
**Authority**: STRATCOM
**Date**: 2025-12-19
**Scope**: CAL-EXP-5 only
**Mutability**: Frozen upon ratification
**Mode**: SHADOW (observational only)

---

## Objective

**Experiment Name**: CAL-EXP-5 — Variance Alignment Test

**Scientific Question**: Does the system avoid FAIL-CLOSE under variance-aligned conditions?

**Goal**: Determine whether variance-aligned arm construction avoids triggering FAIL-CLOSE F5.x codes.

### Scope Fence

CAL-EXP-5 is a **Phase-II FAIL-CLOSE avoidance test**. It is:

| CAL-EXP-5 IS | CAL-EXP-5 IS NOT |
|--------------|------------------|
| A FAIL-CLOSE avoidance test | A capability measurement |
| A variance alignment verification | An uplift experiment |
| A binary pass/fail determination | A performance benchmark |
| An extension of CAL-EXP-4 methodology | An introduction of new metrics |

**Relationship to CAL-EXP-4**: CAL-EXP-5 re-uses CAL-EXP-4 thresholds, schemas, and verifier. It tests whether FAIL-CLOSE can be avoided, whereas CAL-EXP-4 verified that FAIL-CLOSE is correctly triggered.

---

## Baseline and Treatment Arms

CAL-EXP-5 inherits the dual-arm architecture from CAL-EXP-3/CAL-EXP-4.

### Baseline Arm (Control)

| Property | Binding |
|----------|---------|
| Learning | **Disabled** |
| Parameters | Fixed (no adaptation) |
| Seed discipline | Identical to treatment |
| Toolchain fingerprint | Identical to treatment |
| Variance profile | **Identical to treatment** |

### Treatment Arm (Learning ON)

| Property | Binding |
|----------|---------|
| Learning | **Enabled** |
| Allowed knobs | ONLY those defined in CAL-EXP-3 canon |
| Initial state | Identical to baseline |
| Variance profile | **Identical to baseline** |

### Arm Differentiation Constraint

```
ARM_DIFFERENTIATION_CONSTRAINT ≡
    (baseline.learning_enabled = false) ∧
    (treatment.learning_enabled = true) ∧
    (baseline.corpus_hash = treatment.corpus_hash) ∧
    (baseline.seed = treatment.seed) ∧
    (baseline.variance_profile = treatment.variance_profile) ∧
    (baseline.toolchain_fingerprint = treatment.toolchain_fingerprint)
```

### Non-Degeneracy Constraint

```
NON_DEGENERACY ≡
    (treatment.learning_rate > 0) ∧
    (|mean(Δp_treatment) - mean(Δp_baseline)| > 0)
```

---

## Verdict Semantics

### Binary Verdict Only

CAL-EXP-5 produces exactly one of two verdicts: **PASS** or **FAIL**.

No PARTIAL verdict exists.

### FAIL-CLOSE Code Set

```
FAIL_CLOSE_CODES = {F5.1, F5.2, F5.4, F5.5, F5.6}
```

| Code | Name | Trigger |
|------|------|---------|
| F5.1 | Temporal Structure Incompatible | `temporal_structure_pass = false` |
| F5.2 | Variance Ratio Out of Bounds | `variance_ratio_acceptable = false` |
| F5.4 | Missing Audit Artifact | Required artifact not found |
| F5.5 | Schema Validation Failure | Malformed JSON, wrong version |
| F5.6 | Pathological Data | NaN/Inf detected |

### WARN Code Set

```
WARN_CODES = {F5.3, F5.7}
```

| Code | Name | Effect on Verdict | Effect on Claim |
|------|------|-------------------|-----------------|
| F5.3 | Windowed Drift Excessive | None | Cap to L3 |
| F5.7 | IQR Ratio Out of Bounds | None | Cap to L3 |

### Per-Run Verdict Predicate

```
CAL_EXP_5_RUN_PASS(run) ≡
    (F5_failure_codes ∩ FAIL_CLOSE_CODES = ∅)

CAL_EXP_5_RUN_FAIL(run) ≡
    (F5_failure_codes ∩ FAIL_CLOSE_CODES ≠ ∅)
```

### Experiment-Level Verdict Predicate

```
CAL_EXP_5_PASS ≡
    (∀ run ∈ runs: CAL_EXP_5_RUN_PASS(run))

CAL_EXP_5_FAIL ≡
    (∃ run ∈ runs: CAL_EXP_5_RUN_FAIL(run))
```

---

## Explicit Non-Claims

CAL-EXP-5 does NOT claim:

| Non-Claim | Rationale |
|-----------|-----------|
| Capability, intelligence, or generalization | Out of scope for all calibration experiments |
| Learning effectiveness or improvement | CAL-EXP-5 measures FAIL-CLOSE avoidance, not performance |
| Variance alignment implies correctness | Avoidance of FAIL-CLOSE is not a quality assertion |
| Threshold optimality | Thresholds are inherited from CAL-EXP-4 FREEZE |
| Superiority to any baseline or system | No comparison is performed |
| That PASS implies L4+ achievability | PASS only asserts FAIL-CLOSE avoidance |
| That FAIL implies system deficiency | FAIL only asserts FAIL-CLOSE triggered |

---

## Interpretation Guardrails

### If CAL-EXP-5 PASS

**Permitted Statements**:
- "No FAIL-CLOSE F5.x codes were triggered across all runs"
- "System avoided variance ratio incompatibility (F5.2) under these conditions"
- "Variance-aligned arm construction did not trigger FAIL-CLOSE"

**Forbidden Statements**:
- "System can produce variance-aligned arms" (overgeneralization)
- "Variance compatibility is solved" (overclaim)
- Any capability or performance assertion

### If CAL-EXP-5 FAIL

**Permitted Statements**:
- "FAIL-CLOSE F5.x codes were triggered"
- "System did not avoid variance incompatibility under these conditions"
- Specific F5.x codes and values observed

**Forbidden Statements**:
- "System cannot avoid variance incompatibility" (overgeneralization)
- "Thresholds need adjustment" (outside scope)
- Any capability or performance assertion

---

## Thresholds (Inherited from CAL-EXP-4)

All thresholds are inherited from `CAL_EXP_4_FREEZE.md` without modification.

| Parameter | Value | Source |
|-----------|-------|--------|
| `variance_ratio_max` | 2.0 | CAL-EXP-4 FREEZE §2.1 |
| `variance_ratio_min` | 0.5 | CAL-EXP-4 FREEZE §2.1 |
| `iqr_ratio_max` | 2.0 | CAL-EXP-4 FREEZE §2.1 |
| `windowed_drift_max` | 0.05 | CAL-EXP-4 FREEZE §2.1 |
| `min_coverage_ratio` | 1.0 | CAL-EXP-4 FREEZE §2.2 |
| `max_gap_ratio_divergence` | 0.1 | CAL-EXP-4 FREEZE §2.2 |

---

## Pre-Registration Requirements

### Seed Discipline (Inherited)

Per CAL-EXP-3: Seed must be registered before execution in `run_config.json`.

### Window Discipline (Inherited)

Per CAL-EXP-3: Evaluation windows must be pre-registered.

### Variance Profile Discipline (Inherited)

Per CAL-EXP-4: Variance profile parameters must be declared before execution.

---

## Binding Constraints

### What This Document Defines

- The meaning of "FAIL-CLOSE avoidance"
- Binary verdict semantics (PASS/FAIL only)
- Interpretation guardrails for both outcomes
- Inheritance relationship to CAL-EXP-4

### What This Document Does NOT Define

- Implementation details (code, scripts, harnesses)
- New metrics (none introduced)
- New thresholds (all inherited)
- Pilot logic or external data sources

### Document Status

| Property | Value |
|----------|-------|
| Status | SPECIFICATION (BINDING) |
| Mutability | Frozen upon STRATCOM ratification |
| Additive changes | Permitted (clarifications only) |
| Semantic changes | Require STRATCOM re-ratification |
| Implementation authority | NOT GRANTED by this document |

---

**SHADOW MODE** — observational only.

*Precision > optimism.*
