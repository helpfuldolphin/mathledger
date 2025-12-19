# Phase II Index

**Status**: FROZEN (Awaiting Execution Authorization)
**Phase**: Phase II (Verdict Invariance Under Auxiliary Perturbation)
**Created**: 2025-12-19
**Predecessor**: Phase I (CLOSED)

---

## Phase Classification

**Phase II tests governance stability; no capability claims.**

| Property | Value |
|----------|-------|
| Type | Governance invariance test |
| Scope | Determine if verdict is invariant under auxiliary perturbation |
| Claims permitted | Invariance/sensitivity observations only |
| Claims forbidden | Capability, intelligence, generalization, correctness claims |
| Pilot involvement | **FORBIDDEN** |
| Mode | SHADOW (observational only) |

---

## Experiment Summary

| Field | Value |
|-------|-------|
| Phase | Phase II |
| Scientific Question | Is governance verdict invariant under auxiliary parameter perturbation? |
| Verdict Type | Binary (PASS/FAIL only, no PARTIAL) |
| Inherited from Phase I | F5.x predicates, thresholds, schemas, verifier |
| Binding Spec | PHASE_II_VERDICT_INVARIANCE_SPEC.md |
| Freeze Declaration | PHASE_II_VERDICT_INVARIANCE_FREEZE.md |

---

## Authoritative Documents

| Document | Status | Purpose |
|----------|--------|---------|
| [PHASE_II_VERDICT_INVARIANCE_SPEC.md](PHASE_II_VERDICT_INVARIANCE_SPEC.md) | **RATIFIED** | Question crystallization, definitions |
| [PHASE_II_VERDICT_INVARIANCE_FREEZE.md](PHASE_II_VERDICT_INVARIANCE_FREEZE.md) | **FROZEN** | Semantic freeze declaration |
| [PHASE_II_QUESTION_SET.md](PHASE_II_QUESTION_SET.md) | HISTORICAL | Question selection record |
| [PHASE_I_CALIBRATION_CLOSURE.md](PHASE_I_CALIBRATION_CLOSURE.md) | CLOSED | Phase I closure classification |

---

## Relationship to Phase I

| Aspect | Phase I | Phase II |
|--------|---------|----------|
| Questions | Measurement, fail-close behavior | Governance stability |
| Tests | Predicate triggering | Predicate invariance |
| Inherited | — | All Phase I predicates/thresholds |
| Independent | — | Does not depend on Phase I outcomes |
| Reuses runs | — | **NO** (fresh execution required) |

---

## Authoritative Source of Truth

| Authority | Source | Notes |
|-----------|--------|-------|
| Question authority | `PHASE_II_VERDICT_INVARIANCE_SPEC.md` | Defines scientific question |
| Freeze authority | `PHASE_II_VERDICT_INVARIANCE_FREEZE.md` | Defines frozen semantics |
| Predicate authority | `CAL_EXP_4_FREEZE.md` | All F5.x predicates inherited |
| Threshold authority | `CAL_EXP_4_FREEZE.md` | All thresholds inherited |
| Verifier authority | `scripts/verify_cal_exp_4_run.py` | Reused without modification |
| Index authority | This document | Single place for phase status |

---

## Verdict Semantics

### Binary Verdict

| Verdict | Definition |
|---------|------------|
| **PASS** | All tested perturbations produce identical governance verdict |
| **FAIL** | Any perturbation produces different governance verdict |

### Verdict Components

A governance verdict consists of:

| Component | Type |
|-----------|------|
| F5.x codes | Set of failure codes |
| Claim level | L0-L5 |
| Binary verdict | PASS/FAIL |
| Temporal comparability | Boolean |
| Variance comparability | Boolean |

All components must be identical for invariance.

---

## Auxiliary Parameter Categories (Frozen)

| Category | Examples |
|----------|----------|
| Timestamp representation | ISO8601 precision, timezone format |
| JSON serialization order | Key ordering beyond sort_keys |
| Floating-point precision | Decimal places in output |
| Logging verbosity | Debug output presence |
| Filesystem metadata | File creation time |
| Environment metadata | Platform string format |

---

## Execution Gate

### Pre-Execution Requirements

| Step | Status | Artifact |
|------|--------|----------|
| 1. Question selection | **DONE** | `PHASE_II_QUESTION_SET.md` |
| 2. Spec ratified | **DONE** | `PHASE_II_VERDICT_INVARIANCE_SPEC.md` |
| 3. Freeze declaration | **DONE** | `PHASE_II_VERDICT_INVARIANCE_FREEZE.md` |
| 4. Index created | **DONE** | This document |
| 5. Harness plan | **PENDING** | Not yet prepared |
| 6. Execution authorization | **NOT GRANTED** | Awaiting STRATCOM |

### Execution Authorization

Execution requires:
1. All freeze documents committed
2. Harness plan reviewed
3. STRATCOM execution authorization

---

## Execution Record

### Run Inventory

| Run ID | Perturbation | Verdict | Divergence | Artifact Path |
|--------|--------------|---------|------------|---------------|
| (pending) | — | — | — | — |

### Execution Status

| Field | Value |
|-------|-------|
| Execution date | (pending) |
| Runs completed | 0 |
| Status | **FROZEN — AWAITING EXECUTION AUTHORIZATION** |

---

## Harness Plan Status

| Property | Status |
|----------|--------|
| Harness plan prepared | **NO** |
| Perturbation set defined | **NO** |
| Execution authorized | **NO** |

Harness plan preparation is authorized but not yet complete.

---

## Change Control

Modifications to Phase II artifacts require:
1. Update to this index
2. Explicit rationale
3. STRATCOM approval for semantic changes

---

*This index is organizational and traceability only. It does not define new metrics, claims, or pilot logic.*

**SHADOW MODE** — Observational only.

*Precision > optimism.*
