# CAL-EXP-2: P4 Divergence Minimization — Experiment Design

> **Status:** EXPERIMENT DESIGN (NON-CANONICAL)
> **Type:** Calibration Experiment — observational, not normative
> **Date:** 2025-12-13
> **Phase:** CAL-EXP-2

---

## Non-Canonical Notice

This document describes a **calibration experiment**, not a specification change.

- It does NOT modify CLI, exit codes, or artifact schemas
- It does NOT add new metrics or requirements
- It produces **observational data**, not normative definitions
- Results are recorded in `docs/system_law/calibration/audits/`

---

## 1. Canonical References (UNCHANGED)

The following documents govern `run_shadow_audit.py v0.1` and are **NOT modified** by this experiment:

| Document | Path | Role |
|----------|------|------|
| Canonical Contract | `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | CLI, exit codes, artifact layout |
| Scope Guardrails | `docs/system_law/calibration/RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md` | Enforcement protocol |
| Metrics Appendix | `docs/system_law/calibration/RUN_SHADOW_AUDIT_METRICS_V0_1.md` | Metric definitions |
| Metric Versioning | `docs/system_law/Metric_Versioning_Policy_v0.1.md` | Anti-laundering rules |
| Trust Binding | `docs/system_law/calibration/METRICS_AUDIT_TRUST_BINDING_V0_1.md` | Binding points |

---

## 2. DO NOT TOUCH

The following documents are **FROZEN** and MUST NOT be edited as part of CAL-EXP-2:

| Document | Status | Freeze Reason |
|----------|--------|---------------|
| `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | CANONICAL | Defines v0.1 interfaces |
| `RUN_SHADOW_AUDIT_SCOPE_GUARDRAILS.md` | ENFORCEMENT | PR gate protocol |
| `RUN_SHADOW_AUDIT_METRICS_V0_1.md` | FROZEN | Metric formulas locked |
| `Metric_Versioning_Policy_v0.1.md` | FROZEN | No new metrics |
| `METRICS_AUDIT_TRUST_BINDING_V0_1.md` | FROZEN | Binding structure locked |
| `Evidence_Pack_Spec_PhaseX.md` | FROZEN | Bundle schema locked |
| `RUN_SHADOW_AUDIT_V0_1_SPEC.md` | SUPERSEDED | Historical only |
| `RUN_SHADOW_AUDIT_V0_1_1_ADDENDUM.md` | MERGED | Historical only |
| `SHADOW_AUDIT_UX_SPEC_v0.1.md` | SUPERSEDED | Historical only |

**Violation of this list requires escalation and version bump.**

---

## 3. Experiment Objective

**Goal:** _[TO BE DEFINED]_

**Hypothesis:** _[TO BE DEFINED]_

**Success Criteria:** _[TO BE DEFINED]_

---

## 4. Methodology

### 4.1 Input Configuration

_[TO BE DEFINED]_

### 4.2 Execution Protocol

_[TO BE DEFINED]_

### 4.3 Measurement Points

_[TO BE DEFINED]_

---

## 5. Expected Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Raw results | `results/cal_exp_2/` | JSON/JSONL |
| Results summary | `docs/system_law/calibration/audits/CAL_EXP_2_RESULTS.md` | Markdown |

---

## 6. Constraints

| Constraint | Enforcement |
|------------|-------------|
| No new CLI flags | Contract frozen |
| No new exit codes | Contract frozen |
| No new required artifacts | Contract frozen |
| No metric formula changes | Metrics appendix frozen |
| SHADOW mode only | No enforcement semantics |

---

## 7. Sign-Off

| Role | Agent | Date | Status |
|------|-------|------|--------|
| Experiment Design | _[PENDING]_ | | |
| Canonical Compliance | CLAUDE Q | 2025-12-13 | Template ready |
| Execution | _[PENDING]_ | | |
| Results Review | _[PENDING]_ | | |

---

**END OF EXPERIMENT DESIGN TEMPLATE**
