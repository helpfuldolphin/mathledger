# CAL-EXP-5 Index

**Status**: EXECUTED (Phase 3 Complete)
**Phase**: CAL-EXP-5 (Phase-II FAIL-CLOSE Avoidance Test)
**Created**: 2025-12-19
**Predecessor**: CAL-EXP-4 (EXECUTED — Phase 3 Complete)

---

## Experiment Classification

**CAL-EXP-5 is a Phase-II FAIL-CLOSE avoidance test; no capability claims.**

| Property | Value |
|----------|-------|
| Type | FAIL-CLOSE avoidance test |
| Scope | Determine if variance-aligned conditions avoid FAIL-CLOSE |
| Claims permitted | FAIL-CLOSE avoidance claims only |
| Claims forbidden | Capability, intelligence, generalization, uplift claims |
| Pilot involvement | **FORBIDDEN** |
| Mode | SHADOW (observational only) |

---

## Experiment Summary

| Field | Value |
|-------|-------|
| Experiment ID | CAL-EXP-5 |
| Objective | Determine whether variance-aligned arm construction avoids FAIL-CLOSE |
| Scientific Question | Does the system avoid FAIL-CLOSE under variance-aligned conditions? |
| Verdict Type | Binary (PASS/FAIL only, no PARTIAL) |
| Horizon | Inherits CAL-EXP-3 (1000 cycles, 200 warm-up excluded) |
| Binding Spec | CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md |
| Implementation Plan | CAL_EXP_5_IMPLEMENTATION_PLAN.md |
| Freeze Declaration | CAL_EXP_5_FREEZE.md |

---

## Authoritative Documents

| Document | Status | Purpose |
|----------|--------|---------|
| [CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md](CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md) | **BINDING** | Charter, verdict semantics, interpretation guardrails |
| [CAL_EXP_5_IMPLEMENTATION_PLAN.md](CAL_EXP_5_IMPLEMENTATION_PLAN.md) | **FROZEN** | Execution machinery, artifact contract |
| [CAL_EXP_5_FREEZE.md](CAL_EXP_5_FREEZE.md) | **FROZEN** | Semantic freeze declaration |
| `scripts/verify_cal_exp_4_run.py` | **REUSED** | Verifier (CAL-EXP-4 verifier, no modification) |

---

## Relationship to CAL-EXP-4

| Aspect | CAL-EXP-4 | CAL-EXP-5 |
|--------|-----------|-----------|
| Question | "Does fail-close work correctly?" | "Can we avoid fail-close?" |
| Expected outcome | FAIL (prove fail-close triggers) | PASS (avoid fail-close) |
| Verifier | `verify_cal_exp_4_run.py` | Same verifier |
| Thresholds | Defined and frozen | Inherited |
| Schemas | Defined and frozen | Inherited |
| Output directory | `results/cal_exp_4/` | `results/cal_exp_5/` |

---

## Authoritative Source of Truth

| Authority | Source | Notes |
|-----------|--------|-------|
| Contract authority | `CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md` | Defines verdict semantics |
| Implementation authority | `CAL_EXP_5_IMPLEMENTATION_PLAN.md` | Defines execution machinery |
| Threshold authority | `CAL_EXP_4_FREEZE.md` | All thresholds inherited |
| Verifier authority | `scripts/verify_cal_exp_4_run.py` | Reused without modification |
| Index authority | This document | Single place for experiment status |

---

## Artifact Contract Block (Inherited + Extended)

**Source**: `CAL_EXP_5_IMPLEMENTATION_PLAN.md` Section 2

### Authoritative Run-Dir Shape

```
results/cal_exp_5/<run_id>/
├── run_config.json                          # Pre-registered seed, windows, variance profile
├── RUN_METADATA.json                        # Final verdict with F5.x status
├── baseline/
│   ├── cycles.jsonl                         # Per-cycle delta_p values (learning OFF)
│   └── summary.json                         # Baseline arm summary
├── treatment/
│   ├── cycles.jsonl                         # Per-cycle delta_p values (learning ON)
│   └── summary.json                         # Treatment arm summary
├── analysis/
│   ├── uplift_report.json                   # Inherited from CAL-EXP-3
│   └── windowed_analysis.json               # Inherited from CAL-EXP-3
└── validity/
    ├── toolchain_hash.txt                   # Inherited from CAL-EXP-3
    ├── corpus_manifest.json                 # Inherited from CAL-EXP-3
    ├── validity_checks.json                 # Inherited from CAL-EXP-3
    ├── isolation_audit.json                 # Inherited from CAL-EXP-3
    ├── temporal_structure_audit.json        # Inherited from CAL-EXP-4
    └── variance_profile_audit.json          # Inherited from CAL-EXP-4
```

### Required Files

| File | Why Required |
|------|--------------|
| `run_config.json` | Pre-registered seed, windows (F4.3 protection) |
| `RUN_METADATA.json` | Final verdict with `cal_exp_5_verdict` |
| `baseline/cycles.jsonl` | Per-cycle delta_p values for control arm |
| `treatment/cycles.jsonl` | Per-cycle delta_p values for learning arm |
| `validity/temporal_structure_audit.json` | F5.1 checks |
| `validity/variance_profile_audit.json` | F5.2, F5.3, F5.7 checks |

### Experiment Identifier

```json
{
  "experiment_id": "CAL-EXP-5"
}
```

---

## Verdict Semantics

### Binary Verdict

| Verdict | Definition |
|---------|------------|
| **PASS** | `f5_failure_codes ∩ FAIL_CLOSE_CODES = ∅` |
| **FAIL** | `f5_failure_codes ∩ FAIL_CLOSE_CODES ≠ ∅` |

### FAIL-CLOSE Codes

```
FAIL_CLOSE_CODES = {F5.1, F5.2, F5.4, F5.5, F5.6}
```

### WARN Codes (Non-Verdict-Affecting)

```
WARN_CODES = {F5.3, F5.7}
```

---

## Execution Gate

### Pre-Execution Requirements

| Step | Status | Artifact |
|------|--------|----------|
| 1. Spec frozen | **DONE** | `CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md` |
| 2. Implementation plan frozen | **DONE** | `CAL_EXP_5_IMPLEMENTATION_PLAN.md` |
| 3. Freeze declaration | **DONE** | `CAL_EXP_5_FREEZE.md` |
| 4. Index created | **DONE** | This document |
| 5. Pre-Execution Assertion | **DONE** | Issued 2025-12-19 |
| 6. Harness implementation | **DONE** | `scripts/run_cal_exp_5_harness.py` |
| 7. Execution | **DONE** | 3 runs completed |

### Execution Authorization

Execution authorized by STRATCOM directive 2025-12-19.

---

## Execution Record

### Run Inventory

| Run ID | Seed | Verdict | F5 Status | Artifact Path |
|--------|------|---------|-----------|---------------|
| `cal_exp_5_seed42_20251219_125232` | 42 | FAIL | F5.2, F5.3 | `results/cal_exp_5/cal_exp_5_seed42_20251219_125232/` |
| `cal_exp_5_seed43_20251219_125233` | 43 | FAIL | F5.2, F5.3 | `results/cal_exp_5/cal_exp_5_seed43_20251219_125233/` |
| `cal_exp_5_seed44_20251219_125235` | 44 | FAIL | F5.2, F5.3 | `results/cal_exp_5/cal_exp_5_seed44_20251219_125235/` |

### Execution Status

| Field | Value |
|-------|-------|
| Execution date | 2025-12-19 |
| Runs completed | 3 |
| Artifacts per run | 14 |
| Status | **COMPLETE** |

### Aggregate Verdict

| Metric | Value |
|--------|-------|
| Per-run verdicts | FAIL, FAIL, FAIL |
| Experiment-level verdict | **FAIL** |
| FAIL-CLOSE codes triggered | F5.2 (all runs) |
| WARN codes triggered | F5.3 (all runs) |
| Claim cap | L0 (all runs) |

---

## Change Control

Modifications to CAL-EXP-5 artifacts require:
1. Update to this index
2. Explicit rationale
3. STRATCOM approval for semantic changes

---

*This index is organizational and traceability only. It does not define new metrics, claims, or pilot logic.*

**SHADOW MODE** — observational only.

*Precision > optimism.*
