# CAL-EXP-5: Implementation Plan

**Status**: IMPLEMENTATION PLAN (FROZEN)
**Authority**: Derived from `CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md`
**Date**: 2025-12-19
**Scope**: Execution machinery only
**Mutability**: Frozen
**Mode**: SHADOW (observational only)

---

## Purpose

This document translates the binding charter (`CAL_EXP_5_VARIANCE_ALIGNMENT_SPEC.md`) into concrete execution steps. It specifies harness requirements, artifact contracts, and verifier usage.

**This document MUST NOT**:
- Introduce new metrics
- Add new thresholds
- Modify CAL-EXP-4 definitions
- Reference pilot or external data

**Goal**: Execute CAL-EXP-5 using CAL-EXP-4 infrastructure to determine FAIL-CLOSE avoidance.

---

## 1. Compatibility Statement

**CAL-EXP-5 uses CAL-EXP-4 verifier and schemas without modification.**

| Aspect | CAL-EXP-4 | CAL-EXP-5 |
|--------|-----------|-----------|
| Verifier | `verify_cal_exp_4_run.py` | Same verifier |
| Schemas | `temporal_structure_audit.schema.json`, `variance_profile_audit.schema.json` | Same schemas |
| Thresholds | CAL_EXP_4_FREEZE.md | Inherited |
| Artifact layout | `results/cal_exp_4/<run_id>/` | `results/cal_exp_5/<run_id>/` |

**Non-breaking guarantee**: The CAL-EXP-4 verifier is used as-is. The only difference is the output directory and `experiment_id` in artifacts.

---

## 2. Artifact Contract

### 2.1 Directory Structure

```
results/cal_exp_5/<run_id>/
├── run_config.json
├── RUN_METADATA.json
├── baseline/
│   ├── cycles.jsonl
│   └── summary.json
├── treatment/
│   ├── cycles.jsonl
│   └── summary.json
├── analysis/
│   ├── uplift_report.json
│   └── windowed_analysis.json
└── validity/
    ├── toolchain_hash.txt
    ├── corpus_manifest.json
    ├── validity_checks.json
    ├── isolation_audit.json
    ├── temporal_structure_audit.json
    └── variance_profile_audit.json
```

### 2.2 Required Artifacts

| File | Purpose |
|------|---------|
| `run_config.json` | Pre-registered seed, windows, variance profile |
| `RUN_METADATA.json` | Final verdict with F5.x status |
| `baseline/cycles.jsonl` | Per-cycle delta_p values (learning OFF) |
| `treatment/cycles.jsonl` | Per-cycle delta_p values (learning ON) |
| `validity/temporal_structure_audit.json` | F5.1 checks |
| `validity/variance_profile_audit.json` | F5.2, F5.3, F5.7 checks |

### 2.3 Experiment ID

All artifacts MUST use:

```json
{
  "experiment_id": "CAL-EXP-5"
}
```

---

## 3. Harness Requirements

### 3.1 Harness Extension

CAL-EXP-5 harness extends CAL-EXP-4 harness with:

| Change | Description |
|--------|-------------|
| `experiment_id` | Set to "CAL-EXP-5" |
| Output directory | `results/cal_exp_5/` instead of `results/cal_exp_4/` |
| Variance profile | Same as CAL-EXP-4 defaults |

### 3.2 No Algorithmic Changes

The harness MUST NOT modify:
- Learning rules
- Variance profile generation
- Arm execution logic
- Any threshold or parameter

### 3.3 Verdict Computation

```python
def compute_cal_exp_5_verdict(f5_failure_codes: List[str]) -> str:
    """
    Compute CAL-EXP-5 verdict per spec.

    PASS: No FAIL-CLOSE codes triggered
    FAIL: Any FAIL-CLOSE code triggered
    """
    FAIL_CLOSE_CODES = {"F5.1", "F5.2", "F5.4", "F5.5", "F5.6"}

    if set(f5_failure_codes) & FAIL_CLOSE_CODES:
        return "FAIL"
    return "PASS"
```

---

## 4. Verifier Usage

### 4.1 Verifier Command

```bash
python scripts/verify_cal_exp_4_run.py --run-dir results/cal_exp_5/<run_id>/
```

The CAL-EXP-4 verifier is used without modification.

### 4.2 Verdict Interpretation

| Verifier Output | CAL-EXP-5 Interpretation |
|-----------------|--------------------------|
| `f5_failure_codes ∩ FAIL_CLOSE_CODES = ∅` | PASS |
| `f5_failure_codes ∩ FAIL_CLOSE_CODES ≠ ∅` | FAIL |
| WARN codes only (F5.3, F5.7) | PASS (with claim cap) |

---

## 5. Execution Checklist

### 5.1 Pre-Execution

| Step | Action | Artifact |
|------|--------|----------|
| 1 | Register seed | `run_config.json` |
| 2 | Register windows | `run_config.json` |
| 3 | Register variance profile | `run_config.json` |
| 4 | Generate corpus | `validity/corpus_manifest.json` |
| 5 | Record toolchain hash | `validity/toolchain_hash.txt` |

### 5.2 Execution

| Step | Action | Artifact |
|------|--------|----------|
| 6 | Execute baseline arm | `baseline/cycles.jsonl` |
| 7 | Compute baseline summary | `baseline/summary.json` |
| 8 | Execute treatment arm | `treatment/cycles.jsonl` |
| 9 | Compute treatment summary | `treatment/summary.json` |

### 5.3 Post-Execution

| Step | Action | Artifact |
|------|--------|----------|
| 10 | Run inherited validity checks | `validity/validity_checks.json` |
| 11 | Run temporal structure audit | `validity/temporal_structure_audit.json` |
| 12 | Run variance profile audit | `validity/variance_profile_audit.json` |
| 13 | Compute ΔΔp | `analysis/uplift_report.json` |
| 14 | Assign verdict (PASS/FAIL) | `RUN_METADATA.json` |

---

## 6. RUN_METADATA.json Contract

```json
{
  "experiment": "CAL-EXP-5",
  "run_id": "<uuid>",
  "seed": 42,
  "mode": "SHADOW",
  "cal_exp_5_verdict": "<PASS|FAIL>",
  "f5_failure_codes": [],
  "fail_close_triggered": false,
  "warn_codes_triggered": [],
  "claim_cap_level": "<L0-L5|null>",
  "validity_passed": true,
  "generated_at": "<ISO8601>"
}
```

---

## 7. Smoke Checklist

### Commands to Run

```bash
# 1. Run CAL-EXP-5 harness
python scripts/run_cal_exp_5_harness.py --seed 42

# 2. Verify run artifacts using CAL-EXP-4 verifier
python scripts/verify_cal_exp_4_run.py --run-dir results/cal_exp_5/<run_id>/

# 3. Check verdict
python -c "import json; print(json.load(open('results/cal_exp_5/<run_id>/RUN_METADATA.json'))['cal_exp_5_verdict'])"
```

### Expected PASS Line

```
cal_exp_5_verdict: PASS
f5_failure_codes: []
fail_close_triggered: false
```

### Expected FAIL Line

```
cal_exp_5_verdict: FAIL
f5_failure_codes: ['F5.2']
fail_close_triggered: true
```

---

**SHADOW MODE** — observational only.

*Precision > optimism.*
