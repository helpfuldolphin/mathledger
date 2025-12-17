# CAL-EXP-4: Implementation Plan

**Status**: IMPLEMENTATION PLAN (PROVISIONAL)
**Authority**: Derived from `CAL_EXP_4_VARIANCE_STRESS_SPEC.md`
**Date**: 2025-12-14
**Scope**: Execution machinery only
**Mutability**: Editable until execution begins
**Mode**: SHADOW (observational only)

---

## Purpose

This document translates the binding charter (`CAL_EXP_4_VARIANCE_STRESS_SPEC.md`) into concrete execution steps. It specifies harness extensions, artifact contract additions, and verifier logic for detecting temporal structure mismatches.

**This document MUST NOT**:
- Introduce new metrics
- Add new claims beyond the spec
- Modify frozen CAL-EXP-3 definitions
- Reference pilot or external data

**Goal**: Extend CAL-EXP-3 machinery to detect temporal structure mismatches without breaking existing invariants.

---

## 1. Compatibility Statement

**CAL-EXP-4 verifier is an extension of CAL-EXP-3 verifier logic; artifact layout is consistent.**

| Aspect | CAL-EXP-3 | CAL-EXP-4 Extension |
|--------|-----------|---------------------|
| Base verifier | `verify_cal_exp_3_run.py` | Extends with F5.x checks |
| Artifact layout | `results/cal_exp_3/<run_id>/` | Same structure + `validity/temporal_structure_audit.json` |
| Claim ladder | L0-L5 | Same ladder, with capping rules for F5.x failures |
| Validity checks | F1.1-F2.3 | Adds F5.1-F5.5 |

**Non-breaking guarantee**: A CAL-EXP-3 run that passes the CAL-EXP-3 verifier will also pass CAL-EXP-4's inherited checks. CAL-EXP-4 adds additional checks but does not modify existing ones.

---

## 2. Artifact Contract Extension

### 2.1 Extended Directory Structure

```
results/cal_exp_4/<run_id>/
├── run_config.json           # Extended with variance_profile
├── baseline/
│   ├── cycles.jsonl          # Per-cycle Δp values (learning OFF)
│   └── summary.json          # Baseline arm summary statistics
├── treatment/
│   ├── cycles.jsonl          # Per-cycle Δp values (learning ON)
│   └── summary.json          # Treatment arm summary statistics
├── analysis/
│   ├── uplift_report.json    # ΔΔp computation (inherited)
│   └── windowed_analysis.json # Per-window breakdown (inherited)
├── validity/
│   ├── toolchain_hash.txt    # SHA-256 of runtime environment (inherited)
│   ├── corpus_manifest.json  # Hash of input corpus (inherited)
│   ├── validity_checks.json  # Pass/fail for F1.x-F2.x (inherited)
│   ├── isolation_audit.json  # Network/filesystem isolation (inherited)
│   └── temporal_structure_audit.json  # NEW: F5.x checks
└── RUN_METADATA.json         # Final verdict with F5.x status
```

### 2.2 New Required Artifact: temporal_structure_audit.json

**Location**: `validity/temporal_structure_audit.json`

**Schema**:

```json
{
  "schema_version": "1.0.0",
  "experiment": "CAL-EXP-4",
  "baseline_profile": {
    "noise_scale": 0.03,
    "drift_rate": 0.002,
    "spike_probability": 0.05,
    "computed_variance": 0.0025,
    "lag1_autocorrelation": 0.15
  },
  "treatment_profile": {
    "noise_scale": 0.03,
    "drift_rate": 0.002,
    "spike_probability": 0.05,
    "computed_variance": 0.0024,
    "lag1_autocorrelation": 0.14
  },
  "comparability_checks": {
    "noise_scale_match": true,
    "drift_rate_match": true,
    "spike_probability_match": true,
    "variance_ratio": 1.04,
    "autocorrelation_diff": 0.01
  },
  "window_volatility": {
    "W1_baseline_var": 0.0023,
    "W1_treatment_var": 0.0024,
    "W2_baseline_var": 0.0026,
    "W2_treatment_var": 0.0025,
    "W3_baseline_var": 0.0024,
    "W3_treatment_var": 0.0023,
    "W4_baseline_var": 0.0025,
    "W4_treatment_var": 0.0026,
    "max_ratio": 1.08
  },
  "verdict": {
    "temporal_comparability": true,
    "failures_detected": [],
    "claim_cap": null
  },
  "generated_at": "<ISO8601>"
}
```

### 2.3 Extended run_config.json

**Additional required fields**:

```json
{
  "experiment": "CAL-EXP-4",
  "spec_reference": "CAL_EXP_4_VARIANCE_STRESS_SPEC.md",
  "seed": 42,
  "cycles": 1000,
  "windows": { ... },
  "baseline_config": {
    "learning_enabled": false,
    "rfl_active": false
  },
  "treatment_config": {
    "learning_enabled": true,
    "rfl_active": true
  },
  "variance_profile": {
    "noise_scale": 0.03,
    "drift_rate": 0.002,
    "spike_probability": 0.05,
    "registered_at": "<ISO8601>"
  },
  "registered_at": "<ISO8601>"
}
```

### 2.4 Extended RUN_METADATA.json

**Additional fields**:

```json
{
  "experiment": "CAL-EXP-4",
  "run_id": "<uuid>",
  "verdict": "<L0|L1|L2|L3|L4|L5>",
  "delta_delta_p": 0.035,
  "validity_passed": true,
  "temporal_comparability_passed": true,
  "f5_failures": [],
  "claim_permitted": "<string>",
  "generated_at": "<ISO8601>"
}
```

---

## 3. Harness Requirements

### 3.1 What the Harness Must Log

The harness must record sufficient information for the verifier to detect F5.x failures:

| Data Point | Source | Purpose |
|------------|--------|---------|
| `variance_profile` parameters | Pre-registered in run_config.json | F5.1 check |
| Per-cycle Δp values | cycles.jsonl | Compute actual variance |
| Per-window statistics | windowed_analysis.json | F5.3 check |

### 3.2 Temporal Structure Computation

The harness must compute and log (no new metrics—descriptive statistics only):

| Statistic | Formula | Threshold |
|-----------|---------|-----------|
| Computed variance | `var(Δp_values)` | Compare between arms |
| Variance ratio | `max(var_b, var_t) / min(var_b, var_t)` | Mismatch if > 2.0 |
| Lag-1 autocorrelation | `corr(Δp[1:], Δp[:-1])` | Mismatch if abs(diff) > 0.2 |
| Window volatility ratio | `max(w_var) / min(w_var)` across sub-windows | Mismatch if > 2.0 |

**Note**: These are descriptive statistics for validity checking, not new metrics. They do not appear in claims.

### 3.3 Fail-Close Implementation

The harness must implement fail-close behavior:

```python
def compute_temporal_structure_audit(baseline_cycles, treatment_cycles, config):
    """
    Compute temporal structure audit for F5.x checks.

    Returns verdict with temporal_comparability: bool and any claim_cap.
    """
    failures = []

    # F5.1: Variance profile mismatch (from config)
    # Already matched by design if same corpus used

    # F5.2: Compute and compare variances
    baseline_var = compute_variance(baseline_cycles)
    treatment_var = compute_variance(treatment_cycles)
    variance_ratio = max(baseline_var, treatment_var) / min(baseline_var, treatment_var)
    if variance_ratio > 2.0:
        failures.append("F5.1: variance_ratio exceeds 2.0")

    # F5.3: Compute and compare autocorrelations
    baseline_autocorr = compute_lag1_autocorrelation(baseline_cycles)
    treatment_autocorr = compute_lag1_autocorrelation(treatment_cycles)
    autocorr_diff = abs(baseline_autocorr - treatment_autocorr)
    if autocorr_diff > 0.2:
        failures.append("F5.2: autocorrelation_diff exceeds 0.2")

    # F5.3: Window volatility
    window_volatility = compute_window_volatility(baseline_cycles, treatment_cycles)
    if window_volatility["max_ratio"] > 2.0:
        failures.append("F5.3: window_volatility_ratio exceeds 2.0")

    # Determine verdict
    if failures:
        return {
            "temporal_comparability": False,
            "failures_detected": failures,
            "claim_cap": "L3"
        }

    return {
        "temporal_comparability": True,
        "failures_detected": [],
        "claim_cap": None
    }
```

---

## 4. Verifier Extensions

### 4.1 F5.x Check Implementation

The CAL-EXP-4 verifier extends CAL-EXP-3's verifier with:

```python
def check_temporal_structure(run_dir: Path) -> CheckResult:
    """
    Check F5.x: Temporal structure comparability.

    Per CAL_EXP_4_VARIANCE_STRESS_SPEC.md:
    - F5.1: Variance profile mismatch
    - F5.2: Autocorrelation mismatch
    - F5.3: Window volatility mismatch
    - F5.4: Temporal audit missing
    - F5.5: Temporal audit inconclusive
    """
    audit_path = run_dir / "validity" / "temporal_structure_audit.json"

    # F5.4: Check audit exists
    if not audit_path.exists():
        return CheckResult(
            name="F5.4_temporal_audit_missing",
            passed=False,
            expected="validity/temporal_structure_audit.json exists",
            actual="file not found",
            invalidates=False,  # Caps at L2, doesn't invalidate
            claim_cap="L2"
        )

    audit_data = load_json(audit_path)

    # F5.5: Check audit is conclusive
    verdict = audit_data.get("verdict", {})
    if verdict.get("temporal_comparability") is None:
        return CheckResult(
            name="F5.5_temporal_audit_inconclusive",
            passed=False,
            expected="temporal_comparability determined",
            actual="inconclusive",
            invalidates=False,
            claim_cap="L3"
        )

    # F5.1-F5.3: Check for detected failures
    failures = verdict.get("failures_detected", [])
    if failures:
        return CheckResult(
            name="F5.1-F5.3_temporal_mismatch",
            passed=False,
            expected="no temporal structure mismatches",
            actual=f"{len(failures)} failures: {failures}",
            invalidates=False,  # Caps claim, doesn't invalidate
            claim_cap=verdict.get("claim_cap", "L3")
        )

    return CheckResult(
        name="F5.x_temporal_comparability",
        passed=True,
        expected="temporal_comparability=true",
        actual="temporal_comparability=true",
        invalidates=False,
        claim_cap=None
    )
```

### 4.2 Claim Level Assignment with F5.x

```python
def assign_claim_level_cal_exp_4(report: dict, validity: dict, temporal: CheckResult) -> str:
    """
    Assign claim level per CAL-EXP-4 spec.

    Inherits CAL-EXP-3 logic, adds F5.x capping.
    """
    # Run CAL-EXP-3 logic first
    base_level = assign_claim_level_cal_exp_3(report, validity)

    # Apply F5.x capping
    if temporal.claim_cap:
        level_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
        base_idx = level_order.index(base_level)
        cap_idx = level_order.index(temporal.claim_cap)
        return level_order[min(base_idx, cap_idx)]

    return base_level
```

---

## 5. Execution Checklist

### 5.1 Pre-Execution

| Step | Action | Artifact |
|------|--------|----------|
| 1 | Register seed | `run_config.json` |
| 2 | Register windows | `run_config.json` |
| 3 | **Register variance profile** | `run_config.json` (NEW) |
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
| 10 | Run CAL-EXP-3 validity checks | `validity/validity_checks.json` |
| 11 | **Run temporal structure audit** | `validity/temporal_structure_audit.json` (NEW) |
| 12 | Compute ΔΔp | `analysis/uplift_report.json` |
| 13 | Compute windowed analysis | `analysis/windowed_analysis.json` |
| 14 | Assign claim level (with F5.x capping) | `RUN_METADATA.json` |

---

## 6. Failure Detection

### 6.1 Automated Detection

| Failure | Detection Method | Response |
|---------|------------------|----------|
| F1.1-F2.3 | Inherited from CAL-EXP-3 | Abort, report invalid |
| **F5.1 Variance profile mismatch** | Config + computed variance comparison | Cap at L3 |
| **F5.2 Autocorrelation mismatch** | Lag-1 autocorrelation diff > 0.2 | Cap at L3 |
| **F5.3 Window volatility mismatch** | Max window variance ratio > 2.0 | Cap at L3 |
| **F5.4 Temporal audit missing** | File existence check | Cap at L2 |
| **F5.5 Temporal audit inconclusive** | Verdict field check | Cap at L3 |

### 6.2 Manual Review Required

| Failure | Reason |
|---------|--------|
| F2.4 Observer effect | Requires instrumentation review |
| F3.1-F3.4 Interpretive | Requires human judgment on claims |
| Materiality threshold disputes | May require STRATCOM guidance |

---

## 7. Test Scenarios

### 7.1 Expected PASS Scenarios

| Scenario | Variance Profile | Expected Result |
|----------|------------------|-----------------|
| Identical profiles | noise=0.03, drift=0.002, spike=0.05 both arms | L4 possible |
| Near-identical profiles | variance_ratio=1.1, autocorr_diff=0.05 | L4 possible |

### 7.2 Expected FAIL/CAP Scenarios

| Scenario | Variance Profile | Expected Result |
|----------|------------------|-----------------|
| High variance ratio | baseline noise=0.01, treatment noise=0.05 | Capped at L3 |
| Autocorrelation mismatch | baseline autocorr=0.1, treatment autocorr=0.4 | Capped at L3 |
| Missing audit | No temporal_structure_audit.json | Capped at L2 |

---

## 8. Smoke Checklist

### Commands to Run

```bash
# 1. Run CAL-EXP-4 harness (when implemented)
python scripts/run_cal_exp_4_canonical.py --seed 42 --cycles 1000

# 2. Verify run artifacts
python scripts/verify_cal_exp_4_run.py --run-dir results/cal_exp_4/<run_id>/

# 3. Check temporal_structure_audit.json exists
ls results/cal_exp_4/<run_id>/validity/temporal_structure_audit.json

# 4. Validate JSON schema
python -c "import json; json.load(open('results/cal_exp_4/<run_id>/validity/temporal_structure_audit.json'))"
```

### Expected Pass Line

```
=== CAL-EXP-4 RUN VERIFICATION ===
Run Directory: results/cal_exp_4/<run_id>/
...
[PASS] F1.1_toolchain_parity: ...
[PASS] F1.2_corpus_identity: ...
[PASS] F2.3_external_ingestion: ...
[PASS] F5.x_temporal_comparability: temporal_comparability=true
...
SUMMARY: N checks, 0 FAIL, 0 WARN
VERDICT: PASS
```

### Expected Cap Line (Mismatch Scenario)

```
[WARN] F5.1-F5.3_temporal_mismatch: expected=no temporal structure mismatches, actual=1 failures: ['F5.1: variance_ratio exceeds 2.0']
...
CLAIM LEVEL: L3 (capped due to F5.x)
```

---

**SHADOW MODE** — observational only.

*Precision > optimism.*
