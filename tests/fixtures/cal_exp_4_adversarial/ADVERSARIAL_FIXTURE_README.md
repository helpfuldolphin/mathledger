# CAL-EXP-4 Adversarial Fixture

**Purpose**: Measurement Integrity Red Team test fixture
**Status**: SYNTHETIC / NOT CANONICAL / FOR TESTING ONLY

---

## Overview

This fixture is designed to trigger **ALL** F5.x failure modes in the CAL-EXP-4 verifier.
It validates that the verifier correctly identifies incompatible variance profiles and
applies claim capping.

---

## Fixture Parameters

### Temporal Structure Violations

| Parameter | Baseline | Treatment | Threshold | Violation |
|-----------|----------|-----------|-----------|-----------|
| cycle_count | 800 | 750 | match required | **50 cycles missing** |
| monotonic_cycle_indices | true | **false** | true | **Non-monotonic** |
| temporal_coverage_ratio | 1.0 | 0.9375 | >= 1.0 | **Coverage gap** |
| cycle_gap_max | 1 | 5 | comparable | **Gap divergence** |

### Variance Profile Violations

| Parameter | Value | Threshold | Violation |
|-----------|-------|-----------|-----------|
| variance_ratio | **2.5** | max 2.0 | **Exceeds by 25%** |
| windowed_variance_drift | **0.25** | max 0.05 | **Exceeds by 5x** |
| iqr_ratio | **2.92** | max 2.0 | **Exceeds by 46%** |
| window_volatility_ratio | 3.18 (max) | < 2.0 | **Window 3 extreme** |

---

## Expected Verifier Output

### Summary

```
SUMMARY: 32 checks, 9 FAIL, 3 WARN
temporal_comparability: False
variance_comparability: False
claim_cap_applied: True
claim_cap_level: L3
f5_failure_codes: ['F5.1', 'F5.2', 'F5.3', 'F5.7']
VERDICT: FAIL
```

### F5.x Codes Triggered

| Code | Count | Description |
|------|-------|-------------|
| F5.1 | 7 | Temporal structure incompatible |
| F5.2 | 2 | Variance ratio out of bounds |
| F5.3 | 1 | Windowed drift excessive |
| F5.7 | 1 | IQR ratio out of bounds |

### Detailed Check Results

#### Temporal Structure Checks (F5.1)

```
FAIL: temporal:cycle_count_match: expected=true, actual=False [F5.1]
FAIL: temporal:cycle_indices_identical: expected=true, actual=False [F5.1]
FAIL: temporal:coverage_ratio_match: expected=true, actual=False [F5.1]
FAIL: temporal:gap_structure_compatible: expected=true, actual=False [F5.1]
FAIL: temporal:structure_compatible: expected=true, actual=False [F5.1]
PASS: temporal:baseline_monotonic: expected=true, actual=True
FAIL: temporal:treatment_monotonic: expected=true, actual=False [F5.1]
FAIL: temporal:structure_pass: expected=true, actual=False [F5.1]
```

#### Variance Profile Checks (F5.2, F5.3, F5.7)

```
WARN: variance:ratio_acceptable: expected=true, actual=False (ratio=2.5) [F5.2]
WARN: variance:windowed_drift_acceptable: expected=true, actual=False (drift=0.25) [F5.3]
WARN: variance:iqr_ratio_acceptable: expected=true, actual=False (iqr_ratio=2.92) [F5.7]
FAIL: variance:profile_compatible: expected=true, actual=False [F5.2]
FAIL: variance:profile_pass: expected=true, actual=False [F5.2]
```

#### Claim Capping

```
PASS: claim:cap_applied: expected=claim cap from audit, actual=capped to L3
```

---

## Validation Criteria

The verifier MUST:

1. **NOT crash** on this adversarial input
2. Return `temporal_comparability = false`
3. Return `variance_comparability = false`
4. Return `claim_cap_applied = true`
5. Return `claim_cap_level = "L3"`
6. Return all four F5 codes: `F5.1`, `F5.2`, `F5.3`, `F5.7`
7. Return `VERDICT: FAIL`

---

## Usage

```bash
# Run verifier on adversarial fixture
python scripts/verify_cal_exp_4_run.py \
  --run-dir tests/fixtures/cal_exp_4_adversarial

# Expected exit code: 1 (FAIL)
```

---

## Windowed Analysis Detail

The per-window variance ratios show increasing divergence:

| Window | Baseline Var | Treatment Var | Ratio |
|--------|--------------|---------------|-------|
| W1 | 0.008 | 0.015 | 1.875 |
| W2 | 0.009 | 0.022 | 2.444 |
| W3 | 0.011 | 0.035 | **3.182** |
| W4 | 0.012 | 0.028 | 2.333 |

Window 3 shows the most extreme variance ratio (3.18x), indicating
potential regime instability in the treatment arm.

---

**SYNTHETIC DATA - NOT FOR PRODUCTION USE**
