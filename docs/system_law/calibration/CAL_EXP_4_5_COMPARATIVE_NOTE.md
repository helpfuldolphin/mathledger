# CAL-EXP-4/5 Comparative Note

**Status**: ARCHIVAL (non-forward-looking)
**Date**: 2025-12-19
**Mode**: SHADOW (observational only)

---

## 1. Question Alignment

| Aspect | CAL-EXP-4 | CAL-EXP-5 |
|--------|-----------|-----------|
| Scientific Question | Does fail-close work correctly? | Does the system avoid fail-close? |
| Question Type | Verification (prove mechanism works) | Avoidance (avoid triggering mechanism) |
| Expected Outcome | FAIL (fail-close triggers) | PASS (fail-close avoided) |
| Observed Outcome | FAIL | FAIL |
| Question Answered | YES | YES |

### Relationship

CAL-EXP-4 and CAL-EXP-5 are complementary questions about the same fail-close mechanism:

- **CAL-EXP-4**: Confirms fail-close triggers when conditions warrant
- **CAL-EXP-5**: Tests whether fail-close can be avoided

Both experiments used identical thresholds, schemas, and verifier logic.

---

## 2. Invariant Confirmation

### 2.1 Verifier Unchanged

| Property | Value |
|----------|-------|
| Verifier path | `scripts/verify_cal_exp_4_run.py` |
| SHA-256 | `97ef340b9ce0750474b14fc2d599b08a8c30c0f7e778270514d4f7396a8f5e02` |
| Modification | **NONE** |

### 2.2 Thresholds Unchanged

| Threshold | CAL-EXP-4 | CAL-EXP-5 | Identical |
|-----------|-----------|-----------|-----------|
| `VARIANCE_RATIO_MAX` | 2.0 | 2.0 | YES |
| `VARIANCE_RATIO_MIN` | 0.5 | 0.5 | YES |
| `IQR_RATIO_MAX` | 2.0 | 2.0 | YES |
| `WINDOWED_DRIFT_MAX` | 0.05 | 0.05 | YES |
| `CLAIM_CAP_THRESHOLD` | 3.0 | 3.0 | YES |
| `MIN_COVERAGE_RATIO` | 1.0 | 1.0 | YES |
| `MAX_GAP_RATIO_DIVERGENCE` | 0.1 | 0.1 | YES |

**All thresholds identical: YES**

### 2.3 Schemas Unchanged

| Schema | Version | Source |
|--------|---------|--------|
| `temporal_structure_audit` | 1.0.0 | `CAL_EXP_4_FREEZE.md` |
| `variance_profile_audit` | 1.0.0 | `CAL_EXP_4_FREEZE.md` |

**Schema versions identical: YES**

### 2.4 Window Definitions Unchanged

| Window | Range | Both Experiments |
|--------|-------|------------------|
| Warm-up (excluded) | 0-200 | YES |
| Evaluation | 201-1000 | YES |
| W1_early | 201-400 | YES |
| W2_mid | 401-600 | YES |
| W3_late | 601-800 | YES |
| W4_final | 801-1000 | YES |

---

## 3. Predicate Comparison Table

### 3.1 Per-Seed Predicate Status

#### Seed 42

| Predicate | Threshold | CAL-EXP-4 | CAL-EXP-5 | Match |
|-----------|-----------|-----------|-----------|-------|
| `variance_ratio` | 0.5-2.0 | 0.2645 | 0.2645 | YES |
| `variance_ratio_acceptable` | — | FALSE | FALSE | YES |
| `windowed_variance_drift` | ≤0.05 | 0.7447 | 0.7447 | YES |
| `windowed_drift_acceptable` | — | FALSE | FALSE | YES |
| `iqr_ratio` | ≤2.0 | 0.4032 | 0.4032 | YES |
| `iqr_ratio_acceptable` | — | TRUE | TRUE | YES |
| `temporal_structure_pass` | — | TRUE | TRUE | YES |
| `profile_compatible` | — | FALSE | FALSE | YES |
| `f5_failure_codes` | — | [F5.2, F5.3] | [F5.2, F5.3] | YES |
| `claim_level` | — | L0 | L0 | YES |

#### Seed 43

| Predicate | Threshold | CAL-EXP-4 | CAL-EXP-5 | Match |
|-----------|-----------|-----------|-----------|-------|
| `variance_ratio` | 0.5-2.0 | 0.1902 | 0.1902 | YES |
| `variance_ratio_acceptable` | — | FALSE | FALSE | YES |
| `windowed_variance_drift` | ≤0.05 | 0.8182 | 0.8182 | YES |
| `windowed_drift_acceptable` | — | FALSE | FALSE | YES |
| `iqr_ratio` | ≤2.0 | 0.5503 | 0.5503 | YES |
| `iqr_ratio_acceptable` | — | TRUE | TRUE | YES |
| `temporal_structure_pass` | — | TRUE | TRUE | YES |
| `profile_compatible` | — | FALSE | FALSE | YES |
| `f5_failure_codes` | — | [F5.2, F5.3] | [F5.2, F5.3] | YES |
| `claim_level` | — | L0 | L0 | YES |

#### Seed 44

| Predicate | Threshold | CAL-EXP-4 | CAL-EXP-5 | Match |
|-----------|-----------|-----------|-----------|-------|
| `variance_ratio` | 0.5-2.0 | 0.2341 | 0.2341 | YES |
| `variance_ratio_acceptable` | — | FALSE | FALSE | YES |
| `windowed_variance_drift` | ≤0.05 | 0.8546 | 0.8546 | YES |
| `windowed_drift_acceptable` | — | FALSE | FALSE | YES |
| `iqr_ratio` | ≤2.0 | 0.4427 | 0.4427 | YES |
| `iqr_ratio_acceptable` | — | TRUE | TRUE | YES |
| `temporal_structure_pass` | — | TRUE | TRUE | YES |
| `profile_compatible` | — | FALSE | FALSE | YES |
| `f5_failure_codes` | — | [F5.2, F5.3] | [F5.2, F5.3] | YES |
| `claim_level` | — | L0 | L0 | YES |

### 3.2 Predicate Summary

| Predicate | Held in CAL-EXP-4 | Held in CAL-EXP-5 | Consistent |
|-----------|-------------------|-------------------|------------|
| `temporal_structure_pass` | YES (all runs) | YES (all runs) | YES |
| `iqr_ratio_acceptable` | YES (all runs) | YES (all runs) | YES |
| `variance_ratio_acceptable` | NO (all runs) | NO (all runs) | YES |
| `windowed_drift_acceptable` | NO (all runs) | NO (all runs) | YES |
| `profile_compatible` | NO (all runs) | NO (all runs) | YES |

---

## 4. Predicates Violated in Both Experiments

| Predicate | Violation | CAL-EXP-4 Runs | CAL-EXP-5 Runs |
|-----------|-----------|----------------|----------------|
| `variance_ratio_acceptable` | ratio < 0.5 | 3/3 | 3/3 |
| `windowed_drift_acceptable` | drift > 0.05 | 3/3 | 3/3 |

### 4.1 Variance Ratio Violations

| Seed | Observed Ratio | Threshold Min | Violation Magnitude |
|------|----------------|---------------|---------------------|
| 42 | 0.2645 | 0.5 | 0.2355 below min |
| 43 | 0.1902 | 0.5 | 0.3098 below min |
| 44 | 0.2341 | 0.5 | 0.2659 below min |

### 4.2 Windowed Drift Violations

| Seed | Observed Drift | Threshold Max | Violation Magnitude |
|------|----------------|---------------|---------------------|
| 42 | 0.7447 | 0.05 | 14.9x threshold |
| 43 | 0.8182 | 0.05 | 16.4x threshold |
| 44 | 0.8546 | 0.05 | 17.1x threshold |

---

## 5. Predicates Held in Both Experiments

| Predicate | CAL-EXP-4 | CAL-EXP-5 | Consistent |
|-----------|-----------|-----------|------------|
| `temporal_structure_pass` | TRUE (all) | TRUE (all) | YES |
| `iqr_ratio_acceptable` | TRUE (all) | TRUE (all) | YES |
| `cycle_count_match` | TRUE (all) | TRUE (all) | YES |
| `cycle_indices_identical` | TRUE (all) | TRUE (all) | YES |
| `coverage_ratio_match` | TRUE (all) | TRUE (all) | YES |
| `gap_structure_compatible` | TRUE (all) | TRUE (all) | YES |

---

## 6. Outcome Symmetry/Differences

### 6.1 Symmetries

| Property | CAL-EXP-4 | CAL-EXP-5 |
|----------|-----------|-----------|
| F5.2 triggered | YES (all runs) | YES (all runs) |
| F5.3 triggered | YES (all runs) | YES (all runs) |
| Claim cap | L0 (all runs) | L0 (all runs) |
| Artifact count | 14 per run | 14 per run |
| Temporal structure | PASS (all) | PASS (all) |
| IQR ratio | PASS (all) | PASS (all) |

### 6.2 Differences

| Property | CAL-EXP-4 | CAL-EXP-5 |
|----------|-----------|-----------|
| `experiment_id` in artifacts | `CAL-EXP-4` | `CAL-EXP-5` |
| Output directory | `results/cal_exp_4/` | `results/cal_exp_5/` |
| Verdict field | `claim_level` | `cal_exp_5_verdict` |
| Expected outcome | FAIL | PASS |
| Observed outcome | FAIL | FAIL |
| Question answered | YES | YES |

### 6.3 Numerical Equivalence

For identical seeds, all numerical values are identical:

| Metric (seed 42) | CAL-EXP-4 | CAL-EXP-5 | Delta |
|------------------|-----------|-----------|-------|
| `variance_ratio` | 0.2645 | 0.2645 | 0.0000 |
| `windowed_variance_drift` | 0.7447 | 0.7447 | 0.0000 |
| `iqr_ratio` | 0.4032 | 0.4032 | 0.0000 |

This confirms deterministic replay: identical seeds produce identical numerical outputs.

---

## 7. Strictly Permitted Conclusions

The following statements are factual observations with no forward-looking implications:

### 7.1 Invariant Conclusions

1. **Verifier unchanged**: The CAL-EXP-4 verifier was not modified between experiments.

2. **Thresholds unchanged**: All seven threshold values are identical in CAL-EXP-4 and CAL-EXP-5.

3. **Schemas unchanged**: Both experiments use schema version 1.0.0.

4. **Window definitions unchanged**: Evaluation windows (201-1000) and sub-windows are identical.

### 7.2 Predicate Conclusions

5. **F5.2 triggered in both**: Variance ratio was below threshold (< 0.5) in all runs of both experiments.

6. **F5.3 triggered in both**: Windowed drift exceeded threshold (> 0.05) in all runs of both experiments.

7. **Temporal structure held in both**: F5.1 was not triggered in any run.

8. **IQR ratio held in both**: F5.7 was not triggered in any run.

### 7.3 Outcome Conclusions

9. **Both experiments answered their questions**: CAL-EXP-4 confirmed fail-close triggers; CAL-EXP-5 confirmed fail-close was not avoided.

10. **Claim cap identical**: L0 was applied in all runs of both experiments due to F5.2.

11. **Deterministic replay confirmed**: Identical seeds produced identical numerical values across experiments.

---

## 8. Explicitly Not Concluded

This note does **NOT** conclude:

- Whether thresholds should be adjusted
- Whether the synthetic harness is representative
- Whether CAL-EXP-6 should be designed
- Whether learning effectiveness is demonstrated
- Whether the system has any capability
- What changes would produce PASS outcomes

---

**SHADOW MODE** — Archival, non-forward-looking.

*Precision > optimism.*
