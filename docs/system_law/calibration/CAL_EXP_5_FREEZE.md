# CAL-EXP-5: Semantic Freeze Declaration

**Status**: FROZEN
**Authority**: STRATCOM GOVERNANCE
**Effective Date**: 2025-12-19
**Freeze Version**: 1.0.0
**Mode**: SHADOW (observational only)

---

## Declaration

This document declares CAL-EXP-5 semantics as **FROZEN**. All thresholds, failure codes, and verdict rules are inherited from CAL-EXP-4 and are binding.

> **Any change to CAL-EXP-5 semantics requires explicit STRATCOM authorization.**

---

## 1. Inheritance Statement

**CAL-EXP-5 inherits all frozen elements from CAL-EXP-4 without modification.**

| Element | Source | CAL-EXP-5 Status |
|---------|--------|------------------|
| Schema versions | `CAL_EXP_4_FREEZE.md` §1 | Inherited (1.0.0) |
| Variance thresholds | `CAL_EXP_4_FREEZE.md` §2.1 | Inherited |
| Temporal thresholds | `CAL_EXP_4_FREEZE.md` §2.2 | Inherited |
| F5.x failure taxonomy | `CAL_EXP_4_FREEZE.md` §3 | Inherited |
| Verifier | `scripts/verify_cal_exp_4_run.py` | Reused without modification |

---

## 2. Thresholds (Inherited from CAL-EXP-4)

### 2.1 Variance Comparability Thresholds

| Parameter | Threshold | Source |
|-----------|-----------|--------|
| `variance_ratio_max` | **2.0** | CAL_EXP_4_FREEZE §2.1 |
| `variance_ratio_min` | **0.5** | CAL_EXP_4_FREEZE §2.1 |
| `iqr_ratio_max` | **2.0** | CAL_EXP_4_FREEZE §2.1 |
| `windowed_drift_max` | **0.05** | CAL_EXP_4_FREEZE §2.1 |

### 2.2 Temporal Comparability Thresholds

| Parameter | Threshold | Source |
|-----------|-----------|--------|
| `min_coverage_ratio` | **1.0** | CAL_EXP_4_FREEZE §2.2 |
| `max_gap_ratio_divergence` | **0.1** | CAL_EXP_4_FREEZE §2.2 |

---

## 3. F5.x Failure Taxonomy (Inherited)

### 3.1 FAIL-CLOSE Codes

| Code | Name | Effect on Verdict |
|------|------|-------------------|
| **F5.1** | Temporal Structure Incompatible | **FAIL** |
| **F5.2** | Variance Ratio Out of Bounds | **FAIL** |
| **F5.4** | Missing Audit Artifact | **FAIL** |
| **F5.5** | Schema Validation Failure | **FAIL** |
| **F5.6** | Pathological Data | **FAIL** |

### 3.2 WARN Codes (Non-Verdict-Affecting)

| Code | Name | Effect on Verdict | Effect on Claim |
|------|------|-------------------|-----------------|
| **F5.3** | Windowed Drift Excessive | None | Cap to L3 |
| **F5.7** | IQR Ratio Out of Bounds | None | Cap to L3 |

---

## 4. Verdict Semantics (CAL-EXP-5 Specific)

### 4.1 Binary Verdict Only

CAL-EXP-5 produces exactly one of two verdicts: **PASS** or **FAIL**.

No PARTIAL verdict exists.

### 4.2 Verdict Predicate

```
FAIL_CLOSE_CODES = {F5.1, F5.2, F5.4, F5.5, F5.6}

CAL_EXP_5_PASS ≡ (f5_failure_codes ∩ FAIL_CLOSE_CODES = ∅)
CAL_EXP_5_FAIL ≡ (f5_failure_codes ∩ FAIL_CLOSE_CODES ≠ ∅)
```

### 4.3 Verdict Computation

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

## 5. Claim Level Semantics

### 5.1 Claim Hierarchy (Inherited)

| Level | Description |
|-------|-------------|
| L5 | Replicated (3+ independent run-pairs, all L4) |
| L4 | Validated (all comparability checks pass) |
| L3 | Provisional (WARN codes present) |
| L2 | Within Noise |
| L1 | Measured |
| L0 | Incomplete/Invalid |

### 5.2 CAL-EXP-5 Claim Implications

| Verdict | Claim Implication |
|---------|-------------------|
| **PASS** | Up to L4/L5 achievable (per replication rules) |
| **PASS with WARN** | Capped to L3 |
| **FAIL** | L0 (run void) |

---

## 6. Verifier Binding

### 6.1 Verifier Reuse

CAL-EXP-5 uses `scripts/verify_cal_exp_4_run.py` without modification.

### 6.2 Invocation

```bash
python scripts/verify_cal_exp_4_run.py --run-dir results/cal_exp_5/<run_id>/
```

### 6.3 Verdict Extraction

The CAL-EXP-5 harness computes the final verdict from verifier output:

```python
verifier_output = run_verifier(run_dir)
f5_codes = verifier_output["f5_failure_codes"]
verdict = compute_cal_exp_5_verdict(f5_codes)
```

---

## 7. Artifact Contract (CAL-EXP-5 Specific)

### 7.1 Output Directory

```
results/cal_exp_5/<run_id>/
```

### 7.2 Experiment Identifier

All CAL-EXP-5 artifacts MUST use:

```json
{
  "experiment_id": "CAL-EXP-5"
}
```

### 7.3 RUN_METADATA.json Contract

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

## 8. Interpretation Guardrails (Frozen)

### 8.1 If CAL-EXP-5 PASS

**Permitted Statements**:
- "No FAIL-CLOSE F5.x codes were triggered across all runs"
- "System avoided variance ratio incompatibility (F5.2) under these conditions"
- "Variance-aligned arm construction did not trigger FAIL-CLOSE"

**Forbidden Statements**:
- "System can produce variance-aligned arms" (overgeneralization)
- "Variance compatibility is solved" (overclaim)
- Any capability or performance assertion

### 8.2 If CAL-EXP-5 FAIL

**Permitted Statements**:
- "FAIL-CLOSE F5.x codes were triggered"
- "System did not avoid variance incompatibility under these conditions"
- Specific F5.x codes and values observed

**Forbidden Statements**:
- "System cannot avoid variance incompatibility" (overgeneralization)
- "Thresholds need adjustment" (outside scope)
- Any capability or performance assertion

---

## 9. Change Control

### 9.1 Frozen Elements

The following elements are FROZEN and require STRATCOM authorization to modify:

- [ ] Threshold values (inherited from CAL-EXP-4)
- [ ] F5.x failure codes and semantics (inherited)
- [ ] Binary verdict semantics (CAL-EXP-5 specific)
- [ ] Interpretation guardrails

### 9.2 Non-Frozen Elements

The following may be modified without STRATCOM authorization:

- Harness implementation details (not affecting semantics)
- Test fixtures (not affecting production)
- Documentation clarifications (not affecting semantics)

---

## 10. Attestation

This freeze declaration is binding for all CAL-EXP-5 operations.

| Role | Responsibility |
|------|----------------|
| Verifier | Enforce frozen semantics (CAL-EXP-4 verifier) |
| Harness | Generate artifacts per frozen schemas |
| Analysts | Interpret per frozen guardrails |

---

**FROZEN** — No semantic changes without STRATCOM authorization.

**SHADOW MODE** — Observational only, non-gating.

*Precision > optimism.*
