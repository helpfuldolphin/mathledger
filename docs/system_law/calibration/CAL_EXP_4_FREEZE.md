# CAL-EXP-4: Semantic Freeze Declaration

**Status**: FROZEN
**Authority**: STRATCOM GOVERNANCE
**Effective Date**: 2025-12-17
**Freeze Version**: 1.0.0

---

## Declaration

This document declares CAL-EXP-4 semantics as **FROZEN**. All thresholds, failure codes, claim capping rules, and schema versions specified herein are binding.

> **Any change to CAL-EXP-4 semantics requires explicit STRATCOM authorization.**

---

## 1. Schema Versions (Frozen)

| Schema | Path | Version | Status |
|--------|------|---------|--------|
| Temporal Structure Audit | `schemas/cal_exp_4/temporal_structure_audit.schema.json` | **1.0.0** | FROZEN |
| Variance Profile Audit | `schemas/cal_exp_4/variance_profile_audit.schema.json` | **1.0.0** | FROZEN |

Schema version changes require STRATCOM authorization and new freeze declaration.

---

## 2. Thresholds (Frozen)

### 2.1 Variance Comparability Thresholds

| Parameter | Threshold | Semantics |
|-----------|-----------|-----------|
| `variance_ratio_max` | **2.0** | Treatment/baseline variance ratio upper bound |
| `variance_ratio_min` | **0.5** | Treatment/baseline variance ratio lower bound |
| `iqr_ratio_max` | **2.0** | Treatment/baseline IQR ratio upper bound |
| `windowed_drift_max` | **0.05** | Maximum variance drift across sub-windows |
| `claim_cap_threshold` | **3.0** | Ratio threshold triggering claim cap vs fail-close |

### 2.2 Temporal Comparability Thresholds

| Parameter | Threshold | Semantics |
|-----------|-----------|-----------|
| `min_coverage_ratio` | **1.0** | Minimum temporal coverage (100% required) |
| `max_gap_ratio_divergence` | **0.1** | Maximum gap ratio divergence between arms |

### 2.3 Threshold Interpretation

- **Exceeds threshold** = comparability failure
- **Within threshold** = comparability acceptable
- Thresholds are **inclusive** (boundary values pass)

---

## 3. F5.x Failure Taxonomy (Frozen)

### 3.1 Failure Code Registry

| Code | Name | Trigger Condition | Severity |
|------|------|-------------------|----------|
| **F5.1** | Temporal Structure Incompatible | `temporal_structure_pass = false` | FAIL-CLOSE |
| **F5.2** | Variance Ratio Out of Bounds | `variance_ratio_acceptable = false` | FAIL-CLOSE |
| **F5.3** | Windowed Drift Excessive | `windowed_drift_acceptable = false` | WARN/CAP |
| **F5.4** | Missing Audit Artifact | Required artifact not found | FAIL-CLOSE + CAP |
| **F5.5** | Schema Validation Failure | Malformed JSON, missing fields, wrong version | FAIL-CLOSE |
| **F5.6** | Pathological Data | NaN/Inf detected in audit data | FAIL-CLOSE |
| **F5.7** | IQR Ratio Out of Bounds | `iqr_ratio_acceptable = false` | WARN/CAP |

### 3.2 Failure Code Semantics

#### F5.1: Temporal Structure Incompatible

Triggered by ANY of:
- `cycle_count_match = false`
- `cycle_indices_identical = false`
- `coverage_ratio_match = false`
- `gap_structure_compatible = false`
- `temporal_structure_compatible = false`
- `baseline_arm.monotonic_cycle_indices = false`
- `treatment_arm.monotonic_cycle_indices = false`
- `temporal_structure_pass = false`

**Effect**: Run INVALID. No claim permitted.

#### F5.2: Variance Ratio Out of Bounds

Triggered when:
- `variance_ratio > variance_ratio_max` (2.0)
- `variance_ratio < variance_ratio_min` (0.5)
- `profile_compatible = false`
- `variance_profile_pass = false`

**Effect**: Run INVALID or claim capped (per F5.2 severity).

#### F5.3: Windowed Drift Excessive

Triggered when:
- `windowed_variance_drift > windowed_drift_max` (0.05)
- `windowed_drift_acceptable = false`

**Effect**: Claim capped. Run may still be valid at reduced claim level.

#### F5.4: Missing Audit Artifact

Triggered when:
- `validity/temporal_structure_audit.json` not found
- `validity/variance_profile_audit.json` not found

**Effect**: Run INVALID. Claim capped to L3.

#### F5.5: Schema Validation Failure

Triggered by ANY of:
- `schema_version != "1.0.0"`
- `experiment_id != "CAL-EXP-4"`
- Required fields missing
- Malformed JSON

**Effect**: Run INVALID. No claim permitted.

#### F5.6: Pathological Data

Triggered by ANY of:
- NaN values in numeric fields
- Inf values in numeric fields
- `has_nan = true` in arm data
- `has_inf = true` in arm data

**Effect**: Run INVALID. No claim permitted.

#### F5.7: IQR Ratio Out of Bounds

Triggered when:
- `iqr_ratio > iqr_ratio_max` (2.0)
- `iqr_ratio_acceptable = false`

**Effect**: Claim capped. Run may still be valid at reduced claim level.

---

## 4. Claim Capping Rules (Frozen)

### 4.1 Claim Level Hierarchy

| Level | Description | Requirements |
|-------|-------------|--------------|
| L5 | Replicated | 3+ independent run-pairs, all L4 |
| L4 | Validated | All comparability checks pass |
| L3 | Provisional | Comparability concerns present |
| L2 | Within Noise | Effect within noise floor |
| L1 | Measured | Measurements obtained |
| L0 | Incomplete | Run did not complete |

### 4.2 Claim Cap Triggers

| Condition | Cap Level | F5 Code |
|-----------|-----------|---------|
| Missing variance audit | **L3** | F5.4 |
| Variance ratio exceeds threshold (soft) | **L3** | F5.2 |
| Windowed drift exceeds threshold | **L3** | F5.3 |
| IQR ratio exceeds threshold | **L3** | F5.7 |
| Profile incompatible (fail-close) | **L0** | F5.2 |
| Temporal structure incompatible | **L0** | F5.1 |
| Schema validation failure | **L0** | F5.5 |
| Pathological data | **L0** | F5.6 |

### 4.3 Claim Cap Precedence

When multiple cap conditions apply, the **lowest** (most restrictive) cap takes precedence.

```
L0 < L1 < L2 < L3 < L4 < L5
```

---

## 5. Verifier Binding

The verifier (`scripts/verify_cal_exp_4_run.py`) MUST:

1. Validate against frozen schema versions (1.0.0)
2. Apply frozen thresholds exactly as specified
3. Emit F5.x codes per frozen taxonomy
4. Apply claim capping per frozen rules
5. Fail-close on any schema deviation

### 5.1 Verifier Output Contract

```json
{
  "temporal_comparability": "<bool>",
  "variance_comparability": "<bool>",
  "f5_failure_codes": ["<F5.x>", ...],
  "claim_cap_applied": "<bool>",
  "claim_cap_level": "<L0-L5 | null>"
}
```

---

## 6. Change Control

### 6.1 Frozen Elements

The following elements are FROZEN and require STRATCOM authorization to modify:

- [ ] Threshold values (Section 2)
- [ ] F5.x failure codes and semantics (Section 3)
- [ ] Claim capping rules (Section 4)
- [ ] Schema versions (Section 1)
- [ ] Verifier output contract (Section 5.1)

### 6.2 Change Request Process

1. Submit change request to STRATCOM
2. Document rationale and impact analysis
3. Obtain explicit STRATCOM authorization
4. Update freeze declaration with new version
5. Update all dependent artifacts

### 6.3 Non-Frozen Elements

The following may be modified without STRATCOM authorization:

- Verifier implementation details (not affecting semantics)
- Test fixtures (not affecting production)
- Documentation clarifications (not affecting semantics)
- CI workflow configuration (not affecting thresholds)

---

## 7. Attestation

This freeze declaration is binding for all CAL-EXP-4 operations.

| Role | Responsibility |
|------|----------------|
| Verifier | Enforce frozen semantics |
| Harness | Generate artifacts per frozen schemas |
| CI | Validate per frozen thresholds |
| Analysts | Interpret per frozen claim levels |

---

**FROZEN** — No semantic changes without STRATCOM authorization.

**SHADOW MODE** — Observational only, non-gating.
