# Metrics Audit Trust Binding v0.1

> **Status**: AUTHORITATIVE CROSS-REFERENCE
> **Version**: 0.1.0
> **Date**: 2025-12-12
> **Author**: CLAUDE U (Metrics & Anti-Laundering Guardian)

---

## 1. Document Cross-Reference Matrix

This document binds metrics policy to artifact contracts for audit trust.

| Document | Role | Frozen Keys |
|----------|------|-------------|
| `RUN_SHADOW_AUDIT_METRICS_V0_1.md` | Metric definitions & tiers | `legacy_metrics.*`, `primary_metrics.*` |
| `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | CLI & output layout | `--input`, `--output`, exit codes |
| `RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` | Bundle structure | `manifest.json`, `summary.json` (Section 3-4) |
| `Metric_Versioning_Policy_v0.1.md` | Anti-laundering rule | AL-001, non-equivalence strings |
| `METRIC_DEFINITIONS.md` | Formula definitions | `mean_delta_p`, `divergence_rate` |

### 1.1 Binding Points

```
┌─────────────────────────────────────────────────────────────────────┐
│                    run_shadow_audit.py v0.1                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT ──────────────────────────────────────────────── OUTPUT      │
│    │                                                      │         │
│    ▼                                                      ▼         │
│  [Artifact Contract]                              [Metrics Policy]  │
│    - manifest.json                                  - summary.json  │
│    - status_report.txt                              - run_id/       │
│    - evidence_pack/                                                 │
│                                                                     │
│                         ┌─────────────┐                             │
│                         │ summary.json│                             │
│                         └──────┬──────┘                             │
│                                │                                    │
│          ┌─────────────────────┼─────────────────────┐              │
│          ▼                     ▼                     ▼              │
│   legacy_metrics.*     primary_metrics.*   diagnostic_metrics.*    │
│      (FROZEN)              (FROZEN)           (EXPERIMENTAL)        │
│                                                                     │
│                    + truth_vector_v0_1 (ADVISORY)                   │
│                    + anti_laundering {} (REQUIRED)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Metric Tiers in `summary.json`

### 2.1 Required Output Structure

```json
{
  "schema_version": "1.0.0",
  "run_id": "<run_id>",
  "timestamp": "<iso8601>",
  "mode": "SHADOW",

  "legacy_metrics": {
    "divergence_rate": <float>,
    "hard_ok_rate": <float>,
    "safe_region_rate": <float>,
    "divergence_count": <int>,
    "total_cycles": <int>
  },

  "primary_metrics": {
    "mean_delta_p": <float>,
    "omega_mismatch_rate": <float>
  },

  "diagnostic_metrics": {
    "success_div_rate": <float|null>,
    "blocked_div_rate": <float|null>,
    "phase_lag_xcorr": <float|null>
  },

  "truth_vector_v0_1": {
    "schema_version": "0.1.0",
    "status": "ADVISORY",
    "aggregation": "NONE",
    "components": { ... }
  },

  "anti_laundering": {
    "legacy_preserved": true,
    "non_equivalence_warnings": [ ... ],
    "policy_version": "0.1.0"
  },

  "status": "OK"
}
```

### 2.2 Tier Placement Invariants

| Tier | JSON Path | Presence | Mutability |
|------|-----------|----------|------------|
| `legacy_metrics` | `$.legacy_metrics.*` | **REQUIRED** | FROZEN |
| `primary_metrics` | `$.primary_metrics.*` | **REQUIRED** | FROZEN |
| `diagnostic_metrics` | `$.diagnostic_metrics.*` | OPTIONAL | EXPERIMENTAL |
| `truth_vector_v0_1` | `$.truth_vector_v0_1` | **REQUIRED** | ADVISORY |
| `anti_laundering` | `$.anti_laundering` | **REQUIRED** | FROZEN |

---

## 3. What Laundering Looks Like (NEGATIVE EXAMPLES)

### 3.1 Example 1: Overwriting Legacy Metrics

**VIOLATION**: Replacing `legacy_metrics` values with "better" primary values.

```json
{
  "legacy_metrics": {
    "divergence_rate": 0.034
  }
}
```

**Problem**: The `0.034` is actually `mean_delta_p`, not the original binary `divergence_rate` which was `0.87`. This is **metric laundering** — making a bad legacy value look good by substituting a different metric.

**Correct Output**:
```json
{
  "legacy_metrics": {
    "divergence_rate": 0.87
  },
  "primary_metrics": {
    "mean_delta_p": 0.034
  },
  "anti_laundering": {
    "legacy_preserved": true,
    "non_equivalence_warnings": [
      "legacy_metrics.divergence_rate (0.870) ≢ primary_metrics.mean_delta_p (0.034)"
    ]
  }
}
```

---

### 3.2 Example 2: Implicit Equivalence

**VIOLATION**: Renaming without explicit non-equivalence warning.

```json
{
  "metrics": {
    "divergence": 0.034,
    "accuracy": 0.966
  }
}
```

**Problem**: No indication whether `divergence` is `legacy_divergence_rate`, `mean_delta_p`, or something else. Consumers will assume equivalence with historical data. This is **silent laundering**.

**Correct Output**:
```json
{
  "legacy_metrics": {
    "divergence_rate": 0.87
  },
  "primary_metrics": {
    "mean_delta_p": 0.034
  }
}
```

---

### 3.3 Example 3: Missing Legacy Tier

**VIOLATION**: Omitting legacy metrics entirely.

```json
{
  "primary_metrics": {
    "mean_delta_p": 0.034,
    "omega_mismatch_rate": 0.05
  },
  "anti_laundering": {
    "legacy_preserved": false
  }
}
```

**Problem**: Historical comparisons become impossible. Dashboards expecting `divergence_rate` will break. Setting `legacy_preserved: false` is an **admission of laundering**, not a fix.

**Correct Output**: Include both tiers, even if legacy values look "bad."

---

### 3.4 Example 4: Fake Aggregation

**VIOLATION**: Computing a composite score in v0.1.

```json
{
  "truth_vector_v0_1": {
    "aggregation": "WEIGHTED_MEAN",
    "composite_score": 0.92
  }
}
```

**Problem**: v0.1 does NOT implement aggregation. This `composite_score` is fabricated. Per the "No New Science" rule, `aggregation` MUST be `"NONE"`.

**Correct Output**:
```json
{
  "truth_vector_v0_1": {
    "aggregation": "NONE",
    "components": {
      "state_tracking_error": 0.034,
      "safety_mismatch_rate": 0.05,
      "outcome_calibration_score": null
    }
  }
}
```

---

### 3.5 Laundering Detection Heuristics

| Signal | Likely Laundering |
|--------|-------------------|
| `legacy_metrics.divergence_rate` < 0.1 when twin is new | Suspicious (twins rarely start accurate) |
| `legacy_preserved: false` | Admission of violation |
| `anti_laundering` block missing | Non-compliant output |
| `truth_vector_v0_1.aggregation` != `"NONE"` | New science in v0.1 |
| No non-equivalence warnings when legacy ≠ primary | Silent substitution |

---

## 4. Auditor Checklist

### 4.1 Pre-Audit Verification

- [ ] **Policy documents present**
  - [ ] `docs/system_law/Metric_Versioning_Policy_v0.1.md` exists
  - [ ] `docs/system_law/calibration/RUN_SHADOW_AUDIT_METRICS_V0_1.md` exists
  - [x] `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md` exists (canonical)

- [ ] **Schema version matches**
  - [ ] `summary.json` has `schema_version: "1.0.0"`
  - [ ] `truth_vector_v0_1` has `schema_version: "0.1.0"`

### 4.2 Legacy Metrics Audit

- [ ] **Presence check**
  - [ ] `$.legacy_metrics.divergence_rate` exists (float)
  - [ ] `$.legacy_metrics.hard_ok_rate` exists (float)
  - [ ] `$.legacy_metrics.safe_region_rate` exists (float)
  - [ ] `$.legacy_metrics.divergence_count` exists (int)
  - [ ] `$.legacy_metrics.total_cycles` exists (int)

- [ ] **Source verification**
  - [ ] `divergence_rate` computed from `USLAShadowLogger` (binary mismatch)
  - [ ] NOT substituted with `mean_delta_p` or other continuous metric

- [ ] **Range validation**
  - [ ] All rates in [0.0, 1.0]
  - [ ] `total_cycles` > 0
  - [ ] `divergence_count` <= `total_cycles`

### 4.3 Primary Metrics Audit

- [ ] **Presence check**
  - [ ] `$.primary_metrics.mean_delta_p` exists (float)
  - [ ] `$.primary_metrics.omega_mismatch_rate` exists (float)

- [ ] **Source verification**
  - [ ] `mean_delta_p` from `calibration_annex.state_delta_p_mean`
  - [ ] `omega_mismatch_rate` from `TDAPatternClassifier.omega_miss_rate`

- [ ] **Non-equivalence verification**
  - [ ] `primary_metrics.mean_delta_p` ≠ `legacy_metrics.divergence_rate` (unless coincidentally equal)
  - [ ] If equal, verify source is genuinely the same (unlikely for different metrics)

### 4.4 Anti-Laundering Audit

- [ ] **Block presence**
  - [ ] `$.anti_laundering` exists as object

- [ ] **Invariant checks**
  - [ ] `$.anti_laundering.legacy_preserved` == `true`
  - [ ] `$.anti_laundering.policy_version` == `"0.1.0"`

- [ ] **Non-equivalence warnings**
  - [ ] `$.anti_laundering.non_equivalence_warnings` is array
  - [ ] Contains warning for `divergence_rate ≢ mean_delta_p`
  - [ ] Contains warning for `safe_region_rate ≢ omega_mismatch_rate`

### 4.5 Truth Vector Audit

- [ ] **Composition-only check**
  - [ ] `$.truth_vector_v0_1.aggregation` == `"NONE"`
  - [ ] `$.truth_vector_v0_1.status` == `"ADVISORY"`

- [ ] **Component passthrough verification**
  - [ ] `$.truth_vector_v0_1.components.state_tracking_error` == `$.primary_metrics.mean_delta_p`
  - [ ] `$.truth_vector_v0_1.components.safety_mismatch_rate` == `$.primary_metrics.omega_mismatch_rate`

- [ ] **Placeholder check**
  - [ ] `$.truth_vector_v0_1.components.outcome_calibration_score` == `null`

### 4.6 Negative Checks (Must NOT Happen)

- [ ] **No laundering signals**
  - [ ] `$.legacy_preserved` is NOT `false`
  - [ ] No `composite_score` or `aggregated_score` field
  - [ ] No `true_divergence` without version suffix

- [ ] **No silent substitution**
  - [ ] `legacy_metrics` values differ from `primary_metrics` values (unless genuine coincidence)
  - [ ] If values are identical, auditor flags for manual review

### 4.7 CI Validation Commands

```bash
# Validate anti-laundering invariant
jq '.anti_laundering.legacy_preserved' summary.json  # MUST be true

# Validate truth vector composition (not aggregation)
jq '.truth_vector_v0_1.aggregation' summary.json  # MUST be "NONE"

# Validate passthrough
jq '.truth_vector_v0_1.components.state_tracking_error == .primary_metrics.mean_delta_p' summary.json  # MUST be true

# Check for non-equivalence warnings
jq '.anti_laundering.non_equivalence_warnings | length > 0' summary.json  # MUST be true

# Detect suspicious equality (flag for review)
jq '.legacy_metrics.divergence_rate == .primary_metrics.mean_delta_p' summary.json  # If true, INVESTIGATE
```

---

## 5. Audit Sign-Off Template

```markdown
## Shadow Audit Metrics Review — [run_id]

### Auditor: _______________
### Date: _______________

### Checklist Status

- [ ] 4.1 Pre-Audit Verification: PASS / FAIL
- [ ] 4.2 Legacy Metrics Audit: PASS / FAIL
- [ ] 4.3 Primary Metrics Audit: PASS / FAIL
- [ ] 4.4 Anti-Laundering Audit: PASS / FAIL
- [ ] 4.5 Truth Vector Audit: PASS / FAIL
- [ ] 4.6 Negative Checks: PASS / FAIL

### Findings

1. [Finding or "None"]
2. [Finding or "None"]

### Verdict

- [ ] **APPROVED**: Output conforms to Metric Versioning Policy v0.1
- [ ] **REJECTED**: Laundering detected — see findings
- [ ] **FLAGGED**: Manual review required — see findings

### Signature: _______________
```

---

## 6. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                ANTI-LAUNDERING QUICK CHECK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✓ legacy_metrics.* present?              YES = PASS            │
│  ✓ primary_metrics.* present?             YES = PASS            │
│  ✓ anti_laundering.legacy_preserved?      true = PASS           │
│  ✓ truth_vector_v0_1.aggregation?         "NONE" = PASS         │
│  ✓ non_equivalence_warnings present?      YES = PASS            │
│                                                                 │
│  ✗ legacy_metrics values = primary_metrics values?              │
│    If YES → FLAG FOR MANUAL REVIEW (possible laundering)        │
│                                                                 │
│  ✗ composite_score present?               NO = PASS             │
│  ✗ true_divergence without _v1?           NO = PASS             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**END OF CROSS-REFERENCE**

**SHADOW MODE CONTRACT**: This document defines audit procedures for observational metrics only. No audit finding influences governance decisions.
