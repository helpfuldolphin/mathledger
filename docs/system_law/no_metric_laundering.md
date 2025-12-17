# NO METRIC LAUNDERING

**Document Version:** 1.0.0
**Schema Version:** 1.1.0
**SHADOW MODE:** This document is observational guidance only.

---

## Executive Summary

**Metric laundering** occurs when different metrics with distinct semantics are conflated,
renamed, or treated as equivalent without explicit acknowledgment of their differences.

This document establishes the **NO METRIC LAUNDERING** principle for P5 divergence reports:

> **`legacy_outcome_mismatch_rate NOT_EQUIVALENT_TO state_error_mean`**

These metrics measure fundamentally different aspects of twin-real divergence and
**MUST NOT** be treated as interchangeable.

---

## The Problem

### Before v1.1.0: Ambiguous "divergence_rate"

The legacy `divergence_rate` metric was often misinterpreted:

```json
{
  "divergence_rate": 0.15
}
```

**What does 0.15 mean?**
- 15% of cycles had *some* divergence? (outcome mismatch)
- Average state error magnitude? (mean delta)
- Euclidean distance in state space? (vector norm)
- Something else entirely?

This ambiguity allowed **metric laundering**: consumers could interpret the value
however they wished, leading to inconsistent analysis and misleading conclusions.

---

## The Solution: Explicit Metric Versioning

### v1.1.0 introduces `true_divergence_vector_v1`

```json
{
  "divergence_rate": 0.15,
  "true_divergence_vector_v1": {
    "safety_state_mismatch_rate": 0.07,
    "state_error_mean": 0.042,
    "outcome_brier_score": 0.12
  },
  "metric_versioning": {
    "legacy_metrics": ["divergence_rate", "mock_baseline_divergence_rate", "divergence_delta"],
    "true_vector_v1_metrics": ["safety_state_mismatch_rate", "state_error_mean", "outcome_brier_score"],
    "equivalence_note": "legacy_outcome_mismatch_rate NOT_EQUIVALENT_TO state_error_mean",
    "doc_reference": "docs/system_law/no_metric_laundering.md"
  }
}
```

---

## Metric Definitions

### Legacy Metrics (Pre-v1.1.0)

| Metric | Definition | Unit |
|--------|------------|------|
| `divergence_rate` | `divergent_cycles / total_cycles` | ratio [0,1] |
| `mock_baseline_divergence_rate` | P4 mock baseline comparison | ratio [0,1] |
| `divergence_delta` | `divergence_rate - mock_baseline_divergence_rate` | ratio [-1,1] |

**Note:** `divergence_rate` is the **outcome mismatch rate** — the fraction of cycles
where the twin's predicted outcome differs from the real outcome.

### True Vector v1 Metrics (v1.1.0+)

| Metric | Definition | Unit |
|--------|------------|------|
| `safety_state_mismatch_rate` | `(blocked_diverged + omega_diverged) / total_cycles` | ratio [0,1] |
| `state_error_mean` | `mean(abs(H_delta))` | state units |
| `outcome_brier_score` | `mean((twin_prob - outcome_binary)²)` | score [0,1] |

---

## Non-Equivalence Examples

### Example 1: High Outcome Mismatch, Low State Error

```
Scenario: Twin predicts "success" but real system returns "blocked"
          State values are nearly identical (H_delta ≈ 0.001)

legacy:
  divergence_rate = 0.20 (20% outcome mismatch)

true_vector_v1:
  safety_state_mismatch_rate = 0.20 (blocked differs)
  state_error_mean = 0.001 (tiny state difference)
```

**Interpretation:** The twin's *decision boundary* is miscalibrated, not its state tracking.
Laundering these as equivalent would suggest the twin needs full recalibration when only
the decision threshold needs adjustment.

### Example 2: Low Outcome Mismatch, High State Error

```
Scenario: Twin predicts correct outcomes but state values drift significantly
          H_delta averages 0.15 but outcomes still match

legacy:
  divergence_rate = 0.02 (2% outcome mismatch)

true_vector_v1:
  safety_state_mismatch_rate = 0.02
  state_error_mean = 0.15 (significant state drift)
```

**Interpretation:** The twin is tracking outcomes correctly but the underlying state
model is diverging. This could indicate future outcome prediction failures.
Laundering these would hide the state drift warning.

### Example 3: Safety-Critical Distinction

```
Scenario: 10% success divergence, 5% blocked divergence, 2% omega divergence

legacy:
  divergence_rate = 0.17 (all divergence types combined)

true_vector_v1:
  safety_state_mismatch_rate = 0.07 (blocked + omega only)
  outcome_brier_score = 0.18
```

**Interpretation:** `safety_state_mismatch_rate` isolates safety-critical prediction errors
(blocked/omega states) from less critical success prediction errors. The legacy metric
conflates these, potentially masking safety-relevant divergence in a sea of benign errors.

---

## Why This Matters

### 1. Decision Quality

Different metrics warrant different responses:
- High `state_error_mean` → recalibrate state tracking
- High `safety_state_mismatch_rate` → audit safety boundaries
- High `outcome_brier_score` → adjust probability calibration

Laundering these into a single "divergence" number loses actionable specificity.

### 2. Audit Trail

Explicit metric versioning creates a clear audit trail:
- Which metrics were used for decisions?
- Are comparisons across time periods valid?
- What schema version produced this report?

### 3. SHADOW MODE Compliance

In SHADOW MODE, metrics are observational only. Metric laundering can create
false confidence that would be dangerous if governance ever transitions out of shadow mode.

### 4. Calibration Experiments

CAL-EXP-2/3 may tune parameters, but **MUST NOT** redefine metric semantics or rename legacy fields.

**Enforcement chain:**
- **Runtime:** generator code + schema `const` constraints + `TestMetricVersioningBlock` tests
- **Interpretation:** CAL-EXP-2 binding definitions + validity attestation docs (non-runtime)

---

## Implementation Contract

### Report Generators MUST:

1. Include both legacy and true_vector metrics (backwards compatibility)
2. Include `metric_versioning` block with explicit categorization
3. Use `NOT_EQUIVALENT_TO` (ASCII-safe non-equivalence) in equivalence_note
4. Reference this document in `doc_reference`

> **CI Note — ASCII-Safe Policy:** We use ASCII tokens like `NOT_EQUIVALENT_TO` instead of
> Unicode symbols (e.g., `≢`) for Windows CP1252 compatibility. JSON `\uXXXX` escaping works
> but we standardize on ASCII for cross-platform portability in CI pipelines.
>
> **Enforcement trace:**
> - Generator: `scripts/generate_p5_divergence_real_report.py` (emits `equivalence_note`)
> - Schema: `docs/system_law/schemas/p5/p5_divergence_real.schema.json` (const constraint)
> - Tests: `pytest tests/topology/first_light/test_p5_divergence_pipeline_integration.py::TestMetricVersioningBlock -v`

### Consumers MUST:

1. Check `metric_versioning` block before interpreting metrics
2. NOT assume legacy `divergence_rate` equals any true_vector metric
3. NOT perform arithmetic across metric categories without explicit justification
4. Log which metric category was used for any decision

### Warnings MUST:

1. Name the exact metric being evaluated (e.g., `outcome_mismatch_rate=15%`)
2. NOT use generic "divergence" terminology without qualification
3. Cap to single warning per report to avoid alert fatigue

---

## Schema Reference

```json
{
  "metric_versioning": {
    "type": "object",
    "required": ["legacy_metrics", "true_vector_v1_metrics", "equivalence_note", "doc_reference"],
    "properties": {
      "legacy_metrics": {
        "type": "array",
        "items": { "enum": ["divergence_rate", "mock_baseline_divergence_rate", "divergence_delta"] }
      },
      "true_vector_v1_metrics": {
        "type": "array",
        "items": { "enum": ["safety_state_mismatch_rate", "state_error_mean", "outcome_brier_score"] }
      },
      "equivalence_note": { "const": "legacy_outcome_mismatch_rate NOT_EQUIVALENT_TO state_error_mean" },
      "doc_reference": { "const": "docs/system_law/no_metric_laundering.md" }
    }
  }
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-12 | Initial release with v1.1.0 schema |

---

## See Also

- `docs/system_law/schemas/p5/p5_divergence_real.schema.json` — Full schema definition
- `scripts/generate_p5_divergence_real_report.py` — Report generator implementation
- `scripts/generate_first_light_status.py` — Status warning implementation
