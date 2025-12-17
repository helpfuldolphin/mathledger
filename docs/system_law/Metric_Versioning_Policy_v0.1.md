# Metric Versioning Policy v0.1 — `run_shadow_audit.py`

> **Status**: ACTIVE SPECIFICATION
> **Version**: 0.1.1
> **Date**: 2025-12-12
> **Author**: CLAUDE U
> **Canonical Appendix**: `docs/system_law/calibration/RUN_SHADOW_AUDIT_METRICS_V0_1.md`

---

## 0. Canonical Metric Mapping Table

### 0.1 Tier Classification

| Tier | JSON Path | Mutability | Purpose |
|------|-----------|------------|---------|
| `legacy_metrics.*` | FROZEN | Always present if source exists; backward compat |
| `primary_metrics.*` | FROZEN | v0.1 minimal truth set; calibration decisions |
| `diagnostic_metrics.*` | EXPERIMENTAL | Optional deep-dive; may change |

### 0.2 Metric → Tier Mapping

| Metric Name | Tier | Source | JSON Key | Status |
|-------------|------|--------|----------|--------|
| `divergence_rate` | `legacy_metrics` | `USLAShadowLogger.get_summary()` | `divergence_rate` | FROZEN |
| `hard_ok_rate` | `legacy_metrics` | `USLAShadowLogger.get_summary()` | `hard_ok_rate` | FROZEN |
| `safe_region_rate` | `legacy_metrics` | `USLAShadowLogger.get_summary()` | `safe_region_rate` | FROZEN |
| `divergence_count` | `legacy_metrics` | `USLAShadowLogger.get_summary()` | `divergence_count` | FROZEN |
| `total_cycles` | `legacy_metrics` | `USLAShadowLogger.get_summary()` | `total_cycles` | FROZEN |
| `mean_delta_p` | `primary_metrics` | `calibration_annex.state_delta_p_mean` | `mean_delta_p` | FROZEN |
| `omega_mismatch_rate` | `primary_metrics` | `TDAPatternClassifier.omega_miss_rate` | `omega_mismatch_rate` | FROZEN |
| `success_div_rate` | `diagnostic_metrics` | `_extract_divergence_component()` | `success_div_rate` | EXPERIMENTAL |
| `blocked_div_rate` | `diagnostic_metrics` | `_extract_divergence_component()` | `blocked_div_rate` | EXPERIMENTAL |
| `phase_lag_xcorr` | `diagnostic_metrics` | `cal_exp1_report.windows[*]` | `phase_lag_xcorr` | EXPERIMENTAL |
| `delta_bias` | `diagnostic_metrics` | DEPRECATED (heuristic) | `delta_bias` | DEPRECATED |
| `mean_H_error` | `diagnostic_metrics` | per-component recomputation | `mean_H_error` | EXPERIMENTAL |
| `mean_rho_error` | `diagnostic_metrics` | per-component recomputation | `mean_rho_error` | EXPERIMENTAL |
| `mean_tau_error` | `diagnostic_metrics` | per-component recomputation | `mean_tau_error` | EXPERIMENTAL |
| `mean_beta_error` | `diagnostic_metrics` | per-component recomputation | `mean_beta_error` | EXPERIMENTAL |

### 0.3 Anti-Laundering Rule

**INVARIANT AL-001**: Legacy metrics MUST be preserved side-by-side with any new metrics.

```
ANTI-LAUNDERING CONTRACT:

1. NEVER overwrite legacy_metrics.* with derived/primary values
2. NEVER rename legacy_metrics.* to imply equivalence with primary_metrics.*
3. ALWAYS emit non-equivalence strings when legacy and primary metrics differ by >5%
4. ALWAYS preserve legacy_metrics.* even when primary_metrics.* is "better"

Violation = audit failure.
```

**Non-Equivalence String Template**:
```
"⚠ NON-EQUIVALENT: legacy_{name} ({legacy_val:.3f}) ≢ primary_{name} ({primary_val:.3f}) — see Metric_Versioning_Policy_v0.1.md"
```

---

## 1. Metric Naming Conventions

### 1.1 Prefix Taxonomy

| Prefix | Meaning | Mutability in v0.1 |
|--------|---------|-------------------|
| `raw_` | Unprocessed metric from underlying system | FROZEN |
| `legacy_` | Pre-existing metric preserved for continuity | FROZEN |
| `derived_` | Computed from raw/legacy metrics without new logic | FROZEN |
| `true_` | Explicitly versioned "ground truth" metric | EXPERIMENTAL |
| `exp_` | Experimental metric, not for production use | EXPERIMENTAL |

### 1.2 Naming Rules

1. **Legacy preservation**: All existing metrics from `USLAShadowLogger` retain their current names prefixed with `legacy_` when exposed alongside new metrics:
   - `divergence_rate` → also exposed as `legacy_divergence_rate`
   - `hard_ok_rate` → also exposed as `legacy_hard_ok_rate`
   - `safe_region_rate` → also exposed as `legacy_safe_region_rate`

2. **True divergence versioning**: Any "true divergence" metric MUST include explicit version:
   - `true_divergence_v1` (not `true_divergence`)
   - `true_divergence_components_v1`

3. **No implicit equivalence**: Different metric names are NEVER semantically equivalent. If two metrics measure similar things differently, both MUST exist with distinct names.

### 1.3 Non-Equivalence Statements

The following pairs are **explicitly non-equivalent**:

```
legacy_divergence_rate ≢ true_divergence_v1
    Reason: legacy_divergence_rate is binary (real_blocked == sim_blocked),
            true_divergence_v1 is a weighted composite

legacy_hard_ok_rate ≢ derived_safety_compliance_rate
    Reason: legacy measures USLA hard_ok snapshots,
            derived measures outcome-based safety

raw_state_delta_p ≢ true_state_tracking_error_v1
    Reason: raw is instantaneous |p_twin - p_real|,
            true_v1 is windowed mean with bias correction

divergence_rate (from window) ≢ true_divergence_v1
    Reason: window divergence_rate is cycle-level mismatch rate,
            true_divergence_v1 is multi-factor composite
```

---

## 2. Truth Vector v0.1

### 2.1 Available Components (Computable from Existing Artifacts)

| Field | Source | Status |
|-------|--------|--------|
| `state_tracking_error` | `mean_delta_p` from `calibration_annex.py` | AVAILABLE |
| `safety_mismatch_rate` | `omega_miss_rate` from `p5_pattern_classifier.py` | AVAILABLE |
| `outcome_calibration_score` | N/A (requires Brier score) | FUTURE v0.2+ |
| `success_div_rate` | `_extract_divergence_component()` | AVAILABLE |
| `omega_div_rate` | `_extract_divergence_component()` | AVAILABLE |
| `blocked_div_rate` | `_extract_divergence_component()` | AVAILABLE |

### 2.2 Truth Vector Schema (v0.1)

```json
{
  "truth_vector_v1": {
    "schema_version": "1.0.0",
    "status": "ADVISORY",
    "components": {
      "state_tracking_error": <float|null>,
      "safety_mismatch_rate": <float|null>,
      "success_div_rate": <float|null>,
      "omega_div_rate": <float|null>,
      "blocked_div_rate": <float|null>,
      "outcome_calibration_score": null
    },
    "component_availability": {
      "state_tracking_error": "AVAILABLE",
      "safety_mismatch_rate": "AVAILABLE",
      "success_div_rate": "AVAILABLE",
      "omega_div_rate": "AVAILABLE",
      "blocked_div_rate": "AVAILABLE",
      "outcome_calibration_score": "PLACEHOLDER_v0.2"
    },
    "aggregation_method": "NONE_IN_v0.1"
  }
}
```

### 2.3 Brier/Log-Loss Placeholder Policy

**Status**: NOT IMPLEMENTED in v0.1

If Brier score or log-loss calibration is requested:

```json
{
  "outcome_calibration_score": null,
  "outcome_calibration_method": "PLACEHOLDER",
  "outcome_calibration_note": "Brier/log-loss scoring deferred to v0.2. Requires per-cycle probability predictions not currently emitted.",
  "required_for_v0.2": [
    "twin_p_predicted per cycle (probability, not binary)",
    "outcome_actual per cycle",
    "minimum 100 cycles for statistical significance"
  ]
}
```

---

## 3. Output Placement

### 3.1 `run_summary.json`

```json
{
  "schema_version": "0.1.0",
  "run_id": "<uuid>",
  "timestamp": "<iso8601>",

  "legacy_metrics": {
    "divergence_rate": 0.123,
    "hard_ok_rate": 0.95,
    "safe_region_rate": 0.88,
    "divergence_count": 15,
    "total_cycles": 122
  },

  "derived_metrics": {
    "governance_alignment_rate": 0.877,
    "state_delta_p_mean": 0.034,
    "state_delta_p_std": 0.012
  },

  "truth_vector_v1": {
    "schema_version": "1.0.0",
    "status": "ADVISORY",
    "components": { ... },
    "component_availability": { ... },
    "aggregation_method": "NONE_IN_v0.1"
  },

  "metric_versioning": {
    "policy_version": "0.1.0",
    "frozen_fields": ["legacy_metrics.*", "derived_metrics.*"],
    "experimental_fields": ["truth_vector_v1.*"]
  }
}
```

### 3.2 Status Signals Block

When emitting GGFL-compatible status signals:

```json
{
  "signal_type": "shadow_audit",
  "schema_version": "1.0.0",
  "mode": "SHADOW",

  "health_contribution": {
    "status": "OK|WARN|BREACH",
    "divergence_band": "GREEN|YELLOW|RED",
    "advisory_only": true
  },

  "metrics_by_tier": {
    "tier_frozen": {
      "legacy_divergence_rate": 0.123,
      "legacy_hard_ok_rate": 0.95
    },
    "tier_advisory": {
      "true_divergence_v1": null,
      "truth_vector_v1_present": true
    }
  },

  "non_equivalence_warnings": [
    "legacy_divergence_rate ≢ true_divergence_v1 (see Metric_Versioning_Policy_v0.1.md)"
  ]
}
```

### 3.3 Evidence Pack Manifest — Governance Blocks

In `manifest.json` under the `governance` section:

```json
{
  "governance": {
    "shadow_audit_metrics": {
      "policy_version": "0.1.0",
      "policy_doc": "docs/system_law/Metric_Versioning_Policy_v0.1.md",

      "frozen_metrics": {
        "legacy_divergence_rate": { "value": 0.123, "frozen": true },
        "legacy_hard_ok_rate": { "value": 0.95, "frozen": true },
        "legacy_safe_region_rate": { "value": 0.88, "frozen": true }
      },

      "experimental_metrics": {
        "true_divergence_v1": {
          "value": null,
          "experimental": true,
          "reason": "composite scoring deferred to v0.2"
        },
        "truth_vector_v1": {
          "present": true,
          "complete": false,
          "missing_components": ["outcome_calibration_score"]
        }
      },

      "metric_lineage": {
        "legacy_divergence_rate": "USLAShadowLogger.get_summary().divergence_rate",
        "state_tracking_error": "calibration_annex.state_delta_p_mean",
        "safety_mismatch_rate": "TDAPatternClassifier.omega_miss_rate"
      }
    }
  }
}
```

---

## 4. Reconciliation Checklist

When metrics conflict, `run_shadow_audit.py` MUST emit the following reconciliation checklist:

### 4.1 Console Output Format

```
=== METRIC RECONCILIATION CHECKLIST ===

[1] LEGACY vs DERIVED DIVERGENCE
    legacy_divergence_rate:     0.123
    derived_alignment_rate:     0.877 (= 1 - legacy)
    STATUS: CONSISTENT (complementary)

[2] STATE TRACKING METRICS
    raw_mean_delta_p:           0.034
    truth_vector.state_tracking_error: 0.034
    STATUS: CONSISTENT (same source)

[3] SAFETY METRICS
    legacy_hard_ok_rate:        0.950
    safety_mismatch_rate:       0.048
    STATUS: ADVISORY (different definitions, both valid)
    NOTE: hard_ok measures USLA state, mismatch measures omega divergence

[4] TRUE DIVERGENCE v1
    true_divergence_v1:         NOT COMPUTED (v0.1)
    legacy_divergence_rate:     0.123
    STATUS: NON-EQUIVALENT
    NOTE: These metrics are NOT comparable. See policy doc.

[5] CALIBRATION SCORE
    outcome_calibration_score:  PLACEHOLDER (v0.2)
    STATUS: DEFERRED
    REQUIRED: per-cycle probability predictions

=== END RECONCILIATION CHECKLIST ===
```

### 4.2 Machine-Readable Format

```json
{
  "reconciliation_v1": {
    "timestamp": "<iso8601>",
    "checks": [
      {
        "check_id": "legacy_vs_derived_divergence",
        "status": "CONSISTENT",
        "reason": "complementary",
        "values": {
          "legacy_divergence_rate": 0.123,
          "derived_alignment_rate": 0.877
        }
      },
      {
        "check_id": "true_divergence_equivalence",
        "status": "NON_EQUIVALENT",
        "reason": "different_definitions",
        "values": {
          "true_divergence_v1": null,
          "legacy_divergence_rate": 0.123
        },
        "policy_reference": "Metric_Versioning_Policy_v0.1.md#non-equivalence"
      }
    ],
    "overall_status": "RECONCILED_WITH_NOTES"
  }
}
```

---

## 5. Frozen vs Experimental Fields

### 5.1 FROZEN in v0.1 (Must Not Change)

| Field Path | Type | Reason |
|------------|------|--------|
| `legacy_metrics.divergence_rate` | float | Historical continuity |
| `legacy_metrics.hard_ok_rate` | float | Historical continuity |
| `legacy_metrics.safe_region_rate` | float | Historical continuity |
| `legacy_metrics.divergence_count` | int | Historical continuity |
| `legacy_metrics.total_cycles` | int | Historical continuity |
| `derived_metrics.state_delta_p_mean` | float | Computed from existing |
| `derived_metrics.state_delta_p_std` | float | Computed from existing |
| `shadow_mode_compliance.*` | bool | SHADOW MODE contract |

### 5.2 EXPERIMENTAL in v0.1 (May Change in v0.2+)

| Field Path | Type | Notes |
|------------|------|-------|
| `truth_vector_v1.*` | object | Schema may evolve |
| `true_divergence_v1` | float|null | Not computed in v0.1 |
| `reconciliation_v1.*` | object | Checklist may expand |
| `exp_*` | any | Explicitly experimental |

### 5.3 Version Upgrade Contract

When upgrading from v0.1 to v0.2:

1. FROZEN fields MUST remain at their paths with identical semantics
2. EXPERIMENTAL fields MAY be:
   - Renamed with `_v2` suffix
   - Removed with deprecation notice
   - Stabilized to FROZEN status
3. New fields MUST start as EXPERIMENTAL
4. `schema_version` MUST increment

---

## 6. Implementation Notes

### 6.1 Code Integration Points

```python
# In run_shadow_audit.py

from backend.topology.first_light.calibration_annex import load_cal_exp1_annex
from backend.topology.first_light.p5_pattern_classifier import TDAPatternClassifier

def build_truth_vector_v1(annex: dict, classifier: TDAPatternClassifier) -> dict:
    """Build truth vector from existing artifacts."""
    return {
        "schema_version": "1.0.0",
        "status": "ADVISORY",
        "components": {
            "state_tracking_error": annex.get("state_delta_p_mean"),
            "safety_mismatch_rate": classifier.get_p5_topology_extension().attractor_miss_rate,
            "success_div_rate": annex.get("success_div_rate"),
            "omega_div_rate": annex.get("omega_div_rate"),
            "blocked_div_rate": annex.get("blocked_div_rate"),
            "outcome_calibration_score": None,  # v0.2+
        },
        "component_availability": {
            "state_tracking_error": "AVAILABLE" if annex.get("state_delta_p_mean") is not None else "MISSING",
            "safety_mismatch_rate": "AVAILABLE",
            "success_div_rate": "AVAILABLE" if annex.get("success_div_rate") is not None else "MISSING",
            "omega_div_rate": "AVAILABLE" if annex.get("omega_div_rate") is not None else "MISSING",
            "blocked_div_rate": "AVAILABLE" if annex.get("blocked_div_rate") is not None else "MISSING",
            "outcome_calibration_score": "PLACEHOLDER_v0.2",
        },
        "aggregation_method": "NONE_IN_v0.1",
    }
```

### 6.2 Reconciliation Emitter

```python
def emit_reconciliation_checklist(
    legacy_metrics: dict,
    derived_metrics: dict,
    truth_vector: dict,
    output_mode: str = "console",  # or "json"
) -> dict:
    """Emit reconciliation checklist per policy."""
    checks = []

    # Check 1: Legacy vs Derived divergence
    leg_div = legacy_metrics.get("divergence_rate", 0)
    der_align = 1 - leg_div
    checks.append({
        "check_id": "legacy_vs_derived_divergence",
        "status": "CONSISTENT",
        "reason": "complementary",
        "values": {"legacy_divergence_rate": leg_div, "derived_alignment_rate": der_align},
    })

    # Check 4: True divergence non-equivalence
    checks.append({
        "check_id": "true_divergence_equivalence",
        "status": "NON_EQUIVALENT",
        "reason": "different_definitions",
        "values": {"true_divergence_v1": None, "legacy_divergence_rate": leg_div},
        "policy_reference": "Metric_Versioning_Policy_v0.1.md#non-equivalence",
    })

    return {
        "reconciliation_v1": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "overall_status": "RECONCILED_WITH_NOTES",
        }
    }
```

---

## 7. Summary

| Aspect | v0.1 Policy |
|--------|-------------|
| Legacy metrics | FROZEN, preserved with `legacy_` prefix |
| True divergence | `true_divergence_v1` — EXPERIMENTAL, not computed |
| Truth vector | Partial — 5/6 components available |
| Brier/log-loss | PLACEHOLDER only, v0.2+ |
| Reconciliation | Required when metrics conflict |
| Non-equivalence | Explicitly documented, emitted in outputs |

**SHADOW MODE CONTRACT**: All metrics in this policy are observational. No metric influences governance decisions in v0.1.
