# RUN_SHADOW_AUDIT v0.1 — Metrics Appendix

> **Status**: ACTIVE SPECIFICATION
> **Version**: 0.1.0
> **Date**: 2025-12-12
> **Parent Doc**: `docs/system_law/Metric_Versioning_Policy_v0.1.md`
> **Canonical Contract**: `docs/system_law/calibration/RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`

---

## 1. Purpose

This appendix provides the canonical metric definitions, JSON paths, and output structure for `run_shadow_audit.py v0.1`. All metrics are **composition-only** — no new formulas, thresholds, or science.

---

## 2. Output Structure

### 2.1 `summary.json` Schema

```json
{
  "schema_version": "1.0.0",
  "run_id": "<uuid>",
  "timestamp": "<iso8601>",
  "mode": "SHADOW",

  "legacy_metrics": {
    "divergence_rate": 0.123,
    "hard_ok_rate": 0.95,
    "safe_region_rate": 0.88,
    "divergence_count": 15,
    "total_cycles": 122
  },

  "primary_metrics": {
    "mean_delta_p": 0.034,
    "omega_mismatch_rate": 0.052
  },

  "diagnostic_metrics": {
    "success_div_rate": 0.089,
    "blocked_div_rate": 0.012,
    "phase_lag_xcorr": 0.23,
    "mean_H_error": null,
    "mean_rho_error": null,
    "mean_tau_error": null,
    "mean_beta_error": null
  },

  "truth_vector_v0_1": { ... },

  "anti_laundering": {
    "legacy_preserved": true,
    "non_equivalence_warnings": []
  },

  "status": "OK"
}
```

---

## 3. Metric Tiers

> **Terminology Note**: Metrics containing "safe" or "safety" (e.g., `safe_region_rate`,
> `omega_mismatch_rate`/`safety_mismatch_rate`) measure **simulator prediction agreement**,
> not behavioral safety guarantees. "Safe region" refers to the mathematical Ω-region in
> state space (a dynamical systems concept), not certified safe behavior. All metrics are
> SHADOW/ADVISORY only and make no safety claims.

### 3.1 `legacy_metrics.*` — FROZEN

**Invariant**: Always present if source artifact exists. Never modified, never aliased.

| Key | Type | Source | Formula | Notes |
|-----|------|--------|---------|-------|
| `divergence_rate` | float | `USLAShadowLogger.get_summary()` | `divergence_count / total_cycles` | Binary: real_blocked == sim_blocked |
| `hard_ok_rate` | float | `USLAShadowLogger.get_summary()` | `sum(hard_ok) / total_cycles` | USLA hard_ok predicate |
| `safe_region_rate` | float | `USLAShadowLogger.get_summary()` | `sum(in_safe_region) / total_cycles` | USLA safe region membership |
| `divergence_count` | int | `USLAShadowLogger` | count of misaligned cycles | — |
| `total_cycles` | int | `USLAShadowLogger` | total cycles logged | — |

**JSON Path**: `$.legacy_metrics.*`

**Source File**: `backend/topology/usla_shadow.py:244-263`

---

### 3.2 `primary_metrics.*` — FROZEN (v0.1 Minimal Truth Set)

**Invariant**: The two metrics required for calibration decision-making.

| Key | Type | Source | Formula | Notes |
|-----|------|--------|---------|-------|
| `mean_delta_p` | float | `calibration_annex.state_delta_p_mean` | `mean(abs(p_twin - p_real))` | State tracking error |
| `omega_mismatch_rate` | float | `TDAPatternClassifier.omega_miss_rate` | `count(omega_real != omega_twin) / n` | Safety mismatch |

**JSON Path**: `$.primary_metrics.*`

**Source Files**:
- `mean_delta_p`: `backend/topology/first_light/calibration_annex.py:52-57`
- `omega_mismatch_rate`: `backend/topology/first_light/p5_pattern_classifier.py:278-282`

**Why These Two?**
1. `mean_delta_p` — primary calibration objective per `METRIC_DEFINITIONS.md`
2. `omega_mismatch_rate` — safety-critical divergence (attractor miss detection)

---

### 3.3 `diagnostic_metrics.*` — EXPERIMENTAL

**Invariant**: Optional. May be `null` if source unavailable. Schema may change in v0.2+.

| Key | Type | Source | Availability | Notes |
|-----|------|--------|--------------|-------|
| `success_div_rate` | float\|null | `_extract_divergence_component()` | If twin predictions logged | Success prediction mismatch |
| `blocked_div_rate` | float\|null | `_extract_divergence_component()` | If twin predictions logged | Blocked prediction mismatch |
| `phase_lag_xcorr` | float\|null | `cal_exp1_report.windows[*]` | If cal_exp1 ran | Temporal alignment |
| `delta_bias` | float\|null | DEPRECATED | — | **DO NOT USE** |
| `mean_H_error` | float\|null | recomputation required | Requires twin state | Per-component H |
| `mean_rho_error` | float\|null | recomputation required | Requires twin state | Per-component rho |
| `mean_tau_error` | float\|null | recomputation required | Requires twin state | Per-component tau |
| `mean_beta_error` | float\|null | recomputation required | Requires twin state | Per-component beta |

**JSON Path**: `$.diagnostic_metrics.*`

---

## 4. truth_vector_v0_1 — Composition Only

### 4.1 Definition

The `truth_vector_v0_1` is a **pure composition** of existing metrics. It introduces **no new formulas**.

```json
{
  "truth_vector_v0_1": {
    "schema_version": "0.1.0",
    "status": "ADVISORY",
    "aggregation": "NONE",

    "components": {
      "state_tracking_error": <primary_metrics.mean_delta_p>,
      "safety_mismatch_rate": <primary_metrics.omega_mismatch_rate>,
      "success_div_rate": <diagnostic_metrics.success_div_rate | null>,
      "blocked_div_rate": <diagnostic_metrics.blocked_div_rate | null>,
      "outcome_calibration_score": null
    },

    "availability": {
      "state_tracking_error": "AVAILABLE",
      "safety_mismatch_rate": "AVAILABLE",
      "success_div_rate": "CONDITIONAL",
      "blocked_div_rate": "CONDITIONAL",
      "outcome_calibration_score": "FUTURE_v0.2"
    },

    "placeholders": {
      "outcome_calibration_score": {
        "method": "BRIER_OR_LOGLOSS",
        "status": "NOT_IMPLEMENTED",
        "reason": "requires per-cycle probability predictions (not binary)",
        "spec_ref": "METRIC_DEFINITIONS.md#brier-logloss"
      }
    }
  }
}
```

### 4.2 Composition Rules

| Component | Source Path | Transform | Status |
|-----------|-------------|-----------|--------|
| `state_tracking_error` | `primary_metrics.mean_delta_p` | identity (no transform) | FROZEN |
| `safety_mismatch_rate` | `primary_metrics.omega_mismatch_rate` | identity (no transform) | FROZEN |
| `success_div_rate` | `diagnostic_metrics.success_div_rate` | identity (no transform) | CONDITIONAL |
| `blocked_div_rate` | `diagnostic_metrics.blocked_div_rate` | identity (no transform) | CONDITIONAL |
| `outcome_calibration_score` | N/A | placeholder | FUTURE_v0.2 |

### 4.3 What truth_vector_v0_1 Does NOT Do

1. **NO aggregation** — components are not combined into a single score
2. **NO weighting** — no weight factors applied
3. **NO thresholds** — no pass/fail determination
4. **NO Brier/log-loss** — requires probability outputs not yet emitted
5. **NO new formulas** — pure passthrough of existing values

---

## 5. Anti-Laundering Rule

### 5.1 Contract

```
ANTI-LAUNDERING RULE (AL-001)

Purpose: Prevent metric confusion between legacy and primary/truth metrics.

1. PRESERVATION: legacy_metrics.* MUST be emitted alongside primary_metrics.*
2. NO OVERWRITE: primary_metrics values MUST NOT replace legacy_metrics values
3. NO ALIASING: primary_metrics.mean_delta_p ≠ legacy_metrics.divergence_rate (even if numerically close)
4. EXPLICIT WARNINGS: When |legacy - primary| > threshold, emit non-equivalence warning

Violation of AL-001 = audit artifact rejection.
```

### 5.2 Non-Equivalence Detection

When emitting `summary.json`, check for non-equivalence:

```python
def check_non_equivalence(legacy: dict, primary: dict) -> list[str]:
    """
    Returns list of non-equivalence warning strings.
    Threshold: 5% relative difference or 0.01 absolute (whichever is larger).
    """
    warnings = []

    # divergence_rate vs (1 - mean_delta_p) is ALWAYS non-equivalent
    # These metrics measure different things and should never be compared
    warnings.append(
        "legacy_metrics.divergence_rate ≢ primary_metrics.mean_delta_p "
        "(different definitions: binary mismatch vs continuous state error)"
    )

    # safe_region_rate vs omega_mismatch_rate
    leg_sr = legacy.get("safe_region_rate", 0)
    prim_om = primary.get("omega_mismatch_rate", 0)
    # Note: these are complementary but NOT equivalent
    warnings.append(
        f"legacy_metrics.safe_region_rate ({leg_sr:.3f}) ≢ "
        f"primary_metrics.omega_mismatch_rate ({prim_om:.3f}) "
        "(safe_region is USLA predicate; omega_mismatch is twin prediction error)"
    )

    return warnings
```

### 5.3 Output Example

```json
{
  "anti_laundering": {
    "legacy_preserved": true,
    "side_by_side": true,
    "non_equivalence_warnings": [
      "legacy_metrics.divergence_rate ≢ primary_metrics.mean_delta_p (different definitions: binary mismatch vs continuous state error)",
      "legacy_metrics.safe_region_rate (0.880) ≢ primary_metrics.omega_mismatch_rate (0.052) (safe_region is USLA predicate; omega_mismatch is twin prediction error)"
    ],
    "policy_version": "0.1.0",
    "policy_doc": "docs/system_law/Metric_Versioning_Policy_v0.1.md"
  }
}
```

---

## 6. Brier/Log-Loss Placeholder

### 6.1 Status

**NOT IMPLEMENTED** in v0.1.

### 6.2 Requirements for v0.2

To enable Brier or log-loss calibration scoring, the following artifacts must exist:

| Requirement | Current Status | Needed For |
|-------------|----------------|------------|
| Per-cycle `twin_p_predicted` (probability [0,1]) | NOT EMITTED | Brier score |
| Per-cycle `outcome_actual` (binary) | AVAILABLE | Brier score |
| Minimum 100 cycles | AVAILABLE | Statistical significance |
| Calibrated probability outputs from Twin | NOT IMPLEMENTED | Log-loss |

### 6.3 Placeholder Schema

```json
{
  "outcome_calibration_score": null,
  "outcome_calibration": {
    "method": "PLACEHOLDER",
    "available_in": "v0.2",
    "blocker": "twin emits binary predictions, not probabilities",
    "requirements": [
      "twin_p_predicted: float in [0,1] per cycle",
      "outcome_actual: bool per cycle",
      "n_cycles >= 100"
    ]
  }
}
```

---

## 7. Smoke-Test Readiness Checklist

### 7.1 Pre-Run Checks

Before running `run_shadow_audit.py v0.1`:

- [ ] **Source artifacts exist**
  - [ ] `USLAShadowLogger` output available (JSONL or summary)
  - [ ] `calibration_annex` loadable (cal_exp1_report.json or equivalent)
  - [ ] `TDAPatternClassifier` can compute omega_miss_rate

- [ ] **Schema compliance**
  - [ ] Input conforms to expected JSONL schema
  - [ ] Output directory is writable

### 7.2 Output Validation Checks

After running, verify:

- [ ] **summary.json exists** and is valid JSON
- [ ] **schema_version** = `"1.0.0"`
- [ ] **mode** = `"SHADOW"`

#### 7.2.1 legacy_metrics Checks

- [ ] `legacy_metrics.divergence_rate` is present (float)
- [ ] `legacy_metrics.hard_ok_rate` is present (float)
- [ ] `legacy_metrics.safe_region_rate` is present (float)
- [ ] `legacy_metrics.divergence_count` is present (int)
- [ ] `legacy_metrics.total_cycles` is present (int)
- [ ] All values are in valid range [0, 1] for rates

#### 7.2.2 primary_metrics Checks

- [ ] `primary_metrics.mean_delta_p` is present (float)
- [ ] `primary_metrics.omega_mismatch_rate` is present (float)
- [ ] Values are >= 0 (non-negative)

#### 7.2.3 truth_vector_v0_1 Checks

- [ ] `truth_vector_v0_1.schema_version` = `"0.1.0"`
- [ ] `truth_vector_v0_1.status` = `"ADVISORY"`
- [ ] `truth_vector_v0_1.aggregation` = `"NONE"`
- [ ] `truth_vector_v0_1.components.state_tracking_error` == `primary_metrics.mean_delta_p`
- [ ] `truth_vector_v0_1.components.safety_mismatch_rate` == `primary_metrics.omega_mismatch_rate`
- [ ] `truth_vector_v0_1.components.outcome_calibration_score` == `null`
- [ ] `truth_vector_v0_1.placeholders.outcome_calibration_score.status` = `"NOT_IMPLEMENTED"`

#### 7.2.4 Anti-Laundering Checks

- [ ] `anti_laundering.legacy_preserved` == `true`
- [ ] `anti_laundering.non_equivalence_warnings` is array (may be empty)
- [ ] `anti_laundering.policy_version` = `"0.1.0"`

#### 7.2.5 Non-Equivalence Enforcement

- [ ] Non-equivalence warning present for divergence_rate vs mean_delta_p
- [ ] Non-equivalence warning present for safe_region_rate vs omega_mismatch_rate
- [ ] **NO** warning implies these metrics are equivalent

### 7.3 Negative Checks (Must NOT Happen)

- [ ] **NO** `true_divergence` field without version suffix
- [ ] **NO** `outcome_calibration_score` with non-null value
- [ ] **NO** overwriting of legacy_metrics values
- [ ] **NO** aggregated score in truth_vector_v0_1
- [ ] **NO** new formulas or threshold computations

### 7.4 CI Integration Check

```bash
# Validate output against schema
python -m jsonschema -i summary.json docs/system_law/schemas/shadow_audit_summary.schema.json

# Check anti-laundering invariant
jq '.anti_laundering.legacy_preserved' summary.json  # must be true

# Check truth_vector composition
jq '.truth_vector_v0_1.components.state_tracking_error == .primary_metrics.mean_delta_p' summary.json  # must be true

# Check no aggregation
jq '.truth_vector_v0_1.aggregation' summary.json  # must be "NONE"
```

---

## 8. Appendix: JSON Paths Quick Reference

| Metric | JSON Path | Tier |
|--------|-----------|------|
| divergence_rate | `$.legacy_metrics.divergence_rate` | FROZEN |
| hard_ok_rate | `$.legacy_metrics.hard_ok_rate` | FROZEN |
| safe_region_rate | `$.legacy_metrics.safe_region_rate` | FROZEN |
| divergence_count | `$.legacy_metrics.divergence_count` | FROZEN |
| total_cycles | `$.legacy_metrics.total_cycles` | FROZEN |
| mean_delta_p | `$.primary_metrics.mean_delta_p` | FROZEN |
| omega_mismatch_rate | `$.primary_metrics.omega_mismatch_rate` | FROZEN |
| success_div_rate | `$.diagnostic_metrics.success_div_rate` | EXPERIMENTAL |
| blocked_div_rate | `$.diagnostic_metrics.blocked_div_rate` | EXPERIMENTAL |
| phase_lag_xcorr | `$.diagnostic_metrics.phase_lag_xcorr` | EXPERIMENTAL |
| state_tracking_error | `$.truth_vector_v0_1.components.state_tracking_error` | ADVISORY |
| safety_mismatch_rate | `$.truth_vector_v0_1.components.safety_mismatch_rate` | ADVISORY |
| outcome_calibration_score | `$.truth_vector_v0_1.components.outcome_calibration_score` | FUTURE |

---

## 9. Implementation Requirements (v0.1)

### 9.1 Exact JSON Paths — Mandatory Structure

```
summary.json
├── schema_version          : string = "1.0.0"
├── run_id                  : string
├── timestamp               : string (ISO8601)
├── mode                    : string = "SHADOW"
│
├── legacy_metrics/         : object (REQUIRED if USLAShadowLogger ran)
│   ├── divergence_rate     : float [0.0, 1.0]
│   ├── hard_ok_rate        : float [0.0, 1.0]
│   ├── safe_region_rate    : float [0.0, 1.0]
│   ├── divergence_count    : int >= 0
│   └── total_cycles        : int > 0
│
├── primary_metrics/        : object (REQUIRED)
│   ├── mean_delta_p        : float >= 0.0
│   └── omega_mismatch_rate : float [0.0, 1.0]
│
├── diagnostic_metrics/     : object (OPTIONAL)
│   ├── success_div_rate    : float|null
│   ├── blocked_div_rate    : float|null
│   ├── phase_lag_xcorr     : float|null
│   ├── mean_H_error        : float|null
│   ├── mean_rho_error      : float|null
│   ├── mean_tau_error      : float|null
│   └── mean_beta_error     : float|null
│
├── truth_vector_v0_1/      : object (REQUIRED)
│   ├── schema_version      : string = "0.1.0"
│   ├── status              : string = "ADVISORY"
│   ├── aggregation         : string = "NONE"
│   ├── components/         : object
│   │   ├── state_tracking_error      : float (== primary_metrics.mean_delta_p)
│   │   ├── safety_mismatch_rate      : float (== primary_metrics.omega_mismatch_rate)
│   │   ├── success_div_rate          : float|null
│   │   ├── blocked_div_rate          : float|null
│   │   └── outcome_calibration_score : null
│   └── placeholders/       : object
│
├── anti_laundering/        : object (REQUIRED)
│   ├── legacy_preserved    : bool = true
│   ├── non_equivalence_warnings : array<string>
│   └── policy_version      : string = "0.1.0"
│
└── status                  : string ("OK"|"WARN"|"ERROR")
```

### 9.2 Required Presence Rules (Non-Gating)

> **Non-Gating Clarification**: Per canonical contract (`RUN_SHADOW_AUDIT_V0_1_CONTRACT.md`
> lines 35-41), SHADOW mode exit semantics are: **exit 0 = ran to completion (success or
> warnings)**, **exit 1 = script crash only**. Missing metrics MUST NOT cause exit 1.
> Instead, degrade gracefully with `schema_ok=false` and `advisory_warnings`.

| Field | Condition | Action if Missing |
|-------|-----------|-------------------|
| `legacy_metrics.*` | `USLAShadowLogger` output exists | `schema_ok=false`, warn, emit `null` placeholders |
| `primary_metrics.mean_delta_p` | `calibration_annex` loadable | `schema_ok=false`, warn, emit `null` |
| `primary_metrics.omega_mismatch_rate` | `TDAPatternClassifier` ran | `schema_ok=false`, warn, emit `null` |
| `diagnostic_metrics.*` | source available | Set to `null` (do not omit key) |
| `truth_vector_v0_1` | always | Emit with available components; others `null` |
| `anti_laundering` | always | Always emit; `legacy_preserved=false` if legacy missing |

**Graceful Degradation Output** (missing `legacy_metrics` example):
```json
{
  "schema_ok": false,
  "advisory_warnings": [
    "legacy_metrics unavailable: USLAShadowLogger output not found",
    "primary_metrics.mean_delta_p unavailable: calibration_annex not loadable"
  ],
  "legacy_metrics": {
    "divergence_rate": null,
    "hard_ok_rate": null,
    "safe_region_rate": null,
    "divergence_count": null,
    "total_cycles": null
  },
  "primary_metrics": {
    "mean_delta_p": null,
    "omega_mismatch_rate": 0.051
  },
  "anti_laundering": {
    "legacy_preserved": false,
    "non_equivalence_warnings": [
      "legacy_metrics unavailable — cannot verify non-equivalence"
    ]
  },
  "status": "WARN"
}
```

**Exit code remains 0** — script ran to completion. Consumers check `schema_ok` and `status`.

### 9.3 Anti-Laundering Machine Checks (AL-001)

**Implementers MUST enforce these checks programmatically:**

| Check ID | Condition | Violation If |
|----------|-----------|--------------|
| AL-001-A | `legacy_metrics` present alongside `primary_metrics` | `legacy_metrics` missing (degraded = `legacy_preserved=false`) |
| AL-001-B | `legacy_metrics.*` values unchanged from source | Any legacy value differs from `USLAShadowLogger.get_summary()` |
| AL-001-C | `anti_laundering.legacy_preserved` reflects truth | Field lies about preservation state |
| AL-001-D | `non_equivalence_warnings` present when applicable | Array empty when both tiers have values and differ |
| AL-001-E | `truth_vector_v0_1.aggregation == "NONE"` | Any other value (e.g., "WEIGHTED", "MEAN") |
| AL-001-F | `truth_vector_v0_1.components.*` are identity copies | Any transform applied to source values |

**Violation Response** (non-gating): Set `status="WARN"`, add to `advisory_warnings`:
```json
{
  "status": "WARN",
  "advisory_warnings": [
    "AL-001-B violation: legacy_metrics.divergence_rate differs from source (0.87 vs 0.23)"
  ]
}
```

**Exit code remains 0** — AL-001 violations are audit warnings, not script failures.

### 9.4 Acceptance Tests (6 Tests)

**AT-01: Legacy Tier Presence**
- Input: Valid `USLAShadowLogger` output with 100 cycles, 23 divergences
- Expected: `legacy_metrics.divergence_rate == 0.23`, all 5 fields present
- Verify: `jq '.legacy_metrics | keys | length' == 5`

**AT-02: Primary Tier Passthrough**
- Input: `calibration_annex` with `state_delta_p_mean = 0.0472`
- Expected: `primary_metrics.mean_delta_p == 0.0472` (no rounding, no transform)
- Verify: `jq '.primary_metrics.mean_delta_p == 0.0472'`

**AT-03: Truth Vector Composition**
- Input: `primary_metrics.mean_delta_p = 0.034`, `omega_mismatch_rate = 0.051`
- Expected: `truth_vector_v0_1.components.state_tracking_error == 0.034`
- Verify: `jq '.truth_vector_v0_1.components.state_tracking_error == .primary_metrics.mean_delta_p'`

**AT-04: Anti-Laundering Block**
- Input: Any valid run
- Expected: `anti_laundering.legacy_preserved == true`, warnings array has ≥2 entries
- Verify: `jq '.anti_laundering.legacy_preserved and (.anti_laundering.non_equivalence_warnings | length >= 2)'`

**AT-05: No Aggregation**
- Input: Any valid run
- Expected: `truth_vector_v0_1.aggregation == "NONE"`, no `composite_score` field
- Verify: `jq '.truth_vector_v0_1.aggregation == "NONE" and (.truth_vector_v0_1.composite_score == null)'`

**AT-06: Diagnostic Nulls**
- Input: Run without twin state (per-component errors unavailable)
- Expected: `diagnostic_metrics.mean_H_error == null` (key present, value null)
- Verify: `jq '.diagnostic_metrics | has("mean_H_error") and .mean_H_error == null'`

### 9.5 Do Not Implement (v0.1 Forbidden List)

| ID | Forbidden Upgrade | Rationale |
|----|-------------------|-----------|
| **DNI-01** | Scalar composite scoring (e.g., `overall_score = 0.85`) | Requires weighting = new science |
| **DNI-02** | Kalman filter or state estimator for twin | New algorithm = new science |
| **DNI-03** | Adaptive thresholds (e.g., `if divergence > dynamic_threshold`) | New heuristic = new science |
| **DNI-04** | Brier score / log-loss computation | Requires probability predictions not yet emitted |
| **DNI-05** | Weighted aggregation of truth_vector components | `aggregation` MUST remain `"NONE"` |

**If any DNI item is required, it MUST go through v0.2+ spec review with calibration experiments.**

---

## 10. Document Control

| Version | Date | Change | Author |
|---------|------|--------|--------|
| 0.1.0 | 2025-12-12 | Initial appendix | CLAUDE U |
| 0.1.1 | 2025-12-12 | Added Section 9: Implementation Requirements | CLAUDE U |
| 0.1.2 | 2025-12-12 | Fixed exit semantics conflict — non-gating degradation | CLAUDE U |

**Frozen Fields**: All `legacy_metrics.*` and `primary_metrics.*` paths are FROZEN.

**SHADOW MODE CONTRACT**: This appendix defines observational metrics only. No metric defined here influences governance decisions.

---

**END OF APPENDIX**
