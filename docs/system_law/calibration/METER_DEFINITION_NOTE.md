# CAL-EXP-1 Meter Definition Note

**Status**: SHADOW MODE (Advisory Only)  
**Date**: 2025-01-XX  
**Schema Version**: 1.0.0

This document defines the canonical metric definitions used for CAL-EXP-1 calibration analysis, with a focus on divergence decomposition and instrumentation verdicts.

## Divergence Metrics Overview

CAL-EXP-1 calibration uses multiple divergence metrics to assess Twin tracking fidelity. These metrics are decomposed into component rates to provide granular diagnostic information.

### 1. overall_any_divergence_rate (Legacy Composite)

**Definition**: Fraction of cycles where any divergence occurs (binary threshold crossing).

**Formula**:
```
overall_any_divergence_rate = count(is_diverged() == True) / window_size
```

**is_diverged() predicate** (from `data_structures_p4.py:426-434`):
```python
def is_diverged(self) -> bool:
    return (
        self.success_diverged or
        self.blocked_diverged or
        self.omega_diverged or
        self.hard_ok_diverged or
        self.divergence_severity != "NONE"
    )
```

**Source**: `cal_exp1_report.json` windows `divergence_rate` field

**Saturation Behavior**: When `mean_delta_p` hovers near the threshold (0.05), small stochastic variations cause the metric to saturate near 1.0, making it insensitive to improvements in `mean_delta_p`. This is a structural property of the threshold choice, not a failure of the calibration.

**Instrumentation Verdict**: When `overall_any_divergence_rate >= 0.99`, the meter is considered **METER_SATURATED** and alternative metrics (state_divergence_rate or mean_delta_p) should be used for calibration tuning.

### 2. state_divergence_rate (Decomposed State Error)

**Definition**: Fraction of cycles where state error (delta_p) exceeds threshold.

**Formula**:
```
state_diverged = mean_delta_p > 0.05  # state_threshold (hardcoded, PROVISIONAL)
state_divergence_rate = 1.0 if state_diverged else 0.0  # Per-window approximation
```

**Threshold**: 0.05 (hardcoded in `data_structures_p4.py:483`, PROVISIONAL)

**Note**: This is a conservative approximation using `mean_delta_p` as a proxy. If `mean_delta_p > threshold`, the window is classified as fully state-diverged (rate = 1.0). True per-cycle state divergence rates would require recomputing from `divergence_log.jsonl` per-cycle snapshots.

**Source**: Derived from `cal_exp1_report.json` windows `mean_delta_p` field

**Basis Field**: `state_divergence_rate_basis = "proxy_mean_delta_p_threshold"` (recorded in decomposition output)

**⚠️ WARNING: Proxy ≠ Component Divergence**

The `state_divergence_rate` computed via `mean_delta_p > threshold` proxy is **NOT** the same as true per-cycle state divergence. The proxy:
- Uses window-averaged `mean_delta_p` rather than per-cycle `delta_p` values
- Classifies entire windows as fully diverged (1.0) or not diverged (0.0)
- Cannot detect partial window divergence (e.g., 30% of cycles diverged)

True component divergence would require:
- Per-cycle `DivergenceSnapshot` data from `divergence_log.jsonl`
- Computing `divergence_severity != "NONE"` for each cycle
- Aggregating per-cycle state divergence counts

**Use the proxy for diagnostic purposes only. Do not treat it as equivalent to true per-cycle state divergence rates.**

**Advantage Over Legacy Metric**: Provides a separate signal for state tracking errors, distinct from outcome prediction errors. Less prone to saturation when `mean_delta_p` is near threshold.

### 3. outcome_divergence_rate_success (Decomposed Outcome Error)

**Definition**: Fraction of cycles where Twin predicted success incorrectly.

**Formula**:
```
outcome_divergence_rate_success = count(real.success != twin.predicted_success) / window_size
```

**Source**: `DivergenceSnapshot.success_diverged` field (per-cycle data)

**Availability**: **NOT available in aggregated windows**. Requires per-cycle `DivergenceSnapshot` data from `divergence_log.jsonl`, which is not included in `cal_exp1_report.json`.

**Status**: Set to `None` in decomposition output with explicit note explaining missing data requirement.

**Basis Field**: `outcome_divergence_basis = "UNAVAILABLE_NO_PER_CYCLE_COMPONENTS"` (recorded in decomposition output when outcome rates are None)

### 4. outcome_divergence_rate_omega (Decomposed Outcome Error)

**Definition**: Fraction of cycles where Twin predicted omega membership incorrectly.

**Formula**:
```
outcome_divergence_rate_omega = count(real.in_omega != twin.predicted_in_omega) / window_size
```

**Source**: `DivergenceSnapshot.omega_diverged` field (per-cycle data)

**Availability**: **NOT available in aggregated windows**. Requires per-cycle `DivergenceSnapshot` data from `divergence_log.jsonl`.

**Status**: Set to `None` in decomposition output with explicit note explaining missing data requirement.

**Basis Field**: See `outcome_divergence_basis` (applies to all outcome divergence rates).

### 5. outcome_divergence_rate_blocked (Decomposed Outcome Error)

**Definition**: Fraction of cycles where Twin predicted blocked status incorrectly.

**Formula**:
```
outcome_divergence_rate_blocked = count(real.real_blocked != twin.predicted_blocked) / window_size
```

**Source**: `DivergenceSnapshot.blocked_diverged` field (per-cycle data)

**Availability**: **NOT available in aggregated windows**. Requires per-cycle `DivergenceSnapshot` data from `divergence_log.jsonl`.

**Status**: Set to `None` in decomposition output with explicit note explaining missing data requirement.

**Basis Field**: See `outcome_divergence_basis` (applies to all outcome divergence rates).

### 6. mean_delta_p (Continuous Magnitude Metric)

**Definition**: Mean of absolute state error (delta_p) over all cycles in a window.

**Formula**:
```
mean_delta_p = sum(abs(delta_p[i]) for i in window) / window_size
```

**delta_p computation** (from `data_structures_p4.py:454-455`):
```python
delta_p = (H_delta + rho_delta + tau_delta + beta_delta) / 4.0
```

Where:
- `H_delta = abs(real.H - twin.twin_H)`
- `rho_delta = abs(real.rho - twin.twin_rho)`
- `tau_delta = abs(real.tau - twin.twin_tau)`
- `beta_delta = abs(real.beta - twin.twin_beta)`

**Source**: `cal_exp1_report.json` windows `mean_delta_p` field

**Advantage**: Continuous metric that directly responds to state tracking improvements, not gated by binary thresholds. Recommended as primary calibration objective when `overall_any_divergence_rate` is saturated.

## Instrumentation Verdicts

The runtime profile calibration annex includes an `instrumentation_verdict` field to indicate meter health:

- **METER_OK**: All metrics are within normal operating ranges.
- **METER_SATURATED**: `overall_any_divergence_rate >= 0.99`, indicating the legacy composite metric is saturated and alternative metrics should be used.
- **INSUFFICIENT_DATA**: Required fields for decomposition are missing from windows.

When `METER_SATURATED` is detected, the annex includes `instrumentation_notes` explaining:
- Which metric is saturated and why
- Recommendation to use `state_divergence_rate` or `mean_delta_p` for calibration tuning
- Note that saturation is a structural property of the threshold choice, not a calibration failure

## Thresholds (PROVISIONAL)

All thresholds are **PROVISIONAL** and should NOT be adjusted during calibration per doctrine. The model must be improved, not the ruler.

| Threshold | Value | Source | Status |
|-----------|-------|--------|--------|
| state_threshold | 0.05 | data_structures_p4.py:483 | PROVISIONAL |
| threshold_none | 0.01 | data_structures_p4.py:496 | PROVISIONAL |
| threshold_info | 0.05 | data_structures_p4.py:497 | PROVISIONAL |
| threshold_warn | 0.15 | data_structures_p4.py:498 | PROVISIONAL |

## Recommended Calibration Metrics

For calibration tuning (e.g., UPGRADE-1 LR tuning), the recommended primary objective is:

**Primary**: `mean_delta_p` (continuous magnitude, responsive to improvements)

**Secondary** (dashboard signals, not objectives):
- `state_divergence_rate`: Smoke signal for state tracking failures
- `overall_any_divergence_rate`: Gross failure indicator (may be saturated)
- `phase_lag_xcorr`: Temporal alignment indicator

## SHADOW MODE

All divergence decomposition and instrumentation verdicts are **advisory only** and do not gate calibration results or invalidate evidence packs. Missing outcome divergence rates (due to unavailable per-cycle data) are not errors and are documented with explicit reasons.

---

**END OF METER DEFINITION NOTE**

