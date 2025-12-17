# Calibration Metric Definitions

---

## Ratification Block

| Field | Value |
|-------|-------|
| **Status** | FROZEN |
| **Owner** | Calibration Governance |
| **Ratified-by** | STRATCOM (COMPLETE) |
| **Version** | 1.1.0 |
| **Last Updated** | 2025-12-12 |
| **Change Policy** | ADDITIVE ONLY - existing definitions may not be modified |
| **Authority** | This document is the canonical source for metric semantics in the calibration campaign |

---

This document defines the exact formulas and source locations for all metrics used in CAL-EXP-1 calibration.

---

## 1. divergence_rate

**Definition**: Fraction of cycles where `is_diverged()` returns True within a window.

**Formula**:
```
divergence_rate = count(is_diverged() == True) / window_size
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

**Note**: This is an "any divergence" metric. A cycle is diverged if:
- success_diverged: `real.success != twin.predicted_success`
- blocked_diverged: `real.real_blocked != twin.predicted_blocked`
- omega_diverged: `real.in_omega != twin.predicted_in_omega`
- hard_ok_diverged: `real.hard_ok != twin.predicted_hard_ok`
- OR divergence_severity != "NONE" (state divergence)

**Source File**: `backend/topology/first_light/data_structures_p4.py`
**Computed By**: `scripts/run_p5_cal_exp1.py` (CalExp1Runner._finalize_window)

---

## 2. mean_delta_p

**Definition**: Mean of `abs(delta_p)` over all cycles in a window.

**Formula**:
```
mean_delta_p = sum(abs(delta_p[i]) for i in window) / window_size
```

**delta_p computation** (from `data_structures_p4.py:454-455`):
```python
# Composite delta_p (weighted average of state component deltas)
delta_p = (H_delta + rho_delta + tau_delta + beta_delta) / 4.0
```

Where:
- `H_delta = abs(real.H - twin.twin_H)`
- `rho_delta = abs(real.rho - twin.twin_rho)`
- `tau_delta = abs(real.tau - twin.twin_tau)`
- `beta_delta = abs(real.beta - twin.twin_beta)`

**Note**: delta_p is the average absolute state error across all 4 components (H, rho, tau, beta). It measures state tracking fidelity.

**Source File**: `backend/topology/first_light/data_structures_p4.py`
**Computed By**: `scripts/run_p5_cal_exp1.py` (via DivergenceSnapshot.from_observations)

---

## 3. delta_bias (DEPRECATED - HEURISTIC ONLY)

**Definition**: A heuristic proxy for directional bias.

**Formula** (from `run_p5_cal_exp1.py:337`):
```python
delta_bias = mean_delta_p * (0.5 - divergence_rate)
```

**WARNING**: This is NOT true signed bias (Twin - Real). It is a heuristic that:
- When divergence_rate = 1.0: `delta_bias = -0.5 * mean_delta_p`
- When divergence_rate = 0.5: `delta_bias = 0`
- When divergence_rate = 0.0: `delta_bias = +0.5 * mean_delta_p`

This metric conflates magnitude with direction and should NOT be used as the primary calibration metric.

**Source File**: `scripts/run_p5_cal_exp1.py`
**Recommendation**: Replace with true signed bias computation.

---

## 4. success_divergence_rate (DECOMPOSITION)

**Definition**: Fraction of cycles where Twin predicted success incorrectly.

**Formula**:
```
success_divergence_rate = count(real.success != twin.predicted_success) / window_size
```

**Source**: `DivergenceSnapshot.success_diverged` field
**Status**: Available per-cycle but NOT aggregated in cal_exp1_report.json

---

## 5. omega_divergence_rate (DECOMPOSITION)

**Definition**: Fraction of cycles where Twin predicted omega membership incorrectly.

**Formula**:
```
omega_divergence_rate = count(real.in_omega != twin.predicted_in_omega) / window_size
```

**Source**: `DivergenceSnapshot.omega_diverged` field
**Status**: Available per-cycle but NOT aggregated in cal_exp1_report.json

---

## 6. blocked_divergence_rate (DECOMPOSITION)

**Definition**: Fraction of cycles where Twin predicted blocked status incorrectly.

**Formula**:
```
blocked_divergence_rate = count(real.real_blocked != twin.predicted_blocked) / window_size
```

**Source**: `DivergenceSnapshot.blocked_diverged` field
**Status**: Available per-cycle but NOT aggregated in cal_exp1_report.json

---

## 7. state_divergence_rate (DECOMPOSITION)

**Definition**: Fraction of cycles where state error (delta_p) exceeds threshold.

**Formula**:
```
state_diverged = delta_p > 0.05  # hardcoded threshold
state_divergence_rate = count(state_diverged) / window_size
```

**Threshold**: 0.05 (hardcoded in `data_structures_p4.py:483`)

**Source**: `DivergenceSnapshot.divergence_type == "STATE" or "BOTH"`
**Status**: Available per-cycle but NOT aggregated in cal_exp1_report.json

---

## 8. phase_lag_xcorr

**Definition**: Lag-1 autocorrelation of delta_p time series.

**Formula** (from `run_p5_cal_exp1.py:368-393`):
```python
# Compute lag-1 autocorrelation of delta_p series
series = delta_p_values
n = len(series)
mean = sum(series) / n
cov_0 = sum((x - mean) ** 2 for x in series) / n
cov_1 = sum((series[i] - mean) * (series[i+1] - mean) for i in range(n-1)) / (n-1)
phase_lag_xcorr = cov_1 / cov_0
```

**Interpretation**: High positive values indicate Twin lags behind Real state.

**Source File**: `scripts/run_p5_cal_exp1.py`

---

## Threshold Summary

| Threshold | Value | Source | Status |
|-----------|-------|--------|--------|
| state_threshold | 0.05 | data_structures_p4.py:483 | PROVISIONAL |
| threshold_none | 0.01 | data_structures_p4.py:496 | PROVISIONAL |
| threshold_info | 0.05 | data_structures_p4.py:497 | PROVISIONAL |
| threshold_warn | 0.15 | data_structures_p4.py:498 | PROVISIONAL |

**Note**: All thresholds are PROVISIONAL and should NOT be adjusted during calibration per doctrine.

---

## Availability in Artifacts

| Metric | cal_exp1_report.json | synthetic_trace.jsonl | Needs Recomputation |
|--------|---------------------|----------------------|---------------------|
| divergence_rate | YES (per-window) | NO | NO |
| mean_delta_p | YES (per-window) | NO | YES (need twin state) |
| delta_bias | YES (per-window) | NO | NO |
| success_divergence_rate | NO | Partial (real only) | YES |
| omega_divergence_rate | NO | Partial (real only) | YES |
| blocked_divergence_rate | NO | Partial (real only) | YES |
| state_divergence_rate | NO | Partial (real only) | YES |
| phase_lag_xcorr | YES (per-window) | NO | NO |

**Critical Gap**: Divergence decomposition requires both real observations AND twin predictions. The current artifacts only store `synthetic_trace.jsonl` (real telemetry). Twin predictions are not persisted, making full decomposition impossible without re-running.

---

## 8. Phase-Lag Reconciliation (Advisory-Only)

**How to use in reconciliation:** The `explain_phase_lag_vs_divergence()` function in `backend/health/semantic_tda_timeline.py` connects phase-lag index (from semantic-TDA correlation timeline) to divergence decomposition (from `runtime_profile_calibration.decompose_divergence_components`) to distinguish state lag from outcome noise. This function is **advisory-only** and does not trigger enforcement actions. Use it during calibration reconciliation to determine whether high phase lag indicates temporal misalignment in state predictions (STATE_LAG_DOMINANT) or outcome prediction noise (OUTCOME_NOISE_DOMINANT). The function returns an interpretation enum, thresholds used, basis fields indicating data sources, and neutral explanatory notes. See `docs/system_law/TDA_PhaseX_Binding.md` Section 8.10 for full specification.

---

## Recommended Single Metric of Truth

For UPGRADE-1 (per-component LR tuning), the primary objective is **state tracking fidelity**.

**Recommended Metric**: `mean_delta_p`

Rationale:
1. Measures average state error across H, rho, tau, beta
2. Not gated by binary threshold (unlike divergence_rate)
3. Directly responsive to LR changes
4. Available in canonical artifacts

**Secondary Metrics** (dashboard signals, not objectives):
- divergence_rate: Smoke signal for gross failures
- phase_lag_xcorr: Temporal alignment indicator

---

## Metric Classification: Legacy vs Primary

### Legacy Metric: divergence_rate

**Status**: LEGACY (retained for backward compatibility, NOT primary objective)

**Purpose**: Dashboard smoke signal for gross failures.

**Limitations**:
1. Binary threshold creates saturation when mean_delta_p oscillates near 0.05
2. Mixes outcome divergence (success/omega/blocked) with state divergence
3. Not directly responsive to LR tuning improvements
4. Can show no improvement even when state tracking materially improves

**Use Case**: Alarm when divergence_rate exceeds 0.9 for extended periods.

### Primary Metric: mean_delta_p

**Status**: PRIMARY (canonical objective for calibration)

**Purpose**: Measure state tracking fidelity.

**Advantages**:
1. Continuous metric (no threshold saturation)
2. Directly responsive to LR changes
3. Decomposable into per-component errors (H, rho, tau, beta)
4. Tracks actual estimator lag, not binary threshold crossings

**Use Case**: Primary objective function for all CAL-EXP calibration.

### Explicit Non-Equivalence Statement

**divergence_rate and mean_delta_p are NOT equivalent metrics.**

- divergence_rate can remain saturated at 1.0 while mean_delta_p improves
- mean_delta_p can worsen while divergence_rate stays constant
- Improvements in one do NOT imply improvements in the other

**Calibration decisions MUST be based on mean_delta_p, not divergence_rate.**

---

## Single Metric of Truth Rule

### Primary Optimization Metric (Calibration Era)

**Metric**: `mean_delta_p`

**Definition**: Mean absolute state tracking error across H, rho, tau, beta components.

**Use**: ALL calibration experiments (CAL-EXP-1, CAL-EXP-2, etc.) MUST optimize for this metric.

**Success Criterion**: Improvement is defined as `delta(mean_delta_p) < 0` across majority of seeds.

### Secondary Metrics (Dashboard-Only)

The following metrics are retained for observability but MUST NOT be used as optimization targets:

| Metric | Tag | Purpose |
|--------|-----|---------|
| divergence_rate | `DASHBOARD-ONLY` | Smoke signal for gross failures |
| success_divergence_rate | `DASHBOARD-ONLY` | Outcome prediction monitoring |
| omega_divergence_rate | `DASHBOARD-ONLY` | Safe region prediction monitoring |
| blocked_divergence_rate | `DASHBOARD-ONLY` | Governance prediction monitoring |
| state_divergence_rate | `DASHBOARD-ONLY` | Threshold-crossing indicator |
| phase_lag_xcorr | `DASHBOARD-ONLY` | Temporal alignment indicator |
| delta_bias | `DEPRECATED` | Heuristic - do not use |

### Per-Component Metrics (Diagnostic)

| Metric | Tag | Purpose |
|--------|-----|---------|
| mean_H_error | `DIAGNOSTIC` | H tracking analysis |
| mean_rho_error | `DIAGNOSTIC` | rho tracking analysis |
| mean_tau_error | `DIAGNOSTIC` | tau tracking analysis |
| mean_beta_error | `DIAGNOSTIC` | beta tracking analysis |

---

## Audit Checklist for Calibration Experiments

Every calibration experiment MUST report the following:

1. [ ] **Metric Version**: Reference to this document version (currently 1.0.0)
   - *Verify*: Check report header for `metric_definitions_version: 1.0.0`

2. [ ] **Window Size**: Number of cycles per analysis window (e.g., 50)
   - *Verify*: `jq '.window_size' cal_exp1_report.json`

3. [ ] **Total Cycles**: Total cycles observed (e.g., 200)
   - *Verify*: `jq '.total_cycles' cal_exp1_report.json`

4. [ ] **Seeds Used**: List of random seeds (e.g., {42, 43})
   - *Verify*: Check run directory names or `jq '.upgrade_config.seed' cal_exp1_report.json`

5. [ ] **LR Vector**: Full learning rate configuration `{H, rho, tau, beta}`
   - *Verify*: `jq '.upgrade_config.lr_overrides' cal_exp1_report.json`

6. [ ] **Baseline Reference**: Path to baseline run artifacts
   - *Verify*: Confirm `results/p5_cal_exp1/synthetic_seed*/` exists

7. [ ] **Primary Metric Delta**: `delta(mean_delta_p)` for each seed
   - *Verify*: `jq '.comparisons.seed42.deltas.mean_delta_p' upgrade1_reconciliation.json`

8. [ ] **Decomposition Table**: Per-component error deltas (H, rho, tau, beta)
   - *Verify*: `jq '.comparisons.seed42.deltas | {H: .mean_H_error, rho: .mean_rho_error, tau: .mean_tau_error, beta: .mean_beta_error}' upgrade1_reconciliation.json`

9. [ ] **Dominant Contributor**: Which component dominates divergence_rate
   - *Verify*: Run `python scripts/canonical_reconciliation.py` and check DOMINANT output

10. [ ] **Replay Verification**: PASS/FAIL for trace replay equivalence
    - *Verify*: `jq '.verdict' cal_exp1_replay_check.json`

**Non-compliance**: Experiments missing any checklist item are INVALID for decision-making.

---

## Metric Versioning Policy

### Core Principles

1. **Metrics are never deleted.** Once a metric is defined and used in a calibration experiment, it remains in this document permanently.

2. **New metrics are additive and versioned.** When introducing a new metric:
   - Assign a version number (e.g., mean_delta_p_v2)
   - Document the change rationale
   - Reference the experiment that motivated the change

3. **Deprecation is explicit.** Deprecated metrics are marked with `(DEPRECATED)` but never removed.

4. **Formula changes require new metric names.** Changing the formula of an existing metric requires creating a new versioned metric.

### Version History

| Version | Date | Change | Rationale |
|---------|------|--------|-----------|
| 1.0.0 | 2025-12-12 | Initial definitions | CAL-EXP-1 baseline |
| 1.0.0 | 2025-12-12 | delta_bias marked DEPRECATED | Heuristic conflates magnitude with direction |
| 1.0.0 | 2025-12-12 | mean_delta_p designated PRIMARY | Reconciliation identified it as true calibration objective |
| 1.1.0 | 2025-12-12 | Added CAL-EXP-2 baseline values | CAL-EXP-2 canonicalization |
| 1.1.0 | 2025-12-12 | Added UPGRADE-2 hypothesis targets | UPGRADE-2 design draft |
| 1.1.0 | 2025-12-12 | Document promoted to canonical authority | Scope Lock compliance |

### Adding New Metrics

To add a new metric:
1. Assign next available section number
2. Include: Definition, Formula, Source File, Status
3. Add to Availability table
4. Update Version History
5. Do NOT modify existing metric definitions

---

## Appendix A: Metric Names & JSON Paths

### Primary Metric

| Metric | JSON Key | Artifact File | JSON Path | Label |
|--------|----------|---------------|-----------|-------|
| mean_delta_p | `mean_delta_p` | cal_exp1_report.json | `$.windows[*].mean_delta_p` | PRIMARY |
| mean_delta_p | `mean_delta_p` | upgrade1_reconciliation.json | `$.runs.*.full_run.mean_delta_p` | PRIMARY |
| mean_delta_p | `mean_delta_p` | upgrade1_reconciliation.json | `$.runs.*.windows[*].mean_delta_p` | PRIMARY |

### Legacy Divergence Rate

| Metric | JSON Key | Artifact File | JSON Path | Label |
|--------|----------|---------------|-----------|-------|
| divergence_rate | `divergence_rate` | cal_exp1_report.json | `$.windows[*].divergence_rate` | RAW_ANY_MISMATCH |
| divergence_rate | `divergence_rate` | upgrade1_reconciliation.json | `$.runs.*.full_run.divergence_rate` | RAW_ANY_MISMATCH |

**Note**: `divergence_rate` in cal_exp1_report.json uses the harness is_diverged() predicate. The reconciliation script recomputes this from raw trace data.

### Decomposition Components

| Metric | JSON Key | Artifact File | JSON Path | Label |
|--------|----------|---------------|-----------|-------|
| success_divergence_rate | `success_divergence_rate` | upgrade1_reconciliation.json | `$.runs.*.full_run.success_divergence_rate` | DECOMPOSITION |
| omega_divergence_rate | `omega_divergence_rate` | upgrade1_reconciliation.json | `$.runs.*.full_run.omega_divergence_rate` | DECOMPOSITION |
| blocked_divergence_rate | `blocked_divergence_rate` | upgrade1_reconciliation.json | `$.runs.*.full_run.blocked_divergence_rate` | DECOMPOSITION |
| state_divergence_rate | `state_divergence_rate` | upgrade1_reconciliation.json | `$.runs.*.full_run.state_divergence_rate` | DECOMPOSITION |

### Per-Component Error Metrics

| Metric | JSON Key | Artifact File | JSON Path | Label |
|--------|----------|---------------|-----------|-------|
| mean_H_error | `mean_H_error` | upgrade1_reconciliation.json | `$.runs.*.full_run.mean_H_error` | DIAGNOSTIC |
| mean_rho_error | `mean_rho_error` | upgrade1_reconciliation.json | `$.runs.*.full_run.mean_rho_error` | DIAGNOSTIC |
| mean_tau_error | `mean_tau_error` | upgrade1_reconciliation.json | `$.runs.*.full_run.mean_tau_error` | DIAGNOSTIC |
| mean_beta_error | `mean_beta_error` | upgrade1_reconciliation.json | `$.runs.*.full_run.mean_beta_error` | DIAGNOSTIC |

### Deprecated Metrics

| Metric | JSON Key | Artifact File | JSON Path | Label |
|--------|----------|---------------|-----------|-------|
| delta_bias | `delta_bias` | cal_exp1_report.json | `$.windows[*].delta_bias` | DEPRECATED |

### Artifacts NOT in cal_exp1_report.json

The following metrics require recomputation via `scripts/canonical_reconciliation.py`:
- success_divergence_rate
- omega_divergence_rate
- blocked_divergence_rate
- state_divergence_rate
- mean_H_error, mean_rho_error, mean_tau_error, mean_beta_error

---

## Appendix B: Compatibility Contract

### Legacy Dashboard Support

**Requirement**: Old dashboards may still read `divergence_rate` from cal_exp1_report.json.

**Contract**:
1. The `divergence_rate` key MUST remain present in all cal_exp1_report.json artifacts
2. The field MUST continue to use the is_diverged() predicate (any mismatch)
3. New primary metrics (mean_delta_p) are ADDITIVE, not replacement
4. Dashboards reading `divergence_rate` MUST NOT break when new metrics are added

**Migration Path**:
- Phase 1 (current): Both metrics present, divergence_rate labeled RAW_ANY_MISMATCH
- Phase 2 (future): Dashboard updated to display mean_delta_p as primary
- Phase 3 (future): divergence_rate moved to "advanced" view

---

**END OF METRIC DEFINITIONS**

**Document Status**: FROZEN as of 2025-12-12. Additions permitted; modifications to existing definitions prohibited.
