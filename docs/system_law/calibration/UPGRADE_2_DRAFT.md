# UPGRADE-2: Predictive Twin Model

**Status**: PROVISIONAL / DESIGN DRAFT
**Date**: 2025-12-12
**Requires**: CAL-EXP-3 validation before implementation

---

## ⚠️ DRAFT NOTICE

This document is a **design draft**. It has NOT been validated experimentally.

- Do NOT implement without STRATCOM authorization
- Do NOT treat design parameters as canonical
- Validation requires CAL-EXP-3

---

## Scope Lock

**This section prevents hidden acceptance semantics.**

| Constraint | Binding |
|------------|---------|
| Document status | PROVISIONAL / DESIGN DRAFT |
| Acceptance thresholds | **NONE INTRODUCED** — This document defines no acceptance gates |
| Numeric targets | **HYPOTHESIS TARGETS ONLY** — Not gates, not thresholds |
| Metric semantics | **MUST REFERENCE** `docs/system_law/calibration/METRIC_DEFINITIONS.md` |
| Implementation authority | **NOT GRANTED** — Requires explicit STRATCOM authorization |

### What This Document IS

- A design hypothesis for Twin model improvement
- A specification of testable predictions (H1-H4)
- A reference for CAL-EXP-3 protocol design

### What This Document IS NOT

- An acceptance specification
- A production-ready implementation plan
- A ratified metric definition source

---

## Problem Statement

CAL-EXP-2 established that exponential averaging reaches a **convergence floor** at δp ≈ 0.025.

### Root Cause Analysis

The current Twin model uses **reactive tracking**:

```
twin_state(t+1) = (1 - lr) * twin_state(t) + lr * real_state(t)
```

This architecture has inherent limitations:

1. **No predictive capability** — Twin only reacts to observed state, cannot anticipate transitions
2. **Fixed latency** — Response time is proportional to 1/LR, creating irreducible lag
3. **Overshoot during transitions** — Large state changes cause the Twin to "chase" the real state
4. **No velocity modeling** — Twin ignores rate-of-change information

### What Predictive Structure is Missing?

The Twin lacks:

| Missing Component | Effect |
|-------------------|--------|
| **Velocity term** | Cannot predict direction of change |
| **Acceleration term** | Cannot anticipate transition speed |
| **Regime detection** | Cannot distinguish stable vs transitional states |
| **Asymmetric response** | Cannot respond faster to large changes |

---

## UPGRADE-2 Design

### Core Concept: Velocity-Augmented Tracking

Replace simple exponential averaging with a **velocity-aware predictor**:

```
velocity(t) = real_state(t) - real_state(t-1)
predicted_state(t+1) = real_state(t) + velocity(t) * momentum

twin_state(t+1) = (1 - lr) * twin_state(t) + lr * predicted_state(t+1)
```

### State Terms to Model Explicitly

#### 1. Per-Component Velocity (∂H/∂t, ∂ρ/∂t)

Track rate of change for each state component:

```python
self._velocity_H = real_H - self._prev_real_H
self._velocity_rho = real_rho - self._prev_real_rho
```

#### 2. Momentum Coefficient

Apply momentum to velocity predictions:

```python
momentum_H = 0.3  # Tunable: how much to project forward
momentum_rho = 0.2
```

#### 3. Regime Indicator

Detect transition vs stable regimes:

```python
is_transition = abs(velocity_H) > threshold_H or abs(velocity_rho) > threshold_rho
```

#### 4. Adaptive LR Based on Regime

Increase LR during transitions, decrease during stability:

```python
effective_lr_H = lr_H * (1.5 if is_transition else 0.8)
```

### Proposed Implementation

```python
class PredictiveTwinRunner:
    def update_state(self, real_observation: RealCycleObservation) -> None:
        # Compute velocities
        velocity_H = real_observation.H - self._prev_real_H
        velocity_rho = real_observation.rho - self._prev_real_rho

        # Detect regime
        is_transition = (
            abs(velocity_H) > self._transition_threshold_H or
            abs(velocity_rho) > self._transition_threshold_rho
        )

        # Compute predictions with momentum
        predicted_H = real_observation.H + velocity_H * self._momentum_H
        predicted_rho = real_observation.rho + velocity_rho * self._momentum_rho

        # Adaptive LR
        lr_H = self._lr_H * (self._transition_boost if is_transition else self._stable_damp)
        lr_rho = self._lr_rho * (self._transition_boost if is_transition else self._stable_damp)

        # Update twin state toward prediction
        self._H = self._H * (1 - lr_H) + predicted_H * lr_H
        self._rho = self._rho * (1 - lr_rho) + predicted_rho * lr_rho

        # Store for next cycle
        self._prev_real_H = real_observation.H
        self._prev_real_rho = real_observation.rho
```

---

## Hypothesis Targets

> **NOT GUARANTEED** — These are design targets, not validated results. Actual effects will be measured in CAL-EXP-3.

### 1. τ Tracking (No Change Expected)

| Metric | Current (UPGRADE-1) | Target (UPGRADE-2) |
|--------|---------------------|----------------------|
| τ LR | 0.02 (heavily damped) | 0.02 (unchanged) |
| τ tracking error | Low | Low (unchanged) |

τ is already well-tracked because it changes slowly. UPGRADE-2 primarily targets H and ρ.

### 2. Convergence Floor (Primary Target)

| Metric | Current | Target |
|--------|---------|--------|
| Convergence floor | δp ≈ 0.025 | δp ≤ 0.0175 (≥30% reduction) |
| Warm-up cycles | 400 | ≤250 (≥40% reduction) |
| Overshoot magnitude | High | Moderate |

**Design rationale**: Velocity prediction should reduce the "chasing" behavior during transitions.

### 3. Identified Risks

| Risk | Proposed Mitigation |
|------|---------------------|
| Overshoot from momentum | Cap momentum at 0.5, dampen during stability |
| Oscillation | Regime detection gates momentum application |
| Increased complexity | Additional parameters require tuning |

---

## Implementation Interface (REAL-READY)

> Signatures only — no implementation code. For design reference.

```
┌─────────────────────────────────────────────────────────────────┐
│ REALITY LOCK                                                    │
├─────────────────────────────────────────────────────────────────┤
│ REAL-READY: Function signatures/types that match repo structure │
│ SPEC-ONLY:  Pseudocode or future work (not yet in repo)         │
└─────────────────────────────────────────────────────────────────┘
```

### New State Variables Required

**REALITY LOCK: SPEC-ONLY** — These variables do not exist in current TwinRunner.

```python
# In TwinRunner.__init__:
self._prev_real_H: float = 0.5      # Previous real H observation
self._prev_real_rho: float = 0.7    # Previous real rho observation
self._velocity_H: float = 0.0       # Computed H velocity
self._velocity_rho: float = 0.0     # Computed rho velocity
self._is_transition: bool = False   # Regime indicator
```

### Config Surface (CLI Flags / Parameters)

**REALITY LOCK: SPEC-ONLY** — These flags do not exist in current harness.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use-predictive-twin` | bool | False | Enable UPGRADE-2 |
| `--momentum-H` | float | 0.3 | H velocity momentum coefficient |
| `--momentum-rho` | float | 0.2 | ρ velocity momentum coefficient |
| `--transition-threshold-H` | float | 0.02 | H transition detection threshold |
| `--transition-threshold-rho` | float | 0.015 | ρ transition detection threshold |
| `--transition-boost` | float | 1.5 | LR multiplier during transitions |
| `--stable-damp` | float | 0.8 | LR multiplier during stability |

### Integration Point in TwinRunner

**REALITY LOCK: REAL-READY** — References existing method `TwinRunner.update_state()` in `backend/topology/first_light/runner_p4.py`.

```python
# In TwinRunner.update_state(), replace:
#   self._H = self._H * (1 - lr_H) + real_observation.H * lr_H
# With:
#   if self._use_predictive:
#       self._update_state_predictive(real_observation)
#   else:
#       self._update_state_reactive(real_observation)  # Current behavior
```

### Method Signatures

**REALITY LOCK: SPEC-ONLY** — These methods do not exist yet.

```python
def _update_state_predictive(self, real_observation: RealCycleObservation) -> None:
    """UPGRADE-2: Velocity-augmented state update."""
    ...

def _compute_velocity(self, current: float, previous: float) -> float:
    """Compute single-step velocity."""
    ...

def _detect_regime(self) -> bool:
    """Detect transition vs stable regime. Returns True if transitioning."""
    ...

def _compute_effective_lr(self, base_lr: float, is_transition: bool) -> float:
    """Compute regime-adaptive learning rate."""
    ...
```

---

## How CAL-EXP-3 Will Validate H1-H4

> **PROVISIONAL** — Exact thresholds subject to refinement.

### Validation Protocol

| Hypothesis | Metric | Threshold | Measurement |
|------------|--------|-----------|-------------|
| **H1** | Final-phase mean δp | < 0.0175 | Mean δp in cycles 801-1000 |
| **H2** | Cycles to first CONVERGING phase | < 250 | Phase classification transition point |
| **H3** | δp in high-velocity windows | < 60% of UPGRADE-1 baseline | Windows where \|∂H/∂t\| > 0.01 |
| **H4** | Phase-5 variance | ≤ CAL-EXP-2 phase-5 variance | Variance of δp in cycles 801-1000 |

### Pass/Fail Criteria

| Result | Criteria |
|--------|----------|
| **PASS** | H1 + H4 pass, AND (H2 OR H3) pass |
| **PARTIAL** | H1 OR H4 pass, others fail |
| **FAIL** | H1 AND H4 both fail |

### CAL-EXP-3 Configuration

```json
{
  "experiment": "CAL-EXP-3",
  "cycles": 1000,
  "seed": 42,
  "telemetry_adapter": "real",
  "upgrade_2_enabled": true,
  "parameters": "PROVISIONAL (see Provisional Parameter Recommendations)"
}
```

---

## Testable Hypotheses for CAL-EXP-3

### H1: Convergence Floor Reduction

**Hypothesis**: UPGRADE-2 reduces convergence floor by ≥30% (δp < 0.0175).

**Test**: Run CAL-EXP-3 with 1000 cycles, compare final-phase mean δp.

### H2: Warm-Up Time Reduction

**Hypothesis**: UPGRADE-2 reduces warm-up divergence duration by ≥40% (< 250 cycles to recovery).

**Test**: Identify phase transition points, measure cycles to first CONVERGING phase.

### H3: Transition Tracking Improvement

**Hypothesis**: UPGRADE-2 reduces δp during state transitions by ≥40%.

**Test**: Isolate high-velocity windows (|∂H/∂t| > 0.01), compare δp vs UPGRADE-1.

### H4: No Stability Regression

**Hypothesis**: UPGRADE-2 does not increase variance in stable phases.

**Test**: Compare phase-5 variance between CAL-EXP-2 and CAL-EXP-3.

---

## Provisional Parameter Recommendations

| Parameter | Provisional Value | Rationale |
|-----------|-------------------|-----------|
| momentum_H | 0.3 | Moderate forward projection |
| momentum_rho | 0.2 | Slightly damped (ρ changes slower) |
| transition_threshold_H | 0.02 | ~2x typical cycle-to-cycle change |
| transition_threshold_rho | 0.015 | ~2x typical cycle-to-cycle change |
| transition_boost | 1.5 | 50% LR increase during transitions |
| stable_damp | 0.8 | 20% LR reduction during stability |

**These values are PROVISIONAL and require CAL-EXP-3 validation.**

---

## Implementation Checklist (Pending Authorization)

- [ ] Add velocity tracking to TwinRunner
- [ ] Add momentum prediction
- [ ] Add regime detection
- [ ] Add adaptive LR modulation
- [ ] Add CLI flags for UPGRADE-2 parameters
- [ ] Design CAL-EXP-3 protocol
- [ ] Execute CAL-EXP-3
- [ ] Validate hypotheses H1-H4

---

## CAL-EXP-3 Protocol Stub

> **PROVISIONAL** — Protocol subject to refinement before execution.

### Required Artifacts

| Artifact | Path | Format |
|----------|------|--------|
| Run Config | `results/cal_exp_3/<run_id>/run_config.json` | JSON |
| Real Cycles | `results/cal_exp_3/<run_id>/real_cycles.jsonl` | JSONL |
| Twin Predictions | `results/cal_exp_3/<run_id>/twin_predictions.jsonl` | JSONL |
| Divergence Log | `results/cal_exp_3/<run_id>/divergence_log.jsonl` | JSONL |
| Summary | `results/cal_exp_3/<run_id>/p4_summary.json` | JSON |

### Required Fields for Hypothesis Evaluation

**In `real_cycles.jsonl` (per line):**
```json
{
  "cycle": <int>,
  "usla_state": {
    "H": <float>,
    "rho": <float>
  }
}
```

**In `twin_predictions.jsonl` (per line):**
```json
{
  "real_cycle": <int>,
  "twin_state": {
    "H": <float>,
    "rho": <float>
  }
}
```

### Pass/Fail Evaluation Rules

All rules expressed in terms of `mean_delta_p` and decomposition components. **No new global "divergence scalar" introduced.**

#### H1: Convergence Floor Reduction

```
METRIC: mean_delta_p_phase5 = mean(|real.H - twin.H| for cycles 801-1000)
BASELINE: CAL-EXP-2 phase5 mean_delta_p = 0.0254
THRESHOLD: mean_delta_p_phase5 < 0.0175

PASS: mean_delta_p_phase5 < 0.0175
FAIL: mean_delta_p_phase5 >= 0.0175
```

#### H2: Warm-Up Time Reduction

```
METRIC: first_converging_cycle = min(cycle where phase_status == "CONVERGING")
BASELINE: CAL-EXP-2 first_converging_cycle ~ 600
THRESHOLD: first_converging_cycle < 250

PASS: first_converging_cycle < 250
FAIL: first_converging_cycle >= 250
```

#### H3: Transition Tracking Improvement

```
METRIC: mean_delta_p_transitions = mean(|real.H - twin.H| for windows where |dH/dt| > 0.01)
BASELINE: CAL-EXP-2 transition mean_delta_p (compute from baseline data)
THRESHOLD: mean_delta_p_transitions < 0.6 * baseline_transition_delta_p

PASS: mean_delta_p_transitions < 0.6 * baseline
FAIL: mean_delta_p_transitions >= 0.6 * baseline
```

#### H4: No Stability Regression

```
METRIC: variance_phase5 = variance(|real.H - twin.H| for cycles 801-1000)
BASELINE: CAL-EXP-2 phase5 variance = 0.000031
THRESHOLD: variance_phase5 <= baseline_variance

PASS: variance_phase5 <= 0.000031
FAIL: variance_phase5 > 0.000031
```

### Overall Verdict Rules

```
PASS:    H1.PASS AND H4.PASS AND (H2.PASS OR H3.PASS)
PARTIAL: (H1.PASS OR H4.PASS) AND NOT (H1.PASS AND H4.PASS)
FAIL:    H1.FAIL AND H4.FAIL
```

---

## Authorization Required

This design is **NOT AUTHORIZED** for implementation.

**Required before implementation**:
1. STRATCOM review of design
2. CAL-EXP-3 protocol approval
3. Explicit implementation authorization

---

*SHADOW MODE: This is a design document. No code changes have been made.*
