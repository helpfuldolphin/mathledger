# CAL-EXP-1: Baseline Calibration Record

**Status**: FROZEN
**Date**: 2025-12-12
**Schema Version**: 1.0.0

---

## Inputs

| Parameter | Value |
|-----------|-------|
| Adapter | RealTelemetryAdapter (SYNTHETIC mode) |
| Cycles | 200 |
| Window Size | 50 |
| Seeds | {42, 43} |
| Learning Rate | 0.1 (uniform) |
| Noise Scale | 0.02 |
| LR Overrides | None |

---

## Observations

### Divergence Behavior
- **Final divergence_rate**: 1.0 (both seeds)
- **Mean divergence_rate**: 0.985 (seed=42), 0.995 (seed=43)
- **Trend**: Stable at ceiling by window 1

### Bias Characteristics
- **Sign**: Consistently NEGATIVE
- **Magnitude**: -0.02 to -0.03
- **Interpretation**: Twin systematically underestimates H relative to Real

### Variance Stability
- **Order of magnitude**: 10^-4 to 10^-5
- **Status**: Stable (low variance indicates consistent but offset predictions)

### Phase Lag
- **Range**: 0.43 to 0.86 (xcorr)
- **Status**: Persistent across all windows
- **Interpretation**: Twin tracks Real with temporal delay

### Pattern Tags
- **Observed**: NONE, OSCILLATION
- **Not observed**: DRIFT, SPIKE
- **Status**: Heuristic/provisional (not validated)

---

## Verdict

**ADJUST**

The Twin is tracking Real state (low variance) but not converging (divergence_rate = 1.0). The systematic negative bias and persistent phase lag indicate the uniform learning rate is insufficient for timely state tracking.

---

## Authorized Next Step

**UPGRADE-1**: Per-Component Learning Rates

```python
LR = {
    "H": 0.20,
    "rho": 0.15,
    "tau": 0.02,
    "beta": 0.12,
}
```

Rationale:
- Higher H LR to accelerate estimator convergence
- Higher rho LR to reduce phase lag
- Lower tau LR to maintain threshold stability
- Moderate beta LR for blocking state tracking

---

## Artifacts

| File | Purpose |
|------|---------|
| `results/p5_cal_exp1/cal_exp1_dynamics_summary.json` | Machine-readable baseline |
| `results/p5_cal_exp1/CAL_EXP1_SCIENTIST_REPORT.md` | Human-readable analysis |
| `results/p5_cal_exp1/synthetic_seed42/` | Seed=42 run artifacts |
| `results/p5_cal_exp1/synthetic_seed43/` | Seed=43 run artifacts |

---

## Audit Trail

This record is part of the calibration ledger. Referenced by:
- CAL-EXP-1-RERUN (UPGRADE-1 comparison)
- CAL-EXP-2 (if UPGRADE-1 succeeds)

**DO NOT MODIFY** after freeze date.
