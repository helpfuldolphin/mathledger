# CAL-EXP-1 UPGRADE-1 Comparison Report

**Date**: 2025-12-12
**Status**: PROVISIONAL + SHADOW_ONLY
**Schema Version**: 1.0.0

---

## Upgrade Configuration

**UPGRADE-1**: Per-Component Learning Rates

| Component | Baseline LR | UPGRADE-1 LR | Change |
|-----------|-------------|--------------|--------|
| H         | 0.10        | 0.20         | +100%  |
| rho       | 0.10        | 0.15         | +50%   |
| tau       | 0.05        | 0.02         | -60%   |
| beta      | 0.10        | 0.12         | +20%   |

---

## Side-by-Side Results: Seed=42

### Window Metrics Comparison

| Window | Metric | BASELINE | UPGRADE-1 | Delta | Improved? |
|--------|--------|----------|-----------|-------|-----------|
| 0 | divergence_rate | 0.94 | 0.96 | +0.02 | NO |
| 0 | mean_delta_p | 0.0451 | 0.0462 | +0.0011 | NO |
| 0 | delta_bias | -0.0199 | -0.0213 | -0.0014 | NO |
| 0 | phase_lag_xcorr | 0.792 | 0.819 | +0.027 | NO |
| 1 | divergence_rate | 1.00 | 1.00 | 0.00 | SAME |
| 1 | mean_delta_p | 0.0517 | 0.0560 | +0.0043 | NO |
| 1 | delta_bias | -0.0259 | -0.0280 | -0.0021 | NO |
| 1 | phase_lag_xcorr | 0.622 | 0.564 | **-0.058** | **YES** |
| 2 | divergence_rate | 1.00 | 1.00 | 0.00 | SAME |
| 2 | mean_delta_p | 0.0474 | 0.0463 | **-0.0011** | **YES** |
| 2 | delta_bias | -0.0237 | -0.0232 | **+0.0005** | **YES** |
| 2 | phase_lag_xcorr | 0.433 | 0.551 | +0.118 | NO |
| 3 | divergence_rate | 1.00 | 1.00 | 0.00 | SAME |
| 3 | mean_delta_p | 0.0517 | 0.0524 | +0.0007 | NO |
| 3 | delta_bias | -0.0258 | -0.0262 | -0.0004 | NO |
| 3 | phase_lag_xcorr | 0.780 | 0.780 | 0.000 | SAME |

### Summary (Seed=42)

| Metric | BASELINE | UPGRADE-1 | Delta |
|--------|----------|-----------|-------|
| Mean divergence | 0.985 | 0.990 | +0.005 |
| Final delta_bias | -0.0258 | -0.0262 | -0.0004 |
| Mean phase_lag | 0.657 | 0.679 | +0.022 |
| Pattern progression | NONE->OSC->OSC->NONE | NONE->OSC->OSC->NONE | SAME |

---

## Side-by-Side Results: Seed=43

### Window Metrics Comparison

| Window | Metric | BASELINE | UPGRADE-1 | Delta | Improved? |
|--------|--------|----------|-----------|-------|-----------|
| 0 | divergence_rate | 0.98 | 0.96 | **-0.02** | **YES** |
| 0 | mean_delta_p | 0.0450 | 0.0436 | **-0.0014** | **YES** |
| 0 | delta_bias | -0.0216 | -0.0200 | **+0.0016** | **YES** |
| 0 | phase_lag_xcorr | 0.859 | 0.830 | **-0.029** | **YES** |
| 1 | divergence_rate | 1.00 | 1.00 | 0.00 | SAME |
| 1 | mean_delta_p | 0.0575 | 0.0571 | **-0.0004** | **YES** |
| 1 | delta_bias | -0.0287 | -0.0285 | **+0.0002** | **YES** |
| 1 | phase_lag_xcorr | 0.844 | 0.794 | **-0.050** | **YES** |
| 2 | divergence_rate | 1.00 | 1.00 | 0.00 | SAME |
| 2 | mean_delta_p | 0.0616 | 0.0590 | **-0.0026** | **YES** |
| 2 | delta_bias | -0.0308 | -0.0295 | **+0.0013** | **YES** |
| 2 | phase_lag_xcorr | 0.726 | 0.728 | +0.002 | ~SAME |
| 3 | divergence_rate | 1.00 | 1.00 | 0.00 | SAME |
| 3 | mean_delta_p | 0.0624 | 0.0657 | +0.0033 | NO |
| 3 | delta_bias | -0.0312 | -0.0328 | -0.0016 | NO |
| 3 | phase_lag_xcorr | 0.695 | 0.663 | **-0.032** | **YES** |

### Summary (Seed=43)

| Metric | BASELINE | UPGRADE-1 | Delta |
|--------|----------|-----------|-------|
| Mean divergence | 0.995 | 0.990 | **-0.005** |
| Final delta_bias | -0.0312 | -0.0328 | -0.0016 |
| Mean phase_lag | 0.781 | 0.754 | **-0.027** |
| Pattern progression | OSC->NONE->OSC->NONE | NONE->NONE->NONE->NONE | IMPROVED |

---

## Aggregate Analysis

### Delta Summary (UPGRADE-1 vs BASELINE)

| Metric | Seed=42 | Seed=43 | Mean Delta |
|--------|---------|---------|------------|
| Delta(mean divergence) | +0.005 | -0.005 | 0.000 |
| Delta(final_bias) | -0.0004 | -0.0016 | -0.0010 |
| Delta(mean_phase_lag) | +0.022 | -0.027 | -0.0025 |
| Delta(mean_delta_p) | +0.0013 | -0.0003 | +0.0005 |

### Convergence Trend Classification

| Seed | BASELINE Trend | UPGRADE-1 Trend | Assessment |
|------|----------------|-----------------|------------|
| 42 | NONE | NONE | No change |
| 43 | NONE | NONE | No change |

**Overall Trend**: **NONE** (No clear convergence improvement)

---

## Key Question

> "Did UPGRADE-1 reduce estimator lag without introducing instability?"

### Answer

**MIXED RESULTS**

1. **Phase Lag**:
   - Seed=42: Slight increase (+0.022 mean)
   - Seed=43: Modest decrease (-0.027 mean)
   - **Net effect**: ~0.0025 reduction (marginal)

2. **Bias**:
   - Both seeds show slightly worse (more negative) final bias
   - Twin still systematically underestimates

3. **Divergence Rate**:
   - Unchanged at ceiling (1.0)
   - UPGRADE-1 did not reduce divergence

4. **Stability**:
   - No CRITICAL events
   - No instability introduced
   - Pattern progression improved for seed=43 (less oscillation)

5. **Mean delta_p**:
   - Seed=42: Slightly worse (+0.0013)
   - Seed=43: Mixed (early improvement, late degradation)

---

## Verdict

**INSUFFICIENT IMPROVEMENT**

UPGRADE-1 did not achieve the primary goal of reducing divergence rate below 1.0.

- Divergence rate remains at ceiling
- Bias slightly worsened
- Phase lag showed mixed results (one improved, one worsened)
- No convergence trend established

---

## Recommendation

**Do NOT authorize CAL-EXP-2 with current parameters.**

Consider:

1. **Investigate divergence threshold**: The 0.05 threshold for `is_diverged()` may be too strict. Current mean_delta_p (0.045-0.065) consistently exceeds this.

2. **Higher H learning rate**: Try H=0.30 or H=0.35 to accelerate state tracking.

3. **Adjust rho LR direction**: Current rho increase didn't consistently help. Try rho=0.20 for faster stability tracking.

4. **Alternative**: Reconsider the divergence metric itself - is delta_p the right comparison target?

---

## Artifacts

| Run | Path |
|-----|------|
| BASELINE seed=42 | `results/p5_cal_exp1/synthetic_seed42/` |
| BASELINE seed=43 | `results/p5_cal_exp1/synthetic_seed43/` |
| UPGRADE-1 seed=42 | `results/p5_cal_exp1/upgrade1_seed42/` |
| UPGRADE-1 seed=43 | `results/p5_cal_exp1/upgrade1_seed43/` |

---

**Note**: All results are PROVISIONAL + SHADOW_ONLY. No governance modifications occurred.
