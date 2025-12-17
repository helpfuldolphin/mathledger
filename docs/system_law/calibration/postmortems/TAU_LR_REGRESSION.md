# Post-Mortem: Tau LR Regression in UPGRADE-1

---

## Ratification Block

| Field | Value |
|-------|-------|
| **Status** | FROZEN |
| **Owner** | Calibration Governance |
| **Ratified-by** | ARCHITECT (COMPLETE) |
| **Version** | 1.0.0 |
| **Last Updated** | 2025-12-12 |
| **Change Policy** | ADDITIVE ONLY - existing analysis may not be modified |
| **Classification** | ROOT CAUSE ANALYSIS - NON-ENFORCING |

---

## Summary

UPGRADE-1 reduced tau learning rate from 0.05 to 0.02 (-60%), causing tau tracking error to increase by +0.012 to +0.013 across both seeds. This regression offset the gains achieved in H and rho tracking, resulting in a mixed verdict for the overall calibration.

---

## Timeline

| Event | Description |
|-------|-------------|
| CAL-EXP-1 BASELINE | Established baseline with uniform LR=0.1 (tau effective LR=0.05) |
| UPGRADE-1 Design | Per-component LRs: H=0.20, rho=0.15, tau=0.02, beta=0.12 |
| UPGRADE-1 Execution | Runs completed for seeds 42 and 43 |
| Reconciliation | Decomposition revealed tau as dominant regression vector |

---

## Root Cause Analysis

### Why tau is more sensitive than H/rho

1. **Smaller dynamic range**: tau operates in a narrower band (typically 0.15-0.25) compared to H (0.3-0.7) and rho (0.5-0.8). Small absolute errors in tau translate to larger relative errors.

2. **Threshold role**: tau is the decision threshold for omega membership (`H > tau`). Tracking errors in tau directly affect omega prediction accuracy.

3. **Lower baseline LR**: The baseline already used `tau LR = base_LR * 0.5 = 0.05`. This was already conservative. Reducing it further to 0.02 introduced excessive lag.

### Why lowering LR degraded tracking

The TwinRunner state update for tau is:
```python
twin_tau = twin_tau * (1 - lr_tau) + real.tau * lr_tau
twin_tau += rng.gauss(0, noise * 0.5)
```

With `lr_tau = 0.02`:
- Twin tau moves only 2% toward real tau per cycle
- Noise term (gaussian with std=0.01) becomes comparable to learning signal
- Time to converge from initial tau (0.20) to target tau increases 2.5x

**Mathematical analysis**:
- Half-life of convergence: `t_half = ln(2) / lr_tau`
- Baseline (lr=0.05): t_half = 13.9 cycles
- UPGRADE-1 (lr=0.02): t_half = 34.7 cycles

The slower convergence meant tau was still tracking toward baseline values when windows closed, inflating mean_tau_error.

### Evidence from decomposition

| Component | Baseline LR | UPGRADE-1 LR | mean_error Delta (seed42) | mean_error Delta (seed43) |
|-----------|-------------|--------------|--------------------------|--------------------------|
| H | 0.10 | 0.20 (+100%) | -0.0045 (IMPROVED) | -0.0095 (IMPROVED) |
| rho | 0.10 | 0.15 (+50%) | -0.0063 (IMPROVED) | -0.0071 (IMPROVED) |
| tau | 0.05 | 0.02 (-60%) | **+0.0132 (REGRESSED)** | **+0.0116 (REGRESSED)** |
| beta | 0.10 | 0.12 (+20%) | +0.0027 (slight worse) | +0.0038 (slight worse) |

Tau regression (+0.012) exceeded combined H+rho improvement (-0.010 to -0.017), dragging overall mean_delta_p.

---

## Do-Not-Repeat Rules

### Rule 1: Never reduce LR for slow-moving components below baseline

**Statement**: For components with `effective_LR < base_LR` in baseline, do not reduce further without explicit justification.

**Rationale**: Slow-moving components are already at risk of tracking lag. Further reduction amplifies this risk.

**Application**: tau had baseline LR = 0.05 (half of base). UPGRADE-1 should have kept tau >= 0.05.

### Rule 2: Test LR changes directionally before combining

**Statement**: When tuning multiple LRs simultaneously, first verify each change improves its target component in isolation.

**Rationale**: Combined changes obscure which component is regressing.

**Application**: Should have tested tau=0.02 alone to observe degradation before combining with H/rho increases.

### Rule 3: Preserve gains by not changing what works

**Statement**: If a component is not the dominant error source, do not reduce its LR.

**Rationale**: Reducing LR for non-problematic components introduces new regression vectors.

**Application**: tau was not the dominant error in BASELINE (beta was). Reducing tau LR was unnecessary optimization.

### Rule 4: Compare half-life to window size

**Statement**: Ensure convergence half-life is less than 1/4 of window size.

**Rationale**: If half-life > window_size/4, the component will not converge within the measurement window.

**Application**:
- Window size: 50 cycles
- Required half-life: < 12.5 cycles
- UPGRADE-1 tau half-life: 34.7 cycles (VIOLATION)

---

## Corrective Action

### Recommended UPGRADE-2 Configuration

```python
LR = {
    "H": 0.20,    # Keep (improved H tracking)
    "rho": 0.15,  # Keep (improved rho tracking)
    "tau": 0.05,  # RESTORE to baseline (fixing regression)
    "beta": 0.10, # Restore to baseline (marginal effect)
}
```

This preserves the H/rho gains while eliminating the tau regression.

### Alternative: Increase tau LR

If tau tracking is deemed important for future experiments:
```python
"tau": 0.08  # Increase to reduce half-life to ~8.7 cycles
```

This would make tau more responsive, but should be tested in isolation first.

---

## Institutional Memory

This post-mortem documents why UPGRADE-1 produced mixed results despite correctly identifying H and rho as improvement targets. The tau regression was an unforced error caused by reducing an already-conservative LR.

**Key Lesson**: Calibration is not just about increasing LRs for slow components. It also requires not decreasing LRs for components that are already adequately tuned.

---

**Document Status**: FROZEN. This post-mortem is part of the calibration audit trail.

**References**:
- `results/p5_reconciliation/upgrade1_reconciliation.json`
- `results/p5_reconciliation/upgrade1_reconciliation.md`
- `docs/system_law/calibration/METRIC_DEFINITIONS.md`
