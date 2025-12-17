# Replay P5 × True Divergence v1 Metric Contract

> **Status**: SHADOW MODE SPECIFICATION
> **Version**: 1.0.0
> **Date**: 2025-12-12

## Metric Contract

The `replay_p5` signal contributes to True Divergence v1 computation under
the following constraints:

1. **Safety-Weighted, Not Averaged**: Replay determinism rate is treated as a
   safety-weighted metric. A RED band (`determinism_rate < 0.70`) increases
   the True Divergence score but does NOT trigger solo hard blocking.

2. **Advisory-Only Contribution**: The `replay_p5` GGFL signal is always
   `advisory_only=true`. Even when `determinism_band="RED"`, this signal
   cannot escalate to a hard block without corroboration from other signals.

3. **Non-Gating Invariant**: Per SHADOW MODE contract, replay_p5 never
   influences control flow. It provides observational telemetry only.

4. **Deterministic Ordering**: All warning lists and reason arrays in the
   replay_p5 signal are sorted alphabetically for reproducible output.

## True Divergence Integration

```
true_divergence_contribution = {
    "signal": "replay_p5",
    "weight": "safety",           # not "average"
    "hard_block_eligible": false, # cannot solo-block
    "advisory_only": true,
    "corroboration_required": true
}
```

**SHADOW MODE: This contract is observational. No gating logic is implemented.**
