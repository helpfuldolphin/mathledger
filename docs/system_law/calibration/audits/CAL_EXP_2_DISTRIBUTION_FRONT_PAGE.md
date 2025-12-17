# CAL-EXP-2: Distribution Front Page

> **Status**: DISTRIBUTION COPY — LANGUAGE REVIEWED
> **Experiment**: P4 Divergence Minimization (Long-Window Convergence)
> **Mode**: SHADOW — Observational Only
> **Date**: 2025-12-13

---

## What CAL-EXP-2 Is

CAL-EXP-2 is a controlled calibration experiment that measures the effect of extended observation windows (1000 cycles) on Twin state estimator divergence under synthetic test conditions. The experiment observed a convergence floor at `mean_delta_p ≈ 0.025` after approximately 400 warm-up cycles, indicating that exponential averaging reaches a tracking limit with the current parameter configuration. All observations were recorded in SHADOW MODE with no influence on system behavior.

---

## What CAL-EXP-2 Is NOT

- **NOT a capability demonstration** — Results measure parameter tuning effects on a state estimator, not AI capability or intelligence advancement.

- **NOT a safety certification** — "Safe region" metrics refer to mathematical state-space boundaries (Ω-region), not behavioral safety guarantees or regulatory compliance.

- **NOT a deployment qualification** — Phase X artifacts are explicitly pre-production test data under synthetic conditions.

---

## Valid Claims (Three Only)

| Claim | Status | Evidence |
|-------|--------|----------|
| **Reduction observed** | VALID | `mean_delta_p` reduced from ~0.087 (warm-up) to ~0.025 (plateau) under test conditions |
| **Monotone convergence NOT observed** | VALID | Non-monotonic phases recorded (divergence spikes in phases 2-3 before recovery) |
| **No new pathology introduced** | VALID | Experiment completed without crashes, data corruption, or schema violations |

---

## Document Links

| Document | Purpose |
|----------|---------|
| [CAL_EXP_2_Canonical_Record.md](../CAL_EXP_2_Canonical_Record.md) | Canonical experiment results |
| [CAL_EXP_2_VALIDITY_ATTESTATION.md](../CAL_EXP_2_VALIDITY_ATTESTATION.md) | Validity attestation |
| [CAL_EXP_2_DEFINITIONS_BINDING.md](../CAL_EXP_2_DEFINITIONS_BINDING.md) | Metric definitions binding |
| [CAL_EXP_DO_NOT_CLAIM_APPENDIX.md](../CAL_EXP_DO_NOT_CLAIM_APPENDIX.md) | Misinterpretation shield |
| [CAL_EXP_2_DISTRIBUTION_CHECKLIST.md](./CAL_EXP_2_DISTRIBUTION_CHECKLIST.md) | Pre-distribution checklist |

---

## Distribution Footer

```
═══════════════════════════════════════════════════════════════════════════════
CALIBRATION EXPERIMENT NOTICE — NON-NORMATIVE GUIDANCE

This document reports SHADOW MODE observations from controlled synthetic testing.

• Results measure estimator tuning under test conditions only
• "Safe region" is a mathematical construct, not a safety guarantee
• "Divergence reduction" measures parameter adjustment effects, not capability
• All metrics are advisory only and do not influence system behavior
• Phase X artifacts are explicitly pre-production test data

This notice is non-normative guidance for distribution context.
See: docs/system_law/calibration/CAL_EXP_DO_NOT_CLAIM_APPENDIX.md
═══════════════════════════════════════════════════════════════════════════════
```

---

**END OF FRONT PAGE**
