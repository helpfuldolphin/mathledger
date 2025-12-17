# CAL-EXP-3 Reporting Language Constraints

**Single source of truth (code):** `backend/governance/language_constraints.py`

**Claim ladder definition:** `docs/system_law/calibration/CAL_EXP_3_UPLIFT_SPEC.md`

If a phrase sounds impressive, it's probably illegal.

---

## Claim Ladder Mapping

No claim at level N is valid without satisfying all levels < N.

| Level | Claim Type | Allowed Template | Prohibited Escalation |
|-------|------------|------------------|----------------------|
| **L0** | Experiment completed | "Both arms ran to completion." | "Experiment succeeded" |
| **L1** | Measurements obtained | "Δp values computed for both arms." | "Metrics validated" |
| **L2** | ΔΔp computed | "ΔΔp = X (treatment − baseline)." | "Improvement measured" |
| **L3** | ΔΔp exceeds noise | "ΔΔp exceeds noise floor (X > Y)." | "Significant uplift" |
| **L4** | Uplift measured | "Measured uplift of X under CAL-EXP-3 conditions." | "Learning works" |
| **L5** | Uplift replicated | "Uplift replicated across N runs." | "System improved" |

### Mechanism Claims — Always Forbidden

| Forbidden Pattern | Why | Alternative |
|-------------------|-----|-------------|
| "Learning works" | Causal mechanism not measured | "Measured positive ΔΔp with learning enabled" |
| "System improved" | Implies absolute progress | "Treatment arm outperformed baseline in window W" |
| "Intelligence increased" | Term not operationalized | (no alternative—do not claim) |

---

## Forbidden Phrases

| Phrase | Reason |
|--------|--------|
| "improved intelligence" | Anthropomorphizes capability |
| "validated learning" | Implies correctness of adaptation |
| "generalization" | Implies transfer beyond training domain |
| "learned behavior" | Implies autonomous adaptation |
| "intelligence gain" | Overstates capability |
| "cognitive improvement" | Anthropomorphizes |
| "calibration passed" | Implies gate/approval (inherited) |
| "ready for production" | Beyond calibration scope (inherited) |
| "governance approved" | No approval authority (inherited) |

---

## Allowed Phrases

| Phrase | Why Allowed |
|--------|-------------|
| "measured uplift" | Neutral, measurement-focused |
| "observed delta" | Factual observation |
| "metric change" | Descriptive, not evaluative |
| "performance delta" | Quantitative comparison |
| "ΔΔp computed" | Technical, no value judgment |
| "treatment arm" | Neutral experimental terminology |
| "baseline arm" | Neutral experimental terminology |

---

## Approved Templates

| Use Case | Template |
|----------|----------|
| L0 Completion | "Both arms ran to completion under CAL-EXP-3 conditions." |
| L1 Measurement | "Δp values computed: baseline={X}, treatment={Y}." |
| L2 Delta | "ΔΔp = {X} (treatment − baseline)." |
| L3 Noise Floor | "ΔΔp of {X} exceeds noise floor of {Y}." |
| L4 Uplift | "Measured uplift of {X} under CAL-EXP-3 conditions." |
| L5 Replication | "Uplift of {X} replicated across {N} independent runs." |

---

## Required Qualifier

All CAL-EXP-3 result statements must end with:

> **SHADOW MODE — observational only.**
