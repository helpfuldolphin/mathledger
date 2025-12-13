# CAL-EXP-2 Reporting Language Constraints

**Single source of truth (code):** `backend/governance/language_constraints.py`

## Forbidden Phrases

| Phrase | Reason |
|--------|--------|
| "divergence eliminated" | Implies perfection |
| "twin validated" | Implies correctness |
| "calibration passed" | Implies gate/approval |
| "model converged" | Implies finality |
| "accuracy improved" | Implies ground truth |
| "system aligned" | Overloaded term |
| "ready for production" | Beyond calibration scope |
| "governance approved" | No approval authority |
| "monotone improvement achieved" | Invalid per §2 definitions |

## Approved Templates

| Use Case | Template |
|----------|----------|
| Numeric | "Divergence rate measured at X (prior: Y) under CAL-EXP-2 conditions." |
| Delta | "Twin-real delta reduced from X to Y across N cycles." |
| Observation | "Observed lower divergence after parameter adjustment." |
| Summary | "Divergence metrics within calibration band after tuning." |
| Trajectory | "Twin trajectory more closely tracks real runner behavior." |
| Qualified | "State delta at X — calibration target was Y." |

## Required Qualifier

All CAL-EXP-2 result statements must end with:

> **SHADOW MODE — observational only.**
