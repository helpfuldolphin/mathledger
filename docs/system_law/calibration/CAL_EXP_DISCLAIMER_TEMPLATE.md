# Calibration Experiment Disclaimer Template

> **Status**: NON-CANONICAL — TEMPLATE FOR REUSE
> **Version**: 1.0.0
> **Date**: 2025-12-13
> **Purpose**: Standard disclaimer language for CAL-EXP reports
> **Usage**: Copy and adapt for CAL-EXP-1, CAL-EXP-2, etc.

---

## 1. Standard Disclaimer (Copy This)

> **Calibration Experiment Disclaimer**: This experiment measures the effect of parameter adjustments on divergence metrics under controlled synthetic conditions. All results are **SHADOW MODE observations** — they do not represent production behavior, safety guarantees, or capability claims. Divergence reduction indicates improved state tracking by the Twin estimator against synthetic test data; it does not certify system correctness or predict real-world performance. Metrics such as `mean_delta_p` and `divergence_rate` measure simulator agreement, not behavioral correctness. These results require independent replication and are subject to revision as calibration continues.

---

## 2. Forbidden Words (Do Not Use)

| Word/Phrase | Reason |
|-------------|--------|
| breakthrough | Implies paradigm shift |
| solved | Implies completeness |
| proven safe | No safety guarantees |
| optimal | Implies global minimum |
| accurate | Requires benchmark qualification |
| autonomous | Implies agency |
| superior | Comparative claim |
| guarantees | No guarantees in SHADOW mode |
| eliminates | Reduction ≠ elimination |
| validates | Use "tests" or "measures" |
| certifies | No certification authority |
| converged | Use "trending toward" |

---

## 3. Safe Verbs (Prefer These)

| Verb | Usage |
|------|-------|
| **measure** | "We measure divergence rate under test conditions" |
| **observe** | "We observe a reduction in mean_delta_p" |
| **record** | "Results are recorded in the evidence pack" |
| **compare** | "We compare baseline to adjusted parameters" |
| **test** | "We test parameter configurations" |
| **reduce** | "Parameter adjustment reduces error" |
| **adjust** | "We adjust learning rates" |
| **trend** | "Metrics trend toward target" |

---

## 4. Required Context Phrases

Every CAL-EXP report SHOULD include at least one of:

- "SHADOW MODE"
- "synthetic conditions" or "controlled conditions"
- "does not certify" or "does not guarantee"
- Specific numeric deltas (e.g., "from 0.87 to 0.34")

---

## 5. Usage Example

**BAD**:
> "CAL-EXP-2 achieved a breakthrough in divergence reduction, proving the Twin is now accurate and autonomous."

**GOOD**:
> "CAL-EXP-2 measured a reduction in mean_delta_p from 0.087 to 0.034 under synthetic test conditions. This observes improved state tracking but does not certify system correctness. Results are SHADOW MODE only."

---

## 6. Adaptation Notes

When adapting this template:

1. Replace `[CAL-EXP-N]` with specific experiment ID
2. Insert actual numeric results where placeholders appear
3. Add experiment-specific context if needed
4. Do NOT remove the core disclaimer language
5. Do NOT add words from the forbidden list

---

**TEMPLATE STATUS**: This document is a reusable template, not a canonical specification. It does not define system behavior or governance policy.
