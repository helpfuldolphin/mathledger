# CAL-EXP-2 Distribution Checklist

> **Status**: NON-NORMATIVE GUIDANCE
> **Version**: 1.0.0
> **Date**: 2025-12-13
> **Scope**: Pre-distribution review for CAL-EXP-2 results

---

## Purpose

This checklist provides **non-normative guidance** for reviewing CAL-EXP-2 results before external distribution. It is advisory only and does not gate distribution decisions.

---

## Pre-Distribution Checklist

Before sharing CAL-EXP-2 results outside the immediate engineering team, review the following:

### 1. Footer Attachment

- [ ] Distribution footer block is present at end of document
- [ ] Footer references `CAL_EXP_DO_NOT_CLAIM_APPENDIX.md`

### 2. Language Review

- [ ] Document does NOT contain: "validated", "certified", "production-ready"
- [ ] Document does NOT contain: "improved" (use "reduced error" or "adjusted")
- [ ] Document does NOT contain: "breakthrough" (use "measured reduction")

### 3. Context Phrases

- [ ] "SHADOW MODE" appears at least once
- [ ] "synthetic conditions" or "test conditions" appears in results section
- [ ] Numeric claims include measurement context (e.g., "under test conditions")

### 4. Metric Framing

- [ ] `mean_delta_p` described as "state tracking error", not "accuracy"
- [ ] `divergence_rate` described as "estimator agreement", not "performance"
- [ ] "Safe region" accompanied by "mathematical Ω-region" clarification if used

### 5. Claims Check

- [ ] No claims of capability advancement
- [ ] No claims of safety guarantee
- [ ] No claims of deployment readiness
- [ ] No claims of regulatory compliance

### 6. Reference Attachment

- [ ] Link to `CAL_EXP_DO_NOT_CLAIM_APPENDIX.md` included or attached
- [ ] Link to `CAL_EXP_DISCLAIMER_TEMPLATE.md` available if requested

---

## Reviewer Sign-Off (Optional)

| Item | Reviewer | Date |
|------|----------|------|
| Checklist reviewed | _______________ | _______________ |
| Distribution approved | _______________ | _______________ |

---

## Non-Normative Notice

This checklist is **advisory guidance only**. It does not:
- Gate or block distribution decisions
- Constitute a formal review process
- Create compliance obligations
- Replace engineering judgment

Distribution decisions remain with the responsible parties.

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `CAL_EXP_DO_NOT_CLAIM_APPENDIX.md` | Misinterpretation shield |
| `CAL_EXP_DISCLAIMER_TEMPLATE.md` | Reusable disclaimer language |
| `CAL_EXP_2_Canonical_Record.md` | Canonical experiment record |
| `CAL_EXP_2_RESULTS_TEMPLATE.md` | Results template with footer |

---

**END OF CHECKLIST**
