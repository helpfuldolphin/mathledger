# CAL-EXP-4 — Private Design Note (Phase II Draft)

**STATUS**: PRIVATE / NON-BINDING / NOT FOR IMPLEMENTATION
**Author**: Claude Topologist
**Date**: 2025-12-14
**Scope**: Conceptual design only
**Audience**: MathLedger Architect (internal)

---

> **QUARANTINE NOTICE**
>
> This document is a latent design artifact. It is:
> - NOT approved for implementation
> - NOT referenced in external communications
> - NOT part of the canonical calibration sequence
> - NOT a commitment to any future experiment
>
> It exists solely as an internal design note for Phase II planning.

---

## 1. Question

Does the governance substrate maintain validity-check coherence when the corpus exhibits different inter-cycle variance profiles than those present during CAL-EXP-3 calibration?

---

## 2. New Risk Surface

**Risk**: Validity checker temporal coupling failure.

CAL-EXP-3 operated under a fixed inter-cycle variance profile—the magnitude and frequency of state changes between consecutive cycles were determined by hardcoded corpus generation parameters. The validity checks (F1.1-F2.3) were verified against artifacts exhibiting that specific variance profile. The risk is that validity checkers may pass under low-variance conditions but produce spurious or meaningless results under high-variance conditions—not because the underlying system is broken, but because the verifier's notion of "identical input corpus" implicitly assumes a variance profile that is no longer present.

CAL-EXP-3 could not surface this risk because all runs shared the same corpus generation parameters. The verifier was never asked to certify comparability between runs with different variance profiles. A validity check that certifies "identical corpus" based on content hash alone may be insufficient if the temporal structure of that corpus materially affects the system's behavior.

---

## 3. System Law Stress Point

**Stressed component**: Verifier comparability assumption.

The current verifier certifies that two runs are comparable if they share identical corpus content (F1.2: corpus hash match). This check is content-based, not structure-based. The implicit assumption is that corpus identity implies behavioral comparability—that two runs with the same corpus hash can be meaningfully compared regardless of how state changes are distributed across cycles.

CAL-EXP-4 would stress this assumption by parameterizing the corpus's inter-cycle variance profile (e.g., concentrated vs. distributed state changes) while holding content hash constant. The question is not whether the system performs differently under different profiles—the question is whether the verifier's comparability certification remains sound when the temporal distribution of state changes varies.

If the verifier certifies L4 for runs with materially different variance profiles, and those runs are not in fact comparable under the intent of the validity conditions, then the governance apparatus has a soundness gap.

---

## 4. Falsification Criterion

**Failure mode**: The verifier certifies comparability (via corpus hash match) for two runs whose variance profiles differ sufficiently that the "identical input corpus" condition is not meaningfully satisfied.

Concretely: if two runs pass F1.2 (corpus identity) because they share the same set of state values, but differ in how those values are distributed across cycles (e.g., Run A has gradual drift, Run B has sudden jumps), and this difference would cause a reasonable auditor to reject the comparison as apples-to-oranges, then the verifier is unsound.

The unsoundness is specific: the validity condition F1.2 checks content identity but not temporal identity. The verifier would be certifying that runs are comparable when they are not, because the condition is underspecified.

A false negative (verifier rejects runs that are in fact comparable) is preferable to a false positive (verifier certifies comparability for runs that should not be compared). CAL-EXP-4 should be designed to expose false positives in the comparability certification.

---

## Notes

- "Inter-cycle variance profile" refers to the distribution of state-change magnitudes across the cycle sequence—not a new metric, but a property of corpus construction.
- The parameter is a corpus generation knob (e.g., noise scale, drift rate), not a measured output of the system.
- This experiment does not test whether the system "handles" variance better or worse. It tests whether the governance apparatus can distinguish runs that should not be compared.
- If the verifier is already sound under variance profile changes, CAL-EXP-4 would confirm that soundness.
- No changes to the claim ladder, validity conditions, or artifact contract are proposed. The experiment tests completeness of existing conditions.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 0.1 | 2025-12-14 | Initial private design note |

---

**PRIVATE DESIGN — PHASE II — NOT FOR IMPLEMENTATION**
