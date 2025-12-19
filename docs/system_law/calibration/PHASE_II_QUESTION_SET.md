# Phase II Question Set

**Status**: PRE-EXPERIMENTAL (question selection)
**Date**: 2025-12-19
**Mode**: SHADOW (epistemic reframing only)
**Authority**: STRATCOM decision pending

---

## Preamble

Phase I is formally closed. All three experiments (CAL-EXP-3/4/5) answered their questions. Phase II requires a qualitatively different question that:

- Is not a refinement of Phase I questions
- Does not depend on achieving PASS in Phase I
- Does not introduce capability, performance, or intelligence claims
- Is answerable using the existing governance substrate

This document presents three mutually exclusive candidate questions for STRATCOM selection.

---

## Candidate A: Predicate Separability

### Question

**Does the F5.x predicate set partition failure modes into distinguishable equivalence classes under the existing governance substrate?**

### What It Would Establish If Answered

| If YES | If NO |
|--------|-------|
| F5.x codes identify distinct failure modes | F5.x codes conflate distinct failure modes |
| Each F5.x code has independent diagnostic value | Some F5.x codes are redundant or overlapping |
| Governance can attribute failures to specific causes | Governance cannot distinguish between certain failures |

### What It Would NOT Establish

| Non-Claim | Reason |
|-----------|--------|
| Which predicates should be added or removed | Outside scope (no recommendations) |
| Whether current predicates are optimal | Separability ≠ optimality |
| Capability or performance | Question is about governance, not system |
| Threshold validity | Predicates evaluated, not thresholds |

### Is FAIL Epistemically Valuable?

**YES.** A FAIL outcome would establish that F5.x codes conflate failure modes, indicating that the governance substrate has limited diagnostic resolution. This is valuable for understanding the limits of the current predicate set without implying any corrective action.

### Why This Question Cannot Be Asked in Phase I

Phase I asked whether specific predicates trigger (CAL-EXP-4) and whether they can be avoided (CAL-EXP-5). It did not ask whether the predicate set as a whole has sufficient resolution to distinguish failure modes. Phase I observed F5.2 and F5.3 co-occurring, but did not assess whether they represent independent failure modes or are causally linked under the governance model.

---

## Candidate B: Governance Stability Under Replay

### Question

**Is the governance verdict (PASS/FAIL/claim-level) invariant under deterministic replay with perturbed auxiliary parameters that do not affect the frozen predicates?**

### What It Would Establish If Answered

| If YES | If NO |
|--------|-------|
| Governance verdicts are robust to non-predicate variation | Governance verdicts are sensitive to implementation details |
| Frozen predicates fully determine outcomes | Hidden dependencies exist outside frozen predicates |
| Replay is a valid verification method | Replay may produce spurious divergence |

### What It Would NOT Establish

| Non-Claim | Reason |
|-----------|--------|
| What auxiliary parameters should be frozen | Outside scope (no recommendations) |
| Whether sensitivity is desirable or undesirable | Stability ≠ correctness |
| Capability or performance | Question is about governance invariance |
| Threshold validity | Predicates and thresholds not modified |

### Is FAIL Epistemically Valuable?

**YES.** A FAIL outcome would establish that governance verdicts depend on parameters outside the frozen predicate set. This reveals hidden assumptions in the governance model and identifies where the "frozen" boundary is incomplete. This is valuable for understanding governance fragility without implying corrective action.

### Why This Question Cannot Be Asked in Phase I

Phase I established deterministic replay (identical seeds → identical outputs) but did not test whether the governance verdict is invariant under perturbations to auxiliary parameters (e.g., timestamp formatting, JSON serialization order, logging verbosity). Phase I confirmed that the system is deterministic; it did not confirm that governance is robust to non-semantic variation.

---

## Candidate C: Observability Sufficiency

### Question

**Does the current artifact contract provide sufficient observability to distinguish between runs that fail for the same F5.x code but for structurally different reasons?**

### What It Would Establish If Answered

| If YES | If NO |
|--------|-------|
| Artifacts contain enough information to diagnose failures | Artifacts are insufficient for failure diagnosis |
| Post-hoc analysis can identify failure causes | Post-hoc analysis cannot distinguish failure modes |
| Governance substrate supports forensic audit | Governance substrate has observability gaps |

### What It Would NOT Establish

| Non-Claim | Reason |
|-----------|--------|
| What additional artifacts should be added | Outside scope (no recommendations) |
| Whether observability gaps are problematic | Sufficiency ≠ optimality |
| Capability or performance | Question is about governance observability |
| Root cause of any specific failure | Observability ≠ diagnosis |

### Is FAIL Epistemically Valuable?

**YES.** A FAIL outcome would establish that the artifact contract has observability gaps—runs with the same F5.x code cannot be distinguished structurally using the current artifacts. This is valuable for understanding the limits of forensic analysis without implying any artifact additions.

### Why This Question Cannot Be Asked in Phase I

Phase I produced artifacts and verified their presence. It did not assess whether the artifacts contain sufficient information to distinguish between structurally different failures. All Phase I runs produced F5.2, but the artifacts were not analyzed for their diagnostic power—only for their conformance to the contract.

---

## Question Comparison Matrix

| Aspect | Candidate A | Candidate B | Candidate C |
|--------|-------------|-------------|-------------|
| Domain | Predicate structure | Verdict stability | Artifact sufficiency |
| Tests | F5.x separability | Governance invariance | Observability depth |
| Requires new predicates | Possibly | No | No |
| Requires new artifacts | No | No | Possibly (for analysis) |
| FAIL is valuable | YES | YES | YES |
| Depends on PASS in Phase I | NO | NO | NO |
| Uses existing substrate | YES | YES | YES |

---

## Mutual Exclusivity

The three candidates are mutually exclusive in their primary concern:

| Candidate | Primary Concern |
|-----------|-----------------|
| A | **Structure** of the predicate set |
| B | **Stability** of governance verdicts |
| C | **Depth** of artifact observability |

Selecting one does not preclude asking the others in subsequent phases, but each represents a distinct epistemic direction for Phase II.

---

## STRATCOM Decision Required

One candidate must be selected before any execution planning may proceed.

| Decision | Next Step |
|----------|-----------|
| Select Candidate A | Predicate separability analysis |
| Select Candidate B | Governance stability testing |
| Select Candidate C | Observability sufficiency assessment |
| Reject all candidates | Reformulate Phase II scope |

---

**No execution authorized until question selection.**

**SHADOW MODE** — Epistemic reframing only.

*Precision > optimism.*
