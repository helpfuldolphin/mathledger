# MathLedger v0.2.3 Epistemic Coherence Audit
## Acquisition Committee Gate Decision

**Auditor Role:** Hostile External Auditor
**Target:** v0.2.3-audit-path-freshness (CURRENT per /versions/)
**Audit Date:** 2026-01-03
**Commit:** 674bcd16104f37961fe1ce9e200a5b95a9c85bb3
**Scope:** Epistemic coherence, internal consistency, audit-readiness for outreach

---

## EXECUTIVE VERDICT

**FAIL FOR OUTREACH** — Micro-release required before acquisition committee presentation.

**Rationale:** v0.2.3 contains **1 BLOCKING** epistemic error in the Scope Lock that directly contradicts the demo behavior. This creates a credibility hazard: external evaluators following the documentation will encounter a contradiction between what the Scope Lock claims ("v0 has no verifier", "UI does not claim VERIFIED") and what the demo actually does (shows VERIFIED for MV claims).

The system architecture is sound, the tier classification is honest, and the governance model is internally consistent. However, the documentation contains a **factual error** that undermines trust. This must be corrected before outreach.

---

## FINDINGS SUMMARY

| Severity | Count | Gate Impact |
|----------|-------|-------------|
| BLOCKING | 1 | Prevents outreach |
| MAJOR | 2 | Requires fix but not blocking |
| MINOR | 3 | Note for future improvement |

**Total Issues:** 6
**Epistemic Soundness:** PASS (architecture is coherent)
**Documentation Accuracy:** FAIL (contains factual error)
**Audit Readiness:** FAIL (contradictory claims)

---

## BLOCKING FINDINGS

### BLOCKING-1: V0_LOCK.md Line 160 Contains Factual Error

**Location:** `docs/V0_LOCK.md` line 160

**Evidence:**
```markdown
Line 160: "PA terminology hazard: PA claims no longer return `VERIFIED`.
All claims in v0 return `ABSTAINED` with explicit `authority_basis` explanation."
```

**But:**
- `FOR_AUDITORS.md` line 19: "Claim '2+2=4' as **MV** → VERIFIED"
- `FOR_AUDITORS.md` line 22: "Claim '2+2=5' as **MV** → REFUTED"
- `V0_LOCK.md` line 57: "VERIFIED | MV claim parsed AND arithmetic confirmed"

**Analysis:**
The statement "All claims in v0 return ABSTAINED" is **factually false**. MV claims return VERIFIED or REFUTED via the arithmetic validator (`governance/mv_validator.py`). The sentence should read:

> "PA terminology hazard: PA claims no longer return `VERIFIED`. **PA, FV, and ADV claims** in v0 return `ABSTAINED` with explicit `authority_basis` explanation."

**Why BLOCKING:**
A cold auditor reading the Scope Lock will conclude that the demo should never show VERIFIED. When they run the demo and see VERIFIED for MV claims, they will assume:
1. The demo violates the scope, OR
2. The documentation is wrong

Either conclusion destroys credibility. This is a **gate-level epistemic error**.

**Recommendation:**
Correct line 160 to specify "PA, FV, and ADV claims" not "All claims". This is a one-line fix.

---

## MAJOR FINDINGS

### MAJOR-1: "v0 has no verifier" Contradicts Existence of MV Arithmetic Validator

**Location:** `docs/V0_LOCK.md` lines 67, 102

**Evidence:**
- Line 67: "In v0, no claims are mechanically verified"
- Line 102: "The UI does not claim VERIFIED for any claim (v0 has no verifier)"

**But:**
- `governance/mv_validator.py` exists and validates arithmetic claims
- MV claims return VERIFIED when arithmetic is correct
- The system DOES mechanically verify arithmetic

**Analysis:**
The Scope Lock uses "verifier" to mean "formal proof system (Lean/Z3)" but this creates ambiguity. The system HAS a verifier (arithmetic validator) but NOT a formal verifier. The terminology drift makes the document internally contradictory.

**Why MAJOR (not BLOCKING):**
This is a terminology issue, not a factual error. The intended meaning is clear from context: "v0 has no *formal* verifier." But hostile readers will exploit this ambiguity.

**Recommendation:**
Replace "v0 has no verifier" with "v0 has no formal verifier (Lean/Z3/Coq)". Add a note: "MV claims are validated via arithmetic checking, not formal proof."

---

### MAJOR-2: Tier A Count is 10 But Only 9 Invariants Are Explained

**Location:** `docs/invariants_status.md` lines 32-96, `releases/releases.json` v0.2.3

**Evidence:**
- `releases.json` v0.2.3: `"tier_a": 10`
- `invariants_status.md` lists 9 Tier A invariants (sections ### 1 through ### 9)
- v0.2.0 added "Audit Surface Version Field" as 10th Tier A invariant
- This invariant is not explained in `invariants_status.md`

**Analysis:**
The 10th Tier A invariant exists in `releases.json` but has no corresponding section in `invariants_status.md`. This creates a documentation gap: auditors cannot verify what "Audit Surface Version Field" means or how it is enforced.

**Why MAJOR:**
Tier A invariants are the core trust claims. If one is missing from the explanation document, auditors cannot verify it. This undermines the "brutally honest classification" promise.

**Recommendation:**
Add a section "### 10. Audit Surface Version Field" to `invariants_status.md` explaining what this invariant is and how it is enforced. OR remove it from Tier A if it is not actually enforced.

---

## MINOR FINDINGS

### MINOR-1: V0_LOCK.md Claims to Be "Frozen" But Contains v0.2.1 Release Notes

**Location:** `docs/V0_LOCK.md` lines 115, 251-332

**Evidence:**
- Line 115: "Date Locked: 2026-01-02. This scope is frozen."
- Lines 251-332: Release notes for v0.2.1 (locked 2026-01-03)

**Analysis:**
The Scope Lock claims to be frozen as of 2026-01-02, but it contains release notes for v0.2.1 which was locked 2026-01-03. This violates the immutability claim.

**Why MINOR:**
The V0_LOCK.md appears to be a **cumulative document** that tracks all v0.x releases, not a frozen artifact. The "Date Locked" refers to the original v0 scope, not the document itself. This is confusing but not epistemically unsound.

**Recommendation:**
Clarify that V0_LOCK.md is a "living scope document" that tracks v0.x releases, not a frozen artifact. OR split it into separate files per version.

---

### MINOR-2: Field Manual Uses "First Organism" and "Wide Slice" Terms Not Found in Public Docs

**Location:** `docs/field_manual/fm.pdf` page 1 (Abstract)

**Evidence:**
- FM Abstract: "how the 'First Organism' vertical slice and the Wide Slice abstention experiment operationalize these ideas in Phase I"
- These terms do not appear in V0_LOCK, FOR_AUDITORS, invariants_status, or HOW_THE_DEMO_EXPLAINS_ITSELF

**Analysis:**
The Field Manual introduces internal jargon that does not appear in public-facing docs. This could be:
1. Intentional separation of internal design docs from public explanations (GOOD)
2. Terminology drift where different names are used for the same concepts (BAD)

**Why MINOR:**
FOR_AUDITORS.md explicitly states the FM is "not rewritten yet; it is used to surface obligations and gaps." This signals that the FM is an internal artifact, not public documentation. The terminology drift is acceptable for internal docs.

**Recommendation:**
Add a note to FOR_AUDITORS.md explaining that FM terminology may differ from public docs because it is an internal design document.

---

### MINOR-3: "MV Validator Correctness" Is Tier B But Validator Bugs Are Not Detectable

**Location:** `docs/invariants_status.md` lines 99-108

**Evidence:**
- Tier B: "MV Validator Correctness (edge cases)"
- Description: "Edge cases (overflow, division by zero, floating point)"
- Detection: "Logged validation_outcome with parsed values"

**Analysis:**
Tier B is defined as "Logged and replay-visible. Violation is detectable but not prevented."

But MV validator bugs are NOT detectable from logs alone. If the validator has a bug (e.g., overflow), the log will show VERIFIED when it should show REFUTED. The bug is not detectable from the log—it requires independent verification.

**Why MINOR:**
This is a tier classification issue, not an epistemic error. The validator IS logged, so the classification is technically correct. But the "detectable" claim is weak.

**Recommendation:**
Reclassify "MV Validator Correctness" as Tier C (aspirational) with a note: "Validator bugs are not detectable from logs alone. Full correctness requires independent verification or formal proof of the validator."

---

## POSITIVE FINDINGS (Epistemic Strengths)

### ✓ Authority-Bearing vs. Verified Distinction Is Maintained

The docs correctly distinguish "authority-bearing" (enters R_t) from "verified" (mechanically checked). PA is authority-bearing but not verified. This is conceptually sound and consistently applied.

### ✓ Tier Classification Is Brutally Honest

The tier system (A/B/C) is honest about what is enforced vs. aspirational. Tier C explicitly lists "FV Mechanical Verification", "Multi-Model Consensus", "RFL Integration" as NOT enforced. This is credible.

### ✓ "Cannot Enforce" List Is Explicit

`releases.json` includes a `cannot_enforce` list for each version. This is rare in technical documentation and demonstrates epistemic honesty.

### ✓ Template Variable Substitution Works Correctly

FOR_AUDITORS.md uses `{{CURRENT_VERSION}}` templates which are correctly rendered as `/v0.2.3/` in the built HTML. The build script works as intended.

### ✓ Field Manual Exists and Is Substantial

The 195-page Field Manual demonstrates serious architectural thinking. The abstract's focus on "command knowledge" (ability to audit and explain) is epistemically sound.

### ✓ HOW_THE_DEMO_EXPLAINS_ITSELF.md Is Epistemically Careful

The document explicitly lists "What This Demo Refuses to Claim" (lines 119-130). It does not claim alignment, intelligence, safety, or generalization. This is rare and credible.

---

## CROSS-DOCUMENT CONSISTENCY CHECK

| Concept | V0_LOCK | FOR_AUDITORS | invariants_status | HOW_THE_DEMO | Consistent? |
|---------|---------|--------------|-------------------|--------------|-------------|
| Tier A count | 10 (v0.2.0+) | Not specified | 9 explained | Not specified | MAJOR-2 |
| MV returns VERIFIED | ✓ (line 57) | ✓ (line 19) | ✓ (implied) | ✓ (line 25) | PASS |
| "All claims ABSTAINED" | X (line 160) | Contradicts | N/A | N/A | BLOCKING-1 |
| ADV excluded from R_t | ✓ | ✓ | ✓ (Tier A) | ✓ | PASS |
| H_t = SHA256(R_t \|\| U_t) | ✓ | ✓ | ✓ (Tier A) | ✓ | PASS |
| "v0 has no verifier" | X (line 102) | Contradicts | N/A | N/A | MAJOR-1 |
| Abstention is first-class | ✓ | ✓ | ✓ (Tier A) | ✓ (line 29) | PASS |
| No formal verifier | ✓ (Tier C) | ✓ (line 102) | ✓ (Tier C) | N/A | PASS |

**Consistency Score:** 6/8 PASS, 2/8 FAIL

---

## IMMUTABILITY AND VERSION SEMANTICS

### ✓ /versions/ Correctly Shows v0.2.3 as CURRENT

The canonical `releases/releases.json` has `"current_version": "v0.2.3"` and v0.2.3 has `"status": "current"`. This is correct.

### ✓ Prior Versions Correctly Marked as SUPERSEDED

- v0: "superseded-by-v0.2.1"
- v0.2.0: "superseded-by-v0.2.1"
- v0.2.1: "superseded-by-v0.2.2"
- v0.2.2: "superseded-by-v0.2.3"

Version semantics are correct.

### Warning: V0_LOCK.md Immutability Claim Is Ambiguous

The document claims to be "frozen" but contains release notes for multiple versions. This is confusing but not epistemically unsound if V0_LOCK.md is intended as a cumulative document.

---

## ACQUISITION COMMITTEE RISK ASSESSMENT

### Credibility Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Auditor finds contradiction between Scope Lock and demo | HIGH | Fix BLOCKING-1 before outreach |
| "v0 has no verifier" exploited by hostile evaluator | MEDIUM | Fix MAJOR-1 terminology |
| Tier A count mismatch undermines trust | MEDIUM | Fix MAJOR-2 documentation gap |
| Field Manual jargon confuses external readers | LOW | Add clarification note |

### Epistemic Strengths for Acquisition

1. **Honest about limitations:** Tier C explicitly lists what is NOT enforced
2. **Refuses to overclaim:** HOW_THE_DEMO explicitly lists non-claims
3. **Verifiable architecture:** Evidence packs enable independent verification
4. **Substantial design thinking:** 195-page Field Manual demonstrates depth
5. **External audit history:** Three prior audits listed in FOR_AUDITORS

### Recommended Talking Points for Outreach

**DO emphasize:**
- Tier A/B/C classification system (brutal honesty)
- Evidence pack replay verification (independent auditability)
- Authority-bearing vs. verified distinction (epistemic precision)
- "Cannot enforce" list (credible non-claims)

**DO NOT emphasize:**
- "v0 has no verifier" (ambiguous, will be exploited)
- Tier A count (documentation gap)
- Field Manual (internal doc, not ready for external consumption)

---

## RECOMMENDATION

**HOLD OUTREACH** until micro-release v0.2.4 addresses BLOCKING-1.

### Required for v0.2.4

1. **Fix BLOCKING-1:** Correct V0_LOCK.md line 160 to specify "PA, FV, and ADV claims" not "All claims"
2. **Fix MAJOR-1:** Replace "v0 has no verifier" with "v0 has no formal verifier"
3. **Fix MAJOR-2:** Add section explaining 10th Tier A invariant OR remove it from tier count

### Estimated Effort

- **BLOCKING-1:** 1 line change, 5 minutes
- **MAJOR-1:** 3-5 line changes, 15 minutes
- **MAJOR-2:** Add 1 section to invariants_status.md, 30 minutes

**Total:** ~1 hour for micro-release v0.2.4

### Post-Fix Verdict

After v0.2.4 addresses these issues:

**PROCEED WITH OUTREACH** — System is epistemically coherent and audit-ready.

---

## AUDITOR SIGN-OFF

This audit was conducted with hostile intent to identify credibility hazards before acquisition committee presentation. The architecture is sound, the governance model is internally consistent, and the tier classification is honest. However, the documentation contains a factual error (BLOCKING-1) that must be corrected before external evaluation.

The system demonstrates rare epistemic honesty (explicit non-claims, cannot-enforce lists, tier classification). This is a strength. The documentation error is fixable and does not reflect architectural flaws.

**Recommended action:** Micro-release v0.2.4 to fix documentation, then proceed with outreach.

---

**Audit completed:** 2026-01-03
**Auditor:** Hostile External (Manus Agent)
**Next review:** Post-v0.2.4 smoke test
