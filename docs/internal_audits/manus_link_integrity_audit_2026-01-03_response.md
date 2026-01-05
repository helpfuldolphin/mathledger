# Response to Hostile Link Integrity Audit

**Audit**: `manus_link_integrity_audit_2026-01-03.md`
**Response Date**: 2026-01-03
**Responder**: Claude A
**Status**: ACKNOWLEDGED - Findings accepted, fixes planned for v0.2.2

---

## Acknowledgment

This response acknowledges the hostile link integrity audit conducted on 2026-01-03. The audit was designed to break epistemic integrity by finding broken, misleading, or inconsistent links. **It succeeded in identifying real failures.**

We do NOT:
- Dispute the findings
- Soften the language
- Claim the failures were intentional

We DO:
- Accept responsibility for the failures
- Classify each finding by root cause
- Commit to specific fixes in v0.2.2

---

## Finding Classification

### Classification Key

| Code | Meaning |
|------|---------|
| DOC | Documentation bug - text says X, reality is Y |
| BUILD | Build pipeline bug - metadata/timestamps inconsistent |
| ARCH | Architectural decision that was underdocumented |
| NAV | Navigation/discoverability failure |

---

## BLOCKING Findings Acknowledged

### BLOCKER A: Demo/Archive Version Ambiguity

**Finding**: Demo shows `v0.2.0-demo-lock` but auditor checklist instructs verifying `v0.2.1-cohesion`

**Classification**: ARCH + DOC

**Root Cause**:
- Single `/demo/` instance serves all archive versions (architectural decision)
- This was never documented
- Auditor checklist was written assuming version-specific demo

**Acknowledged Failures**:
1. Checklist Step 1 says "verify banner shows v0.2.1-cohesion" - IMPOSSIBLE
2. No documentation explains the shared demo architecture
3. Creates immediate credibility failure for any auditor

**Fix Target**: v0.2.2

---

### BLOCKER B: Broken Auditor Instructions

**Finding**: Steps 3-5 of auditor checklist cannot be completed

**Classification**: DOC + NAV

**Root Cause**:
- Step 3: "Download Evidence Pack" button does not exist in boundary demo
- Step 4: Evidence Pack Verifier exists but is not linked anywhere
- Step 5: Depends on Steps 3-4, therefore impossible

**Acknowledged Failures**:
1. Auditor checklist was not tested against live system
2. Evidence Pack Verifier at `/v0.2.1/evidence-pack/verify/` is hidden
3. Instructions describe functionality that doesn't exist in demo UX

**Fix Target**: v0.2.2

---

### BLOCKER C: Immutability Signaling vs Build Metadata

**Finding**: Footer shows different commit/time than manifest.json

**Classification**: BUILD

**Root Cause**:
- Footer shows: commit 27a94c8a, time 18:55:59Z
- Manifest shows: build_commit cd2507d5, time 19:46:25Z
- Site was rebuilt but footer template used content commit, not build commit

**Acknowledged Failures**:
1. "Content commit" vs "build commit" distinction was not documented
2. Footer incorrectly suggests site was built from content commit
3. Creates appearance of tampering when it's actually a build pipeline quirk

**Fix Target**: v0.2.2

---

### BLOCKER D: Repository Verifiability

**Finding**: Verification instructions say "clone the repository" but no URL provided

**Classification**: DOC

**Root Cause**:
- Instructions contain placeholder "your-org/mathledger"
- Actual repo URL (github.com/helpfuldolphin/mathledger) was not substituted
- This was fixed in some docs but not all

**Acknowledged Failures**:
1. Verification instructions are impossible to follow
2. Core epistemic claim (source verifiability) is undermined
3. This is a documentation oversight, not an access restriction

**Fix Target**: v0.2.2

---

### BLOCKER E: v0.2.0 Claims CURRENT Status

**Finding**: v0.2.0 archive page shows "Status: CURRENT" but /versions/ shows "SUPERSEDED"

**Classification**: BUILD + ARCH

**Root Cause**:
- Archive pages are built at lock time with status at that moment
- When v0.2.1 was released, v0.2.0 archive was NOT rebuilt (by design)
- This means archived status is frozen, but /versions/ shows live status

**Acknowledged Failures**:
1. Immutability semantics were not explained
2. "Status" field in archived pages creates confusion
3. Should either: (a) never show status in archives, or (b) document this behavior

**Fix Target**: v0.2.2

---

## NON-BLOCKING Findings Acknowledged

| Finding | Classification | Severity | Notes |
|---------|---------------|----------|-------|
| Silent redirect to /v0.2.1 | ARCH | MINOR | Acceptable if documented |
| Relative vs absolute paths | BUILD | MINOR | Cosmetic inconsistency |
| "90-Second Proof" naming | DOC | MINOR | Already fixed in v0.2.1 |
| Hostile Rehearsal version label | DOC | MINOR | Content version vs archive version |
| v0 superseded-by inconsistency | BUILD | MINOR | Same root cause as BLOCKER E |

---

## What the Audit Got RIGHT

The audit correctly identified that:

1. **Auditor path is broken** - Steps cannot be completed
2. **Version signaling is inconsistent** - Multiple sources of truth
3. **Build metadata is confusing** - Content commit vs build commit not explained
4. **Repository URL is missing** - Verification impossible
5. **Demo sharing is undocumented** - Architectural decision never explained

These are **real failures** at the interface layer.

---

## What the Audit Did NOT Break

The audit did NOT find failures in:

1. **Hashing correctness** - SHA256 computations are sound
2. **Abstention preservation** - ADV excluded from R_t
3. **Authority boundaries** - DraftProposal never enters committed paths
4. **Replay verification** - Same code paths used
5. **Determinism** - Same inputs produce same outputs

The epistemic core is intact. The failures are at the documentation and build layers.

---

## Fix Commitment

| Blocker | Fix Version | Fix Type |
|---------|-------------|----------|
| A: Demo version ambiguity | v0.2.2 | DOC + possibly ARCH |
| B: Broken auditor instructions | v0.2.2 | DOC + NAV |
| C: Build metadata confusion | v0.2.2 | DOC + BUILD |
| D: Repository URL missing | v0.2.2 | DOC |
| E: v0.2.0 status confusion | v0.2.2 | DOC |

---

## Governance Note

This audit and response are preserved as evidence that:

1. We accept hostile external review
2. We acknowledge failures without softening
3. We commit to specific fixes with version targets
4. We distinguish interface failures from core failures

v0.2.1 is CLOSED with known defects. v0.2.2 will address them.

---

**Response Status**: COMPLETE
**Next Action**: Design v0.2.2 fixes (see v0.2.2 planning document)
