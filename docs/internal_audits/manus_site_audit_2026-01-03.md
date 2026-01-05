# MathLedger Cold-Start Audit Report

**Date:** January 3, 2026  
**Auditor Perspective:** External safety lead/auditor/potential acquirer with no prior context

---

## Part 1 — First Contact (10–15 seconds)

### What I See Above the Fold

**Headings:**
- "MathLedger — Version v0.2.1"
- "Version v0.2.1 Archive" with "ARCHIVE" badge
- "Hosted Interactive Demo"

**Status Banner:**
- Green-highlighted box at top showing "Status: CURRENT"
- Tag: v0.2.1-cohesion | Commit: 27a94c8a5813 | Locked: 2026-01-03

**Tier Information:**
- "Tier A (enforced): 10"
- "Tier B (logged): 1"
- "Tier C (aspirational): 3"

**Warnings/Non-Claims (prominently displayed in red/warning text):**
"What this version cannot enforce:"
- No Lean/Z3 verifier: FV claims always return ABSTAINED
- Single template partitioner: no multi-model consensus
- No learning loop: RFL not active
- MV edge cases: overflow, float precision not fully covered

**Calls to Action:**
- Navigation tabs: Scope, Explanation, Invariants, Fixtures, Evidence, All Versions
- Large green button: "Open Interactive Demo"
- Link: "5-minute auditor checklist" (described as "New to MathLedger? Start with...")

**Additional Visual Elements:**
- Statement: "Interactive demo is hosted; archive remains immutable."
- Statement: "This is the archive for MathLedger version v0.2.1. All artifacts below are static, verifiable, and immutable."

### After ~10 Seconds: What I Believe This Project Is

**What I believe this project is:**
MathLedger appears to be a **versioned epistemic archive system** that demonstrates some form of mathematical or logical claim verification with explicit enforcement tiers. It presents itself as an immutable, verifiable artifact with a hosted interactive demo. The emphasis on "what this version cannot enforce" and the tier system (A/B/C) suggests it's a rigorous demonstration of claim verification with explicit limitations.

**What I believe it is explicitly NOT:**
- NOT a production-ready verification system (given the prominent warnings about missing Lean/Z3 verifier, no multi-model consensus, no learning loop)
- NOT claiming to verify all mathematical edge cases (explicitly calls out overflow and float precision gaps)
- NOT a general-purpose AI demo (the language is unusually technical and limitation-focused)
- NOT making implicit promises (the "What this version cannot enforce" section is more prominent than typical feature lists)

**Key Quoted Phrases:**
- "What this version cannot enforce" (unusual negative framing)
- "static, verifiable, and immutable"
- "Tier A (enforced): 10 / Tier B (logged): 1 / Tier C (aspirational): 3"
- "archive remains immutable"

---

## Part 2 — Archive Understanding

### Scope Lock Exploration

**What v0 Demonstrates:**
The Scope Lock document states that "v0 is a governance demo" that demonstrates:
1. UVIL (User-Verified Input Loop): Human binding mechanism for authority
2. Trust Classes: FV, MV, PA (authority-bearing) vs ADV (exploration-only)
3. Dual Attestation Roots: U_t (UI), R_t (reasoning), H_t (composite)
4. Determinism: Same inputs produce same outputs, replayable
5. Exploration/Authority Boundary: DraftProposal never enters hash-committed paths

**What v0 Does NOT Demonstrate (explicit exclusions):**
- RFL learning loop: No curriculum, no policy, no uplift
- Multi-model arena: Single template partitioner, no LiteLLM, no model competition
- Agent tools: No code execution, no sandbox, no E2B
- Real verifier: No Lean, no Z3, no mechanical proof checking
- Production auth: No user accounts, no API keys, no rate limiting
- Persistence: In-memory only, restart-loss accepted
- Long-running agents: Synchronous request/response only

**Critical Statement:**
"v0 is a governance substrate demo, not a capability demo."

The document explicitly states what v0 IS and what it does NOT show:
- It shows: The boundary between exploration and authority is real, testable, and replayable; the system stops when it cannot verify
- It does NOT show: "That the system is intelligent / That the system is aligned / That the system is safe / That verification works (v0 has no verifier)"

**Terminology:**
Three outcome values are defined:
- VERIFIED: MV claim parsed AND arithmetic confirmed (e.g., "2 + 2 = 4")
- REFUTED: MV claim parsed AND arithmetic failed (e.g., "2 + 2 = 5")
- ABSTAINED: Cannot mechanically verify (PA, FV, unparseable MV, ADV-only)

**Lock Statement:**
The document includes explicit "Allowed Iterations" vs "Forbidden Iterations" and a formal lock statement dated 2026-01-02.

---

### Explanation Page Exploration

**Core Mechanism:**
The Explanation document describes how the demo separates exploration from authority:
- **Exploration phase:** System generates a DraftProposal with random identifier; nothing is committed; editing is free
- **Authority phase:** When user clicks "Commit," system creates CommittedPartitionSnapshot with content-derived identifier (hash); claims become immutable

**Key Design Principle:**
"exploration identifiers never appear in committed data" - this is described as "not an implementation detail. It is the design."

**Three Outcomes:**
- VERIFIED: machine-checkable proof that claim holds
- REFUTED: machine-checkable proof that claim does not hold
- ABSTAINED: system did not find either (described as "not a failure" but "the correct output")

**Critical Quote:**
"This stopping is correctness, not caution. A system that produces confident outputs when it lacks grounds for confidence is broken."

**ADV (Advisory) Trust Class:**
ADV content is "explicitly inert" - it does not enter the reasoning root (R_t), does not contribute to attestation, is recorded but has no authority. The document states: "This is not humility theater... It is making a structural distinction: suggestions exist, but they are not claims."

**Hash System:**
- U_t: UI Merkle root (commits to human actions)
- R_t: Reasoning Merkle root (commits to system's established claims)
- H_t: Composite root (binds U_t and R_t together)

**Evidence Pack:**
Self-contained JSON file enabling independent verification through replay. Document explicitly states what replay does NOT prove: "Replay verification proves structural integrity, not truth... It proves nothing about what that trail represents."

**Explicit Non-Claims Section:**
The document has a section titled "What This Demo Refuses to Claim" stating the demo does NOT claim:
- System is aligned with human values
- System is intelligent
- System is safe
- System will behave well in novel situations

It claims only: "the system's governance is legible."

---

### Invariants Page Exploration

**Document Description:**
"This document provides a brutally honest classification of governance invariants."

**Tier Classification System:**
- **Tier A:** Cryptographically or structurally enforced. Violation is impossible without detection. (10 invariants)
- **Tier B:** Logged and replay-visible. Violation is detectable but not prevented. (1 invariant)
- **Tier C:** Documented but not enforced in v0. Aspirational. (3 invariants)

**Tier A Invariants (10 total):**
1. Canonicalization Determinism
2. H_t = SHA256(R_t || U_t)
3. ADV Excluded from R_t
4. Content-Derived IDs
5. Replay Uses Same Code Paths
6. Double-Commit Returns 409
7. No Silent Authority
8. Trust-Class Monotonicity
9. Abstention Preservation
10. Audit Surface Version Field

**Tier B (1 invariant):**
- MV Validator Correctness (edge cases) - logged but not hard-gated

**Tier C (3 invariants - aspirational, not in v0):**
1. FV Mechanical Verification (no Lean/Z3 verifier)
2. Multi-Model Consensus (single template partitioner)
3. RFL Integration (no learning loop)

**Notable Feature:**
The invariants table includes columns for "How It Can Be Violated Today" and "Current Detection" - showing explicit acknowledgment of limitations and attack surfaces.

---

### Fixtures Page Exploration

**Description:**
Regression test fixtures for version v0.2.1. Each fixture contains input and expected output JSON files.

**9 Test Fixtures:**
1. adv_only
2. mixed_mv_adv
3. mv_arithmetic_refuted
4. mv_arithmetic_verified
5. mv_only
6. pa_only
7. same_claim_as_adv
8. same_claim_as_pa
9. underdetermined_navier_stokes

**Verification:**
- Checksum verification available via index.json with SHA256 checksums
- Regression harness can be run locally with: `uv run python tools/run_demo_cases.py`

**Observation:**
The fixture names are descriptive and cover different trust class scenarios (ADV, MV, PA) and outcomes (verified, refuted). The "underdetermined_navier_stokes" fixture suggests testing with complex mathematical claims that cannot be verified.

---

### Evidence Pack Page Exploration

**Purpose:**
"The evidence pack enables independent replay verification. An auditor can recompute attestation hashes without running the demo."

**What Replay Verification Proves:**
- The recorded hashes match what the inputs produce
- The attestation trail has not been tampered with
- Determinism: same inputs produce same outputs

**What Replay Verification Does NOT Prove:**
- That the claims are true
- That the verification was sound
- That the system behaved safely

**Replay Instructions:**
Concrete command-line instructions provided for running replay verification locally.

---

### Part 2 Summary: What MathLedger Claims and Refuses to Claim

**What MathLedger Claims:**
1. It is a **governance substrate demo**, not a capability demo
2. It demonstrates a structural boundary between exploration and authority that is cryptographically enforced
3. It provides deterministic, replayable attestations with content-derived identifiers
4. It has 10 Tier A invariants that are cryptographically or structurally enforced
5. The system's governance is **legible** - you can see what the human committed and what the system established
6. Replay verification proves structural integrity (that the audit trail is intact)

**What MathLedger Refuses to Claim:**
1. That the system is intelligent, aligned, or safe
2. That verification works (v0 has no verifier - all FV claims return ABSTAINED)
3. That it demonstrates capability (explicitly: "not a capability demo")
4. That replay verification proves truth or soundness (only structural integrity)
5. That it generalizes to production or handles all edge cases
6. That it has multi-model consensus, learning loops, or real mechanical verification

**What is Unusually Explicit or Disciplined:**

The most striking feature is the **prominence of non-claims and limitations**. The homepage displays "What this version cannot enforce" in red warning text before any feature descriptions. Every major document includes explicit sections on what is NOT claimed or NOT proven. The Invariants document is titled "brutally honest classification" and includes a column for "How It Can Be Violated Today." The Evidence Pack page explicitly states what replay does NOT prove. This negative framing is far more prominent than in typical AI demos, which usually emphasize capabilities.

The **tier system** (A/B/C) provides unusual transparency about enforcement levels - most systems would claim everything is "secure" without distinguishing between cryptographically enforced, logged-but-not-prevented, and aspirational invariants.

The **terminology discipline** is rigorous: ABSTAINED is treated as a "first-class outcome" rather than a failure, and the system explicitly refuses to return VERIFIED when it cannot mechanically verify.

**What Feels Confusing or Underspecified:**

1. **Target audience ambiguity:** The site assumes significant technical sophistication (terms like "Merkle root," "RFC 8785-style canonicalization," "content-derived IDs") without clear onboarding for different audience levels.

2. **"Governance demo" framing:** While the site repeatedly states this is a "governance substrate demo," it's not immediately clear what problem this solves or why governance without capability matters. The value proposition requires significant inference.

3. **Missing context for "FM":** The Invariants page references "FM Section" repeatedly (§1.5, §4, etc.) but doesn't explain what FM is or link to it.

4. **UVIL acronym:** "User-Verified Input Loop" is mentioned but not explained in depth on the homepage - requires clicking through to understand.

5. **Transition to demo:** While the green "Open Interactive Demo" button is visible, there's no clear narrative bridge explaining "now that you understand the governance claims, here's how to see them in action."

**Where I Had to "Work" to Understand:**

1. Understanding the distinction between "authority-bearing" and "exploration-only" required reading multiple documents
2. Grasping why a demo with no real verifier is valuable took time - the point is governance infrastructure, not verification capability
3. The relationship between the archive (static documentation) and the demo (interactive) wasn't immediately clear
4. Understanding what "epistemic archive" means in practice
5. Connecting the tier system to actual enforcement mechanisms required reading code examples

---

## Part 3 — Transition to the Demo

### How the Site Points to the Interactive Demo

**Link Visibility:**
The link is **obvious**. There is a large green button labeled "Open Interactive Demo" prominently displayed in a light green box titled "Hosted Interactive Demo" near the top of the archive page. The button uses high contrast (dark green on light green background) and is the only call-to-action button on the page.

**Framing:**
The framing is **clear but minimal**. The box states: "Interactive demo is hosted; archive remains immutable." This establishes the relationship between the two artifacts (demo is live, archive is static) but doesn't explain what the demo will show or why you should use it.

Below the button, there's a secondary link: "New to MathLedger? Start with the 5-minute auditor checklist" - this provides an alternative entry point for newcomers.

**Continuation vs. Separate Thing:**
It feels like **intentionally separated but related artifacts**. The framing "Interactive demo is hosted; archive remains immutable" explicitly distinguishes them. The archive is presented as the authoritative, immutable documentation, while the demo is positioned as a separate hosted application. This separation feels deliberate - the archive documents what the demo does, and the demo demonstrates what the archive describes.

### Transition Feel

The transition feels **intentionally gated** rather than seamless. The archive requires you to understand the governance claims (Scope Lock, Explanation, Invariants) before you interact with the demo. There's no narrative flow like "Now let's see this in action!" - instead, the demo is presented as a parallel artifact that you can access when ready.

The transition is **not confusing** - the button is clear and the relationship is stated. However, it is **somewhat technical** in that it assumes you understand what "hosted" vs "immutable archive" means and why that distinction matters.

---

## Part 4 — Interactive Demo Evaluation

### Initial Demo Interface Observations

**Header Banner:**
Black banner at top: "GOVERNANCE DEMO (not capability)" with version info: v0.2.0 | v0.2.0-demo-lock | 27a94c8a5813

**Framing Text (above the fold):**
Three key statements displayed prominently:
1. "The system does not decide what is true. It decides what is justified under a declared verification route."
2. "This demo will stop more often than you expect. It reports what it cannot verify."
3. "If you are looking for a system that always has an answer, this demo is not it." (in italics)

**Demo Section:**
- Title: "Same Claim, Different Authority"
- Button: "Run 90-Second Proof" (white button with red background)
- Scenario dropdown with 6 options:
  - MV Only (Mechanically Validated)
  - Mixed MV + ADV
  - PA Only (User Attestation)
  - ADV Only (Exploration)
  - Underdetermined (Open Problem)
  - Custom Input

**Two-Stream Display:**
- Left panel: "EXPLORATION STREAM (NOT AUTHORITY)" - currently shows "Select a scenario or enter custom input."
- Right panel: "AUTHORITY STREAM (BOUND)" - currently shows "Nothing committed yet. Authority stream is empty."

**Documentation Sidebar:**
Links to 5 documents including Scope Lock, How the Demo Explains Itself, Invariants Status

**Quick Reference:**
- FV: Formal proof (ABSTAINED in v0)
- MV: Mechanical validation (arithmetic only)
- PA: User attestation (ABSTAINED)
- ADV: Advisory (excluded from R_t)

---

### "Same Claim, Different Authority" Demo Observations

**Timing:**
The demo completed in approximately **5-8 seconds** (not 90 seconds as the button name suggests). The animation showed results appearing sequentially with brief pauses between each item.

**Animation Sequence:**
1. Button changed to "Running..." state
2. Four claims appeared sequentially in a dark box:
   - Item 1: ADV (Advisory) - "2 + 2 = 4" → ABSTAINED (appeared first)
   - Item 2: PA (Attested) - "2 + 2 = 4" → ABSTAINED (appeared after ~2 seconds)
   - Item 3: MV (Validated) - "2 + 2 = 4" → VERIFIED (appeared after ~4 seconds)
   - Item 4: MV (False) - "3 * 3 = 8" → REFUTED (appeared after ~6 seconds)
3. Button changed to "Run Again" when complete

**Visual Clarity:**
The demo is **visually clear**. Each item shows:
- Trust class label (ADV, PA, MV)
- Claim text in quotes
- Arrow separator
- Outcome in colored text (ABSTAINED in orange, VERIFIED in green, REFUTED in red)
- Explanation text below (e.g., "Excluded from authority stream", "Arithmetic validator confirmed")

**Color Coding:**
- ABSTAINED: Orange text
- VERIFIED: Green text
- REFUTED: Red text

**Summary Statement:**
Below the four items, a summary appears: "Same claim text, different trust class → different outcome. Same trust class, different truth → VERIFIED vs REFUTED."

**Understandability Without Reading Docs:**
The outcomes are **partially understandable** without documentation:
- The color coding (green = good, red = bad, orange = neutral/uncertain) is intuitive
- The explanation text for each outcome provides context
- The summary statement clarifies the point being demonstrated
- However, understanding WHY ADV is excluded or WHY PA returns ABSTAINED requires reading the documentation

**Errors Observed:**
None. The demo ran smoothly without errors.

**Delays:**
No unexpected delays. The animation was smooth and sequential.

**Unexpected Behavior:**
The button name "Run 90-Second Proof" is misleading - the demo takes only 5-8 seconds, not 90 seconds. This creates an expectation mismatch.

**"What does this prove?" Expandable:**
There is an expandable section titled "What does this prove?" below the demo results. The expandable appears to be present but I could not confirm if it opened when clicked (it may require a second click or may be collapsed by default).

---

### Additional Demo Interface Observations

**Two-Stream Display:**
The demo interface clearly separates:
- **EXPLORATION STREAM (NOT AUTHORITY):** Left panel showing "Select a scenario or enter custom input."
- **AUTHORITY STREAM (BOUND):** Right panel showing "Nothing committed yet. Authority stream is empty."

This visual separation reinforces the governance boundary described in the documentation.

**Scenario Dropdown:**
Six predefined scenarios available:
1. MV Only (Mechanically Validated)
2. Mixed MV + ADV
3. PA Only (User Attestation)
4. ADV Only (Exploration)
5. Underdetermined (Open Problem)
6. Custom Input

The scenario selector appears to be the entry point for the full interactive demo flow (as opposed to the "90-Second Proof" which is a standalone demonstration).

**Overall Demo Assessment:**
- **Animation timing:** ~5-8 seconds (not 90 seconds as button name suggests)
- **Visual clarity:** High - color coding, clear labels, explanation text
- **Understandability:** Moderate - basic outcomes are clear, but deeper understanding of WHY requires documentation
- **Errors:** None observed
- **Delays:** None observed
- **Unexpected behavior:** Button name "Run 90-Second Proof" is misleading about timing

---

## Part 5 — Coherence Check

### Do mathledger.ai (archive) and mathledger.ai/demo Feel Like One Coherent System?

**Answer: One coherent epistemic system.**

The archive and demo feel like **intentionally separated but tightly coupled artifacts** that form a coherent whole. They are not "stitched together" - they appear designed from the start to work as a documentation-demonstration pair.

**Evidence of Coherence:**

1. **Consistent Terminology:**
   - Trust classes (FV, MV, PA, ADV) are defined identically in both
   - Outcomes (VERIFIED, REFUTED, ABSTAINED) match exactly
   - "Governance substrate demo, not capability demo" appears in both
   - The tier system (A/B/C) is referenced consistently

2. **Matching Non-Claims:**
   - Archive states: "No Lean/Z3 verifier: FV claims always return ABSTAINED"
   - Demo Quick Reference states: "FV: Formal proof (ABSTAINED in v0)"
   - Archive states: "MV arithmetic validator only"
   - Demo footer states: "MV arithmetic validator only"

3. **Demo Behavior Matches Archive's Non-Claims:**
   - The demo shows PA returning ABSTAINED with explanation "Authority-bearing but no validator" - exactly as the archive predicts
   - The demo shows ADV being "Excluded from authority stream" - exactly as the Explanation document describes
   - The demo shows MV arithmetic validation working for "2+2=4" and "3*3=8" - matching the archive's scope of "arithmetic only"

4. **Cross-Linking:**
   - The demo includes a documentation sidebar linking to the same documents available in the archive
   - The archive prominently features the "Open Interactive Demo" button
   - Both reference the same version numbers and commit hashes

5. **Shared Framing:**
   - Both emphasize limitations over capabilities
   - Both use the phrase "governance substrate"
   - Both explicitly state what the system does NOT do

**Minor Inconsistencies:**

1. **Version number discrepancy:** The demo shows "v0.2.0" while the archive shows "v0.2.1" - this suggests the demo may not have been updated to the latest archive version, though both reference similar commit hashes.

2. **"90-Second Proof" naming:** The button name doesn't match the actual timing (~5-8 seconds), which could indicate the demo was updated but the button label wasn't changed, or the name refers to something other than wall-clock time.

**Verdict:**
The archive and demo form **one coherent epistemic system**. The demo is a live instantiation of what the archive documents. The terminology is consistent, the behavior matches the documented limitations, and the non-claims in the archive are validated by the demo's actual behavior. The separation between them (immutable archive vs. hosted demo) is itself part of the coherent design - the archive documents what the demo does, and the demo proves the archive's claims are testable.

---

## Part 6 — Acquirer / Safety Lead Lens

### What Feels Unusually Rigorous or Novel?

**1. Negative Capability Framing**
The prominence of non-claims and limitations is unprecedented in AI demos. Most systems lead with capabilities; MathLedger leads with "What this version cannot enforce" in red warning text on the homepage. This inverted framing is a genuine novelty in the field.

**2. Tiered Enforcement Transparency**
The Tier A/B/C classification system provides unusual granularity about what is actually enforced versus aspirational. The "How It Can Be Violated Today" column in the Invariants table is something I have never seen in production systems, let alone demos. This level of attack surface transparency is typically reserved for internal security documentation.

**3. Abstention as First-Class Outcome**
Treating ABSTAINED as a legitimate, non-failure outcome - and structurally enforcing that it cannot be silently converted to a claim - is a rigorous design choice. Most AI systems are incentivized to always produce an answer; MathLedger's architecture makes "I don't know" a core feature.

**4. Exploration/Authority Boundary Enforcement**
The structural separation between DraftProposal (with random IDs) and CommittedPartitionSnapshot (with content-derived IDs) is more than documentation - it's enforced in the code with ValueError exceptions. The claim that "exploration identifiers never appear in committed data" is verifiable.

**5. Replay Verification with Explicit Non-Claims**
The evidence pack system provides cryptographic replay verification while explicitly stating it proves "structural integrity, not truth." This distinction between "the audit trail is intact" and "the claims are correct" is philosophically sophisticated and rarely articulated this clearly.

**6. Immutable Versioned Archives**
The epistemic archive concept - where each version is locked with commit hashes, checksums, and explicit scope locks - creates a verifiable historical record. The "Date Locked" timestamps and "This is an epistemic archive. Content is immutable once published" footer establish accountability.

**7. Governance Without Capability**
The framing as a "governance substrate demo, not capability demo" is conceptually novel. Most AI safety work focuses on making capable systems safe; MathLedger demonstrates governance infrastructure before adding capability. This is architecturally backwards from typical AI development, and that's the point.

### What Feels Unfinished, Underspecified, or Missing?

**1. No Failed Verification Examples**
The site shows VERIFIED, REFUTED, and ABSTAINED outcomes, but doesn't demonstrate what happens when verification infrastructure itself fails (e.g., validator crashes, hash computation errors, Byzantine failures). A "failure modes" page would strengthen credibility.

**2. Threat Model Absence**
There is no explicit threat model. Who is the adversary? What attacks is the system designed to resist? What attacks is it explicitly NOT designed to resist? The Invariants page shows "How It Can Be Violated Today" but doesn't frame this as adversarial threat modeling.

**3. Missing "For Auditors" Entry Point Clarity**
While there's a "5-minute auditor checklist" link, it's not prominent enough. An auditor landing on the homepage would need to infer that this is the right starting point. A clearer "If you're auditing this system, start here" banner would help.

**4. No Comparison to Existing Standards**
The site doesn't position MathLedger relative to existing audit standards (SOC 2, ISO 27001), AI governance frameworks (NIST AI RMF), or formal verification approaches (Coq, Isabelle). This makes it harder to assess whether MathLedger is complementary, competitive, or orthogonal to existing approaches.

**5. Scalability and Performance Claims Absent**
There are no claims about performance, throughput, or scalability. Can this handle 1000 claims? 1 million? Is there a performance model? While this is consistent with "governance substrate only," it leaves open questions about production viability.

**6. Multi-Party Scenarios Underexplored**
The demo shows single-user flows. What happens when multiple parties commit conflicting claims? How are disputes resolved? The UVIL (User-Verified Input Loop) suggests human-in-the-loop, but multi-stakeholder scenarios aren't demonstrated.

**7. Integration Guidance Missing**
There's no "How to Integrate MathLedger" guide. If I wanted to use this in my organization, what would that look like? Is it a library, a service, a protocol? The local execution instructions exist but aren't framed as integration guidance.

**8. Economic and Incentive Model Absent**
There's no discussion of incentives. Why would users commit claims? Why would validators participate? In production, governance systems need incentive alignment, but this is not addressed (which is fine for v0, but should be acknowledged as future work).

### What Would I Want to See Next? (Concrete Requests)

**1. A Live Example of a Failed Verification**
Show a case where the verification infrastructure itself fails (not just ABSTAINED, but an actual system error). Demonstrate how the system handles and reports infrastructure failures. Include this in the fixtures with expected error states.

**2. A Threat Model Page**
Create a document titled "Threat Model and Attack Surface" that explicitly lists:
- Adversaries the system is designed to resist (e.g., malicious validators, data tampering)
- Adversaries the system is NOT designed to resist (e.g., compromised hardware, social engineering)
- Attack vectors and mitigations for each Tier A invariant
- Explicitly out-of-scope threats

**3. A "For Auditors" Landing Page**
Create a dedicated entry point at `/for-auditors` that provides:
- 30-second summary of what to audit
- Links to the 5 most critical documents in priority order
- Checklist of verification steps with estimated time for each
- Expected outputs for each verification step
- Contact information for questions

**4. A "Comparison to Existing Approaches" Document**
Add a page that positions MathLedger relative to:
- Traditional audit frameworks (SOC 2, ISO 27001)
- AI governance frameworks (NIST AI RMF, EU AI Act)
- Formal verification systems (Lean, Coq, Z3)
- Blockchain/distributed ledger approaches
Explain what MathLedger does that these don't, and what these do that MathLedger doesn't.

**5. A "Failure Modes and Limitations" Page**
Create a comprehensive document listing:
- Known failure modes (with examples)
- Edge cases not covered by current validators
- Scalability limits (if any)
- Performance characteristics (latency, throughput)
- Degradation behavior under load or attack

**6. Evidence Pack Tamper Detection Demo**
Add an interactive demo that:
- Shows a valid evidence pack with PASS verification
- Allows the user to modify a field
- Re-runs verification and shows FAIL with specific diff
This would make the tamper detection claim tangible.

**7. Multi-Stakeholder Scenario**
Add a fixture or demo showing:
- Two users committing conflicting claims about the same fact
- How the system records both without choosing a winner
- How the attestation structure preserves both perspectives
This would demonstrate governance in contested scenarios.

**8. Integration Example**
Provide a concrete example of integrating MathLedger into an existing system:
- Sample code for a Python application
- API documentation (if applicable)
- Deployment guide
- Monitoring and observability recommendations

---

## Part 7 — Verdict

### Core Value Proposition (One Sentence)

MathLedger provides a cryptographically enforced governance substrate that separates AI system exploration from authority-bearing claims, making the boundary between "what the system suggested" and "what was committed as justified" structurally verifiable and replayable, with abstention as a first-class outcome when verification cannot be established.

### Single Biggest Improvement for Credibility (One Sentence)

Add a prominent threat model page that explicitly names the adversaries the system is designed to resist and those it is not, with concrete attack scenarios and corresponding Tier A invariant protections, because the current documentation demonstrates unusual rigor in showing limitations but stops short of framing those limitations as adversarial threat modeling, which is what auditors and acquirers need to assess whether the governance substrate is fit for their threat environment.

---

## Summary of Key Findings

**Strengths:**
- Unprecedented transparency about limitations and non-claims
- Rigorous tier system (A/B/C) with explicit enforcement levels
- Structural enforcement of exploration/authority boundary
- Abstention treated as first-class outcome, not failure
- Immutable versioned archives with cryptographic verification
- Terminology consistency between documentation and demo
- Evidence pack replay verification with explicit scope

**Weaknesses:**
- No explicit threat model or adversary framing
- Missing failed verification examples (infrastructure failures, not just ABSTAINED)
- Target audience ambiguity (assumes high technical sophistication)
- "90-Second Proof" button name misleading (actually ~5-8 seconds)
- No comparison to existing audit/governance frameworks
- Integration guidance absent
- Multi-stakeholder scenarios underexplored

**Overall Assessment:**
MathLedger represents a genuinely novel approach to AI governance by prioritizing legibility and structural enforcement over capability. The negative framing (leading with limitations) and tiered transparency (A/B/C invariants) are unprecedented in AI demos. The system is coherent across archive and demo, with terminology and behavior matching documented claims. However, the value proposition requires significant inference - it's not immediately clear why governance without capability matters or who the target user is. The addition of a threat model and "For Auditors" entry point would significantly increase credibility and usability for the stated audience (safety leads, auditors, acquirers).

---

