> **Audit target:** v0.2.2 archive
>
> `/demo` always serves CURRENT version (v0.2.3 at time of audit)
>
> This is expected; historical demos are not hosted per-version.

---

# EPISTEMIC INTEGRITY AUDIT: MathLedger v0.2.2
## Field Manual vs Archive vs Demo


**Audit Date:** 2026-01-03
**Auditor:** Hostile External (Epistemic Focus)
**Goal:** Identify conceptual contradictions, spec ambiguities, invariant gaps


---


## PHASE 1: FIELD MANUAL (Scope Lock) - Key Claims


### What v0 Demonstrates (POSITIVE CLAIMS)


1. **UVIL (User-Verified Input Loop):** Human binding mechanism for authority
2. **Trust Classes:** FV, MV, PA (authority-bearing) vs ADV (exploration-only)
3. **Dual Attestation Roots:** U_t (UI), R_t (reasoning), H_t (composite)
4. **Determinism:** Same inputs produce same outputs, replayable
5. **Exploration/Authority Boundary:** DraftProposal never enters hash-committed paths


### What v0 Does NOT Demonstrate (NEGATIVE CLAIMS)


1. **No RFL learning loop** - No curriculum, no policy, no uplift
2. **No multi-model arena** - Single template partitioner
3. **No agent tools** - No code execution, no sandbox
4. **No real verifier** - No Lean, no Z3, no mechanical proof checking
5. **No production auth** - No user accounts, no API keys
6. **No persistence** - In-memory only
7. **No long-running agents** - Synchronous request/response only


### Critical Framing


> "v0 is a governance substrate demo, not a capability demo."


> "It does not show:
> - That the system is intelligent
> - That the system is aligned
> - That the system is safe
> - That verification works (v0 has no verifier)"


### Terminology Constraints


**Outcome Values:**
- **VERIFIED:** MV claim parsed AND arithmetic confirmed (e.g., "2 + 2 = 4")
- **REFUTED:** MV claim parsed AND arithmetic failed (e.g., "2 + 2 = 5")
- **ABSTAINED:** Cannot mechanically verify (PA, FV, unparseable MV, ADV-only)


**Authority-Bearing:**
- FV, MV, PA: authority-bearing (enter R_t)
- ADV: exploration-only (excluded from R_t)


**CRITICAL CONSTRAINT:**
> "v0 does NOT claim that authority-bearing means 'verified':
> PA is authority-bearing because a human attests, not because the system verified
> In v0, no claims are mechanically verified"


**WAIT - CONTRADICTION DETECTED:**


The Scope Lock says:
> "In v0, no claims are mechanically verified"


But the Outcome Values table says:
> "VERIFIED: MV claim parsed AND arithmetic confirmed (e.g., '2 + 2 = 4')"


**Question:** If "no claims are mechanically verified," how can MV claims return VERIFIED?


**Possible interpretation:** The "arithmetic validator" is NOT considered "mechanical verification" in the FV sense (no Lean/Z3). It's just string parsing + Python arithmetic.


**But this is AMBIGUOUS.** An auditor would ask: "Is the arithmetic validator mechanical verification or not?"


---


## INVARIANTS VERIFIED (from Scope Lock)


**T1:** proposal_id never appears in hash-committed payloads
**T2:** ADV claims excluded from R_t computation
**T3:** H_t = SHA256(R_t || U_t) deterministic
**T4:** Double-commit of proposal_id returns 409 Conflict
**T5:** /run_verification rejects raw proposal_id
**T6:** Outcomes are VERIFIED/REFUTED only via MV arithmetic validator; else ABSTAINED
**T7:** PA claims show "authority-bearing but not mechanically verified"
**T8:** UI clearly separates Exploration Stream from Authority Stream
**T9:** Evidence pack contains ONLY minimum for replay
**T10:** Replay verification uses SAME code paths as live
**T11:** No external API calls required for replay verification
**T12:** Tampered evidence pack correctly returns FAIL with diff
**T13:** UI_COPY dictionary contains all required self-explanation keys
**T14:** No capability claims ("safe", "aligned", "intelligent") in UI copy
**T15:** ABSTAINED explanation mentions "first-class outcome"
**T16:** Abstention preservation gate enforced on all R_t entries


---


## TIER COUNTS (from Scope Lock header)


- **Tier A (enforced):** 10
- **Tier B (logged):** 1
- **Tier C (aspirational):** 3


**WAIT - CONTRADICTION DETECTED:**


Scope Lock header says:
- Tier A: 10
- Tier B: 1
- Tier C: 3


But v0.2.0 release notes say:
- Tier A: 9
- Tier B: 1
- Tier C: 3


**Question:** Which is correct? Did Tier A increase from 9 to 10 in v0.2.1 or v0.2.2?


**This is a SPEC AMBIGUITY.** Need to check Invariants page.


---


## FORBIDDEN ITERATIONS (from Scope Lock)


Do not add to v0:
- Model selection or switching
- Tool execution or sandboxing
- External API calls in demo flow
- Persistence beyond session
- **Claims of verification when no verifier exists**
- **Claims of safety, alignment, or intelligence**


**Key constraint:** "Claims of verification when no verifier exists"


**But T6 says:** "Outcomes are VERIFIED/REFUTED only via MV arithmetic validator"


**Question:** Does the arithmetic validator count as a "verifier"? If not, why does MV return VERIFIED?


---


## PRESSURE POINTS IDENTIFIED SO FAR


1. **"No claims are mechanically verified" vs "VERIFIED outcome for MV"** - Spec ambiguity
2. **Tier A count mismatch** (10 vs 9) - Interface issue
3. **"No verifier" vs "arithmetic validator"** - Terminology gap


Continuing to Invariants page...


---




## PHASE 1 (CONTINUED): INVARIANTS PAGE


### Tier Counts (from Invariants page header)


**Header says:**
- Tier A (enforced): 10
- Tier B (logged): 1
- Tier C (aspirational): 3


**Body says:**
- "Tier A: Enforced (9 invariants)"
- Lists 9 invariants (1-9)
- Tier B: 1 invariant (MV Validator Correctness)
- Tier C: 3 invariants (FV, Multi-Model, RFL)


**CONTRADICTION #1: TIER A COUNT MISMATCH**


**Header:** 10
**Body:** 9
**Severity:** Interface issue (confusing but not epistemic)


**Hypothesis:** The 10th Tier A invariant may be "Audit Surface Version Field" mentioned in Scope Lock but not listed in Invariants page body.


---


### Invariants Table - "How It Can Be Violated Today"


This is **EXCEPTIONAL TRANSPARENCY**. The table explicitly lists how each invariant can be violated.


**Key findings:**


| Invariant | Tier | How It Can Be Violated Today |
|-----------|------|------------------------------|
| Canonicalization | A | Impossible without golden test failure |
| H_t computation | A | Impossible - structural code constraint |
| ADV excluded | A | Impossible - raises ValueError |
| Content-derived IDs | A | Impossible - uses SHA256 |
| Replay same paths | A | Impossible - same imports |
| Double-commit 409 | A | Impossible - set check |
| No Silent Authority | A | Impossible - require_epoch_root() gate mandatory |
| Trust-Class Monotonicity | A | Impossible - gate mandatory at commit |
| Abstention Preservation | A | Impossible - gate mandatory |
| **MV Validator Correctness** | **B** | **Edge cases in arithmetic parsing; non-arithmetic claims** |


**PRESSURE POINT #1: MV VALIDATOR CORRECTNESS IS TIER B**


**Implication:** The MV validator (which produces VERIFIED/REFUTED outcomes) is **NOT cryptographically enforced**. It's logged but can be violated.


**Contradiction with Scope Lock:**


Scope Lock says:
> "Outcomes are VERIFIED/REFUTED only via MV arithmetic validator; else ABSTAINED" (T6)


This sounds like Tier A (enforced). But Invariants page says MV Validator Correctness is **Tier B** (logged, not prevented).


**Question for acquisition committee:**


If MV Validator Correctness is Tier B, then:
1. Can a buggy validator produce VERIFIED for an incorrect claim?
2. Would this be detectable in replay?
3. Is the "VERIFIED" outcome trustworthy?


**Answer from Invariants page:**
> "Violation Path: Edge cases (overflow, division by zero, floating point)"
> "Detection: Logged validation_outcome with parsed values"


**So:** Yes, a buggy validator CAN produce wrong VERIFIED outcomes. Detection is via logs (replay-visible), not prevention.


**This is honest but undermines the "VERIFIED" claim.**


---


### Tier C: Aspirational (3 invariants)


1. **FV Mechanical Verification**
   - FM Reference: §1.5
   - Current State: FV trust class exists; no Lean/Z3 verifier
   - **Status: All FV claims return ABSTAINED**


2. **Multi-Model Consensus**
   - FM Reference: §10 (uncharted surface area)
   - Current State: Single template partitioner
   - Status: Not in v0 scope


3. **RFL Integration**
   - FM Reference: §7-8
   - Current State: No learning loop
   - Status: Not in v0 scope


**PRESSURE POINT #2: FV ALWAYS RETURNS ABSTAINED**


**Scope Lock says:**
> "No Lean/Z3 verifier: FV claims always return ABSTAINED"


**Invariants page confirms:**
> "All FV claims return ABSTAINED"


**Question:** If FV always returns ABSTAINED, why does it exist as a trust class?


**Possible answer:** FV is a **placeholder** for Phase II. It's in the schema but not functional in v0.


**But this creates ambiguity:** An auditor seeing "FV" in the demo might assume it works.


---


### "What this version cannot enforce" (header box)


- No Lean/Z3 verifier: FV claims always return ABSTAINED
- Single template partitioner: no multi-model consensus
- No learning loop: RFL not active
- **MV edge cases: overflow, float precision not fully covered**


**PRESSURE POINT #3: MV EDGE CASES NOT FULLY COVERED**


**This is the same issue as "MV Validator Correctness is Tier B".**


**The header explicitly says:**
> "MV edge cases: overflow, float precision not fully covered"


**Implication:** MV's VERIFIED outcome is **conditional** on the claim being within the validator's coverage.


**Question:** Does the demo or UI explain this limitation?


Need to check demo and UI copy.


---


## SUMMARY OF PRESSURE POINTS SO FAR


1. **Tier A count mismatch** (10 vs 9) - Interface issue
2. **MV Validator Correctness is Tier B, not A** - Invariant gap
3. **MV edge cases not fully covered** - Spec ambiguity (what counts as "covered"?)
4. **FV always returns ABSTAINED** - Interface issue (why does it exist?)
5. **"No claims are mechanically verified" vs "VERIFIED outcome for MV"** - Terminology gap


Continuing to Explanation page and demo...


---




## PHASE 2: EXPLANATION PAGE - CRITICAL CONTRADICTION FOUND


### Outcome Definitions (from Explanation page)


> "The demo produces three possible outcomes: VERIFIED, REFUTED, or ABSTAINED.
>
> **VERIFIED means the system found a machine-checkable proof that the claim holds.**
> **REFUTED means the system found a machine-checkable proof that the claim does not hold.**
> ABSTAINED means the system did not find either."


**THIS IS A DIRECT CONTRADICTION WITH SCOPE LOCK.**


**Scope Lock says:**
> "In v0, no claims are mechanically verified"
> "No Lean/Z3 verifier: FV claims always return ABSTAINED"
> "What v0 Does NOT Demonstrate: Real verifier: No Lean, no Z3, no mechanical proof checking"


**Explanation page says:**
> "VERIFIED means the system found a machine-checkable proof"


**PRESSURE POINT #6: VERIFIED OUTCOME CONTRADICTS "NO VERIFIER" CLAIM**


**Classification:** SPEC AMBIGUITY / INVARIANT GAP


**Analysis:**


The Explanation page claims VERIFIED means "machine-checkable proof." But:
1. Scope Lock says "no mechanical proof checking"
2. Invariants page says MV Validator Correctness is Tier B (not enforced)
3. Header says "MV edge cases: overflow, float precision not fully covered"


**Possible interpretations:**


**Interpretation A (Generous):**
- "Machine-checkable proof" means "Python arithmetic check" (not formal verification)
- The MV validator checks `a op b = c` by parsing and evaluating in Python
- This is "machine-checkable" but not "formal verification" (Lean/Z3)
- The Scope Lock's "no mechanical verification" refers to formal verification only


**Interpretation B (Strict):**
- "Machine-checkable proof" implies formal verification
- The Explanation page is overclaiming
- The MV validator is just string parsing + Python eval
- This should be called "arithmetic validation," not "machine-checkable proof"


**Interpretation C (Hostile):**
- The terminology is deliberately ambiguous
- "Machine-checkable proof" sounds impressive
- But it's just Python arithmetic on a narrow domain
- This is marketing language, not technical precision


**Acquisition committee question:**


"Does VERIFIED mean 'formally verified' or 'Python arithmetic check'?"


**Answer from Field Manual:** Need to check if FM defines "machine-checkable proof."


---


### What Replay Does NOT Prove (from Explanation page)


> "Replay verification proves structural integrity, not truth. A PASS result means: the recorded hashes match what the inputs produce. It does not mean: the claims are correct, the verification was sound, or the system behaved safely. The evidence pack proves the audit trail is intact. It proves nothing about what that trail represents."


**THIS IS EXCELLENT NEGATIVE FRAMING.**


The Explanation page explicitly says replay verification does NOT prove:
- Claims are correct
- Verification was sound
- System behaved safely


**This is consistent with Scope Lock's negative framing.**


**But it creates a tension with the VERIFIED outcome:**


If "VERIFIED means machine-checkable proof," but "replay does not prove verification was sound," then what does VERIFIED actually mean?


**Possible answer:** VERIFIED means "the system claimed to verify it," not "it was actually verified."


**But this is confusing.**


---


### "What This Demo Refuses to Claim" (from Explanation page)


> "This demo does not claim the system is aligned with human values.
> This demo does not claim the system is intelligent.
> This demo does not claim the system is safe.
> This demo does not claim the system will behave well in novel situations.
>
> This demo claims one thing: the system's governance is legible."


**EXCELLENT. This is consistent with Scope Lock.**


---


### "How to Read the Demo as Evidence" (from Explanation page)


**Justified conclusions:**
- The system separates exploration from authority at the data model level.
- The system produces deterministic attestations for committed content.
- The system refuses to include ADV content in authority-bearing paths.
- The system stops and reports ABSTAINED when it cannot verify or refute.
- The human's actions and the system's reasoning are recorded in separate, verifiable structures.


**Explicitly not justified:**
- The system will behave correctly in all cases.
- The system's verification is complete or sound in any formal sense.
- The system's abstention is always appropriate.
- The governance structure prevents all...


**This is EXCEPTIONAL transparency.**


**But note:** "The system's verification is complete or sound in any formal sense" is explicitly NOT justified.


**This contradicts the VERIFIED outcome definition:** "VERIFIED means the system found a machine-checkable proof."


**If verification is not sound in any formal sense, how can VERIFIED mean "machine-checkable proof"?**


---


## SUMMARY OF CONTRADICTIONS SO FAR


1. **Tier A count mismatch** (10 vs 9) - Interface issue
2. **MV Validator Correctness is Tier B** - Invariant gap
3. **MV edge cases not fully covered** - Spec ambiguity
4. **FV always returns ABSTAINED** - Interface issue
5. **"No claims are mechanically verified" vs "VERIFIED outcome for MV"** - Terminology gap
6. **"VERIFIED means machine-checkable proof" vs "verification not sound in any formal sense"** - SPEC AMBIGUITY / CONTRADICTION


**PRESSURE POINT #6 is the most serious.**


An acquisition committee would ask:


"If VERIFIED does not mean 'formally verified,' what does it mean?"


"If verification is not sound, why call it VERIFIED?"


"Is this a governance demo or a verification demo?"


---


## NEXT STEPS


1. Check Field Manual for definition of "machine-checkable proof"
2. Check demo UI to see how VERIFIED is presented
3. Check if demo UI explains MV validator limitations


---




## PHASE 3: INTERACTIVE DEMO - UI ANALYSIS


### CRITICAL FINDING: DEMO VERSION MISMATCH


**Archive version being audited:** v0.2.2
**Demo version:** v0.2.3
**Demo tag:** v0.2.3-audit-path-freshness
**Demo commit:** 674bcd16104f


**Archive commit (v0.2.2):** 27a94c8a5813


**PRESSURE POINT #7: DEMO RUNS DIFFERENT VERSION THAN ARCHIVE**


**Classification:** INTERFACE ISSUE / DEPLOYMENT GAP


**Impact:** An auditor reviewing v0.2.2 archive cannot test the demo that matches the archive. The demo is running v0.2.3, which may have different behavior.


**This was flagged in previous audits as a BLOCKING issue.** Still not fixed.


---


### Trust Class Descriptions (from demo UI sidebar)


**FV: Formal proof**
→ ABSTAINED in v0 (no prover)


**MV: Mechanical validation**
→ **Arithmetic only in v0**


**PA: User attestation**
→ ABSTAINED (no verifier)


**ADV: Advisory**
→ EXCLUDED FROM R_t


---


### KEY OBSERVATION: MV SAYS "MECHANICAL VALIDATION" NOT "MACHINE-CHECKABLE PROOF"


**Demo UI says:** "MV: Mechanical validation → Arithmetic only in v0"


**Explanation page says:** "VERIFIED means the system found a machine-checkable proof"


**PRESSURE POINT #8: TERMINOLOGY INCONSISTENCY**


**Classification:** SPEC AMBIGUITY


**Analysis:**


The demo UI uses the term **"mechanical validation"** for MV.


The Explanation page uses the term **"machine-checkable proof"** for VERIFIED.


These are NOT the same thing:
- "Mechanical validation" suggests automated checking (could be simple parsing)
- "Machine-checkable proof" suggests formal verification (Lean/Z3 level)


**The demo UI is more accurate:** "Arithmetic only in v0" makes it clear this is NOT formal verification.


**The Explanation page is overclaiming:** "machine-checkable proof" is too strong for "arithmetic parsing + Python eval."


---


### Demo Framing (from UI header)


> "The system does not decide what is true. It decides what is justified under a declared verification route."


**EXCELLENT FRAMING.** This is consistent with Scope Lock.


> "This demo will stop more often than you expect. It reports what it cannot verify."


**GOOD. Negative framing.**


> "If you are looking for a system that always has an answer, this demo is not it."


**EXCELLENT. Sets expectations correctly.**


---


### Demo Footer


> "v0.2.3 (v0.2.3-audit-path-freshness) | Governance substrate only | MV arithmetic validator only | Scope Lock"


**Key phrase:** "MV arithmetic validator only"


**This is the most accurate description of what MV does.**


**But it contradicts Explanation page's "machine-checkable proof."**


---


### "What does 'justified' mean?" (expandable section)


Need to expand this to see what it says...


---


## SUMMARY OF FINDINGS SO FAR


### CONTRADICTIONS


1. **"No claims are mechanically verified" (Scope Lock) vs "VERIFIED means machine-checkable proof" (Explanation)** - MAJOR
2. **"Verification not sound in any formal sense" (Explanation) vs "VERIFIED means machine-checkable proof" (Explanation)** - MAJOR
3. **Tier A count: 10 (header) vs 9 (body)** - MINOR


### SPEC AMBIGUITIES


4. **"Machine-checkable proof" vs "mechanical validation" vs "arithmetic validator"** - What does VERIFIED actually mean?
5. **MV Validator Correctness is Tier B** - Can produce wrong VERIFIED outcomes
6. **"MV edge cases not fully covered"** - What is covered? What isn't?


### INTERFACE ISSUES


7. **Demo version mismatch** (v0.2.3 vs v0.2.2 archive) - BLOCKING for audit
8. **FV exists but always returns ABSTAINED** - Why is it in the schema?


### INVARIANT GAPS


9. **MV Validator Correctness is Tier B, not A** - VERIFIED outcome is not cryptographically enforced


---


## NEXT STEPS


1. Run MV scenario to see how VERIFIED is presented in demo
2. Check if demo explains MV validator limitations
3. Expand "What does 'justified' mean?" section
4. Check "What Gets Rejected" tests


---




## PHASE 3 (CONTINUED): DEMO INTERACTION LIMITED


**Note:** The demo dropdown does not appear to be responding to selection. This may be a UI issue or the demo may require additional interaction steps.


**However, sufficient information has been gathered from:**
1. Field Manual (Scope Lock, Invariants, Explanation)
2. Demo UI framing and trust class descriptions
3. Archive documentation


**Moving to compilation of findings.**


---


## PHASE 4: EPISTEMIC CONTRADICTIONS IDENTIFIED


### MAJOR CONTRADICTIONS


#### 1. "VERIFIED" TERMINOLOGY OVERREACH


**Scope Lock says:**
> "In v0, no claims are mechanically verified"
> "No Lean/Z3 verifier: FV claims always return ABSTAINED"
> "What v0 Does NOT Demonstrate: Real verifier: No Lean, no Z3, no mechanical proof checking"


**Explanation page says:**
> "VERIFIED means the system found a machine-checkable proof that the claim holds."


**Demo UI says:**
> "MV: Mechanical validation → Arithmetic only in v0"


**Analysis:**


The term **"machine-checkable proof"** (Explanation page) is **too strong** for what MV actually does (arithmetic parsing + Python eval).


The demo UI is more accurate: "Arithmetic only in v0" makes it clear this is NOT formal verification.


**Recommended terminology:**
- "Arithmetic validation" (accurate)
- "Mechanical validation" (acceptable)
- "Machine-checkable proof" (overclaims)


**Why this matters:**


"Machine-checkable proof" has a specific meaning in formal methods: a proof that can be checked by a proof assistant (Lean, Coq, Z3). MV's arithmetic validator is NOT this.


An acquisition committee would ask: "If you claim 'machine-checkable proof,' why do you also say 'no mechanical proof checking'?"


**Classification:** SPEC AMBIGUITY / TERMINOLOGY GAP


---


#### 2. "VERIFICATION NOT SOUND" VS "VERIFIED OUTCOME"


**Explanation page says:**
> "Replay verification proves structural integrity, not truth."
> "Explicitly not justified: The system's verification is complete or sound in any formal sense."


**But also says:**
> "VERIFIED means the system found a machine-checkable proof."


**Analysis:**


If "verification is not sound in any formal sense," then the VERIFIED outcome should not be called "VERIFIED."


**Recommended alternatives:**
- "VALIDATED" (checked by validator, not formally verified)
- "CONFIRMED" (arithmetic check passed)
- "ACCEPTED" (meets MV criteria)


**Why this matters:**


"VERIFIED" implies correctness. But if "verification is not sound," then VERIFIED claims can be wrong.


The Invariants page confirms this: MV Validator Correctness is Tier B (logged, not enforced). Edge cases can produce wrong VERIFIED outcomes.


**Classification:** SPEC AMBIGUITY / INVARIANT GAP


---


### SPEC AMBIGUITIES


#### 3. MV VALIDATOR COVERAGE UNDEFINED


**Invariants page header says:**
> "MV edge cases: overflow, float precision not fully covered"


**Questions:**
- What IS covered?
- What ISN'T covered?
- How does an auditor know if a claim is within coverage?


**No specification found.**


**Impact:** An auditor cannot determine if a VERIFIED outcome is trustworthy without knowing the validator's coverage.


**Classification:** SPEC AMBIGUITY


---


#### 4. TIER A COUNT MISMATCH


**Header:** Tier A (enforced): 10
**Body:** Tier A: Enforced (9 invariants)


**Lists 9 invariants (1-9).**


**Hypothesis:** The 10th may be "Audit Surface Version Field" mentioned in Scope Lock T16 but not listed in Invariants body.


**Classification:** INTERFACE ISSUE (minor)


---


### INTERFACE ISSUES


#### 5. DEMO VERSION MISMATCH (RECURRING)


**Archive:** v0.2.2 (commit 27a94c8a5813)
**Demo:** v0.2.3 (commit 674bcd16104f)


**This was flagged as BLOCKING in previous audits. Still not fixed.**


**Impact:** Cannot verify archive claims against demo behavior.


**Classification:** DEPLOYMENT GAP (blocking for audit)


---


#### 6. FV EXISTS BUT ALWAYS ABSTAINS


**Scope Lock:** "No Lean/Z3 verifier: FV claims always return ABSTAINED"
**Invariants:** "All FV claims return ABSTAINED"
**Demo UI:** "FV: Formal proof → ABSTAINED in v0 (no prover)"


**Question:** Why does FV exist in the schema if it always returns ABSTAINED?


**Possible answer:** Placeholder for Phase II.


**But:** This creates ambiguity. A user seeing "FV" might assume it works.


**Recommendation:** Either:
1. Remove FV from v0 schema (add in Phase II when it works)
2. Add explicit warning: "FV is a placeholder for Phase II. It does not function in v0."


**Classification:** INTERFACE ISSUE (confusing but not blocking)


---


### INVARIANT GAPS


#### 7. MV VALIDATOR CORRECTNESS IS TIER B


**Invariants page:**
> "MV Validator Correctness: Tier B (Logged and replay-visible. Violation is detectable but not prevented.)"
> "Violation Path: Edge cases (overflow, division by zero, floating point)"


**Impact:**


The VERIFIED outcome is NOT cryptographically enforced. A buggy validator can produce VERIFIED for incorrect claims.


Detection is via logs (replay-visible), not prevention.


**This undermines the VERIFIED claim.**


**Classification:** INVARIANT GAP


---


## PHASE 5: CLASSIFICATION AND RANKING


### PRESSURE POINTS (RANKED BY SEVERITY)


---
