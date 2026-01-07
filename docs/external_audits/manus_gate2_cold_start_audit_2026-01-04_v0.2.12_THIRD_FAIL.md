\# Gate 2: Cold-Start Audit — v0.2.12 (Clarified Version Semantics)



\*\*Date:\*\* 2026-01-04  

\*\*Role:\*\* Epistemic Gatekeeper (Cold External Auditor)  

\*\*Constraints:\*\* Zero context, zero URL guessing, no repo access, no assumptions



\*\*Clarification:\*\* Documentation versions (claim sets) may advance independently of demo capability versions. Demo is expected to run latest demo-capability version, which may be lower than CURRENT documentation version. This should be explicitly documented in FOR\_AUDITORS.



---



\## Phase 1: Identify CURRENT Documentation Version





\### Step 1: Navigate to https://mathledger.ai/



\*\*Result:\*\* ✅ Redirects to /v0.2.12/



\*\*Landing Page Observations:\*\*

\- Status: LOCKED (see /versions/ for current status)

\- Tag: v0.2.12-versioning-doctrine

\- Commit: 15cc70f60732

\- Locked: 2026-01-05



\*\*Key Claim on Archive Page:\*\*

> "This archive is immutable. The hosted demo at /demo/ is the live instantiation of this same version."



\*\*Critical Observation:\*\* Archive page claims demo is "live instantiation of this same version" (v0.2.12). This contradicts the clarification that demo may run a different version. Need to verify if FOR\_AUDITORS explicitly documents version independence.



\*\*Next Action:\*\* Click /versions/ to confirm CURRENT status, then navigate to FOR\_AUDITORS to check for explicit documentation of demo version independence.





\### Step 2: Confirm CURRENT via /versions/



\*\*URL:\*\* https://mathledger.ai/versions/



\*\*Result:\*\* ✅ v0.2.12 confirmed as CURRENT



\*\*Canonical Registry Shows:\*\*

\- v0.2.10 | Demo | SUPERSEDED BY V0.2.11 | 2026-01-04 | 55d12f4

\- v0.2.11 | Demo | SUPERSEDED BY V0.2.12 | 2026-01-05 | f6f142f

\- v0.2.12 | Demo | \*\*CURRENT\*\* | 2026-01-05 | 15cc70f



\*\*Phase 1 Assessment:\*\* ✅ PASS (v0.2.12 confirmed as CURRENT documentation version)



---



\## Phase 2: Verify FOR\_AUDITORS Explicitly Documents Demo Version Independence



\*\*Critical Question:\*\* Does FOR\_AUDITORS explicitly state that demo may run a different version than the documentation version?



\*\*Archive Page Claim:\*\* "The hosted demo at /demo/ is the live instantiation of this same version."



This claim suggests demo should match documentation version. Need to check if FOR\_AUDITORS clarifies or contradicts this.





\## Phase 2: Verify FOR\_AUDITORS Explicitly Documents Demo Version Independence



\*\*URL:\*\* https://mathledger.ai/v0.2.12/docs/for-auditors/



\*\*Result:\*\* ❌ FAIL (version coherence is contradictory, not understandable)



\*\*FOR\_AUDITORS Step 1.3 Instructs:\*\*

> "Verify the version banner shows v0.2.12-versioning-doctrine"



\*\*Critical Finding:\*\* FOR\_AUDITORS does \*\*NOT\*\* explicitly state that demo may run a different version than documentation version. Instead, it instructs auditors to verify the demo banner shows v0.2.12-versioning-doctrine.



\*\*Contradictions Detected:\*\*



1\. \*\*Archive Page Claims:\*\* "The hosted demo at /demo/ is the live instantiation of this same version." (v0.2.12)

2\. \*\*FOR\_AUDITORS Step 1.3:\*\* "Verify the version banner shows v0.2.12-versioning-doctrine"

3\. \*\*Clarification Provided:\*\* "Documentation versions may advance independently of demo capability versions. Demo is expected to run latest demo-capability version, which may be lower than CURRENT documentation version."



\*\*Analysis:\*\* A cold auditor reading FOR\_AUDITORS with zero context will:

1\. Read Step 1.3 expecting demo to show v0.2.12-versioning-doctrine

2\. Navigate to /demo/

3\. See v0.2.10-demo-reliability

4\. \*\*Conclude Step 1.3 has failed\*\*



There is \*\*no explicit documentation\*\* in FOR\_AUDITORS explaining that demo version independence is intentional. The clarification contradicts both the archive page claim and the FOR\_AUDITORS Step 1.3 instruction.



---



\## GATE 2 VERDICT



\*\*GATE 2: FAIL — Version coherence is contradictory, not understandable\*\*



\*\*Reason:\*\* FOR\_AUDITORS Step 1.3 instructs auditors to verify demo banner shows v0.2.12-versioning-doctrine, but demo shows v0.2.10-demo-reliability. The clarification that "documentation versions may advance independently of demo capability versions" is \*\*not documented anywhere a cold auditor can find it\*\*. This creates an epistemic contradiction: the site claims demo is "live instantiation of this same version" while simultaneously running a different version, with no explanation provided to auditors.



\*\*For version coherence to be "understandable and non-contradictory," FOR\_AUDITORS must explicitly state:\*\*

\- Demo may run a different version than documentation version

\- This is intentional, not a failure

\- How to verify which demo version is running

\- How to map demo version to documentation version



\*\*None of this is present in FOR\_AUDITORS or any other clickable documentation.\*\*





