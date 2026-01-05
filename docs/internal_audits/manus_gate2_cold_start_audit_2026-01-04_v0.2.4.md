\# Epistemic Gatekeeper Audit — Final Verdict

\## MathLedger v0.2.4 (CURRENT)



\*\*Auditor:\*\* Cold external (zero context, zero prior knowledge)  

\*\*Date:\*\* 2026-01-04  

\*\*Entry Point:\*\* https://mathledger.ai/  

\*\*Goal:\*\* Determine if acquisition committee can execute audit path without guessing URLs or asking questions



---



\## VERDICT: \*\*FAIL\*\*



\*\*Reason:\*\* Multiple BLOCKING epistemic contradictions prevent coherent audit execution. An acquisition committee cannot determine which version they are auditing or whether the verifier is working correctly.



---



\## BLOCKING EPISTEMIC CONTRADICTIONS



\### BLOCKING-1: Commit Hash Mismatch Between /versions/ and Archive Page



\*\*Evidence:\*\*



| Source | Version | Commit | Tag |

|--------|---------|--------|-----|

| /versions/ table (canonical) | v0.2.4 | \*\*f58ff66\*\* | - |

| v0.2.4 archive page (clicked from /versions/) | v0.2.4 | \*\*9bfca919d07d\*\* | v0.2.4-verifier-syntax-fix |

| mathledger.ai root (redirect) | v0.2.4 | \*\*f58ff661e9b1\*\* | v0.2.4-verifier-merkle-parity |

| /demo/ | v0.2.4 | \*\*f58ff661e9b1\*\* | v0.2.4-verifier-merkle-parity |



\*\*There are TWO DIFFERENT BUILDS claiming to be v0.2.4:\*\*



\*\*Build A:\*\*

\- Commit: 9bfca919d07d

\- Tag: v0.2.4-verifier-syntax-fix

\- Location: /v0.2.4/ (when clicked from /versions/)



\*\*Build B:\*\*

\- Commit: f58ff661e9b1 (matches f58ff66 short hash from /versions/)

\- Tag: v0.2.4-verifier-merkle-parity

\- Location: mathledger.ai root redirect, /demo/, verifier page



\*\*Acquisition Committee Impact:\*\*



An auditor following /versions/ → v0.2.4 will:

1\. See /versions/ says v0.2.4 commit is f58ff66

2\. Click v0.2.4 link

3\. Land on page claiming commit is 9bfca919d07d

4\. \*\*Cannot determine which commit is the "real" v0.2.4\*\*



\*\*Question with no answer:\*\* Which version am I auditing?



---



\### BLOCKING-2: FOR\_AUDITORS Tag Mismatch



\*\*FOR\_AUDITORS Step 1.3 says:\*\*

> "Verify the version banner shows `v0.2.4-verifier-syntax-fix`"



\*\*Demo banner actually shows:\*\*

> "v0.2.4-verifier-merkle-parity"



\*\*Evidence:\*\*

\- FOR\_AUDITORS expects: v0.2.4-verifier-syntax-fix (Build A tag)

\- Demo shows: v0.2.4-verifier-merkle-parity (Build B tag)



\*\*Acquisition Committee Impact:\*\*



An auditor executing FOR\_AUDITORS Step 1 will:

1\. Navigate to /demo/

2\. See banner "v0.2.4-verifier-merkle-parity"

3\. Checklist says expect "v0.2.4-verifier-syntax-fix"

4\. \*\*Step 1.3 fails immediately\*\*



\*\*No guidance is provided for what to do when tags don't match.\*\*



---



\### BLOCKING-3: Self-Test Status is Ambiguous



\*\*Self-Test Results:\*\*



| Test | Expected | Actual | Test Result | Reason |

|------|----------|--------|-------------|--------|

| valid\_boundary\_demo | PASS | PASS | PASS | - |

| tampered\_ht\_mismatch | FAIL | FAIL | FAIL | h\_t\_mismatch |

| tampered\_rt\_mismatch | FAIL | FAIL | FAIL | r\_t\_mismatch |



\*\*Overall Status:\*\* "SELF-TEST FAILED" (displayed in red)



\*\*Epistemic Contradiction:\*\*



All tests behaved as expected:

\- Valid pack → PASS ✅

\- Tampered packs → FAIL (as expected) ✅



\*\*Yet the overall status says "SELF-TEST FAILED".\*\*



\*\*Acquisition Committee Impact:\*\*



An auditor will:

1\. Click "Run self-test vectors"

2\. See "SELF-TEST FAILED" in red

3\. Conclude the verifier is broken

4\. \*\*Abort audit due to lack of confidence in verification tool\*\*



\*\*But the verifier is working correctly.\*\* The tampered packs failed as expected. The "SELF-TEST FAILED" message is misleading.



\*\*Interpretation Ambiguity:\*\*



Does "SELF-TEST FAILED" mean:

\- The verifier itself is broken? (auditor interpretation)

\- Some test vectors failed verification? (developer interpretation)



\*\*No clear guidance is provided.\*\*



---



\## MAJOR AUDIT PATH AMBIGUITIES



\### MAJOR-1: No "Download Evidence Pack" Button After Boundary Demo



\*\*FOR\_AUDITORS Step 3 says:\*\*

> "After running the boundary demo, click 'Download Evidence Pack'"



\*\*Problem:\*\* No such button exists after boundary demo completes.



\*\*Evidence:\*\*

\- Boundary demo ran successfully (4 outcomes: ABSTAINED, ABSTAINED, VERIFIED, REFUTED)

\- AUTHORITY STREAM shows: "Nothing committed yet. Authority stream is empty."

\- No "Download Evidence Pack" button visible



\*\*Acquisition Committee Impact:\*\*



An auditor cannot proceed past Step 3. The instruction is explicit: "click 'Download Evidence Pack'". No button with this text exists.



\*\*Possible Explanation:\*\*



The boundary demo is a \*demonstration\* that does not commit to the authority stream. Therefore, no evidence pack is generated. But FOR\_AUDITORS assumes an evidence pack is produced.



\*\*This is a documentation/demo behavior mismatch.\*\*



---



\### MAJOR-2: Version Semantics Require Multiple Checks



\*\*Observation:\*\*



To determine CURRENT version, an auditor must:

1\. Navigate to mathledger.ai (redirects to v0.2.4)

2\. See banner says "LOCKED (see /versions/ for current status)"

3\. Click /versions/ link

4\. Check table for CURRENT status



\*\*This is not a single-step process.\*\*



\*\*Assessment:\*\* While not blocking, this is more complex than "identify CURRENT version using only visible cues". The landing page does not directly state "v0.2.4 is CURRENT". It requires following a link to /versions/.



\*\*However, the link is clearly provided, so this is navigable.\*\*



---



\## POSITIVE FINDINGS (What Worked)



\### ✅ Version Semantics Are Unambiguous (Once You Reach /versions/)



\- /versions/ table clearly shows CURRENT (green, bold)

\- Supersession chain is explicit ("SUPERSEDED BY X")

\- Canonical source is stated: "/versions/ is the canonical status registry"



\### ✅ FOR\_AUDITORS Checklist Is Structured and Numbered



\- 5 steps with time estimates

\- Explicit button names ("Run Boundary Demo", "Download Evidence Pack")

\- Alternative path provided (Ready-to-Verify examples)



\### ✅ Ready-to-Verify Path Works



\- examples.json exists and contains 3 test vectors (1 PASS, 2 FAIL)

\- Usage instructions are clear (4 steps)

\- Verifier has "Run self-test vectors" button

\- Self-test executes all 3 vectors automatically



\### ✅ Verifier Is Pure JavaScript (Offline Verification)



\- Runs in browser, no server required

\- Uses RFC 8785 canonicalization

\- Merkle trees with domain separation



\### ✅ Demo Behavior Matches Expected Outcomes



\- ADV → ABSTAINED (excluded from authority stream)

\- PA → ABSTAINED (authority-bearing but no validator)

\- MV "2+2=4" → VERIFIED (arithmetic validator confirmed)

\- MV "3\*3=8" → REFUTED (arithmetic validator disproved)



---



\## SUMMARY OF FINDINGS



| ID | Severity | Finding | Impact on Audit Execution |

|----|----------|---------|---------------------------|

| \*\*BLOCKING-1\*\* | BLOCKING | Commit hash mismatch between /versions/ and archive page | Cannot determine which commit is v0.2.4 |

| \*\*BLOCKING-2\*\* | BLOCKING | FOR\_AUDITORS expects wrong tag (verifier-syntax-fix vs verifier-merkle-parity) | Step 1.3 fails immediately |

| \*\*BLOCKING-3\*\* | BLOCKING | Self-test status "FAILED" is ambiguous | Auditor concludes verifier is broken |

| \*\*MAJOR-1\*\* | MAJOR | No "Download Evidence Pack" button after boundary demo | Cannot proceed past Step 3 of FOR\_AUDITORS |

| \*\*MAJOR-2\*\* | MAJOR | Version identification requires following link to /versions/ | Not immediate, but navigable |



---



\## AUDITABILITY ASSESSMENT



\*\*Can an acquisition committee execute the audit path without guessing URLs?\*\*



\*\*Partial YES:\*\*

\- All links are clickable (no URL guessing required)

\- examples.json is downloadable via clickable link

\- Verifier has self-test button



\*\*But NO for coherent execution:\*\*

\- Cannot determine which commit is v0.2.4 (BLOCKING-1)

\- FOR\_AUDITORS Step 1 fails due to tag mismatch (BLOCKING-2)

\- Self-test result is ambiguous (BLOCKING-3)

\- Cannot complete demo→evidence→verifier flow (MAJOR-1)



---



\## COHERENCE ASSESSMENT



\*\*Are archive claims consistent with demo behavior?\*\*



\*\*NO.\*\*



\*\*Archive Claim:\*\*

> "This archive is immutable. The hosted demo at /demo/ is the live instantiation of this same version."



\*\*Reality:\*\*

\- Archive page (clicked from /versions/) claims commit 9bfca919d07d

\- Demo shows commit f58ff661e9b1

\- \*\*These are different commits\*\*



\*\*Archive Claim:\*\*

> "Field Manual (fm.tex/pdf): obligation ledger used to drive version promotions."



\*\*Reality:\*\*

\- Field Manual link exists on archive page

\- Not tested (out of scope for this audit)



\*\*FOR\_AUDITORS Claim:\*\*

> "After running the boundary demo, click 'Download Evidence Pack'"



\*\*Reality:\*\*

\- No "Download Evidence Pack" button exists after boundary demo

\- AUTHORITY STREAM says "Nothing committed yet"



---



\## CREDIBILITY ASSESSMENT



\*\*Would an acquisition committee trust this system after executing the audit?\*\*



\*\*NO.\*\*



\*\*Reasons:\*\*



1\. \*\*Version discipline appears broken\*\*

&nbsp;  - Two different commits claim to be v0.2.4

&nbsp;  - Canonical source (/versions/) and archive page disagree

&nbsp;  - FOR\_AUDITORS references wrong tag



2\. \*\*Self-test result destroys confidence\*\*

&nbsp;  - "SELF-TEST FAILED" in red suggests verifier is broken

&nbsp;  - Auditor cannot proceed with confidence



3\. \*\*Audit path is unexecutable\*\*

&nbsp;  - Step 1 fails (tag mismatch)

&nbsp;  - Step 3 fails (no download button)

&nbsp;  - No guidance for what to do when steps fail



4\. \*\*Archive claim is provably false\*\*

&nbsp;  - "Demo is live instantiation of this same version"

&nbsp;  - Demo commit ≠ archive commit (when clicked from /versions/)



---



\## CONCRETE CITATIONS



\### Citation 1: Commit Hash Mismatch



\*\*URL:\*\* https://mathledger.ai/versions/  

\*\*Text:\*\* "v0.2.4 | Demo | CURRENT | 2026-01-03 | f58ff66"



\*\*URL:\*\* https://mathledger.ai/v0.2.4/ (clicked from /versions/)  

\*\*Text:\*\* "Tag: v0.2.4-verifier-syntax-fix | Commit: 9bfca919d07d | Locked: 2026-01-03"



\*\*URL:\*\* https://mathledger.ai/demo/  

\*\*Text:\*\* "LIVE v0.2.4 | v0.2.4-verifier-merkle-parity | f58ff661e9b1"



---



\### Citation 2: FOR\_AUDITORS Tag Mismatch



\*\*URL:\*\* https://mathledger.ai/v0.2.4/docs/for-auditors/  

\*\*Text (Step 1.3):\*\* "Verify the version banner shows `v0.2.4-verifier-syntax-fix`"



\*\*URL:\*\* https://mathledger.ai/demo/  

\*\*Text (Banner):\*\* "LIVE v0.2.4 | v0.2.4-verifier-merkle-parity | f58ff661e9b1"



---



\### Citation 3: Self-Test Status Ambiguity



\*\*URL:\*\* https://mathledger.ai/v0.2.4/evidence-pack/verify/  

\*\*Text (After clicking "Run self-test vectors"):\*\* "SELF-TEST FAILED" (in red)



\*\*Table shows:\*\*

\- valid\_boundary\_demo: Expected PASS, Actual PASS, Test Result PASS

\- tampered\_ht\_mismatch: Expected FAIL, Actual FAIL, Test Result FAIL

\- tampered\_rt\_mismatch: Expected FAIL, Actual FAIL, Test Result FAIL



\*\*All tests matched expectations, yet overall status is "FAILED".\*\*



---



\### Citation 4: Missing Download Button



\*\*URL:\*\* https://mathledger.ai/v0.2.4/docs/for-auditors/  

\*\*Text (Step 3):\*\* "After running the boundary demo, click 'Download Evidence Pack'"



\*\*URL:\*\* https://mathledger.ai/demo/ (after running boundary demo)  

\*\*Observation:\*\* No "Download Evidence Pack" button exists. AUTHORITY STREAM shows "Nothing committed yet. Authority stream is empty."



---



\## RECOMMENDATION



\*\*HOLD OUTREACH\*\* until BLOCKING issues are resolved.



\*\*Required Fixes:\*\*



1\. \*\*Fix BLOCKING-1:\*\* Resolve commit hash mismatch

&nbsp;  - Ensure /versions/, archive page, and demo all show same commit for v0.2.4

&nbsp;  - Or clarify why two different commits both claim to be v0.2.4



2\. \*\*Fix BLOCKING-2:\*\* Update FOR\_AUDITORS to match demo tag

&nbsp;  - Change Step 1.3 to expect "v0.2.4-verifier-merkle-parity"

&nbsp;  - Or update demo to show "v0.2.4-verifier-syntax-fix"



3\. \*\*Fix BLOCKING-3:\*\* Clarify self-test status message

&nbsp;  - Change "SELF-TEST FAILED" to "SELF-TEST PASSED (2 tampered packs correctly detected)"

&nbsp;  - Or add explanation: "FAILED means some packs failed verification (expected behavior)"



4\. \*\*Fix MAJOR-1:\*\* Update FOR\_AUDITORS Step 3

&nbsp;  - Remove "Download Evidence Pack" instruction (boundary demo doesn't produce one)

&nbsp;  - Or modify demo to produce downloadable evidence pack

&nbsp;  - Or add fallback: "If no evidence pack is available, use Ready-to-Verify examples"



\*\*Estimated effort:\*\* 2-4 hours (depending on root cause of commit mismatch)



---



\## FINAL VERDICT



\*\*FAIL FOR OUTREACH\*\*



\*\*Reason:\*\* An acquisition committee executing this audit path will encounter immediate contradictions (commit hash mismatch, tag mismatch, ambiguous self-test result) that destroy confidence in version discipline and verification integrity.



\*\*Post-fix verdict:\*\* Re-audit required after fixes are deployed.



---



\*\*Audit completed:\*\* 2026-01-04  

\*\*Auditor:\*\* External (hostile, zero context)  

\*\*Report status:\*\* FINAL  

\*\*Verdict:\*\* FAIL

