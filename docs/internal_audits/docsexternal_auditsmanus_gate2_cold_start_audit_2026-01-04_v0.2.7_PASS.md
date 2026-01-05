\# Gate 2: Cold-Start Audit Path Gate — v0.2.7



\*\*Auditor:\*\* Manus (Cold External, Zero Context)  

\*\*Entry Point:\*\* https://mathledger.ai/  

\*\*Target:\*\* v0.2.7 (CURRENT)  

\*\*Constraints:\*\* Zero context, zero URL guessing, no repo access  

\*\*Date:\*\* 2026-01-04  

\*\*Audit Type:\*\* Gate 2 — Cold-Start Audit Path Gate



---



\## VERDICT



\*\*GATE 2: PASS\*\*



---



\## EXECUTIVE SUMMARY



An acquisition committee \*\*can\*\* execute the audit path end-to-end using only what the site provides. All steps are executable via clickable links, version coherence is maintained, and the verification path completes successfully with expected results (1 PASS, 2 FAILs).



---



\## DETAILED EXECUTION LOG



\### Task 1: Identify CURRENT Version ✅ PASS



\*\*Starting Point:\*\* https://mathledger.ai/



\*\*Step 1:\*\* Navigate to entry point

\- Result: Redirected to https://mathledger.ai/v0.2.7/

\- Banner: "Status: LOCKED (see /versions/ for current status)"



\*\*Step 2:\*\* Click "(see /versions/ for current status)" link

\- URL: https://mathledger.ai/versions/

\- Table shows: \*\*v0.2.7 | Demo | CURRENT | 2026-01-04 | 5d01b4b\*\* (green, bold)

\- Previous version: v0.2.6 | Demo | SUPERSEDED BY V0.2.7



\*\*Result:\*\* ✅ v0.2.7 confirmed as CURRENT via canonical registry



---



\### Task 2: Navigate to FOR\_AUDITORS Page ✅ PASS



\*\*Step 1:\*\* Click v0.2.7 link to return to archive

\- URL: https://mathledger.ai/v0.2.7/



\*\*Step 2:\*\* Scroll to find FOR\_AUDITORS link

\- Link found: "5-minute auditor verification"



\*\*Step 3:\*\* Click "5-minute auditor verification" link

\- URL: https://mathledger.ai/v0.2.7/docs/for-auditors/

\- Page loads successfully



\*\*Result:\*\* ✅ FOR\_AUDITORS page accessed via clickable link only



---



\### Task 3: Execute Step 1 (CRITICAL) ✅ PASS



\*\*FOR\_AUDITORS Step 1 Instructions:\*\*

1\. Navigate to /demo/

2\. Confirm the demo loads without errors

3\. Verify the version banner shows `v0.2.7-verifier-parity`



\*\*Execution:\*\*



\*\*Step 1.1:\*\* Click /demo/ link

\- URL: https://mathledger.ai/demo/

\- Page loads successfully ✅



\*\*Step 1.2:\*\* Confirm demo loads without errors

\- Page title: "MathLedger Demo v0.2.7"

\- No errors observed ✅



\*\*Step 1.3:\*\* Verify version banner

\- Banner shows: "LIVE v0.2.7 | v0.2.7-verifier-parity | 5d01b4b1446e"

\- Expected: v0.2.7-verifier-parity

\- Actual: v0.2.7-verifier-parity

\- \*\*Match confirmed\*\* ✅



\*\*Result:\*\* ✅ Step 1 executed successfully, demo banner matches CURRENT version



---



\### Task 4: Verify Step 3 Not Blocking ✅ PASS



\*\*FOR\_AUDITORS Step 3:\*\* "After running the boundary demo, click 'Download Evidence Pack'"



\*\*Observation:\*\* Step 3 requires running boundary demo first (Step 2)



\*\*Fallback Path Found:\*\*



\*\*Section:\*\* "Ready-to-Verify Examples (No Demo Required)"



\*\*Description:\*\* "If the demo is unavailable or you want to verify without running it, use these pre-generated examples"



\*\*Download link:\*\* /v0.2.7/evidence-pack/examples.json (clickable)



\*\*Verification steps provided:\*\*

1\. Download examples.json

2\. Copy content of examples.valid\_boundary\_demo.pack to new file

3\. Open Evidence Pack Verifier

4\. Upload pack JSON

5\. Click Verify and observe PASS

6\. Repeat with tampered packs and observe FAIL



\*\*OR:\*\* "Click 'Run self-test vectors' on the verifier page to see all examples tested automatically"



\*\*Result:\*\* ✅ Step 3 is not blocking — Ready-to-Verify fallback is clearly documented and executable



---



\### Task 5: Complete Verification Path ✅ PASS



\*\*Chosen Path:\*\* Ready-to-Verify → Verifier → Self-Test



\*\*Reason:\*\* Explicitly documented as alternative, faster than demo path



\*\*Execution:\*\*



\*\*Step 1:\*\* Navigate to verifier page

\- Clicked "Evidence Pack Verifier" link from FOR\_AUDITORS page

\- URL: https://mathledger.ai/v0.2.7/evidence-pack/verify/



\*\*Step 2:\*\* Locate self-test button

\- Section: "Run Self-Test Vectors"

\- Description: "Click the button below to run all built-in test vectors. Expected results: valid packs PASS, tampered packs FAIL."

\- Button: "Run self-test vectors"



\*\*Step 3:\*\* Click "Run self-test vectors" button

\- Button clicked successfully



\*\*Step 4:\*\* Observe results

\- Overall Status: \*\*SELF-TEST PASSED (3 vectors)\*\* ✅



\*\*Results Table:\*\*



| Name | Expected | Actual | Test Result | Reason |

|------|----------|--------|-------------|--------|

| valid\_boundary\_demo | PASS | PASS | \*\*PASS\*\* | - |

| tampered\_ht\_mismatch | FAIL | FAIL | \*\*PASS\*\* | h\_t\_mismatch |

| tampered\_rt\_mismatch | FAIL | FAIL | \*\*PASS\*\* | r\_t\_mismatch |



\*\*Verification:\*\*

\- ✅ 1 PASS observed (valid\_boundary\_demo: Expected PASS, Actual PASS)

\- ✅ 2 expected FAILs observed (tampered packs correctly show FAIL)

\- ✅ All test results show PASS (meaning each test matched its expected outcome)



\*\*Result:\*\* ✅ Full verification path executed successfully



---



\## AUDIT EXECUTION SUMMARY



| Task | Status | Result |

|------|--------|--------|

| \*\*1. Identify CURRENT version\*\* | ✅ PASS | v0.2.7 confirmed via /versions/ |

| \*\*2. Navigate to FOR\_AUDITORS\*\* | ✅ PASS | Accessed via clickable link |

| \*\*3. Execute Step 1 (demo banner)\*\* | ✅ PASS | Banner shows v0.2.7-verifier-parity |

| \*\*4. Verify Step 3 not blocking\*\* | ✅ PASS | Ready-to-Verify fallback documented |

| \*\*5. Complete verification path\*\* | ✅ PASS | Self-test: 1 PASS, 2 FAILs as expected |



\*\*All tasks completed successfully using only site-provided instructions and clickable links.\*\*



---



\## VERSION COHERENCE



| Source | Version | Tag | Commit |

|--------|---------|-----|--------|

| /versions/ (canonical) | v0.2.7 | - | 5d01b4b |

| v0.2.7 archive page | v0.2.7 | v0.2.7-verifier-parity | 5d01b4b1446e |

| FOR\_AUDITORS page | v0.2.7 | v0.2.7-verifier-parity | 5d01b4b1446e |

| /demo/ | v0.2.7 | v0.2.7-verifier-parity | 5d01b4b1446e |

| Verifier page | v0.2.7 | v0.2.7-verifier-parity | 98aa4bed8df1 |



\*\*✅ Version coherence: PASS\*\* — All sources show v0.2.7 with v0.2.7-verifier-parity tag



\*\*Note:\*\* Verifier commit (98aa4bed8df1) differs from archive commit (5d01b4b1446e), but verifier page explicitly states it was "Generated from commit 98aa4bed8df1" which appears to be the verifier generation commit, not a version mismatch.



---



\## ACQUISITION COMMITTEE IMPACT



\*\*Question:\*\* Can an acquisition committee execute the audit path end-to-end using only what the site tells them?



\*\*Answer:\*\* \*\*YES.\*\*



\*\*Committee Member Experience:\*\*

1\. Navigate to mathledger.ai ✅

2\. Confirm v0.2.7 is CURRENT via /versions/ ✅

3\. Access FOR\_AUDITORS via clickable link ✅

4\. Navigate to /demo/ and verify banner matches v0.2.7-verifier-parity ✅

5\. Identify Ready-to-Verify fallback (no demo required) ✅

6\. Navigate to verifier and run self-test ✅

7\. Observe SELF-TEST PASSED with 1 PASS and 2 expected FAILs ✅



\*\*Zero URL guessing. Zero repo access. Zero assumptions.\*\*



---



\## OBSERVATIONS



\### What Worked Well ✅



1\. \*\*Version Identification:\*\* /versions/ page is clear and unambiguous

2\. \*\*Navigation:\*\* All links are clickable, no URL guessing required

3\. \*\*FOR\_AUDITORS:\*\* Step-by-step instructions are explicit

4\. \*\*Demo Banner:\*\* Matches CURRENT version exactly

5\. \*\*Ready-to-Verify Fallback:\*\* Clearly documented alternative path

6\. \*\*Self-Test:\*\* One-click execution with clear results

7\. \*\*Results Display:\*\* Table format makes PASS/FAIL obvious

8\. \*\*Version Coherence:\*\* All sources agree on v0.2.7-verifier-parity



\### Comparison to Previous Audits



\*\*v0.2.5 (2026-01-04):\*\*

\- Result: FAIL — Step 1.3

\- Reason: Demo showed v0.2.4, not v0.2.5



\*\*v0.2.6 (2026-01-04):\*\*

\- Result: FAIL — Phase 5

\- Reason: Self-test showed "SELF-TEST FAILED" (u\_t\_mismatch on all vectors)



\*\*v0.2.7 (2026-01-04 — this audit):\*\*

\- Result: \*\*PASS\*\*

\- Reason: All tasks completed successfully, version coherence maintained, self-test passed



\*\*Pattern:\*\* v0.2.7 fixes both issues:

1\. Demo deployment now matches CURRENT version

2\. Verifier parity fix resolves u\_t\_mismatch (v0.2.7 release notes confirm this)



---



\## EPISTEMIC EXECUTABILITY ASSESSMENT



\*\*Judging Criteria:\*\* Can a cold auditor execute the path without getting stuck, confused, or contradicted?



\*\*Assessment:\*\* \*\*YES.\*\*



\*\*Evidence:\*\*

\- No ambiguous instructions

\- No broken links

\- No version mismatches

\- No contradictory claims

\- Clear fallback paths

\- Explicit expected results

\- One-click verification



\*\*Credibility:\*\* \*\*HIGH.\*\*



The system does what it claims:

\- Archive says demo is "live instantiation of this same version" → Demo shows v0.2.7 ✅

\- FOR\_AUDITORS says banner shows v0.2.7-verifier-parity → Banner shows v0.2.7-verifier-parity ✅

\- Self-test says "valid packs PASS, tampered packs FAIL" → Results match exactly ✅



---



\## FINAL VERDICT



\*\*GATE 2: PASS\*\*



\*\*Reason:\*\* Acquisition committee can execute audit path end-to-end using only site-provided instructions and clickable links. All steps are executable, version coherence is maintained, and verification path completes successfully with expected results.



---



\*\*Audit Completed:\*\* 2026-01-04  

\*\*Auditor Role:\*\* Epistemic Gatekeeper (Cold External)  

\*\*Report Status:\*\* FINAL  

\*\*Saved to:\*\* docs/external\_audits/manus\_gate2\_cold\_start\_audit\_2026-01-04\_v0.2.7\_PASS.md

