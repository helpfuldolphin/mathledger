\# Gate 2: Cold-Start Epistemic Dunk — v0.2.9 (Post-Deployment)



\*\*Auditor:\*\* Manus (Cold External, Zero Context)  

\*\*Entry Point:\*\* https://mathledger.ai/  

\*\*Target:\*\* v0.2.9 (CURRENT)  

\*\*Constraints:\*\* Zero context, zero URL guessing, no repo access  

\*\*Date:\*\* 2026-01-04 (post-deployment update)



\*\*Core Question:\*\* Can audit path be executed end-to-end?



---



\## PHASE 1: Identify CURRENT Version (Post-Deployment)



\*\*Action:\*\* Navigate to https://mathledger.ai/ and verify CURRENT status



\*\*Landing page:\*\* https://mathledger.ai/ → redirects to https://mathledger.ai/v0.2.9/



\*\*Canonical registry:\*\* https://mathledger.ai/versions/



\*\*Version table:\*\*



| Version | Status | Locked | Commit |

|---------|--------|--------|--------|

| v0.2.8 | SUPERSEDED BY V0.2.9 | 2026-01-04 | ebb69ab |

| \*\*v0.2.9\*\* | \*\*CURRENT\*\* | 2026-01-04 | f01d43b |



\*\*Result:\*\* ✅ \*\*PASS — Phase 1\*\*



\*\*v0.2.9 confirmed as CURRENT\*\* per canonical registry.



---



\## PHASE 2: Navigate to FOR\_AUDITORS and Execute Step 1



\*\*Action:\*\* Navigate to FOR\_AUDITORS page via clickable link



\*\*FOR\_AUDITORS page:\*\* https://mathledger.ai/v0.2.9/docs/for-auditors/



\*\*FOR\_AUDITORS Step 1 Instructions:\*\*

1\. Navigate to /demo/

2\. Confirm the demo loads without errors

3\. Verify the version banner shows `v0.2.9-abstention-terminal`



\*\*Execution:\*\*



\*\*Step 1.1:\*\* Navigate to /demo/

\- \*\*URL:\*\* https://mathledger.ai/demo/

\- \*\*Result:\*\* ✅ Demo loads successfully



\*\*Step 1.2:\*\* Confirm demo loads without errors

\- \*\*Result:\*\* ✅ Page loads without errors



\*\*Step 1.3:\*\* Verify version banner shows `v0.2.9-abstention-terminal`

\- \*\*Expected:\*\* v0.2.9-abstention-terminal

\- \*\*Actual:\*\* v0.2.9 | v0.2.9-abstention-terminal | f01d43b14c57

\- \*\*Result:\*\* ✅ \*\*PASS\*\*



\*\*Banner text (top of page):\*\*

> "LIVE v0.2.9 | v0.2.9-abstention-terminal | f01d43b14c57"



\*\*Page title:\*\* "MathLedger Demo v0.2.9"



\*\*Result:\*\* ✅ \*\*PASS — Phase 2 (FOR\_AUDITORS Step 1)\*\*



---



\## PHASE 3: Verify Step 3 Not Blocking



\*\*Action:\*\* Check if Ready-to-Verify fallback is explicit and executable



\*\*FOR\_AUDITORS Step 3 Verification:\*\*



\*\*Step 3 instruction:\*\*

> "After running the boundary demo, click 'Download Evidence Pack'"



\*\*Ready-to-Verify Fallback Section:\*\*



\*\*Heading:\*\* "Ready-to-Verify Examples (No Demo Required)"



\*\*Explicit statement:\*\*

> "If the demo is unavailable or you want to verify without running it, use these pre-generated examples:"



\*\*Download link:\*\* "Download Examples" → /v0.2.9/evidence-pack/examples.json



\*\*Examples provided:\*\*



| Example | Expected Result | Purpose |

|---------|----------------|---------|

| valid\_boundary\_demo | PASS | Shows a correctly-formed evidence pack from a boundary demo |

| tampered\_ht\_mismatch | FAIL (h\_t mismatch) | Demonstrates detection of tampered H\_t field |

| tampered\_rt\_mismatch | FAIL (r\_t mismatch) | Demonstrates detection of tampered reasoning artifacts |



\*\*Verification Steps (explicit):\*\*

1\. Download examples.json

2\. Copy the content of examples.valid\_boundary\_demo.pack to a new file

3\. Open the Evidence Pack Verifier

4\. Upload the pack JSON

5\. Click Verify and observe PASS

6\. Repeat with tampered\_ht\_mismatch.pack and observe FAIL



\*\*Alternative (built-in self-test):\*\*

> "Or use the verifier's built-in self-test: Click 'Run self-test vectors' on the verifier page to see all examples tested automatically."



\*\*Result:\*\* ✅ \*\*PASS — Phase 3 (Step 3 Not Blocking)\*\*



\*\*Findings:\*\*

\- Step 3 is NOT blocking

\- Ready-to-Verify fallback is explicitly documented

\- Fallback is executable via clickable links

\- Built-in self-test provides zero-setup verification path



---



\## PHASE 4: Complete One Full Verification Path



\*\*Action:\*\* Execute Ready-to-Verify → self-test → PASS path (no demo required)



\*\*Verifier page:\*\* https://mathledger.ai/v0.2.9/evidence-pack/verify/



\*\*Action:\*\* Click "Run self-test vectors" button



\*\*Result:\*\* ❌ \*\*SELF-TEST FAILED\*\*



\*\*Self-Test Results Table:\*\*



| Name | Expected | Actual | Pass/Fail | Reason |

|------|----------|--------|-----------|--------|

| valid\_boundary\_demo | PASS | FAIL | \*\*FAIL\*\* | u\_t\_mismatch |

| tampered\_ht\_mismatch | FAIL | FAIL | PASS | u\_t\_mismatch |

| tampered\_rt\_mismatch | FAIL | FAIL | PASS | u\_t\_mismatch |



\*\*Overall Status:\*\* SELF-TEST FAILED (red)



\*\*Critical Finding:\*\*

\- \*\*valid\_boundary\_demo\*\* expected PASS, got FAIL

\- All 3 test vectors report "u\_t\_mismatch"

\- Cannot complete verification path: no PASS observed



\*\*Per audit instructions:\*\*

> "Complete one verification path... You must observe: At least one PASS, Two expected FAILs"



\*\*Result:\*\* ❌ \*\*FAIL — Phase 4 (Complete Verification Path)\*\*



\*\*Reason:\*\* Self-test shows 0 PASS, 3 FAIL. Cannot demonstrate working verification.



---



\## FINAL VERDICT



\*\*GATE 2: FAIL — Phase 4: Self-test shows 0 PASS (expected 1 PASS, 2 FAIL); all vectors report u\_t\_mismatch\*\*

