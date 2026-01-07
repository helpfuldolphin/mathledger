\# Gate 2: Cold-Start Audit — MathLedger v0.2.13



\*\*Auditor:\*\* Manus (Epistemic Gatekeeper)  

\*\*Date:\*\* 2026-01-05  

\*\*Target:\*\* v0.2.13 (CURRENT)  

\*\*Constraints:\*\* Zero context, zero URL guessing, no repo access, no assumptions



---



\## FINAL VERDICT



\*\*GATE 2: FAIL — Task B Step 5: examples.json not found (404), cannot complete verification path\*\*



---



\## Executive Summary



v0.2.13 made significant progress on version coherence documentation (addressing v0.2.12 FAIL findings), but introduced a BLOCKING regression: \*\*examples.json is missing\*\*, preventing cold auditors from completing any verification path.



---



\## Audit Results by Task



| Task | Status | Details |

|------|--------|---------|

| A: Confirm CURRENT version | ✅ PASS | v0.2.13 confirmed via /versions/ |

| B: Navigate to FOR\_AUDITORS | ✅ PASS | Accessible via clickable links |

| C: Verify demo/docs version independence documented | ✅ PASS | Explicitly documented with "Version Coherence Note" |

| B: Execute Step 1 (demo banner check) | ✅ PASS | Demo shows v0.2.10-demo-reliability as expected |

| B: Complete one verification path | ❌ FAIL | examples.json returns 404, self-test blocked |



---



\## BLOCKING FINDING



\*\*Finding:\*\* examples.json missing (HTTP 404)



\*\*URL Tested:\*\* https://mathledger.ai/v0.2.13/evidence-pack/examples.json



\*\*Impact:\*\*

\- Self-test path (FOR\_AUDITORS Step 3 recommended): BLOCKED

\- Ready-to-Verify path (no demo required): BLOCKED

\- Cannot observe "1 PASS, 2 FAILs" as required by Gate 2



\*\*FOR\_AUDITORS Claims:\*\*

\- Archive page: "📥 Download Examples (PASS + FAIL)" (link returns 404)

\- Step 3: "Click 'Run self-test vectors'" (produces "Error: examples.json not found")

\- Ready-to-Verify: "Download examples.json" (file does not exist)



\*\*Verifier Error Message:\*\*

> "Error: examples.json not found"



---



\## POSITIVE FINDINGS



✅ \*\*Version Coherence Documentation (Addresses v0.2.12 FAIL)\*\*



FOR\_AUDITORS now includes prominent "Version Coherence Note":



> "The hosted demo runs the latest demo-capability version (currently v0.2.10). Documentation versions may advance independently. This is intentional: documentation releases do not require demo redeployment. Version coherence is determined by /versions/, not by demo parity. See: Version Number Doctrine."



\*\*Step 1.3 Updated:\*\*

> "Verify the demo banner shows the latest demo-capability version (currently v0.2.10-demo-reliability). The demo may lag behind the CURRENT documentation version; this is intentional."



\*\*Assessment:\*\* This resolves the v0.2.12 version coherence contradiction. A cold auditor can now understand why demo shows v0.2.10 when documentation is v0.2.13.



✅ \*\*Demo Banner Check:\*\* Demo correctly shows v0.2.10-demo-reliability as documented



✅ \*\*Navigation:\*\* All links clickable, zero URL guessing required



✅ \*\*Abstention Semantics:\*\* Explicitly documented as "correct behavior and not a failure mode"



---



\## Detailed Execution Log



\### Task A: Confirm CURRENT Version ✅



1\. \*\*Navigate to /versions/:\*\* https://mathledger.ai/versions/

2\. \*\*Result:\*\* v0.2.13 | Demo | CURRENT | 2026-01-05 | f6adede

3\. \*\*Status:\*\* ✅ PASS



\### Task B: Navigate to FOR\_AUDITORS ✅



1\. \*\*Landing page:\*\* https://mathledger.ai/ (redirects to /v0.2.13/)

2\. \*\*Archive page observation:\*\* "Version Coherence Note" visible

3\. \*\*FOR\_AUDITORS link:\*\* Clickable link found in "For Auditors: 3-Step Verification" box

4\. \*\*URL:\*\* https://mathledger.ai/v0.2.13/docs/for-auditors/

5\. \*\*Status:\*\* ✅ PASS



\### Task C: Verify Demo/Docs Version Independence Documented ✅



\*\*FOR\_AUDITORS Page Content:\*\*



> \*\*Version Coherence Note:\*\* The hosted demo runs the latest demo-capability version (currently v0.2.10). Documentation versions may advance independently. This is intentional: documentation releases do not require demo redeployment. Version coherence is determined by /versions/, not by demo parity. See: Version Number Doctrine.



\*\*Key Elements:\*\*

\- ✅ Explicitly states demo runs "latest demo-capability version (currently v0.2.10)"

\- ✅ Explicitly states "Documentation versions may advance independently"

\- ✅ Explicitly states "This is intentional"

\- ✅ Provides clickable link to "Version Number Doctrine"

\- ✅ Step 1.3 no longer requires demo banner to match documentation version

\- ✅ Clarifies version coherence determined by /versions/, not demo parity



\*\*Status:\*\* ✅ PASS (demo/docs version independence explicitly documented)



\### Task B: Execute Step 1 (Demo Banner Check) ✅



\*\*FOR\_AUDITORS Step 1.1:\*\* Navigate to /demo/

\- \*\*URL:\*\* https://mathledger.ai/demo/

\- \*\*Result:\*\* ✅ Demo loads without errors



\*\*FOR\_AUDITORS Step 1.2:\*\* Confirm demo loads without errors

\- \*\*Result:\*\* ✅ Page loaded successfully, no error messages



\*\*FOR\_AUDITORS Step 1.3:\*\* Verify demo banner shows latest demo-capability version

\- \*\*Expected:\*\* v0.2.10-demo-reliability (per FOR\_AUDITORS instruction)

\- \*\*Actual:\*\* "LIVE v0.2.10 | v0.2.10-demo-reliability | 55d12f49dc44"

\- \*\*Result:\*\* ✅ PASS (banner matches expected version)



\*\*Status:\*\* ✅ PASS



\### Task B: Complete One Verification Path ❌



\*\*Chosen Path:\*\* Ready-to-Verify → Self-Test (recommended path per FOR\_AUDITORS Step 3)



\*\*Step 1:\*\* Navigate to verifier

\- \*\*URL:\*\* https://mathledger.ai/v0.2.13/evidence-pack/verify/

\- \*\*Result:\*\* ✅ Page loads



\*\*Step 2:\*\* Click "Run self-test vectors"

\- \*\*Action:\*\* Clicked button (index 5)

\- \*\*Result:\*\* ❌ "Error: examples.json not found"



\*\*Step 3:\*\* Verify examples.json exists

\- \*\*URL:\*\* https://mathledger.ai/v0.2.13/evidence-pack/examples.json

\- \*\*HTTP Response:\*\* 404 NOT FOUND

\- \*\*Result:\*\* ❌ File does not exist



\*\*Status:\*\* ❌ FAIL (BLOCKING - cannot complete verification path)



---



\## Comparison to Previous Audits



| Version | Gate 2 Status | Reason |

|---------|---------------|--------|

| v0.2.12 | FAIL | Version coherence contradictory (demo v0.2.10, docs v0.2.12, not explained) |

| \*\*v0.2.13\*\* | \*\*FAIL\*\* | \*\*examples.json missing (404), verification path blocked\*\* |



\*\*Progress:\*\* v0.2.13 resolved v0.2.12's version coherence issue but introduced new blocking regression.



---



\## Recommendation



\*\*HOLD OUTREACH\*\* until examples.json is restored.



\*\*Fix Required:\*\*

1\. Restore /v0.2.13/evidence-pack/examples.json (likely deployment error)

2\. Verify self-test shows "SELF-TEST PASSED (3 vectors)"

3\. Verify 1 PASS (valid\_boundary\_demo) and 2 FAILs (tampered cases)



\*\*Estimated Effort:\*\* 5-15 minutes (likely deployment/build issue, not code change)



\*\*Post-fix verdict:\*\* Re-run Gate 2 Task B Step 5 to verify verification path completes.



---



\## Audit Trail



\*\*Entry Point:\*\* https://mathledger.ai/  

\*\*Canonical Registry:\*\* https://mathledger.ai/versions/  

\*\*FOR\_AUDITORS:\*\* https://mathledger.ai/v0.2.13/docs/for-auditors/  

\*\*Demo:\*\* https://mathledger.ai/demo/  

\*\*Verifier:\*\* https://mathledger.ai/v0.2.13/evidence-pack/verify/  

\*\*examples.json (404):\*\* https://mathledger.ai/v0.2.13/evidence-pack/examples.json



---



\*\*GATE 2: FAIL — Task B Step 5: examples.json not found (404), cannot complete verification path\*\*

