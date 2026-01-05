\# Gate 2: Cold-Start Audit Path Gate — v0.2.6



\*\*Entry Point:\*\* https://mathledger.ai/  

\*\*Target:\*\* v0.2.6 (CURRENT)  

\*\*Method:\*\* Zero context, zero URL guessing  

\*\*Date:\*\* 2026-01-04



---



\## VERDICT



\*\*GATE 2: FAIL — Phase 5 (self-test verification)\*\*



---



\## REASON



FOR\_AUDITORS instructs auditors to use the Ready-to-Verify path:

> "Click 'Run self-test vectors' on the verifier page to see all examples tested automatically."



Verifier self-test result:

> \*\*SELF-TEST FAILED\*\* (red)



\*\*Self-Test Results:\*\*



| Name | Expected | Actual | Pass/Fail | Reason |

|------|----------|--------|-----------|--------|

| valid\_boundary\_demo | PASS | FAIL | \*\*FAIL\*\* | u\_t\_mismatch |

| tampered\_ht\_mismatch | FAIL | FAIL | PASS | u\_t\_mismatch |

| tampered\_rt\_mismatch | FAIL | FAIL | PASS | u\_t\_mismatch |



\*\*Expected:\*\* valid\_boundary\_demo shows PASS  

\*\*Actual:\*\* valid\_boundary\_demo shows FAIL with u\_t\_mismatch



\*\*All 3 test vectors report u\_t\_mismatch\*\*, suggesting the verifier's U\_t computation does not match examples.json.



---



\## EXECUTION SUMMARY



| Phase | Task | Status | Details |

|-------|------|--------|---------|

| \*\*1\*\* | Identify CURRENT version | ✅ PASS | v0.2.6 confirmed via /versions/ |

| \*\*2\*\* | Follow FOR\_AUDITORS via clickable links | ✅ PASS | All links navigable, no URL guessing |

| \*\*3\*\* | Verify Step 1.3 (demo banner) | ✅ PASS | v0.2.6-verifier-correctness matches |

| \*\*4\*\* | Verify Step 3 not blocking | ✅ PASS | Ready-to-Verify fallback present |

| \*\*5\*\* | Complete verification path | ❌ \*\*FAIL\*\* | Self-test shows "SELF-TEST FAILED" |



---



\## WHAT WORKED ✅



\*\*Phase 1:\*\* /versions/ clearly shows v0.2.6 as CURRENT (green, bold)  

\*\*Phase 2:\*\* FOR\_AUDITORS accessible via "5-minute auditor verification" link  

\*\*Phase 3:\*\* Demo banner matches v0.2.6-verifier-correctness (no version mismatch)  

\*\*Phase 4:\*\* Ready-to-Verify fallback clearly documented with examples.json download link  

\*\*Navigation:\*\* All links clickable, zero URL guessing required  

\*\*Version Coherence:\*\* All sources agree on v0.2.6-verifier-correctness



---



\## WHAT FAILED ❌



\*\*Phase 5:\*\* Verifier self-test shows "SELF-TEST FAILED" in red  

\*\*Expected:\*\* valid\_boundary\_demo shows PASS (1 PASS, 2 FAIL overall)  

\*\*Actual:\*\* valid\_boundary\_demo shows FAIL (0 PASS, 3 FAIL overall)  

\*\*Reason:\*\* All 3 test vectors report "u\_t\_mismatch"



---



\## BLOCKING FINDING



\*\*URL:\*\* https://mathledger.ai/v0.2.6/evidence-pack/verify/



\*\*What Happened:\*\*

1\. Navigated to verifier page ✅

2\. Clicked "Run self-test vectors" button ✅

3\. Verifier displayed "SELF-TEST FAILED" in red ❌

4\. valid\_boundary\_demo (expected PASS) showed FAIL ❌



\*\*Why It Matters:\*\*



An acquisition committee member following FOR\_AUDITORS will:

1\. Read: "Click 'Run self-test vectors' on the verifier page"

2\. Click the button

3\. See "SELF-TEST FAILED" in red

4\. Observe valid pack shows FAIL instead of PASS

5\. \*\*Conclude the verifier is broken\*\*



\*\*No guidance is provided\*\* for what to do when self-test fails.



---



\## ACQUISITION COMMITTEE IMPACT



\*\*Question:\*\* Can an acquisition committee execute the audit path end-to-end using only what the site tells them?



\*\*Answer:\*\* \*\*NO.\*\*



\*\*Reason:\*\* The Ready-to-Verify path (explicitly documented as "no demo required") fails at the final step. The verifier's self-test shows FAILED status, indicating the verification tool itself is broken.



\*\*Committee Member Experience:\*\*

1\. /versions/ says v0.2.6 is CURRENT ✅

2\. FOR\_AUDITORS says use Ready-to-Verify path ✅

3\. Download examples.json ✅

4\. Open verifier ✅

5\. Click "Run self-test vectors" ✅

6\. See "SELF-TEST FAILED" ❌

7\. \*\*Stuck.\*\* Cannot complete verification path.



---



\## VERSION COHERENCE CHECK



| Source | Version | Tag | Commit |

|--------|---------|-----|--------|

| /versions/ (canonical) | v0.2.6 | - | 62799ae |

| v0.2.6 archive page | v0.2.6 | v0.2.6-verifier-correctness | 62799ae82a62 |

| FOR\_AUDITORS page | v0.2.6 | v0.2.6-verifier-correctness | 62799ae82a62 |

| /demo/ | v0.2.6 | v0.2.6-verifier-correctness | 62799ae82a62 |

| Verifier page | v0.2.6 | v0.2.6-verifier-correctness | 62799ae82a62 |



\*\*✅ Version coherence: PASS\*\* (all sources agree)



---



\## CITATIONS



\*\*Citation 1: /versions/ declares v0.2.6 CURRENT\*\*

\- URL: https://mathledger.ai/versions/

\- Text: "v0.2.6 | Demo | CURRENT | 2026-01-04 | 62799ae"



\*\*Citation 2: FOR\_AUDITORS Ready-to-Verify instruction\*\*

\- URL: https://mathledger.ai/v0.2.6/docs/for-auditors/

\- Text: "Or use the verifier's built-in self-test: Click 'Run self-test vectors' on the verifier page to see all examples tested automatically."



\*\*Citation 3: Verifier self-test result\*\*

\- URL: https://mathledger.ai/v0.2.6/evidence-pack/verify/

\- Status: "SELF-TEST FAILED" (red)

\- valid\_boundary\_demo: Expected PASS, Actual FAIL, Reason: u\_t\_mismatch



\*\*Citation 4: examples.json expected results\*\*

\- URL: https://mathledger.ai/v0.2.6/evidence-pack/examples.json

\- valid\_boundary\_demo.expected\_verdict: "PASS"

\- tampered\_ht\_mismatch.expected\_verdict: "FAIL"

\- tampered\_rt\_mismatch.expected\_verdict: "FAIL"



---



\## TECHNICAL OBSERVATION



\*\*All 3 test vectors report "u\_t\_mismatch"\*\*, suggesting:

\- Either the verifier's U\_t computation changed between examples.json generation and verifier deployment

\- Or the examples.json test vectors contain incorrect U\_t values

\- Or the verifier's Merkle tree implementation does not match the Python implementation used to generate examples.json



\*\*Version tag is "v0.2.6-verifier-correctness"\*\*, which suggests this version was specifically intended to fix verifier correctness issues. The self-test failure indicates this was not achieved.



---



\## FINAL VERDICT



\*\*GATE 2: FAIL — Phase 5 (self-test verification)\*\*



\*\*Exact Failure Point:\*\* FOR\_AUDITORS Ready-to-Verify path → "Run self-test vectors" → SELF-TEST FAILED



\*\*Reason:\*\* Verifier self-test shows FAILED status. valid\_boundary\_demo (expected PASS) shows FAIL with u\_t\_mismatch. Cannot complete "examples→verifier→PASS" verification path.



\*\*Audit Path Status:\*\* BLOCKED (cannot demonstrate working verification)



---



\*\*Audit Completed:\*\* 2026-01-04  

\*\*Auditor Role:\*\* Epistemic Gatekeeper (Cold External)  

\*\*Report Status:\*\* FINAL

