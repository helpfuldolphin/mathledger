\# Gate 3 Runtime Audit — Verifier Correctness

\## MathLedger v0.2.4



\*\*Auditor:\*\* Claude Chrome (Hostile Runtime Auditor)  

\*\*Date:\*\* 2026-01-04  

\*\*Target Version:\*\* v0.2.4 (CURRENT)  

\*\*Scope:\*\* Browser-executed verifier correctness, UI truthfulness, auditor confidence



---



\## VERDICT: ❌ FAIL (BLOCKING)



\*\*Reason:\*\* The verifier’s self-test logic incorrectly reports failure when all test vectors behave exactly as expected.  

This creates a false negative that would cause an acquisition committee to incorrectly conclude the verifier is broken.



---



\## Executive Summary



The cryptographic verification itself is \*\*correct\*\*:

\- Valid packs pass

\- Tampered packs fail

\- Hash mismatches are detected accurately



However, the \*\*self-test reporting layer is wrong\*\*:

\- When Expected = FAIL and Actual = FAIL, the test should be marked \*\*PASS\*\*

\- Instead, the UI marks these as \*\*FAIL\*\*, and the banner shows \*\*“SELF-TEST FAILED”\*\*



This breaks audit confidence.



---



\## A. Version Synchronization — PASS



| Check | Expected | Observed | Result |

|------|----------|----------|--------|

| `/versions/` CURRENT | v0.2.4 | v0.2.4 | ✅ |

| `/demo/` banner | v0.2.4 | v0.2.4 | ✅ |

| Demo tag | v0.2.4-verifier-merkle-parity | Match | ✅ |



---



\## B. Verifier Runtime Integrity — PASS



\- Hard refresh performed (Ctrl+Shift+R)

\- DevTools console monitored before load

\- \*\*Zero JavaScript errors\*\*

\- Verifier UI loads correctly



\*\*URL tested:\*\*  

https://mathledger.ai/v0.2.4/evidence-pack/verify/



---



\## C. Self-Test Vectors — ❌ FAIL (BLOCKING)



\### Expected Behavior



| Test Vector | Expected Outcome |

|------------|------------------|

| valid\_boundary\_demo | PASS |

| tampered\_ht\_mismatch | FAIL |

| tampered\_rt\_mismatch | FAIL |



If all match expectations → \*\*SELF-TEST PASSED\*\*



---



\### Observed Behavior



\- Table renders correctly

\- Individual results match expectations

\- \*\*Overall banner shows:\*\* `SELF-TEST FAILED` ❌



\#### Table Observed



| Name | Expected | Actual | Displayed Result |

|-----|----------|--------|------------------|

| valid\_boundary\_demo | PASS | PASS | PASS |

| tampered\_ht\_mismatch | FAIL | FAIL | FAIL ❌ |

| tampered\_rt\_mismatch | FAIL | FAIL | FAIL ❌ |



---



\### Root Cause



The self-test logic is inverted:



\- It treats `FAIL` as a test failure, even when the failure is \*\*expected\*\*

\- UI conflates \*verification result\* with \*test verdict\*



This is a \*\*presentation-layer bug\*\*, not a cryptographic one.



---



\## D. Manual Tamper Test — PASS (Minor UX Issue)



\- Pasted `tampered\_ht\_mismatch` pack

\- Clicked Verify

\- Result: `FAIL`

\- Hash comparison clearly highlights `h\_t` mismatch



\*\*Minor Issue:\*\*  

Status text shows only `FAIL` without explicit reason string (e.g., “FAIL: h\_t mismatch”).  

The visual diff is clear, but explicit text would improve clarity.



---



\## E. Demo → Auditor Path — PASS



\- `/demo/` loads v0.2.4

\- Boundary demo runs correctly

\- “Open Auditor Tool” link points to `/v0.2.4/evidence-pack/verify/`

\- No 404s, no redirects

\- Final URL correct



---



\## Findings Summary



| ID | Severity | Finding |

|----|----------|--------|

| F1 | \*\*BLOCKING\*\* | Self-test incorrectly reports FAILED when all vectors match expectations |

| F2 | MINOR | Manual FAIL result lacks explicit reason text |



---



\## Recommendation



\*\*OUTREACH: NO-GO\*\* until F1 is fixed.



\### Required Fix



\- Change self-test verdict logic:

&nbsp; - A test \*\*passes\*\* when `Actual == Expected`, even if both are FAIL

\- Update banner logic:

&nbsp; - Show `SELF-TEST PASSED (3 vectors)` when all comparisons match

\- Optional:

&nbsp; - Add explicit reason string to manual FAIL status



After fix, re-run Gate 3.



---



\*\*Audit Status:\*\* FINAL  

\*\*Next Step:\*\* Micro-fix + re-run Claude Chrome Gate 3  

