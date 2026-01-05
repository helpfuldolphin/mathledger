\# Gate 3 Runtime Verifier Audit — MathLedger v0.2.5



\*\*Auditor:\*\* Claude Chrome (Hostile Runtime Auditor)  

\*\*Target Version:\*\* v0.2.5  

\*\*URL:\*\* https://mathledger.ai/v0.2.5/evidence-pack/verify/  

\*\*Timestamp (UTC):\*\* 2026-01-04T03:29:54.312Z  



---



\## VERDICT



\*\*GATE 3: FAIL\*\*



---



\## EXECUTIVE SUMMARY



The runtime verifier fails to correctly validate evidence packs.  

The self-test mechanism reports \*\*SELF-TEST FAILED\*\* even when all test vectors produce the correct Expected vs Actual outcomes.



This breaks the audit confidence model: a functioning verifier is presented as broken.



---



\## STEP-BY-STEP RESULTS



\### 1. Navigation

✅ Successfully navigated to verifier page.



\### 2. Hard Refresh

✅ Hard refresh performed with DevTools open.



\### 3. Console Check

✅ \*\*PASS\*\* — Zero JavaScript errors observed.



\### 4. Self-Test Execution

❌ \*\*FAIL — CRITICAL\*\*



\*\*Banner displayed:\*\* 

SELF-TEST FAILED

\*\*Expected banner:\*\* 

SELF-TEST PASSED (3 vectors)

\### Self-Test Results Table



| Name | Expected | Actual | Pass/Fail | Reason |

|-----|----------|--------|-----------|--------|

| valid\_boundary\_demo | PASS | FAIL | FAIL | u\_t\_mismatch |

| tampered\_ht\_mismatch | FAIL | FAIL | FAIL | u\_t\_mismatch |

| tampered\_rt\_mismatch | FAIL | FAIL | FAIL | u\_t\_mismatch |



---



\## ANALYSIS



\### Critical Defects



1\. \*\*Incorrect hash computation\*\*

&nbsp;  - `valid\_boundary\_demo` should PASS but fails with `u\_t\_mismatch`

&nbsp;  - Indicates JS verifier computes different U\_t/R\_t/H\_t than the evidence pack



2\. \*\*Broken self-test semantics\*\*

&nbsp;  - For tampered vectors: Expected=FAIL and Actual=FAIL should be a \*\*passing test\*\*

&nbsp;  - UI marks these as FAIL, conflating verification outcome with test verdict



3\. \*\*Incorrect failure reason\*\*

&nbsp;  - All failures report `u\_t\_mismatch`

&nbsp;  - Tampered tests should report `h\_t\_mismatch` or `r\_t\_mismatch`



---



\## MANUAL TAMPER CHECK



⚠️ \*\*PARTIAL PASS\*\*



\- Manual paste of `tampered\_ht\_mismatch` pack

\- Verification result: \*\*FAIL\*\* (correct)

\- Hash comparison clearly shows mismatch

\- \*\*Issue:\*\* Reason string reports `u\_t\_mismatch` instead of expected `h\_t\_mismatch`



---



\## EVIDENCE ARTIFACTS



\- 📸 Screenshot: Self-test table showing `SELF-TEST FAILED`

\- 📸 Screenshot: Manual FAIL with hash mismatch visible

\- 📋 Console log: Zero errors

\- 🔗 Final URL: https://mathledger.ai/v0.2.5/evidence-pack/verify/



---



\## FINAL VERDICT



\*\*GATE 3: FAIL\*\*



\*\*Reason:\*\*  

The verifier’s runtime hash computation does not match the evidence packs, and the self-test UI incorrectly reports failure even when expected outcomes are met.



This must be fixed before outreach.

