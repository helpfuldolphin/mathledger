\# Gate 3 Runtime Verifier Audit — MathLedger v0.2.10



\*\*Auditor:\*\* Claude Chrome (Hostile Runtime Auditor)  

\*\*Target Version:\*\* v0.2.10  

\*\*URL:\*\* https://mathledger.ai/v0.2.10/evidence-pack/verify/  

\*\*Timestamp (UTC):\*\* 2026-01-05T00:13:33.055Z  

\*\*Scope:\*\* Runtime verifier correctness + self-test truthfulness  



---



\## VERDICT



\*\*GATE 3: FAIL\*\*



---



\## ARTIFACTS



\- Screenshot: Self-test results show `SELF-TEST FAILED` (red banner)

\- Console log: Zero JavaScript errors (clean console)

\- Final URL: https://mathledger.ai/v0.2.10/evidence-pack/verify/



---



\## SELF-TEST RESULTS OBSERVED



\*\*Banner:\*\* `SELF-TEST FAILED`  

\*\*Expected:\*\* `SELF-TEST PASSED (3 vectors)`



| Name | Expected | Actual | Test Result | Reason |

|------|----------|--------|------------|--------|

| valid\_boundary\_demo | PASS | FAIL | FAIL | u\_t\_mismatch |

| tampered\_ht\_mismatch | FAIL | FAIL | PASS | u\_t\_mismatch |

| tampered\_rt\_mismatch | FAIL | FAIL | PASS | u\_t\_mismatch |



---



\## CRITICAL FAILURES



1\. \*\*Known-good vector fails\*\*  

&nbsp;  - `valid\_boundary\_demo` should PASS but returns FAIL with `u\_t\_mismatch`.



2\. \*\*Reason string is wrong / non-specific\*\*  

&nbsp;  - Tampered vectors should report `h\_t\_mismatch` and `r\_t\_mismatch`, but all rows show `u\_t\_mismatch`.



3\. \*\*Self-test banner is non-truthful\*\*  

&nbsp;  - Because the known-good vector fails, the banner correctly shows FAILED — but this blocks outreach.



---



\## FINAL VERDICT



\*\*GATE 3: FAIL — valid\_boundary\_demo expected PASS but got FAIL with u\_t\_mismatch.\*\*



