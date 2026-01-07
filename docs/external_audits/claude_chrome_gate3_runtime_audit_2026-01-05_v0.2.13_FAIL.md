\# Gate 3 Runtime Verifier Audit — MathLedger v0.2.13 (FAIL)



\*\*Auditor:\*\* Claude Chrome (Hostile Runtime Auditor)  

\*\*Target Version:\*\* v0.2.13  

\*\*URL:\*\* https://mathledger.ai/v0.2.13/evidence-pack/verify/  

\*\*Date:\*\* 2026-01-05  



---



\## VERDICT



\*\*GATE 3: FAIL\*\*



---



\## EXECUTIVE SUMMARY



The verifier self-test cannot execute because the required test vector file is missing.



---



\## FAILURE DETAILS



\- Action: Clicked \*\*“Run self-test vectors”\*\*

\- Observed: Error message:



`Error: examples.json not found`



\- Expected: Banner \*\*“SELF-TEST PASSED (3 vectors)”\*\* plus results table.



---



\## ROOT CAUSE (Observed)



The verifier depends on:



`/v0.2.13/evidence-pack/examples.json`



This file is missing (404), so the self-test cannot load vectors.



---



\## FINAL VERDICT



\*\*GATE 3: FAIL — Self-test vectors failed with "Error: examples.json not found".\*\*

