\# Hostile Runtime Audit: MathLedger v0.2.4 Verifier + Demo Path (PRE-SYNC)





\*\*Auditor:\*\* Claude Chrome (runtime breaker)  

\*\*Date:\*\* 2026-01-03  

\*\*Target Version:\*\* v0.2.4 (per `/versions/`)  

\*\*Verdict:\*\* OUTREACH-NO-GO (at time of test)  

\*\*Note:\*\* This audit was executed while deployment sync issues were still present (demo reported v0.2.3 while `/versions/` reported v0.2.4).





---





\## Step-by-Step Results





\### Step 1 — Identify CURRENT version

\*\*PASS\*\* ✅  

\- CURRENT: `v0.2.4`  

\- URL: `https://mathledger.ai/versions/`





---





\### Step 2 — Version-pinned artifacts in CURRENT archive

\*\*PASS\*\* ✅  

From `https://mathledger.ai/v0.2.4/`:





\- \*\*Open Auditor Tool\*\* → `/v0.2.4/evidence-pack/verify/`  

\- \*\*Download Examples (PASS + FAIL)\*\* → `/v0.2.4/evidence-pack/examples.json`  





Both are correctly version-pinned to `v0.2.4`.





---





\### Step 3 — v0.2.4 verifier self-test

\*\*FAIL\*\* ❌  

\- URL: `https://mathledger.ai/v0.2.4/evidence-pack/verify/`  

\- Console errors on load: \*\*None\*\* ✅  

\- Self-test status: \*\*"SELF-TEST FAILED"\*\* (expected \*\*"SELF-TEST PASSED (3 vectors)"\*\*)





Observed table:





| Name                 | Expected | Actual | Pass/Fail | Reason        |

|----------------------|----------|--------|----------|---------------|

| valid\_boundary\_demo  | PASS     | PASS   | PASS     | —             |

| tampered\_ht\_mismatch | FAIL     | FAIL   | FAIL     | h\_t\_mismatch  |

| tampered\_rt\_mismatch | FAIL     | FAIL   | FAIL     | r\_t\_mismatch  |





\*\*Issue:\*\* The \*test-scoring\* logic appears inverted/confused: tampered vectors are \*\*supposed\*\* to FAIL verification, but the harness marks them as a failed test case rather than “test PASS (expected FAIL matched).” The UI is conflating “verification verdict” with “test verdict.”





---





\### Step 4 — Manual tamper test

\*\*PASS\*\* ✅  

\- Input: `tampered\_ht\_mismatch.pack` pasted into verifier  

\- Result: `FAIL` with `h\_t mismatch` detected  

\- UI did not freeze at "Waiting…"





---





\### Step 5 — Demo-to-auditor path

\*\*FAIL\*\* ❌  

\- Expected: `/demo/` serves CURRENT `v0.2.4` and links to correct auditor tool  

\- Actual: demo reports \*\*v0.2.3\*\* while `/versions/` says CURRENT is \*\*v0.2.4\*\*

\- Demo “Open Auditor Tool” link:

&nbsp; - `/demo/v0.2.3/evidence-pack/verify/` → \*\*404 Not Found\*\* (`{"detail":"Not Found"}`)





---





\### Step 6 — Download evidence pack and verify end-to-end

\*\*NOT TESTED\*\* (blocked by Step 5 failure)





---





\## Console Errors

None observed on v0.2.4 verifier page after hard refresh.





---





\## Screenshots Captured

\- `ss\_2495k0l3m`: Self-test table shows “SELF-TEST FAILED”

\- `ss\_94807a14n`: Manual tamper FAIL works

\- `ss\_5083lezz4`: Demo “Open Auditor Tool” 404





---





\## Top 3 Reasons for NO-GO (at time of test)





1\. \*\*Self-test harness scoring is wrong\*\*  

&nbsp;  The verifier correctly computes PASS/FAIL, but the harness marks expected-FAIL vectors as failures.





2\. \*\*Demo → Auditor tool link 404s\*\*  

&nbsp;  Demo points to a non-existent `/demo/v0.2.3/evidence-pack/verify/` path.





3\. \*\*Demo version mismatch\*\*  

&nbsp;  Demo is v0.2.3 while CURRENT is v0.2.4, making the “3-step audit flow” non-executable.





---





\## Observations





\- The verifier’s cryptographic verification appears correct (manual tamper detection works).

\- The self-test presentation layer is incorrect (test verdict vs verify verdict conflation).

\- Version sync between archive and hosted demo is broken at time of audit.

\- `examples.json` appears well-structured and version-pinned.







