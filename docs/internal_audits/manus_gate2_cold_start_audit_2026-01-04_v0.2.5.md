\# Gate 2: Cold-Start Audit Path Gate — v0.2.5



\*\*Entry Point:\*\* https://mathledger.ai/  

\*\*Target:\*\* v0.2.5 (CURRENT)  

\*\*Auditor:\*\* Cold external (zero context, no URL guessing)  

\*\*Date:\*\* 2026-01-04



---



\## VERDICT



\*\*GATE 2: FAIL — Step 1.3 (version banner verification)\*\*



---



\## REASON



FOR\_AUDITORS Step 1.3 instructs:

> "Verify the version banner shows `v0.2.5-self-test-clarity`"



Demo banner shows:

> "LIVE v0.2.4 | v0.2.4-verifier-merkle-parity | f58ff661e9b1"



\*\*Expected:\*\* v0.2.5-self-test-clarity  

\*\*Actual:\*\* v0.2.4-verifier-merkle-parity



\*\*Step 1.3 fails immediately.\*\* Audit path is blocked.



---



\## EXECUTION SUMMARY



| Task | Status | Details |

|------|--------|---------|

| \*\*Task 1:\*\* Identify CURRENT version | ✅ PASS | v0.2.5 confirmed via /versions/ |

| \*\*Task 2:\*\* Follow FOR\_AUDITORS Step 1.1 | ✅ PASS | Navigated to /demo/ via clickable link |

| \*\*Task 2:\*\* Follow FOR\_AUDITORS Step 1.2 | ✅ PASS | Demo loads without errors |

| \*\*Task 2:\*\* Follow FOR\_AUDITORS Step 1.3 | ❌ FAIL | Version banner shows v0.2.4, not v0.2.5 |

| \*\*Task 3:\*\* Critical check (Step 3) | ⏸️ NOT REACHED | Blocked at Step 1.3 |

| \*\*Task 4:\*\* Complete verification path | ⏸️ NOT REACHED | Blocked at Step 1.3 |

| \*\*Task 5:\*\* Version coherence check | ⚠️ PARTIAL | Inconsistency detected |



---



\## VERSION COHERENCE CHECK (PARTIAL)



| Source | Version | Tag | Commit |

|--------|---------|-----|--------|

| /versions/ (canonical) | v0.2.5 | - | 20b5811 |

| v0.2.5 archive page | v0.2.5 | v0.2.5-self-test-clarity | 20b5811e207c |

| FOR\_AUDITORS page | v0.2.5 | v0.2.5-self-test-clarity | 20b5811e207c |

| \*\*/demo/\*\* | \*\*v0.2.4\*\* | \*\*v0.2.4-verifier-merkle-parity\*\* | \*\*f58ff661e9b1\*\* |



\*\*Inconsistency:\*\* Demo version does not match CURRENT version declared in /versions/.



---



\## DETAILED FINDINGS



\### BLOCKING: Demo Version Mismatch



\*\*URL:\*\* https://mathledger.ai/demo/  

\*\*Expected:\*\* v0.2.5-self-test-clarity  

\*\*Actual:\*\* v0.2.4-verifier-merkle-parity  

\*\*Impact:\*\* Acquisition committee member cannot verify they are testing CURRENT version



\*\*Exact Step:\*\* FOR\_AUDITORS Step 1.3  

\*\*Instruction:\*\* "Verify the version banner shows `v0.2.5-self-test-clarity`"  

\*\*Result:\*\* Banner shows v0.2.4-verifier-merkle-parity instead



\*\*Why It Matters:\*\*



An auditor following FOR\_AUDITORS from zero context will:

1\. Confirm v0.2.5 is CURRENT via /versions/ ✅

2\. Navigate to /demo/ as instructed ✅

3\. See banner shows v0.2.4 ❌

4\. \*\*Cannot proceed\*\* — instruction explicitly says verify v0.2.5-self-test-clarity



\*\*Disallowed Actions (per Gate 2 instructions):\*\*

\- ❌ Cannot guess URLs

\- ❌ Cannot assume intended behavior

\- ❌ Cannot ignore failed steps



\*\*Instruction:\*\* "If any step fails, stop and report."



---



\## WHAT WORKED



✅ \*\*Task 1 (Identify CURRENT):\*\* /versions/ clearly shows v0.2.5 as CURRENT (green, bold)  

✅ \*\*Navigation:\*\* All links clickable, no URL guessing required  

✅ \*\*FOR\_AUDITORS Access:\*\* Accessible via "5-minute auditor verification" link from archive  

✅ \*\*Demo Loads:\*\* /demo/ loads without errors  

✅ \*\*Archive Coherence:\*\* v0.2.5 archive page, FOR\_AUDITORS page, and /versions/ all agree on version



---



\## WHAT FAILED



❌ \*\*Demo Version:\*\* /demo/ shows v0.2.4 instead of v0.2.5  

❌ \*\*Step 1.3:\*\* Version banner verification fails  

❌ \*\*Version Coherence:\*\* Demo contradicts /versions/ canonical source  

❌ \*\*Audit Path:\*\* Cannot proceed past Step 1 of 5-step checklist



---



\## ACQUISITION COMMITTEE IMPACT



\*\*Question:\*\* Can an acquisition committee execute the audit path end-to-end using only what the site tells them?



\*\*Answer:\*\* \*\*NO.\*\*



\*\*Reason:\*\* Step 1.3 of FOR\_AUDITORS checklist is unexecutable. The instruction is explicit: verify the banner shows v0.2.5-self-test-clarity. The banner shows v0.2.4-verifier-merkle-parity. No guidance is provided for what to do when versions mismatch.



\*\*Committee Member Experience:\*\*

1\. /versions/ says v0.2.5 is CURRENT

2\. FOR\_AUDITORS says verify demo shows v0.2.5-self-test-clarity

3\. Demo shows v0.2.4-verifier-merkle-parity

4\. \*\*Stuck.\*\* Cannot determine which version to audit.



---



\## CITATIONS



\*\*Citation 1: /versions/ declares v0.2.5 CURRENT\*\*

\- URL: https://mathledger.ai/versions/

\- Text: "v0.2.5 | Demo | CURRENT | 2026-01-04 | 20b5811"



\*\*Citation 2: FOR\_AUDITORS expects v0.2.5-self-test-clarity\*\*

\- URL: https://mathledger.ai/v0.2.5/docs/for-auditors/

\- Text (Step 1.3): "Verify the version banner shows `v0.2.5-self-test-clarity`"



\*\*Citation 3: Demo shows v0.2.4\*\*

\- URL: https://mathledger.ai/demo/

\- Text (Banner): "LIVE v0.2.4 | v0.2.4-verifier-merkle-parity | f58ff661e9b1"

\- Text (Page Title): "MathLedger Demo v0.2.4"



---



\## FINAL VERDICT



\*\*GATE 2: FAIL — Step 1.3 (version banner verification)\*\*



\*\*Exact Failure Point:\*\* FOR\_AUDITORS Step 1.3  

\*\*Reason:\*\* Demo shows v0.2.4, not v0.2.5 as required by checklist  

\*\*Audit Path Status:\*\* BLOCKED (cannot proceed past Step 1 of 5)



---



\*\*Audit Completed:\*\* 2026-01-04  

\*\*Auditor Role:\*\* Epistemic Gatekeeper (Cold External)  

\*\*Report Status:\*\* FINAL

