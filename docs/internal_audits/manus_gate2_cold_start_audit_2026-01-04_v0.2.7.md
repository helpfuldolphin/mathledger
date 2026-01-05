\# Gate 2: Cold-Start Audit Path Gate — v0.2.7



\*\*Auditor:\*\* Manus (Cold External, Zero Context)  

\*\*Entry Point:\*\* https://mathledger.ai/  

\*\*Target:\*\* v0.2.7 (CURRENT)  

\*\*Method:\*\* Zero context, zero URL guessing  

\*\*Date:\*\* 2026-01-04  

\*\*Audit Type:\*\* Gate 2 — Cold-Start Audit Path Gate



---



\## VERDICT



\*\*GATE 2: FAIL — Step 1.3 (demo banner verification)\*\*



---



\## EXECUTIVE SUMMARY



\*\*Target:\*\* v0.2.7 (confirmed as CURRENT via canonical registry)  

\*\*Failure Point:\*\* Demo shows v0.2.6, not v0.2.7  

\*\*Result:\*\* Cannot complete FOR\_AUDITORS Step 1.3



\*\*Audit terminated at Phase 2 (Step 1.3).\*\* Demo version does not match CURRENT version declared in canonical registry.



---



\## DETAILED FINDINGS



\### Phase 1: Identify CURRENT Version ✅ PASS



\*\*Landing Page:\*\*

\- URL: https://mathledger.ai/

\- Redirected to: https://mathledger.ai/v0.2.7/

\- Banner: "Status: LOCKED (see /versions/ for current status)"



\*\*Canonical Registry Check:\*\*

\- URL: https://mathledger.ai/versions/

\- Table entry: \*\*v0.2.7 | Demo | CURRENT | 2026-01-04 | 5d01b4b\*\*

\- Previous version: v0.2.6 | Demo | SUPERSEDED BY V0.2.7 | 2026-01-04 | 62799ae



\*\*✅ v0.2.7 confirmed as CURRENT via canonical registry\*\*



---



\### Phase 2: Follow FOR\_AUDITORS Checklist ❌ FAIL



\*\*FOR\_AUDITORS Page:\*\*

\- URL: https://mathledger.ai/v0.2.7/docs/for-auditors/

\- Accessed via clickable link: "5-minute auditor verification"

\- Page loads successfully ✅



\*\*Step 1.1: Navigate to /demo/\*\* ✅ PASS

\- Clicked /demo/ link

\- URL: https://mathledger.ai/demo/



\*\*Step 1.2: Confirm demo loads without errors\*\* ✅ PASS

\- Demo loads successfully

\- No errors observed



\*\*Step 1.3: Verify version banner\*\* ❌ FAIL

\- FOR\_AUDITORS instructs: "Verify the version banner shows `v0.2.7-verifier-parity`"

\- Demo banner shows: "LIVE v0.2.6 | v0.2.6-verifier-correctness | 62799ae82a62"

\- \*\*Expected:\*\* v0.2.7-verifier-parity

\- \*\*Actual:\*\* v0.2.6-verifier-correctness



---



\## BLOCKING FINDING



\*\*Location:\*\* https://mathledger.ai/demo/



\*\*What Happened:\*\*

1\. Confirmed v0.2.7 is CURRENT via /versions/ ✅

2\. Navigated to FOR\_AUDITORS page ✅

3\. Followed Step 1.1: Navigate to /demo/ ✅

4\. Followed Step 1.2: Confirm demo loads ✅

5\. Followed Step 1.3: Verify version banner ❌



\*\*Expected:\*\* v0.2.7-verifier-parity  

\*\*Actual:\*\* v0.2.6-verifier-correctness



\*\*Why It Matters:\*\*



An acquisition committee member following FOR\_AUDITORS from zero context will:

1\. Confirm v0.2.7 is CURRENT ✅

2\. Navigate to FOR\_AUDITORS ✅

3\. Navigate to /demo/ ✅

4\. See v0.2.6 banner ❌

5\. \*\*Conclude version mismatch\*\* — Cannot proceed with Step 1.3



\*\*No guidance is provided\*\* for what to do when demo version does not match CURRENT.



---



\## VERSION COHERENCE



| Source | Version | Tag | Commit |

|--------|---------|-----|--------|

| /versions/ (canonical) | v0.2.7 | - | 5d01b4b |

| v0.2.7 archive page | v0.2.7 | v0.2.7-verifier-parity | 5d01b4b1446e |

| FOR\_AUDITORS page | v0.2.7 | v0.2.7-verifier-parity | 5d01b4b1446e |

| \*\*/demo/\*\* | \*\*v0.2.6\*\* | \*\*v0.2.6-verifier-correctness\*\* | \*\*62799ae82a62\*\* |



\*\*❌ Version coherence: FAIL\*\* — Demo does not match CURRENT



---



\## CITATIONS



\*\*Citation 1: Canonical Registry\*\*

\- URL: https://mathledger.ai/versions/

\- Text: "v0.2.7 | Demo | CURRENT | 2026-01-04 | 5d01b4b"



\*\*Citation 2: FOR\_AUDITORS Step 1.3\*\*

\- URL: https://mathledger.ai/v0.2.7/docs/for-auditors/

\- Text: "Verify the version banner shows `v0.2.7-verifier-parity`"



\*\*Citation 3: Demo Banner\*\*

\- URL: https://mathledger.ai/demo/

\- Text: "LIVE v0.2.6 | v0.2.6-verifier-correctness | 62799ae82a62"



\*\*Citation 4: v0.2.7 Archive Claim\*\*

\- URL: https://mathledger.ai/v0.2.7/

\- Text: "The hosted demo at /demo/ is the live instantiation of this same version."



---



\## AUDIT EXECUTION LOG



| Phase | Task | Status | Result |

|-------|------|--------|--------|

| \*\*1\*\* | Identify CURRENT version | ✅ PASS | v0.2.7 confirmed via /versions/ |

| \*\*2\*\* | Follow FOR\_AUDITORS | ❌ FAIL | Step 1.3 failed (demo shows v0.2.6) |

| 3 | Verify Step 3 not blocking | ⏸️ NOT REACHED | Blocked at Phase 2 |

| 4 | Complete verification path | ⏸️ NOT REACHED | Blocked at Phase 2 |



---



\## ACQUISITION COMMITTEE IMPACT



\*\*Question:\*\* Can an acquisition committee execute the audit path end-to-end using only what the site tells them?



\*\*Answer:\*\* \*\*NO.\*\*



\*\*Reason:\*\* FOR\_AUDITORS Step 1.3 explicitly requires demo banner to show v0.2.7-verifier-parity. Demo shows v0.2.6-verifier-correctness instead.



\*\*Committee Member Experience:\*\*

1\. /versions/ says v0.2.7 is CURRENT ✅

2\. v0.2.7 archive says "demo is live instantiation of this same version" ✅

3\. FOR\_AUDITORS says verify banner shows v0.2.7-verifier-parity ✅

4\. Navigate to /demo/ ✅

5\. See v0.2.6-verifier-correctness banner ❌

6\. \*\*Stuck.\*\* Cannot proceed with Step 1.3.



\*\*Credibility Impact:\*\*



\- Canonical registry declares v0.2.7 as CURRENT

\- v0.2.7 archive claims demo is "live instantiation of this same version"

\- Demo shows v0.2.6 instead

\- \*\*Archive claim is provably false\*\*



---



\## OBSERVATIONS



\*\*Landing Page Behavior:\*\*

\- Landing page redirects to v0.2.7 (suggests it's current)

\- v0.2.7 archive page exists and is accessible

\- v0.2.7 is listed as CURRENT in /versions/



\*\*Demo Behavior:\*\*

\- Demo shows v0.2.6-verifier-correctness

\- Demo commit (62799ae82a62) matches v0.2.6 commit in /versions/

\- Demo does not match v0.2.7 commit (5d01b4b1446e)



\*\*Possible Explanations:\*\*

1\. Demo deployment is stale (not updated to v0.2.7)

2\. /demo/ endpoint is not versioned (always shows latest deployed, not latest CURRENT)

3\. v0.2.7 was marked CURRENT before demo was deployed



---



\## COMPARISON TO PREVIOUS AUDITS



\*\*Previous Gate 2 Audits:\*\*



\*\*v0.2.5 (2026-01-04):\*\*

\- Result: FAIL — Step 1.3

\- Reason: Demo showed v0.2.4, not v0.2.5

\- Finding: Demo version mismatch



\*\*v0.2.6 (2026-01-04):\*\*

\- Result: FAIL — Phase 5

\- Reason: Self-test showed "SELF-TEST FAILED"

\- Finding: Verifier u\_t\_mismatch on all test vectors



\*\*v0.2.7 (2026-01-04 — this audit):\*\*

\- Result: FAIL — Step 1.3

\- Reason: Demo shows v0.2.6, not v0.2.7

\- Finding: Demo version mismatch (same issue as v0.2.5)



\*\*Pattern:\*\* Demo deployment lags behind version promotion to CURRENT.



---



\## FINAL VERDICT



\*\*GATE 2: FAIL — Step 1.3 (demo banner verification)\*\*



\*\*Exact Failure Point:\*\* FOR\_AUDITORS Step 1.3 requires v0.2.7-verifier-parity, demo shows v0.2.6-verifier-correctness



\*\*Reason:\*\* Demo version does not match CURRENT version declared in canonical registry



\*\*Audit Path Status:\*\* BLOCKED at Phase 2 (Step 1.3)



\*\*Cannot complete:\*\* Demo→evidence→verifier→PASS path requires demo to be v0.2.7



---



\*\*Audit Completed:\*\* 2026-01-04  

\*\*Auditor Role:\*\* Epistemic Gatekeeper (Cold External)  

\*\*Report Status:\*\* FINAL  

\*\*Saved to:\*\* docs/external\_audits/manus\_gate2\_cold\_start\_audit\_2026-01-04\_v0.2.7.md



