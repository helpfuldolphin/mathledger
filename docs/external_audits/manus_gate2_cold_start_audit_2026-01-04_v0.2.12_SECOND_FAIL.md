\# Gate 2: Cold-Start Audit — v0.2.12 (Post-CDN Update)



\*\*Date:\*\* 2026-01-04  

\*\*Role:\*\* Epistemic Gatekeeper (Cold External Auditor)  

\*\*Constraints:\*\* Zero context, zero URL guessing, no repo access, no assumptions



---



\## Task A: Site-Wide Coherence (CURRENT)





\### Step 1: Navigate to https://mathledger.ai/



\*\*Result:\*\* ✅ Redirects to https://mathledger.ai/v0.2.12/



\*\*Landing Page Observations:\*\*

\- Status: LOCKED (see /versions/ for current status)

\- Tag: v0.2.12-versioning-doctrine

\- Commit: 15cc70f60732

\- Locked: 2026-01-05



\*\*Next Action:\*\* Click "(see /versions/ for current status)" link to confirm CURRENT status





\### Step 2: Confirm CURRENT status via /versions/



\*\*URL:\*\* https://mathledger.ai/versions/



\*\*Result:\*\* ✅ PASS



\*\*Canonical Registry Shows:\*\*

\- v0.2.10 | Demo | SUPERSEDED BY V0.2.11 | 2026-01-04 | 55d12f4

\- v0.2.11 | Demo | SUPERSEDED BY V0.2.12 | 2026-01-05 | f6f142f

\- v0.2.12 | Demo | \*\*CURRENT\*\* | 2026-01-05 | 15cc70f



\*\*Version Coherence:\*\* ✅ PASS

\- Landing page (mathledger.ai/) redirects to /v0.2.12/

\- /versions/ (canonical registry) shows v0.2.12 as CURRENT

\- Site's redirect and /versions/ \*\*agree\*\*



\*\*Task A Assessment:\*\* ✅ PASS (v0.2.12 confirmed as CURRENT, version coherence maintained)



---



\## Task B: Execute Audit Path End-to-End (CURRENT)





\### Step 1: Navigate to FOR\_AUDITORS via clickable link



\*\*URL:\*\* https://mathledger.ai/v0.2.12/docs/for-auditors/



\*\*Result:\*\* ✅ PASS



\*\*FOR\_AUDITORS Page Observations:\*\*

\- Version: v0.2.12-versioning-doctrine

\- Commit: 15cc70f60732

\- 5-Minute Verification Checklist present

\- Step 1.3 instructs: "Verify the version banner shows v0.2.12-versioning-doctrine"



\*\*Next Action:\*\* Navigate to /demo/ to verify Step 1.3 (demo banner matches CURRENT version)





\### Step 2: Verify Step 1.3 - Demo banner matches CURRENT version



\*\*URL:\*\* https://mathledger.ai/demo/



\*\*Result:\*\* ❌ FAIL



\*\*Demo Banner Shows:\*\*

\- "LIVE v0.2.10 | v0.2.10-demo-reliability | 55d12f49dc44"



\*\*FOR\_AUDITORS Step 1.3 Expects:\*\*

\- "Verify the version banner shows v0.2.12-versioning-doctrine"



\*\*Expected:\*\* v0.2.12-versioning-doctrine  

\*\*Actual:\*\* v0.2.10-demo-reliability



\*\*Analysis:\*\* Demo is running v0.2.10 (SUPERSEDED), not v0.2.12 (CURRENT). This is a version coherence failure. A cold auditor following FOR\_AUDITORS Step 1.3 will immediately fail at this step.



---



\## GATE 2 VERDICT



\*\*GATE 2: FAIL — Task B Step 2 (FOR\_AUDITORS Step 1.3): Demo banner shows v0.2.10, expected v0.2.12-versioning-doctrine\*\*



---



\## Step-by-Step Results



| Task | Step | Status | Result |

|------|------|--------|--------|

| A | Navigate to mathledger.ai | ✅ PASS | Redirects to /v0.2.12/ |

| A | Confirm CURRENT via /versions/ | ✅ PASS | v0.2.12 is CURRENT |

| B | Navigate to FOR\_AUDITORS | ✅ PASS | Accessed successfully |

| B | Verify demo banner (Step 1.3) | ❌ FAIL | Demo shows v0.2.10, not v0.2.12 |

| B | Complete verification path | ⏸️ NOT REACHED | Blocked at Step 1.3 |



---



\*\*Audit stopped immediately at first failure per instructions.\*\*





