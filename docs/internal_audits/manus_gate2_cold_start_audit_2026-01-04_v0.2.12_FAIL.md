\# Gate 2: Cold-Start Audit — v0.2.12



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



\*\*Result:\*\* ❌ FAIL



\*\*Canonical Registry Shows:\*\*

\- v0.2.10 | Demo | \*\*CURRENT\*\* | 2026-01-04 | 55d12f4

\- v0.2.11 | NOT LISTED

\- v0.2.12 | NOT LISTED



\*\*Expected:\*\* v0.2.12 is CURRENT  

\*\*Actual:\*\* v0.2.10 is CURRENT



\*\*Contradiction:\*\*

\- Landing page (mathledger.ai/) redirects to /v0.2.12/

\- /versions/ (canonical registry) shows v0.2.10 as CURRENT

\- v0.2.12 does not appear in version table



\*\*Analysis:\*\* The site's redirect and /versions/ do NOT agree. Landing page redirects to v0.2.12, but canonical registry shows v0.2.10 as CURRENT. This is a version coherence failure.



---



\## GATE 2 VERDICT



\*\*GATE 2: FAIL — Task A Step 2: /versions/ shows v0.2.10 as CURRENT, not v0.2.12 (version coherence failure)\*\*



---



\## Step-by-Step Results



| Task | Step | Status | Result |

|------|------|--------|--------|

| A | Navigate to mathledger.ai | ✅ PASS | Redirects to /v0.2.12/ |

| A | Confirm CURRENT via /versions/ | ❌ FAIL | /versions/ shows v0.2.10 as CURRENT, not v0.2.12 |

| B | Navigate to FOR\_AUDITORS | ⏸️ NOT REACHED | Blocked at Task A Step 2 |

| B | Verify demo banner | ⏸️ NOT REACHED | Blocked at Task A Step 2 |

| B | Complete verification path | ⏸️ NOT REACHED | Blocked at Task A Step 2 |



---



\*\*Audit stopped immediately at first failure per instructions.\*\*





