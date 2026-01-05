\# BLOCKING FINDING: Demo Version Mismatch



\*\*Severity:\*\* BLOCKING  

\*\*Discovery Time:\*\* Phase 5 (Demo Coherence Verification)  

\*\*Impact:\*\* Acquisition committee embarrassment risk



---



\## THE PROBLEM



\*\*Expected:\*\* /demo/ shows v0.2.4 (CURRENT as per /versions/)  

\*\*Actual:\*\* /demo/ shows v0.2.3 (SUPERSEDED)



\*\*Evidence:\*\*

\- /versions/ table: v0.2.4 marked as CURRENT (green, bold)

\- /demo/ page title: "MathLedger Demo v0.2.3"

\- /demo/ banner: "LIVE v0.2.3 | v0.2.3-audit-path-freshness | 674bcd16104f"

\- /demo/ footer: "v0.2.3 (v0.2.3-audit-path-freshness)"



---



\## WHY THIS IS BLOCKING



\*\*Acquisition Committee Scenario:\*\*



1\. Auditor follows instructions: "Start at /versions/, identify CURRENT"

2\. Auditor sees v0.2.4 is CURRENT (commit 9bfca91)

3\. FOR\_AUDITORS says: "Navigate to /demo/, verify version banner shows v0.2.4-verifier-syntax-fix"

4\. Auditor clicks /demo/ link from v0.2.4 archive

5\. \*\*Demo shows v0.2.3 (commit 674bcd1), not v0.2.4\*\*



\*\*Founder is asked live:\*\*

> "Your documentation says v0.2.4 is CURRENT. Your demo shows v0.2.3. Which version are we evaluating?"



\*\*No good answer exists:\*\*

\- If demo is correct: /versions/ is wrong (version registry is broken)

\- If /versions/ is correct: demo is stale (deployment is broken)

\- Either way: \*\*immutability claim is violated\*\* (demo changed without version bump)



---



\## TECHNICAL ROOT CAUSE



\*\*Hypothesis 1: Demo is not version-pinned\*\*

\- /demo/ is a single endpoint, not versioned (/demo/v0.2.4/)

\- Demo may be manually updated, separate from archive builds

\- Archive says "hosted demo at /demo/ is the live instantiation of this same version"

\- This claim is \*\*provably false\*\* for v0.2.4



\*\*Hypothesis 2: v0.2.4 was released but demo was not updated\*\*

\- Archive built at 2026-01-04T00:53:50Z (per v0.2.4 footer)

\- Demo still shows v0.2.3 (commit 674bcd1)

\- This means v0.2.4 archive was published \*\*without updating the demo\*\*



---



\## ACQUISITION RISK



\*\*What the committee will think:\*\*



1\. \*\*Version discipline is broken\*\*

&nbsp;  - If CURRENT version's demo isn't live, what does "CURRENT" mean?

&nbsp;  - Are there other version mismatches we haven't found?



2\. \*\*Immutability claim is suspect\*\*

&nbsp;  - Archive says "demo is the live instantiation of this version"

&nbsp;  - Demo shows different version

&nbsp;  - Either the claim is wrong, or the demo changed without versioning



3\. \*\*Audit path is unexecutable\*\*

&nbsp;  - FOR\_AUDITORS checklist assumes /demo/ matches CURRENT

&nbsp;  - Auditor cannot verify v0.2.4 because demo is v0.2.3

&nbsp;  - "5-minute verification" is impossible



4\. \*\*Founder credibility hit\*\*

&nbsp;  - This is the \*\*first thing\*\* an auditor checks (Step 1 of FOR\_AUDITORS)

&nbsp;  - Failure at Step 1 taints all subsequent claims

&nbsp;  - "If they can't keep demo in sync, why trust the cryptography?"



---



\## RECOMMENDED FIX



\*\*Option A: Update /demo/ to v0.2.4 immediately\*\*

\- Deploy v0.2.4 demo to /demo/ endpoint

\- Verify banner shows "v0.2.4-verifier-syntax-fix | 9bfca91"

\- Risk: If v0.2.4 has bugs, this breaks the live demo



\*\*Option B: Use versioned demo URLs\*\*

\- Change /demo/ to redirect to /demo/v0.2.4/

\- Each version has its own demo endpoint

\- Archive claim becomes: "hosted demo at /demo/v0.2.4/ is the live instantiation"

\- This requires architecture change



\*\*Option C: Demote v0.2.4 to SUPERSEDED, promote v0.2.3 back to CURRENT\*\*

\- If v0.2.4 isn't ready for demo, don't call it CURRENT

\- Update /versions/ to mark v0.2.3 as CURRENT

\- Risk: Looks like a rollback (why was v0.2.4 released?)



---



\## VERDICT



\*\*OUTREACH-NO-GO\*\* until this is fixed.



\*\*Reason:\*\* An auditor executing FOR\_AUDITORS Step 1 will immediately discover the version mismatch. This is not a subtle bug. It's the first thing they check. Founder will be asked to explain live, and there is no good explanation.



\*\*Timeline:\*\* Fix required before any acquisition committee demo.



\*\*Confidence:\*\* 100% (verified by direct navigation to /demo/)



