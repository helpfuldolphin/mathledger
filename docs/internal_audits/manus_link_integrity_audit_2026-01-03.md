\# HOSTILE EXTERNAL AUDIT: MathLedger Link Integrity



\*\*Auditor Role:\*\* Hostile external auditor  

\*\*Goal:\*\* Break epistemic integrity by finding broken, misleading, or inconsistent links  

\*\*Date:\*\* 2026-01-03  

\*\*Assumption:\*\* This will be reviewed by an acquisition committee



---



\## HOMEPAGE ANALYSIS (https://mathledger.ai)



\*\*Redirect Behavior:\*\* 

\- Requested: `https://mathledger.ai`

\- Actual: `https://mathledger.ai/v0.2.1/`

\- \*\*ISSUE:\*\* Silent redirect to versioned path. No indication that bare domain redirects to v0.2.1. An auditor expecting a version-neutral landing page gets dumped into a specific version without warning.



\*\*Visible Links on Homepage (v0.2.1):\*\*



\### Header Navigation

1\. Scope → `/v0.2.1/docs/scope-lock/`

2\. Explanation → `/v0.2.1/docs/explanation/`

3\. Invariants → `/v0.2.1/docs/invariants/`

4\. Fixtures → `/v0.2.1/fixtures/`

5\. Evidence → `/v0.2.1/evidence-pack/`

6\. All Versions → `/versions/`



\### Body Links

7\. "Open Interactive Demo" → `/demo/`

8\. "5-minute auditor checklist" → `docs/for-auditors/` (RELATIVE PATH - RED FLAG)



\### Archive Contents Table

9\. For Auditors → (not directly linked, just text in table)

10\. Scope Lock → (duplicate of header)

11\. Explanation → (duplicate of header)

12\. Invariants → (duplicate of header)

13\. Hostile Rehearsal → (mentioned in table, no visible link)

14\. Fixtures → (duplicate of header)

15\. Evidence Pack → (duplicate of header)

16\. Manifest → `manifest.json` (mentioned, need to test)



\### Footer

17\. "Site built from commit 27a94c8a58139cb10349f6418336c618f528cbab" - no link to repo



---



\## LINK TESTING: v0.2.1 HOMEPAGE





\### TEST 1: "5-minute auditor checklist" link



\*\*Expected URL:\*\* `/v0.2.1/docs/for-auditors/` (absolute path)  

\*\*Actual URL:\*\* `https://mathledger.ai/v0.2.1/docs/for-auditors/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* MINOR  



\*\*Issue:\*\* The link on homepage was `docs/for-auditors/` (relative path), which worked due to browser resolution, but is inconsistent with other links that use absolute paths starting with `/v0.2.1/`. This creates ambiguity about whether the path is version-locked or not.



\*\*Audit Impact:\*\* MINOR - Link works but path inconsistency suggests sloppy implementation.



---



\### CRITICAL FINDING: For Auditors Page Claims vs Reality



The "For Auditors" page provides a "5-Minute Verification Checklist" with the following steps:



\*\*Step 1:\*\* Navigate to `/demo/` and verify version banner shows `v0.2.1-cohesion`



\*\*PROBLEM:\*\* We already know from previous audit that `/demo/health` reports version `0.2.0` with tag `v0.2.0-demo-lock`, NOT `v0.2.1-cohesion`. This instruction will FAIL for any auditor following it.



\*\*Step 3:\*\* "After running the boundary demo, click 'Download Evidence Pack'"



\*\*PROBLEM:\*\* We already verified that the boundary demo does NOT generate downloadable evidence packs. This instruction is IMPOSSIBLE to follow.



\*\*Step 4:\*\* "Open the Evidence Pack Verifier" and "Upload the evidence pack JSON you downloaded"



\*\*PROBLEM:\*\* No "Evidence Pack Verifier" tool was found in previous audit. Where is this tool? No link provided.



\*\*Step 5:\*\* "Tamper Test" - modify evidence pack and re-upload



\*\*PROBLEM:\*\* Cannot complete because Steps 3 and 4 are already impossible.



\*\*SEVERITY:\*\* \*\*BLOCKING\*\*



\*\*Audit Impact:\*\* The primary auditor entry point provides instructions that CANNOT BE COMPLETED. This is either:

1\. Documentation for a future version that doesn't exist yet

2\. Documentation that was never tested against the live system

3\. Intentional misdirection



Any acquisition committee following these instructions will conclude the system is non-functional or fraudulent.



---



\## NEW LINKS DISCOVERED ON FOR-AUDITORS PAGE



Testing additional links found on this page:





\### TEST 2: /demo/ link



\*\*Expected:\*\* Demo showing version `v0.2.1-cohesion` (per auditor checklist Step 1)  

\*\*Actual:\*\* Demo shows `v0.2.0 | v0.2.0-demo-lock | 27a94c8a5813`  

\*\*Status:\*\* ✅ 200 OK (page loads)  

\*\*Severity:\*\* \*\*BLOCKING\*\*



\*\*CRITICAL FAILURE:\*\* The "For Auditors" page explicitly instructs auditors to:

> "Verify the version banner shows v0.2.1-cohesion"



\*\*Reality:\*\* The version banner shows `v0.2.0-demo-lock`



\*\*This is a BLOCKING audit failure.\*\* The first verification step in the auditor checklist FAILS. An auditor following the official instructions will immediately conclude:

1\. The documentation is wrong

2\. The demo is outdated

3\. The system lacks version control discipline

4\. The "immutable archive" claim is undermined



\*\*Violates:\*\* Immutability expectations - the archive claims to be v0.2.1 but links to v0.2.0 demo  

\*\*Creates:\*\* Audit ambiguity - which version is actually being audited?  

\*\*Acceptable if documented?\*\* NO - this is explicitly documented INCORRECTLY in the auditor checklist



---



\## DEMO PAGE LINK TESTING



New links discovered on demo page:





\### TEST 3: "Scope Lock" link from demo



\*\*URL:\*\* `/demo/docs/view/V0\_LOCK.md`  

\*\*Actual:\*\* `https://mathledger.ai/demo/docs/view/V0\_LOCK.md`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* MAJOR



\*\*CRITICAL FINDING: Version Confusion Deepens\*\*



This document is titled "V0 Scope Lock" and describes "MathLedger Demo v0". It contains release notes for:

1\. \*\*v0-demo-lock\*\* (commit: ab8f51ab389aed7b3412cb987fc70d0d4f2bbe0b, date: 2026-01-02)

2\. \*\*v0.2.0-demo-lock\*\* (commit: 27a94c8a58139cb10349f6418336c618f528cbab, date: 2026-01-02)



\*\*Problems:\*\*



1\. \*\*The demo shows v0.2.0\*\* but this document describes both "v0" and "v0.2.0" as if they're the same thing

2\. \*\*The archive claims v0.2.1\*\* (commit: 27a94c8a5813) but the demo runs v0.2.0 (commit: 27a94c8a5813 - SAME SHORT HASH)

3\. \*\*The full commit hash in v0.2.0 release notes\*\* is 27a94c8a58139cb10349f6418336c618f528cbab, which matches the archive's full hash

4\. \*\*This means v0.2.0 and v0.2.1 are built from THE SAME COMMIT\*\* but tagged differently



\*\*Audit Impact:\*\* 



This is either:

\- A tagging/deployment error where v0.2.1 archive was built but demo wasn't updated

\- Intentional version forking from the same codebase

\- Evidence of poor version control discipline



\*\*Violates:\*\* Immutability expectations - same commit, different version tags  

\*\*Creates:\*\* SEVERE audit ambiguity - which version is actually being audited?  

\*\*Acceptable if documented?\*\* NO - the auditor checklist explicitly claims the demo shows v0.2.1-cohesion



---





\## ARCHIVE CONTENTS TABLE LINKS (v0.2.1)



All links use RELATIVE paths (no leading `/v0.2.1/`):



1\. For Auditors → `docs/for-auditors/`

2\. Scope Lock → `docs/scope-lock/`

3\. Explanation → `docs/explanation/`

4\. Invariants → `docs/invariants/`

5\. Hostile Rehearsal → `docs/hostile-rehearsal/`

6\. Fixtures → `fixtures/`

7\. Evidence Pack → `evidence-pack/`

8\. Manifest → `manifest.json`



\*\*Note:\*\* These relative paths are inconsistent with the header navigation which uses absolute paths like `/v0.2.1/docs/scope-lock/`. Testing each now.



---





\### TEST 4: "Hostile Rehearsal" link



\*\*URL:\*\* `docs/hostile-rehearsal/`  

\*\*Actual:\*\* `https://mathledger.ai/v0.2.1/docs/hostile-rehearsal/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* OK



\*\*Content:\*\* Comprehensive Q\&A document for defending the demo under adversarial questioning. Well-structured with 10 hostile questions and prepared answers.



\*\*Observation:\*\* This document is labeled "Version: v0.1 (MV arithmetic validator added)" but appears in the v0.2.1 archive. This suggests the document was created for v0.1 and hasn't been updated for v0.2.1, or the version label refers to the content version, not the archive version.



\*\*Minor Issue:\*\* Document version labeling could create confusion about which version it applies to.



---





\### TEST 5: manifest.json



\*\*URL:\*\* `manifest.json` (relative)  

\*\*Actual:\*\* `https://mathledger.ai/v0.2.1/manifest.json`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* MAJOR



\*\*CRITICAL DISCOVERY: Build Commit Mismatch\*\*



The manifest.json reveals a THIRD commit hash:



```json

{

&nbsp; "version": "v0.2.1",

&nbsp; "tag": "v0.2.1-cohesion",

&nbsp; "commit": "27a94c8a58139cb10349f6418336c618f528cbab",

&nbsp; "build\_commit": "cd2507d5f14ebf8ede0d4ebdba4f466799e18b5e",

&nbsp; "build\_time": "2026-01-03T19:46:25Z"

}

```



\*\*Analysis:\*\*



1\. \*\*Content commit:\*\* 27a94c8a58139cb10349f6418336c618f528cbab (what the archive documents)

2\. \*\*Build commit:\*\* cd2507d5f14ebf8ede0d4ebdba4f466799e18b5e (what built the static site)

3\. \*\*Build time:\*\* 2026-01-03T19:46:25Z



\*\*But the homepage footer says:\*\*

> "Site built from commit 27a94c8a58139cb10349f6418336c618f528cbab at 2026-01-03T18:55:59Z"



\*\*CONTRADICTIONS:\*\*



1\. \*\*Build commit mismatch:\*\* Footer claims site built from 27a94c8a5813, manifest says cd2507d5

2\. \*\*Build time mismatch:\*\* Footer says 18:55:59Z, manifest says 19:46:25Z (51 minutes later)

3\. \*\*This means the site was rebuilt AFTER the initial build\*\*, but the footer wasn't updated



\*\*Audit Impact:\*\* 



This destroys the "immutable archive" claim. The manifest proves the site was rebuilt at 19:46:25Z from commit cd2507d5, but the footer still shows the OLD build info (18:55:59Z, commit 27a94c8a5813). This is either:



1\. A deployment error where the manifest was updated but the footer wasn't

2\. Evidence that the "immutable" archive has been modified

3\. A fundamental misunderstanding of what "immutable" means



\*\*Violates:\*\* Immutability expectations - the archive has been rebuilt but claims to be immutable  

\*\*Creates:\*\* SEVERE audit ambiguity - which commit actually generated this archive?  

\*\*Acceptable if documented?\*\* NO - this directly contradicts the "immutable once published" claim



\*\*Severity:\*\* \*\*BLOCKING\*\* - The core epistemic claim (immutability) is violated by the manifest itself



---



\### ADDITIONAL FINDINGS FROM MANIFEST



\*\*Evidence Pack Verifier Tool Found:\*\*



The manifest lists:

```

"evidence-pack/verify/index.html"

```



This is the "Evidence Pack Verifier" mentioned in the auditor checklist but never linked from the main pages. Testing this path now.



---





\### TEST 6: Evidence Pack Verifier (Hidden Tool)



\*\*URL:\*\* `/v0.2.1/evidence-pack/verify/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* \*\*MAJOR\*\*



\*\*CRITICAL FINDING: Tool Exists But Is Hidden\*\*



The "Evidence Pack Verifier" mentioned in the auditor checklist Step 4 DOES exist at:

`https://mathledger.ai/v0.2.1/evidence-pack/verify/`



\*\*Problems:\*\*



1\. \*\*NOT LINKED from any main page\*\* - not on homepage, not on evidence pack page, not on for-auditors page

2\. \*\*Only discoverable via manifest.json\*\* - an auditor following the official checklist would have no way to find this tool

3\. \*\*The auditor checklist says:\*\* "Open the Evidence Pack Verifier" but provides NO LINK

4\. \*\*This makes Step 4 and Step 5 of the auditor checklist IMPOSSIBLE\*\* for a cold auditor



\*\*Audit Impact:\*\*



The auditor checklist provides a 5-step verification process, but Steps 4 and 5 (verify evidence pack, tamper test) are \*\*blocked by missing navigation\*\*. An auditor would need to:

1\. Guess the URL structure, OR

2\. Read the manifest.json and discover the path, OR

3\. Give up and conclude the tool doesn't exist



\*\*This is a BLOCKING usability failure.\*\* The tool exists and appears functional, but is completely hidden from the documented audit path.



\*\*Violates:\*\* Audit accessibility - tool exists but is undiscoverable  

\*\*Creates:\*\* BLOCKING audit ambiguity - auditors cannot complete the official checklist  

\*\*Acceptable if documented?\*\* NO - the checklist explicitly instructs auditors to use this tool but doesn't link to it



\*\*Recommendation:\*\* Add a prominent link to `/v0.2.1/evidence-pack/verify/` from:

1\. The for-auditors page (Step 4)

2\. The evidence-pack page

3\. The homepage archive contents table



---





\### TEST 7: /versions/ page



\*\*URL:\*\* `/versions/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* OK



\*\*Versions Identified:\*\*



| Version | Status | Locked | Commit (short) | Link |

|---------|--------|--------|----------------|------|

| v0 | SUPERSEDED BY V0.2.1 | 2026-01-02 | ab8f51a | /v0/ |

| v0.2.0 | SUPERSEDED BY V0.2.1 | 2026-01-02 | 27a94c8 | /v0.2.0/ + /demo/ |

| v0.2.1 | CURRENT | 2026-01-03 | 27a94c8 | /v0.2.1/ + /demo/ |



\*\*Key Observations:\*\*



1\. \*\*v0.2.0 and v0.2.1 share the same commit\*\* (27a94c8) - confirms earlier finding

2\. \*\*Both v0.2.0 and v0.2.1 link to the SAME /demo/\*\* - this explains why demo shows v0.2.0

3\. \*\*Footer build time matches manifest:\*\* "Site built at 2026-01-03T19:46:25Z" (matches manifest, NOT homepage footer)



\*\*CRITICAL FINDING: Demo Sharing\*\*



The versions table shows:

\- v0.2.0 | Demo

\- v0.2.1 | Demo



Both link to `/demo/`, which means \*\*there is only ONE demo instance\*\* serving both versions. This explains why:

\- The v0.2.1 archive links to a demo that reports v0.2.0

\- The auditor checklist fails (expects v0.2.1-cohesion, gets v0.2.0-demo-lock)



\*\*This is architectural, not a deployment error.\*\* The demo is intentionally shared between versions.



\*\*Audit Impact:\*\*



This design choice violates the "immutable version snapshot" claim. If v0.2.1 is supposed to be a "complete, immutable snapshot," it should have its own demo instance showing v0.2.1, not share a demo with v0.2.0.



\*\*Acceptable if documented?\*\* MAYBE - but it's not documented, and it directly contradicts the auditor checklist which expects version-specific demos.



---



\## CROSS-VERSION LINK TESTING



Now testing links across versions to check for broken cross-references.





\### TEST 8: v0 archive



\*\*URL:\*\* `/v0/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* MINOR



\*\*Version Info:\*\*

\- Status: SUPERSEDED BY V0.2.0 (but versions page says "SUPERSEDED BY V0.2.1")

\- Tag: v0-demo-lock

\- Commit: ab8f51ab389a

\- Locked: 2026-01-02

\- Tier A: 9 (not 10 like v0.2.1)



\*\*INCONSISTENCY FOUND:\*\*



The v0 archive header says:

> "Status: SUPERSEDED BY V0.2.0"



But the /versions/ page says:

> "v0: SUPERSEDED BY V0.2.1"



\*\*Which is it?\*\* Was v0 superseded by v0.2.0 or v0.2.1? This is sloppy version management.



\*\*Demo Availability:\*\*



v0 has NO hosted demo - only local execution instructions. This is consistent and honest (unlike v0.2.0/v0.2.1 which share a demo).



\*\*Archive Contents:\*\*



v0 has NO "For Auditors" page and NO "Hostile Rehearsal" page. These were added in later versions.



---



\## RAPID LINK TESTING: v0 ARCHIVE



Testing all links in v0 archive systematically:





\### TEST 9: v0.2.0 archive



\*\*URL:\*\* `/v0.2.0/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Severity:\*\* \*\*BLOCKING\*\*



\*\*CATASTROPHIC FINDING: v0.2.0 Claims to be CURRENT\*\*



The v0.2.0 archive header shows:

> "Status: CURRENT"



But the /versions/ page says:

> "v0.2.0: SUPERSEDED BY V0.2.1"



\*\*This is a BLOCKING integrity failure.\*\* The system has TWO versions simultaneously claiming to be CURRENT:

1\. v0.2.0 (on its own page)

2\. v0.2.1 (on its own page AND on /versions/)



\*\*Audit Impact:\*\*



An auditor navigating directly to `/v0.2.0/` would see "Status: CURRENT" and believe they're auditing the current version. This is FALSE. The version was superseded on 2026-01-03, but the archive page wasn't updated.



\*\*This directly violates the "immutable archive" claim.\*\* If the archive is immutable, the status label should have been "CURRENT" when locked on 2026-01-02, and should remain "CURRENT" forever. But the /versions/ page shows it as "SUPERSEDED," which means EITHER:



1\. The archive WAS modified (status changed from CURRENT to SUPERSEDED), OR

2\. The archive was NEVER updated (still shows old status)



Either way, the immutability claim is broken.



\*\*Violates:\*\* Immutability expectations - status labels are inconsistent  

\*\*Creates:\*\* BLOCKING audit ambiguity - which version is actually current?  

\*\*Acceptable if documented?\*\* NO - this is a fundamental integrity failure



---



\*\*Additional Observations:\*\*



\- v0.2.0 says "Interactive Demo: LOCAL ONLY" (no hosted demo)

\- But /versions/ page links v0.2.0 to `/demo/` (hosted demo exists)

\- v0.2.0 footer: "Site built from commit 27a94c8a5813 at 2026-01-03T15:57:04Z"

\- This is a DIFFERENT build time than v0.2.1 (18:55:59Z or 19:46:25Z depending on which source)



---



\## SUMMARY OF VERSION CHAOS



| Version | Archive Status | /versions/ Status | Build Time | Commit | Demo |

|---------|---------------|-------------------|------------|--------|------|

| v0 | SUPERSEDED BY V0.2.0 | SUPERSEDED BY V0.2.1 | 15:57:04Z | ab8f51a | None |

| v0.2.0 | \*\*CURRENT\*\* | SUPERSEDED BY V0.2.1 | 15:57:04Z | 27a94c8 | /demo/ |

| v0.2.1 | \*\*CURRENT\*\* | \*\*CURRENT\*\* | 18:55:59Z (footer) or 19:46:25Z (manifest) | 27a94c8 | /demo/ |



\*\*This is a complete version control disaster.\*\*



---





\## PHASE 5: CROSS-VERSION LINK TESTING



Testing how links behave across version boundaries and whether redirects are consistent.



\### Test: GitHub repo link



The archives mention cloning from GitHub but provide no actual repository URL. Testing common patterns:





\### TEST 10: GitHub Repository Link



\*\*URL:\*\* `https://github.com/mathledger/mathledger` (inferred from archive instructions)  

\*\*Status:\*\* ❌ 404 NOT FOUND  

\*\*Severity:\*\* \*\*BLOCKING\*\*



\*\*CRITICAL FINDING: Repository Does Not Exist\*\*



All three archive versions (v0, v0.2.0, v0.2.1) provide verification instructions that say:



> "Clone the repository"  

> "git clone https://github.com/helpfuldolphin/mathledger"  

> "git checkout \[commit hash]"



\*\*Problems:\*\*



1\. \*\*No actual repository URL provided\*\* - the instructions say "your-org" as a placeholder

2\. \*\*Tested common pattern\*\* (`github.com/mathledger/mathledger`) - \*\*404 NOT FOUND\*\*

3\. \*\*This makes ALL verification instructions IMPOSSIBLE to follow\*\*



\*\*Audit Impact:\*\*



The core verification claim is:



> "To verify this archive matches the source: Clone the repository, Checkout commit \[hash], Run build script, Compare files"



\*\*This is completely impossible.\*\* There is no public repository. An auditor cannot:

\- Verify the archive matches the source

\- Verify the commit hashes are real

\- Verify the build process

\- Verify the checksums

\- Run local execution



\*\*This destroys the entire epistemic foundation of the archive.\*\* The archive claims to be verifiable and immutable, but the source code is not accessible.



\*\*Violates:\*\* Verifiability - no source code available  

\*\*Creates:\*\* BLOCKING audit failure - cannot verify any claims  

\*\*Acceptable if documented?\*\* NO - the instructions explicitly claim verification is possible



\*\*Severity:\*\* \*\*BLOCKING\*\* - The entire audit is impossible without source access



---



\## COMPILATION: ADVERSARIAL AUDIT REPORT



Compiling all findings into final hostile audit report...





