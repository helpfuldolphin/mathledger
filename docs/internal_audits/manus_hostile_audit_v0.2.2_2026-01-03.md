\# HOSTILE AUDIT: MathLedger v0.2.2

\## For Acquisition Committee Review



\*\*Audit Date:\*\* 2026-01-03  

\*\*Auditor:\*\* Hostile External  

\*\*Target Version:\*\* v0.2.2  

\*\*Correct GitHub Repo:\*\* https://github.com/helpfuldolphin/mathledger



---



\## STEP 1: HOMEPAGE REDIRECT CHAIN



\*\*Test:\*\* Navigate to `https://mathledger.ai` and record redirect



\*\*Requested URL:\*\* `https://mathledger.ai`  

\*\*Final URL:\*\* `https://mathledger.ai/v0.2.2/`  

\*\*HTTP Status:\*\* 200 OK (redirect occurred)



\*\*Redirect Chain:\*\*

```

https://mathledger.ai → https://mathledger.ai/v0.2.2/

```



\*\*Landing Page Title:\*\* "Version v0.2.2 Archive — MathLedger v0.2.2"



\*\*Is versioned archive obvious?\*\* ✅ \*\*YES\*\*



Evidence:

1\. URL explicitly shows `/v0.2.2/`

2\. Page title includes "Version v0.2.2 Archive"

3\. Status shows: `LOCKED (see /versions/ for current status)` - does NOT claim CURRENT

4\. Tag: `v0.2.2-link-integrity`

5\. All header links are version-pinned (e.g., `/v0.2.2/docs/scope-lock/`)



\*\*Observation:\*\* Major improvement over v0.2.1. Status correctly defers to `/versions/` for CURRENT/SUPERSEDED determination.



---



\## STEP 2: SYSTEMATIC LINK TESTING



Testing every header link and Archive Contents table link...





\### Links Identified on Homepage



\*\*Header Links (visible in navigation):\*\*

1\. Scope → `/v0.2.2/docs/scope-lock/`

2\. Explanation → `/v0.2.2/docs/explanation/`

3\. Invariants → `/v0.2.2/docs/invariants/`

4\. Fixtures → `/v0.2.2/fixtures/`

5\. Evidence → `/v0.2.2/evidence-pack/`

6\. All Versions → `/versions/`



\*\*Prominent Action Links:\*\*

7\. Open Hosted Demo → `/demo/`

8\. Open Auditor Tool → `/v0.2.2/evidence-pack/verify/`

9\. 5-minute auditor verification → `/v0.2.2/docs/for-auditors/`

10\. Download Examples (PASS + FAIL) → `/v0.2.2/evidence-pack/examples.json`

11\. Open verifier → `/v0.2.2/evidence-pack/verify/`

12\. Open demo → `/demo/`



\*\*Archive Contents Table Links (need to scroll to see):\*\*

\- For Auditors

\- Scope Lock

\- Explanation

\- Invariants

\- Hostile Rehearsal

\- Fixtures

\- Evidence Pack

\- Manifest



\*\*Key Observation:\*\* v0.2.2 prominently displays:

\- Direct link to auditor tool (was hidden in v0.2.1)

\- Direct link to examples.json (new feature)

\- Direct link to for-auditors checklist (was hard to find in v0.2.1)



Now testing each link systematically...



---





\### Archive Contents Table Links (All Version-Pinned)



\*\*Confirmed Links:\*\*

1\. For Auditors → `/v0.2.2/docs/for-auditors/`

2\. Scope Lock → `/v0.2.2/docs/scope-lock/`

3\. Explanation → `/v0.2.2/docs/explanation/`

4\. Invariants → `/v0.2.2/docs/invariants/`

5\. Hostile Rehearsal → `/v0.2.2/docs/hostile-rehearsal/`

6\. Fixtures → `/v0.2.2/fixtures/`

7\. Evidence Pack → `/v0.2.2/evidence-pack/`

8\. Manifest → `/v0.2.2/manifest.json`



\*\*Critical Observation:\*\* ALL links are now version-pinned with absolute paths (e.g., `/v0.2.2/...`). This is the fix claimed in "Release Delta: v0.2.2" - "All archive table links now version-pinned (absolute paths)".



This is a MAJOR IMPROVEMENT over v0.2.1, which used relative paths that could break cross-version navigation.



Now testing each link systematically for 404s or unexpected redirects...



---



\## LINK TESTING RESULTS





\### TEST 1: For Auditors Link



\*\*URL:\*\* `/v0.2.2/docs/for-auditors/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Result:\*\* Page loads correctly



\*\*BLOCKING ISSUE FOUND:\*\*



\*\*Step 1 of checklist says:\*\*

> "Verify the version banner shows v0.2.1-cohesion"



\*\*Problem:\*\* This is the v0.2.2 archive, but the checklist still references v0.2.1-cohesion. This is STALE DOCUMENTATION.



\*\*Expected:\*\* Checklist should say "Verify the version banner shows v0.2.2-link-integrity"



\*\*Impact:\*\* An auditor following this checklist will:

1\. Open the demo

2\. See version "v0.2.0-demo-lock" (if demo hasn't been updated)

3\. Expect to see "v0.2.1-cohesion"

4\. Conclude the verification FAILED



\*\*This is a BLOCKING issue\*\* - the official auditor checklist contains incorrect version information.



\*\*Additional Issues Found:\*\*



1\. \*\*"Download Examples" link\*\* in the checklist points to:

&nbsp;  - `releases/evidence\_pack\_examples.v0.2.1.json`

&nbsp;  - This references v0.2.1, not v0.2.2



2\. \*\*GitHub clone instructions\*\* say:

&nbsp;  ```

&nbsp;  git checkout v0.2.1-cohesion

&nbsp;  ```

&nbsp;  Should be: `git checkout v0.2.2-link-integrity`



3\. \*\*External Audits table\*\* lists audits of v0.2.1, not v0.2.2



\*\*Severity:\*\* BLOCKING - The for-auditors checklist is the PRIMARY verification document, and it contains multiple references to the wrong version.



---





\## STEP 4: EVIDENCE PACK PAGE VERIFICATION



\*\*URL:\*\* `/v0.2.2/evidence-pack/`  

\*\*Status:\*\* ✅ 200 OK



\### Required Links Verification



\*\*✅ PASS:\*\* Evidence pack page contains visible links to:



1\. \*\*Auditor Tool:\*\* `/v0.2.2/evidence-pack/verify/`

&nbsp;  - Link text: "Open Auditor Tool"

&nbsp;  - Prominently displayed in "Verification Tools" section

&nbsp;  - ✅ Correctly version-pinned



2\. \*\*Examples.json:\*\* `/v0.2.2/evidence-pack/examples.json`

&nbsp;  - Link text: "📥 Download Example Packs"

&nbsp;  - Prominently displayed in "Verification Tools" section

&nbsp;  - ✅ Correctly version-pinned



\*\*Observation:\*\* This is a MAJOR IMPROVEMENT over v0.2.1, where:

\- The auditor tool was hidden (no link from evidence pack page)

\- The examples.json file didn't exist



\*\*Additional Findings:\*\*



1\. \*\*Replay instructions reference correct version:\*\*

&nbsp;  - Says "checkout v0.2.2-link-integrity" ✅

&nbsp;  - This is consistent with the archive version



2\. \*\*Files section\*\* lists:

&nbsp;  - `input.json` (relative link)

&nbsp;  - This appears to be example input, not a complete evidence pack



Now testing the auditor tool and examples.json links...



---





\## STEP 5: EVIDENCE PACK VERIFIER TOOL



\*\*URL:\*\* `/v0.2.2/evidence-pack/verify/`  

\*\*Status:\*\* ✅ 200 OK  

\*\*Tool loads:\*\* ✅ YES



\### Interface Observations



\*\*Features:\*\*

\- Pure JavaScript (runs in browser, no server)

\- Uses RFC 8785 canonicalization

\- Manual verification interface with textarea

\- Upload button and Verify button

\- Status display showing "Waiting..."



\*\*BLOCKING ISSUE: NO BUILT-IN SELF-TEST VECTORS\*\*



\*\*Expected:\*\* The verifier should have built-in self-test vectors that can be run to confirm PASS/FAIL behavior without requiring external files.



\*\*Actual:\*\* The verifier only provides a manual verification interface. To test it, an auditor must:

1\. Download examples.json separately

2\. Open examples.json

3\. Copy/paste individual examples into the textarea

4\. Click Verify



\*\*This violates the audit instruction:\*\* "run the built-in self-test vectors"



\*\*There are no built-in self-test vectors.\*\* The tool requires manual input.



\*\*Workaround:\*\* Download examples.json and test manually.



Now downloading examples.json to test the verifier...



---





\### Examples.json Downloaded



\*\*URL:\*\* `/v0.2.2/evidence-pack/examples.json`  

\*\*Status:\*\* ✅ 200 OK  

\*\*File loads:\*\* ✅ YES



\*\*MAJOR ISSUE: STALE VERSION REFERENCES IN EXAMPLES.JSON\*\*



\*\*Problem 1: Usage instructions reference wrong version\*\*



```json

"usage\_instructions": {

&nbsp; "step\_2": "Open https://mathledger.ai/v0.2.1/evidence-pack/verify/",

}

```



\*\*Expected:\*\* Should say `/v0.2.2/evidence-pack/verify/`  

\*\*Actual:\*\* Says `/v0.2.1/evidence-pack/verify/`



\*\*Problem 2: Pack version field references wrong version\*\*



All three example packs contain:

```json

"pack\_version": "v0.2.1"

```



\*\*Expected:\*\* Should say `"pack\_version": "v0.2.2"`  

\*\*Actual:\*\* Says `"pack\_version": "v0.2.1"`



\*\*Impact:\*\* An auditor following the instructions in examples.json will:

1\. Be directed to the v0.2.1 verifier (wrong version)

2\. Upload packs labeled as v0.2.1 to the v0.2.2 verifier

3\. Experience confusion about which version they're testing



\*\*Severity:\*\* MAJOR - The examples file is a key verification artifact, and it contains incorrect version information throughout.



\*\*Examples Provided:\*\*

1\. `valid\_boundary\_demo` - Expected: PASS

2\. `tampered\_ht\_mismatch` - Expected: FAIL (h\_t set to zeros)

3\. `tampered\_rt\_mismatch` - Expected: FAIL (reasoning artifacts modified)



The examples appear structurally correct (contain u\_t, r\_t, h\_t hashes and proper evidence pack structure), but the version metadata is wrong.



---





\## STEP 6: STATUS SEMANTICS VALIDATION



\*\*Requirement:\*\* Confirm no version page claims CURRENT, and only /versions/ asserts CURRENT/SUPERSEDED.



\### /versions/ Page Analysis



\*\*URL:\*\* `/versions/`  

\*\*Status:\*\* ✅ 200 OK



\*\*Version Status Table:\*\*



| Version | Status | Locked | Commit |

|---------|--------|--------|--------|

| v0 | SUPERSEDED BY V0.2.1 | 2026-01-02 | ab8f51a |

| v0.2.0 | SUPERSEDED BY V0.2.1 | 2026-01-02 | 27a94c8 |

| v0.2.1 | SUPERSEDED BY V0.2.2 | 2026-01-03 | 27a94c8 |

| v0.2.2 | \*\*CURRENT\*\* | 2026-01-03 | 27a94c8 |



\*\*✅ PASS:\*\* /versions/ correctly asserts v0.2.2 as CURRENT



\*\*Archive Integrity Statement:\*\*

> "Each version directory is immutable once deployed. Superseded versions remain fully navigable. Prior versions are never modified; only their status label changes."



\*\*MINOR ISSUE: Immutability Contradiction\*\*



The statement "Prior versions are never modified; only their status label changes" is \*\*contradictory\*\*.



\*\*Analysis:\*\*

\- If "status label changes," then the version HAS been modified

\- The claim is that version directories are immutable, but status labels change

\- This suggests status labels are stored OUTSIDE the version directories (e.g., in /versions/)



\*\*This is actually CORRECT architecture\*\* - status is determined by /versions/, not by individual version pages. But the wording is confusing and could be interpreted as a violation of immutability.



\*\*Better wording:\*\* "Prior version directories are never modified. Status labels are maintained by /versions/ and may change as new versions are released."



\### Individual Version Page Status Check



\*\*v0.2.2 homepage status:\*\* `LOCKED (see /versions/ for current status)`  

\*\*v0.2.1 homepage status:\*\* (need to check)  

\*\*v0.2.0 homepage status:\*\* (need to check)



Checking older versions to confirm they don't claim CURRENT...



---





\### BLOCKING ISSUE: v0.2.1 STILL CLAIMS "CURRENT"



\*\*URL:\*\* `/v0.2.1/`  

\*\*Status Field:\*\* `Status: CURRENT` (displayed in green)



\*\*Expected:\*\* Should show `Status: LOCKED (see /versions/ for current status)` like v0.2.2



\*\*Actual:\*\* Shows `Status: CURRENT` even though /versions/ says "SUPERSEDED BY V0.2.2"



\*\*This directly violates the v0.2.2 release claim:\*\*



From v0.2.2 Release Delta:

> "Changed: Version pages show LOCKED status (not CURRENT/SUPERSEDED)"



\*\*Reality:\*\* v0.2.1 was NOT updated to show LOCKED status. It still claims CURRENT.



\*\*Impact:\*\* This is a CRITICAL failure of the immutability model:



1\. \*\*Contradictory status:\*\* /versions/ says v0.2.1 is SUPERSEDED, but v0.2.1 itself says CURRENT

2\. \*\*Violates stated design:\*\* v0.2.2 claims to have fixed this issue, but v0.2.1 wasn't updated

3\. \*\*Breaks trust:\*\* If "immutable archives" can have their status changed, they're not immutable



\*\*Two possible interpretations:\*\*



\*\*Interpretation A (Immutability Violation):\*\*

\- v0.2.1 archive WAS modified to change status from CURRENT to LOCKED

\- But the modification failed or wasn't deployed

\- This violates "immutable once published"



\*\*Interpretation B (Design Flaw):\*\*

\- v0.2.1 archive was NEVER modified (correctly immutable)

\- But it was published with "Status: CURRENT" baked in

\- This means EVERY version will forever claim CURRENT when published

\- The /versions/ page is the only source of truth, but individual archives contradict it



\*\*Either way, this is BLOCKING for acquisition.\*\*



Checking v0.2.0 to see if the pattern holds...



---





\### v0.2.0 Also Claims CURRENT



\*\*URL:\*\* `/v0.2.0/`  

\*\*Status Field:\*\* `Status: CURRENT` (displayed in green)



\*\*Pattern Confirmed:\*\* ALL versions claim "Status: CURRENT" when you visit their individual archive pages:



| Version | Individual Page Says | /versions/ Says | Contradiction? |

|---------|---------------------|-----------------|----------------|

| v0.2.0 | CURRENT | SUPERSEDED BY V0.2.1 | ✅ YES |

| v0.2.1 | CURRENT | SUPERSEDED BY V0.2.2 | ✅ YES |

| v0.2.2 | LOCKED (see /versions/) | CURRENT | ❌ NO |



\*\*Root Cause Analysis:\*\*



v0.2.2 is the FIRST version to implement the "LOCKED" status pattern. All prior versions (v0.2.0, v0.2.1) were published with "Status: CURRENT" baked into their static HTML.



\*\*This reveals a fundamental design flaw:\*\*



1\. \*\*Before v0.2.2:\*\* Status was baked into static archive pages

2\. \*\*After v0.2.2:\*\* Status defers to /versions/ with "LOCKED" label

3\. \*\*Problem:\*\* Old archives can't be updated (immutability), so they forever claim CURRENT



\*\*The v0.2.2 "fix" only applies to v0.2.2 and future versions.\*\* It doesn't fix the contradiction for v0.2.0 and v0.2.1.



\*\*Acquisition Committee Implication:\*\*



This is NOT a violation of immutability (old archives weren't modified). But it IS a violation of \*\*epistemic consistency\*\*:



\- An auditor visiting v0.2.1 directly sees "CURRENT"

\- An auditor visiting /versions/ sees "SUPERSEDED"

\- Both can't be true



\*\*Recommended Fix:\*\* Add a banner to v0.2.0 and v0.2.1 archives (via JavaScript or server-side redirect) that says:



> "⚠️ This version shows 'Status: CURRENT' but has been superseded. See /versions/ for current status."



But this would require modifying the archives, violating immutability.



\*\*Alternative:\*\* Accept that old archives have stale status labels, and document this as a known limitation in /versions/.



---





\## STEP 7: METADATA CONSISTENCY CHECK



\*\*URL:\*\* `/v0.2.2/manifest.json`  

\*\*Status:\*\* ✅ 200 OK



\### Manifest vs Footer Comparison



\*\*Homepage Footer Claims:\*\*

```

Site built from commit 27a94c8a58139cb10349f6418336c618f528cbab at 2026-01-03T21:27:01Z

```



\*\*Manifest.json Fields:\*\*

```json

{

&nbsp; "version": "v0.2.2",

&nbsp; "tag": "v0.2.2-link-integrity",

&nbsp; "commit": "27a94c8a58139cb10349f6418336c618f528cbab",

&nbsp; "build\_time": "2026-01-03T21:27:01Z",

&nbsp; "build\_commit": "dcf8d601bdd31401e3e59b8054220e7d67101762"

}

```



\### Consistency Analysis



\*\*✅ PASS - No Contradictions\*\*



| Field | Footer | Manifest | Match? |

|-------|--------|----------|--------|

| Commit | 27a94c8a5813...cbab | 27a94c8a5813...cbab | ✅ YES |

| Build Time | 2026-01-03T21:27:01Z | 2026-01-03T21:27:01Z | ✅ YES |



\*\*Key Observations:\*\*



1\. \*\*Build commit field:\*\* Manifest includes `build\_commit: dcf8d601...` which is NOT shown in footer

&nbsp;  - This is the commit that built the site (different from content commit)

&nbsp;  - Footer only shows content commit (27a94c8a5813)

&nbsp;  - This is CORRECT and expected



2\. \*\*Status field:\*\* Manifest says `"status": "current"` (lowercase)

&nbsp;  - Homepage says `Status: LOCKED (see /versions/ for current status)`

&nbsp;  - This is a MINOR inconsistency but not contradictory

&nbsp;  - Manifest status is metadata, homepage status is user-facing



3\. \*\*All file checksums present:\*\* Manifest includes SHA256 hashes for 31 files

&nbsp;  - This enables verification of archive integrity

&nbsp;  - ✅ Checksums can be independently verified



\*\*Comparison with v0.2.1 (Previous Blocking Issue):\*\*



In v0.2.1, we found:

\- Footer said: built at 18:55:59Z from commit 27a94c8a

\- Manifest said: built at 19:46:25Z from commit cd2507d5

\- \*\*This was a BLOCKING contradiction\*\*



In v0.2.2:

\- Footer and manifest agree on build time and commit

\- \*\*No contradiction\*\*



\*\*Verdict:\*\* ✅ Metadata is consistent. No blocking issues.



---





