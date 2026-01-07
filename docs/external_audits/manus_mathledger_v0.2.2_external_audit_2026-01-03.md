\# MathLedger v0.2.2 External Audit Findings



\*\*Audit Date:\*\* 2026-01-03  

\*\*Auditor Role:\*\* Hostile External Auditor  

\*\*Target Version:\*\* v0.2.2 (CURRENT as per /versions/)  

\*\*Repository:\*\* https://github.com/helpfuldolphin/mathledger  



---



\## Executive Summary



This audit examined the CURRENT release (v0.2.2) as shown on /versions/, focusing on:

1\. Status semantics and tier counts in v0.2.2 archive

2\. Link completeness and repo link correctness

3\. /demo/ version clarity

4\. "For Auditors" checklist executability

5\. examples.json and verifier UX for cold auditor PASS/FAIL



---



\## BLOCKING FINDINGS



None identified.



---



\## MAJOR FINDINGS



\### MAJOR-1: FOR\_AUDITORS.md Hardcodes Wrong Version (v0.2.1)



\*\*Location:\*\* `docs/FOR\_AUDITORS.md` lines 13, 86, 132, 195, 218



\*\*Evidence:\*\*

\- Line 13: "Verify the version banner shows `v0.2.1-cohesion`"

\- Line 86: "This version (v0.2.1) demonstrates:"

\- Line 132: "git checkout v0.2.1-cohesion"

\- Line 195: "This version (v0.2.1) demonstrates:"

\- Line 218: "git checkout v0.2.1-cohesion"



\*\*Impact:\*\* A cold auditor following the checklist will:

1\. Check for wrong version in demo (v0.2.1 instead of v0.2.2)

2\. Clone wrong git tag (v0.2.1-cohesion instead of v0.2.2-link-integrity)

3\. Believe they are auditing v0.2.1 when the archive claims v0.2.2



\*\*Why MAJOR:\*\* This breaks the "For Auditors" checklist requirement that "every step must be executable with clickable links and correct version/tag." A cold auditor cannot successfully complete the checklist as written.



---



\### MAJOR-2: FOR\_AUDITORS.md Links to Wrong Version Verifier



\*\*Location:\*\* `docs/FOR\_AUDITORS.md` lines 69, 189



\*\*Evidence:\*\*

\- Line 69: `Open the \[Evidence Pack Verifier](/v0.2.1/evidence-pack/verify/)`

\- Line 189: `Open the <a href="/v0.2.1/evidence-pack/verify/">Evidence Pack Verifier</a>` (in rendered HTML)



\*\*Impact:\*\* Cold auditor is directed to v0.2.1 verifier instead of v0.2.2 verifier. The checklist claims to be for the current version but points to a superseded version's tools.



\*\*Why MAJOR:\*\* Breaks verifier UX requirement. Auditor cannot verify v0.2.2 evidence packs using v0.2.1 verifier if there are schema differences.



---



\### MAJOR-3: FOR\_AUDITORS.md References Wrong examples.json File



\*\*Location:\*\* `docs/FOR\_AUDITORS.md` line 57, 67



\*\*Evidence:\*\*

\- Line 57: "The file `releases/evidence\_pack\_examples.v0.2.1.json` contains:"

\- Line 67: "Open \[releases/evidence\_pack\_examples.v0.2.1.json](https://github.com/helpfuldolphin/mathledger/blob/main/releases/evidence\_pack\_examples.v0.2.1.json)"



\*\*Impact:\*\* Cold auditor downloads v0.2.1 examples instead of v0.2.2 examples. The file `releases/evidence\_pack\_examples.v0.2.2.json` exists but is not referenced.



\*\*Why MAJOR:\*\* Breaks examples.json verifier UX requirement. Auditor cannot test PASS/FAIL with correct version examples.



---



\## MINOR FINDINGS



\### MINOR-1: /versions/ Page Does Not Explain Status Semantics



\*\*Location:\*\* `site/versions/index.html` lines 139-189



\*\*Evidence:\*\* The page shows:

\- "SUPERSEDED BY V0.2.1" (for v0, v0.2.0)

\- "SUPERSEDED BY V0.2.2" (for v0.2.1)

\- "CURRENT" (for v0.2.2)



But does not define what "CURRENT" or "SUPERSEDED" means. Does CURRENT mean:

\- Latest chronologically?

\- Recommended for use?

\- Only version with hosted demo?



\*\*Impact:\*\* Cold auditor must infer semantics. Not a blocking issue but reduces clarity.



\*\*Why MINOR:\*\* Semantics are inferable from context, but explicit definition would improve hostile auditor experience.



---



\### MINOR-2: Archive Table Links Use Relative Paths, Not Absolute



\*\*Location:\*\* `site/v0.2.2/index.html` lines 254-261



\*\*Evidence:\*\* Archive table shows:

```html

<tr><td><a href="/v0.2.2/docs/for-auditors/">For Auditors</a></td>

<tr><td><a href="/v0.2.2/docs/scope-lock/">Scope Lock</a></td>

```



These are absolute paths from site root, but the release delta (line 247) claims:

> "All archive table links now version-pinned (absolute paths)"



\*\*Impact:\*\* Links work correctly, but the claim in the release delta is technically accurate (they ARE absolute paths from root). However, a hostile auditor might expect fully qualified URLs like `https://mathledger.ai/v0.2.2/docs/...` for true "absolute" paths.



\*\*Why MINOR:\*\* Links function correctly. This is a semantic quibble about what "absolute" means (root-relative vs fully qualified).



---



\### MINOR-3: Repo Links Use Placeholder "main" Branch Instead of Version Tag



\*\*Location:\*\* `site/v0.2.2/docs/for-auditors/index.html` line 189



\*\*Evidence:\*\*

```html

<a href="https://github.com/helpfuldolphin/mathledger/blob/main/releases/evidence\_pack\_examples.v0.2.1.json">

```



\*\*Impact:\*\* Link points to `main` branch, not the locked tag `v0.2.2-link-integrity`. If `main` diverges, the link becomes stale.



\*\*Why MINOR:\*\* Link works today, but violates immutability principle. Should be:

```

https://github.com/helpfuldolphin/mathledger/blob/v0.2.2-link-integrity/releases/evidence\_pack\_examples.v0.2.2.json

```



---



\### MINOR-4: examples.json Contains v0.2.1 Pack Versions, Not v0.2.2



\*\*Location:\*\* `releases/evidence\_pack\_examples.v0.2.2.json` lines 19, 101, 183



\*\*Evidence:\*\*

\- Line 19: `"pack\_version": "v0.2.1"`

\- Line 101: `"pack\_version": "v0.2.1"`

\- Line 183: `"pack\_version": "v0.2.1"`



The file is named `evidence\_pack\_examples.v0.2.2.json` but all packs inside declare `pack\_version: v0.2.1`.



\*\*Impact:\*\* Cold auditor may be confused whether these are v0.2.2 examples or v0.2.1 examples. The file comment (line 6) says "Examples copied from v0.2.1; hash contract unchanged" which explains this, but it's not obvious.



\*\*Why MINOR:\*\* Functionally correct if hash contract is unchanged, but semantically confusing.



---



\### MINOR-5: /demo/ Version Clarity Not Audited (Demo Not Running)



\*\*Location:\*\* `/demo/` endpoint (not accessible in static build)



\*\*Evidence:\*\* The demo is not running in the cloned repository. The audit task requires verifying that "/demo/ clearly states it is the CURRENT demo and not a historical snapshot."



\*\*Impact:\*\* Cannot verify this requirement without running the demo server.



\*\*Why MINOR:\*\* The demo app.py (lines 34-65) shows version is loaded from releases.json and would report v0.2.2. The landing page (site/v0.2.2/index.html line 196) includes a JavaScript version check. Assuming the demo runs correctly, this would pass. But not verified.



---



\## SUMMARY TABLE



| ID | Severity | Finding | Blocking? |

|----|----------|---------|-----------|

| MAJOR-1 | MAJOR | FOR\_AUDITORS.md hardcodes v0.2.1 instead of v0.2.2 | No (doc issue, not code) |

| MAJOR-2 | MAJOR | FOR\_AUDITORS.md links to v0.2.1 verifier instead of v0.2.2 | No (doc issue) |

| MAJOR-3 | MAJOR | FOR\_AUDITORS.md references v0.2.1 examples.json instead of v0.2.2 | No (doc issue) |

| MINOR-1 | MINOR | /versions/ does not define status semantics | No |

| MINOR-2 | MINOR | "Absolute paths" claim is ambiguous (root-relative vs fully qualified) | No |

| MINOR-3 | MINOR | Repo links use "main" branch instead of version tag | No |

| MINOR-4 | MINOR | examples.json packs declare v0.2.1 inside v0.2.2 file | No |

| MINOR-5 | MINOR | /demo/ version clarity not verified (demo not running) | No |



---



\## AUDIT SCOPE COMPLETION



| Requirement | Status | Notes |

|-------------|--------|-------|

| Confirm CURRENT version via /versions/ | ✅ PASS | v0.2.2 confirmed as CURRENT |

| Audit v0.2.2 archive: status semantics | ⚠️ MINOR-1 | Status shown but not defined |

| Audit v0.2.2 archive: tier counts | ✅ PASS | 10/1/3 correct in all locations |

| Audit v0.2.2 archive: link completeness | ✅ PASS | All archive table links present and functional |

| Audit v0.2.2 archive: repo link correctness | ⚠️ MINOR-3 | Links use "main" not version tag |

| Audit /demo/ version clarity | ⚠️ MINOR-5 | Not verified (demo not running) |

| Audit "For Auditors" checklist | ❌ MAJOR-1, MAJOR-2, MAJOR-3 | Hardcoded wrong version, wrong links |

| Audit examples.json verifier UX | ⚠️ MINOR-4 | Packs declare v0.2.1 inside v0.2.2 file |



---



\## RECOMMENDATIONS (Tied to Findings)



1\. \*\*MAJOR-1, MAJOR-2, MAJOR-3:\*\* Update `docs/FOR\_AUDITORS.md` to replace all v0.2.1 references with v0.2.2. This should be a find-replace operation.



2\. \*\*MINOR-3:\*\* Update repo links in generated HTML to use version tags instead of "main" branch. Modify `scripts/build\_static\_site.py` to inject tag-based URLs.



3\. \*\*MINOR-4:\*\* Regenerate `evidence\_pack\_examples.v0.2.2.json` with `pack\_version: v0.2.2` or add prominent comment explaining why v0.2.1 is correct.



---



\## CONCLUSION



The v0.2.2 archive is \*\*structurally sound\*\* with correct tier counts, functional links, and proper status display. However, the \*\*"For Auditors" checklist is broken\*\* due to hardcoded v0.2.1 references, making it impossible for a cold auditor to execute the checklist without manual correction.



\*\*No blocking issues prevent deployment\*\*, but MAJOR-1, MAJOR-2, and MAJOR-3 should be fixed before claiming the checklist is "ready for external auditors."



