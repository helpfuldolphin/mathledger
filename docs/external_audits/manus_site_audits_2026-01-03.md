\# MathLedger: Top 5 Friction Points and Top 5 Credibility Signals



\## Top 5 Friction Points for a Cold Auditor



\### 1. \*\*Version Mismatch Between Archive and Demo\*\*

\*\*Severity:\*\* High  

\*\*Description:\*\* The archive shows v0.2.1 as CURRENT, but the demo runs v0.2.0. The `/demo/health` endpoint reports version 0.2.0 with tag v0.2.0-demo-lock, while the homepage shows v0.2.1 with tag v0.2.1-cohesion. Both share the same commit hash (27a94c8a5813), but the version inconsistency creates immediate doubt about whether the demo actually demonstrates what the archive documents.



\*\*Impact:\*\* A cold auditor's first verification step fails. This undermines trust in the entire system's coherence and raises questions about deployment processes and version control discipline.



\*\*Fix:\*\* Either update the demo to v0.2.1 or add a prominent notice explaining why the demo lags behind the archive version.



---



\### 2. \*\*No Ready-to-Verify Evidence Pack Available\*\*

\*\*Severity:\*\* High  

\*\*Description:\*\* The archive's "Evidence Pack" section contains only an `input.json` file (input specification), not a complete evidence pack with attestation hashes (U\_t, R\_t, H\_t). An auditor cannot immediately download and verify a pre-generated evidence pack without running the demo through a full scenario.



\*\*Impact:\*\* The core audit claim - "An auditor can recompute attestation hashes without running the demo" - cannot be verified from the static archive alone. This creates a catch-22: to audit the system, you must first use the system.



\*\*Fix:\*\* Provide at least one complete, pre-generated evidence pack in the archive with all hashes and replay verification instructions that work out-of-the-box.



---



\### 3. \*\*Missing "For Auditors" Entry Point\*\*

\*\*Severity:\*\* Medium  

\*\*Description:\*\* While there's a link to a "5-minute auditor checklist," it's not prominent enough and doesn't serve as a clear entry point. A cold auditor landing on the homepage must infer where to start. The site assumes familiarity with concepts like UVIL, trust classes, and Merkle roots without providing a guided path.



\*\*Impact:\*\* High cognitive load for first-time auditors. The site is optimized for demonstrating rigor to those who already understand what they're looking for, not for onboarding new auditors.



\*\*Fix:\*\* Create a dedicated `/for-auditors` landing page with:

\- "Start here if you're auditing this system for the first time"

\- Numbered steps with estimated time for each

\- Links to critical documents in priority order

\- Expected outputs for each verification step



---



\### 4. \*\*"90-Second Proof" Button Name Misleading\*\*

\*\*Severity:\*\* Low-Medium  

\*\*Description:\*\* The boundary demo button is labeled "Run 90-Second Proof" but the actual execution time varies significantly (observed: ~24 seconds in this test, ~5-8 seconds in previous test). The name creates an expectation that isn't met.



\*\*Impact:\*\* Minor credibility hit. In a system that emphasizes precision and explicit non-claims, a misleading button label stands out as inconsistent with the overall discipline.



\*\*Fix:\*\* Rename to "Run Boundary Demo" or "Same Claim, Different Authority Demo" to avoid timing expectations. If "90-Second" refers to something other than wall-clock time, explain it.



---



\### 5. \*\*No Failed Verification Examples\*\*

\*\*Severity:\*\* Medium  

\*\*Description:\*\* The site shows VERIFIED, REFUTED, and ABSTAINED outcomes, but doesn't demonstrate what happens when the verification infrastructure itself fails (e.g., validator crashes, hash computation errors, tampered evidence pack). There's no "negative test" showing the system catching an attack or failure.



\*\*Impact:\*\* An auditor cannot verify that the system actually detects tampering or failures. The claim "tampered evidence pack correctly returns FAIL with diff" is documented but not demonstrated in the archive.



\*\*Fix:\*\* Add a fixture or demo showing:

\- A tampered evidence pack with modified hash

\- The replay verification detecting the tampering

\- The specific error message and diff output



---



\## Top 5 Credibility Signals for a Cold Auditor



\### 1. \*\*Prominent Non-Claims and Limitations\*\*

\*\*Strength:\*\* Exceptional  

\*\*Description:\*\* The homepage displays "What this version cannot enforce" in red warning text before any feature descriptions. Every major document includes explicit sections on what is NOT claimed or NOT proven. The Invariants page is titled "brutally honest classification" and includes a column for "How It Can Be Violated Today."



\*\*Why It Matters:\*\* This inverted framing is unprecedented in AI demos. Most systems lead with capabilities; MathLedger leads with limitations. This signals intellectual honesty and reduces the risk of overclaiming - a critical concern in AI safety.



\*\*Auditor Confidence:\*\* High. The system is more likely to be trustworthy if it explicitly states its boundaries than if it makes broad claims.



---



\### 2. \*\*Tiered Enforcement Transparency (A/B/C)\*\*

\*\*Strength:\*\* Exceptional  

\*\*Description:\*\* The tier system provides granular transparency about what is cryptographically enforced (Tier A: 10 invariants), logged but not prevented (Tier B: 1 invariant), and aspirational (Tier C: 3 invariants). The Invariants table shows "How It Can Be Violated Today" and "Current Detection" for each.



\*\*Why It Matters:\*\* This level of attack surface transparency is typically reserved for internal security documentation. Publishing it demonstrates confidence in the architecture and enables meaningful external audit.



\*\*Auditor Confidence:\*\* High. The system distinguishes between "we claim this is secure" and "we can prove this is secure," which is rare in AI governance demos.



---



\### 3. \*\*Immutable Versioned Archives with Cryptographic Verification\*\*

\*\*Strength:\*\* High  

\*\*Description:\*\* Each version is locked with:

\- Explicit lock date (2026-01-03)

\- Git commit hash (27a94c8a58139cb10349f6418336c618f528cbab)

\- Version tag (v0.2.1-cohesion)

\- Build timestamp (2026-01-03T18:55:59Z)

\- Checksum manifest (manifest.json)

\- Verification instructions (clone repo, checkout commit, compare files)



\*\*Why It Matters:\*\* This creates a verifiable historical record. An auditor can cryptographically verify that the archive matches the source code at a specific point in time. The "epistemic archive" concept is philosophically rigorous.



\*\*Auditor Confidence:\*\* High. The archive is not just documentation - it's a verifiable artifact with tamper detection.



---



\### 4. \*\*Abstention as First-Class Outcome\*\*

\*\*Strength:\*\* High  

\*\*Description:\*\* ABSTAINED is treated as a legitimate, non-failure outcome. The system structurally enforces that it cannot be silently converted to a claim (Tier A invariant: "Abstention Preservation"). The documentation states: "This stopping is correctness, not caution."



\*\*Why It Matters:\*\* Most AI systems are incentivized to always produce an answer. MathLedger's architecture makes "I don't know" a core feature, which is critical for safety-critical applications where false confidence is more dangerous than no answer.



\*\*Auditor Confidence:\*\* High. The system's willingness to abstain signals that it won't produce confident outputs when it lacks grounds for confidence.



---



\### 5. \*\*Consistent Terminology Across Archive and Demo\*\*

\*\*Strength:\*\* High  

\*\*Description:\*\* Trust classes (FV, MV, PA, ADV), outcomes (VERIFIED, REFUTED, ABSTAINED), and governance concepts (UVIL, R\_t, U\_t, H\_t) are defined identically in the archive and demo. The demo behavior matches the documented limitations (e.g., PA returns ABSTAINED, ADV excluded from R\_t).



\*\*Why It Matters:\*\* Terminology consistency indicates that the archive and demo were designed together, not "stitched together" after the fact. This coherence suggests the governance claims are testable and the demo is a genuine instantiation of the documented architecture.



\*\*Auditor Confidence:\*\* Medium-High. While the version mismatch (v0.2.1 vs v0.2.0) undermines this somewhat, the core terminology and behavior remain consistent.



---



\## Summary Table



| Friction Points | Severity | Impact on Audit |

|----------------|----------|-----------------|

| 1. Version mismatch (archive v0.2.1 vs demo v0.2.0) | High | First verification fails |

| 2. No ready-to-verify evidence pack | High | Core audit claim unverifiable |

| 3. Missing "For Auditors" entry point | Medium | High cognitive load |

| 4. Misleading button name ("90-Second Proof") | Low-Medium | Minor credibility hit |

| 5. No failed verification examples | Medium | Cannot verify tamper detection |



| Credibility Signals | Strength | Confidence Boost |

|--------------------|----------|------------------|

| 1. Prominent non-claims and limitations | Exceptional | Reduces overclaiming risk |

| 2. Tiered enforcement transparency (A/B/C) | Exceptional | Enables meaningful audit |

| 3. Immutable versioned archives | High | Cryptographic verification |

| 4. Abstention as first-class outcome | High | Safety-critical feature |

| 5. Consistent terminology | High | Coherent system design |



---



\## Net Assessment



\*\*Overall Credibility:\*\* \*\*High\*\*, but undermined by specific execution gaps.



The \*\*credibility signals\*\* are genuinely exceptional - the negative framing, tiered transparency, and abstention-as-feature are novel in AI governance. However, the \*\*friction points\*\* create immediate barriers to audit:



1\. The version mismatch makes the first verification step fail

2\. The missing evidence pack makes the core audit claim unverifiable without running the demo



These are \*\*fixable issues\*\* that don't undermine the fundamental design, but they significantly impact the "cold auditor" experience. The system demonstrates unusual rigor in design but needs better execution in deployment and artifact availability.



\*\*Recommendation:\*\* Fix the top 2 friction points (version consistency + pre-generated evidence pack) to match the exceptional credibility signals. The system has earned trust through its design discipline - don't lose it through deployment gaps.





