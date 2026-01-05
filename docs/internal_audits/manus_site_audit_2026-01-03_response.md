# MathLedger Audit Response: Manus Site Audit 2026-01-03

**Response Date:** 2026-01-03
**Responding Party:** Claude C (adversarial review persona)
**Original Audit:** `docs/external_audits/manus_site_audit_2026-01-03.md`
**Response Type:** Categorized findings with disposition

---

## Response Categories

| Category | Count | Meaning |
|----------|-------|---------|
| **Accepted** | 5 | We will implement changes |
| **Intentional Friction** | 4 | We will NOT change; rationale provided |
| **Out-of-Scope v0.x** | 8 | Explicitly deferred to v1+ |

---

## Section A: Accepted Findings (Will Change)

### A1: "90-Second Proof" Button Name Misleading

**Manus Finding (Part 4):**
> "The button name '90-Second Proof' is misleading - the demo takes only 5-8 seconds, not 90 seconds."

**Disposition:** ACCEPTED

**Concrete Change:**
| Item | Value |
|------|-------|
| File | `demo/app.py` |
| Change Type | Copy edit |
| Current Text | `"Run 90-Second Proof"` |
| New Text | `"Run Boundary Demo"` |
| Line (approx) | Search for `90-Second` in boundary demo section |

**Rationale:** The "90 seconds" was intended to mean "understand in 90 seconds" but reads as animation duration. Clearer to remove the timing claim entirely.

**Verification Step:**
```powershell
curl.exe -s https://mathledger.ai/demo/ | findstr "90-Second"
# Expected: No output (string removed)
```

---

### A2: Version Number Discrepancy (Demo v0.2.0 vs Archive v0.2.1)

**Manus Finding (Part 5):**
> "Version number discrepancy: The demo shows 'v0.2.0' while the archive shows 'v0.2.1'"

**Disposition:** ACCEPTED

**Concrete Change:**
| Item | Value |
|------|-------|
| File | `demo/app.py` |
| Change Type | Version constant update |
| Current Value | `BUILD_VERSION = "v0.2.0"` |
| New Value | `BUILD_VERSION = "v0.2.1"` |
| Secondary File | `releases/releases.json` |
| Secondary Change | Update `current_version` field |

**Rationale:** Demo and archive must report the same version for coherence.

**Verification Step:**
```powershell
curl.exe -s https://mathledger.ai/demo/healthz | findstr "version"
# Expected: "version": "v0.2.1"

curl.exe -s https://mathledger.ai/v0.2.1/manifest.json | findstr "version"
# Expected: "version": "v0.2.1"
```

---

### A3: "FM Section" References Unexplained

**Manus Finding (Part 2):**
> "The Invariants page references 'FM Section' repeatedly (§1.5, §4, etc.) but doesn't explain what FM is or link to it."

**Disposition:** ACCEPTED

**Concrete Change:**
| Item | Value |
|------|-------|
| File | `scripts/build_static_site.py` (invariants page builder) |
| Change Type | Add tooltip/link |
| Current | `FM §1.5` (unlinked) |
| New | `FM §1.5` with tooltip: "FM = Field Manual (internal technical specification)" |
| Alternative | Add footnote at bottom of invariants page |

**Rationale:** External auditors should not need to guess what FM means.

**Verification Step:**
```powershell
curl.exe -s https://mathledger.ai/v0.2.1/docs/invariants/ | findstr "Field Manual"
# Expected: Match found (tooltip or footnote text)
```

---

### A4: "For Auditors" Entry Point Insufficiently Prominent

**Manus Finding (Part 6):**
> "While there's a '5-minute auditor checklist' link, it's not prominent enough. An auditor landing on the homepage would need to infer that this is the right starting point."

**Disposition:** ACCEPTED

**Concrete Change:**
| Item | Value |
|------|-------|
| File | `scripts/build_static_site.py` (landing page builder) |
| Change Type | Add prominent banner |
| Location | Above "Open Interactive Demo" button |
| New Element | Warning-styled box: "Auditing MathLedger? Start with the [5-minute checklist](/docs/ACQUIRER_QA_CHECKLIST.md)" |

**Rationale:** Auditors are a primary audience; entry point should be unmissable.

**Verification Step:**
```powershell
curl.exe -s https://mathledger.ai/v0.2.1/ | findstr -i "auditing"
# Expected: "Auditing MathLedger?" text found
```

---

### A5: Evidence Pack Verifier Page 404

**Manus Finding:** (Discovered during QA checklist verification)
> Demo links to `/v0.2.0/evidence-pack/verify/` but page returns 404.

**Disposition:** ACCEPTED

**Concrete Change:**
| Item | Value |
|------|-------|
| File | `scripts/build_static_site.py` |
| Change Type | Build check enforcement |
| Action | Ensure verifier page is generated and deployed |
| Secondary | Cloudflare Pages redeployment required |

**Rationale:** Demo cannot link to non-existent page.

**Verification Step:**
```powershell
curl.exe -sI https://mathledger.ai/v0.2.1/evidence-pack/verify/
# Expected: HTTP/1.1 200 OK
```

---

## Section B: Intentional Friction (Will NOT Change)

### B1: Target Audience Assumes High Technical Sophistication

**Manus Finding (Part 2):**
> "The site assumes significant technical sophistication (terms like 'Merkle root,' 'RFC 8785-style canonicalization,' 'content-derived IDs') without clear onboarding for different audience levels."

**Disposition:** INTENTIONAL FRICTION - NO CHANGE

**Rationale:**
1. MathLedger is a **governance substrate**, not a consumer product
2. Target audience is safety leads, auditors, and technical acquirers who understand cryptographic primitives
3. Simplifying terminology would introduce imprecision that undermines the rigor claim
4. The "5-minute auditor checklist" provides a gentler entry point for those who need it

**Supporting Evidence:**
> From V0_LOCK.md: "This is a governance substrate demo, not a capability demo."

The technical language IS the product. Dumbing it down would be dishonest.

---

### B2: Negative Framing More Prominent Than Features

**Manus Finding (Part 1):**
> "The most striking feature is the prominence of non-claims and limitations."

**Disposition:** INTENTIONAL FRICTION - NO CHANGE

**Rationale:**
1. This is the core design philosophy, not a bug
2. Leading with limitations prevents capability inflation
3. Auditors value explicit non-claims over implicit promises
4. The HOSTILE_DEMO_REHEARSAL.md Red Lines table exists precisely because overclaiming is dangerous

**Supporting Evidence:**
> From HOSTILE_DEMO_REHEARSAL.md: "If asked 'what can this do?', redirect to 'what does this *refuse* to do?'"

The negative framing is our primary differentiator from marketing-driven AI demos.

---

### B3: "Governance Demo" Value Proposition Requires Inference

**Manus Finding (Part 2):**
> "While the site repeatedly states this is a 'governance substrate demo,' it's not immediately clear what problem this solves or why governance without capability matters."

**Disposition:** INTENTIONAL FRICTION - NO CHANGE

**Rationale:**
1. If you don't understand why governance without capability matters, you are not the target audience
2. Explaining "why this matters" would require capability claims we refuse to make
3. The value proposition is self-evident to safety researchers: you can audit the boundary before trusting the interior
4. Adding marketing copy would undermine the "no claims" discipline

**Supporting Evidence:**
> From Explanation doc: "This stopping is correctness, not caution. A system that produces confident outputs when it lacks grounds for confidence is broken."

The audience who needs this explained is not our audience.

---

### B4: Demo Animation Timing (~5-8 seconds)

**Manus Finding (Part 4):**
> "The demo completed in approximately 5-8 seconds (not 90 seconds as the button name suggests)."

**Disposition:** INTENTIONAL FRICTION - PARTIAL CHANGE

**Rationale:**
- The button name "90-Second Proof" is being changed (see A1)
- But the animation timing itself (5-8 seconds) is intentional
- Sequential reveal helps users read each outcome before the next appears
- Instant display would overwhelm; too slow would bore

**Action:** Button rename only. Animation timing unchanged.

---

## Section C: Out-of-Scope for v0.x (Explicitly Deferred)

### C1: Threat Model Page

**Manus Finding (Part 6):**
> "Create a document titled 'Threat Model and Attack Surface' that explicitly lists adversaries the system is designed to resist and those it is not."

**Disposition:** DEFERRED TO v1.0

**Rationale:**
- v0.x is a governance demo, not a production system
- Threat modeling against a demo creates false precision
- When verifiers exist (v1+), threat model becomes meaningful

**Tracking:** Add to `docs/ROADMAP.md` under v1.0 requirements

---

### C2: Failed Verification Examples (Infrastructure Failures)

**Manus Finding (Part 6):**
> "Show a case where the verification infrastructure itself fails (not just ABSTAINED, but an actual system error)."

**Disposition:** DEFERRED TO v1.0

**Rationale:**
- v0 has minimal infrastructure (in-memory, single node)
- Infrastructure failure modes are v1 concerns (persistence, multi-node)
- Current demo shows governance boundaries, not failure handling

**Tracking:** Add fixture `infrastructure_failure` in v1.0

---

### C3: Comparison to Existing Standards (SOC 2, NIST AI RMF, etc.)

**Manus Finding (Part 6):**
> "Position MathLedger relative to existing audit standards, AI governance frameworks, formal verification systems."

**Disposition:** DEFERRED TO v1.0

**Rationale:**
- Comparison requires stable feature set to compare against
- v0.x is intentionally minimal; comparison would be mostly "not applicable"
- v1.0 with real verifiers enables meaningful comparison

**Tracking:** Add to `docs/ROADMAP.md` as v1.0 documentation requirement

---

### C4: Scalability and Performance Claims

**Manus Finding (Part 6):**
> "There are no claims about performance, throughput, or scalability."

**Disposition:** DEFERRED TO v1.0+

**Rationale:**
- v0 is in-memory, restart-loss-accepted
- Performance claims on a demo are meaningless
- When persistence exists, benchmarking becomes relevant

**Tracking:** Performance passport framework exists but is not public in v0.x

---

### C5: Multi-Stakeholder Scenarios

**Manus Finding (Part 6):**
> "What happens when multiple parties commit conflicting claims about the same fact?"

**Disposition:** DEFERRED TO v1.0

**Rationale:**
- v0 is single-user demo
- Multi-party governance requires auth (not in v0)
- Conflict resolution is a v1 feature, not governance substrate

**Tracking:** Multi-party attestation in v1.0 scope

---

### C6: Integration Guidance

**Manus Finding (Part 6):**
> "There's no 'How to Integrate MathLedger' guide."

**Disposition:** DEFERRED TO v1.0

**Rationale:**
- v0 is not designed for integration; it's a demo
- Integration guides for demos create false expectations
- When API stabilizes (v1), integration docs become appropriate

**Tracking:** Add to v1.0 documentation plan

---

### C7: Economic and Incentive Model

**Manus Finding (Part 6):**
> "There's no discussion of incentives. Why would users commit claims? Why would validators participate?"

**Disposition:** DEFERRED TO v2.0+

**Rationale:**
- v0/v1 are infrastructure demos, not incentive mechanisms
- Economic models are out of scope for governance substrate
- This is a research direction, not a v1 deliverable

**Tracking:** Mark as "future research" in roadmap

---

### C8: UVIL Acronym Explanation Depth

**Manus Finding (Part 2):**
> "'User-Verified Input Loop' is mentioned but not explained in depth on the homepage."

**Disposition:** DEFERRED TO v0.2.2

**Rationale:**
- Adding UVIL explanation to landing page is low priority
- Explanation doc covers UVIL in detail
- Can add tooltip in minor release

**Tracking:** Minor enhancement, not blocking for v0.2.1

---

## Auditor Re-Test Script (5 Minutes)

After changes are deployed, Manus (or any auditor) can verify with:

```powershell
# ============================================
# MathLedger Audit Response Verification
# Estimated time: 5 minutes
# ============================================

Write-Host "=== A1: Button Name Fix ===" -ForegroundColor Cyan
$demo = (Invoke-WebRequest -Uri "https://mathledger.ai/demo/" -UseBasicParsing).Content
if ($demo -match "90-Second") {
    Write-Host "FAIL: '90-Second' still present" -ForegroundColor Red
} else {
    Write-Host "PASS: '90-Second' removed" -ForegroundColor Green
}

Write-Host "`n=== A2: Version Consistency ===" -ForegroundColor Cyan
$demoHealth = (Invoke-WebRequest -Uri "https://mathledger.ai/demo/healthz" -UseBasicParsing).Content | ConvertFrom-Json
$expectedVersion = "v0.2.1"
if ($demoHealth.version -eq $expectedVersion) {
    Write-Host "PASS: Demo version = $expectedVersion" -ForegroundColor Green
} else {
    Write-Host "FAIL: Demo version = $($demoHealth.version), expected $expectedVersion" -ForegroundColor Red
}

Write-Host "`n=== A3: FM Explanation ===" -ForegroundColor Cyan
$invariants = (Invoke-WebRequest -Uri "https://mathledger.ai/$expectedVersion/docs/invariants/" -UseBasicParsing).Content
if ($invariants -match "Field Manual") {
    Write-Host "PASS: FM explanation found" -ForegroundColor Green
} else {
    Write-Host "FAIL: FM explanation missing" -ForegroundColor Red
}

Write-Host "`n=== A4: Auditor Entry Point ===" -ForegroundColor Cyan
$landing = (Invoke-WebRequest -Uri "https://mathledger.ai/$expectedVersion/" -UseBasicParsing).Content
if ($landing -match "(?i)audit") {
    Write-Host "PASS: Auditor entry point found" -ForegroundColor Green
} else {
    Write-Host "FAIL: Auditor entry point missing" -ForegroundColor Red
}

Write-Host "`n=== A5: Verifier Page Exists ===" -ForegroundColor Cyan
try {
    $verifier = Invoke-WebRequest -Uri "https://mathledger.ai/$expectedVersion/evidence-pack/verify/" -UseBasicParsing
    if ($verifier.StatusCode -eq 200) {
        Write-Host "PASS: Verifier page returns 200" -ForegroundColor Green
    }
} catch {
    Write-Host "FAIL: Verifier page returns $($_.Exception.Response.StatusCode)" -ForegroundColor Red
}

Write-Host "`n=== B1-B4: Intentional Friction (No Change Expected) ===" -ForegroundColor Cyan
Write-Host "SKIP: These are documented as intentional, no verification needed" -ForegroundColor Yellow

Write-Host "`n=== C1-C8: Deferred Items (Not in v0.2.1) ===" -ForegroundColor Cyan
Write-Host "SKIP: These are explicitly deferred to v1.0+" -ForegroundColor Yellow

Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "Run the boundary demo manually to verify animation still works."
Write-Host "Check demo sidebar links to documentation."
Write-Host "Download evidence pack and verify in auditor tool."
```

---

## Change Implementation Tracking

| Finding | Status | PR/Commit | Verified |
|---------|--------|-----------|----------|
| A1: Button name | PENDING | - | [ ] |
| A2: Version sync | PENDING | - | [ ] |
| A3: FM explanation | PENDING | - | [ ] |
| A4: Auditor banner | PENDING | - | [ ] |
| A5: Verifier deploy | PENDING | - | [ ] |

---

## Constraints Observed

- [x] No new claims added
- [x] No edits to fm.tex/fm.pdf
- [x] Response memo only, no implementation

---

**SAVE TO REPO: YES**

**Path:** `docs/external_audits/manus_site_audit_2026-01-03_response.md`
