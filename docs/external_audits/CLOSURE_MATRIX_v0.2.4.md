# CLOSURE MATRIX: MathLedger v0.2.4

**Audit Date:** 2026-01-04
**Auditor Role:** Closure Auditor (Claude D)
**Target Version:** v0.2.4 (tag: v0.2.4-verifier-merkle-parity)
**Commit:** f58ff661e9b163c7babaa23d0df49d6f956cb6f7
**Prior Matrix:** CLOSURE_MATRIX_v0.2.3.md

---

## Executive Summary

| Severity | Total Findings | Fixed | By Design | Deferred | NOT CLOSED |
|----------|----------------|-------|-----------|----------|------------|
| BLOCKING | 11 | 9 | 0 | 0 | **2** |
| MAJOR | 13 | 11 | 2 | 0 | 0 |
| MINOR | 11 | 5 | 5 | 1 | 0 |

**Gate Decision:** **OUTREACH-NO-GO**

**Blocking Issues:**
1. Self-test reports "SELF-TEST FAILED" due to expected_reason string mismatch (spaces vs underscores)
2. FOR_AUDITORS Step 3 references non-existent "Download Evidence Pack" button

---

## Delta Since Previous Closure Matrix (v0.2.3 → v0.2.4)

### New Audits Added

| Audit File | Date | Auditor | Key Findings |
|------------|------|---------|--------------|
| claude_chrome_verifier_runtime_audit_2026-01-03_v0.2.3.md | 2026-01-03 | Claude Chrome | JS syntax error in v0.2.3 verifier |
| claude_chrome_verifier_audit_2026-01-03_v0.2.4_pre_sync.md | 2026-01-03 | Claude Chrome | Self-test scoring, demo sync issues |
| manus_demo_version_mismatch_2026-01-04_v0.2.4.md | 2026-01-04 | Manus | Demo v0.2.3 vs /versions/ v0.2.4 |
| claude_chrome_gate3_runtime_audit_2026-01-04_v0.2.4.md | 2026-01-04 | Claude Chrome | Self-test verdict logic inverted |
| manus_gate2_cold_start_audit_2026-01-04_v0.2.4.md | 2026-01-04 | Manus | Commit mismatch, tag mismatch, self-test ambiguous |

### v0.2.4 Delta (from releases.json)

**Fixed:**
- valid_boundary_demo self-test now returns PASS (was FAIL due to u_t_mismatch in v0.2.3)
- JS verifier uses Merkle tree with domain separation (Python parity)

**Changed:**
- Added domain constants: DOMAIN_UI_LEAF, DOMAIN_REASONING_LEAF, DOMAIN_LEAF, DOMAIN_NODE
- Added merkleRoot(), computeUiRoot(), computeReasoningRoot() functions
- examples.json regenerated with correct Merkle-based hashes
- Added test_verifier_golden_alignment.py (11 tests)
- Added test_evidence_pack_python_replay.py (8 tests)

---

## Closure Matrix — Findings Carried Forward from v0.2.3

All 28 findings from v0.2.3 remain CLOSED. No regressions detected.

| Finding ID | Status in v0.2.3 | Status in v0.2.4 | Notes |
|------------|------------------|------------------|-------|
| AUD-2026-01-03-B01 through B08 | Fixed | Fixed | No regressions |
| AUD-2026-01-03-M01 through M11 | Fixed/By Design | Fixed/By Design | No regressions |
| AUD-2026-01-03-N01 through N09 | Fixed/By Design/Deferred | Fixed/By Design/Deferred | No regressions |

---

## Closure Matrix — New Findings (v0.2.4)

### BLOCKING Findings

| Finding ID | Source Audit | Finding Summary | Status | Evidence | Regression Guard |
|------------|--------------|-----------------|--------|----------|------------------|
| AUD-2026-01-04-B01 | claude_chrome_verifier_runtime_audit_v0.2.3.md | v0.2.3 verifier JS SyntaxError (Unicode escape) | **Fixed** | Commit f58ff66; v0.2.4 verifier loads without errors | `tests/site/test_verifier_js_integrity.py` |
| AUD-2026-01-04-B02 | claude_chrome_verifier_audit_v0.2.4_pre_sync.md | Self-test scoring inverted (expected-FAIL marked as test failure) | **Fixed** | `testPack()` returns `pass: (expected === actual)`; line 267-270 | `tests/governance/test_verifier_golden_alignment.py` |
| AUD-2026-01-04-B03 | manus_demo_version_mismatch_v0.2.4.md | Demo shows v0.2.3 while /versions/ shows v0.2.4 | **Fixed** | Demo now serves v0.2.4; Deploy-by-Tag Doctrine added | `tools/e2e_audit_path.ps1` Phase 5 |
| AUD-2026-01-04-B04 | manus_gate2_cold_start_audit_v0.2.4.md | Commit hash mismatch: /versions/ vs archive page | **Fixed** | releases.json v0.2.4 commit is f58ff66; all pages now consistent | Build script uses releases.json as single source |
| AUD-2026-01-04-B05 | manus_gate2_cold_start_audit_v0.2.4.md | FOR_AUDITORS tag mismatch (verifier-syntax-fix vs verifier-merkle-parity) | **Fixed** | FOR_AUDITORS.md:13 uses `{{CURRENT_TAG}}` template variable | Build-time substitution from releases.json |
| AUD-2026-01-04-B06 | manus_gate2_cold_start_audit_v0.2.4.md | Self-test status "SELF-TEST FAILED" when all expectations match | **NOT CLOSED** | examples.json uses "h_t mismatch" (space); verifier uses "h_t_mismatch" (underscore) | **FIX REQUIRED: Align expected_reason strings** |
| AUD-2026-01-04-B07 | manus_gate2_cold_start_audit_v0.2.4.md | Archive claims demo is "live instantiation of same version" but commits differ | **Fixed** | Demo and archive now serve same commit (f58ff66) | Deploy-by-Tag Doctrine |

### MAJOR Findings

| Finding ID | Source Audit | Finding Summary | Status | Evidence | Regression Guard |
|------------|--------------|-----------------|--------|----------|------------------|
| AUD-2026-01-04-M01 | claude_chrome_verifier_audit_v0.2.4_pre_sync.md | Demo "Open Auditor Tool" link 404s | **Fixed** | Link now points to `/v0.2.4/evidence-pack/verify/`; resolves correctly | Link prefix derived from DEMO_VERSION constant |
| AUD-2026-01-04-M02 | claude_chrome_gate3_runtime_audit_v0.2.4.md | Manual FAIL result lacks explicit reason text | **By Design** | Visual hash diff is clear; status shows PASS/FAIL | Improvement suggestion, not blocking |
| AUD-2026-01-04-M03 | manus_gate2_cold_start_audit_v0.2.4.md | No "Download Evidence Pack" button after boundary demo | **NOT CLOSED** | FOR_AUDITORS.md:27 says "click 'Download Evidence Pack'" but button doesn't exist | **FIX REQUIRED: Update FOR_AUDITORS Step 3 or add button** |
| AUD-2026-01-04-M04 | manus_gate2_cold_start_audit_v0.2.4.md | Version identification requires following link to /versions/ | **By Design** | Landing page shows LOCKED status with link to /versions/; documented architecture | FOR_AUDITORS.md:188 |

### MINOR Findings

| Finding ID | Source Audit | Finding Summary | Status | Evidence | Regression Guard |
|------------|--------------|-----------------|--------|----------|------------------|
| AUD-2026-01-04-N01 | claude_chrome_verifier_audit_v0.2.4_pre_sync.md | Step 6 not tested (blocked by prior failure) | **By Design** | Cascade dependency; post-fix, Step 6 is executable | E2E audit path tests complete flow |
| AUD-2026-01-04-N02 | claude_chrome_verifier_audit_v0.2.4_pre_sync.md | Audit conducted during deployment sync window | **By Design** | PRE-SYNC designation preserved as evidence of transient state | Deploy-by-Tag Doctrine prevents recurrence |

---

## Open Findings Detail

### AUD-2026-01-04-B06: Self-Test String Mismatch (BLOCKING)

**Problem:**
- `examples.json` uses: `"expected_reason": "h_t mismatch"` (with space)
- `verifier/index.html` line 269 checks: `expectedReason==="h_t_mismatch"` (with underscore)
- Result: Self-test reports "SELF-TEST FAILED" even though all tests match expectations

**Files Affected:**
- `site/v0.2.4/evidence-pack/examples.json` lines 97, 179
- `site/v0.2.4/evidence-pack/verify/index.html` lines 267-269
- `releases/evidence_pack_examples.v0.2.4.json` lines 97, 179

**Required Fix:**
Change examples.json `expected_reason` values from:
- `"h_t mismatch"` → `"h_t_mismatch"`
- `"r_t mismatch"` → `"r_t_mismatch"`

**Verification:**
```
GET /v0.2.4/evidence-pack/verify/
Click "Run self-test vectors"
Expected: "SELF-TEST PASSED (3 vectors)"
```

---

### AUD-2026-01-04-M03: Missing Download Button (MAJOR → Upgraded to BLOCKING for audit path)

**Problem:**
FOR_AUDITORS.md Step 3 says:
> "After running the boundary demo, click 'Download Evidence Pack'"

But the boundary demo does NOT commit to the authority stream. It's demonstration mode only. There is no "Download Evidence Pack" button.

**Evidence:**
- AUTHORITY STREAM shows: "Nothing committed yet. Authority stream is empty."
- No download button exists after boundary demo completes

**Required Fix (one of):**
1. Update FOR_AUDITORS Step 3 to remove download instruction, OR
2. Add fallback: "If no evidence pack is available, use Ready-to-Verify examples", OR
3. Modify demo to produce downloadable evidence pack from boundary demo

**Verification:**
```
Follow FOR_AUDITORS checklist from Step 1 through Step 5
Expected: All steps executable without guessing
```

---

## Evidence Pack Verification

### examples.json Version Correctness
- **File:** `site/v0.2.4/evidence-pack/examples.json`
- **pack_version field:** `v0.2.4` (correct)
- **usage_instructions.step_2:** `https://mathledger.ai/v0.2.4/evidence-pack/verify/` (correct)
- **expected_reason values:** INCORRECT (spaces instead of underscores)

### Verifier Self-Test Logic
- **File:** `site/v0.2.4/evidence-pack/verify/index.html`
- **testPack() function:** Correctly computes `pass = (expected === actual)`
- **runSelfTest() function:** Correctly aggregates allPass
- **String literals:** Use underscores (h_t_mismatch, r_t_mismatch, u_t_mismatch)

### Tier A Invariant Count
- **releases.json:** `tier_a: 10`
- **invariants_status.md:** "Tier A: Enforced (10 invariants)"
- **Status:** Consistent

---

## Cross-Document Consistency Verification

| Claim | V0_LOCK.md | FOR_AUDITORS.md | invariants_status.md | HOW_THE_DEMO | Consistent |
|-------|------------|-----------------|----------------------|--------------|------------|
| Tier A count = 10 | ✓ | N/A | ✓ | N/A | PASS |
| MV returns VERIFIED | ✓ | ✓ | ✓ | ✓ | PASS |
| PA/FV/ADV → ABSTAINED | ✓ | ✓ | ✓ | ✓ | PASS |
| "v0 has no formal verifier" | ✓ | ✓ | ✓ | ✓ | PASS |
| ADV excluded from R_t | ✓ | ✓ | ✓ | ✓ | PASS |
| H_t = SHA256(R_t \|\| U_t) | ✓ | ✓ | ✓ | ✓ | PASS |

**Consistency Score:** 6/6 PASS

---

## Browser Verification Required (Gates)

### Claude Chrome (Runtime Breaker) Tasks

**Gate 3 Re-Run (Post-Fix):**

| Step | URL | Action | Pass Criteria |
|------|-----|--------|---------------|
| 1 | `https://mathledger.ai/versions/` | Identify CURRENT | v0.2.4 shown in green, commit f58ff66 |
| 2 | `https://mathledger.ai/demo/` | Check version banner | Shows `v0.2.4-verifier-merkle-parity | f58ff661e9b1` |
| 3 | `https://mathledger.ai/v0.2.4/evidence-pack/verify/` | Hard refresh (Ctrl+Shift+R) | Zero console errors |
| 4 | Same | Click "Run self-test vectors" | Status shows "SELF-TEST PASSED (3 vectors)" |
| 5 | Same | Table rows | valid_boundary_demo: PASS; tampered_ht_mismatch: PASS; tampered_rt_mismatch: PASS |
| 6 | Same | Paste tampered_ht_mismatch pack manually | Status shows "FAIL" with h_t mismatch highlighted |
| 7 | `https://mathledger.ai/demo/` | Click "Open Auditor Tool" | Navigates to `/v0.2.4/evidence-pack/verify/` (no 404) |

### Manus (Cold-Start Epistemic Gatekeeper) Tasks

**Gate 2 Re-Run (Post-Fix):**

| Step | Action | Pass Criteria |
|------|--------|---------------|
| 1 | Navigate to `https://mathledger.ai/` | Page loads; see LOCKED banner with link to /versions/ |
| 2 | Click /versions/ link | v0.2.4 shown as CURRENT (green, bold) |
| 3 | Click v0.2.4 link | Archive page shows commit f58ff661e9b1, tag v0.2.4-verifier-merkle-parity |
| 4 | Navigate to `/demo/` | Banner shows v0.2.4 with matching commit/tag |
| 5 | Run Boundary Demo | 4 outcomes: ABSTAINED, ABSTAINED, VERIFIED, REFUTED |
| 6 | Follow FOR_AUDITORS checklist | All 5 steps executable without guessing URLs |
| 7 | Use Ready-to-Verify path | Download examples.json → run self-test → "SELF-TEST PASSED (3 vectors)" |
| 8 | Verify `/v0.2.4/docs/for-auditors/` | Step 1.3 shows `{{CURRENT_TAG}}` or resolved value matching demo banner |

---

## Gate Decision

### **OUTREACH-NO-GO**

**Reason:** 2 BLOCKING issues remain open:

1. **AUD-2026-01-04-B06:** Self-test reports FAILED due to string mismatch
   - Impact: Acquisition committee sees "SELF-TEST FAILED" and loses confidence
   - Fix: Align expected_reason strings in examples.json (spaces → underscores)

2. **AUD-2026-01-04-M03:** FOR_AUDITORS Step 3 unexecutable
   - Impact: Auditor cannot complete checklist as written
   - Fix: Update Step 3 documentation or add fallback instruction

### Micro-Release Required

**Recommended version:** v0.2.5

**Changes required:**
1. Update `releases/evidence_pack_examples.v0.2.4.json`:
   - Line 97: `"expected_reason": "h_t_mismatch"` (underscore)
   - Line 179: `"expected_reason": "r_t_mismatch"` (underscore)
2. Regenerate `site/v0.2.4/evidence-pack/examples.json` from corrected source
3. Update FOR_AUDITORS.md Step 3 to include fallback:
   - "If no evidence pack is available after boundary demo, use Ready-to-Verify examples (Step 3b)"

**Post-fix verification:**
```powershell
.\tools\e2e_audit_path.ps1
# Expected: All checks pass including Phase 3 self-test
```

---

## Audit Trail

| Audit File | Date | Auditor | Version Audited | Findings Status |
|------------|------|---------|-----------------|-----------------|
| manus_site_audit_2026-01-03.md | 2026-01-03 | Manus | v0.2.1 | All closed |
| manus_link_integrity_audit_2026-01-03.md | 2026-01-03 | Manus | v0.2.1 | All closed |
| manus_hostile_audit_v0.2.2_2026-01-03.md | 2026-01-03 | Manus | v0.2.2 | All closed |
| manus_epistemic_integrity_audit_2026-01-03_v0.2.2.md | 2026-01-03 | Manus | v0.2.2/v0.2.3 | All closed |
| manus_epistemic_coherence_audit_2026-01-03_v0.2.3.md | 2026-01-03 | Manus | v0.2.3 | All closed |
| claude_extension_hostile_audit_2026-01-03_v0.2.3.md | 2026-01-03 | Claude Chrome | v0.2.3 | All closed |
| claude_chrome_verifier_runtime_audit_2026-01-03_v0.2.3.md | 2026-01-03 | Claude Chrome | v0.2.3 | All closed |
| claude_chrome_verifier_audit_2026-01-03_v0.2.4_pre_sync.md | 2026-01-03 | Claude Chrome | v0.2.4 | All closed |
| manus_demo_version_mismatch_2026-01-04_v0.2.4.md | 2026-01-04 | Manus | v0.2.4 | All closed |
| claude_chrome_gate3_runtime_audit_2026-01-04_v0.2.4.md | 2026-01-04 | Claude Chrome | v0.2.4 | 1 BLOCKING open |
| manus_gate2_cold_start_audit_2026-01-04_v0.2.4.md | 2026-01-04 | Manus | v0.2.4 | 2 BLOCKING open |

---

**Closure Matrix Generated:** 2026-01-04
**Auditor:** Claude D (Closure Auditor)
**Next Review:** Post micro-release (v0.2.5)

---

**SAVE TO REPO: YES**

Reason: This matrix documents the current closure state and identifies 2 BLOCKING issues that require a micro-release before outreach. The matrix provides actionable evidence for the required fixes.
