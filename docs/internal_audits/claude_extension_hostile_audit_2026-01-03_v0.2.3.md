# HOSTILE AUDIT: MathLedger v0.2.3 (Claude Chrome / Opus)

**Audit Date:** 2026-01-03
**Auditor Role:** External hostile auditor for acquisition committee
**Target Version:** v0.2.3 (tag: v0.2.3-audit-path-freshness)
**Verdict:** **FAIL (not outreach-ready)**

## Executive Summary

The **Evidence Pack Verifier** — the core auditor artifact — was **non-functional** due to JavaScript errors.
This blocks the primary audit workflow (demo → evidence pack → verifier → PASS/FAIL).

## Blocking Issues

### BLOCKING-1: Verifier JavaScript broken (self-test + manual verification fail)
- **URL:** https://mathledger.ai/v0.2.3/evidence-pack/verify/
- **Expected:** "Run self-test vectors" executes and prints PASS/FAIL table; manual verification works.
- **Actual:** JS errors prevent any verification.
- **Console Evidence:**
  - `SyntaxError: Invalid Unicode escape sequence (verify/:48:406)`
  - `ReferenceError: runSelfTest is not defined (verify/:22:123)`
- **Impact:** Auditors cannot verify evidence packs; audit path is non-executable.

## Major Issues

### MAJOR-1: Demo "Open Auditor Tool" link returns 404
- **Broken URL:** https://mathledger.ai/demo/v0.2.3/evidence-pack/verify/
- **Expected:** https://mathledger.ai/v0.2.3/evidence-pack/verify/
- **Impact:** Users following the demo workflow cannot reach verifier.

### MAJOR-2: Fixture directory links return 404 (UI navigation broken)
- **Broken example:** https://mathledger.ai/v0.2.3/fixtures/mv_arithmetic_verified/
- **Working workaround:** https://mathledger.ai/v0.2.3/fixtures/mv_arithmetic_verified/input.json
- **Impact:** Fixture UI page links are unusable; auditors cannot browse fixtures via the site.

### MAJOR-3: v0.2.1 archive still displays "Status: CURRENT"
- **URL:** https://mathledger.ai/v0.2.1/
- **Expected:** LOCKED / superseded semantics, deferring to /versions/
- **Impact:** Misleads auditors who land on older versions directly.

## Minor Issues

### MINOR-1: Demo banner commit differs from archive commit
- **Demo banner:** `v0.2.3 | ... | 674bcd16104f`
- **Archive commit:** `27a94c8a5813`
- **Impact:** Potential confusion about what is being audited.

### MINOR-2: "DEMO OUT OF SYNC" warning element exists but not visible
- **URL:** https://mathledger.ai/v0.2.3/
- **Finding:** hidden warning text exists but does not render; mismatch detection may not fire.

## What Worked / Positive Notes

- LOCKED status pattern + /versions as canonical authority looks correct (v0.2.2+).
- Explicit non-claims maintained ("not capability", no "safe/aligned/intelligent").
- Tiered invariants A/B/C presented clearly.
- examples.json appears sound (pack_version correct, PASS+FAIL cases).
- Demo boundary demonstration runs and shows 4 outcome types.

## URLs Referenced

| URL | Result |
|---|---|
| https://mathledger.ai | 302 → /v0.2.3/ |
| https://mathledger.ai/versions/ | 200 OK |
| https://mathledger.ai/v0.2.3/ | 200 OK |
| https://mathledger.ai/demo/ | 200 OK |
| https://mathledger.ai/demo/healthz | 200 OK |
| https://mathledger.ai/v0.2.3/evidence-pack/verify/ | 200 OK but JS broken |
| https://mathledger.ai/demo/v0.2.3/evidence-pack/verify/ | 404 |
| https://mathledger.ai/v0.2.3/fixtures/mv_arithmetic_verified/ | 404 |
| https://mathledger.ai/v0.2.3/fixtures/mv_arithmetic_verified/input.json | 200 |

## Recommendations (Auditor)

1) Fix verifier JS immediately; add build-time assertion that verifier loads without JS errors.
2) Add CI assertions for link integrity (archive contents + fixtures + demo→verifier routing).
3) Ensure mismatch warnings are visible and do not rely solely on client-side JS.
4) Add E2E audit-path smoke test (demo → evidence pack → verifier PASS).
5) Backfill superseded archives with LOCKED semantics (or dynamic banner).

**End of report.**
