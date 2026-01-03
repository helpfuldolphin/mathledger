# Release Closure: v0.2.1

**Document Type**: Formal Release Closure
**Version**: v0.2.1
**Tag**: v0.2.1-cohesion
**Commit**: 27a94c8a58139cb10349f6418336c618f528cbab
**Date Locked**: 2026-01-03
**Closure Date**: 2026-01-03

---

## Purpose

This document formally closes v0.2.1 after external audit review. It records which findings were addressed, which are deferred, and establishes that no further changes will be made to v0.2.1 artifacts.

---

## External Audits Reviewed

| Audit | Date | Source | Scope |
|-------|------|--------|-------|
| Cold-Start Site Audit | 2026-01-03 | `docs/external_audits/manus_site_audit_2026-01-03.md` | Full site + demo evaluation from external auditor perspective |

---

## Findings Summary

### Critical Findings (Addressed in v0.2.1)

| # | Finding | Severity | Addressed? | Evidence |
|---|---------|----------|------------|----------|
| 1 | **Version discrepancy** - Demo showed v0.2.0 while archive showed v0.2.1 | HIGH | YES | Anti-drift gate added (`tools/check_hosted_demo_matches_release.py`), release pin validation in `/health` endpoint |
| 2 | **"90-Second Proof" misleading** - Button name didn't match actual timing (~5-8s) | MEDIUM | YES | Renamed to "Run Boundary Demo (approx 8s)" with clarification note |
| 3 | **"For Auditors" entry point missing** | MEDIUM | YES | Created `docs/FOR_AUDITORS.md` with 5-minute checklist |
| 4 | **Evidence pack tamper detection demo needed** | MEDIUM | YES | Created `releases/evidence_pack_examples.v0.2.1.json` with PASS/FAIL examples |

### Non-Critical Findings (Deferred to Future Versions)

| # | Finding | Severity | Deferred To | Rationale |
|---|---------|----------|-------------|-----------|
| 5 | **No explicit threat model** | LOW | v0.3.x | Requires adversary analysis; beyond v0 governance-only scope |
| 6 | **No failed verification examples** (infrastructure failures) | LOW | v0.3.x | v0 demonstrates governance, not failure modes |
| 7 | **No comparison to existing standards** (SOC 2, NIST AI RMF) | LOW | v0.3.x | Positioning document is marketing, not governance |
| 8 | **Scalability/performance claims absent** | LOW | v0.3.x | v0 is capability-agnostic by design |
| 9 | **Multi-stakeholder scenarios underexplored** | LOW | v0.3.x | Requires dispute resolution design |
| 10 | **Integration guidance missing** | LOW | v0.3.x | Requires stable API; v0 is demo-only |
| 11 | **Economic/incentive model absent** | LOW | v1.x | Production concern, not v0 scope |

---

## Strengths Validated by Audit

The external audit validated the following as genuinely novel or rigorous:

1. **Negative Capability Framing** - "What this version cannot enforce" prominence
2. **Tiered Enforcement Transparency** - A/B/C classification with explicit violation modes
3. **Abstention as First-Class Outcome** - ABSTAINED is not failure
4. **Exploration/Authority Boundary Enforcement** - DraftProposal never enters committed data
5. **Replay Verification with Explicit Non-Claims** - Proves structural integrity, not truth
6. **Immutable Versioned Archives** - Commit hashes, checksums, lock dates
7. **Governance Without Capability** - Substrate before capability

---

## Artifacts Included in v0.2.1

| Artifact | Path | Purpose |
|----------|------|---------|
| Demo | `demo/app.py` | Hosted interactive demo |
| API | `backend/api/uvil.py` | UVIL endpoints |
| Governance | `governance/` | Trust classes, validators, authority gate |
| Fixtures | `fixtures/` | 9 regression test cases |
| Evidence Pack Examples | `releases/evidence_pack_examples.v0.2.1.json` | Pre-computed PASS/FAIL packs |
| Static Archive | `site/v0.2.1/` | Immutable HTML archive |
| Auditor Guide | `docs/FOR_AUDITORS.md` | 5-minute verification checklist |
| Anti-Drift Gate | `tools/check_hosted_demo_matches_release.py` | CI-grade version check |

---

## Invariant Status at Closure

| Tier | Count | Description |
|------|-------|-------------|
| A | 10 | Cryptographically or structurally enforced |
| B | 1 | Logged and replay-visible |
| C | 3 | Documented but not enforced |

**Tier A Invariants (10)**:
1. Canonicalization Determinism (RFC 8785-style)
2. H_t = SHA256(R_t || U_t)
3. ADV Excluded from R_t
4. Content-Derived IDs
5. Replay Uses Same Code Paths
6. Double-Commit Returns 409
7. No Silent Authority
8. Trust-Class Monotonicity
9. Abstention Preservation
10. Audit Surface Version Field

---

## Closure Statement

**No further changes will be made to v0.2.1 artifacts.**

This closure applies to:
- All files in `site/v0.2.1/`
- All fixtures in `fixtures/` at tag v0.2.1-cohesion
- The `releases/evidence_pack_examples.v0.2.1.json` file
- All documentation referenced in `releases/releases.json` for v0.2.1

Any changes required will be released as v0.2.2 or later.

---

## Verification

To verify this closure:

```bash
# 1. Check tag exists
git tag -l "v0.2.1-cohesion"

# 2. Verify commit matches
git log -1 --format='%H' v0.2.1-cohesion
# Expected: 27a94c8a58139cb10349f6418336c618f528cbab

# 3. Run anti-drift gate
uv run python tools/check_hosted_demo_matches_release.py
# Expected: PASS

# 4. Verify evidence pack examples
uv run pytest tests/governance/test_evidence_pack_examples.py -v
# Expected: 16 passed
```

---

## Sign-Off

| Role | Date | Status |
|------|------|--------|
| External Audit Review | 2026-01-03 | COMPLETE |
| Findings Triage | 2026-01-03 | COMPLETE |
| Closure Documentation | 2026-01-03 | COMPLETE |

**Closed by**: Claude A
**Closure date**: 2026-01-03

---

*This document is part of the formal release process for MathLedger. It establishes an immutable record of what was reviewed, what was addressed, and what the version claims at the time of closure.*
