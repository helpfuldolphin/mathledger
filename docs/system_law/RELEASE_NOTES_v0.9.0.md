# Release Notes: v0.9.0-governance-p0

**Release Date**: 2025-12-17
**Tag**: `v0.9.0-governance-p0`
**Commit**: `e84c67f3fa0239683a23372a0b2d679b21a952e4`

---

## Summary

This release completes Phase 2 governance definition work identified in the STRATCOM Red Team Response. It establishes canonical SHADOW MODE semantics, adds cryptographic manifest signing, and implements documentation conformance enforcement.

---

## What Changed

### 1. SHADOW MODE Governance (Phase 2 Complete)

| Document | Purpose |
|----------|---------|
| `SHADOW_MODE_CONTRACT.md` | Canonical definition of SHADOW-OBSERVE and SHADOW-GATED sub-modes |
| `SHADOW_GRADUATION_POLICY.md` | Explicit prohibition of implicit graduation; re-verification required |
| `PHASE_2_GOVERNANCE_CLOSURE.md` | Closure record confirming Phase 2 completion |

**Key Change**: All references to "SHADOW MODE" without sub-mode qualification now default to SHADOW-OBSERVE per contract §1.1.

### 2. Manifest Signing (P0.2)

| Component | Path |
|-----------|------|
| `sign_manifest.py` | Ed25519 detached signature generation |
| `verify_manifest_signature.py` | Signature verification with fail-close semantics |
| `generate_signing_keypair.py` | Keypair generation utility |
| `MANIFEST_SIGNING_GUIDE.md` | Operator documentation |
| `MANIFEST_SIGNING_SMOKE_TEST.md` | Verification checklist |

**Key Change**: Evidence packs can now be cryptographically signed, providing provenance binding (not tamper-evidence without trusted timestamping).

### 3. Documentation Conformance

| Commit | Change |
|--------|--------|
| `a703d2b` | Replace prohibited "observational only" with contract-compliant language |
| `e84c67f` | Add legacy allowlist for pre-contract SHADOW MODE references |

**Key Change**: Evaluator-facing documentation now conforms to SHADOW_MODE_CONTRACT.md §4.1.

### 4. Shadow Release Gate

| Feature | Description |
|---------|-------------|
| Legacy allowlist | Pre-contract docs in `docs/system_law/` exempt from UNQUALIFIED warnings |
| `--strict` mode | CLI flag to enforce full compliance (no legacy exemptions) |
| Warning tracking | `legacy_warnings_suppressed` count in gate reports |

---

## What Did NOT Change

| Component | Status |
|-----------|--------|
| CAL-EXP-1/2/3 results | Frozen; no modifications |
| First Light harnesses | Frozen; determinism preserved |
| Dual-root attestation formula | `H_t = SHA256(R_t \|\| U_t)` unchanged |
| Evidence pack structure | Schema v1.0.0 unchanged |
| Lean verification logic | No changes to proof checking |

---

## Verification Commands

Run these three commands to verify the release:

```bash
# 1. Run manifest signing tests (19 tests)
uv run python -m pytest tests/evidence/test_manifest_signing.py tests/evidence/test_manifest_signing_e2e.py -v

# 2. Run shadow release gate tests
uv run python -m pytest tests/health/test_shadow_release_gate.py -v

# 3. Verify governance documents exist
ls -la docs/system_law/SHADOW_MODE_CONTRACT.md docs/system_law/SHADOW_GRADUATION_POLICY.md docs/system_law/PHASE_2_GOVERNANCE_CLOSURE.md
```

**Expected Results**:
- Command 1: 19 passed
- Command 2: All tests pass
- Command 3: All three files exist

---

## CI Status

| Workflow | Status | Run ID |
|----------|--------|--------|
| Core Loop Verification | Pending | — |
| Shadow Audit Gate | Pending | — |
| System Law Index Check | Pending | — |

*Note: CI links will be updated after tag push.*

---

## Provenance

### Commits in This Release (Phase 2 Scope)

| Commit | Description |
|--------|-------------|
| `15eea6d` | Add SHADOW_MODE_CONTRACT.md |
| `c6dcde7` | Add SHADOW_GRADUATION_POLICY.md |
| `a703d2b` | Documentation conformance (prohibited language replacement) |
| `70f65f7` | Phase 2 Governance Closure Record |
| `f211395` | Manifest signing implementation |
| `e84c67f` | Shadow release gate legacy allowlist |

### Deferred to Phase 3

| Item | Current State |
|------|---------------|
| Manifest signing (GPG/org key) | Dev key only; org key setup pending |
| Path traversal protection | Not implemented |
| Lean fallback enforcement | Partial |
| Isolation audit labeling | Self-reported |

---

## Breaking Changes

None. This release is backwards-compatible with v0.8.x.

---

## Upgrade Instructions

1. Pull latest changes: `git fetch && git checkout v0.9.0-governance-p0`
2. Sync dependencies: `uv sync`
3. Run verification commands above

---

**Release Captain**: Claude Code
**Reviewed By**: STRATCOM

