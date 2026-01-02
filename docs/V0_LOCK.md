# V0 Scope Lock

This document defines the scope of MathLedger Demo v0. It exists to prevent drift.

---

## What v0 Demonstrates

v0 is a governance demo. It demonstrates:

1. **UVIL (User-Verified Input Loop)**: Human binding mechanism for authority
2. **Trust Classes**: FV, MV, PA (authority-bearing) vs ADV (exploration-only)
3. **Dual Attestation Roots**: U_t (UI), R_t (reasoning), H_t (composite)
4. **Determinism**: Same inputs produce same outputs, replayable
5. **Exploration/Authority Boundary**: DraftProposal never enters hash-committed paths

---

## What v0 Does NOT Demonstrate

v0 explicitly excludes:

- **RFL learning loop**: No curriculum, no policy, no uplift
- **Multi-model arena**: Single template partitioner, no LiteLLM, no model competition
- **Agent tools**: No code execution, no sandbox, no E2B
- **Real verifier**: No Lean, no Z3, no mechanical proof checking
- **Production auth**: No user accounts, no API keys, no rate limiting
- **Persistence**: In-memory only, restart-loss accepted
- **Long-running agents**: Synchronous request/response only

---

## What v0 IS

v0 is a **governance substrate demo**, not a capability demo.

It shows:
- The boundary between exploration and authority is real (enforced in code)
- The boundary is testable (fixtures + harness)
- The boundary is replayable (determinism + content-derived IDs)
- The system stops when it cannot verify (ABSTAINED is correct behavior)

It does not show:
- That the system is intelligent
- That the system is aligned
- That the system is safe
- That verification works (v0 has no verifier)

---

## Terminology Constraints

v0 uses these outcome values:

| Outcome | Meaning in v0 |
|---------|---------------|
| VERIFIED | MV claim parsed AND arithmetic confirmed (e.g., "2 + 2 = 4") |
| REFUTED | MV claim parsed AND arithmetic failed (e.g., "2 + 2 = 5") |
| ABSTAINED | Cannot mechanically verify (PA, FV, unparseable MV, ADV-only) |

v0 uses "authority-bearing" to mean "accepted into reasoning stream (R_t)":
- FV, MV, PA: authority-bearing (enter R_t)
- ADV: exploration-only (excluded from R_t)

v0 does NOT claim that authority-bearing means "verified":
- PA is authority-bearing because a human attests, not because the system verified
- In v0, no claims are mechanically verified

---

## Allowed Iterations

Within v0 scope, you may iterate on:

- Template partitioner logic
- UI copy and explanation text
- Trust class selection UX
- Fixture set for regression testing
- Determinism tests and golden hashes

---

## Forbidden Iterations

Do not add to v0:

- Model selection or switching
- Tool execution or sandboxing
- External API calls in demo flow
- Persistence beyond session
- Claims of verification when no verifier exists
- Claims of safety, alignment, or intelligence

---

## Lock Statement

v0 is complete when:

1. The demo runs locally with a single command
2. The harness passes all 5 cases
3. The UI clearly shows exploration vs authority streams
4. The UI does not claim VERIFIED for any claim (v0 has no verifier)
5. The fixtures are stable (golden hashes do not drift)

v0 is not complete when:

1. Features are added beyond the scope above
2. Outcomes are mislabeled (e.g., PA shown as "VERIFIED")
3. The exploration/authority boundary is blurred

---

## Date Locked

2026-01-02

This scope is frozen. Changes require explicit scope unlock with rationale.

---

## Release Notes: v0-demo-lock

**Tag**: `v0-demo-lock`
**Commit**: `ab8f51ab389aed7b3412cb987fc70d0d4f2bbe0b`
**Date**: 2026-01-02

### Key Deliverables

| Artifact | Path | Purpose |
|----------|------|---------|
| Demo entrypoint | `demo/app.py` | Single-command runnable: `uv run python demo/app.py` |
| API endpoints | `backend/api/uvil.py` | `/propose_partition`, `/commit_uvil`, `/run_verification`, `/evidence_pack`, `/replay_verify` |
| Governance models | `governance/uvil.py` | DraftProposal, CommittedPartitionSnapshot, attestation |
| Trust classes | `governance/trust_class.py` | FV, MV, PA, ADV enum + authority-bearing check |
| MV validator | `governance/mv_validator.py` | Arithmetic equality validator for MV claims |
| Regression harness | `tools/run_demo_cases.py` | 9 demo cases + evidence pack tests |
| Fixtures | `fixtures/<case>/` | 9 cases including mv_arithmetic_verified, same_claim_as_pa |

### Invariants Verified

1. **T1**: proposal_id never appears in hash-committed payloads
2. **T2**: ADV claims excluded from R_t computation
3. **T3**: H_t = SHA256(R_t || U_t) deterministic
4. **T4**: Double-commit of proposal_id returns 409 Conflict
5. **T5**: /run_verification rejects raw proposal_id
6. **T6**: Outcomes are VERIFIED/REFUTED only via MV arithmetic validator; else ABSTAINED
7. **T7**: PA claims show "authority-bearing but not mechanically verified"
8. **T8**: UI clearly separates Exploration Stream from Authority Stream

### Evidence Pack Invariants

9. **T9**: Evidence pack contains ONLY minimum for replay: uvil_events, reasoning_artifacts, u_t, r_t, h_t
10. **T10**: Replay verification uses SAME code paths as live (compute_ui_root, compute_reasoning_root, compute_composite_root)
11. **T11**: No external API calls required for replay verification
12. **T12**: Tampered evidence pack correctly returns FAIL with diff

### Critical Fix

- **PA terminology hazard**: PA claims no longer return `VERIFIED`. All claims in v0 return `ABSTAINED` with explicit `authority_basis` explanation.

### How to Verify

```bash
# Start demo
uv run python demo/app.py

# Run regression harness (in another terminal)
uv run python tools/run_demo_cases.py

# Expected: 9/9 cases succeed

# Run evidence pack tests
uv run python tools/run_demo_cases.py --evidence-pack-tests

# Expected: All evidence pack tests PASS

# Run UI self-explanation tests
uv run python tools/run_demo_cases.py --ui-tests

# Expected: UI Self-Explanation Tests: ALL PASSED
```

---

## Release Notes: v0.2.0

**Tag**: `v0.2.0-ui-self-explanation`
**Date**: 2026-01-02

### What Changed

This release adds UI self-explanation to the demo. The demo now explains itself in real-time.

### New Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| UI_COPY dictionary | `demo/app.py:83` | Canonical self-explanation strings |
| /ui_copy endpoint | `demo/app.py` | Returns canonical copy for regression |
| Copy drift test | `tests/governance/test_ui_copy_drift.py` | Prevents semantic drift |
| UI regression | `tools/run_demo_cases.py --ui-tests` | Runtime verification |
| Docs sidebar | Demo UI | Links to 5 key documentation files |
| Abstention preservation | `governance/uvil.py` | Tier A gate on R_t |

### UI Integration Points (9 total)

1. Framing box with expandable "What does justified mean?"
2. Trust class tooltips and note
3. Transition note on commit
4. Outcome-specific explanations (ABSTAINED, VERIFIED, REFUTED)
5. Evidence pack expandable explanation
6. Hash label tooltips (U_t, R_t, H_t)
7. Boundary demo expandable breakdown
8. ADV exclusion badge tooltips
9. Governance error templates

### Invariants Added

13. **T13**: UI_COPY dictionary contains all required self-explanation keys
14. **T14**: No capability claims ("safe", "aligned", "intelligent") in UI copy
15. **T15**: ABSTAINED explanation mentions "first-class outcome"
16. **T16**: Abstention preservation gate enforced on all R_t entries

### How to Verify

```bash
# Run copy drift tests (offline, no server needed)
uv run pytest tests/governance/test_ui_copy_drift.py -v

# Run UI self-explanation tests (requires server)
uv run python demo/app.py  # In one terminal
uv run python tools/run_demo_cases.py --ui-tests  # In another

# Verify abstention preservation
uv run pytest tests/governance/test_abstention_preservation.py -v
```

### Breaking Changes

None. v0.2.0 is fully backward compatible with v0.1.0 demo cases.
