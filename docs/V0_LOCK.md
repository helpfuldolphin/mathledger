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
| ABSTAINED | System cannot mechanically verify (always true in v0) |
| VERIFIED | Reserved for when real verifier exists (not used in v0) |
| REFUTED | Reserved for when real verifier exists (not used in v0) |

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
**Commit**: `2bc6bfb6496ab0cee2a8877d0b4b3e2edaedd16b`
**Date**: 2026-01-02

### Key Deliverables

| Artifact | Path | Purpose |
|----------|------|---------|
| Demo entrypoint | `demo/app.py` | Single-command runnable: `uv run python demo/app.py` |
| API endpoints | `backend/api/uvil.py` | `/propose_partition`, `/commit_uvil`, `/run_verification` |
| Governance models | `governance/uvil.py` | DraftProposal, CommittedPartitionSnapshot, attestation |
| Trust classes | `governance/trust_class.py` | FV, MV, PA, ADV enum + authority-bearing check |
| Regression harness | `tools/run_demo_cases.py` | 5 demo cases with fixture capture |
| Fixtures | `fixtures/<case>/` | mv_only, mixed_mv_adv, pa_only, adv_only, underdetermined |

### Invariants Verified

1. **T1**: proposal_id never appears in hash-committed payloads
2. **T2**: ADV claims excluded from R_t computation
3. **T3**: H_t = SHA256(R_t || U_t) deterministic
4. **T4**: Double-commit of proposal_id returns 409 Conflict
5. **T5**: /run_verification rejects raw proposal_id
6. **T6**: All v0 outcomes are ABSTAINED (no verifier exists)
7. **T7**: PA claims show "authority-bearing but not mechanically verified"
8. **T8**: UI clearly separates Exploration Stream from Authority Stream

### Critical Fix

- **PA terminology hazard**: PA claims no longer return `VERIFIED`. All claims in v0 return `ABSTAINED` with explicit `authority_basis` explanation.

### How to Verify

```bash
# Start demo
uv run python demo/app.py

# Run regression harness (in another terminal)
uv run python tools/run_demo_cases.py

# Expected: 5/5 cases succeed, all outcomes ABSTAINED
```
