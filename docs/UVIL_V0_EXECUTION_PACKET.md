# UVIL v0 + Trust Classes v0: Execution Packet

**Status:** GO FOR IMPLEMENTATION
**Subordinate to:** fm.tex (Field Manual)
**Scope:** Single-model governance demo, no multi-model arena logic

---

## 1. GO / NO-GO VERDICT

**GO**

- **Leaf contract locked**: `compute_ui_root()` and `compute_reasoning_root()` expect raw dict payloads; dual_root handles canonicalization internally - verified by reading `attestation/dual_root.py`.
- **Exploration/authority boundary defined**: DraftProposal (random IDs) never enters hash-committed paths; CommittedPartitionSnapshot (content-derived IDs) is the sole input to U_t.
- **No external blockers**: In-memory storage accepted for v0, no LiteLLM/E2B dependencies, single-model scope explicitly scoped.

---

## 2. PR STACK

| PR | Title | Files | Tests | Risk | Commit Message |
|----|-------|-------|-------|------|----------------|
| **PR1** | `feat(governance): add TrustClass enum` | `governance/trust_class.py` | `tests/governance/test_trust_class.py` | Low | `feat(governance): add TrustClass enum (FV/MV/PA/ADV)` |
| **PR2** | `feat(governance): add UVIL data models` | `governance/uvil.py` | `tests/governance/test_uvil_models.py` | Medium | `feat(governance): add DraftProposal, CommittedClaim, UVIL_Event, ReasoningArtifact` |
| **PR3** | `feat(governance): add attestation builders` | `governance/uvil.py` (extend) | `tests/governance/test_uvil_attestation.py` | High | `feat(governance): add build_uvil_event_payload, compute_full_attestation` |
| **PR4** | `feat(api): add /propose_partition endpoint` | `backend/api/uvil.py` | `tests/api/test_uvil_propose.py` | Medium | `feat(api): add POST /propose_partition (exploration only)` |
| **PR5** | `feat(api): add /commit_uvil endpoint` | `backend/api/uvil.py` (extend) | `tests/api/test_uvil_commit.py` | High | `feat(api): add POST /commit_uvil with proposal_id validation` |
| **PR6** | `feat(api): add /run_verification endpoint` | `backend/api/uvil.py` (extend) | `tests/api/test_uvil_verify.py` | High | `feat(api): add POST /run_verification (rejects proposal_id)` |
| **PR7** | `test(governance): add T1-T3 invariant tests` | `tests/governance/test_uvil_v0_invariants.py` | - | Medium | `test(governance): add ADV-leakage, immutability tests (T1-T3)` |
| **PR8** | `test(governance): add T4-T8 invariant tests` | `tests/governance/test_uvil_v0_invariants.py` (extend) | - | High | `test(governance): add determinism, exclusion tests (T4-T8)` |
| **PR9** | `feat(demo): add minimal Streamlit app` | `demo/app.py`, `demo/config.py` | `tests/demo/test_app_smoke.py` | Low | `feat(demo): add Streamlit UVIL demo with baseline/governed modes` |
| **PR10** | `docs: add UVIL_V0_SPEC.md` | `docs/UVIL_V0_SPEC.md` | - | Low | `docs: add UVIL v0 + Trust Classes v0 specification` |

**Total: 10 PRs** (under 12 limit)

---

## 3. INVARIANT CHECKLIST

For `.github/pull_request_template.md`:

```markdown
## UVIL v0 Invariant Checklist

Before merging, confirm ALL of these:

- [ ] **INV-1**: ADV claims NEVER appear in R_t computation
- [ ] **INV-2**: CommittedClaim is frozen (`@dataclass(frozen=True)`)
- [ ] **INV-3**: UVIL_Event is frozen (`@dataclass(frozen=True)`)
- [ ] **INV-4**: ReasoningArtifact is frozen (`@dataclass(frozen=True)`)
- [ ] **INV-5**: `compute_ui_root()` receives raw dict payloads, NOT pre-hashed strings
- [ ] **INV-6**: `compute_reasoning_root()` receives raw dict payloads, NOT pre-hashed strings
- [ ] **INV-7**: DraftProposal.proposal_id NEVER appears in any hash-committed payload
- [ ] **INV-8**: `/run_verification` rejects requests containing `proposal_id`
- [ ] **INV-9**: `/commit_uvil` requires `proposal_id` to exist in draft store
- [ ] **INV-10**: Committed IDs derived from `sha256(canonical_json(content))`, not random
- [ ] **INV-11**: Same seed + same input produces identical (U_t, R_t, H_t)
- [ ] **INV-12**: No wall-clock timestamps in hash-committed fields
- [ ] **INV-13**: Default suggested trust class is ADV (user must promote)
- [ ] **INV-14**: Outcome enum restricted to {VERIFIED, REFUTED, ABSTAINED}
- [ ] **INV-15**: Double-commit of same proposal_id returns 409 Conflict
```

---

## 4. CRITICAL LEAF CONTRACT

**Source:** `attestation/dual_root.py`

```
compute_ui_root(ui_events: Sequence[RawLeaf]) -> str
compute_reasoning_root(proof_events: Sequence[RawLeaf]) -> str

Where RawLeaf = Union[str, bytes, Mapping[str, Any], Sequence[Any], int, float, bool, None]
```

**CRITICAL**: These functions expect RAW PAYLOADS (dicts), NOT pre-hashed strings.

Internal processing:
1. `_canonicalize_leaf(value)` - handles dict via `rfc8785_canonicalize()`
2. `hash_ui_leaf(canonical_value)` / `hash_reasoning_leaf(canonical_value)` - applies domain separation
3. `crypto_merkle_root(leaf_hashes)` - builds Merkle tree

**Domain separation tags:**
- UI leaves: `b"\xA1ui-leaf"`
- Reasoning leaves: `b"\xA0reasoning-leaf"`

---

## 5. GOLDEN HASH GOVERNANCE PROCEDURE

```
GOLDEN HASH GOVERNANCE (v0)
===========================

1. WHEN TO RECORD
   - After PR10 merges (full stack complete)
   - Run:
     uv run python -c "
     from governance.uvil import compute_full_attestation, UVIL_Event, ReasoningArtifact
     from governance.trust_class import TrustClass
     evt = UVIL_Event('evt_golden', 'COMMIT', 'cps_golden', 'user_golden', 1)
     art = ReasoningArtifact('art_golden', 'claim_golden', TrustClass.FV, {'golden': True})
     u, r, h = compute_full_attestation([evt], [art])
     print(f'GOLDEN: u_t={u}, r_t={r}, h_t={h}')
     "

2. RECORD LOCATION
   - File: tests/governance/golden_attestation.json
   - Contents: {"u_t": "...", "r_t": "...", "h_t": "...", "seed": "synthetic_v0"}

3. CI ENFORCEMENT
   - Add test: test_golden_attestation_no_drift()
   - Fails if computed != recorded
   - Any drift = BLOCKING, requires explicit approval

4. DRIFT HANDLING
   - If intentional change: update golden_attestation.json + add changelog entry
   - If unintentional: REVERT, investigate canonicalization/hashing change

5. NEVER
   - Never update golden without PR review
   - Never bypass CI check
   - Never use wall-clock or random values in golden fixtures
```

---

## 6. FILE TREE

```
governance/
  trust_class.py          # TrustClass enum (FV/MV/PA/ADV), Outcome enum
  uvil.py                 # Data models + attestation builders

backend/api/
  __init__.py             # Router exports
  uvil.py                 # /propose_partition, /commit_uvil, /run_verification

tests/governance/
  __init__.py
  test_uvil_v0_invariants.py  # T1-T8 invariant tests
  golden_attestation.json     # (after PR10)

docs/
  UVIL_V0_EXECUTION_PACKET.md  # This file
  UVIL_V0_SPEC.md              # (PR10)
```

---

## 7. HARD CONSTRAINTS

- **DO NOT** amend fm.tex/fm.pdf
- **DO NOT** introduce multi-model arena logic
- **DO NOT** introduce LiteLLM/E2B in v0
- **DraftProposal/proposal_id** must NEVER enter any hash-committed payload
- **compute_ui_root** and **compute_reasoning_root** must be called with raw dict payloads only
- **Storage**: In-memory dict, accept restart-loss for v0

---

## 8. TRUST CLASS DEFINITIONS

Per FM sec:trust-classes:

| Class | Name | Authority | R_t Entry |
|-------|------|-----------|-----------|
| FV | Formally Verified | Yes | Allowed |
| MV | Mechanically Validated | Yes | Allowed |
| PA | Procedurally Attested | Yes | Allowed |
| ADV | Advisory | No | NEVER |

**Default suggested trust class**: ADV (user must explicitly promote)

---

## 9. EXPLORATION vs AUTHORITY BOUNDARY

```
EXPLORATION PHASE (no hash commitment)
======================================
DraftProposal {
  proposal_id: str      # Random UUID - NEVER enters H_t
  claims: [DraftClaim]  # Mutable suggestions
  created_at: datetime  # Wall-clock, not committed
}

AUTHORITY PHASE (hash committed, frozen)
========================================
CommittedPartitionSnapshot {
  committed_partition_id: str  # sha256(canonical(claims))
  claims: Tuple[CommittedClaim]  # Frozen
  commit_epoch: int  # Monotonic counter
}
```

---

## 10. ATTESTATION FORMULA

```
U_t = compute_ui_root([build_uvil_event_payload(e) for e in uvil_events])
R_t = compute_reasoning_root([build_reasoning_artifact_payload(a) for a in reasoning_artifacts])
H_t = compute_composite_root(R_t, U_t)  # SHA256(R_t || U_t)
```

Note: FM specifies `Hash("EPOCH:" || R_t || U_t)` but repo uses `SHA256(R_t || U_t)`.
This is a tracked discrepancy - do not modify FM.

---

## END OF EXECUTION PACKET
