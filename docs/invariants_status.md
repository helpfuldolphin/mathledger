# Invariants Status: FM vs. v0 Implementation

This document provides a brutally honest classification of governance invariants.

**Tier Classification:**
- **Tier A**: Cryptographically or structurally enforced. Violation is impossible without detection.
- **Tier B**: Logged and replay-visible. Violation is detectable but not prevented.
- **Tier C**: Documented but not enforced in v0. Aspirational.

---

## Invariants Table

| Invariant | FM Section | Tier | How It Can Be Violated Today | Current Detection | Planned Enforcement |
|-----------|------------|------|------------------------------|-------------------|---------------------|
| **Canonicalization Determinism** | §1.5 | A | Impossible without golden test failure | `test_golden_evidence_pack.py` | Done |
| **H_t = SHA256(R_t \|\| U_t)** | §1.5 | A | Impossible - structural code constraint | `compute_composite_root()` | Done |
| **ADV Excluded from R_t** | §1.5 | A | Impossible - `build_reasoning_artifact_payload` raises ValueError | Structural rejection | Done |
| **Content-Derived IDs** | §1.5 | A | Impossible - `derive_committed_id()` uses SHA256 | Code structure | Done |
| **Replay Uses Same Code Paths** | — | A | Impossible - same imports in `replay_verify` | Structural | Done |
| **Double-Commit Returns 409** | — | A | Impossible - `_committed_proposal_ids` set check | API enforcement | Done |
| **No Silent Authority** | §4 | **A** | Impossible - `require_epoch_root()` gate mandatory for evidence pack | `authority_gate.require_epoch_root()` | Done (v0.1) |
| **Trust-Class Monotonicity** | §6 | **A** | Impossible - `require_trust_class_monotonicity()` gate mandatory at commit | `trust_monotonicity.require_trust_class_monotonicity()` | Done (v0.1) |
| **Abstention Preservation** | §4.1 | **A** | Impossible - `require_abstention_preservation()` gate mandatory | `abstention_preservation.require_abstention_preservation()` | Done (v0.1) |
| **Audit Surface Version Field** | — | **A** | Impossible - manifest.json includes source reference | Build assertion verifies | Done (v0.2.0) |
| **MV Validator Correctness** | — | B | Edge cases in arithmetic parsing; non-arithmetic claims | Logged validation_outcome | Additional validators |
| **FV Mechanical Verification** | §1.5 | C | No Lean/Z3 verifier exists | — | Phase II |
| **Multi-Model Consensus** | §10 | C | Single template partitioner | — | Phase II |
| **RFL Integration** | §7-8 | C | No learning loop in v0 | — | Phase II |
| **USLA** | §10 | C | No constitutional constraint layer | — | Phase III |
| **TDA Mind Scanner** | §10.3 | C | No topological coherence monitoring | — | Phase III |

---

## Tier A: Enforced (10 invariants)

These cannot be violated without cryptographic or structural failure detection.

### 1. Canonicalization Determinism
- **FM Reference**: §1.5 ("Canonical identity is sacred")
- **Enforcement**: Golden hash test (`test_golden_evidence_pack.py`)
- **Detection**: CI fails if `GOLDEN_U_T`, `GOLDEN_R_T`, `GOLDEN_H_T` drift
- **Status**: ✓ Complete

### 2. H_t Computation
- **FM Reference**: §1.5 ("$H_t = \text{Hash}(\texttt{EPOCH:} \| R_t \| U_t)$")
- **Enforcement**: `compute_composite_root(r_t, u_t)` in `attestation/dual_root.py`
- **Detection**: Structural - only one code path exists
- **Status**: ✓ Complete

### 3. ADV Excluded from R_t
- **FM Reference**: §1.5, §4 ("forbidden to promote into authority-bearing state")
- **Enforcement**: `build_reasoning_artifact_payload()` raises `ValueError` for ADV
- **Detection**: Structural rejection + replay verification
- **Status**: ✓ Complete

### 4. Content-Derived IDs
- **FM Reference**: §1.5 ("canonical normalization and hashing")
- **Enforcement**: `derive_committed_id()` computes SHA256 of canonical JSON
- **Detection**: Any ID change means content changed
- **Status**: ✓ Complete

### 5. Replay Uses Same Code Paths
- **Enforcement**: `replay_verify` imports `compute_ui_root`, `compute_reasoning_root`, `compute_composite_root`
- **Detection**: No duplicated canonicalization logic
- **Status**: ✓ Complete

### 6. No Silent Authority (PROMOTED v0.1)
- **FM Reference**: §4 ("Nothing that influences durable learning authority may occur silently")
- **Enforcement**: `governance/authority_gate.py:require_epoch_root()`
- **Detection**: `SilentAuthorityViolation` exception on any gate failure
- **Gate Location**: `backend/api/uvil.py:get_evidence_pack()` - mandatory before producing canonical evidence pack
- **Fail Mode**: Fail-closed (HTTP 422, no evidence pack produced)
- **Tests**: `tests/governance/test_no_silent_authority.py` (17 tests)
- **Status**: ✓ Complete (v0.1)

### 7. Double-Commit Returns 409
- **Enforcement**: `_committed_proposal_ids` set check in `commit_uvil()`
- **Detection**: HTTP 409 response on duplicate commit attempt
- **Status**: ✓ Complete

### 8. Trust-Class Monotonicity (PROMOTED v0.1)
- **FM Reference**: §6 ("Once a claim is committed, its trust class is immutable")
- **Enforcement**: `governance/trust_monotonicity.py:require_trust_class_monotonicity()`
- **Detection**: `TrustClassMonotonicityViolation` exception on any trust-class change attempt
- **Gate Location**: `backend/api/uvil.py:commit_uvil()` - mandatory before committing claims
- **Fail Mode**: Fail-closed (HTTP 422, commit rejected)
- **Tests**: `tests/governance/test_trust_class_monotonicity.py` (22 tests)
- **Status**: ✓ Complete (v0.1)

### 9. Abstention Preservation (PROMOTED v0.1)
- **FM Reference**: §4.1 ("abstention is a typed outcome... first-class ledger artifact")
- **Enforcement**: `governance/abstention_preservation.py:require_abstention_preservation()`
- **Detection**: `AbstentionPreservationViolation` exception on missing/null/invalid outcome
- **Gate Location**: Before R_t computation - mandatory for all reasoning artifacts
- **Fail Mode**: Fail-closed (no evidence pack without valid outcomes)
- **Tests**: `tests/governance/test_abstention_preservation.py` (25+ tests)
- **Status**: ✓ Complete (v0.1)

### 10. Audit Surface Version Field (PROMOTED v0.2.0)
- **FM Reference**: — (operational invariant, not in FM)
- **Enforcement**: `releases/releases.json` is the single source of truth for version metadata
- **Detection**: Build assertion verifies deployed artifacts match releases.json entries
- **Gate Location**: `tools/predeploy_gate.py` - mandatory before deployment
- **Fail Mode**: Fail-closed (deployment blocked if mismatch detected)
- **Tests**: `tests/governance/test_release_metadata_guard.py`
- **Status**: ✓ Complete (v0.2.0)

**What Is Enforced**:

1. Every release has a canonical entry in `releases/releases.json` with:
   - `version`: Semantic version string (e.g., "v0.2.3")
   - `tag`: Git tag name (e.g., "v0.2.3-audit-path-freshness")
   - `commit`: Short commit hash for verification
   - `status`: Release lifecycle state ("active", "closed")

2. Evidence pack artifacts include version reference:
   - `releases/evidence_pack_examples.{version}.json` → `pack_version` field matches version
   - `releases/evidence_pack_verifier_vectors.{version}.json` → `metadata.version` matches version

3. Hosted demo health endpoint (`/health`) returns:
   - `build_commit`, `build_tag` matching releases.json current_version entry
   - `release_pin.is_stale` flag for drift detection

**How Auditors Verify**:

```bash
# 1. Check releases.json current_version
cat releases/releases.json | jq '.current_version'

# 2. Verify deployed demo matches
curl -s https://mathledger.ai/demo/health | jq '.build_tag, .build_commit'

# 3. Run predeploy gate (should pass for correctly deployed system)
uv run python tools/predeploy_gate.py --check-health
```

**Promotion Date**: 2026-01-02 (v0.2.0)

---

## Tier B: Logged but Not Hard-Gated (1 invariant)

These are detectable via replay or logs but not prevented at runtime.

### 1. MV Validator Correctness
- **Current State**: Arithmetic validator handles `a op b = c` pattern
- **Violation Path**: Edge cases (overflow, division by zero, floating point)
- **Detection**: Logged validation_outcome with parsed values
- **Planned Enforcement**: Additional validators, edge case handling

---

## Tier C: Aspirational (3 invariants in current release, 5 total documented)

These are documented but not implemented in v0. Only FV, Multi-Model, and RFL are tracked in releases.json.

### 1. FV Mechanical Verification
- **FM Reference**: §1.5, throughout
- **Current State**: FV trust class exists; no Lean/Z3 verifier
- **Status**: All FV claims return ABSTAINED

### 2. Multi-Model Consensus
- **FM Reference**: §10 (uncharted surface area)
- **Current State**: Single template partitioner
- **Status**: Not in v0 scope

### 3. RFL Integration
- **FM Reference**: §7-8
- **Current State**: No learning loop
- **Status**: Not in v0 scope

### 4. USLA (Unified System Law Architecture)
- **FM Reference**: §10, throughout
- **Current State**: No constitutional constraint layer
- **Status**: Phase III scope

### 5. TDA Mind Scanner
- **FM Reference**: §10.3, throughout
- **Current State**: No topological coherence monitoring
- **Status**: Phase III scope

---

## Promotion Complete: No Silent Authority → Tier A (v0.1)

**Invariant**: No Silent Authority (FM §4)

**Previous State**: Tier B (logged, not hard-gated)

**Current State**: Tier A (structurally enforced)

**Implementation (v0.1)**:

```python
# governance/authority_gate.py

def require_epoch_root(request: AuthorityUpdateRequest) -> str:
    """
    Verify epoch root before allowing authority-bearing output.

    FAIL-CLOSED: Any validation failure raises SilentAuthorityViolation.
    """
    if request.uvil_events is None:
        raise SilentAuthorityViolation("uvil_events is None")
    if request.reasoning_artifacts is None:
        raise SilentAuthorityViolation("reasoning_artifacts is None")

    computed_u_t = compute_ui_root(request.uvil_events)
    computed_r_t = compute_reasoning_root(request.reasoning_artifacts)
    computed_h_t = compute_composite_root(computed_r_t, computed_u_t)

    if request.claimed_h_t and computed_h_t != request.claimed_h_t:
        raise SilentAuthorityViolation("H_t mismatch - tampering detected")

    return computed_h_t
```

**Enforcement Location**:

- `backend/api/uvil.py:get_evidence_pack()` - mandatory gate before producing canonical evidence pack
- `governance/authority_gate.py:create_gated_evidence_pack_data()` - helper that enforces gate

**What Is Gated (v0.1 "Durable Influence")**:

1. Producing evidence packs (canonical attestation output)
2. Any data marked `authority_gate_passed: True`

**Tests**:

- `tests/governance/test_no_silent_authority.py` (17 tests)
- Cases: missing inputs, tampered artifacts, tampered events, happy path, bypass prevention

**Promotion Date**: 2026-01-02

---

## Promotion Complete: Trust-Class Monotonicity → Tier A (v0.1)

**Invariant**: Trust-Class Monotonicity (FM §6)

**Previous State**: Tier B (logged, humans could over-assign)

**Current State**: Tier A (structurally enforced)

**Implementation (v0.1)**:

```python
# governance/trust_monotonicity.py

def require_trust_class_monotonicity(
    claims: List[Dict[str, Any]],
    committed_partition_id: str,
    record_violation: bool = True,
) -> None:
    """
    Verify trust-class monotonicity for a batch of claims.

    FAIL-CLOSED: First violation raises TrustClassMonotonicityViolation.
    """
    for claim in claims:
        verify_trust_class_immutable(
            claim_id=claim["claim_id"],
            claim_text=claim["claim_text"],
            trust_class=claim["trust_class"],
            rationale=claim.get("rationale", ""),
            committed_partition_id=committed_partition_id,
            record_violation=record_violation,
        )
```

**Enforcement Location**:

- `backend/api/uvil.py:commit_uvil()` - mandatory gate before committing claims
- `governance/trust_monotonicity.py:verify_trust_class_immutable()` - single claim verification

**What Is Gated (v0.1)**:

1. Any attempt to commit a claim_id with a different trust class than originally committed
2. Any content tampering (claim_text mismatch for same claim_id)

**Error Response** (HTTP 422):
```json
{
  "error_code": "TRUST_CLASS_MONOTONICITY_VIOLATION",
  "message": "Trust-class monotonicity FAILED: Cannot change trust class...",
  "committed_partition_id": "...",
  "claim_id": "sha256:...",
  "attempted_change": {"from": "MV", "to": "FV"}
}
```

**Tests**:

- `tests/governance/test_trust_class_monotonicity.py` (22 tests)
- Cases: upgrades blocked, downgrades blocked, valid new artifacts, structured error, audit log, no contamination

**Promotion Date**: 2026-01-02

---

## Promotion Complete: Abstention Preservation → Tier A (v0.1)

**Invariant**: Abstention Preservation (FM §4.1)

**Previous State**: Tier B (logged, downstream could ignore)

**Current State**: Tier A (structurally enforced)

**Implementation (v0.1)**:

```python
# governance/abstention_preservation.py

def require_abstention_preservation(
    reasoning_artifacts: List[Dict[str, Any]],
    record_violation: bool = True,
) -> None:
    """
    Verify abstention preservation for a batch of reasoning artifacts.

    FAIL-CLOSED: First violation raises AbstentionPreservationViolation.
    """
    for idx, artifact in enumerate(reasoning_artifacts):
        verify_outcome_present(artifact, artifact_index=idx, record_violation=record_violation)
```

**Enforcement Location**:

- Before R_t computation - mandatory for all reasoning artifacts
- `governance/abstention_preservation.py:verify_outcome_present()` - single artifact verification

**What Is Gated (v0.1)**:

1. Missing `validation_outcome` field → violation
2. Null/None `validation_outcome` → violation
3. Invalid outcome value (not VERIFIED/REFUTED/ABSTAINED) → violation
4. Aggregation without explicit ABSTAINED handling → violation

**Error Response** (HTTP 422):
```json
{
  "error_code": "ABSTENTION_PRESERVATION_VIOLATION",
  "message": "Abstention preservation FAILED: validation_outcome is null...",
  "artifact_index": 2,
  "claim_id": "sha256:...",
  "violation_type": "NULL_VALUE",
  "details": {"received_value": null}
}
```

**Tests**:

- `tests/governance/test_abstention_preservation.py` (25+ tests)
- Cases: missing outcome, null outcome, invalid outcome, valid outcomes, aggregation, coercion detection

**Promotion Date**: 2026-01-02

---

## Tracked Issues: FM vs. Repo Discrepancies

| Issue | FM Claim | Repo State | Resolution |
|-------|----------|------------|------------|
| H_t formula | `Hash("EPOCH:" || R_t || U_t)` | `SHA256(R_t || U_t)` | Domain prefix differs; documented as v0 simplification |
| Trust classes | 4 defined (FV, MV, PA, ADV) | 4 implemented | ✓ Aligned |
| Dual attestation | U_t + R_t + H_t | Implemented | ✓ Aligned |
| Abstention outcome | First-class artifact | Implemented | ✓ Aligned |

For a comprehensive 20-point pressure analysis, see: [SPEC_PRESSURE_AUDIT_V0.2.x.md](SPEC_PRESSURE_AUDIT_V0.2.x.md)

---

## Governance Note

This document increases credibility by being honest about what is enforced vs. aspirational.

- **Tier A**: Show these. They are cryptographic proof.
- **Tier B**: Acknowledge these. They are logged but not gated.
- **Tier C**: Do not claim these work. They are Phase II.

**Date**: 2026-01-02
**Author**: Claude A (v0 Evidence Pack Closure Pass)
