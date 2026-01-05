# MathLedger v0: System Boundary Memo

**Version**: 0.1
**Date**: 2026-01-02
**Audience**: Safety leads, governance reviewers, senior engineers, acquisition committees

---

## Purpose of v0

MathLedger v0 is a demonstration system that implements a subset of the Formal Model (FM) governance framework. Its purpose is to provide a working reference implementation of:

1. User-Verified Input Loop (UVIL) for claim partitioning
2. Dual-root attestation (U_t, R_t, H_t)
3. Trust-class assignment (FV, MV, PA, ADV)
4. Evidence pack generation with cryptographic hashes
5. Replay verification using identical code paths

v0 is not a production system. It is a demonstration of governance structure.

---

## What Is Mechanically Enforced (Tier A)

The following invariants are structurally enforced. Violation is impossible without cryptographic or structural failure detection.

| Invariant | Enforcement Mechanism |
|-----------|----------------------|
| Canonicalization Determinism | Golden hash tests fail on any drift |
| H_t = SHA256(R_t \|\| U_t) | Single code path in `compute_composite_root()` |
| ADV Excluded from R_t | `build_reasoning_artifact_payload()` raises ValueError |
| Content-Derived IDs | `derive_committed_id()` computes SHA256 of canonical JSON |
| Replay Uses Same Code Paths | Replay imports same attestation functions |
| No Silent Authority | `require_epoch_root()` gate before evidence pack output |
| Double-Commit Returns 409 | `_committed_proposal_ids` set check |
| Trust-Class Monotonicity | `require_trust_class_monotonicity()` gate at commit |

Each Tier A invariant has fail-closed semantics. On violation, the operation is rejected with a structured error response.

---

## What Is Detectable but Not Prevented (Tier B)

The following invariants are logged and replay-visible but not hard-gated at runtime.

| Invariant | Detection Method | Violation Path |
|-----------|------------------|----------------|
| Abstention First-Class | Logged in `validation_outcome` field | Downstream metrics could filter ABSTAINED |
| MV Validator Correctness | Logged validation outcome | Edge cases in arithmetic parsing |

Tier B invariants are visible in evidence packs and replay logs. They are not prevented at runtime in v0.

---

## What Is Explicitly Out of Scope (Tier C)

The following are documented in the FM but not implemented in v0.

| Invariant | Current State |
|-----------|---------------|
| FV Mechanical Verification | No Lean/Z3 verifier. All FV claims return ABSTAINED. |
| Multi-Model Consensus | Single template partitioner. No multi-model voting. |
| RFL Integration | No learning loop. No curriculum updates. |

Tier C invariants are aspirational. They are not claimed to work.

---

## Authority Model

v0 implements a four-class trust hierarchy:

| Class | Meaning | v0 Behavior |
|-------|---------|-------------|
| FV (Formally Verified) | Claim has machine-checkable proof (Lean/Z3, Phase II) | Returns ABSTAINED (no verifier in v0) |
| MV (Mechanically Validated) | Claim is testable by deterministic validator (limited coverage) | Returns VERIFIED/REFUTED/ABSTAINED |
| PA (Procedurally Attested) | Claim attested by human authority | Returns ABSTAINED (authority noted) |
| ADV (Advisory) | Claim is commentary, not truth-bearing | Excluded from R_t entirely |

Authority flows through the dual-root attestation:
- **U_t**: Merkle root of user interaction events
- **R_t**: Merkle root of reasoning artifacts (excludes ADV)
- **H_t**: SHA256(R_t || U_t) — composite epoch root

Only claims with `authority_gate_passed: True` are included in canonical evidence packs.

---

## Abstention Semantics

ABSTAINED is a typed outcome, not a failure. It means:

- The claim was submitted for verification
- The system had no verifier capable of producing VERIFIED or REFUTED
- The claim is recorded with its trust class and rationale
- The claim is included in attestation (for MV, PA) but not asserted as true or false

ABSTAINED claims:
- Are included in R_t (for MV, PA)
- Are excluded from R_t (for ADV)
- Are visible in evidence packs
- Are replayable with identical outcome

The system does not default to VERIFIED or REFUTED when uncertain.

---

## Replayability & Non-Silent Drift

Every evidence pack contains:

1. `uvil_events`: All user interaction events
2. `reasoning_artifacts`: All claims with trust class and outcome
3. `u_t`, `r_t`, `h_t`: Attestation roots
4. `replay_instructions`: Commands to reproduce attestation

Replay verification uses the same code paths as original computation. If replay produces different hashes, the evidence pack is invalid.

The No Silent Authority gate ensures that no authority-bearing output is produced without passing attestation verification. Any drift between claimed and computed H_t raises `SilentAuthorityViolation`.

---

## Why This Is Safe to Evaluate

v0 is safe to evaluate because:

1. **Scope is explicit**: Tier A/B/C classification makes enforcement boundaries clear
2. **Fail-closed semantics**: Tier A violations reject operations, not produce bad output
3. **No hidden authority**: The No Silent Authority gate is mandatory for evidence packs
4. **Abstention is honest**: The system does not claim verification it cannot perform
5. **Replay is deterministic**: Anyone can verify attestation roots from evidence packs
6. **Trust classes are immutable**: Once committed, claims cannot change trust class
7. **ADV is excluded**: Advisory content cannot contaminate reasoning attestation

The system does not:
- Claim to verify formal proofs (FV returns ABSTAINED; Lean/Z3 is Phase II)
- Claim to handle all mathematical claims (MV covers only `a op b = c` for integers)
- Claim multi-model consensus (single partitioner)
- Claim learning integration (no curriculum updates)

**Terminology note**: "Machine-checkable proof" applies only to FV (Formal Verification via Lean/Z3), which is not implemented in v0. MV (Mechanical Validation) uses a procedural arithmetic checker with limited coverage—this is not formal proof, merely deterministic evaluation.

---

**Closing Statement**

MathLedger v0 does not decide what is true; it decides what is justified, and it makes that decision replayable.

---

**Author**: Claude A (v0.1 System Boundary Memo)
**Audit Date**: 2026-01-02
