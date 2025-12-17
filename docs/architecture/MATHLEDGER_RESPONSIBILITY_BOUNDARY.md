# MathLedger Responsibility Boundary

**STATUS**: DESCRIPTIVE / AUDIT-GRADE BOUNDARY STATEMENT (2025-12-13)

**Derivation**: This boundary is extracted strictly from verified claims in:
- `TEMPORAL_CAUSAL_SEMANTICS_EXTRACTION.md` (24 VERIFIED, 1 PARTIAL)
- `HARMONIC_100X_REBASED_ON_TEMPORAL_CONTRACT.md` (failure mapping)

No new claims, mechanisms, or capabilities are introduced.

---

## 1. Scope Declaration

MathLedger is a **provenance and ordering infrastructure** for mathematical proof artifacts. It operates at the **ledger and attestation layer**, recording what was generated, when it was sealed, and in what sequence. It enforces monotonicity constraints on block ordering, requires dual attestation before policy updates, and isolates observational governance from enforcement actions.

MathLedger does not attempt to be a verifier, a proof generator, a formalization engine, or a training system. It does not evaluate the mathematical correctness of proofs, the fidelity of formalizations, or the quality of model outputs. It records and orders artifacts that external systems produce.

---

## 2. Responsibility Matrix

| Category | Responsibility Type | Explanation |
|----------|---------------------|-------------|
| **Enforced Guarantees** | MathLedger is responsible for enforcing these conditions via code guards with test coverage. | Block height monotonicity (BARR-02 via FORB-04: `check_monotone_ledger()` rejects `curr_height <= prev_height`). Hash chain integrity (BARR-02 via FORB-05: `prev_hash` must match prior block). Timestamp non-regression (BARR-02 via FORB-06: `curr_ts >= prev_ts`). Dual attestation prerequisite (BARR-01: `compute_composite_root()` raises `ValueError` if R_t or U_t missing). RFL gate fail-closed (BARR-03/04 via FORB-07: `RFLEventGate` rejects invalid attestations). SHADOW mode non-interference (FORB-02/08: all observations `action="LOGGED_ONLY"`; P3/P4 runners reject `shadow_mode=False`). AI proof isolation (FORB-01: `ShadowModeFilter.should_include()` returns False for `source_type="external_ai"`). |
| **Observed & Detectable Conditions** | MathLedger can observe these conditions and surface them in signals or artifacts, but does not prevent them. | Non-deterministic proof generation: detected via `determinism_rate` signal at Attestation clock; immutable H_t enables replay comparison. Proof log truncation: detected via `entry_count` mismatch in proof snapshot at Block clock; post-hoc only. Distribution shift: detected via `tier_skew_detector` alarm at TDA window clock; no prevention barrier. Manifest tampering: detected via Merkle root verification in evidence pack; BARR-01 commits H_t. Semantic drift: detected via `semantic_drift` tile at corpus level; not formalization-level. DAG orphans: detected via `ProofDag.validate()` at Block clock; FORB-05 surfaces as FK constraint failure. |
| **Recorded but Not Evaluated Conditions** | MathLedger records these in the attestation chain but does not evaluate their quality or correctness. | Proof content: R_t commits Merkle root over proofs; BARR-01 is content-agnostic. Formalization specs: H_t commits whatever Lean spec was provided; no semantic gate. Statement count: `statement_count` in block metadata; no triviality filter. Environment hash: `env_hash` in manifest captures pip freeze; no Lean toolchain version. Code provenance: `provenance.git.commit_sha` recorded; no model checkpoint. |
| **Out-of-Scope Conditions** | MathLedger does not govern, track, or evaluate these. They are structurally outside the temporal/causal contract. | Verifier soundness: Lean kernel correctness assumed; no cross-verifier validation. Mathlib soundness: assumed sound; not tracked in any clock domain. Model weights: not part of Attestation clock (H_t has no model hash). Training data: no clock domain tracks training corpus; outside governance scope. Formalization fidelity: natural language → Lean spec binding not audited. Abstention: not tracked in any governance signal. Benchmark validity: external to system; no holdout set governance. Proof quality/novelty: binary type-check only; no pedagogical or diversity metric. Infrastructure availability: telemetry gaps detectable; no HA governance. Namespace collisions: Lean-internal; below governance layer. |

---

## 3. Detection vs Prevention Boundary

### Detectable but Not Preventable

| Failure Class | Why Detectable | Why Not Preventable | Clock/Barrier Surface |
|---------------|----------------|---------------------|----------------------|
| **Non-deterministic proof generation** | BARR-01 creates immutable H_t; replay comparison reveals divergence; `determinism_rate` signal drops | No barrier gates on model sampling; proof generation occurs before attestation | Attestation clock; `determinism_rate` signal |
| **Proof log truncation** | `entry_count` in proof snapshot enables post-hoc comparison | Truncation occurs before Block clock (`seal_block_with_dual_roots()` never invoked) | Block clock; `entry_count` mismatch |
| **Distribution shift** | `tier_skew_detector` raises alarm at TDA window boundary | No barrier prevents shift; TDA is observational only (FORB-03) | TDA window clock; drift radar alarm |
| **Semantic drift** | `semantic_drift` tile detects corpus-level divergence | BARR-02 enforces sequence, not semantic coherence; H_t is content-agnostic | Block clock; `semantic_drift` signal |
| **Environment version skew** | `env_hash` in manifest enables cross-run comparison | BARR-02 checks block sequence, not verifier version; toolchain not in attestation | Attestation clock; `env_hash` mismatch |
| **Curriculum trajectory degradation** | `curriculum_drift_tile` tracks policy trajectory | BARR-04 requires valid H_t, but H_t doesn't encode proof quality | RFL epoch clock; drift tile |

### Cascade Example: "Ghost Theorem Epidemic" (Cascade 2)

This cascade illustrates the detection/prevention boundary:

- **Trigger**: Proof log truncation (proofs lost before sealing)
- **Why barriers don't prevent it**: BARR-02 (monotone guard) validates blocks that exist; truncation occurs *before* `seal_block_with_dual_roots()` is called. FORB-05 (hash chain) validates *present* blocks, not missing entries.
- **Detection locus**: Block clock, post-hoc. `entry_count` mismatch in proof snapshot. `ProofDag.validate()` detects orphans if truncated proofs are referenced as parents.
- **Barrier limitation**: No write-ahead logging barrier exists. Truncation is not a temporal ordering violation; it is an event that never enters the clock domain.

### Cascade Example: "TTT Weight Rot" (Cascade 4)

- **Trigger**: Model weights change without checkpointing
- **Why barriers don't prevent it**: Model checkpoint is not part of Attestation clock. H_t = SHA256(R_t || U_t) commits proof content, not proof provenance. CLK-06 (RFL epoch) tracks policy, not model weights.
- **Detection locus**: Attestation clock, symptom-level. `determinism_rate` drops when replay produces different proof for same input.
- **Barrier limitation**: BARR-01 creates immutable record of *what* was generated, but not *which model* generated it. Detection is via consequence (non-reproducibility), not cause (weight drift).

---

## 4. What MathLedger Does Not Claim

MathLedger does not claim responsibility for:

- **Model quality**: MathLedger does not evaluate, measure, or attest to the capability, accuracy, or reliability of any proof-generating model. The attestation chain commits artifacts; it does not endorse them.

- **Proof novelty**: MathLedger does not assess whether a proof is novel, interesting, or mathematically significant. `statement_count` is tracked; triviality is not filtered.

- **Formalization fidelity**: MathLedger does not verify that a Lean specification faithfully captures the intended natural-language theorem. H_t commits the spec that was provided; no semantic audit exists.

- **Training data hygiene**: MathLedger does not track, audit, or attest to the provenance, contamination status, or holdout compliance of any training corpus. Training data is outside the governance scope.

- **Benchmark validity**: MathLedger does not validate that benchmark scores reflect genuine capability. External benchmark selection is not governed.

- **Verifier soundness**: MathLedger assumes the Lean kernel is correct. No cross-verifier quorum exists. If a kernel bug is discovered, MathLedger can identify tainted subtrees via `ProofDag` lineage, but it cannot prevent acceptance of invalid proofs.

---

## 5. Partner-Safe Boundary Statement

MathLedger provides **auditable provenance and ordering** for mathematical proof artifacts. It records what proofs were generated, seals them into monotonically ordered blocks with cryptographic attestation, and enforces non-interference between observational governance and enforcement actions. Partners integrating with MathLedger gain an immutable audit trail of what was committed and when, with replay capability for determinism verification. MathLedger does not evaluate the correctness, quality, or fidelity of proofs it records. It does not certify, endorse, or validate the systems that generate those proofs. The value proposition is provenance infrastructure: knowing what happened, in what order, with what attestation—not asserting that what happened was correct.

---

## 6. Final Boundary Sentence

**MathLedger is responsible for block ordering, hash chain integrity, dual attestation prerequisites, and SHADOW mode non-interference; records proof artifacts, formalization specs, and environment hashes without evaluating their correctness; detects non-determinism, truncation, drift, and tampering post-hoc via replay and signal comparison; and explicitly does not govern model weights, training data, formalization fidelity, verifier soundness, or benchmark validity.**

---

## Appendix: Evidence Traceability

| Responsibility Claim | Verified Claim ID | Test Evidence |
|---------------------|-------------------|---------------|
| Block height monotonicity | FORB-04 | `test_height_violation` |
| Hash chain integrity | FORB-05 | `test_hash_chain_violation` |
| Timestamp non-regression | FORB-06 | `test_timestamp_violation` |
| Dual attestation prerequisite | BARR-01 | `test_compute_composite_root_invalid_r_t` |
| RFL gate fail-closed | BARR-03, FORB-07 | `test_rfl_event_gate` |
| SHADOW non-interference | FORB-02, FORB-08 | `test_p3_harness_no_governance_api_calls`, `test_p3_harness_shadow_mode_enforced_in_config` |
| AI proof isolation | FORB-01 | `test_external_ai_blocked` |
| Determinism detection | BARR-01 | `test_compute_composite_root_valid` (H_t immutability enables replay) |
| DAG orphan detection | FORB-05 | `proof_parents` FK constraint; `ProofDag.validate()` |

---

*Document generated: 2025-12-13*
*Derivation status: All claims traceable to VERIFIED temporal/causal contract*
*Acceptance criteria: No future-looking language; no competitive language; detection ≠ prevention*
