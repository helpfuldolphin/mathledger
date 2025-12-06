# VCP 2.1 Execution Plan: The First Organism

**Role:** Global Orchestrator
**Target:** `mathledger` Repository
**Goal:** Transform Spanning Set → Minimal Basis
**Vibe:** [VIBE_SPEC_MATHLEDGER_VCP21.md](../docs/VIBE_SPEC_MATHLEDGER_VCP21.md)

---

## Layer 2: Multi-Agent Decomposition (Task Graph)

We decompose the refactor into 7 thematic tracks. Each track consists of atomic micro-tasks.

### Track A: Substrate & Normalization (The Axiomatic Core)
*Focus: `backend/logic`, `backend/fol_eq`, `backend/axiom_engine`*
- **A1:** Implement strict RFC 8785 JSON canonicalizer (`backend/basis/canon.py`).
- **A2:** Refactor `Formula` class to be immutable and hash-deterministic.
- **A3:** Extract `Axiom` and `InferenceRule` definitions into a pure data layer (no logic in definitions).
- **A4:** Unify normalization logic: one path for all expressions (Prop, FOL, Eq).

### Track B: Derivation & Lean (The Engine)
*Focus: `backend/axiom_engine`, `backend/lean_proj`*
- **B1:** Isolate `LeanInterface` – pure input/output wrapper around Lean 4 CLI.
- **B2:** Refactor `DerivationEngine` to accept explicit seeds (no global random).
- **B3:** Implement "Proof-or-Abstain" wrapper: failures are typed exceptions, not `False` or `None`.

### Track C: Ledger & Dual Attestation (The Memory)
*Focus: `backend/ledger`, `backend/crypto`*
- **C1:** Define `BlockHeader` and `ProofRecord` Pydantic schemas matching Whitepaper V2.
- **C2:** Implement `DualAttestation` class: binds `(R_t, U_t, H_t)`.
- **C3:** Refactor Merkle Tree implementation to use the A1 canonicalizer.

### Track D: RFL & Curriculum (The Brain)
*Focus: `backend/rfl`, `backend/frontier`*
- **D1:** Formalize `ReflexiveForgetting` algorithm as a standalone pure function.
- **D2:** Extract `Curriculum` state machine: explicit transitions between "slices".

### Track E: API & Interface (The Skin)
*Focus: `backend/api`*
- **E1:** Create v2 API schemas (strict, versioned).
- **E2:** Implement `APIShim` to translate v1 calls to v2 logic during transition.

### Track F: Security & Runtime (The Immune System)
*Focus: `backend/security`, `backend/worker.py`*
- **F1:** Hardening: Add input validation middleware to all entrypoints.
- **F2:** Worker isolation: Ensure worker crashes do not corrupt ledger state.

### Track G: Determinism Harness (The Test)
*Focus: `tests/`, `backend/repro`*
- **G1:** Create `FirstOrganismHarness`: Runs the B-D-C loop in a sandbox.
- **G2:** Determinism verification script: Runs G1 twice, asserts sha256 equality of all outputs.

---

## Layer 3: MDAP Execution Details

### Micro-Task Specs

#### A1: RFC 8785 Canonicalizer
- **Files:** `backend/basis/canon.py`
- **Invariants:** Must match `jcs` reference implementation output.
- **MOV:** Unit tests with edge-case JSONs (unicode, sorting).

#### C2: Dual Attestation
- **Files:** `backend/crypto/attestation.py`
- **Invariants:** `verify(sign(m, k), k) == True`. `R_t` derived from Modus Ponens DAG.
- **MOV:** Integration test generating a dummy chain and verifying signatures.

#### G1: First Organism Harness
- **Files:** `tests/integration/harness_v2.py`
- **Invariants:** End-to-end execution without network calls (mocked Lean if needed for speed).
- **MOV:** Runs in < 30s.

*(Full spec for all tasks implies adherence to VSD constraints)*

---

## Layer 4: Global Reviewer Hooks

The Global Reviewer Agent must intervene at these checkpoints:

1.  **Post-Track A (Substrate):** Verify foundation is solid before building the engine.
    *   *Check:* Are hash functions consistent across the codebase?
2.  **Post-Track C (Ledger):** Verify data structure integrity.
    *   *Check:* Is the Dual Attestation scheme correctly implemented?
3.  **Post-Track G (First Organism):** The Final Exam.
    *   *Check:* Does the `harness_v2.py` produce identical artifacts on two runs?

---

## Layer 5: Telemetry & Learning

All agents executing tasks must append to `ops/telemetry_vcp21.jsonl`:

```json
{
  "task_id": "A1",
  "status": "success",
  "agent": "cursor-small",
  "timestamp": "2023-10-27T10:00:00Z",
  "files_touched": ["backend/basis/canon.py"],
  "vibe_score": 0.95,
  "notes": "Replaced custom json.dumps with jcs library."
}
```

**Architecture Update Trigger:**
If `vibe_score` drops below 0.7 on any task, HALT and request human/supervisor intervention.
