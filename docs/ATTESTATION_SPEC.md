# Dual-Root Attestation Specification

This document captures the canonical style guide for Rₜ, Uₜ and Hₜ inside MathLedger. The First Organism integration test (`tests/integration/test_first_organism.py`) acts as the **Minimum Viable Demonstration of Life (MVDP)**, proving the operational reality of the dual-root attestation chain.

## Terminology

| Symbol | Meaning |
| --- | --- |
| `Rₜ` | Reasoning Merkle root over deterministic proof artifacts (`compute_reasoning_root`). |
| `Uₜ` | Human/UI Merkle root over canonicalized UI events (`compute_ui_root`). |
| `Hₜ = SHA256(Rₜ ∥ Uₜ)` | Composite epistemic root (`compute_composite_root`). |
| `AttestationLeaf` | Dataclass storing `canonical_value`, `leaf_hash`, and inclusion proof for either stream. |
| `attestation_metadata` | JSON blob emitted by `generate_attestation_metadata()` containing algorithm identifiers, counts, and serialized leaves. |

All leaf canonicalization flows through RFC 8785 JSON normalization to keep the hashing deterministic regardless of whitespace or ordering. Reasoning leaves are hashed with a dedicated reasoning leaf domain tag; UI leaves use a distinct UI domain tag. Internal nodes keep their own node domain tag before reaching the root.

## Canonical workflow

1. **Capture UI events** via `backend/ledger/ui_events.capture_ui_event` (or the attestation API `POST /attestation/ui-event`). The thread-safe store timestamps and canonicalizes each payload and exposes artifacts via `materialize_ui_artifacts()`.  
2. **Seed proofs** and feed them into `seal_block_with_dual_roots()`:  
   - The proof payload becomes a reasoning leaf.  
   - Either the caller supplies UI events or the helper drains the event store.  
3. `seal_block_with_dual_roots()` returns a block record with:  
   - `reasoning_merkle_root`, `ui_merkle_root`, `composite_attestation_root`.  
   - `attestation_metadata` containing `reasoning_leaves`, `ui_leaves`, event counts, and algorithm identifiers (`"SHA256"`, `"v2"`).  
4. **Consumers** can recompute `Rₜ`/`Uₜ` from the stored leaves and verify `Hₜ` deterministically via `compute_composite_root()`. The leaves expose inclusion proofs that can be verified via `backend.crypto.hashing.verify_merkle_proof`.

## First Organism Example (Single-Leaf)

The test `tests/integration/test_first_organism.py` illustrates the chain with a simple abstain statement. The sequence is documented below.

### 1. Input payloads

The canonical First Organism example uses these exact payloads (mirrored in `tests/test_dual_root_attestation.py::test_first_organism_sample_metadata`):

```json
{
  "proof": {
    "statement": "p -> p",
    "statement_hash": "c3f5b3205153cf1c5a2b8a0c3694a7c3",
    "status": "abstain",
    "prover": "lean-interface",
    "verification_method": "lean-disabled",
    "reason": "mock dominant statement"
  },
  "ui_event": {
    "event_type": "select_statement",
    "statement_hash": "c3f5b3"
  }
}
```

### 2. Captured leaves

After RFC 8785 canonicalization:

| Stream | Canonical Value |
|--------|-----------------|
| Reasoning | `{"prover":"lean-interface","reason":"mock dominant statement","statement":"p -> p","statement_hash":"c3f5b3205153cf1c5a2b8a0c3694a7c3","status":"abstain","verification_method":"lean-disabled"}` |
| UI | `{"event_type":"select_statement","statement_hash":"c3f5b3"}` |

Domain-separated leaf hashes:

| Stream | Leaf Hash |
|--------|-----------|
| Reasoning | `98db47d99674abe1f6be5ba5672660b8057a616c6d5423a478470dfc1b344a26` |
| UI | `d692382bd19d9113c44ecdadc37fc3839139bba0d4c0e471e3f4d76ce187a58e` |

### 3. Attestation metadata

With the single-leaf trees the Merkle roots equal the leaf hashes; the composite root is:

```json
{
  "reasoning_merkle_root": "142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902",
  "ui_merkle_root": "3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee",
  "composite_attestation_root": "6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59",
  "reasoning_event_count": 1,
  "ui_event_count": 1,
  "attestation_version": "v2",
  "algorithm": "SHA256",
  "composite_formula": "SHA256(R_t || U_t)",
  "leaf_hash_algorithm": "sha256",
  "reasoning_leaves": [
    {
      "original_index": 0,
      "sorted_index": 0,
      "canonical_value": "{\"prover\":\"lean-interface\",\"reason\":\"mock dominant statement\",\"statement\":\"p -> p\",\"statement_hash\":\"c3f5b3205153cf1c5a2b8a0c3694a7c3\",\"status\":\"abstain\",\"verification_method\":\"lean-disabled\"}",
      "leaf_hash": "98db47d99674abe1f6be5ba5672660b8057a616c6d5423a478470dfc1b344a26",
      "merkle_proof": []
    }
  ],
  "ui_leaves": [
    {
      "original_index": 0,
      "sorted_index": 0,
      "canonical_value": "{\"event_type\":\"select_statement\",\"statement_hash\":\"c3f5b3\"}",
      "leaf_hash": "d692382bd19d9113c44ecdadc37fc3839139bba0d4c0e471e3f4d76ce187a58e",
      "merkle_proof": []
    }
  ],
  "system": "pl"
}
```

### 4. Deterministic recomputation

The integration test (`tests/integration/test_first_organism.py`) and unit test (`tests/test_dual_root_attestation.py::test_first_organism_sample_metadata`) both verify recomputability:

```python
from attestation.dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)

# Extract canonical values from stored leaves
reasoning_canonicals = [leaf["canonical_value"] for leaf in metadata["reasoning_leaves"]]
ui_canonicals = [leaf["canonical_value"] for leaf in metadata["ui_leaves"]]

# Recompute using the exact same primitives
recomputed_r = compute_reasoning_root(reasoning_canonicals)
recomputed_u = compute_ui_root(ui_canonicals)
recomputed_h = compute_composite_root(recomputed_r, recomputed_u)

# Assert determinism
assert recomputed_r == "142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902"
assert recomputed_u == "3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee"
assert recomputed_h == "6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59"
```

### 5. Dual attestation gate

The RFL runner records `RunLedgerEntry` instances pointing back to `Hₜ`. If the attestation metadata is present, the runner's `_dual_attest()` helper inspects both `coverage` and `uplift` bootstrap results—`verify_metabolism()` then uses the recomputed `Hₜ` from the block for policy updates (abstention mass is reflected in the ledger entry; `dual_attestation_records` is annotated with the computed hash).

---

## Multi-Leaf Example

The test `tests/test_dual_root_attestation.py::TestMultiLeafAttestation` exercises Merkle tree construction with 3 reasoning leaves and 2 UI leaves. This validates non-trivial proof paths and sorted index assignment.

### 1. Input payloads

```json
{
  "reasoning": [
    {"statement": "p -> p", "statement_hash": "c3f5b3205153cf1c5a2b8a0c3694a7c3", "status": "success", "prover": "lean-interface"},
    {"statement": "(p -> q) -> (p -> q)", "statement_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", "status": "success", "prover": "lean-interface"},
    {"statement": "p -> (q -> p)", "statement_hash": "d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2", "status": "abstain", "prover": "lean-interface", "reason": "non-tautology"}
  ],
  "ui": [
    {"event_type": "select_statement", "statement_hash": "c3f5b3"},
    {"event_type": "toggle_abstain", "statement_hash": "d7e8f9", "value": true}
  ]
}
```

### 2. Leaf hashes and proofs

| Stream | Index | Leaf Hash | Proof Length |
|--------|-------|-----------|--------------|
| Reasoning | 0 | `b06a76968fd774e33550a10fbc58ad8e4f897f0932d5a99a4b4bb5f8223b07c8` | 2 |
| Reasoning | 1 | `d0591631fe41174777215473932ea01789e21ffbe2effaa3c09d1c1d96b82d27` | 2 |
| Reasoning | 2 | `db7aacc075ff81bdc355da1ca8d8a74a964ed995851aa677099a57f77c81bf52` | 2 |
| UI | 0 | `d692382bd19d9113c44ecdadc37fc3839139bba0d4c0e471e3f4d76ce187a58e` | 1 |
| UI | 1 | `792b3a2593f5cb7d190a6d8e9db2527ac2405cd859aafb943c6f7129892c5e24` | 1 |

### 3. Merkle roots

```
R_t = a17dfa7651b8d88e575e398da11c53ad85cc3f9a9623e1253733e42407768e78
U_t = 4556737241f76696ab1eefb321ab62540eea34fc6c374ffbacd1d549248e5ece
H_t = e8469a887f99abd5f59df6e409ca7bd07f5c1342cec7877a41bc01a3b517ca22
```

### 4. Verification

```python
from attestation.dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)
from backend.crypto.hashing import verify_merkle_proof

# Recompute roots from stored leaves
recomputed_r = compute_reasoning_root([leaf["canonical_value"] for leaf in block["reasoning_leaves"]])
recomputed_u = compute_ui_root([leaf["canonical_value"] for leaf in block["ui_leaves"]])
recomputed_h = compute_composite_root(recomputed_r, recomputed_u)

assert recomputed_r == "a17dfa7651b8d88e575e398da11c53ad85cc3f9a9623e1253733e42407768e78"
assert recomputed_u == "4556737241f76696ab1eefb321ab62540eea34fc6c374ffbacd1d549248e5ece"
assert recomputed_h == "e8469a887f99abd5f59df6e409ca7bd07f5c1342cec7877a41bc01a3b517ca22"

# Verify each leaf's inclusion proof
for leaf in block["reasoning_leaves"]:
    proof_path = [(sibling, is_left) for sibling, is_left in leaf["merkle_proof"]]
    assert verify_merkle_proof(leaf["leaf_hash"], proof_path, block["reasoning_merkle_root"])
```

---

## RFC 8785 Canonicalization

All attestation metadata uses RFC 8785 (JCS) for deterministic JSON serialization:

1. **Keys sorted lexicographically**: `{"a":1,"b":2}` not `{"b":2,"a":1}`
2. **No whitespace**: Compact encoding with no spaces after `:` or `,`
3. **Numbers as-is**: No scientific notation normalization
4. **Unicode escaping**: Only escape control characters

The `attestation.dual_root.canonicalize_reasoning_artifact()` and `canonicalize_ui_artifact()` functions enforce this via `substrate.crypto.core.rfc8785_canonicalize()`.

---

## Merkle Proof Edge Cases

The test class `tests/test_dual_root_attestation.py::TestMerkleProofEdgeCases` validates:

| Case | Leaves | Expected Proof Length |
|------|--------|----------------------|
| Single leaf | 1 | Tree-dependent (leaf ≠ root) |
| Two leaves | 2 | 1 sibling each |
| Power of 2 | 4 | 2 siblings each |
| Non-power of 2 | 5 | Variable (still valid) |

Tampered leaves or siblings must fail `verify_merkle_proof()`.

---

## Canonical Integration Artifact

The canonical JSON artifact below is auto-generated by `tests/integration/test_first_organism.py`, stored in `artifacts/first_organism/attestation.json`, and synchronized to `apps/ui/test/fixtures/first_organism_attestation.json` for downstream consumers.

```json
{
  "blockNumber": 1,
  "blockHash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "reasoningMerkleRoot": "142a7c15ac8d44d2e13aa1006febdaa46f66cc4d69b89ebae5cba472e8b47902",
  "uiMerkleRoot": "3c9a33d055fd8f7f61d2dbe1f9835889efea13427837a2213355f26495cfedee",
  "compositeAttestationRoot": "6a006e789be39105a37504e12305f5358baad273e15fb757f266c0f46b116e59",
  "reasoningLeaves": [
    {
      "originalIndex": 0,
      "sortedIndex": 0,
      "canonicalValue": "{\"prover\":\"lean-interface\",\"reason\":\"mock dominant statement\",\"statement\":\"p -> p\",\"statement_hash\":\"c3f5b3205153cf1c5a2b8a0c3694a7c3\",\"status\":\"abstain\",\"verification_method\":\"lean-disabled\"}",
      "leafHash": "98db47d99674abe1f6be5ba5672660b8057a616c6d5423a478470dfc1b344a26",
      "merkleProof": []
    }
  ],
  "uiLeaves": [
    {
      "originalIndex": 0,
      "sortedIndex": 0,
      "canonicalValue": "{\"event_type\":\"select_statement\",\"statement_hash\":\"c3f5b3\"}",
      "leafHash": "d692382bd19d9113c44ecdadc37fc3839139bba0d4c0e471e3f4d76ce187a58e",
      "merkleProof": []
    }
  ],
  "metadata": {
    "attestation_version": "v2",
    "reasoning_event_count": 1,
    "ui_event_count": 1,
    "composite_formula": "SHA256(R_t || U_t)",
    "leaf_hash_algorithm": "sha256",
    "algorithm": "SHA256",
    "system": "pl"
  }
}
```

**To regenerate:**  
1. Run `pytest tests/integration/test_first_organism.py -k closed_loop_happy_path`.  
2. Copy `artifacts/first_organism/attestation.json` to `apps/ui/test/fixtures/first_organism_attestation.json`.  
3. Re-run `tests/test_dual_root_attestation.py::test_first_organism_sample_metadata` to confirm recomputability.

---

Documenting this flow makes the First Organism test the canonical reference for any future dual-root proof; promote it to the basis repo by citing `tests/integration/test_first_organism.py` and `docs/FIRST_ORGANISM.md` when deriving Wave 1 documentation.
