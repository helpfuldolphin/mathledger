# MathLedger Post-Quantum Migration: Safety Case v0.9

**Document Version**: 0.9 (Draft for External Review)  
**Author**: Manus-H, PQ Migration General  
**Date**: December 10, 2025  
**Classification**: Public

---

## 1. Introduction and Threat Model

This document presents the safety case for MathLedger's post-quantum (PQ) cryptographic migration. The primary threat addressed is the potential for a cryptographically relevant quantum computer (CRQC) to break the security assumptions of classical hash functions, specifically SHA-256, which underpins the integrity of the ledger's historical data [1]. A successful attack could compromise the immutability of the blockchain, enabling an adversary to forge transactions, rewrite history, or create invalid blocks that appear valid.

The migration architecture is designed to transition the ledger from a classical hash function (SHA-256) to a quantum-resistant alternative (e.g., SHA3-256) without a hard fork, service interruption, or loss of historical verifiability. This safety case demonstrates that the migration process itself is secure and maintains the integrity of the chain at all stages.

Our threat model assumes a sophisticated adversary with the following capabilities:

- **Preimage Attack**: The ability to find an input `m` for a given hash output `h` such that `H(m) = h`.
- **Second Preimage Attack**: The ability to find a different input `m'` for a given input `m` such that `H(m) = H(m')`.
- **Collision Attack**: The ability to find two different inputs `m1` and `m2` such that `H(m1) = H(m2)`.

We assume the adversary may succeed in breaking the preimage resistance of **either** the legacy hash function (SHA-256) **or** the new PQ hash function, but not both simultaneously during the migration period.

---

## 2. Dual-Commitment Protection Against Preimage Breaks

The core of the migration's security rests on a **dual-commitment** scheme. During the migration phases (specifically Phases 2-4), block headers contain two separate Merkle roots: one computed with the legacy algorithm (`legacy_merkle_root`) and one with the PQ algorithm (`pq_merkle_root`). These two roots are then bound together into a single, final commitment.

> **Definition: Dual Commitment**
> A cryptographic scheme that binds two independent commitments, `C_A` and `C_B`, into a single meta-commitment, `C_meta`. The security of `C_meta` holds even if the cryptographic assumptions of one of the underlying commitments are broken.

In our implementation, the dual commitment is constructed as follows:

`dual_commitment = H_PQ(domain_prefix | legacy_merkle_root | pq_merkle_root)`

Where:
- `H_PQ` is the new, quantum-resistant hash function.
- `domain_prefix` is a constant string (`DUAL_COMMIT:`) to prevent collisions with other hash domains.
- `legacy_merkle_root` is the Merkle root of the block's transactions computed with SHA-256.
- `pq_merkle_root` is the Merkle root of the block's transactions computed with the PQ hash algorithm.

This construction provides a robust defense against the failure of either hash function.

### 2.1. Security Argument: Protection Against Preimage Attack on SHA-256

Let us assume an adversary has successfully broken the preimage resistance of SHA-256. Their goal is to forge a block by creating a malicious set of transactions that hashes to the same `legacy_merkle_root` as a valid block.

1.  **Adversary's Action**: The adversary finds a malicious transaction set `T_malicious` such that `SHA256(T_malicious) = legacy_merkle_root`.
2.  **The Hurdle**: To create a valid block, the adversary must also produce a valid `dual_commitment`. This commitment is `H_PQ(domain_prefix | legacy_merkle_root | pq_merkle_root)`. Even though they have the correct `legacy_merkle_root`, they do not know the correct `pq_merkle_root` for their malicious transaction set.
3.  **The Catch**: The `pq_merkle_root` is computed with the quantum-resistant hash function `H_PQ`. Since we assume `H_PQ` is still secure, the adversary cannot find a second preimage for the `pq_merkle_root` of the valid transaction set. They are forced to compute a new `pq_merkle_root_malicious = H_PQ(T_malicious)`.
4.  **The Failure**: The adversary now has `legacy_merkle_root` and `pq_merkle_root_malicious`. They must compute a new dual commitment: `dual_commitment_malicious = H_PQ(domain_prefix | legacy_merkle_root | pq_merkle_root_malicious)`. Because `pq_merkle_root_malicious` is different from the original `pq_merkle_root`, the resulting `dual_commitment_malicious` will not match the `dual_commitment` in the valid block header.

**Conclusion**: An adversary who breaks SHA-256 cannot forge a block because they are unable to create a valid dual commitment. The security of the quantum-resistant hash function protects the integrity of the legacy hash within the commitment structure. The cost of forging the block is equivalent to breaking the preimage resistance of the PQ hash function, which we assume is computationally infeasible.

### 2.2. Security Argument: Protection Against Preimage Attack on the PQ Hash

This scenario is symmetric. Assume the adversary breaks the PQ hash function but not SHA-256.

1.  **Adversary's Action**: The adversary finds `T_malicious` such that `H_PQ(T_malicious) = pq_merkle_root`.
2.  **The Hurdle**: The adversary must still produce a valid `dual_commitment`. However, the dual commitment itself is computed using `H_PQ`. Since the adversary has broken `H_PQ`, they can theoretically find a preimage for the `dual_commitment`.
3.  **The Catch**: The input to the dual commitment hash is `domain_prefix | legacy_merkle_root | pq_merkle_root`. The adversary knows the target `dual_commitment` and the `pq_merkle_root`. To find a valid input, they must also know the `legacy_merkle_root`. However, this root is computed with SHA-256, which is still secure. The adversary cannot find a `legacy_merkle_root_malicious` that matches the original without breaking SHA-256.
4.  **The Failure**: The adversary is stuck. They cannot construct the correct input to the `H_PQ` function for the dual commitment because one of its components, `legacy_merkle_root`, is protected by a secure hash function.

**Conclusion**: The dual-commitment scheme creates a symbiotic security relationship. The integrity of the block is protected as long as at least one of the two hash functions remains secure against preimage attacks.

| **Scenario** | **Adversary's Capability** | **Security Guarantee** |
| :--- | :--- | :--- |
| SHA-256 Broken | Preimage attack on SHA-256 | Forgery is prevented by the preimage resistance of the PQ hash function used in the dual commitment. |
| PQ Hash Broken | Preimage attack on PQ hash | Forgery is prevented by the preimage resistance of SHA-256, which protects a key input to the dual commitment. |

---

## 3. Epoch Activation Invariants

An **epoch** is a period of blocks governed by a specific set of consensus rules, including a designated canonical hash algorithm. The transition from one epoch to another is a critical event that must be handled with extreme care to prevent security vulnerabilities. Our design enforces three core invariants for epoch activation.

> **Definition: Epoch Invariants**
> A set of rules that ensure the deterministic, unambiguous, and secure transition of the blockchain from one consensus regime to another.

1.  **Immutability**: Once an epoch is registered and its start block is finalized, its parameters (start block, algorithm ID, rule version) cannot be changed. This prevents retroactive alterations to the chain's history or consensus rules.
2.  **Monotonicity**: Epochs must be ordered by their `start_block` number. A new epoch can only be scheduled to start after the current epoch's start block. This ensures a linear and forward-moving progression of consensus rules.
3.  **Continuity**: The block range of epochs must be contiguous. There can be no gaps or overlaps between epochs. The end of one epoch is implicitly defined by the start of the next.

These invariants are enforced by the `EpochManager` and the consensus validation logic. A new epoch proposal is rejected if its `start_block` is not in the future. The epoch registry ensures that epochs are stored in a sorted, contiguous manner.

This strict, deterministic ordering of epochs is essential for the chain-of-custody guarantee.

---

## 4. Chain-of-Custody Guarantee Across Transitions

A primary challenge in cryptographic migration is maintaining a verifiable chain of history. Each block must point unambiguously to its parent, even if the parent was sealed with a different hash algorithm. This is the **chain-of-custody guarantee**.

Our architecture achieves this by using a dual `prev_hash` linkage during migration phases.

-   `prev_hash`: The hash of the parent block's header, computed using the **parent's** hash algorithm. This maintains the legacy chain.
-   `pq_prev_hash`: The hash of the parent block's header, computed using the **current block's** PQ hash algorithm. This establishes the new PQ chain.

### 4.1. The Epoch Transition

Consider a transition at block `N`, where block `N-1` was sealed with SHA-256 (Epoch 1) and block `N` is the first block to be sealed with a dual commitment using SHA3-256 (Epoch 2).

-   **Block `N-1` (Legacy)**:
    -   `header_hash = SHA256(header_N-1)`
-   **Block `N` (Dual Commitment)**:
    -   `prev_hash = header_hash` (from `N-1`, computed with SHA-256)
    -   `pq_prev_hash = SHA3-256(header_N-1)`

When a node validates block `N`, it performs two checks:

1.  It fetches block `N-1`, computes its header hash using the algorithm of Epoch 1 (SHA-256), and verifies that it matches `block_N.prev_hash`.
2.  It computes the hash of block `N-1`'s header using the algorithm of Epoch 2 (SHA3-256) and verifies that it matches `block_N.pq_prev_hash`.

This dual validation ensures that block `N` is correctly and unambiguously linked to block `N-1` in **both** cryptographic domains. The `prev_hash` field provides backward compatibility for legacy clients, while the `pq_prev_hash` field builds the forward-looking, quantum-resistant chain.

### 4.2. Security Argument: Preventing Forking Attacks at the Boundary

An adversary might attempt to create a fork at the epoch boundary by creating a malicious block `N'` that points to a valid `N-1` but contains different transactions.

1.  **Adversary's Action**: Creates a malicious block `N'` with a different transaction set.
2.  **The Hurdle**: Block `N'` must have a valid header. This means it must have a valid `dual_commitment` and valid `prev_hash` / `pq_prev_hash` fields.
3.  **The Failure**: As established in Section 2, the adversary cannot create a valid `dual_commitment` for `N'` without breaking both hash functions. Even if they could, they would also need to correctly compute both `prev_hash` (using SHA-256) and `pq_prev_hash` (using the PQ hash) of the parent block `N-1`. Any inconsistency in this linkage would be detected by the consensus validation rules.

**Conclusion**: The dual-linkage mechanism, combined with the dual-commitment scheme, ensures a secure and verifiable chain of custody across epoch boundaries. Historical verification remains possible because the epoch system tells the validator which algorithm to use for any given block number. A client can sync from genesis, applying the correct hash function for each epoch, and arrive at the same current chain state as a client that has been online the whole time.

---

## 5. References

[1] National Institute of Standards and Technology (NIST). (2024). *Post-Quantum Cryptography*. [https://csrc.nist.gov/projects/post-quantum-cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)

[2] Bernstein, D. J., & Lange, T. (2017). *Post-quantum cryptography*. Nature, 549(7671), 188-194.
