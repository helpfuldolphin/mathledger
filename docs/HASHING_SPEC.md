# Hashing & Canonicalization Specification

## Overview

This document defines the single source of truth for identity generation in the First Organism (MathLedger).
All identities (statements, proofs, blocks) must be derived via the canonical path:

$$ \mathrm{hash}(s) = \mathrm{SHA256}(\mathcal{D} || \mathcal{E}(\mathcal{N}(s))) $$

Where:
- $\mathcal{N}$ is the Normalization function (AST reduction)
- $\mathcal{E}$ is the Encoding function (UTF-8 / ASCII strictness)
- $\mathcal{D}$ is the Domain Separation Tag
- $||$ denotes concatenation

## Domain Separation

To prevent second-preimage attacks (e.g. CVE-2012-2459), all hashes are prefixed with a single byte domain tag:

| Domain | Tag | Usage |
|--------|-----|-------|
| `LEAF` | `0x00` | Merkle Tree Leaf Nodes |
| `NODE` | `0x01` | Merkle Tree Internal Nodes |
| `STMT` | `0x02` | Statement Content Identity |
| `BLCK` | `0x03` | Block Header Identity |
| `FED_` | `0x04` | Federation Namespace |
| `NODE_`| `0x05` | Node Attestation |
| `DOSSIER_` | `0x06` | Celestial Dossier |
| `ROOT_`| `0x07` | Root Hash |

## 1. Statement Canonicalization ($\mathcal{N}$)

Statements are normalized before hashing to ensure semantic equivalence maps to the same identity.

### Rules

1.  **Implication Right-Association**:
    -   `p -> (q -> r)` remains `p->q->r`
    -   `p -> q -> r` becomes `p->q->r`
    -   `(p -> q) -> r` preserves structure

2.  **Commutative Flattening**:
    -   Conjunctions (`/\`) and Disjunctions (`\/`) are flattened and sorted lexicographically.
    -   `q /\ p` becomes `p/\q`
    -   `p /\ (q /\ r)` becomes `p/\q/\r`
    -   Duplicates are removed: `p /\ p` becomes `p`

3.  **Unicode Normalization**:
    -   All Unicode logic symbols are converted to their ASCII equivalents.
    -   `→`, `⇒`, `⟹` become `->`
    -   `∧`, `⋀` become `/\\`
    -   `∨`, `⋁` become `\\/`
    -   `¬`, `￢`, `~` become `~`
    -   `↔`, `⇔` become `<->`

4.  **Whitespace**: All whitespace is removed.

5.  **Parentheses**: Redundant outer parentheses are removed. `((p))` -> `p`.

### Implementation
Refer to `normalization.canon.normalize`.

## 2. Byte Encoding ($\mathcal{E}$)

1.  **Input**: Normalized string (ASCII-only guaranteed by $\mathcal{N}$).
2.  **Encoding**: UTF-8 (which is identical to ASCII for the normalized subset).
3.  **Verification**: The `canonical_bytes` function enforces that the output is strict ASCII.

## 3. Hashing

All hashing uses **SHA-256**.

### Statement Hash
```python
h = sha256(DOMAIN_STMT + canonical_bytes(statement))
```

### Block Hash
```python
h = sha256(DOMAIN_BLCK + canonical_json(block_header))
```

### Merkle Root
The Merkle Tree construction is strictly defined:
1.  **Leaves**: `L_i = sha256(DOMAIN_LEAF + leaf_content)`
2.  **Sorting**: Leaves are sorted by their hash values.
3.  **Nodes**: `N = sha256(DOMAIN_NODE + left_child + right_child)`
4.  **Odd Nodes**: If a level has an odd number of nodes, the last node is duplicated.

## 4. Ad-Hoc Hashing

**Strictly Forbidden.**
No consumer should perform `hashlib.sha256(str(s).encode())`. All hashing must flow through `backend.crypto.core`.
