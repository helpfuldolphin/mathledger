# The Great Condensation: Architecture Decision Record (ADR-001)

**Status:** DRAFT  
**Author:** Gemini B (Cartographer)  
**Date:** 2025-11-27

## 1. Objective
To consolidate the MathLedger "Organism" into a single, hermetic directory: `basis/`. This enforces the distinction between the immutable core (Law) and the supporting infrastructure (Economy/Metabolism).

## 2. The Current Topology (Entropy State)
The codebase currently suffers from "Sprawl," with core logic scattered across multiple root directories and the `backend/` monolith.

| Component | Current Location(s) | Status |
|---|---|---|
| **Logic/Canon** | `normalization/`, `backend/logic/` | Split/Shimmed |
| **Crypto** | `substrate/crypto/`, `backend/crypto/` | Split |
| **Ledger** | `ledger/`, `backend/ledger/` | Split |
| **Attestation** | `attestation/`, `backend/crypto/dual_root.py` | Split |
| **Derivation** | `derivation/`, `backend/axiom_engine/` | Split |
| **RFL** | `rfl/`, `backend/rfl/` | Split |
| **Curriculum** | `curriculum/` | Root |

## 3. The Target Topology (Basis State)
All "Organism" code must reside within `basis/`. The `basis/` directory must generally **not** import from outside itself (except standard libs).

```text
basis/
├── attestation/    # Dual root logic (Law)
├── core/           # Shared types (HexDigest, Block)
├── crypto/         # Canonical hashing (SHA256)
├── curriculum/     # Learning schedule
├── derivation/     # Axiom engine & inference rules
├── ledger/         # Block structure & ingestion
├── logic/          # Normalization & Truth Tables
├── rfl/            # Reinforced Feedback Loop core
└── substrate/      # Lean interface & formal proofs
```

## 4. Migration Strategy (The Condensation)

### Phase 1: The Logic Core (Foundation)
*   **Move**: `normalization/*` → `basis/logic/`
*   **Deprecate**: `backend/logic/canon.py` (delete shims)
*   **Action**: Update all imports from `normalization` to `basis.logic`.

### Phase 2: The Crypto Layer
*   **Move**: `substrate/crypto/*` → `basis/crypto/`
*   **Merge**: `backend/crypto/hashing.py` → `basis/crypto/hash.py`
*   **Action**: Standardize on `basis.crypto`.

### Phase 3: The Structure Layer (Ledger/Attestation)
*   **Move**: `attestation/` → `basis/attestation/`
*   **Move**: `ledger/` → `basis/ledger/`
*   **Action**: Ensure `basis/ledger` only depends on `basis/crypto` and `basis/logic`.

### Phase 4: The Engine Layer (Derivation/RFL)
*   **Move**: `derivation/` → `basis/derivation/`
*   **Move**: `rfl/` → `basis/rfl/`
*   **Refactor**: Extract pure logic from `backend/axiom_engine` if missing in `basis/derivation`.
*   **Cleanup**: Delete `backend/rfl` shims.

## 5. Dependency Firewall
Once migrated, we will enforce the following rules via CI:
1.  `basis/*` CANNOT import `backend/*`
2.  `basis/*` CANNOT import `ops/*`
3.  `basis/*` CANNOT import `apps/*`
4.  `basis/*` CANNOT import `interface/*`

## 6. Immediate Action Plan (Next Steps)
1.  **Execute Phase 1**: Move normalization to `basis/logic`.
2.  **Repair Imports**: Run global search-and-replace for `normalization.canon` -> `basis.logic.canon`.
3.  **Verify Tests**: Ensure `pytest tests/test_canon.py` passes.

## 7. Entropy Quarantine
The following directories are slated for archival/deletion post-migration:
*   `backend/logic`
*   `backend/rfl`
*   `normalization` (root)
*   `substrate` (root python parts)
*   `attestation` (root)
*   `ledger` (root)

