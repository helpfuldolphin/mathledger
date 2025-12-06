# VSD.md — Vibe Specification Document for MathLedger

**Version:** 1.0
**Status:** Active
**Protocol:** VCP 2.1 / First Organism
**Author:** Claude A (Vibe Orchestrator & Intent Compiler)

---

## 1. Executive Summary

This VSD defines the **canonical vibe** for MathLedger: a verifiable ledger of mathematical truths. The codebase currently contains **redundant, experimental, and partially broken code** from multiple agent passes. This document establishes the criteria for what belongs in the **basis** (minimal spanning set) versus what remains as **exploratory slop** (archived but never promoted).

The immediate priority is achieving **First Organism** — the closed-loop integration test that proves:
1. UI event → Curriculum Gate → Derivation → Lean verification/abstention → Ledger Ingestion → Dual-Root Attestation → RFL metric recording

Until First Organism passes cleanly with deterministic, recomputable `H_t`, **no outreach**.

---

## 2. Structural Vibe

### 2.1 Directory Hierarchy (Target State)

```
mathledger/
├── basis/                    # CANONICAL: Minimal spanning set
│   ├── core/types.py         # Immutable value types (Block, DualAttestation, etc.)
│   ├── crypto/hash.py        # Single SHA-256 implementation with domain separation
│   ├── attestation/dual.py   # Dual-root attestation (R_t, U_t, H_t)
│   ├── ledger/block.py       # Block sealing primitives
│   ├── curriculum/ladder.py  # CurriculumTier, CurriculumIndex
│   └── logic/normalizer.py   # Formula normalization (if promoted)
├── normalization/            # CANONICAL: Logic canonicalization
│   ├── canon.py              # Unicode→ASCII, right-assoc →, sorted ∧/∨
│   ├── taut.py               # Tautology checking
│   └── truthtab.py           # Truth table evaluation
├── attestation/              # CANONICAL: Dual-root attestation implementation
│   └── dual_root.py          # RFC 8785, Merkle trees, composite root
├── derivation/               # CANONICAL: Derivation pipeline (new)
│   ├── pipeline.py           # Statement processing pipeline
│   ├── axioms.py             # K/S schema instantiation
│   ├── derive_rules.py       # MP inference, pattern recognition
│   └── verification.py       # 3-layer verification
├── curriculum/               # CANONICAL: Curriculum gates
│   └── gates.py              # Gate specifications
├── interface/api/            # CANONICAL: FastAPI orchestrator
│   ├── app.py                # API server
│   └── schemas.py            # Pydantic models
├── backend/                  # TRANSITIONAL: Being refactored
│   ├── axiom_engine/         # Legacy derivation (shimmed to derivation/)
│   ├── crypto/               # Legacy hashing (consolidate to basis/crypto/)
│   ├── ledger/               # Legacy blocking (extract to basis/ledger/)
│   ├── worker.py             # Lean verification worker (keep)
│   ├── rfl/                  # RFL evidence gathering (keep)
│   └── frontier/             # Curriculum/ratchet (consolidate to curriculum/)
├── tests/                    # Test suite
├── docs/                     # Documentation
├── migrations/               # Database schema
└── archive/                  # SLOP: Preserved but never promoted
    ├── substrate/            # Duplicate formal machinery
    ├── experimental/         # Root-level experiments
    └── consensus_variants/   # Unused consensus implementations
```

### 2.2 What Counts as "Basis-Grade" Code

Code is **basis-grade** if and only if it:

1. **Is used by First Organism path**: Must be reachable from `tests/integration/test_first_organism.py`
2. **Has single responsibility**: One module, one purpose, no ambient state
3. **Is deterministic**: Same inputs → same outputs, no timestamps, no PRNG without explicit seed
4. **Is crypto-rigorous**: Uses domain separation, follows HASH_PIPELINE_SPEC
5. **Is test-covered**: Unit tests + integration test coverage
6. **Has no external dependencies outside stdlib + approved list**: `hashlib`, `dataclasses`, `typing`

**Approved external dependencies for basis:**
- None for core modules (stdlib only)
- `pydantic` for API schemas only
- `psycopg` / `redis` for persistence layer only

### 2.3 What Counts as "Slop"

Code is **slop** if it:

1. **Duplicates basis functionality**: Multiple hash implementations, parallel module trees
2. **Is unreachable from First Organism**: Experimental consensus, federation protocols
3. **Has non-deterministic behavior**: Unseed timestamps, ambient PRNG, floating-point comparisons
4. **Is schema-intolerant**: Hard-codes column names, breaks on schema changes
5. **Is root-level experimental**: `bootstrap_metabolism.py`, `phase_ix_attestation.py`, `rfl_gate.py`

**Slop treatment:**
- Preserve in `archive/` for historical reference
- Never import from archived modules
- Do not delete (may contain research insights)
- Never promote without full review

---

## 3. Semantic Vibe

### 3.1 Determinism Contract

Every function in the basis must satisfy:

```
∀ inputs i: f(i) at time t₁ = f(i) at time t₂
```

**Violations to eliminate:**
- `datetime.now()` → use `deterministic_timestamp(seed)`
- `uuid.uuid4()` → use `deterministic_uuid(seed, index)`
- `dict` iteration order → use `sorted()` or `collections.OrderedDict`
- Floating-point comparisons → use `math.isclose()` with explicit tolerance

### 3.2 Crypto-Rigorous Hashing

All hashing follows the whitepaper identity:

```
hash(s) = SHA256(DOMAIN_TAG || canonical_bytes(s))
```

**Domain tags (from `basis/crypto/hash.py`):**
| Tag | Byte | Purpose |
|-----|------|---------|
| `DOMAIN_LEAF` | `0x00` | Merkle leaf |
| `DOMAIN_NODE` | `0x01` | Merkle internal node |
| `DOMAIN_STMT` | `0x02` | Statement hash |
| `DOMAIN_BLOCK` | `0x03` | Block payload |
| `DOMAIN_REASONING_EMPTY` | `0x10` | Empty reasoning tree |
| `DOMAIN_UI_EMPTY` | `0x11` | Empty UI tree |

**Single source of truth:** `basis/crypto/hash.py`

**Violations to eliminate:**
- `tools/verify_merkle.py` — missing domain separation
- `substrate/crypto/hashing.py` — duplicate of basis
- `backend/crypto/hashing.py` + `backend/crypto/core.py` — dual implementations

### 3.3 Proof-or-Abstain Semantics

Every derivation attempt results in exactly one of:
1. **SUCCESS**: Proof verified (pattern match, truth table, or Lean)
2. **FAILURE**: Proof rejected (disproof found)
3. **ABSTAIN**: Insufficient resources to decide (timeout, depth limit)

**Abstentions are recorded in the ledger**, not silently dropped. The dual-root attestation includes abstention events in `R_t`.

### 3.4 First Organism Loop (Protocol)

```
U_t → R_t → H_t → RFL

Where:
  U_t = UI events (user/AI input)
  R_t = Reasoning Merkle root (proofs + abstentions)
  H_t = SHA256(R_t || U_t)  # Composite attestation root
  RFL = Reflexive evidence gathering (coverage, uplift, velocity)
```

**Gate conditions for curriculum advancement:**
- Coverage CI lower bound ≥ 0.92
- Abstention rate ≤ configured threshold
- Velocity ≥ min proofs/sec
- All metrics from 40-run experiment

---

## 4. Aesthetic Vibe

### 4.1 Investor-Grade Quality

Code that touches the First Organism path must be:
- **Auditable**: Clear data flow, explicit dependencies
- **Documented**: Docstrings with input/output contracts
- **Typed**: Type annotations on all public interfaces
- **Testable**: Unit tests for pure functions, integration tests for pipelines

### 4.2 Research-Grade Rigor

Experimental code in `backend/rfl/`, `backend/causal/`, etc. may be less polished but must:
- Be clearly marked as experimental
- Not be imported by basis modules
- Have at least smoke tests

### 4.3 No Hacks in the Basis

The basis must not contain:
- Magic constants without documentation
- Commented-out code
- `# TODO` without associated issue
- `try/except: pass` (silent swallowing)
- `print()` for logging (use proper logger or remove)
- Schema-tolerance hacks (dynamic column detection belongs in shim layer)

### 4.4 Module Boundaries

```
basis/           → Pure, stdlib-only, no I/O
normalization/   → Pure, stdlib-only, no I/O
attestation/     → Pure + hashlib, no I/O
derivation/      → May call verifier, no direct DB
interface/       → I/O layer, calls persistence
backend/         → Legacy, transitional, may have I/O
```

---

## 5. Import Paths (Canonical)

### 5.1 For New Code

```python
# Logic / normalization
from normalization.canon import normalize, canonical_bytes

# Cryptography
from basis.crypto.hash import sha256_hex, merkle_root, DOMAIN_STMT

# Attestation
from attestation.dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)

# Core types
from basis.core.types import Block, DualAttestation, BlockHeader

# Derivation (when promoted)
from derivation.pipeline import DerivationPipeline

# Curriculum
from basis.curriculum.ladder import CurriculumTier, CurriculumIndex
```

### 5.2 Deprecated Imports (Shimmed)

```python
# DO NOT USE — shimmed to normalization/
from backend.logic.canon import normalize  # DEPRECATED

# DO NOT USE — shimmed to interface/api/
from backend.orchestrator.app import app  # DEPRECATED
```

---

## 6. Verification Checklist

Before any code is promoted to basis:

- [ ] **Determinism**: No `datetime.now()`, no unseeded PRNG
- [ ] **Domain separation**: Uses correct `DOMAIN_*` tag
- [ ] **Type annotations**: All public functions annotated
- [ ] **Docstrings**: Purpose, inputs, outputs, raises
- [ ] **Unit tests**: ≥80% coverage
- [ ] **Integration path**: Reachable from First Organism
- [ ] **No slop imports**: Does not import from `archive/`, `substrate/`, or deprecated modules
- [ ] **Schema tolerance**: Uses shim layer if touching DB

---

## 7. First Organism Acceptance Criteria

The First Organism is **ALIVE** when:

1. `tests/integration/test_first_organism.py` passes
2. `H_t` is recomputable from stored leaves
3. Coverage CI lower bound ≥ 0.92
4. Uplift CI lower bound > 1.0
5. No nondeterministic timestamps in attestation path
6. All hashes use domain separation per HASH_PIPELINE_SPEC

**Until these criteria are met, the codebase is in "pre-life" state.**

---

## 8. Change Control

Any modification to:
- `basis/**`
- `normalization/**`
- `attestation/**`
- `tests/integration/test_first_organism.py`

Requires:
1. Unit test coverage for changed code
2. First Organism integration test still passes
3. No new slop imports introduced
4. Documentation updated if behavior changes

---

*This VSD is the source of truth for MathLedger's canonical vibe. Deviations require explicit justification and approval.*
