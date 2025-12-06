# VSD PHASE 2: Architectural Freeze & Decomposition Blueprint

> **STATUS: PHASE II — NOT YET IMPLEMENTED**
>
> This document describes **target architecture** beyond the current Phase I prototype.
> It is **NOT part of Evidence Pack v1** and should not be cited as existing functionality.
>
> The `basis/` package exists on disk but is **not yet imported by any production code**.
> All Phase I code uses the transitional modules (`normalization/`, `attestation/`, `rfl/`).
>
> **The existence of additional RFL logs does not activate any Phase II modules.**

**Status:** PHASE II TARGET | **Version:** 2.1-P2 | **Scope:** FUTURE DECOMPOSITION
**Author:** Claude A — Vibe Orchestrator & Structural Harmonizer
**Date:** 2025-11-29
**Prerequisite:** VCP 2.1 (VIBE_SPEC_MATHLEDGER_VCP21.md), DECOMPOSITION_PLAN.md

---

## 0. Phase I vs Phase II Scope

This section explicitly classifies what exists and works today (Phase I) versus what this document describes as target architecture (Phase II).

### ✅ Phase I — Actually Exists and Runs Today

The following are **proven and tested** in Evidence Pack v1:

| Component | Location | Evidence |
|-----------|----------|----------|
| First Organism closed-loop test | `tests/integration/test_first_organism.py` | Passes, produces `attestation.json` |
| Determinism test | `tests/integration/test_first_organism_determinism.py` | Passes |
| Dual-root attestation | `attestation/dual_root.py` | Used by FO test |
| Normalization | `normalization/canon.py` | Used by derivation |
| RFL Runner | `rfl/runner.py` | Used by FO test |
| Determinism helpers | `backend/repro/determinism.py`, `substrate/repro/determinism.py` | Used by FO harness |
| Sealed attestation artifact | `artifacts/first_organism/attestation.json` | On disk |
| 1000-cycle Dyno Chart | `artifacts/rfl/` (baseline vs RFL) | On disk |

### 🕒 Phase II — Described Here But NOT Implemented

The following are **target architecture only** and do not exist in working form:

| Component | Status | Notes |
|-----------|--------|-------|
| `basis/` as single source of truth | Files exist but **nothing imports them** | Zero consumers in codebase |
| Law → Economy → Metabolism enforcement | Conceptual only | No CI gate enforces this |
| Forbidden imports CI check for `basis/` | Not implemented | Described in Section 6.3 |
| Shim removal (2025-12-01) | Not done | Shims still active |
| `mypy --strict basis/` gate | Not in CI | Aspirational |
| RFC 8785 migration to `basis/crypto/json.py` | Not started | Section 8.2 |
| Micro-task queue (`ops/microtasks/`) | Not populated | DECOMPOSITION_PLAN.md describes schema only |
| MDAP micro-agents | Not implemented | Phase II theory |
| Wide slice theory curves | Not implemented | Phase II |
| ΔH scaling laws | Not implemented | Phase II |
| Imperfect verifier sandbox | Not implemented | Phase II |

### Critical Distinction

**Phase I proves**: The First Organism closed loop runs, produces deterministic $H_t$, and the RFL runner consumes it.

**Phase II describes**: How to refactor the codebase to consolidate all logic into `basis/` with strict layer boundaries.

### Phase I / Phase II Module Boundary

| Layer | Phase I (Active) | Phase II (Target) |
|-------|------------------|-------------------|
| Normalization | `normalization/canon.py` | `basis/logic/normalizer.py` |
| Attestation | `attestation/dual_root.py` | `basis/attestation/dual.py` |
| RFL Runner | `rfl/runner.py` | (same — already canonical) |
| Crypto | `substrate/crypto/core.py` | `basis/crypto/hash.py` |
| Determinism | `substrate/repro/determinism.py` | (same — already canonical) |

**All Phase I evidence was produced using the "Phase I (Active)" modules.** The `basis/` package has zero consumers regardless of what logs exist.

### Phase I RFL Evidence Snapshot

The following RFL log files exist on disk. Only `fo_rfl_50.jsonl` is canonical Phase I evidence.

| File | Cycles | Status | Phase I Claim? |
|------|--------|--------|----------------|
| `results/fo_rfl_50.jsonl` | 50 | **Complete** — Canonical run | ✅ YES |
| `results/fo_rfl.jsonl` | ~330 | **Degenerate** — lean-disabled, all-abstain | ❌ NO (not uplift evidence) |
| `artifacts/phase_ii/fo_series_1/fo_1000_baseline/` | 1000 | Baseline (no RFL) | ✅ YES (baseline only) |
| `artifacts/phase_ii/fo_series_1/fo_1000_rfl/` | 1000 | RFL comparison | ✅ YES (Dyno Chart source) |

**Clarifications:**
- `fo_rfl.jsonl` (~330 cycles) is NOT uplift evidence. It ran with `lean-disabled` and produced all-abstain events.
- The canonical RFL evidence for Phase I claims remains `fo_rfl_50.jsonl`.
- The 1000-cycle runs are used for the Dyno Chart comparison (baseline vs RFL abstention rates).
- No log file activates Phase II modules — all were produced with Phase I code paths.

### Phase II Uplift Evidence Gate (NOT YET ACTIVATED)

> **Gate Status: INACTIVE**
>
> No existing RFL logs qualify as uplift evidence. This gate defines future eligibility criteria only.

This section defines the **preconditions** under which a future RFL experiment could influence VSD governance decisions (e.g., `basis/` promotion, architectural changes, or claims of demonstrated uplift).

#### Eligibility Criteria

For an RFL experiment to be considered as potential uplift evidence, ALL of the following must be satisfied:

| Criterion | Requirement | Rationale |
|-----------|-------------|-----------|
| **Non-degenerate slice** | Experiment must use a slice with formal verification enabled (Lean for FOL+, or truth-table for propositional logic) and produces non-trivial proof attempts | 100% abstention runs are plumbing tests, not uplift |
| **Minimum cycle count** | N ≥ 100 cycles with successful proof attempts | Statistical power requires sufficient samples |
| **Baseline comparison** | Paired baseline run (same slice, same seed, RFL disabled) must exist | Uplift is relative to baseline, not absolute |
| **Statistically meaningful difference** | Abstention rate difference must be significant (e.g., p < 0.05 or bootstrap CI excludes zero) | Noise is not signal |
| **Determinism verified** | Both runs must pass determinism checks (same seed → same $H_t$) | Non-deterministic runs cannot be trusted |

#### Preregistration Requirements

Before an experiment can qualify as uplift evidence, the following must be documented **before** the run:

1. **Hypothesis**: What specific metric improvement is expected?
2. **Slice configuration**: Exact slice parameters (depth, atoms, gates)
3. **Success criteria**: Numeric thresholds that would constitute "uplift"
4. **Seed**: The deterministic seed to be used
5. **Cycle count**: Planned number of cycles

#### Manifest Requirements

After completion, the experiment must produce:

- [ ] `experiment_manifest.json` with preregistration hash
- [ ] `baseline_log.jsonl` and `rfl_log.jsonl` in paired directories
- [ ] `attestation.json` for both runs with matching $H_t$ on replay
- [ ] `statistical_summary.json` with CI bounds and p-values

#### What Uplift Evidence Would NOT Automatically Grant

Even if an experiment passes all criteria above, it would:

- **NOT** automatically promote `basis/` to production
- **NOT** override Phase I architectural decisions
- **NOT** constitute proof of the broader RFL thesis

It would only:

- Qualify as **one input** to future governance discussions
- Enable further experiments under similar conditions
- Allow cautious claims bounded by the specific slice/configuration tested

#### Current Status

**As of this writing, this gate is INACTIVE.**

| Existing Log | Passes Gate? | Reason |
|--------------|--------------|--------|
| `fo_rfl_50.jsonl` | ❌ NO | No baseline comparison, no preregistration |
| `fo_rfl.jsonl` | ❌ NO | Degenerate (100% abstention), lean-disabled |
| 1000-cycle runs | ❌ NO | No preregistration, exploratory only |

**No governance decision may cite "uplift" until this gate is satisfied.**

---

## 1. Purpose

This document describes the **target architecture** for MathLedger decomposition. It is intended as a roadmap for Phase II work, **not a description of current state**.

When fully implemented, it will define:

1. The canonical `basis/` package layout after decomposition
2. The **Law → Economy → Metabolism** vibe boundaries
3. The import graph with canonical paths and deprecated shims
4. The Hash / Normalization / Attestation single-source taxonomy
5. The **First Organism Determinism Envelope** — files allowed to affect $H_t$

---

## 2. The Three Vibes: Law → Economy → Metabolism

MathLedger's architecture is organized into three conceptual layers, each with distinct responsibilities, purity guarantees, and permissible dependencies.

### 2.1 Layer Definitions

```
┌─────────────────────────────────────────────────────────────────┐
│                         METABOLISM                               │
│  (RFL Runner, Curriculum Gates, Policy Updates, Ledger Entries) │
│  • Consumes H_t from Law layer                                  │
│  • Produces RunLedgerEntry, PolicyReward, SymbolicDescent       │
│  • MAY read config, metrics, historical state                   │
│  • MUST NOT affect H_t (post-seal)                              │
├─────────────────────────────────────────────────────────────────┤
│                          ECONOMY                                 │
│  (Derivation Engine, Lean Interface, Proof Search, Queues)      │
│  • Consumes axioms, rules, seeds                                │
│  • Produces proofs, candidates, verification results            │
│  • Feeds into attestation pipeline                              │
│  • MUST use deterministic primitives from Law layer             │
├─────────────────────────────────────────────────────────────────┤
│                            LAW                                   │
│  (Normalization, Hashing, Attestation, Block Sealing)           │
│  • Pure functions, no I/O, no side effects                      │
│  • Defines canonical identity: hash(s) = SHA256(D || E(N(s)))   │
│  • Produces R_t, U_t, H_t                                       │
│  • THE SINGLE SOURCE OF CRYPTOGRAPHIC TRUTH                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Dependency Rules

| From Layer  | May Import From           | May NOT Import From |
|-------------|---------------------------|---------------------|
| **Law**     | Python stdlib, typing     | Economy, Metabolism, I/O, DB, Redis |
| **Economy** | Law, Python stdlib        | Metabolism, Wall-clock time (except metrics) |
| **Metabolism** | Law, Economy, DB, Redis | Nothing is forbidden (but must not mutate H_t) |

### 2.3 Vibe Boundary Enforcement

```python
# LAW LAYER: basis/
# Pure, deterministic, zero external dependencies
from basis.logic.normalizer import normalize       # Pure
from basis.crypto.hash import sha256_hex           # Pure
from basis.attestation.dual import composite_root  # Pure

# ECONOMY LAYER: backend/axiom_engine/, substrate/
# Derivation logic, must use Law primitives
from backend.axiom_engine.derive import derive_step  # Uses basis.*
from substrate.repro.determinism import SeededRNG    # Deterministic

# METABOLISM LAYER: rfl/, backend/frontier/
# Consumes H_t, produces policy updates
from rfl.runner import RFLRunner                     # Uses H_t
from backend.frontier.curriculum import CurriculumSystem  # Gate evaluation
```

---

## 3. Canonical Basis Package Layout (Phase II Target)

> **Phase II**: The `basis/` package exists on disk but is not yet imported by production code.
> This section describes the **target state** after decomposition is complete.

After decomposition (Phase II), `basis/` will become the **single source of cryptographic and logical truth**:

```
basis/
├── __init__.py              # Re-exports all public symbols
├── core/
│   ├── __init__.py
│   └── types.py             # HexDigest, NormalizedFormula, Block, DualAttestation
├── crypto/
│   ├── __init__.py
│   └── hash.py              # Domain-separated SHA-256, Merkle operations
├── logic/
│   ├── __init__.py
│   └── normalizer.py        # Canonical normalization (N function)
├── ledger/
│   ├── __init__.py
│   └── block.py             # seal_block, block_to_dict, block_json
├── attestation/
│   ├── __init__.py
│   └── dual.py              # R_t, U_t, H_t computation and verification
├── curriculum/
│   ├── __init__.py
│   └── ladder.py            # CurriculumLadder, tier definitions
└── docs/
    └── invariants.md        # Mathematical invariants (reference)
```

### 3.1 Public API Surface (Phase II Target)

After migration, all external consumers SHOULD import from `basis` or `basis.*`.

**Current State (Phase I)**: No code imports from `basis/`. All imports use `normalization/`, `attestation/`, `rfl/`.

```python
# CANONICAL IMPORTS (Use These)
from basis import (
    # Types
    Block, BlockHeader, HexDigest, NormalizedFormula, DualAttestation,
    # Logic
    normalize, normalize_pretty, are_equivalent, atoms,
    # Crypto
    sha256_hex, hash_statement, hash_block, merkle_root,
    compute_merkle_proof, verify_merkle_proof,
    # Attestation
    reasoning_root, ui_root, composite_root,
    build_attestation, verify_attestation,
    # Ledger
    seal_block, block_to_dict, block_json,
    # Curriculum
    CurriculumLadder, ladder_from_dict, ladder_from_json,
)
```

---

## 4. Hash / Normalization / Attestation Single-Source Taxonomy

### 4.1 The Hash Identity Formula

$$
\mathrm{hash}(s) = \mathrm{SHA256}(\mathcal{D} \| \mathcal{E}(\mathcal{N}(s)))
$$

| Symbol | Name | Implementation | Location |
|--------|------|----------------|----------|
| $\mathcal{N}$ | Normalization | `normalize(expr)` | `basis/logic/normalizer.py:175` |
| $\mathcal{E}$ | Encoding | `str.encode("utf-8")` | `basis/crypto/hash.py:28-33` |
| $\mathcal{D}$ | Domain Tag | `DOMAIN_*` constants | `basis/crypto/hash.py:17-22` |

### 4.2 Domain Separation Tags (Canonical)

| Domain | Tag Byte | Hex | Usage |
|--------|----------|-----|-------|
| `DOMAIN_LEAF` | `0x00` | `\x00` | Merkle tree leaf nodes |
| `DOMAIN_NODE` | `0x01` | `\x01` | Merkle tree internal nodes |
| `DOMAIN_STMT` | `0x02` | `\x02` | Statement content identity |
| `DOMAIN_BLOCK` | `0x03` | `\x03` | Block header identity |
| `DOMAIN_REASONING_EMPTY` | `0x10` | `\x10` | Empty reasoning tree |
| `DOMAIN_UI_EMPTY` | `0x11` | `\x11` | Empty UI tree |

### 4.3 Attestation Root Computation

| Root | Symbol | Formula | Implementation |
|------|--------|---------|----------------|
| Reasoning | $R_t$ | `merkle_root(reasoning_leaves)` | `basis/crypto/hash.py:148-152` |
| UI | $U_t$ | `merkle_root(ui_leaves)` | `basis/crypto/hash.py:155-159` |
| Composite | $H_t$ | `SHA256(R_t \|\| U_t)` | `basis/attestation/dual.py` |

### 4.4 Normalization Rules (Canonical)

1. **Unicode → ASCII**: All Unicode logic symbols map to ASCII equivalents
2. **Implication**: Right-associative chaining (`p->q->r`)
3. **Conjunction/Disjunction**: Lexicographically sorted, deduplicated
4. **Whitespace**: Completely stripped in canonical form
5. **Parentheses**: Redundant outer parentheses removed

---

## 5. Import Graph: Canonical Paths and Deprecated Shims

> **Phase II**: This section describes the target import structure. Currently (Phase I),
> the "transitional" paths are the actual active paths used by all code.

### 5.1 Canonical Import Paths (Phase II Target)

```
┌────────────────────────────────────────────────────────────────────┐
│                     CANONICAL (Use These)                          │
├────────────────────────────────────────────────────────────────────┤
│ basis/                     → Primary entry point                   │
│ basis/logic/normalizer     → normalize, are_equivalent, atoms      │
│ basis/crypto/hash          → sha256_hex, merkle_root, hash_*       │
│ basis/attestation/dual     → composite_root, build_attestation     │
│ basis/ledger/block         → seal_block, block_to_dict             │
│ basis/curriculum/ladder    → CurriculumLadder                      │
├────────────────────────────────────────────────────────────────────┤
│ normalization/canon        → Active normalizer (transitional)      │
│ attestation/dual_root      → Active dual-root (transitional)       │
│ rfl/runner                 → RFLRunner (canonical)                 │
│ substrate/repro/determinism→ Deterministic primitives (canonical)  │
│ substrate/crypto/core      → RFC 8785 (transitional to basis)      │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 Deprecated Shims (Schedule for Removal)

| Deprecated Path | Redirects To | Removal Date |
|-----------------|--------------|--------------|
| `backend/logic/canon.py` | `normalization/canon` | 2025-12-01 |
| `backend/crypto/dual_root.py` | `attestation/dual_root` | 2025-12-01 |
| `backend/rfl/runner.py` | `rfl/runner` | 2025-12-01 |
| `backend/orchestrator/app.py` | `interface/api/app` | 2025-12-15 |

### 5.3 Transitional Modules (Active in Phase I)

These are the **actual active paths** used by Phase I code:

| Current Location (ACTIVE) | Target in basis/ | Migration Status |
|---------------------------|------------------|------------------|
| `normalization/canon.py` | `basis/logic/normalizer.py` | Files exist, **not migrated** (no consumers) |
| `attestation/dual_root.py` | `basis/attestation/dual.py` | Files exist, **not migrated** (no consumers) |
| `substrate/crypto/core.py` | `basis/crypto/hash.py` | **PARTIAL** — some functions duplicated |
| `backend/ledger/blockchain.py` | `basis/ledger/block.py` | Files exist, **not migrated** (no consumers) |

**Clarification**: "DONE" in previous version was misleading. The `basis/` files exist but nothing imports them. The transitional modules remain the active code paths.

---

## 6. Forbidden Imports for basis/ (Phase II Constraint)

> **Phase II**: This constraint is not yet enforced in CI. It is a design goal.

When `basis/` becomes the active code path, it MUST remain pure. The following imports will be **STRICTLY FORBIDDEN** within `basis/**/*.py`:

### 6.1 Forbidden Module Categories

```python
# FORBIDDEN IN basis/
import os                    # Filesystem I/O
import sys                   # Runtime state
import time                  # Wall-clock time
import datetime              # Wall-clock time (use deterministic_timestamp externally)
import random                # Entropy source
import uuid                  # Random UUIDs
import asyncio               # Async I/O
import threading             # Concurrency
import multiprocessing       # Concurrency
import socket                # Network I/O
import requests              # Network I/O
import httpx                 # Network I/O
import redis                 # External service
import psycopg               # Database
import sqlalchemy            # Database
```

### 6.2 Allowed Dependencies in basis/

```python
# ALLOWED IN basis/
import hashlib               # Cryptographic hashing (stdlib)
import json                  # Serialization (stdlib)
import re                    # Regex (stdlib)
import functools             # Caching (lru_cache)
from typing import *         # Type hints
from dataclasses import *    # Data structures
```

### 6.3 Enforcement Mechanism (Phase II — Not Yet Implemented)

```bash
# CI check: grep for forbidden imports in basis/
# NOTE: This check is NOT currently in CI. It is a Phase II goal.
grep -rE "^(import|from) (os|sys|time|datetime|random|uuid|asyncio|threading|redis|psycopg)" basis/
# Expected output: (empty)
```

---

## 7. First Organism Determinism Envelope

> **Phase I — This section describes actual tested behavior.**
> The determinism contract is enforced by `tests/integration/test_first_organism_determinism.py`.

The **Determinism Envelope** defines exactly which files may affect the composite attestation root $H_t$.

### 7.1 Files Allowed to Affect $H_t$ (Phase I Active Paths)

These files are **inside the envelope** and subject to determinism constraints.

**Note**: In Phase I, the active paths are `normalization/`, `attestation/`, etc. — not `basis/`.

| File | Role | Determinism Requirement | Phase I Active? |
|------|------|------------------------|-----------------|
| `normalization/canon.py` | Active normalizer | Pure function | ✅ YES |
| `attestation/dual_root.py` | Active dual-root | Pure function | ✅ YES |
| `substrate/repro/determinism.py` | Timestamp/UUID helpers | Seed-dependent only | ✅ YES |
| `backend/repro/determinism.py` | Timestamp/UUID helpers | Seed-dependent only | ✅ YES |
| `backend/axiom_engine/derive.py` | Derivation engine | Must use SeededRNG | ✅ YES |
| `rfl/runner.py` | RFL metabolism | Consumes $H_t$, produces entries | ✅ YES |
| `basis/logic/normalizer.py` | Formula normalization | Pure function, LRU cached | ❌ NOT USED |
| `basis/crypto/hash.py` | All hashing operations | Pure function, domain-tagged | ❌ NOT USED |
| `basis/attestation/dual.py` | $R_t$, $U_t$, $H_t$ computation | Pure function | ❌ NOT USED |

### 7.2 Files Outside the Envelope

These files MUST NOT affect $H_t$ computation:

| Category | Files | Reason |
|----------|-------|--------|
| **API/HTTP** | `backend/orchestrator/app.py` | Network timing |
| **Database** | `backend/models/*.py` | Execution order |
| **Redis** | `backend/worker.py` | Queue timing |
| **Metrics** | `backend/metrics/*.py` | Wall-clock latency |
| **Logging** | `rfl/experiment_logging.py` | Side effects |
| **Scripts** | `scripts/*.ps1` | Operational |

### 7.3 Forbidden Primitives in Envelope

Per `docs/DETERMINISM_CONTRACT.md`:

| Forbidden | Replacement |
|-----------|-------------|
| `datetime.now()` | `deterministic_timestamp(seed)` |
| `datetime.utcnow()` | `deterministic_timestamp(seed)` |
| `time.time()` | `deterministic_unix_timestamp(seed)` |
| `uuid.uuid4()` | `deterministic_uuid(content)` |
| `random.*` | `SeededRNG(seed)` |
| `os.urandom` | **Absolutely forbidden** |
| `dict` iteration | `sorted(d.items())` |

### 7.4 Determinism Verification Test

```python
# tests/integration/test_first_organism_determinism.py
def test_bitwise_determinism():
    """Two runs with same seed must produce identical H_t."""
    result1 = run_first_organism_deterministic(seed=42)
    result2 = run_first_organism_deterministic(seed=42)

    assert result1.composite_root == result2.composite_root
    assert result1.run_hash == result2.run_hash
```

---

## 8. Redline: What Must Move into basis/ (Phase II Roadmap)

> **Phase II**: This section describes migration work that has NOT been done.

### 8.1 Files in basis/ (Exist But Not Used)

The following files exist on disk but have **zero consumers**:

- [ ] `basis/logic/normalizer.py` — Exists, not imported
- [ ] `basis/crypto/hash.py` — Exists, not imported
- [ ] `basis/attestation/dual.py` — Exists, not imported
- [ ] `basis/ledger/block.py` — Exists, not imported
- [ ] `basis/curriculum/ladder.py` — Exists, not imported
- [ ] `basis/core/types.py` — Exists, not imported

**Phase II Work**: Update all consumers to import from `basis/` instead of transitional modules.

### 8.2 Must Migrate (Phase II — Not Started)

| Source | Target | Priority | Status |
|--------|--------|----------|--------|
| `substrate/crypto/core.py:rfc8785_canonicalize` | `basis/crypto/json.py` | P0 | NOT DONE |
| `attestation/dual_root.py:generate_attestation_metadata` | `basis/attestation/dual.py` | P1 | NOT DONE |
| Consumer imports → `basis/` | All files using normalization/attestation | P0 | NOT DONE |

### 8.3 Must Delete After Migration (Phase II — Blocked)

Cannot delete until consumers are migrated:

| File | Replacement | Status |
|------|-------------|--------|
| `backend/logic/canon.py` | `basis/logic/normalizer.py` | BLOCKED — shim still has consumers |
| `backend/crypto/dual_root.py` | `basis/attestation/dual.py` | BLOCKED — shim still has consumers |
| `backend/rfl/runner.py` | `rfl/runner.py` | BLOCKED — shim still has consumers |

---

## 9. Compliance Checklist for PR Merge (Phase II — Not Enforced)

> **Phase II**: These gates are not currently enforced in CI. This is a target checklist.

Every PR touching files in the Determinism Envelope SHOULD pass this checklist:

### 9.1 Pre-Merge Gate (Automated)

- [ ] **No Forbidden Imports**: `grep` check passes for `basis/`
- [ ] **Type Hints**: `mypy --strict basis/` passes
- [ ] **Tests Pass**: `pytest tests/test_canon.py tests/test_hash_canonization.py tests/test_dual_root_attestation.py`
- [ ] **Determinism Test**: `pytest tests/integration/test_first_organism_determinism.py`
- [ ] **No Wall-Clock**: `grep -r "datetime.now\|time.time\|uuid.uuid4" basis/ backend/repro/ attestation/` returns empty

### 9.2 Pre-Merge Gate (Manual Review)

- [ ] **Vibe Boundary**: Imports respect Law → Economy → Metabolism hierarchy
- [ ] **Single Source**: No duplicate implementations of normalize/hash/attest
- [ ] **Domain Tags**: Any new hash operation uses appropriate `DOMAIN_*` tag
- [ ] **Determinism Contract**: Any timestamp/UUID uses `substrate/repro/determinism` helpers

### 9.3 First Organism Smoke Test

```bash
# Must pass before any envelope file is merged
pytest tests/integration/test_first_organism.py -k closed_loop -v
pytest tests/integration/test_first_organism_determinism.py -v
```

---

## 10. Summary: Phase I Reality vs Phase II Target

### Phase I Reality (What Actually Runs Today)

| Aspect | Active Source | Evidence |
|--------|--------------|----------|
| **Normalization** | `normalization/canon.py` | Used by FO test |
| **Hashing** | `substrate/crypto/core.py` | Used by attestation |
| **Attestation** | `attestation/dual_root.py` | Used by FO test |
| **Timestamps** | `substrate/repro/determinism.py` | Used by FO harness |
| **UUIDs** | `substrate/repro/determinism.py` | Used by FO harness |
| **RFL Entry** | `rfl/runner.py` | Used by FO test |

### Phase II Target (After Decomposition)

| Aspect | Target Source | Status |
|--------|--------------|--------|
| **Normalization** | `basis/logic/normalizer.py` | NOT YET USED |
| **Hashing** | `basis/crypto/hash.py` | NOT YET USED |
| **Attestation** | `basis/attestation/dual.py` | NOT YET USED |
| **Timestamps** | `substrate/repro/determinism.py` | Already canonical |
| **UUIDs** | `substrate/repro/determinism.py` | Already canonical |
| **RFL Entry** | `rfl/runner.py` | Already canonical |

---

## 11. Appendix: ASCII Import Dependency Graph

```
                    ┌─────────────────────┐
                    │      basis/         │
                    │   (LAW LAYER)       │
                    │                     │
                    │  logic/normalizer   │
                    │  crypto/hash        │
                    │  attestation/dual   │
                    │  ledger/block       │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ normalization/   │ │ attestation/     │ │ substrate/       │
│ (Transitional)   │ │ (Transitional)   │ │ (ECONOMY)        │
│                  │ │                  │ │                  │
│ canon.py ────────┼─┼─►dual_root.py    │ │ repro/determinism│
│ (→basis/logic)   │ │  (→basis/attest) │ │ crypto/core      │
└──────────────────┘ └──────────────────┘ └────────┬─────────┘
                                                   │
                               ┌───────────────────┼───────────────────┐
                               │                   │                   │
                               ▼                   ▼                   ▼
                    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
                    │ backend/         │ │ rfl/             │ │ backend/         │
                    │ axiom_engine/    │ │ (METABOLISM)     │ │ frontier/        │
                    │ (ECONOMY)        │ │                  │ │ (METABOLISM)     │
                    │                  │ │ runner.py        │ │                  │
                    │ derive.py        │ │ config.py        │ │ curriculum.py    │
                    │ rules.py         │ │ bootstrap_stats  │ │ gates.py         │
                    └──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 12. Document Status

**This document describes PHASE II target architecture.**

It is NOT:
- Part of Evidence Pack v1
- A description of current working code
- Grounds for claiming `basis/` is the single source of truth

It IS:
- A roadmap for future decomposition work
- A design specification for code consolidation
- A reference for Phase II implementation

**Phase I Evidence Pack v1 relies on**:
- `tests/integration/test_first_organism.py` — The actual test
- `artifacts/first_organism/attestation.json` — The sealed artifact
- `normalization/`, `attestation/`, `rfl/` — The actual active code paths

*— Claude A, Vibe Orchestrator (Reviewer-2 Mode)*
