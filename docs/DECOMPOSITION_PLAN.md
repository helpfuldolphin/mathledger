# DECOMPOSITION PLAN: VCP 2.1 Micro-Task Architecture

**Author:** Claude B — Decomposition Architect
**Status:** DRAFT → **FROZEN (Phase I/II Classification Applied)**
**Version:** 1.1
**Parent:** VSD VCP 2.1, `ops/vcp21_plan_first_organism.md`
**Updated:** 2025-11-30

---

## ⚠️ PHASE I vs PHASE II CLASSIFICATION (CRITICAL)

> **Mode:** SOBER TRUTH / REVIEWER-2
> **Evidence Pack v1 Status:** PARTIALLY READY

### Phase I (Completed / In-Scope for Evidence Pack v1)

**Hard Evidence Files (Actually Exist and Are Sealed):**
- `artifacts/first_organism/attestation.json` — FO closed-loop attestation
- `results/fo_rfl_50.jsonl` — Canonical 50-cycle RFL run
- `artifacts/phase_ii/fo_series_1/fo_1000_baseline/experiment_log.jsonl` — 1000-cycle baseline
- `artifacts/figures/rfl_abstention_rate.png` — Abstention rate figure
- `artifacts/figures/rfl_dyno_chart.png` — Dyno chart (baseline vs RFL)

**Tracks with Phase I evidence:**
- Track A (Substrate & Normalization) — Partially complete, code exists
- Track C (Ledger & Dual Attestation) — Core functions complete, used in attestation.json
- Track G (Determinism Harness) — FO tests exist and pass

### Phase-I RFL Evidence Status

| File | Cycles | Status | What It Proves |
|------|--------|--------|----------------|
| `results/fo_rfl_50.jsonl` | 50 | **CANONICAL Phase I RFL** | RFL code path executes. All cycles abstain. |
| `results/fo_rfl.jsonl` | ~330 | Phase I (degenerate) | Extended run, all-abstain. Confirms loop mechanics. |

**CRITICAL CLARIFICATION:**
- Both files are **Phase I evidence**, NOT Phase II.
- Neither demonstrates **RFL uplift** (improved derivation due to RFL).
- These logs prove RFL code executes. They do NOT prove RFL improves anything.
- Any claim of "RFL uplift" requires Phase II experiments **not yet run**.

### Phase II (Deferred — NOT in Evidence Pack v1)

**The following are explicitly NOT IMPLEMENTED and must NOT be claimed:**
- ΔH scaling theory — *Not activated by current RFL logs.*
- Imperfect verifier theory — *Not activated by current RFL logs.*
- Wave-1 basis promotion — *Not activated by current RFL logs.*
- MDAP micro-agents — *Not activated by current RFL logs.*
- Lean sandbox isolation — *Not activated by current RFL logs.*
- Wide slice theory curves — *Not activated by current RFL logs.*
- Logistic abstention theory — *Not activated by current RFL logs.*

**Tracks that are Phase II:**
- Track D (RFL & Curriculum) — Wide slice theory is Phase II. *Not activated by current RFL logs.*
- Track E (API & Interface) — API versioning is Phase II. *Not activated by current RFL logs.*
- Track F (Security & Runtime) — Token rotation is Phase II. *Not activated by current RFL logs.*
- Most of Track B (Derivation & Lean) — Lean sandbox is Phase II. *Not activated by current RFL logs.*

**CONSTRAINT:** No Phase II task may implicitly depend on RFL uplift data from current logs.

### Phase II Uplift Task Interlock

When Phase II begins, uplift experiments will follow these rules:

| Rule | Description |
|------|-------------|
| **No Retroactive Reinterpretation** | Phase II tasks may NOT reinterpret Phase I logs (fo_rfl_50.jsonl, fo_rfl.jsonl) as uplift evidence. They must run NEW experiments. |
| **Governance Gate Required** | All uplift tasks require `VSD_PHASE_2_uplift_gate` approval before starting. |
| **Pre-registered Success Criteria** | UPLIFT-001 must define success criteria BEFORE UPLIFT-002 runs experiments. |
| **Null Results Are Valid** | If uplift is not demonstrated, that is a valid scientific finding to document. |
| **Phase I Remains Frozen** | Evidence Pack v1 is never modified. Phase II creates Evidence Pack v2 if warranted. |

**Phase II Uplift Tasks (defined in `DECOMPOSITION_PHASE_PLAN.json`):**

| ID | Title | Status | Depends On |
|----|-------|--------|------------|
| UPLIFT-001 | Design non-degenerate uplift slice | not_started | — |
| UPLIFT-002 | Run controlled uplift experiment | not_started | UPLIFT-001 |
| UPLIFT-003 | Analyze uplift results | not_started | UPLIFT-002 |
| UPLIFT-004 | Document findings (positive or null) | not_started | UPLIFT-003 |
| UPLIFT-005 | Integrate into Evidence Pack v2 (if positive) | not_started | UPLIFT-004 |

**Interlock with Existing Categories:**

| Category | Relationship to Uplift Tasks |
|----------|------------------------------|
| CURR-* | Curriculum slice design (UPLIFT-001) may inform these, but they remain Phase II until uplift demonstrated |
| RFL-004, RFL-006 | Become relevant only after UPLIFT-002 produces non-degenerate logs |
| DET-005 | Replay capability useful for reproducing uplift experiments |
| DOC-* | Uplift theory documentation deferred until UPLIFT-004 completes |
| SEC-*, TYPE-* | Independent of uplift — remain Phase II on their own timeline |

### See Also
- `ops/microtasks/PHASE_I_LIVE_TASKS.md` — Filtered list of Phase I tasks only
- `DECOMPOSITION_PHASE_PLAN.json` — Full 85-task DAG with phase classification + PHASE_II_RFL_UPLIFT_ROADMAP
- `docs/evidence/EVIDENCE_PACK_V1_AUDIT_CURSOR_O.md` — Evidence audit

---

## 1. Overview

This document specifies the decomposition of VCP 2.1 epics into **MDAP-compatible micro-tasks**. Each micro-task satisfies:

| Property | Requirement |
|----------|-------------|
| **Atomic** | Single edit, single check — completable in < 60 seconds |
| **Context-Minimized** | Only relevant file snippet + error context |
| **Externally Validated** | At least one automated oracle (test, type checker, linter) |

The micro-task queue is stored in `ops/microtasks/microtask_queue.jsonl`.

---

## 2. Micro-Task Taxonomy

The taxonomy categorizes tasks by their nature and validation requirements:

| Type Code | Category | Validation Methods |
|-----------|----------|-------------------|
| `HASH` | Hash Canonicalization Fixes | `pytest tests/test_hash_canonization.py`, `mypy` |
| `DUAL` | Dual-Root Attestation Wiring | `pytest tests/test_dual_root_attestation.py` |
| `CURR` | Curriculum Gate Edge-Cases | `pytest tests/frontier/test_curriculum_gates.py` |
| `RFL` | RFL Runner Contract Fixes | `pytest tests/rfl/`, `mypy` |
| `API` | API Schema Alignment | `pytest tests/integration/test_api_endpoints.py` |
| `SEC` | Security/Env Enforcement | `pytest tests/`, environment check scripts |
| `DEPR` | Deprecation Shim Removal | Import check, `pytest` |
| `TYPE` | Type Annotation Addition | `mypy --strict` |
| `NORM` | Normalization Logic Fixes | `pytest tests/test_canon.py` |
| `DET` | Determinism Enforcement | `pytest tests/test_determinism_*.py` |
| `DOC` | Docstring/Comment Addition | `pydocstyle`, manual review |

---

## 3. Track-by-Track Decomposition

### Track A: Substrate & Normalization (The Axiomatic Core)

**Epic:** Harden `backend/logic`, `backend/fol_eq`, `backend/axiom_engine` for mathematical precision.

**Current State Analysis:**
- `backend/logic/canon.py` → Deprecated shim, redirects to `normalization/canon.py`
- `normalization/canon.py` → Active normalizer with LRU cache
- `basis/logic/normalizer.py` → Duplicate implementation
- `basis/crypto/hash.py` → Domain-separated SHA-256 with proper prefixes

**Completion Criteria:**
- [ ] Single canonical normalization path (no duplicates)
- [ ] All shims removed or marked for removal
- [ ] 100% type coverage on public APIs
- [ ] RFC 8785 JSON serialization for all hashed artifacts

**Micro-Task Series:**

#### A.1: Unify Normalization Implementations
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| A1.1 | Add deprecation warning to `basis/logic/normalizer.py` | `basis/logic/normalizer.py` | 1-15 | `pytest -k normalize`, `python -c "from basis.logic.normalizer import normalize"` | P0 | - |
| A1.2 | Update `basis/logic/normalizer.py` to import from `normalization.canon` | `basis/logic/normalizer.py` | 1-200 | `pytest tests/test_canon.py` | P0 | A1.1 |
| A1.3 | Add `__all__` export list to `normalization/canon.py` | `normalization/canon.py` | 1-20 | `mypy normalization/canon.py` | P1 | - |
| A1.4 | Add type annotation to `_map_unicode` function | `normalization/canon.py` | 54-59 | `mypy --strict normalization/canon.py` | P2 | A1.3 |
| A1.5 | Add type annotation to `_strip_outer_parens` function | `normalization/canon.py` | 62-80 | `mypy --strict normalization/canon.py` | P2 | A1.3 |
| A1.6 | Add return type annotation to `_split_top` function | `normalization/canon.py` | 83-99 | `mypy --strict normalization/canon.py` | P2 | A1.3 |
| A1.7 | Add type annotation to `_flatten` function | `normalization/canon.py` | 102-108 | `mypy --strict normalization/canon.py` | P2 | A1.3 |

#### A.2: Hash Pipeline Hardening
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| A2.1 | Add RFC 8785 JSON canonicalizer import check | `basis/crypto/hash.py` | 1-20 | `python -c "from basis.crypto.hash import sha256_hex"` | P0 | - |
| A2.2 | Add docstring to `_ensure_bytes` function | `basis/crypto/hash.py` | 28-33 | `pydocstyle basis/crypto/hash.py` | P2 | - |
| A2.3 | Add `__all__` export list to `basis/crypto/hash.py` | `basis/crypto/hash.py` | 1-20 | `mypy basis/crypto/hash.py` | P1 | - |
| A2.4 | Verify domain tag uniqueness (no duplicates) | `basis/crypto/hash.py` | 15-23 | `pytest tests/test_hash_canonization.py` | P0 | - |
| A2.5 | Add type annotation to `merkle_root` leaves parameter | `basis/crypto/hash.py` | 66-86 | `mypy --strict basis/crypto/hash.py` | P2 | A2.3 |

#### A.3: Deprecation Shim Cleanup
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| A3.1 | Update removal date in `backend/logic/canon.py` deprecation warning | `backend/logic/canon.py` | 1-15 | `grep "2025-12" backend/logic/canon.py` | P1 | - |
| A3.2 | Add `# TODO(remove ...)` comment to `backend/logic/taut.py` if deprecated | `backend/logic/taut.py` | 1-10 | manual review | P2 | - |
| A3.3 | Verify `backend/logic/__init__.py` exports are minimal | `backend/logic/__init__.py` | * | `python -c "from backend.logic import *"` | P1 | - |

---

### Track B: Derivation & Lean (The Engine) — ⚠️ PARTIALLY PHASE II

> **WARNING:** `LeanInterface` isolation and Lean sandbox are **Phase II**.
> Phase I uses truth-table verification only (hermetic mode, no Lean kernel).
> Seed management (B1.x) is Phase I. Lean isolation (B2.x) is Phase II.

**Epic:** Isolate `LeanInterface`, enforce explicit seeds, implement Proof-or-Abstain.

**Current State Analysis:**
- `backend/axiom_engine/derive.py` → Large monolithic file (637 lines)
- Uses `_GLOBAL_SEED = 0` for determinism
- Contains inline `load_axioms`, `load_derived_statements`, `upsert_statement`, `enqueue_job` methods
- Imports from `backend.repro.determinism` for timestamp generation

**Completion Criteria:**
- [ ] All random operations use explicit seed parameters
- [ ] `LeanInterface` is a standalone pure wrapper
- [ ] All verification functions return typed results or raise exceptions

**Micro-Task Series:**

#### B.1: Extract Seed Management
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| B1.1 | Add docstring to `_GLOBAL_SEED` explaining determinism contract | `backend/axiom_engine/derive.py` | 23-24 | `pydocstyle` | P1 | - |
| B1.2 | Add type annotation `_GLOBAL_SEED: int = 0` | `backend/axiom_engine/derive.py` | 23 | `mypy backend/axiom_engine/derive.py` | P1 | - |
| B1.3 | Add `seed` parameter to `_run_smoke_pl` function signature | `backend/axiom_engine/derive.py` | 297 | `pytest tests/test_derive.py` | P0 | - |
| B1.4 | Replace `_GLOBAL_SEED` usage with passed `seed` in `deterministic_timestamp` calls | `backend/axiom_engine/derive.py` | 137-140 | `pytest tests/test_derive.py` | P0 | B1.3 |
| B1.5 | Add seed parameter to `_upsert_statement` function | `backend/axiom_engine/derive.py` | 88 | `mypy`, `pytest` | P1 | B1.3 |
| B1.6 | Add seed parameter to `_insert_proof` function | `backend/axiom_engine/derive.py` | 155 | `mypy`, `pytest` | P1 | B1.3 |

#### B.2: Lean Interface Isolation
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| B2.1 | Create `backend/axiom_engine/lean_interface.py` stub file | (new file) | - | `python -c "import backend.axiom_engine.lean_interface"` | P0 | - |
| B2.2 | Add `LeanInterface` class with `__init__` accepting project dir | `backend/axiom_engine/lean_interface.py` | 1-30 | `mypy` | P0 | B2.1 |
| B2.3 | Add `verify(statement: str) -> VerificationResult` method | `backend/axiom_engine/lean_interface.py` | 30-60 | `mypy` | P0 | B2.2 |
| B2.4 | Add `VerificationResult` dataclass with `success`, `output`, `error` fields | `backend/axiom_engine/lean_interface.py` | 10-25 | `mypy` | P0 | B2.1 |
| B2.5 | Add type annotation to return type of `verify` method | `backend/axiom_engine/lean_interface.py` | 30 | `mypy --strict` | P1 | B2.3 |

#### B.3: Proof-or-Abstain Pattern
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| B3.1 | Add `VerificationError` exception class to `backend/axiom_engine/__init__.py` | `backend/axiom_engine/__init__.py` | * | `python -c "from backend.axiom_engine import VerificationError"` | P0 | - |
| B3.2 | Add `AbstractionError` exception class for explicit abstentions | `backend/axiom_engine/__init__.py` | * | `python -c "from backend.axiom_engine import AbstractionError"` | P0 | - |
| B3.3 | Replace `return False` with `raise VerificationError` in `_is_tauto_with_timeout` | `backend/axiom_engine/derive.py` | 330-347 | `pytest tests/test_derive.py` | P0 | B3.1 |
| B3.4 | Add try/except wrapper in caller to handle `VerificationError` | `backend/axiom_engine/derive.py` | 520 | `pytest tests/test_derive.py` | P0 | B3.3 |

---

### Track C: Ledger & Dual Attestation (The Memory)

**Epic:** Implement Whitepaper V2 block structures, wire dual attestation.

**Current State Analysis:**
- `basis/attestation/dual.py` → Clean implementation with `build_attestation`, `verify_attestation`
- `backend/crypto/dual_root.py` → Deprecated shim to `attestation.dual_root`
- `attestation/dual_root.py` → Another implementation with `AttestationLeaf`, `AttestationTree`
- `backend/ledger/blockchain.py` → Block construction
- `backend/ledger/blocking.py` → Block sealing

**Completion Criteria:**
- [ ] Single attestation implementation (no duplicates)
- [ ] `BlockHeader` and `ProofRecord` Pydantic schemas match Whitepaper V2
- [ ] All Merkle operations use `basis/crypto/hash.py` canonicalizer

**Micro-Task Series:**

#### C.1: Attestation Unification
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| C1.1 | Add deprecation warning to `attestation/dual_root.py` redirecting to `basis/attestation/dual.py` | `attestation/dual_root.py` | 1-15 | `python -c "from attestation.dual_root import compute_composite_root"` | P0 | - |
| C1.2 | Update `backend/crypto/dual_root.py` to import from `basis/attestation/dual` | `backend/crypto/dual_root.py` | 10-30 | `pytest tests/test_dual_root_attestation.py` | P0 | C1.1 |
| C1.3 | Add `__all__` export list to `basis/attestation/dual.py` | `basis/attestation/dual.py` | 1-15 | `mypy basis/attestation/dual.py` | P1 | - |
| C1.4 | Add type annotation to `composite_root` parameters | `basis/attestation/dual.py` | 25-32 | `mypy --strict basis/attestation/dual.py` | P2 | C1.3 |

#### C.2: Block Schema Alignment
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| C2.1 | Verify `basis/ledger/block.py` exports `Block` dataclass | `basis/ledger/block.py` | * | `python -c "from basis.ledger.block import Block"` | P0 | - |
| C2.2 | Add `reasoning_merkle_root` field to Block if missing | `basis/ledger/block.py` | * | `mypy basis/ledger/block.py` | P0 | C2.1 |
| C2.3 | Add `ui_merkle_root` field to Block if missing | `basis/ledger/block.py` | * | `mypy basis/ledger/block.py` | P0 | C2.1 |
| C2.4 | Add `composite_attestation_root` field to Block | `basis/ledger/block.py` | * | `mypy basis/ledger/block.py` | P0 | C2.1 |
| C2.5 | Add `attestation_metadata` field to Block | `basis/ledger/block.py` | * | `mypy basis/ledger/block.py` | P1 | C2.1 |

#### C.3: Merkle Canonicalization
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| C3.1 | Verify `backend/ledger/blocking.py` uses `basis/crypto/hash.merkle_root` | `backend/ledger/blocking.py` | * | `grep "merkle_root" backend/ledger/blocking.py` | P0 | - |
| C3.2 | Replace any inline Merkle computation with `basis/crypto/hash.merkle_root` | `backend/ledger/blocking.py` | * | `pytest tests/test_first_organism_ledger.py` | P0 | C3.1 |
| C3.3 | Add test asserting Merkle root determinism across runs | `tests/test_first_organism_ledger.py` | * | `pytest tests/test_first_organism_ledger.py` | P1 | C3.2 |

---

### Track D: RFL & Curriculum (The Brain) — ⚠️ PHASE II DEFERRED

> **WARNING:** Wide slice theory, curriculum state machine, and RFL scaling are **Phase II**.
> The only Phase I evidence is the 50-cycle RFL run (`results/fo_rfl_50.jsonl`).
> Do NOT claim curriculum advancement or wide slice testing as completed.

**Epic:** Formalize reflexive forgetting, extract curriculum state machine.

**Current State Analysis:**
- `backend/rfl/runner.py` → Deprecated shim to `rfl/runner.py`
- `rfl/runner.py` → Active RFL runner with `RFLRunner`, `RflResult`
- `backend/frontier/curriculum.py` → Full curriculum implementation (996 lines)
- `basis/curriculum/ladder.py` → Another curriculum abstraction

**Completion Criteria:**
- [ ] `ReflexiveForgetting` as standalone pure function
- [ ] `Curriculum` state machine with explicit slice transitions
- [ ] All RFL operations pass through `rfl/` module

**Micro-Task Series:**

#### D.1: RFL Shim Cleanup
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| D1.1 | Verify `backend/rfl/runner.py` deprecation warning is present | `backend/rfl/runner.py` | 1-20 | `python -c "from backend.rfl.runner import RFLRunner"` 2>&1 | P0 | - |
| D1.2 | Add removal date `TODO(remove after 2025-12-01)` to deprecation comment | `backend/rfl/runner.py` | 1-5 | `grep "2025-12" backend/rfl/runner.py` | P1 | - |
| D1.3 | Update `backend/rfl/__init__.py` to re-export from `rfl/` | `backend/rfl/__init__.py` | * | `pytest tests/rfl/` | P0 | D1.1 |

#### D.2: Curriculum State Machine
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| D2.1 | Add `__all__` export list to `backend/frontier/curriculum.py` | `backend/frontier/curriculum.py` | 1-20 | `mypy backend/frontier/curriculum.py` | P1 | - |
| D2.2 | Add type annotation to `_to_float` return type | `backend/frontier/curriculum.py` | 138-144 | `mypy --strict` | P2 | D2.1 |
| D2.3 | Add type annotation to `_to_int` return type | `backend/frontier/curriculum.py` | 147-153 | `mypy --strict` | P2 | D2.1 |
| D2.4 | Add type annotation to `_first_available` return type | `backend/frontier/curriculum.py` | 156-167 | `mypy --strict` | P2 | D2.1 |
| D2.5 | Verify `CurriculumSlice.from_dict` handles missing fields gracefully | `backend/frontier/curriculum.py` | 267-281 | `pytest tests/frontier/test_curriculum_gates.py` | P0 | - |
| D2.6 | Add validation for `monotonic_axes` in `CurriculumSystem._validate_monotonicity` | `backend/frontier/curriculum.py` | 342-361 | `pytest tests/frontier/test_curriculum_gates.py` | P0 | - |

#### D.3: RFL Bootstrap Stats
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| D3.1 | Add docstring to `rfl/bootstrap_stats.py` module | `rfl/bootstrap_stats.py` | 1-15 | `pydocstyle rfl/bootstrap_stats.py` | P2 | - |
| D3.2 | Add type annotations to bootstrap CI computation function | `rfl/bootstrap_stats.py` | * | `mypy --strict rfl/bootstrap_stats.py` | P1 | - |
| D3.3 | Add test for bootstrap CI with edge case (empty sample) | `tests/rfl/test_bootstrap_stats.py` | * | `pytest tests/rfl/test_bootstrap_stats.py` | P0 | - |

---

### Track E: API & Interface (The Skin) — ⚠️ PHASE II DEFERRED

> **WARNING:** API versioning, v1→v2 translation, and deprecation headers are **Phase II**.
> Phase I only requires the basic endpoints that support attestation retrieval.

**Epic:** Create v2 API schemas, implement `APIShim` for v1→v2 translation.

**Current State Analysis:**
- `backend/orchestrator/app.py` → Deprecated shim to `interface/api/app.py`
- Contains inline FastAPI app with middleware, routes
- Heavy schema-tolerant logic for DB column detection
- Pydantic schemas in `backend/api/schemas.py`

**Completion Criteria:**
- [ ] Strict v2 API schemas with versioning
- [ ] All v1 endpoints have deprecation headers
- [ ] Input validation via Pydantic on all endpoints

**Micro-Task Series:**

#### E.1: Schema Versioning
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| E1.1 | Verify `backend/api/schemas.py` exports all response models | `backend/api/schemas.py` | * | `python -c "from backend.api.schemas import MetricsResponse"` | P0 | - |
| E1.2 | Add `api_version: str = "v1"` field to `HealthResponse` | `backend/api/schemas.py` | * | `mypy backend/api/schemas.py` | P1 | E1.1 |
| E1.3 | Add `api_version: str = "v1"` field to `MetricsResponse` | `backend/api/schemas.py` | * | `mypy backend/api/schemas.py` | P1 | E1.1 |
| E1.4 | Add `api_version: str = "v1"` field to `BlockLatestResponse` | `backend/api/schemas.py` | * | `mypy backend/api/schemas.py` | P1 | E1.1 |

#### E.2: Input Validation Hardening
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| E2.1 | Add `Field(min_length=64, max_length=64)` to hash parameter validation | `backend/orchestrator/app.py` | 1127-1134 | `pytest tests/integration/test_api_endpoints.py` | P0 | - |
| E2.2 | Add regex validation `Field(regex=r'^[a-f0-9]{64}$')` to hash parameters | `backend/orchestrator/app.py` | * | `pytest tests/integration/test_api_endpoints.py` | P0 | E2.1 |
| E2.3 | Add `Body(..., embed=True)` to `record_ui_event` endpoint | `backend/orchestrator/app.py` | 822-835 | `pytest tests/integration/test_attestation_api.py` | P1 | - |

#### E.3: Deprecation Headers
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| E3.1 | Add `Deprecation` header to `/ui` endpoint response | `backend/orchestrator/app.py` | 544-593 | manual review | P2 | - |
| E3.2 | Add `Sunset` header with date to deprecated endpoints | `backend/orchestrator/app.py` | * | manual review | P2 | E3.1 |

---

### Track F: Security & Runtime (The Immune System) — ⚠️ PHASE II DEFERRED

> **WARNING:** Token rotation, Ed25519 signing, and worker isolation are **Phase II**.
> Phase I security is limited to basic API key authentication already in place.

**Epic:** Harden input validation, ensure worker isolation.

**Current State Analysis:**
- `backend/orchestrator/app.py` has `RequestSizeLimiter`, `FixedWindowRateLimiter` middleware
- `require_api_key` dependency for authentication
- `backend/security/runtime_env.py` for environment variable management
- Worker in `backend/worker.py`

**Completion Criteria:**
- [ ] All external inputs validated via Pydantic
- [ ] Worker crashes cannot corrupt ledger state
- [ ] Environment variables validated at startup

**Micro-Task Series:**

#### F.1: Input Validation Middleware
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| F1.1 | Add type annotation to `RequestSizeLimiter.__init__` | `backend/orchestrator/app.py` | 78-80 | `mypy backend/orchestrator/app.py` | P1 | - |
| F1.2 | Add type annotation to `FixedWindowRateLimiter.__init__` | `backend/orchestrator/app.py` | 105-109 | `mypy backend/orchestrator/app.py` | P1 | - |
| F1.3 | Add `_buckets` cleanup task to prevent memory leak | `backend/orchestrator/app.py` | 110 | `pytest tests/integration/test_api_endpoints.py` | P0 | - |

#### F.2: Environment Validation
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| F2.1 | Add `validate_required_env_vars()` function to `backend/security/runtime_env.py` | `backend/security/runtime_env.py` | * | `pytest tests/` | P0 | - |
| F2.2 | Call `validate_required_env_vars()` in app lifespan startup | `backend/orchestrator/app.py` | 911-924 | `pytest tests/integration/test_api_endpoints.py` | P0 | F2.1 |
| F2.3 | Add environment variable documentation to docstring | `backend/security/runtime_env.py` | 1-20 | `pydocstyle` | P2 | - |

#### F.3: Worker Isolation
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| F3.1 | Add `try/except` wrapper around main worker loop | `backend/worker.py` | * | `pytest tests/test_worker_fallback.py` | P0 | - |
| F3.2 | Add `WorkerError` exception class | `backend/worker.py` | * | `python -c "from backend.worker import WorkerError"` | P1 | - |
| F3.3 | Ensure worker writes to separate DB transaction | `backend/worker.py` | * | `pytest tests/test_worker_fallback.py` | P0 | F3.1 |

---

### Track G: Determinism Harness (The Test)

**Epic:** Create `FirstOrganismHarness`, verify bit-perfect replay.

**Current State Analysis:**
- `tests/integration/test_first_organism.py` → Main integration test
- `tests/integration/test_first_organism_determinism.py` → Determinism tests
- `tests/test_determinism_first_organism.py` → Unit-level determinism tests
- `backend/repro/determinism.py` → Deterministic timestamp utilities

**Completion Criteria:**
- [ ] `FirstOrganismHarness` runs B-D-C loop in sandbox
- [ ] Two runs produce SHA-256 identical outputs
- [ ] Harness completes in < 30s

**Micro-Task Series:**

#### G.1: Harness Infrastructure
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| G1.1 | Create `tests/integration/harness_v2.py` file | (new file) | - | `python -c "import tests.integration.harness_v2"` | P0 | - |
| G1.2 | Add `FirstOrganismHarness` class with `__init__(seed: int)` | `tests/integration/harness_v2.py` | 1-30 | `mypy tests/integration/harness_v2.py` | P0 | G1.1 |
| G1.3 | Add `run() -> HarnessResult` method | `tests/integration/harness_v2.py` | 30-60 | `mypy tests/integration/harness_v2.py` | P0 | G1.2 |
| G1.4 | Add `HarnessResult` dataclass with `artifacts_hash`, `duration_seconds` | `tests/integration/harness_v2.py` | 10-25 | `mypy tests/integration/harness_v2.py` | P0 | G1.1 |
| G1.5 | Add `derive()` method calling derivation engine | `tests/integration/harness_v2.py` | 60-90 | `pytest tests/integration/harness_v2.py` | P0 | G1.3 |
| G1.6 | Add `verify()` method calling Lean (or mock) | `tests/integration/harness_v2.py` | 90-120 | `pytest tests/integration/harness_v2.py` | P0 | G1.3 |
| G1.7 | Add `attest()` method calling attestation | `tests/integration/harness_v2.py` | 120-150 | `pytest tests/integration/harness_v2.py` | P0 | G1.3 |

#### G.2: Determinism Verification
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| G2.1 | Add `test_harness_determinism` test function | `tests/integration/harness_v2.py` | 150-180 | `pytest tests/integration/harness_v2.py::test_harness_determinism` | P0 | G1.7 |
| G2.2 | Assert SHA-256 of artifacts matches between two runs | `tests/integration/harness_v2.py` | 170-180 | `pytest tests/integration/harness_v2.py::test_harness_determinism` | P0 | G2.1 |
| G2.3 | Add timeout assertion `< 30s` | `tests/integration/harness_v2.py` | 175 | `pytest tests/integration/harness_v2.py::test_harness_determinism` | P1 | G2.1 |

#### G.3: Existing Test Alignment
| ID | Description | File | Range | Validator | Priority | Deps |
|----|-------------|------|-------|-----------|----------|------|
| G3.1 | Verify `test_first_organism_determinism.py` uses deterministic timestamps | `tests/integration/test_first_organism_determinism.py` | * | `pytest tests/integration/test_first_organism_determinism.py` | P0 | - |
| G3.2 | Add seed parameter to `test_first_organism_closed_loop_happy_path` | `tests/integration/test_first_organism.py` | * | `pytest tests/integration/test_first_organism.py` | P0 | - |
| G3.3 | Add assertion for Merkle root reproducibility | `tests/integration/test_first_organism.py` | * | `pytest tests/integration/test_first_organism.py` | P0 | G3.2 |

---

## 4. Priority Legend

| Priority | Meaning | Target Completion |
|----------|---------|-------------------|
| P0 | Blocking for First Organism | Immediate |
| P1 | Required for VCP 2.1 release | Within sprint |
| P2 | Nice-to-have, technical debt | Backlog |

---

## 5. Validation Matrix

| Validator | Command | Description |
|-----------|---------|-------------|
| `pytest` | `pytest tests/` | Full test suite |
| `pytest -k <pattern>` | `pytest -k normalize` | Filtered tests |
| `mypy` | `mypy --strict <module>` | Type checking |
| `mypy` (loose) | `mypy <module>` | Basic type checking |
| `pydocstyle` | `pydocstyle <module>` | Docstring style |
| `grep` | `grep "pattern" file` | Text search |
| `python -c` | `python -c "import ..."` | Import validation |

---

## 6. Dependency Graph

```
Track A (Substrate)
    ├── A1.1 → A1.2 → A1.3 → A1.4-A1.7 (parallel)
    ├── A2.1 → A2.2-A2.5 (parallel)
    └── A3.1-A3.3 (parallel)

Track B (Derivation)
    ├── B1.1 → B1.2 → B1.3 → B1.4-B1.6 (parallel)
    ├── B2.1 → B2.2 → B2.3-B2.5 (parallel)
    └── B3.1 → B3.2 → B3.3 → B3.4

Track C (Ledger) [depends on A2]
    ├── C1.1 → C1.2 → C1.3 → C1.4
    ├── C2.1 → C2.2-C2.5 (parallel)
    └── C3.1 → C3.2 → C3.3

Track D (RFL)
    ├── D1.1 → D1.2 → D1.3
    ├── D2.1 → D2.2-D2.6 (parallel)
    └── D3.1 → D3.2 → D3.3

Track E (API) [depends on A, C]
    ├── E1.1 → E1.2-E1.4 (parallel)
    ├── E2.1 → E2.2 → E2.3
    └── E3.1 → E3.2

Track F (Security) [depends on E]
    ├── F1.1-F1.3 (parallel)
    ├── F2.1 → F2.2 → F2.3
    └── F3.1 → F3.2 → F3.3

Track G (Determinism) [depends on A, B, C, D]
    ├── G1.1 → G1.2 → G1.3 → G1.4 → G1.5-G1.7 (parallel)
    ├── G2.1 → G2.2 → G2.3
    └── G3.1-G3.3 (parallel)
```

---

## 7. Global Reviewer Checkpoints

1. **Post-Track A:** Hash functions consistent? `pytest tests/test_hash_canonization.py`
2. **Post-Track C:** Dual attestation correct? `pytest tests/test_dual_root_attestation.py`
3. **Post-Track G:** Harness produces identical artifacts? `pytest tests/integration/harness_v2.py::test_harness_determinism`

---

## 8. Micro-Task Queue Location

All micro-tasks are serialized to:
```
ops/microtasks/microtask_queue.jsonl
```

Each line follows the schema:
```json
{
  "id": "A1.1",
  "type": "DEPR",
  "scope": {"file": "basis/logic/normalizer.py", "range": "1-15"},
  "validator": ["pytest -k normalize", "python -c \"from basis.logic.normalizer import normalize\""],
  "priority": "P0",
  "dependencies": []
}
```
