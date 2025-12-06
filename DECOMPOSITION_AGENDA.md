# DECOMPOSITION_AGENDA.md — Epic-Level Workstreams for MathLedger

**Version:** 1.0
**Status:** Active
**Protocol:** VCP 2.1 / First Organism
**Author:** Claude A (Vibe Orchestrator & Intent Compiler)

---

## Overview

This document decomposes the MathLedger refactoring into **epic-level workstreams**, each annotated with scope, risks, impact on First Organism, and recommended agent assignments.

**Priority Framework:**
- **P0**: Blocks First Organism — must be fixed immediately
- **P1**: First Organism path — directly affects loop integrity
- **P2**: Slop reduction — improves maintainability
- **P3**: Future-proofing — nice to have, defer

---

## Epic 1: Consolidate Hashing to Single Source of Truth

**Priority:** P0
**Scope:** Eliminate 4+ duplicate hash implementations, establish `basis/crypto/hash.py` as canonical

### Current State (Problem)

| Module | Location | Domain Sep | Status |
|--------|----------|-----------|--------|
| `basis/crypto/hash.py` | Lines 36-159 | ✓ | **CANONICAL** |
| `backend/crypto/hashing.py` | Lines 34-133 | ✓ | DUPLICATE |
| `backend/crypto/core.py` | Lines 86-156 | ✓ | DUPLICATE |
| `substrate/crypto/hashing.py` | Lines 36-124 | ✓ | DUPLICATE |
| `tools/verify_merkle.py` | Lines 27-46 | ✗ | **DIVERGENT** |

### Target State

1. `basis/crypto/hash.py` is the **only** SHA-256 implementation
2. All other modules re-export from basis
3. `tools/verify_merkle.py` fixed to use domain separation

### Risks

- **HIGH**: Changing hash computation breaks all existing statement hashes
- **MITIGATION**: Only fix `tools/verify_merkle.py` (verification tool); production code already uses domain separation

### Impact on First Organism

- **DIRECT**: Attestation recomputation depends on consistent hashing
- **BLOCKING**: If `H_t` computation differs between modules, First Organism fails

### Micro-Tasks

1. **Audit**: Grep all `hashlib.sha256` calls, verify domain tag presence
2. **Fix**: Update `tools/verify_merkle.py` to use `DOMAIN_LEAF`/`DOMAIN_NODE`
3. **Shim**: Make `backend/crypto/hashing.py` re-export from `basis/crypto/hash.py`
4. **Shim**: Make `backend/crypto/core.py` re-export from `basis/crypto/hash.py`
5. **Archive**: Move `substrate/crypto/` to `archive/substrate/crypto/`
6. **Test**: Run `test_first_organism.py`, verify `H_t` recomputation

### Agent Assignment

- **Claude B (Substrate Architect)**: Design shim layer
- **Claude C (Code Surgeon)**: Execute shim changes
- **Claude D (Test Guardian)**: Verify no hash regression

---

## Epic 2: Remove Nondeterministic Timestamps from First Organism Path

**Priority:** P0
**Scope:** Ensure all timestamps in attestation path use seeded deterministic generation

### Current State (Problem)

Potential `datetime.now()` or `time.time()` calls in:
- `attestation/dual_root.py` — metadata timestamps
- `ledger/blocking.py` — block timestamps
- `backend/rfl/runner.py` — run timestamps

### Target State

All timestamps in First Organism path use:
```python
from backend.repro.determinism import deterministic_timestamp
ts = deterministic_timestamp(seed=GLOBAL_SEED)
```

### Risks

- **MEDIUM**: May affect existing tests that assert on specific timestamps
- **MITIGATION**: Update tests to use deterministic seeds

### Impact on First Organism

- **DIRECT**: Nondeterministic timestamps make `H_t` unreproducible
- **BLOCKING**: Integration test cannot verify determinism

### Micro-Tasks

1. **Audit**: Grep for `datetime.now`, `time.time`, `time.monotonic` in First Organism path
2. **Identify**: Map each timestamp source to its callsite
3. **Replace**: Inject deterministic timestamp generator
4. **Test**: Verify `H_t` is identical across multiple runs

### Agent Assignment

- **Claude E (Determinism Enforcer)**: Audit and replace timestamps
- **Claude D (Test Guardian)**: Verify determinism contract

---

## Epic 3: Collapse Dual-Root Logic into Canonical Location

**Priority:** P1
**Scope:** Establish `attestation/dual_root.py` + `basis/attestation/dual.py` as canonical

### Current State (Problem)

| Module | Purpose | Status |
|--------|---------|--------|
| `attestation/dual_root.py` | Full RFC 8785 + Merkle + composite | **CANONICAL** |
| `basis/attestation/dual.py` | Thin wrapper around basis/crypto | **CANONICAL WRAPPER** |
| `backend/crypto/dual_root.py` | Stub implementation | DEPRECATED |
| `backend/phase_ix/attestation.py` | Experimental variant | EXPERIMENTAL |

### Target State

1. `attestation/dual_root.py` is the **only** implementation
2. `basis/attestation/dual.py` re-exports from `attestation/dual_root.py`
3. `backend/crypto/dual_root.py` → deprecation shim
4. `backend/phase_ix/attestation.py` → `archive/phase_ix/`

### Risks

- **LOW**: Mostly consolidation, no behavior change
- **MITIGATION**: Comprehensive test coverage before refactoring

### Impact on First Organism

- **DIRECT**: First Organism calls `compute_composite_root()`
- **ENABLING**: Cleaner import paths reduce confusion

### Micro-Tasks

1. **Audit**: Identify all imports of dual-root functions
2. **Shim**: `backend/crypto/dual_root.py` → re-export from `attestation/`
3. **Archive**: Move `backend/phase_ix/` to `archive/phase_ix/`
4. **Update**: Fix all import paths in `backend/ledger/blocking.py`
5. **Test**: First Organism still passes

### Agent Assignment

- **Claude B (Substrate Architect)**: Design import hierarchy
- **Claude C (Code Surgeon)**: Execute consolidation
- **Claude F (Archivist)**: Move deprecated code to archive

---

## Epic 4: Archive Unused Consensus and Federation Code

**Priority:** P2
**Scope:** Move unreachable experimental code to `archive/`

### Current State (Problem)

| Module | Purpose | Used by First Organism? |
|--------|---------|------------------------|
| `backend/ledger/consensus/harmony_v1_1.py` | Consensus variant | NO |
| `backend/ledger/consensus/celestial_dossier_v2.py` | Consensus variant | NO |
| `backend/ledger/v4/interfederation.py` | Federation protocol | NO |
| `backend/ledger/v4/stellar.py` | Stellar integration | NO |
| `backend/governance/` | Governance/replay | NO |
| `backend/consensus/harmony.py` | Single consensus impl | NO |

### Target State

1. All unreachable modules moved to `archive/`
2. Clear separation between live code and research

### Risks

- **LOW**: Archiving does not change runtime behavior
- **MITIGATION**: Preserve git history with move, not delete

### Impact on First Organism

- **NONE**: These modules are unreachable from First Organism path

### Micro-Tasks

1. **Audit**: Confirm no imports from First Organism path
2. **Create**: `archive/consensus_variants/`
3. **Move**: `backend/ledger/consensus/` → `archive/consensus_variants/`
4. **Move**: `backend/ledger/v4/` → `archive/federation_v4/`
5. **Move**: `backend/governance/` → `archive/governance/`
6. **Test**: First Organism still passes

### Agent Assignment

- **Claude F (Archivist)**: Execute all moves
- **Claude D (Test Guardian)**: Verify no import breakage

---

## Epic 5: Archive Root-Level Experimental Files

**Priority:** P2
**Scope:** Move 21 root-level Python files to `archive/experimental/`

### Current State (Problem)

Root directory contains experimental scripts:
```
bootstrap_metabolism.py  (13KB)
bridge.py                (3KB)
phase_ix_attestation.py  (10KB)
rfl_gate.py              (20KB)
test_dual_attestation.py (1KB)
test_integration_v05.py  (14KB)
test_integrity_audit.py  (13KB)
test_migration_validation.py (6KB)
test_v05_integration.py  (16KB)
verify_dual_root.py      (9KB)
verify_local_schema.py   (6KB)
...
```

### Target State

1. Root directory contains only:
   - `pyproject.toml`, `pytest.ini`, `Makefile`, `README.md`, `CLAUDE.md`
   - Canonical module directories
2. All experimental scripts in `archive/experimental/`

### Risks

- **MEDIUM**: Some scripts may be referenced by CI or documentation
- **MITIGATION**: Grep for references before moving

### Impact on First Organism

- **NONE**: These are standalone scripts, not imported

### Micro-Tasks

1. **Audit**: Grep for imports/references to root-level scripts
2. **Create**: `archive/experimental/`
3. **Move**: All experimental `.py` files to archive
4. **Update**: Any documentation references
5. **Test**: CI still passes

### Agent Assignment

- **Claude F (Archivist)**: Execute moves
- **Claude G (Docs Maintainer)**: Update references

---

## Epic 6: Archive substrate/ Parallel Implementation

**Priority:** P2
**Scope:** Move entire `substrate/` tree to `archive/`

### Current State (Problem)

`substrate/` is a parallel implementation of core machinery:
```
substrate/
├── crypto/hashing.py     # Duplicate of backend/crypto
├── crypto/core.py        # Duplicate
├── security/runtime_env.py
├── repro/determinism.py
├── lean/
├── logic/
└── bridge/
```

### Target State

1. `substrate/` → `archive/substrate/`
2. No imports from `substrate/` in live code

### Risks

- **MEDIUM**: Some code may import from substrate
- **MITIGATION**: Grep for `from substrate` imports first

### Impact on First Organism

- **UNKNOWN**: Need to audit before moving

### Micro-Tasks

1. **Audit**: Grep for `from substrate` or `import substrate`
2. **Decide**: If imports exist, create shim layer first
3. **Move**: `substrate/` → `archive/substrate/`
4. **Test**: First Organism still passes

### Agent Assignment

- **Claude B (Substrate Architect)**: Audit imports
- **Claude F (Archivist)**: Execute move

---

## Epic 7: Normalize Logic Module Imports

**Priority:** P1
**Scope:** Establish `normalization/` as canonical, deprecate `backend/logic/`

### Current State (Problem)

| Module | Status |
|--------|--------|
| `normalization/canon.py` | **CANONICAL** |
| `normalization/taut.py` | **CANONICAL** |
| `normalization/truthtab.py` | **CANONICAL** |
| `backend/logic/canon.py` | DEPRECATED SHIM |
| `backend/logic/taut.py` | DEPRECATED SHIM |
| `backend/logic/truthtab.py` | DEPRECATED SHIM |
| `basis/logic/normalizer.py` | ALTERNATIVE (review needed) |

### Target State

1. All imports use `from normalization.canon import ...`
2. `backend/logic/*.py` are thin shims with deprecation warnings
3. `basis/logic/normalizer.py` either merged or archived

### Risks

- **LOW**: Shims already in place
- **MITIGATION**: Verify shims emit deprecation warnings

### Impact on First Organism

- **INDIRECT**: First Organism uses normalization for hash computation

### Micro-Tasks

1. **Audit**: Grep for `from backend.logic` imports
2. **Verify**: Shims emit `DeprecationWarning`
3. **Decide**: Merge or archive `basis/logic/normalizer.py`
4. **Update**: Any direct imports to use `normalization/`
5. **Test**: First Organism still passes

### Agent Assignment

- **Claude C (Code Surgeon)**: Update imports
- **Claude D (Test Guardian)**: Verify no regression

---

## Epic 8: Consolidate API to interface/api/

**Priority:** P1
**Scope:** Establish `interface/api/app.py` as canonical FastAPI server

### Current State (Problem)

| Module | Status |
|--------|--------|
| `interface/api/app.py` | **CANONICAL** |
| `interface/api/schemas.py` | **CANONICAL** |
| `backend/orchestrator/app.py` | DEPRECATED SHIM |
| `backend/api/schemas.py` | UNCERTAIN |

### Target State

1. All API imports use `from interface.api import app, ...`
2. `backend/orchestrator/app.py` is shim with deprecation warning
3. Single schema definition in `interface/api/schemas.py`

### Risks

- **MEDIUM**: May affect deployment scripts
- **MITIGATION**: Verify `start_api_server.py` uses correct import

### Impact on First Organism

- **INDIRECT**: API used for metrics but not core loop

### Micro-Tasks

1. **Audit**: Grep for `from backend.orchestrator` imports
2. **Verify**: Shim emits `DeprecationWarning`
3. **Consolidate**: Merge `backend/api/schemas.py` into `interface/api/schemas.py`
4. **Update**: `start_api_server.py` to use canonical import
5. **Test**: API endpoints still work

### Agent Assignment

- **Claude C (Code Surgeon)**: Execute consolidation
- **Claude H (API Specialist)**: Verify endpoint behavior

---

## Epic 9: Document and Freeze First Organism Test

**Priority:** P0
**Scope:** Ensure `tests/integration/test_first_organism.py` is the canonical reference

### Current State

The test exists and defines the First Organism loop, but:
- May have flaky assertions
- May not cover all edge cases
- May not document expected artifacts

### Target State

1. Test is fully documented with inline comments
2. Test covers: derivation, verification, attestation, RFL
3. Test asserts: `H_t` recomputation, coverage gate, determinism
4. Test produces: `artifacts/first_organism/attestation.json`

### Risks

- **LOW**: Adding documentation/assertions doesn't change behavior
- **MITIGATION**: Run test before and after changes

### Impact on First Organism

- **DEFINING**: This test IS First Organism

### Micro-Tasks

1. **Review**: Current test coverage and assertions
2. **Document**: Add docstrings explaining each phase
3. **Add**: Assertions for `H_t` recomputation
4. **Add**: Assertions for determinism (run twice, same `H_t`)
5. **Add**: Artifact generation for evidence

### Agent Assignment

- **Claude D (Test Guardian)**: Enhance test
- **Claude G (Docs Maintainer)**: Document test

---

## Epic 10: Create Deprecation Shim Framework

**Priority:** P1
**Scope:** Standardize how deprecated modules emit warnings

### Current State

Shims exist but may not:
- Emit consistent warning format
- Log to correct logger
- Include migration instructions

### Target State

All deprecated modules use:
```python
import warnings
from functools import wraps

def deprecated(replacement: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{fn.__module__}.{fn.__name__} is deprecated. "
                f"Use {replacement} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return fn(*args, **kwargs)
        return wrapper
    return decorator
```

### Risks

- **LOW**: Adding warnings doesn't change behavior
- **MITIGATION**: Test that warnings don't break CI

### Impact on First Organism

- **NONE**: Warnings are informational

### Micro-Tasks

1. **Create**: `basis/shim/deprecation.py` with standard decorator
2. **Apply**: To `backend/logic/*.py` shims
3. **Apply**: To `backend/orchestrator/app.py` shim
4. **Apply**: To `backend/crypto/dual_root.py` shim
5. **Test**: Verify warnings appear in logs

### Agent Assignment

- **Claude B (Substrate Architect)**: Design framework
- **Claude C (Code Surgeon)**: Apply to all shims

---

## Priority Summary

| Epic | Priority | Impact | Effort | Agent |
|------|----------|--------|--------|-------|
| 1. Consolidate Hashing | P0 | HIGH | MEDIUM | B, C, D |
| 2. Deterministic Timestamps | P0 | HIGH | LOW | E, D |
| 9. Document First Organism Test | P0 | HIGH | LOW | D, G |
| 3. Collapse Dual-Root | P1 | MEDIUM | MEDIUM | B, C, F |
| 7. Normalize Logic Imports | P1 | MEDIUM | LOW | C, D |
| 8. Consolidate API | P1 | MEDIUM | MEDIUM | C, H |
| 10. Deprecation Framework | P1 | LOW | LOW | B, C |
| 4. Archive Consensus | P2 | LOW | LOW | F, D |
| 5. Archive Root Scripts | P2 | LOW | LOW | F, G |
| 6. Archive substrate/ | P2 | LOW | MEDIUM | B, F |

---

## Execution Order

**Phase 1: First Organism Integrity (P0)**
1. Epic 2: Remove nondeterministic timestamps
2. Epic 1: Consolidate hashing (fix `tools/verify_merkle.py` first)
3. Epic 9: Document and freeze First Organism test

**Phase 2: Import Path Cleanup (P1)**
4. Epic 10: Create deprecation framework
5. Epic 7: Normalize logic imports
6. Epic 3: Collapse dual-root
7. Epic 8: Consolidate API

**Phase 3: Slop Archival (P2)**
8. Epic 5: Archive root scripts
9. Epic 4: Archive consensus
10. Epic 6: Archive substrate

---

## Agent Roster

| Agent | Role | Expertise |
|-------|------|-----------|
| Claude A | Vibe Orchestrator | Intent compilation, constraint definition |
| Claude B | Substrate Architect | Module design, import hierarchy |
| Claude C | Code Surgeon | Refactoring, shim implementation |
| Claude D | Test Guardian | Test coverage, regression prevention |
| Claude E | Determinism Enforcer | Timestamp injection, PRNG seeding |
| Claude F | Archivist | Code archival, git history preservation |
| Claude G | Docs Maintainer | Documentation updates, reference fixing |
| Claude H | API Specialist | FastAPI, Pydantic, endpoint behavior |

---

*This agenda is the tactical plan for achieving First Organism. Epics should be executed in order unless dependencies require resequencing.*
