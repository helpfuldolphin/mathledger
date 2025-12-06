# Proposed Architecture Structure

## Current State vs. Proposed State

### Current Architecture (Fragmented)

```
mathledger/
├── backend/
│   ├── crypto/              [Partially centralized]
│   │   ├── hashing.py       ✓ Used by 6 files
│   │   ├── auth.py
│   │   └── handshake.py
│   ├── orchestrator/
│   │   └── app.py           ⚠ 769 lines, complexity F-E
│   ├── logic/
│   │   └── canon.py         ⚠ complexity F (50)
│   └── axiom_engine/
│       └── derive.py        ⚠ 985 lines, complexity E
│
├── scripts/                 [Mixed responsibilities]
│   ├── generate_allblue_epoch_seal.py  ⚠ 816 lines, local crypto
│   ├── generate_allblue_fleet_state.py ⚠ local crypto
│   └── [Many others with duplicate crypto]
│
└── tools/                   [Ad-hoc utilities, many duplicates]
    ├── ci/
    │   ├── rfc8785_canon.py            ⚠ duplicate canonicalize
    │   ├── critical_path_intel.py      ⚠ duplicate canonicalize
    │   ├── velocity_plan.py            ⚠ duplicate canonicalize
    │   └── velocity_proof_pack.py      ⚠ duplicate canonicalize
    ├── repro/
    │   ├── autofix_drift_v3.py         ⚠ duplicate sha256 + canonicalize
    │   ├── autofix_drift_v3_2.py       ⚠ duplicate sha256 + canonicalize
    │   └── generate_attestation.py     ⚠ duplicate sha256 + canonicalize
    └── perf/
        └── perf_gate.py                ⚠ duplicate sha256 + canonicalize
```

**Problems:**
- 🔴 12 duplicate canonicalization implementations
- 🔴 6 duplicate SHA-256 implementations (only 6 files use centralized)
- 🔴 3 duplicate Merkle implementations
- 🔴 No domain separation in local crypto implementations → security risk
- 🔴 High complexity in core modules (F-grade functions)
- 🔴 Files >900 lines with mixed responsibilities

---

### Proposed Architecture (Consolidated)

```
mathledger/
├── backend/
│   ├── core/                          [NEW: Shared utilities]
│   │   ├── crypto/
│   │   │   ├── __init__.py
│   │   │   ├── hashing.py             ✓ Existing (domain separation)
│   │   │   ├── canon.py               [NEW] RFC 8785 canonicalization
│   │   │   ├── auth.py                ✓ Existing
│   │   │   └── handshake.py           ✓ Existing
│   │   ├── output/                    [NEW: Standardized logging]
│   │   │   ├── __init__.py
│   │   │   └── status.py              [NEW] Pass/fail/abstain
│   │   └── audit/                     [NEW: Metrics utilities]
│   │       ├── __init__.py
│   │       └── metrics.py
│   │
│   ├── orchestrator/                  [REFACTORED: Split responsibilities]
│   │   ├── app.py                     ✓ Slimmed down (setup only)
│   │   ├── routes.py                  [NEW] FastAPI route definitions
│   │   ├── handlers.py                [NEW] Business logic
│   │   └── ui.py                      [NEW] UI rendering
│   │
│   ├── logic/                         [REFACTORED: Reduced complexity]
│   │   ├── canon.py                   ✓ Refactored (complexity < C)
│   │   └── parser.py                  [NEW] Extracted parsing logic
│   │
│   ├── axiom_engine/                  [REFACTORED: Tests extracted]
│   │   ├── derive.py                  ✓ Production logic only (<600 lines)
│   │   └── strategies.py              [NEW] Derivation strategies
│   │
│   ├── api/                           ✓ Existing
│   ├── generator/                     ✓ Existing
│   ├── ledger/                        ✓ Existing
│   └── worker.py                      ✓ Existing (complexity < C)
│
├── scripts/                           [CONSUMERS: No local crypto]
│   ├── generate_allblue_epoch_seal.py ✓ Uses backend.core.crypto
│   ├── generate_allblue_fleet_state.py ✓ Uses backend.core.crypto
│   └── [All scripts import from backend.core]
│
├── tools/                             [CONSUMERS: No local crypto]
│   ├── ci/
│   │   ├── rfc8785_canon.py           ✓ Uses backend.core.crypto.canon
│   │   ├── critical_path_intel.py     ✓ Uses backend.core.crypto
│   │   ├── velocity_plan.py           ✓ Uses backend.core.crypto
│   │   └── velocity_proof_pack.py     ✓ Uses backend.core.crypto
│   ├── repro/
│   │   ├── autofix_drift_v3.py        ✓ Uses backend.core.crypto
│   │   ├── autofix_drift_v3_2.py      ✓ Uses backend.core.crypto
│   │   └── generate_attestation.py    ✓ Uses backend.core.crypto
│   └── perf/
│       └── perf_gate.py               ✓ Uses backend.core.crypto
│
└── tests/                             [EXPANDED: Tests from production code]
    ├── axiom_engine/                  [NEW] Extracted smoke tests
    │   └── test_derive_smoke.py
    └── [Existing test structure]
```

**Benefits:**
- ✅ Single source of truth for crypto operations
- ✅ Domain separation enforced everywhere
- ✅ Reduced complexity (no F-grade functions)
- ✅ Clear module boundaries
- ✅ Improved testability
- ✅ Better security posture

---

## Module Dependency Flow

### Current (Scattered dependencies)
```
scripts/generate_allblue_epoch_seal.py
  └─> [local rfc8785_canonicalize()]  ❌ Duplicate

tools/ci/velocity_plan.py
  └─> [local canonicalize_json()]     ❌ Duplicate

tools/repro/autofix_drift_v3.py
  └─> [local compute_sha256()]        ❌ Duplicate
  └─> [local rfc8785_canonicalize()]  ❌ Duplicate

backend/orchestrator/app.py
  └─> backend.crypto.hashing           ✓ Centralized

Result: Inconsistent, difficult to maintain, security risk
```

### Proposed (Centralized dependencies)
```
scripts/generate_allblue_epoch_seal.py
  └─> backend.core.crypto.canon        ✓ Centralized
  └─> backend.core.crypto.hashing      ✓ With domain separation

tools/ci/velocity_plan.py
  └─> backend.core.crypto.canon        ✓ Centralized
  └─> backend.core.output.status       ✓ Standardized

tools/repro/autofix_drift_v3.py
  └─> backend.core.crypto.canon        ✓ Centralized
  └─> backend.core.crypto.hashing      ✓ With domain separation

backend/orchestrator/app.py
  └─> backend.core.crypto.hashing      ✓ Centralized
  └─> backend.orchestrator.handlers    ✓ Separated concerns

Result: Consistent, maintainable, secure
```

---

## File Size Before/After

| File | Current | After Refactor | Change |
|------|---------|----------------|--------|
| backend/axiom_engine/derive.py | 985 | ~600 | -385 lines (tests moved) |
| scripts/generate_allblue_epoch_seal.py | 816 | ~750 | -66 lines (crypto imports) |
| backend/orchestrator/app.py | 769 | ~300 | -469 lines (split into 4 modules) |
| backend/logic/canon.py | 347 | ~250 | -97 lines (parser extracted) |
| backend/testing/hermetic_v2.py | 681 | ~650 | -31 lines (crypto imports) |

**Total reduction:** ~1,048 lines of redundant/misplaced code

---

## Complexity Before/After

| Function | Current Grade | After Refactor | Technique |
|----------|---------------|----------------|-----------|
| backend/logic/canon.py::normalize() | F (50) | B (8) | Decompose into parse/transform/emit |
| backend/orchestrator/app.py::statements_endpoint() | F (45) | B (7) | Extract handlers, use strategy pattern |
| backend/orchestrator/app.py::ui_statement_detail() | E (36) | B (8) | Move UI logic to separate module |
| backend/axiom_engine/derive.py::_run_smoke_pl() | E (32) | - | Move to tests/ directory |
| backend/logic/canon.py::_split_top() | D (28) | B (9) | Extract parser combinator helpers |

**Target:** All functions ≤ Grade B (cyclomatic complexity ≤ 10)

---

## Security Improvement

### Current State
```python
# In tools/repro/autofix_drift_v3.py (line 116)
def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
    # ❌ No domain separation
    # ❌ Vulnerable to second-preimage attacks
```

### After Refactor
```python
# All files use centralized module
from backend.core.crypto.hashing import sha256_hex, DOMAIN_STMT

statement_hash = sha256_hex(statement, domain=DOMAIN_STMT)
# ✅ Domain separation enforced
# ✅ Protected against CVE-2012-2459 type attacks
```

---

## Migration Path

### Phase 1: Foundation (HIGH PRIORITY)
1. Create `backend/core/crypto/canon.py`
2. Create `backend/core/output/status.py`
3. Add tests for new modules
4. No breaking changes

### Phase 2: Migration (HIGH PRIORITY)
1. Update tools/ to use backend.core.crypto
2. Update scripts/ to use backend.core.crypto
3. Update backend/testing/ to use backend.core.crypto
4. Run full test suite after each batch

### Phase 3: Refactoring (MEDIUM PRIORITY)
1. Split backend/orchestrator/app.py
2. Extract backend/logic/parser.py
3. Simplify backend/logic/canon.py
4. Move tests from production code

### Phase 4: Cleanup (LOW PRIORITY)
1. Remove all local crypto implementations
2. Add linting rules to prevent regressions
3. Update CI workflows
4. Create composite GitHub Actions

---

## Success Criteria

- [ ] Zero local implementations of canonicalize/sha256/merkle
- [ ] All functions ≤ Grade B complexity (cyclomatic ≤ 10)
- [ ] All files ≤ 600 lines (except generated)
- [ ] Code duplication < 3%
- [ ] All cryptographic operations use domain separation
- [ ] 100% of tools/scripts use backend.core modules
- [ ] Pre-commit hooks prevent new local crypto implementations

---

## Related Documents

- [OVERSIGHT_REPORT.md](./OVERSIGHT_REPORT.md) - Full analysis
- [REFACTOR_QUICK_REFERENCE.md](./REFACTOR_QUICK_REFERENCE.md) - Developer guide
- [../../CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines
- [../../AGENTS.md](../../AGENTS.md) - Agent-specific guidelines

---

*Last updated: 2025-11-02*
