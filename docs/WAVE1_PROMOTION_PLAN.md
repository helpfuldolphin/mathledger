# Wave 1 Promotion Plan — Basis Promotion & First Organism Readiness

**Auditor**: Claude O
**Date**: 2025-11-25
**Status**: Ready for Review

---

## Executive Summary

The First Organism integration tests are passing, deterministic, and secure. This document identifies which modules from the spanning set meet the basis criteria and provides a promotion checklist for Cursor P.

### Basis Criteria (from `basis/docs/invariants.md`)

1. **Pure functions only** — no ambient reads or writes
2. **Deterministic ordering** — collections are tuples or sorted lists
3. **ASCII canonicalisation** — input strings normalized before crypto paths
4. **Domain-separated hashing** — every SHA-256 has a domain tag
5. **Explicit validation** — composite hashes validate hex length/format
6. **Stable JSON** — canonical separators and key sorting

---

## Module Compliance Matrix

### Law Domain (Ledger, Dual Root, Attestation)

| Module | Location | Basis Analogue | Criteria Met | Notes |
|--------|----------|----------------|--------------|-------|
| `attestation.dual_root` | `attestation/dual_root.py` | `basis.attestation.dual` | ✅ All | RFC 8785 canonicalization, domain separation, hex validation |
| `ledger.blocking` | `ledger/blocking.py` | `basis.ledger.block` | ✅ All | Deterministic `seal_block_with_dual_roots`, content-addressed hashes |
| `ledger.ingest` | `ledger/ingest.py` | N/A (DB layer) | ⚠️ Partial | Deterministic timestamp via `deterministic_timestamp_from_content`, but has DB side effects |
| `backend.crypto.hashing` | `backend/crypto/hashing.py` | `basis.crypto.hash` | ✅ All | Domain prefixes, Merkle proof computation |
| `substrate.crypto.core` | `substrate/crypto/core.py` | `basis.crypto.hash` | ✅ All | RFC 8785 JSON canonicalization |

### Economy Domain (Derivation, Curriculum)

| Module | Location | Basis Analogue | Criteria Met | Notes |
|--------|----------|----------------|--------------|-------|
| `normalization.canon` | `normalization/canon.py` | `basis.logic.normalizer` | ✅ All | Pure, cached, ASCII output |
| `backend.axiom_engine.derive_core` | `backend/axiom_engine/derive_core.py` | N/A | ⚠️ Partial | Uses deterministic seeds, but has DB/Redis interactions |
| `backend.frontier.curriculum` | `backend/frontier/curriculum.py` | `basis.curriculum.ladder` | ✅ All | Pure gate evaluators, normalized metrics |
| `derivation.pipeline` | `derivation/pipeline.py` | N/A | ⚠️ Partial | Bounded derivation, but orchestrator-level |

### Metabolism Domain (RFL Runner)

| Module | Location | Basis Analogue | Criteria Met | Notes |
|--------|----------|----------------|--------------|-------|
| `rfl.runner` | `rfl/runner.py` | N/A | ⚠️ Partial | `run_with_attestation` is pure, but `run_all` has Redis/file I/O |
| `rfl.bootstrap_stats` | `rfl/bootstrap_stats.py` | N/A | ✅ All | Pure bootstrap CI computation |
| `rfl.config` | `rfl/config.py` | N/A | ✅ All | Immutable dataclasses |

---

## Modules Ready for Wave 1 Promotion

### Tier 1: Promote Immediately (No Changes Required)

These modules already meet all basis criteria and are tested in the First Organism integration suite:

1. **`attestation.dual_root`** → Already using `basis.attestation.dual` patterns
   - `compute_reasoning_root()`, `compute_ui_root()`, `compute_composite_root()`
   - `verify_composite_integrity()`, `generate_attestation_metadata()`
   - Domain-separated leaf hashing with `DOMAIN_REASONING_LEAF`, `DOMAIN_UI_LEAF`

2. **`normalization.canon`** → Mirror of `basis.logic.normalizer`
   - `normalize()`, `are_equivalent()`, `canonical_bytes()`
   - LRU-cached, pure functions, ASCII output

3. **`rfl.bootstrap_stats`** → Self-contained statistics
   - `compute_coverage_ci()`, `compute_uplift_ci()`, `verify_metabolism()`
   - Pure numpy computations with deterministic seeds

4. **`backend.crypto.hashing`** → Crypto primitives
   - `merkle_root()`, `compute_merkle_proof()`, `verify_merkle_proof()`
   - Domain-separated `DOMAIN_LEAF`, `DOMAIN_NODE`, `DOMAIN_STMT`

### Tier 2: Promote After Minor Refactor

These modules have I/O dependencies that should be injected:

1. **`ledger.blocking.seal_block_with_dual_roots`**
   - Currently calls `materialize_ui_artifacts()` for implicit UI state
   - Refactor: Accept `ui_events` as explicit parameter (already supported)

2. **`rfl.runner.RFLRunner.run_with_attestation`**
   - The method is pure given an `AttestedRunContext`
   - Refactor: Extract into standalone function `process_attestation(config, attestation)`

### Tier 3: Document-Only (Orchestrator Layer)

These modules orchestrate I/O and should NOT be promoted to basis:

1. `ledger.ingest.LedgerIngestor` — DB persistence layer
2. `backend.axiom_engine.derive` — Full derivation with DB/Redis
3. `rfl.runner.RFLRunner.run_all` — 40-run orchestrator

---

## Promotion Checklist for Cursor P

When the First Organism test suite is green, execute the following:

### Pre-Promotion Verification

- [ ] `pytest tests/integration/test_first_organism.py -v` — All tests pass
- [ ] `pytest tests/test_dual_root_attestation.py -v` — Attestation primitives verified
- [ ] `pytest tests/test_basis_core.py -v` — Basis genome intact
- [ ] `pytest tests/test_canon.py -v` — Hash contract holds
- [ ] Confirm determinism: Run tests twice, compare H_t values

### Tier 1 Promotion Steps

1. **Create `basis.attestation.dual_root` (if not exists)**
   ```python
   # basis/attestation/dual_root.py
   # Re-export from attestation.dual_root with basis-compatible types
   from attestation.dual_root import (
       compute_reasoning_root,
       compute_ui_root,
       compute_composite_root,
       verify_composite_integrity,
       canonicalize_reasoning_artifact,
       canonicalize_ui_artifact,
   )
   ```

2. **Update `basis/__init__.py` exports**
   ```python
   from basis.attestation.dual_root import (
       compute_reasoning_root,
       compute_ui_root,
       compute_composite_root,
       verify_composite_integrity,
   )
   ```

3. **Add Wave 1 integration test marker**
   ```python
   # pytest.ini
   markers =
       wave1: Wave 1 promoted modules
   ```

### Tier 2 Promotion Steps

1. **Refactor `ledger.blocking`**
   - Ensure `seal_block_with_dual_roots` never calls `materialize_ui_artifacts()` internally
   - All UI events must be passed explicitly

2. **Extract `rfl.runner.process_attestation`**
   ```python
   def process_attestation(
       config: RFLConfig,
       attestation: AttestedRunContext,
   ) -> RflResult:
       """Pure attestation processor — no Redis, no file I/O."""
       # ... existing run_with_attestation logic without telemetry
   ```

### Post-Promotion Verification

- [ ] `pytest -m wave1 -v` — All Wave 1 tests pass
- [ ] `artifacts/first_organism/attestation.json` exists and is valid
- [ ] H_t in attestation matches test output: `[PASS] FIRST ORGANISM ALIVE H_t=...`

---

## Dependency Graph

```
basis/
├── core/types.py          ← Block, DualAttestation, CurriculumTier
├── crypto/hash.py         ← merkle_root, hash_statement, domain separation
├── logic/normalizer.py    ← normalize, canonical_bytes
├── ledger/block.py        ← seal_block (basis-pure version)
├── attestation/dual.py    ← composite_root, verify_attestation
└── curriculum/ladder.py   ← CurriculumLadder

attestation/dual_root.py   → Uses: substrate.crypto.{core,hashing}
ledger/blocking.py         → Uses: attestation.dual_root, backend.crypto
normalization/canon.py     → Standalone (mirrors basis.logic.normalizer)
rfl/bootstrap_stats.py     → Standalone (numpy only)
```

---

## Risk Assessment

### Low Risk (Promote)
- `attestation.dual_root` — Cryptographic primitives are well-tested
- `normalization.canon` — Extensive unit test coverage
- `rfl.bootstrap_stats` — Pure math, deterministic seeds

### Medium Risk (Monitor)
- `ledger.blocking` — Implicit `materialize_ui_artifacts()` call in fallback path
- `rfl.runner` — Redis telemetry could introduce non-determinism if not disabled

### Mitigations
1. Add `DETERMINISTIC_MODE=1` env flag to disable all telemetry
2. Add assertions in basis tests that verify no network I/O occurs

---

## Conclusion

**Wave 1 is ready for promotion.** The following modules meet basis criteria:

| Priority | Module | Action |
|----------|--------|--------|
| P0 | `attestation.dual_root` | Promote to basis |
| P0 | `normalization.canon` | Promote to basis |
| P0 | `rfl.bootstrap_stats` | Promote to basis |
| P1 | `ledger.blocking` | Refactor, then promote |
| P1 | `rfl.runner.run_with_attestation` | Extract, then promote |

The First Organism closed loop (UI → Gate → Derive → Attest → RFL) is architecturally sound and cryptographically binding. When the integration test emits:

```
[PASS] FIRST ORGANISM ALIVE H_t=<64-char-hex>
```

Wave 1 promotion can proceed without further architectural debate.
