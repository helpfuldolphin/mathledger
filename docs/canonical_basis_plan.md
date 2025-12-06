# Canonical Basis Plan

> **STATUS: PHASE II — NOT YET IMPLEMENTED**
>
> This document describes a **planned extraction** of code into `basis/`.
> The `basis/` package exists on disk but is **not yet imported by any production code**.
> All Phase I code uses `normalization/`, `attestation/`, `rfl/` directly.
>
> **The existence of additional RFL logs does not activate any Phase II modules.**
>
> See `docs/VSD_PHASE_2.md` for the full Phase II target architecture specification.

## Intent

The repository currently contains overlapping "spanning set" implementations accumulated across months of autonomous agent work. Our objective is to extract a minimal, orthogonal basis that preserves the strongest primitives (deterministic logic normalisation, domain-separated hashing, ledger sealing, dual-attestation) while discarding redundancy through segregation into a fresh `basis/` package.

This plan documents the decisions, invariants, and phased extraction steps.

## Design Principles

1. **Deterministic semantics** – all public functions are pure, parameterised, and stable across re-execution. No hidden environment reads or global mutation.
2. **Cryptographic hygiene** – domain-separated SHA-256, explicit encodings, and explicit validation of hex inputs.
3. **Curriculum ladder alignment** – curriculum assets are versioned and reproducible; transformations carry explicit provenance.
4. **Dual attestation discipline** – reasoning and human/UI streams remain separately attestable and composable.
5. **Canonical normalisation** – propositional/FOL expressions collapse to a single stable form for hashing, proof selection, and curriculum indexing.
6. **Orthogonal modules** – each subsystem exposes a single responsibility with typed data structures and narrow interfaces.

## Canonical Modules

| Subsystem | Source of truth in spanning set | Canonical extraction |
|-----------|---------------------------------|-----------------------|
| Logic canonicalisation | `backend/logic/canon.py` (best normaliser) | `basis/logic/normalizer.py` exposing `normalize`, `normalize_pretty`, `are_equivalent`, `atoms` |
| Cryptographic hashing | `backend/crypto/hashing.py` | `basis/crypto/hash.py` providing domain tags, `sha256_hex`, `merkle_root`, proof helpers |
| Ledger sealing | `backend/ledger/blockchain.py` | `basis/ledger/block.py` with `seal_block`, `merkle_root`, deterministic header typing |
| Dual attestation | `backend/crypto/dual_root.py`, `verify_dual_root.py` | `basis/attestation/dual.py` bundling Merkle aggregation, composite hash, metadata struct |
| Curriculum ladder | `backend/rfl/config.py`, `artifacts/rfl`, docs | `basis/curriculum/ladder.py` defining typed curriculum tiers and deterministic ladder renderer |
| Config + typing | scattered | `basis/core/types.py` for shared typed aliases and dataclasses |

## Target Directory Skeleton

```
basis/
  __init__.py
  core/
    __init__.py
    types.py
  crypto/
    __init__.py
    hash.py
  logic/
    __init__.py
    normalizer.py
  ledger/
    __init__.py
    block.py
  attestation/
    __init__.py
    dual.py
  curriculum/
    __init__.py
    ladder.py
  docs/
    invariants.md
```

Each module will export a narrow public surface and re-export the most frequently used functions via `basis/__init__.py`.

## Extraction Phases

**Current Status**: Phase 1 partially complete (files exist), Phases 2-4 not started.

1. **Module drafting (this pass)** – create fresh canonical modules populated with distilled logic from the current best implementations, cut to the minimal deterministic core.
   - **Status**: Files created in `basis/`, but **no consumers migrated**.
2. **Verification harness** – port the most relevant tests (`test_dual_attestation.py`, crypto hash checks, canon normaliser regressions) to reference `basis.*`.
   - **Status**: NOT DONE. Tests still use `normalization/`, `attestation/`.
3. **Integration map** – document how existing services (FastAPI wrapper, workers) can migrate by replacing imports.
   - **Status**: NOT DONE.
4. **Deprecation sweep** – mark legacy modules with docstring pointers to `basis.*` (non-destructive) once confidence is established.
   - **Status**: NOT DONE. Shims exist but point to transitional modules, not `basis/`.

## Determinism Checklist

- Normaliser caches remain bounded and keyed solely on input strings.
- Hashing functions enforce ASCII encoding and domain tags.
- Ledger sealing uses explicit timestamps supplied by callers; no implicit `time.time()`.
- Attestation metadata is pure dict construction.
- Curriculum ladder builder consumes versioned JSON assets or in-repo static fixtures.

## Open Items

- Formal proofs for normaliser equivalence (Lean project integration) – future work.
- Curriculum dataset pruning – requires stakeholder input before deleting derived artefacts.
- API/worker alignment – schedule after canonical package stabilises.

## Phase I / Phase II Module Boundary

All Phase I evidence was produced using these active modules:

| Layer | Phase I (Active) | Phase II (Target) |
|-------|------------------|-------------------|
| Normalization | `normalization/canon.py` | `basis/logic/normalizer.py` |
| Attestation | `attestation/dual_root.py` | `basis/attestation/dual.py` |
| RFL Runner | `rfl/runner.py` | (same — already canonical) |
| Crypto | `substrate/crypto/core.py` | `basis/crypto/hash.py` |

The `basis/` package has **zero consumers**. No RFL log file — regardless of cycle count — was produced using `basis/` modules.

## Phase I RFL Evidence Status

| File | Status | Phase I Claim? |
|------|--------|----------------|
| `results/fo_rfl_50.jsonl` | **Canonical** — 50-cycle complete run | ✅ YES |
| `results/fo_rfl.jsonl` | **Degenerate** — ~330 cycles, lean-disabled, all-abstain | ❌ NO |
| 1000-cycle baseline/RFL | Dyno Chart comparison data | ✅ YES (for abstention comparison) |

The canonical Phase I RFL evidence remains `fo_rfl_50.jsonl`. Additional logs do not constitute uplift evidence.

## Uplift as Input to Basis Promotion (Phase II Policy)

> **STATUS: NO CURRENT RFL LOGS SATISFY THESE CONDITIONS; `basis/` REMAINS UNPROMOTED.**

This section defines how future uplift experiments could become **one input** into `basis/` promotion decisions, should such experiments ever pass the VSD Uplift Evidence Gate (see `docs/VSD_PHASE_2.md`).

### Relationship Between Uplift and Promotion

```
┌─────────────────────────────────────────────────────────────────┐
│                    UPLIFT ≠ PROMOTION                           │
│                                                                 │
│  Uplift Evidence (if it existed) would be ONE input among:      │
│  • Test coverage of basis/ modules                              │
│  • Consumer migration readiness                                 │
│  • CI gate implementation                                       │
│  • Type coverage (mypy --strict)                                │
│  • Security audit of cryptographic paths                        │
│                                                                 │
│  Uplift alone is INSUFFICIENT to trigger promotion.             │
└─────────────────────────────────────────────────────────────────┘
```

### Preconditions for Uplift to Influence Promotion

If a future RFL experiment were to pass the VSD Uplift Evidence Gate, it could influence `basis/` promotion decisions **only if**:

1. **Gate satisfied**: The experiment passes ALL criteria in `VSD_PHASE_2.md § Phase II Uplift Evidence Gate`
2. **Scope match**: The uplift was demonstrated using code paths that would benefit from `basis/` consolidation
3. **No regression**: Migrating to `basis/` would not break the demonstrated uplift
4. **Independent validation**: At least one other experiment (different slice or seed) shows consistent direction

### What Uplift Would NOT Do

Even with valid uplift evidence:

- `basis/` would **NOT** auto-promote without explicit migration work
- Existing Phase I clarifiers would **NOT** be retroactively loosened
- Claims would remain bounded to the specific configuration tested

### Current Status

| Question | Answer |
|----------|--------|
| Do any RFL logs pass the Uplift Evidence Gate? | **NO** |
| Is `basis/` promoted? | **NO** |
| Are there consumers of `basis/`? | **NO** |
| Can uplift be cited in governance decisions? | **NO** |

**The `basis/` package remains unpromoted. No uplift evidence exists that satisfies the gate criteria.**

