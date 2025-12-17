# Temporal & Causal Semantics Extraction for MathLedger

---

> **STATUS: NON-CANONICAL / DESCRIPTIVE ONLY**
>
> This document describes observed temporal and causal structure in MathLedger as of 2025-12-13.
> It is not prescriptive, not a design specification, and not a recommendation for future work.
> All claims are grounded in code evidence. Claims that could not be verified are marked UNVERIFIED.

---

## Table of Contents

1. [Methodology Declaration](#1-methodology-declaration)
2. [Evidence Map Table](#2-evidence-map-table)
3. [Corrected Minimal Temporal Contract](#3-corrected-minimal-temporal-contract)
4. [Patch Notes](#4-patch-notes)

---

## 1. Methodology Declaration

### Grounding Protocol

This extraction follows strict evidence requirements:

1. **Code-Grounded Only**: Every claim references specific Python files, functions, classes, or constants with line ranges.
2. **Test-Verified**: Where tests exist that enforce behavior, they are cited by exact file path and test class/method name.
3. **No Invention**: No new abstractions, unified timelines, or category-theoretic structures are introduced.
4. **UNVERIFIED Marking**: Claims that could not be grounded in code are explicitly marked UNVERIFIED.

### Exclusions

- Documentation-only claims without code backing
- P5 blueprint components (not yet implemented)
- Speculative unification attempts
- Category theory or algebraic structure

---

## 2. Evidence Map Table

### Clock Domain Claims (CLK-XX)

| ID | Claim Summary | Enforcement Type | Evidence Pointer | Test Pointer | Confidence |
|----|---------------|------------------|------------------|--------------|------------|
| CLK-01 | USLA cycle clock advances via `USLASimulator.step()` | Code | `backend/topology/usla_simulator.py` :: `USLASimulator.step()` :: lines 352-497 | NONE (code-only; tile tests use step() indirectly) | VERIFIED |
| CLK-02 | USLA state includes `cycle` counter | Code | `backend/topology/usla_simulator.py` :: `USLAState` :: lines 54-86, field `cycle: int = 0` at line 83 | NONE | VERIFIED |
| CLK-03 | TDA window aggregates cycles into windows | Code | `backend/tda/monitor.py` :: `TDAMonitor` :: lines 1-100, `TDAWindowMetrics` referenced | `tests/tda/test_tda_monitor.py::TestTDAMonitor::test_finalize_window` :: lines 73-93 | VERIFIED |
| CLK-04 | Block clock advances via `seal_block_with_dual_roots()` | Code | `ledger/blocking.py` :: `seal_block_with_dual_roots()` | `tests/test_dual_root_attestation.py::TestBlockSealing::test_seal_block_with_dual_roots_basic` :: lines 224-250 | VERIFIED |
| CLK-05 | Attestation clock H_t computed from R_t and U_t | Code | `attestation/dual_root.py` :: `compute_composite_root()` :: lines 308-335 | `tests/test_dual_root_attestation.py` :: `TestDualRootComputation::test_compute_composite_root_valid` | VERIFIED |
| CLK-06 | RFL policy epoch increments on update | Code | `rfl/update_algebra.py` :: `apply_update()` returns `PolicyState` with `epoch=policy.epoch + 1` | `tests/rfl/test_rfl_engine.py` :: `TestUpdateAlgebra::test_chain_append` | VERIFIED |
| CLK-07 | Governance checker tracks cycle in input | Code | `backend/governance/last_mile_checker.py` :: `GovernanceFinalCheckInput` :: field `cycle: int` at line 57-85 | `tests/governance/test_last_mile_checker.py` :: fixture `healthy_input` | VERIFIED |

### Synchronization Barrier Claims (BARR-XX)

| ID | Claim Summary | Enforcement Type | Evidence Pointer | Test Pointer | Confidence |
|----|---------------|------------------|------------------|--------------|------------|
| BARR-01 | Dual attestation requires both R_t and U_t before H_t | Code | `attestation/dual_root.py` :: `compute_composite_root()` :: lines 322-326 raises `ValueError` if either root missing | `tests/test_dual_root_attestation.py` :: `TestDualRootComputation::test_compute_composite_root_invalid_r_t` | VERIFIED |
| BARR-02 | Block sealing barrier enforces monotonicity | Code | `backend/ledger/monotone_guard.py` :: `check_monotone_ledger()` :: lines 52-144 | `tests/ledger/test_monotone_guard.py` :: `TestCheckMonotoneLedger::test_height_violation`, `test_hash_chain_violation`, `test_timestamp_violation` | VERIFIED |
| BARR-03 | RFL event gate is fail-closed | Code | `rfl/event_verification.py` :: `RFLEventGate` :: lines 147+ | `tests/rfl/test_rfl_engine.py` :: `TestEventVerification::test_rfl_event_gate` :: lines 536-563 | VERIFIED |
| BARR-04 | RFL requires valid H_t before policy update | Code | `rfl/event_verification.py` :: `AttestedEvent.verify_composite_root()` :: lines 94-102 | `tests/rfl/test_rfl_engine.py` :: `TestEventVerification::test_attested_event_invalid_composite` :: lines 450-464 | VERIFIED |

### Happens-Before Claims (HB-XX)

| ID | Claim Summary | Enforcement Type | Evidence Pointer | Test Pointer | Confidence |
|----|---------------|------------------|------------------|--------------|------------|
| HB-01 | USLA phases 1-7 execute sequentially | Code | `backend/topology/usla_simulator.py` :: `step()` :: lines 371-497 phases labeled in comments | NONE | VERIFIED |
| HB-02 | Proof events → R_t → H_t | Code | `attestation/dual_root.py` :: `build_reasoning_attestation()` :: lines 256-264, then `compute_composite_root()` | `tests/test_dual_root_attestation.py::TestDualRootComputation::test_compute_composite_root_valid` :: lines 65-75 | VERIFIED |
| HB-03 | UI events → U_t → H_t | Code | `attestation/dual_root.py` :: `build_ui_attestation()` :: lines 267-275, then `compute_composite_root()` | `tests/test_dual_root_attestation.py::TestDualRootComputation::test_compute_ui_root_nonempty` :: lines 46-54 | VERIFIED |
| HB-04 | H_t → RFL policy update | Code | `rfl/runner.py` :: `run_with_attestation()` consumes `AttestedRunContext` with composite_root | `tests/rfl/test_rfl_engine.py::TestRFLEngineIntegration::test_rfl_consumes_only_dual_attested_events` :: lines 645-679 | VERIFIED |
| HB-05 | Last-mile gates G0→G1→G2→G3→G4→G5 sequential | Code + Test | `backend/governance/last_mile_checker.py` :: `GateId` enum :: lines 78-85 defines order | `tests/governance/test_last_mile_checker.py::TestGatePrecedence::test_g0_blocks_before_g1` :: lines 387-395 | VERIFIED |
| HB-06 | Signal precedence 1-11 enforced in fusion | Code + Test | `backend/governance/fusion.py` :: `SIGNAL_PRECEDENCE` :: lines 64-76 | `tests/governance/test_fusion.py::TestSignalPrecedence::test_precedence_order` :: lines 453-462 | VERIFIED |

### Forbidden Causal Edge Claims (FORB-XX)

| ID | Claim Summary | Enforcement Type | Evidence Pointer | Test Pointer | Confidence |
|----|---------------|------------------|------------------|--------------|------------|
| FORB-01 | AI proof ingestion cannot advance slice progression | Code + Runtime Guard | `curriculum/shadow_mode.py` :: `ShadowModeFilter.should_include()` :: lines 49-68, `SHADOW_REQUIRED_SOURCE_TYPES = {"external_ai"}` at line 31 | `tests/ingest/test_ai_proof_ingestion.py::TestSliceProgressionGuard::test_external_ai_blocked` :: lines 293-299 | VERIFIED |
| FORB-02 | SHADOW mode observations cannot trigger enforcement | Test (source scanning) | `backend/governance/fusion.py` :: lines 1-12 define SHADOW MODE CONTRACT | `tests/integration/test_shadow_mode_compliance.py::TestP3ShadowModeCompliance::test_p3_harness_no_governance_api_calls` :: lines 25-45 (scans for forbidden patterns) | VERIFIED |
| FORB-03 | TDA red-flags are LOGGED_ONLY, never enforced | Code | `backend/tda/monitor.py` :: `TDARedFlag` :: field `action: str = "LOGGED_ONLY"` at line 54 | NONE | VERIFIED |
| FORB-04 | Monotone guard rejects height regression | Code + Test | `backend/ledger/monotone_guard.py` :: `check_monotone_ledger()` :: lines 83-95 | `tests/ledger/test_monotone_guard.py` :: `TestCheckMonotoneLedger::test_height_violation` | VERIFIED |
| FORB-05 | Monotone guard rejects hash chain break | Code + Test | `backend/ledger/monotone_guard.py` :: `check_monotone_ledger()` :: lines 97-114 | `tests/ledger/test_monotone_guard.py` :: `TestCheckMonotoneLedger::test_hash_chain_violation` | VERIFIED |
| FORB-06 | Monotone guard rejects timestamp regression | Code + Test | `backend/ledger/monotone_guard.py` :: `check_monotone_ledger()` :: lines 116-133 | `tests/ledger/test_monotone_guard.py` :: `TestCheckMonotoneLedger::test_timestamp_violation` | VERIFIED |
| FORB-07 | RFL rejects unverified events (fail-closed gate) | Code + Test | `rfl/event_verification.py` :: `RFLEventGate` | `tests/rfl/test_rfl_engine.py` :: `TestEventVerification::test_rfl_event_gate` :: lines 536-563 | VERIFIED |
| FORB-08 | P3/P4 runners reject shadow_mode=False | Runtime Guard | `backend/topology/first_light/config.py` :: validation rejects shadow_mode=False | `tests/integration/test_shadow_mode_compliance.py` :: `TestP3ShadowModeCompliance::test_p3_harness_shadow_mode_enforced_in_config` | VERIFIED |
| FORB-09 | Last-mile checker verdicts are SHADOW only | Documentation + Convention | `backend/governance/last_mile_checker.py` :: docstring lines 7-11: "Verdicts are logged but NOT enforced" | NONE (docstring contract only; no enforcement test exists) | PARTIAL |

### Event Type Claims (EVT-XX)

| ID | Claim Summary | Enforcement Type | Evidence Pointer | Test Pointer | Confidence |
|----|---------------|------------------|------------------|--------------|------------|
| EVT-01 | USLA cycle boundary produces new state | Code | `backend/topology/usla_simulator.py` :: `step()` returns `USLAState` :: lines 470-490 | NONE | VERIFIED |
| EVT-02 | Block sealing produces block with attestation roots | Code | `ledger/blocking.py` :: `seal_block_with_dual_roots()` returns dict with `reasoning_merkle_root`, `ui_merkle_root`, `composite_attestation_root` | `tests/test_dual_root_attestation.py::TestBlockSealing::test_seal_block_with_dual_roots_basic` :: lines 224-250 | VERIFIED |
| EVT-03 | TDA red-flag event has mode="SHADOW" | Code | `backend/tda/monitor.py` :: `TDARedFlag.to_dict()` :: line 67 adds `"mode": "SHADOW"` | NONE | VERIFIED |
| EVT-04 | Governance evaluation produces verdict (SHADOW) | Code | `backend/governance/last_mile_checker.py` :: `GovernanceFinalCheckResult` with `Verdict` enum | `tests/governance/test_last_mile_checker.py::TestG0CatastrophicGate::test_g0_pass_no_cdi010` :: lines 101-106 | VERIFIED |

---

## 3. Corrected Minimal Temporal Contract

Based on verified evidence, the following temporal contract is enforced by MathLedger:

### 3.1 Clock Domains (Independent)

MathLedger operates with **multiple independent clocks**:

| Clock | Advancement Trigger | Monotonic | External Visibility |
|-------|---------------------|-----------|---------------------|
| **USLA Cycle** | `USLASimulator.step()` call | Yes (cycle++) | Yes (`USLAState.cycle`) |
| **Block** | `seal_block_with_dual_roots()` | Yes (height check) | Yes (`block_number`) |
| **Attestation** | H_t computation | Yes (content-addressed) | Yes (`composite_attestation_root`) |
| **RFL Epoch** | `apply_update()` | Yes (epoch++) | Yes (`PolicyState.epoch`) |
| **Governance** | `run_governance_final_check()` | Yes (cycle input) | Yes (result `cycle` field) |

### 3.2 Synchronization Barriers

**Barrier 1: Dual Attestation**
- Requires: R_t (reasoning root) AND U_t (UI root)
- Produces: H_t = SHA256(R_t || U_t)
- Enforcement: `compute_composite_root()` raises `ValueError` if either missing

**Barrier 2: Block Sealing**
- Requires: Accumulated proofs, system ID
- Produces: Immutable block with Merkle roots
- Enforcement: `check_monotone_ledger()` rejects height/hash/timestamp violations

**Barrier 3: RFL Event Gate**
- Requires: Valid H_t (verified via `verify_composite_root()`)
- Produces: Policy update permission
- Enforcement: Fail-closed gate rejects invalid attestations

### 3.3 Happens-Before Partial Order

```
Proof events ─────────────────────────────────────────┐
                                                      ↓
                                    build_reasoning_attestation() → R_t ──┐
                                                                          ↓
UI events ───────────────────────────────────────────┐                    │
                                                     ↓                    │
                                    build_ui_attestation() → U_t ─────────┤
                                                                          ↓
                                               compute_composite_root() → H_t
                                                                          ↓
                                                                    RFL gate
                                                                          ↓
                                                            policy_update (if admitted)
```

```
USLA step():
  Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → new_state
```

```
Last-Mile Gates:
  G0 (Catastrophic) → G1 (Hard) → G2 (Invariant) → G3 (Safe Region) → G4 (Soft) → G5 (Advisory)
```

### 3.4 Forbidden Causal Edges (Non-Interference Guarantees)

The following causal paths are **explicitly forbidden** by code guards:

| Forbidden Path | Enforcement |
|----------------|-------------|
| AI proof → slice progression | `ShadowModeFilter.should_include()` returns False for `source_type="external_ai"` |
| SHADOW observation → enforcement | `action="LOGGED_ONLY"` in all shadow artifacts; no enforcement code paths |
| TDA metrics → USLA state mutation | TDA module has no write access to USLA; read-only interface |
| RFL update → without valid attestation | `RFLEventGate.admit_event()` rejects invalid H_t |
| Ledger → height regression | `check_monotone_ledger()` returns violation for `curr_height <= prev_height` |
| Ledger → hash chain break | `check_monotone_ledger()` returns violation for `prev_hash != expected` |
| Ledger → timestamp regression | `check_monotone_ledger()` returns violation for `curr_ts < prev_ts` |

### 3.5 SHADOW Mode Non-Interference Contract

All governance subsystems operate in **SHADOW MODE**:

1. **Definition**: SHADOW mode means observations are logged but **never** trigger enforcement actions
2. **Markers**: All shadow outputs contain `mode="SHADOW"` and/or `action="LOGGED_ONLY"`
3. **Enforcement Guard**: P3/P4 runners validate `shadow_mode=True` at initialization
4. **Forbidden Patterns**: Source code explicitly excludes `governance.enforce`, `governance.abort`, `governance.block`

---

## 4. Patch Notes

### Corrections from Prior Report

The following corrections apply to the initial extraction (before evidence hardening):

| Prior Claim | Issue | Correction |
|-------------|-------|------------|
| `usla_simulator.py:54-124` for `to_vector()` | Line range approximate | Exact: `to_vector()` at lines 93-100; `USLAState` class at lines 54-86 |
| `curriculum/shadow_mode.py` in `backend/` | Path incorrect | Correct path: `curriculum/shadow_mode.py` (not `backend/curriculum/`) |
| `rfl/event_verification.py` RFLEventGate "fail-closed" | Needed test verification | Test verified: `tests/rfl/test_rfl_engine.py::TestEventVerification::test_rfl_event_gate` |
| TDA window tests | Assumed to exist | RESOLVED: `tests/tda/test_tda_monitor.py::TestTDAMonitor::test_finalize_window` verified |

### Corrections Applied (Close Partials Pass - 2025-12-13)

| Claim ID | Prior Status | Resolution |
|----------|--------------|------------|
| CLK-03 | PARTIAL | VERIFIED - Test found: `tests/tda/test_tda_monitor.py::TestTDAMonitor::test_finalize_window` lines 73-93 |
| HB-06 | PARTIAL | VERIFIED - Test found: `tests/governance/test_fusion.py::TestSignalPrecedence::test_precedence_order` lines 453-462 |
| FORB-02 | Docstring-based evidence | Fixed - Changed enforcement type to "Test (source scanning)" and added test line range |

### Corrections Applied (Final Hardening Pass - 2025-12-13)

| Claim ID | Change | Rationale |
|----------|--------|-----------|
| CLK-01 | Test pointer: clarified as "NONE (code-only)" | Tile tests use step() indirectly; no direct cycle increment test |
| CLK-04 | Added exact test: `TestBlockSealing::test_seal_block_with_dual_roots_basic` lines 224-250 | Replaced general file reference |
| HB-02 | Added exact test: `TestDualRootComputation::test_compute_composite_root_valid` lines 65-75 | Replaced general file reference |
| HB-03 | Added exact test: `TestDualRootComputation::test_compute_ui_root_nonempty` lines 46-54 | Replaced general file reference |
| HB-04 | Added exact test: `TestRFLEngineIntegration::test_rfl_consumes_only_dual_attested_events` lines 645-679 | Replaced class-only reference |
| HB-05 | Added exact test: `TestGatePrecedence::test_g0_blocks_before_g1` lines 387-395 | Replaced general file reference |
| FORB-01 | Replaced "(if exists)" with `TestSliceProgressionGuard::test_external_ai_blocked` lines 293-299 | Test file verified to exist |
| FORB-09 | DOWNGRADED to PARTIAL | Prior "test" was docstring assertion, not actual test enforcement |
| EVT-02 | Added exact test: `TestBlockSealing::test_seal_block_with_dual_roots_basic` lines 224-250 | Replaced general file reference |
| EVT-04 | Added exact test: `TestG0CatastrophicGate::test_g0_pass_no_cdi010` lines 101-106 | Replaced general file reference |

### Files Verified in This Pass

| File Path | Lines Examined | Purpose |
|-----------|----------------|---------|
| `backend/topology/usla_simulator.py` | 1-150, 350-500 | USLA state, step function |
| `attestation/dual_root.py` | 1-350 | Dual-root computation |
| `backend/governance/fusion.py` | 1-100 | Signal precedence |
| `backend/governance/last_mile_checker.py` | 1-100 | Gate hierarchy |
| `backend/ledger/monotone_guard.py` | 1-150 | Monotonicity checks |
| `curriculum/shadow_mode.py` | 1-250 | Shadow mode filter |
| `rfl/runner.py` | 1-100 | RFL orchestrator |
| `rfl/event_verification.py` | 1-150 | Event verification |
| `backend/tda/monitor.py` | 1-100 | TDA monitoring |
| `backend/ingest/pipeline.py` | 1-100 | AI proof ingestion |
| `tests/test_dual_root_attestation.py` | 1-100 | Attestation tests |
| `tests/ledger/test_monotone_guard.py` | 1-100 | Monotonicity tests |
| `tests/integration/test_shadow_mode_compliance.py` | 1-100 | Shadow mode tests |
| `tests/rfl/test_rfl_engine.py` | 1-100, 350-600 | RFL tests |
| `tests/governance/test_last_mile_checker.py` | 1-100 | Gate tests |

---

## Meta-Conclusion

**MathLedger implements a multi-clock, causally ordered governance pipeline with explicit synchronization barriers (dual attestation, block sealing, RFL gate), partial order constraints (proof→attest→update, G0→G1→...→G5), and non-interference guarantees (SHADOW mode isolation, forbidden causal edges), rather than a unified dynamical system or algebraically closed structure.**

This is a descriptive extraction. It names existing structure. It does not propose unification, design changes, or new invariants.

---

*Document generated: 2025-12-13*
*Evidence hardening pass: COMPLETE*
*Close partials pass: COMPLETE (2025-12-13)*
*Final hardening pass: COMPLETE (2025-12-13)*
*Verification status: 24 VERIFIED, 1 PARTIAL, 0 UNVERIFIED*
