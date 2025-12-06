# Chaos Harness Phase I Specification
**Formal Specification Document — Do Not Implement**

**Author:** GEMINI N  
**Status:** Specification Only — Phase I Frozen  
**Date:** 2025-01-XX  
**Evidence Pack v1:** Consolidation Phase

---

## Executive Summary

This document formalizes the specification for a future chaos testing harness targeting First Organism (FO) infrastructure resilience. **All scenarios described herein are designated Phase II (future implementation). Phase I is explicitly marked as "Do Not Implement Yet."**

**Current State:**
- ✅ Risk documentation exists: `docs/RFL_EXPERIMENT_RISKS.md` (Section 5: First Organism Specific Risks, Section 6: Future Chaos Testing TODO)
- ✅ First Organism test suite exists: `tests/integration/test_first_organism.py`
- ❌ No chaos harness implementation exists
- ❌ No infrastructure failure injection utilities exist
- ❌ No chaos test execution framework exists

**This specification is a forward-looking design document. No implementation work should proceed until explicitly authorized.**

---

## Phase I: Specification Only — DO NOT IMPLEMENT

### Phase I Scope

Phase I consists solely of:
1. ✅ **Risk Documentation** (COMPLETE) — Documented in `docs/RFL_EXPERIMENT_RISKS.md`
2. ✅ **Formal Specification** (THIS DOCUMENT) — Formalization of future requirements
3. ❌ **Implementation** — EXPLICITLY FORBIDDEN until Phase II authorization

### Phase I Deliverables

| Deliverable | Status | Location |
|------------|--------|----------|
| Risk catalog | ✅ Complete | `docs/RFL_EXPERIMENT_RISKS.md` Section 5 |
| TODO anchors | ✅ Complete | `docs/RFL_EXPERIMENT_RISKS.md` Section 6 |
| Formal spec | ✅ Complete | This document |
| Test implementation | ❌ Forbidden | N/A |
| Infrastructure utilities | ❌ Forbidden | N/A |
| CI integration | ❌ Forbidden | N/A |

### Phase I Constraints

**Hard Boundaries:**
- NO test code creation
- NO infrastructure failure injection utilities
- NO container/process control mechanisms
- NO chaos test execution framework
- NO CI/CD integration
- NO pytest markers or fixtures beyond specification

**Rationale:** Evidence Pack v1 consolidation requires hardening existing artifacts, not building new systems. Chaos harness implementation is deferred to Phase II pending SPARK stability and explicit authorization.

---

## Phase II: Future Implementation (NOT AUTHORIZED)

### Phase II Prerequisites

Before any Phase II work may commence, the following conditions MUST be met:

1. **SPARK Test Stability**
   - SPARK tests (`@pytest.mark.first_organism`) must be consistently green across 10+ consecutive CI runs
   - Zero flaky test failures in SPARK suite
   - Evidence: CI logs showing stable SPARK execution

2. **First Organism Path Stability**
   - FO test suite executes reliably in local and CI environments
   - All 5 FO test phases (UI event → curriculum gate → derivation → dual-attestation → RFL metabolism) pass consistently
   - Evidence: Test execution logs showing deterministic FO behavior

3. **Migration System Reliability**
   - Database migrations complete without errors
   - No incomplete migration states observed
   - Evidence: Migration execution logs showing clean completion

4. **Explicit Authorization**
   - Written authorization from project maintainer to proceed with Phase II
   - Evidence: Git commit or issue comment explicitly authorizing chaos harness implementation

**Current Status:** Prerequisites NOT MET. Phase II work is FORBIDDEN.

---

## Phase II: TestFirstOrganismChaos Specification

### Overview

When Phase II is authorized, implement a comprehensive chaos test suite `TestFirstOrganismChaos` that validates FO behavior under infrastructure failures. All scenarios below are Phase II only.

### Scope Limitation: Hermetic RFL Runs

**Important:** RFL runs (e.g., `fo_rfl.jsonl`) are hermetic and operate independently of database infrastructure. Chaos scenarios that target Postgres/Redis failures **cannot be meaningfully applied** to hermetic RFL execution paths.

**Chaos testing scope:**
- ✅ **DB-backed FO tests** — Chaos scenarios apply to tests that require Postgres/Redis (e.g., ledger writes, attestation sealing, block creation)
- ❌ **Hermetic RFL runs** — Chaos scenarios do not apply to standalone RFL execution that produces `fo_rfl.jsonl` without database dependencies

**Rationale:** Hermetic RFL runs are designed to be infrastructure-independent. Injecting infrastructure failures into hermetic execution paths would test the wrong failure mode (hermetic runs should not depend on infrastructure in the first place).

**Chaos testing remains strictly Phase-II and DB-backed only.**

### Scenario 1: Postgres Mid-Run Termination

**Objective:** Verify FO test behavior when Postgres becomes unavailable during active execution.

**Failure Injection:**
- Kill Postgres container/process during an active FO test run
- Timing: Inject failure at mid-execution (after test start, before completion)

**Verification Criteria:**
- [ ] Test fails gracefully with clear error reporting (no silent skips)
- [ ] Partial ledger writes are rolled back (transaction integrity verified)
- [ ] No corrupted `H_t` values are emitted (Merkle root integrity preserved)
- [ ] Test can be re-run cleanly after Postgres recovery (recovery path validated)

**Implementation Notes:**
- Use container orchestration (Docker Compose) or process control to terminate Postgres
- Capture test output and verify error messages are diagnostic (not silent)
- Inspect database state post-failure to confirm rollback
- Verify `H_t` values in test artifacts are either absent (failure) or valid (if emitted before failure)

**Status:** Phase II — NOT IMPLEMENTED

---

### Scenario 2: Redis Drop During Attestation

**Objective:** Verify FO test behavior when Redis connectivity is lost during dual-attestation sealing.

**Failure Injection:**
- Terminate Redis connectivity during the dual-attestation sealing phase
- Timing: Inject failure specifically during `seal_block_with_dual_roots` or attestation state persistence

**Verification Criteria:**
- [ ] Attestation state is not lost (state preservation mechanism verified)
- [ ] Test either retries or fails with clear diagnostics (no silent failures)
- [ ] No partial attestation artifacts are written to disk (atomicity verified)
- [ ] Recovery path exists for resuming attestation after Redis restoration (resumability verified)

**Implementation Notes:**
- Use network partition simulation or container termination to drop Redis
- Verify attestation artifacts directory contains either complete artifacts or no artifacts (not partial)
- Test recovery by restoring Redis and verifying test can resume/complete

**Status:** Phase II — NOT IMPLEMENTED

---

### Scenario 3: Parallel FO Execution

**Objective:** Verify FO test behavior when multiple FO tests run simultaneously against the same database.

**Failure Injection:**
- Run two FO tests simultaneously against the same database instance
- Timing: Both tests start concurrently and execute in parallel

**Verification Criteria:**
- [ ] Cycle indices remain unique and monotonic (no duplicate cycle indices)
- [ ] Block headers do not conflict (no conflicting block creation)
- [ ] `H_t` values are deterministic per test run (not corrupted by interleaving)
- [ ] Database locks prevent ledger corruption (concurrency control verified)
- [ ] Both tests complete with valid, independent attestation seals (isolation verified)

**Implementation Notes:**
- Use pytest-xdist or multiprocessing to run tests in parallel
- Verify cycle indices in database are unique across both test runs
- Inspect block headers to ensure no conflicts
- Compare `H_t` values from both runs to verify determinism (each run produces consistent `H_t` independent of interleaving)

**Status:** Phase II — NOT IMPLEMENTED

---

## Phase II: Implementation Framework Specification

### Test Structure

When Phase II is authorized, create:

```
tests/integration/test_first_organism_chaos.py
```

**Test Markers:**
- `@pytest.mark.chaos` — Mark all chaos tests
- `@pytest.mark.phase_ii` — Mark as Phase II (future)

**Environment Gating:**
- All chaos tests MUST be gated behind `CHAOS_HARNESS=true` environment variable
- Tests MUST skip (not fail) when `CHAOS_HARNESS` is not set
- Rationale: Keep chaos tests separate from standard SPARK runs

### Infrastructure Utilities

When Phase II is authorized, create infrastructure failure injection utilities:

**Container Control:**
- Utility to start/stop/restart Docker containers (Postgres, Redis)
- Utility to simulate network partitions
- Utility to terminate processes gracefully

**State Management:**
- Utility to capture database state before/after failure
- Utility to capture attestation artifacts state
- Utility to verify transaction rollback

**Recovery Testing:**
- Utility to restore infrastructure after failure
- Utility to verify test resumability
- Utility to validate state consistency post-recovery

**Status:** Phase II — NOT IMPLEMENTED

### CI/CD Integration

When Phase II is authorized, integrate chaos tests into CI/CD:

**Execution Model:**
- Chaos tests run periodically (not on every PR)
- Gated behind `CHAOS_HARNESS=true` environment variable
- Separate CI job/workflow for chaos execution
- Not blocking for standard PR merges

**Reporting:**
- Chaos test results reported separately from SPARK results
- Failure artifacts captured (logs, database dumps, attestation artifacts)
- Results stored in `artifacts/chaos/` directory

**Status:** Phase II — NOT IMPLEMENTED

---

## Evidence Grounding

### Canonical Truth Source

**Reference:** `docs/RFL_PHASE_I_TRUTH_SOURCE.md` — Single source of truth for Phase-I RFL behavior and evidence.

This specification aligns with the canonical truth:
- RFL runs (e.g., `fo_rfl.jsonl`) are hermetic and file-based in Phase I
- Phase I has zero empirical RFL uplift (100% abstention by design)
- Chaos testing applies only to DB-backed FO tests, not hermetic RFL runs
- All Phase-II scenarios are explicitly marked as future implementation

### Existing Artifacts Referenced

1. **Risk Documentation**
   - File: `docs/RFL_EXPERIMENT_RISKS.md`
   - Sections: 5 (First Organism Specific Risks), 6 (Future Chaos Testing TODO)
   - Status: ✅ Exists on disk, non-empty, schema-valid

2. **First Organism Test Suite**
   - File: `tests/integration/test_first_organism.py`
   - Status: ✅ Exists on disk, non-empty
   - Evidence: Test file contains 5-phase FO test implementation
   - Note: Test has graceful skip behavior when DB/Redis unavailable (line 30: "skip gracefully when Postgres/Redis are down")

3. **No Chaos Implementation**
   - Search: No files matching `*chaos*.md` or `*CHAOS*.md` in docs/
   - Search: No test files matching `*chaos*.py` in tests/
   - Status: ❌ No chaos harness implementation exists

### Provenance

- **Risk Documentation:** Created by GEMINI N in Evidence Pack v1 consolidation phase
- **This Specification:** Created by GEMINI N in Evidence Pack v1 consolidation phase
- **Forward-Looking Claims:** All scenarios marked as Phase II (future), not current implementation

---

## Reviewer-2 Hardening

### Claims Verification

| Claim | Evidence | Status |
|-------|----------|--------|
| Risk documentation exists | `docs/RFL_EXPERIMENT_RISKS.md` exists, non-empty | ✅ Verified |
| FO test suite exists | `tests/integration/test_first_organism.py` exists, non-empty | ✅ Verified |
| Chaos harness does not exist | No chaos test files found in codebase | ✅ Verified |
| Phase I is specification only | This document explicitly marks Phase I as "Do Not Implement" | ✅ Verified |
| Phase II requires authorization | Prerequisites and authorization requirements documented | ✅ Verified |

### Inconsistencies

None identified. All forward-looking claims are explicitly marked as Phase II (future).

### Missing Data

- No chaos test execution logs (expected — not implemented)
- No infrastructure failure injection utilities (expected — not implemented)
- No CI/CD chaos test integration (expected — not implemented)

**Status:** All missing items are expected and documented as Phase II (future).

---

## Summary

**Phase I Status:** ✅ COMPLETE
- Risk documentation: ✅ Complete
- Formal specification: ✅ Complete (this document)
- Implementation: ❌ Forbidden (as intended)

**Phase II Status:** ❌ NOT AUTHORIZED
- Prerequisites: ❌ Not met
- Authorization: ❌ Not granted
- Implementation: ❌ Forbidden

**Next Steps:**
1. Wait for SPARK stability (10+ consecutive green runs)
2. Wait for explicit Phase II authorization
3. Only then proceed with Phase II implementation

**This specification is frozen. No Phase I implementation work should proceed.**

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Maintainer:** GEMINI N  
**Review Status:** Evidence Pack v1 Consolidation Phase

