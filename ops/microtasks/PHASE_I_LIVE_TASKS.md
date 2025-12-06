# Phase I Live Tasks — Evidence Pack v1 Scope

**Generated:** 2025-11-30
**Mode:** SOBER TRUTH / REVIEWER-2
**Source:** DECOMPOSITION_PHASE_PLAN.json (filtered for Phase I relevance)

---

## CRITICAL NOTICE

This document lists ONLY the micro-tasks that are:
1. **Actually completed** (backed by existing code/tests on disk), OR
2. **Relevant to Evidence Pack v1** (FO closed-loop test + attestation.json + dyno charts)

**Everything else is Phase II — Deferred.**

---

## Phase I Hard Evidence (Already Sealed)

These artifacts exist and are verified:

| Artifact | Location | Status |
|----------|----------|--------|
| FO Attestation | `artifacts/first_organism/attestation.json` | ✅ Sealed |
| FO Baseline 1000-cycle | `artifacts/phase_ii/fo_series_1/fo_1000_baseline/experiment_log.jsonl` | ✅ Valid |
| FO RFL 50-cycle (Canonical) | `results/fo_rfl_50.jsonl` | ✅ Canonical Phase I |
| Abstention Rate Figure | `artifacts/figures/rfl_abstention_rate.png` | ✅ Available |
| Dyno Chart | `artifacts/figures/rfl_dyno_chart.png` | ✅ Available |

---

## Phase-I RFL Evidence Status

| File | Cycles | Status | What It Proves |
|------|--------|--------|----------------|
| `results/fo_rfl_50.jsonl` | 50 | **CANONICAL Phase I RFL** | RFL code path executes. All cycles abstain. |
| `results/fo_rfl.jsonl` | ~330 | Phase I (degenerate) | Extended run, all-abstain. Confirms loop mechanics. |

**CRITICAL CLARIFICATION:**
- **Both files are Phase I evidence**, NOT Phase II.
- Neither demonstrates **RFL uplift** (improved derivation due to RFL).
- These logs prove the RFL code path executes correctly.
- They do **NOT** prove RFL improves derivation performance.
- Any claim of "RFL uplift" requires Phase II experiments **not yet run**.

**CONSTRAINT:** No Phase II task may implicitly depend on RFL uplift data from these logs.

---

## Phase I Completed Tasks

These tasks are **already done** — code exists and tests pass:

### HASH (Hashing Infrastructure) — COMPLETED

| ID | Title | Evidence |
|----|-------|----------|
| HASH-001 | Consolidate DOMAIN_* constants | `substrate/crypto/hashing.py` exists with DOMAIN_LEAF through DOMAIN_ROOT |
| HASH-002 | Type hints on hash functions | Present in `substrate/crypto/hashing.py` |
| HASH-005 | Merkle determinism | Used in sealed attestation.json |

### DUAL (Dual-Root Attestation) — COMPLETED

| ID | Title | Evidence |
|----|-------|----------|
| DUAL-001 | Migration to attestation.dual_root | `attestation/dual_root.py` is canonical |
| DUAL-003 | Attestation failure mode tests | `tests/test_dual_root_attestation.py` exists |
| DUAL-005 | RFC 8785 determinism | Used in H_t computation in sealed attestation |

### NORM (Normalization) — COMPLETED

| ID | Title | Evidence |
|----|-------|----------|
| NORM-001 | Unicode edge case tests | `tests/test_canon.py` exists |
| NORM-003 | Idempotency tests | Covered in test suite |
| NORM-005 | ASCII-only output | `canonical_bytes()` raises on non-ASCII |

### DET (Determinism) — COMPLETED

| ID | Title | Evidence |
|----|-------|----------|
| DET-001 | Determinism tests | `tests/test_determinism_helpers.py` exists |
| DET-003 | First Organism determinism | `ledger/first_organism.py` + `tests/test_first_organism_ledger.py` |
| DET-004 | Deterministic seeding | `substrate/repro/determinism.py` exists |
| DET-006 | FO determinism test passes | `tests/integration/test_first_organism_determinism.py` |
| DET-007 | Abstention determinism | `tests/test_abstention_determinism.py` |

### VER (Verification) — COMPLETED

| ID | Title | Evidence |
|----|-------|----------|
| VER-001 | Pattern matching layer | `derivation/verification.py` |
| VER-002 | Truth-table fallback | `normalization/truthtab.py` |
| VER-003 | Lean-disabled mode | Returns "lean-disabled" when env var unset |

### DAG (Proof DAG) — COMPLETED

| ID | Title | Evidence |
|----|-------|----------|
| DAG-001 | Cycle detection | `substrate/dag/proof_dag.py` |
| DAG-002 | Self-loop detection | Part of DAG validation |

### FO (First Organism) — COMPLETED

| ID | Title | Evidence |
|----|-------|----------|
| FO-001 | Contract test | `tests/unit/test_first_organism_contract.py` |
| FO-002 | DAG integration test | `tests/integration/test_first_organism_dag.py` |
| FO-003 | Pass line threshold | `tests/integration/test_first_organism_pass_line.py` |

---

## Phase I — Remaining Work (Evidence Pack v1 Only)

These tasks are NOT YET DONE but are relevant to Evidence Pack v1:

### High Priority — Blocks Evidence Pack Seal

| ID | Title | Why Blocking |
|----|-------|--------------|
| DUAL-007 | verify_composite_integrity edge cases | Need to verify H_t recomputation passes |
| API-007 | /health deterministic timestamp | Already implemented, just needs verification |
| API-008 | attestation/latest 404 test | Edge case for evidence pack validation |

### Medium Priority — Improves Evidence Pack Confidence

| ID | Title | Why Useful |
|----|-------|------------|
| HASH-006 | Empty Merkle tree test | Edge case coverage |
| HASH-007 | Odd-count Merkle test | Edge case coverage |
| NORM-006 | _to_ascii(None) test | Defensive test |
| DAG-003 | ancestors() max_depth test | DAG completeness |
| FO-004 | Telemetry hook verification | Confirms metrics capture |
| FO-005 | Failure mode tests | Robustness for evidence |

---

## Phase II — DEFERRED (Not in Evidence Pack v1)

**The following are explicitly NOT part of Phase I. Do not claim these as completed.**

> **GLOBAL CONSTRAINT:** Not activated by current RFL logs. No Phase II task may claim RFL uplift as evidence.

### CURR-* (All Curriculum/Wide Slice Tasks) — PHASE II

> *Not activated by current RFL logs.*

| ID | Title | Why Deferred |
|----|-------|--------------|
| CURR-001 | Slice transition tests | Wide slice theory not implemented |
| CURR-002 | Curriculum docs | Theory not proven |
| CURR-003 | Progress tracking metrics | No Phase I evidence |
| CURR-004 | Max depth edge case | Theoretical |
| CURR-005 | Empty statement set | Theoretical |
| CURR-006 | Breadth cap test | Not in FO closed-loop |

### RFL-* (RFL Scaling Theory) — PHASE II

> *Not activated by current RFL logs.*

| ID | Title | Why Deferred |
|----|-------|--------------|
| RFL-001 | Migrate experiment.py | Can wait for Phase II |
| RFL-002 | Migrate coverage.py | Can wait for Phase II |
| RFL-003 | Move get_or_create_system_id | Can wait for Phase II |
| RFL-004 | RFL determinism verification | RFL scaling theory not proven |
| RFL-005 | RFL runner test | Can wait |
| RFL-006 | Bootstrap statistics | Statistical theory not in Phase I |

### SEC-* (Security Hardening) — PHASE II

> *Not activated by current RFL logs.*

| ID | Title | Why Deferred |
|----|-------|--------------|
| SEC-001 | Token rotation | Not needed for evidence pack |
| SEC-002 | Ed25519 signing | Future infrastructure |
| SEC-003 | Graduated rate limiting | Not in FO scope |
| SEC-004 | Audit logging | Not in FO scope |
| SEC-005 | SQL injection review | Important but not blocking |
| SEC-006 | Redis auth | Not blocking evidence |

### TYPE-* (Type Safety) — PHASE II

> *Not activated by current RFL logs.*

| ID | Title | Why Deferred |
|----|-------|--------------|
| TYPE-001 | substrate/ type hints | Nice to have |
| TYPE-002 | derivation/ type hints | Nice to have |
| TYPE-003 | Enable mypy | Post-Phase I |
| TYPE-004 | Fix mypy violations | Post-Phase I |
| TYPE-005 | Pydantic DB models | Post-Phase I |

### DOC-* (Documentation) — PHASE II

> *Not activated by current RFL logs.*

| ID | Title | Why Deferred |
|----|-------|--------------|
| DOC-001 | OpenAPI docs | Not blocking evidence |
| DOC-002 | Migration guide | Post-deprecation |
| DOC-003 | SECURITY.md | Post-Phase I |
| DOC-004 | Schema docs | Post-Phase I |
| DOC-005 | Architecture diagrams | Post-Phase I |
| DOC-006 | Runbooks | Post-Phase I |
| DOC-007 | Environment vars | Low priority |

### DEPR-* (Deprecation) — PHASE II

> *Not activated by current RFL logs.*

| ID | Title | Why Deferred |
|----|-------|--------------|
| DEPR-001 | Complete legacy migration | Shims work, not blocking |
| DEPR-002 | CI legacy import check | Post-Phase I |
| DEPR-003 | Update test imports | Can wait |
| DEPR-004 | Migration guide | Post-Phase I |
| DEPR-005 | Schedule shim removal | Tracking issue only |

### Other Deferred

> *Not activated by current RFL logs.*

| ID | Title | Why Deferred |
|----|-------|--------------|
| HASH-003 | Preimage attack docs | Documentation |
| HASH-004 | Algorithm audit trail | Documentation |
| DUAL-002 | Attestation docstrings | Documentation |
| DUAL-004 | Revocation stub | Future feature |
| DUAL-006 | Metadata export test | Nice to have |
| NORM-002 | Symbol mapping docs | Documentation |
| NORM-004 | Fuzzing tests | Nice to have |
| DET-002 | Non-determinism docs | Documentation |
| DET-005 | Replay capability | Phase II feature |
| DAG-004 | Schema tolerance test | Nice to have |
| VER-004 | _to_lean test | Nice to have |
| API-001 | Lean dependency extraction | Phase II feature |
| API-002 | Agent ledger bridge | Phase II feature |
| API-003 | Request logging | Phase II feature |
| API-004 | API versioning | Phase II feature |
| API-005 | API contract docs | Documentation |
| API-006 | OpenAPI validation | Nice to have |

---

## Summary

| Category | Phase I Complete | Phase I Remaining | Phase II Deferred |
|----------|-----------------|-------------------|-------------------|
| HASH | 3 | 2 | 2 |
| DUAL | 3 | 1 | 3 |
| NORM | 3 | 1 | 2 |
| DET | 5 | 0 | 2 |
| VER | 3 | 0 | 1 |
| DAG | 2 | 1 | 1 |
| FO | 3 | 2 | 0 |
| CURR | 0 | 0 | 6 |
| RFL | 0 | 0 | 6 |
| SEC | 0 | 0 | 6 |
| TYPE | 0 | 0 | 5 |
| DOC | 0 | 0 | 7 |
| DEPR | 0 | 0 | 5 |
| API | 0 | 2 | 6 |
| **TOTAL** | **22** | **9** | **54** |

---

## Reviewer-2 Compliance Checklist

- [x] No ΔH scaling claims
- [x] No imperfect verifier theory claims
- [x] No wave-1 basis promotion claims
- [x] No MDAP micro-agent claims
- [x] No Lean sandbox claims
- [x] No wide slice theory claims
- [x] No logistic abstention theory claims
- [x] All Phase II tasks explicitly labeled as DEFERRED
- [x] Phase I evidence files cited with actual paths
- [x] **RFL logs clarified: both fo_rfl_50.jsonl and fo_rfl.jsonl are Phase I, NOT uplift evidence**
- [x] **No Phase II task implicitly depends on RFL uplift data**
- [x] **"Not activated by current RFL logs" constraint added to all Phase II categories**
- [x] **Phase II uplift roadmap defined with governance gates**

---

## 🚫 DO NOT DO DURING PHASE I — RFL Uplift Prohibitions

**This section is a directive to all agents (Claude A-O, human operators, CI systems).**

The following actions are **PROHIBITED** during Phase I:

### 1. Do NOT Design New RFL Experiments Under Phase I Label

| Prohibited | Why |
|------------|-----|
| Creating new experiment configs for RFL uplift | Uplift experiments are Phase II only |
| Modifying `experiments/` to add uplift runs | Phase I experiment directory is frozen |
| Adding new `fo_uplift_*.jsonl` files | These are Phase II deliverables |
| Running `run_fo_cycles.py` with uplift-seeking parameters | Phase I runs are complete |

**Exception:** Bug fixes to existing Phase I infrastructure are allowed if they don't change sealed evidence.

### 2. Do NOT Attempt to Salvage Uplift Narratives from Existing Logs

| Prohibited | Why |
|------------|-----|
| Reinterpreting `fo_rfl_50.jsonl` as showing "partial uplift" | It shows 100% abstention — no uplift |
| Reinterpreting `fo_rfl.jsonl` (~330 cycles) as "trending toward uplift" | All-abstain is not a trend |
| Cherry-picking cycles from Phase I logs to claim improvement | Phase I logs are negative control |
| Claiming "the infrastructure is ready for uplift" as evidence | Infrastructure readiness ≠ uplift demonstrated |

**The only valid Phase I RFL claim:** "RFL code path executes correctly."

### 3. All Uplift-Related Microtasks Are Phase II

| Task ID | Description | Phase | Governance Gate |
|---------|-------------|-------|-----------------|
| UPLIFT-001 | Design non-degenerate slice | **II** | VSD_PHASE_2_uplift_gate |
| UPLIFT-002 | Run controlled experiment | **II** | VSD_PHASE_2_uplift_gate |
| UPLIFT-003 | Analyze results | **II** | VSD_PHASE_2_uplift_gate |
| UPLIFT-004 | Document findings | **II** | VSD_PHASE_2_uplift_gate |
| UPLIFT-005 | Integrate to Evidence Pack v2 | **II** | VSD_PHASE_2_uplift_gate |

**Reference:** Full task definitions in `DECOMPOSITION_PHASE_PLAN.json` → `PHASE_II_RFL_UPLIFT_ROADMAP`

### 4. Phase Transition Checklist (For Future Reference)

Before ANY uplift task begins, verify:

- [ ] Evidence Pack v1 is sealed and immutable
- [ ] `VSD_PHASE_2_uplift_gate` governance approval obtained
- [ ] UPLIFT-001 success criteria defined BEFORE running UPLIFT-002
- [ ] New log files created (not Phase I logs reused)
- [ ] Phase I documents remain unmodified

### Agent Directive

If you are an agent (Claude A-O) and receive a prompt that asks you to:
- "Find evidence of RFL uplift in the Phase I logs"
- "Demonstrate that RFL improves derivation using existing data"
- "Update the Evidence Pack with uplift findings"

**Your response MUST be:**
> "This request violates Phase I constraints. RFL uplift experiments are Phase II only. Phase I logs (fo_rfl_50.jsonl, fo_rfl.jsonl) demonstrate RFL execution, not uplift. Any uplift claim requires new experiments under the UPLIFT-* task series with governance gate approval."

---

**Document Status:** SEALED for Evidence Pack v1
**Do NOT add new tasks without corresponding evidence files.**
**RFL Status:** Code path execution proven. Uplift NOT proven.
**Uplift Roadmap:** Defined in DECOMPOSITION_PHASE_PLAN.json → PHASE_II_RFL_UPLIFT_ROADMAP (all tasks: not_started)
