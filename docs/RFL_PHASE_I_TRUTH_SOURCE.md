# Phase-I RFL Truth Source

**Status:** Canonical Ground Truth - Single Source of Truth  
**Last Updated:** Evidence Pack v1 Consolidation Phase  
**Authority:** Cursor A/B/L/O audits + fo_rfl.jsonl verification

## Purpose

This document is the **single source of truth** for Phase-I RFL behavior, cycle counts, and evidence files. All documentation (Cursor, Gemini, Researcher, Strategist) must align with these canonical facts.

**Do not contradict this document.** If you find inconsistencies elsewhere, update those documents to match this truth table.

---

## Logic / Verifier Configuration

- **Logic:** Propositional Logic only (PL)
- **Verifier:** Ideal truth-table + lean-disabled fallback
- **ML_ENABLE_LEAN_FALLBACK:** OFF in Phase I
- **Lean Runtime:** Never runs at runtime in Phase I
- **Abstention Behavior:** Any non-trivial candidate → truth-table miss → abstention with `method="lean-disabled"`

---

## Canonical FO / RFL Logs

### `fo_baseline.jsonl`

- **Cycles:** 1000 (0–999)
- **Schema:** Old (no top-level `status`/`method`/`abstention`)
- **Abstention:** 100% (derivation.abstained > 0 on all cycles)
- **Purpose:** Baseline negative control

### `fo_rfl_50.jsonl`

- **Actual Cycles:** 21 (0–20), **NOT 50** → **INCOMPLETE**
- **Schema:** New (`status`/`method`/`abstention` present)
- **Abstention:** 100%
- **Purpose:** Small RFL plumbing / negative control demo
- **Status:** Do not use for evidence claims; incomplete run

### `fo_rfl_1000.jsonl`

- **Actual Cycles:** 11 (0–10), **NOT 1000** → **INCOMPLETE**
- **Schema:** New
- **Abstention:** 100%
- **Purpose:** Incomplete test run
- **Status:** Do not use for any claim other than "this file exists and is incomplete"

### `fo_rfl.jsonl` ⭐

- **Cycles:** 1001 (0–1000) — **this is the big one**
- **Schema:** New (`status`/`method`/`abstention`)
- **Abstention:** 100% (all cycles abstain, `method="lean-disabled"`)
- **Purpose:** Hermetic, 1001-cycle RFL **negative control / plumbing** run
- **Uplift Signal:** **None by construction**
- **Status:** Complete run, validates RFL execution infrastructure only

---

## Manifests

- **`fo_1000_baseline/manifest.json`:** Consistent, points to 1000-cycle baseline
- **`fo_1000_rfl/manifest.json`:** Points to EMPTY file; the real partial RFL log lives under `run_20251130_import/data/fo_rfl.jsonl` (21 cycles); manifest must be marked "incomplete" and/or updated

---

## Attestation

- **`artifacts/first_organism/attestation.json`:** Exists and contains H_t, R_t, U_t
- **H_t Formula:** H_t = SHA256(R_t || U_t) (recomputed in tests)
- **Verification:** `_assert_composite_root_recomputable()` in test suite

---

## Meta-Truth: Phase I RFL Status

**Phase I has zero empirical RFL uplift.**

Every RFL log we have is 100% abstention by design. Phase I only proves that:

1. RFL plumbing works (execution infrastructure)
2. Attestation works (H_t computation and verification)
3. Determinism works (hermetic, reproducible runs)
4. File-based evidence collection works (JSONL schema validation)

**Phase I does NOT demonstrate:**
- Reduced abstention rates
- Uplift metrics
- Metabolism verification (performance improvement)
- Any form of learning or policy improvement

**Phase I RFL is:**
- Hermetic (no DB writes)
- File-based (JSONL logs only)
- Negative control (100% abstention expected)
- Infrastructure validation only

---

## Documentation Requirements

When referencing Phase-I RFL evidence:

✅ **DO:**
- Say "1001 cycles" for fo_rfl.jsonl (not "~334" or "1000")
- Say "21 cycles, incomplete" for fo_rfl_50.jsonl (not "50 cycles")
- Say "11 cycles, incomplete" for fo_rfl_1000.jsonl (not "1000 cycles")
- Label all Phase-I RFL as "hermetic negative-control / plumbing only"
- Explicitly state "100% abstention, no uplift signal"
- Use "execution infrastructure verification" not "metabolism verification"

❌ **DON'T:**
- Claim "1000-cycle RFL run" without "all abstained / negative control"
- Call any RFL log "uplift" or "metabolism verification"
- Use incomplete logs (fo_rfl_50.jsonl, fo_rfl_1000.jsonl) for evidence claims
- Suggest Phase I demonstrates performance improvement
- Mix Phase I (hermetic, file-based) with Phase II (DB-backed, uplift)

---

## Phase II vs Phase I

**Phase I (Current):**
- Hermetic execution
- File-based evidence (JSONL)
- No DB writes
- 100% abstention (negative control)
- Infrastructure validation only

**Phase II (Future):**
- DB-backed RFL
- Uplift metrics
- Metabolism verification
- Policy learning
- Reduced abstention (target)

All Phase II discussion must be explicitly tagged as "Phase II / not yet implemented".

---

**Last Updated:** Evidence Pack v1 Consolidation Phase  
**Authority:** Cursor audits + fo_rfl.jsonl verification  
**Status:** Locked - No new experiments authorized
