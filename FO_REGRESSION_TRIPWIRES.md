# First Organism Regression Tripwires

**Status:** Evidence Pack v1 - Static Contract Verification  
**Location:** `tests/unit/test_first_organism_contract.py`  
**Purpose:** Lightweight "tripwire" tests that guarantee no one accidentally breaks SPARK's core assumptions during refactoring.

## Overview

These regression tests are **static contract checks** - they read source code and verify structural invariants without requiring database connections, Redis, or runtime execution. They act as sentinels that fail loudly if critical SPARK components are refactored away.

## Test Inventory

### 1. `test_first_organism_test_exists()`

**Contract:** The critical test function `test_first_organism_closed_loop_happy_path` must exist in `tests/integration/test_first_organism.py`.

**Verification Method:**
- Parses the test file AST
- Collects all `ast.FunctionDef` nodes
- Verifies the target function name is present

**Failure Impact:** If this test fails, the SPARK closed-loop test has been removed or renamed, breaking the certification pipeline.

**Evidence Grounding:**
- File: `tests/integration/test_first_organism.py`
- Function: `test_first_organism_closed_loop_happy_path` (line 493)

---

### 2. `test_first_organism_pass_string_present()`

**Contract:** The string `[PASS] FIRST ORGANISM ALIVE` must appear in the source files.

**Verification Method:**
- String search in `tests/integration/test_first_organism.py`
- String search in `tests/integration/conftest.py` (where `log_first_organism_pass()` is defined)

**Failure Impact:** If this test fails, the canonical certification line has been removed, breaking Cursor P's detection mechanism.

**Evidence Grounding:**
- String emitted by: `log_first_organism_pass()` in `tests/integration/conftest.py` (line 1153)
- Referenced in: Test docstrings and certification documentation

---

### 3. `test_attestation_file_path_contract()`

**Contract:** The attestation artifact path `artifacts/first_organism/attestation.json` must be referenced in the test file.

**Verification Method:**
- String search for the path in source code
- AST parsing to verify `_write_attestation_artifact()` function exists
- Extraction of function body to verify path is hardcoded

**Failure Impact:** If this test fails, the attestation artifact location has changed, breaking external certification tooling.

**Evidence Grounding:**
- Path defined in: `_write_attestation_artifact()` function (line 289)
- Used by: `test_first_organism_closed_loop_happy_path` and other FO tests

---

### 4. `test_attestation_h_t_recomputes()`

**Contract:** The test must verify that sealed attestation H_t recomputes from R_t||U_t using canonical functions.

**Verification Method:**
- Verifies imports from `attestation.dual_root`: `compute_composite_root`, `verify_composite_integrity`
- Verifies helper function `_assert_composite_root_recomputable()` exists
- Verifies helper function uses both canonical functions
- Verifies recomputability check appears in source (string search for "recomput")

**Failure Impact:** If this test fails, the H_t Invariant (H_t = SHA256(R_t || U_t)) verification has been removed or broken.

**Evidence Grounding:**
- Helper function: `_assert_composite_root_recomputable()` (line 330)
- Canonical functions: `attestation.dual_root.compute_composite_root()`, `verify_composite_integrity()`
- Called by: `test_first_organism_chain_integrity()` (line 473)

---

### 5. `test_rfl_evidence_presence_consistency()`

**Contract:** RFL evidence files must either be empty (marked incomplete) or non-empty with verifiable JSONL schema.

**Verification Method:**
- Verifies `verify_jsonl_schema()` function exists in `tools/devin_e_toolbox/artifact_verifier.py`
- Verifies empty file handling exists in `experiments/validate_fo_logs.py` (returns `{"exists": True, "empty": True}`)
- Static check only - does NOT validate actual files or assert cycle counts/uplift

**Failure Impact:** If this test fails, the validation infrastructure for RFL evidence files is missing, allowing invalid or inconsistent evidence to enter the Evidence Pack.

**Evidence Grounding:**
- Schema validator: `tools/devin_e_toolbox/artifact_verifier.py::verify_jsonl_schema()` (line 38)
- Empty file handler: `experiments/validate_fo_logs.py::validate_log()` (line 9)
- Contract: Evidence files must be either empty (marked incomplete) OR valid JSONL

**Important Constraints:**
- ❌ Does NOT assert uplift metrics
- ❌ Does NOT assert cycle counts
- ❌ Does NOT validate actual file contents (runtime concern)
- ✅ Only verifies that validation infrastructure exists

---

## Static Invariant Summary

| Invariant | Test Function | Critical Path |
|-----------|--------------|---------------|
| SPARK test exists | `test_first_organism_test_exists()` | `test_first_organism_closed_loop_happy_path` |
| Certification string present | `test_first_organism_pass_string_present()` | `log_first_organism_pass()` |
| Attestation path contract | `test_attestation_file_path_contract()` | `_write_attestation_artifact()` |
| H_t recomputability | `test_attestation_h_t_recomputes()` | `_assert_composite_root_recomputable()` |
| RFL evidence consistency | `test_rfl_evidence_presence_consistency()` | `verify_jsonl_schema()` / `validate_fo_logs.py` |

## Design Principles

1. **Static Analysis Only:** No runtime execution, no database dependencies, no external services
2. **AST-Based Verification:** Uses Python's `ast` module for robust function detection
3. **String Search Fallback:** For patterns that don't require AST parsing
4. **Clear Failure Messages:** Explains what broke and why, with context
5. **Evidence-Grounded:** Every check references actual on-disk files and line numbers

## Execution

```bash
# Run all regression tripwires
uv run pytest tests/unit/test_first_organism_contract.py -v

# Run specific test
uv run pytest tests/unit/test_first_organism_contract.py::test_attestation_h_t_recomputes -v
```

## Maintenance

**When to Update:**
- If `test_first_organism_closed_loop_happy_path` is renamed → update test 1
- If certification string format changes → update test 2
- If attestation path changes → update test 3
- If H_t recomputability check is refactored → update test 4
- If RFL evidence validation logic moves or changes → update test 5

**What NOT to Do:**
- Do not add dynamic tests that require runtime execution
- Do not add tests that require database/Redis connections
- Do not add tests that generate synthetic data
- Do not expand into full-flow integration testing (that's what `test_first_organism.py` is for)

## Reviewer-2 Hardening

These tests are designed to pass Reviewer-2 scrutiny:

✅ **No hypotheticals:** Every check references actual files and functions  
✅ **No forward-looking claims:** Only verifies what exists on disk  
✅ **No synthetic data:** Pure static analysis of source code  
✅ **Clear provenance:** Every assertion traces to specific file locations  
✅ **Minimal scope:** Only critical SPARK contracts, nothing extraneous  

## Related Documentation

- `tests/integration/test_first_organism.py` - The actual integration tests
- `tests/integration/conftest.py` - Certification logging helpers
- `attestation/dual_root.py` - Canonical H_t computation functions
- `docs/RFL_PHASE_I_TRUTH_SOURCE.md` - **Canonical Phase-I RFL truth table** (single source of truth)

---

## Phase-I RFL Alignment

These tripwire tests are **Phase-I compliant**:

✅ **No cycle count assertions** - Tripwire #5 only verifies validation infrastructure exists  
✅ **No uplift assertions** - All tests are static contract checks, not performance claims  
✅ **Evidence-grounded** - All assertions reference actual on-disk files and functions  
✅ **Hermetic** - No DB/Redis dependencies, pure static analysis  

For Phase-I RFL evidence facts (cycle counts, schema, abstention rates), see `docs/RFL_PHASE_I_TRUTH_SOURCE.md`.

---

**Last Updated:** Evidence Pack v1 Consolidation Phase  
**Mode:** Reviewer-2 / Evidence-Pack-Sealing  
**Status:** Complete - No new experiments authorized  
**Authority:** Aligned with `docs/RFL_PHASE_I_TRUTH_SOURCE.md`

