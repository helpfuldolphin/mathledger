# SPARK Dry-Run Harness Contract

**Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Status:** Evidence Pack v1 - Sealed

## Purpose

This document defines the **guarantees and limitations** of `scripts/dry_run_spark.py`, the SPARK dry-run test harness. It serves as the contract between the tool and its users, explicitly stating what the tool **does** and **does not** provide.

## Core Guarantee: Zero Side Effects

The dry-run harness **guarantees** that it will:

1. **Never** attempt database connections (Postgres, Redis)
2. **Never** execute test fixtures
3. **Never** run actual test code
4. **Never** modify filesystem state
5. **Never** write to databases or external services
6. **Never** trigger network requests

This guarantee is **absolute** and **non-negotiable**. The tool is designed to be safe to run in any environment without infrastructure dependencies.

## What the Dry-Run Provides

### 1. Import Validation

**Guarantee:** The tool will attempt to import `tests.integration.test_first_organism` and report success or failure.

**What this means:**
- If import succeeds: The test module is syntactically valid and all immediate dependencies are available
- If import fails: The error message indicates what is missing (module, dependency, syntax error)

**What this does NOT mean:**
- ❌ Tests will run successfully
- ❌ All runtime dependencies are satisfied
- ❌ Database schemas are correct
- ❌ Fixtures will execute without errors

### 2. Test Discovery

**Guarantee:** The tool will identify test functions marked with `@pytest.mark.first_organism` using static inspection.

**What this means:**
- Lists test function names that have the marker (function-level or module-level)
- Reports total count of test functions found
- Indicates whether module-level marker applies to all tests

**What this does NOT mean:**
- ❌ Tests are executable (fixtures may fail, dependencies may be missing)
- ❌ Tests will pass when run
- ❌ Test collection is complete (pytest may discover additional tests via plugins)
- ❌ Test markers are correctly registered in pytest configuration

### 3. Fixture Inspection

**Guarantee:** The tool will verify that key fixture functions exist in `tests.integration.conftest` without executing them.

**What this means:**
- Confirms fixture functions are defined
- Reports which expected fixtures are found vs. missing
- Verifies `EnvironmentMode` class and `detect_environment_mode()` function exist

**What this does NOT mean:**
- ❌ Fixtures will execute successfully
- ❌ Fixture dependencies are satisfied
- ❌ Database connections will work when fixtures run
- ❌ Fixture scope and lifecycle are correct

## What the Dry-Run Does NOT Provide

### 1. Test Execution Guarantees

The dry-run **cannot** tell you:
- Whether tests will pass or fail
- Whether fixtures will execute without errors
- Whether database migrations are applied
- Whether test data is available

### 2. Infrastructure Validation

The dry-run **cannot** tell you:
- Whether Postgres is reachable
- Whether Redis is available
- Whether database schemas are correct
- Whether connection strings are valid

### 3. Runtime Dependency Validation

The dry-run **cannot** tell you:
- Whether all Python dependencies are installed correctly
- Whether environment variables are set correctly
- Whether external services (Lean, etc.) are available
- Whether file permissions are correct

### 4. Test Correctness

The dry-run **cannot** tell you:
- Whether test logic is correct
- Whether assertions will pass
- Whether test data is valid
- Whether test isolation is maintained

### 5. Evidence Validation

The dry-run **cannot** tell you:
- Whether evidence files exist
- Whether evidence files are non-empty
- Whether evidence files conform to expected schemas
- Whether evidence files contain valid data

**Critical:** The dry-run harness validates **imports and test collection only**. It does **not** inspect, validate, or verify any evidence artifacts.

## Phase-I Evidence Path

The SPARK test suite produces evidence artifacts in the `results/` directory. These files are **not** validated by the dry-run harness.

### Evidence Files (Phase-I)

The following evidence files are produced by SPARK test execution:

1. **`results/fo_baseline.jsonl`**
   - Baseline experiment results
   - JSONL format (one JSON object per line)
   - Contains baseline throughput, coverage, and abstention metrics

2. **`results/fo_rfl_50.jsonl`**
   - RFL experiment results (50-run variant)
   - JSONL format (one JSON object per line)
   - Contains RFL metabolism verification data for 50-run experiments

3. **`results/fo_rfl.jsonl`**
   - RFL experiment results (full variant)
   - JSONL format (one JSON object per line)
   - Contains RFL metabolism verification data for full experiments

### Evidence Inspection

**To inspect evidence presence and validity, operators must:**

1. **Use Evidence Pack scripts** (not the dry-run harness):
   - `scripts/mk_evidence_pack.py` - Generates evidence pack with validation
   - `scripts/create_sealed_evidence_pack.py` - Creates sealed evidence pack
   - Other Evidence Pack tooling as documented

2. **Manually verify file existence:**
   ```bash
   ls -lh results/fo_*.jsonl
   ```

3. **Check file contents:**
   ```bash
   head -n 1 results/fo_baseline.jsonl  # Verify non-empty
   wc -l results/fo_*.jsonl            # Count records
   ```

**The dry-run harness does NOT:**
- ❌ Check if evidence files exist
- ❌ Validate evidence file schemas
- ❌ Verify evidence file contents
- ❌ Report evidence file status

Evidence validation is **out of scope** for the dry-run harness. Use dedicated Evidence Pack scripts for evidence inspection and validation.

## Output Formats

### Human-Readable (Default)

When run without `--json`, the tool outputs formatted text suitable for terminal viewing.

**Format:**
- Section headers with visual separators
- Status indicators (✓, ✗, ⚠)
- Next steps guidance

**Exit Codes:**
- `0`: All checks passed (import, discovery, fixtures)
- `1`: One or more checks failed

### JSON Output (`--json`)

When run with `--json`, the tool outputs structured JSON suitable for programmatic consumption.

**Schema:**
```json
{
  "success": boolean,
  "import_status": "success" | "import_error" | "error" | null,
  "test_collection": {
    "tests": [string],
    "module_has_marker": boolean,
    "total_functions": integer
  },
  "fixtures": {
    "found": [string],
    "missing": [string],
    "environment_mode_available": boolean,
    "detect_function_available": boolean
  },
  "errors": [string]
}
```

**Exit Codes:**
- `0`: All checks passed
- `1`: One or more checks failed

## Limitations and Edge Cases

### 1. Module-Level Markers

If the test module uses `pytestmark = [pytest.mark.first_organism]` at module level, **all** test functions inherit the marker. The dry-run will list all test functions in this case, but cannot distinguish between explicitly marked and inherited markers.

### 2. Dynamic Test Generation

Tests generated dynamically (e.g., via `pytest_generate_tests`) are **not** detected by static inspection. The dry-run only finds functions defined statically in the module.

### 3. Fixture Detection

Fixture detection relies on Python's `inspect` module. Fixtures defined via decorators are detected, but fixtures defined via other mechanisms may not be found. The tool uses a fallback check (`hasattr`) to catch fixtures that exist but aren't detected via decorator inspection.

### 4. Import Side Effects

While the tool avoids **intentional** side effects, importing Python modules can trigger module-level code execution. If `test_first_organism.py` or `conftest.py` contain module-level code that performs side effects (file I/O, network requests, etc.), those will execute during import.

**Mitigation:** The test modules are designed to avoid side effects at import time. If side effects occur, they are considered a bug in the test module, not the dry-run tool.

## Usage Recommendations

### When to Use

✅ **Use the dry-run when:**
- Verifying test module structure before running tests
- Checking if test markers are correctly applied
- Validating fixture definitions exist
- Debugging import errors
- CI/CD pre-flight checks (no infrastructure required)

### When NOT to Rely On

❌ **Do NOT rely on dry-run for:**
- Confirming tests will pass
- Validating infrastructure availability
- Checking database schema correctness
- Verifying test data exists
- Confirming runtime dependencies

## Evidence Pack Compliance

This contract is part of Evidence Pack v1. All claims are:

- ✅ **Verifiable:** Can be confirmed by inspecting `scripts/dry_run_spark.py`
- ✅ **Conservative:** States limitations explicitly
- ✅ **Evidence-grounded:** Based on actual implementation, not hypotheticals
- ✅ **Reviewer-2 hardened:** Written for skeptical reviewers

## Revision History

- **v1.0 (2025-01-XX):** Initial contract, Evidence Pack v1 seal

---

**Contract Seal:** This document describes the **actual behavior** of `scripts/dry_run_spark.py` as implemented. No forward-looking claims. No hypothetical capabilities. Only what exists on disk.

