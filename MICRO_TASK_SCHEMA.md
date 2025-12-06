# MICRO_TASK_SCHEMA.md — Task Decomposition for MDAP Layer

**Version:** 1.0
**Status:** Active
**Protocol:** VCP 2.1 / First Organism
**Author:** Claude A (Vibe Orchestrator & Intent Compiler)

---

## Overview

This document defines the **micro-task schemas** for the MDAP (Micro-Differential Agent Protocol) layer. Each micro-task is a single, atomic operation that can be executed by a downstream agent in one pass.

**Micro-Task Properties:**
- **Atomic**: One edit, one check, one verification
- **Idempotent**: Running twice produces same result
- **Testable**: Has a clear pass/fail criterion
- **Bounded**: Completes in <5 minutes

---

## Task Type 1: AUDIT

**Purpose:** Discover all instances of a pattern without modification.

### Schema

```yaml
task_type: AUDIT
pattern: "<regex or string pattern>"
scope: "<glob pattern or directory>"
output: "<structured report format>"
criterion: "<what constitutes complete audit>"
```

### Examples

**AUDIT-001: Find all hashlib.sha256 calls**
```yaml
task_type: AUDIT
pattern: "hashlib\\.sha256\\("
scope: "**/*.py"
exclude: ".venv/**, archive/**"
output: |
  File | Line | Has Domain Tag | Context
criterion: "All occurrences catalogued with domain tag presence"
```

**AUDIT-002: Find all datetime.now() calls**
```yaml
task_type: AUDIT
pattern: "datetime\\.now\\(|time\\.time\\(|time\\.monotonic\\("
scope: "attestation/**, backend/ledger/**, backend/rfl/**"
output: |
  File | Line | In First Organism Path? | Replacement Needed?
criterion: "All timestamp sources identified"
```

**AUDIT-003: Find all imports from deprecated modules**
```yaml
task_type: AUDIT
pattern: "from backend\\.logic|from backend\\.orchestrator\\.app|from substrate"
scope: "**/*.py"
exclude: ".venv/**, archive/**"
output: |
  File | Line | Deprecated Import | Canonical Replacement
criterion: "All deprecated imports listed with migration path"
```

---

## Task Type 2: FIX

**Purpose:** Apply a single, targeted fix to one file.

### Schema

```yaml
task_type: FIX
file: "<path to file>"
problem: "<description of issue>"
solution: "<description of fix>"
before: "<code snippet before>"
after: "<code snippet after>"
test: "<how to verify fix>"
```

### Examples

**FIX-001: Add domain separation to verify_merkle.py**
```yaml
task_type: FIX
file: "tools/verify_merkle.py"
problem: "Merkle root computation missing domain tags"
solution: "Import DOMAIN_LEAF/DOMAIN_NODE from basis, apply to hashing"
before: |
  nodes = [hashlib.sha256(x).digest() for x in level]
after: |
  from basis.crypto.hash import DOMAIN_LEAF, DOMAIN_NODE
  nodes = [hashlib.sha256(DOMAIN_LEAF + x).digest() for x in level]
test: "Run tools/verify_merkle.py on known block, compare to expected root"
```

**FIX-002: Replace datetime.now() in attestation**
```yaml
task_type: FIX
file: "attestation/dual_root.py"
problem: "Nondeterministic timestamp in metadata"
solution: "Inject deterministic timestamp generator"
before: |
  "timestamp": datetime.now().isoformat()
after: |
  from backend.repro.determinism import deterministic_timestamp
  "timestamp": deterministic_timestamp(seed=ATTESTATION_SEED).isoformat()
test: "Run attestation twice, verify identical timestamps"
```

---

## Task Type 3: SHIM

**Purpose:** Create a deprecation shim that re-exports from canonical location.

### Schema

```yaml
task_type: SHIM
deprecated_module: "<path to deprecated module>"
canonical_module: "<path to canonical module>"
exports: "<list of names to re-export>"
warning_message: "<deprecation warning text>"
test: "<how to verify shim works>"
```

### Examples

**SHIM-001: backend/crypto/hashing.py → basis/crypto/hash.py**
```yaml
task_type: SHIM
deprecated_module: "backend/crypto/hashing.py"
canonical_module: "basis/crypto/hash"
exports:
  - sha256_hex
  - sha256_bytes
  - merkle_root
  - DOMAIN_LEAF
  - DOMAIN_NODE
  - DOMAIN_STMT
  - DOMAIN_BLOCK
warning_message: |
  backend.crypto.hashing is deprecated. Use basis.crypto.hash instead.
test: |
  from backend.crypto.hashing import sha256_hex
  assert sha256_hex("test") == expected_hash
  # Should emit DeprecationWarning
```

**SHIM-002: backend/logic/canon.py → normalization/canon.py**
```yaml
task_type: SHIM
deprecated_module: "backend/logic/canon.py"
canonical_module: "normalization.canon"
exports:
  - normalize
  - normalize_pretty
  - canonical_bytes
warning_message: |
  backend.logic.canon is deprecated. Use normalization.canon instead.
test: |
  from backend.logic.canon import normalize
  assert normalize("p → q") == "p->q"
  # Should emit DeprecationWarning
```

---

## Task Type 4: ARCHIVE

**Purpose:** Move code to archive directory while preserving git history.

### Schema

```yaml
task_type: ARCHIVE
source: "<path or glob to source>"
destination: "<path in archive/>"
reason: "<why archiving>"
references_to_update: "<list of files that may reference source>"
test: "<how to verify archive is complete>"
```

### Examples

**ARCHIVE-001: Move substrate/ to archive/**
```yaml
task_type: ARCHIVE
source: "substrate/"
destination: "archive/substrate/"
reason: "Parallel implementation superseded by basis/"
references_to_update:
  - "Any file importing from substrate"
test: |
  1. git mv substrate/ archive/substrate/
  2. grep -r "from substrate" --include="*.py" | should be empty
  3. pytest tests/ still passes
```

**ARCHIVE-002: Move root-level experimental scripts**
```yaml
task_type: ARCHIVE
source: |
  bootstrap_metabolism.py
  bridge.py
  phase_ix_attestation.py
  rfl_gate.py
  test_dual_attestation.py
  test_integration_v05.py
  test_integrity_audit.py
  test_migration_validation.py
  test_v05_integration.py
  verify_local_schema.py
destination: "archive/experimental/"
reason: "Root-level experimental scripts not used by First Organism"
references_to_update: []
test: |
  1. git mv <files> archive/experimental/
  2. pytest tests/ still passes
  3. Root directory is clean
```

---

## Task Type 5: CONSOLIDATE

**Purpose:** Merge multiple implementations into single canonical location.

### Schema

```yaml
task_type: CONSOLIDATE
sources:
  - "<path 1>"
  - "<path 2>"
  - "..."
destination: "<canonical path>"
strategy: "<how to merge: pick_one | union | diff_merge>"
conflicts: "<known conflicts and resolution>"
test: "<how to verify consolidation>"
```

### Examples

**CONSOLIDATE-001: Merge schema definitions**
```yaml
task_type: CONSOLIDATE
sources:
  - "backend/api/schemas.py"
  - "interface/api/schemas.py"
  - "backend/orchestrator/schemas.py"
destination: "interface/api/schemas.py"
strategy: "union"
conflicts:
  - "MetricsResponse may have different fields → use superset"
test: |
  1. Compare all schemas, create union
  2. Update imports in all files
  3. Run API tests
```

---

## Task Type 6: TEST

**Purpose:** Add or enhance test coverage for a specific module.

### Schema

```yaml
task_type: TEST
target_module: "<path to module under test>"
test_file: "<path to test file>"
coverage_goal: "<percentage or specific functions>"
test_cases:
  - name: "<test name>"
    input: "<test input>"
    expected: "<expected output>"
test: "<how to verify tests pass>"
```

### Examples

**TEST-001: Add determinism test for attestation**
```yaml
task_type: TEST
target_module: "attestation/dual_root.py"
test_file: "tests/test_dual_root_determinism.py"
coverage_goal: "100% of compute_composite_root"
test_cases:
  - name: "test_composite_root_deterministic"
    input: |
      r_t = "142a7c15..."
      u_t = "3c9a33d0..."
    expected: |
      H_t = "6a006e78..." (same every time)
  - name: "test_multiple_runs_identical"
    input: |
      Run attestation twice with same inputs
    expected: |
      Both runs produce identical H_t
test: "pytest tests/test_dual_root_determinism.py -v"
```

---

## Task Type 7: DOCUMENT

**Purpose:** Add or update documentation for a module or process.

### Schema

```yaml
task_type: DOCUMENT
target: "<path to module or doc file>"
doc_type: "<docstring | readme | spec>"
sections:
  - "<section to add/update>"
template: "<documentation template to follow>"
test: "<how to verify documentation is complete>"
```

### Examples

**DOCUMENT-001: Add docstrings to basis/crypto/hash.py**
```yaml
task_type: DOCUMENT
target: "basis/crypto/hash.py"
doc_type: "docstring"
sections:
  - "Module docstring with usage examples"
  - "sha256_hex: input/output contract"
  - "merkle_root: algorithm description"
  - "Domain tag table"
template: |
  """
  <Brief description>

  Args:
      <param>: <description>

  Returns:
      <description>

  Raises:
      <exception>: <when>

  Example:
      >>> <usage>
  """
test: "pydocstyle basis/crypto/hash.py"
```

---

## Task Type 8: VERIFY

**Purpose:** Run verification checks without making changes.

### Schema

```yaml
task_type: VERIFY
check_name: "<name of verification>"
command: "<command to run>"
success_criterion: "<what output indicates success>"
failure_action: "<what to do on failure>"
```

### Examples

**VERIFY-001: First Organism integration test**
```yaml
task_type: VERIFY
check_name: "First Organism Loop"
command: "pytest tests/integration/test_first_organism.py -v"
success_criterion: "All tests pass, H_t recomputation succeeds"
failure_action: "Block all other tasks until fixed"
```

**VERIFY-002: No deprecated imports in basis**
```yaml
task_type: VERIFY
check_name: "Basis import purity"
command: |
  grep -r "from backend\\.logic\|from substrate\|from backend\\.orchestrator" basis/
success_criterion: "No matches found"
failure_action: "Create FIX task for each violation"
```

**VERIFY-003: Hash determinism**
```yaml
task_type: VERIFY
check_name: "Hash determinism contract"
command: |
  python -c "
  from basis.crypto.hash import sha256_hex, DOMAIN_STMT
  h1 = sha256_hex('p->p', domain=DOMAIN_STMT)
  h2 = sha256_hex('p->p', domain=DOMAIN_STMT)
  assert h1 == h2, 'Hash is nondeterministic!'
  print('PASS: Hash is deterministic')
  "
success_criterion: "PASS: Hash is deterministic"
failure_action: "Critical bug - investigate immediately"
```

---

## Execution Protocol

### For MDAP Agents

1. **Receive** micro-task from queue
2. **Parse** task schema
3. **Execute** task type handler
4. **Report** result (PASS/FAIL/BLOCKED)
5. **Log** artifacts and side effects

### Task Dependencies

Tasks may have dependencies expressed as:
```yaml
depends_on:
  - "AUDIT-001"
  - "VERIFY-001"
blocks:
  - "CONSOLIDATE-001"
```

### Idempotency Contract

All tasks must be idempotent:
- Running `FIX-001` twice produces same result as once
- Running `SHIM-001` twice produces same shim (no duplicate warnings)
- Running `ARCHIVE-001` twice is safe (second run is no-op)

---

## Task Queue Priority

| Priority | Task Types | Execution Order |
|----------|------------|-----------------|
| P0 | VERIFY (First Organism) | First |
| P0 | FIX (blocking issues) | Second |
| P1 | SHIM (deprecation) | Third |
| P1 | CONSOLIDATE (imports) | Fourth |
| P2 | ARCHIVE (slop removal) | Fifth |
| P2 | DOCUMENT | Sixth |
| P3 | TEST (enhancement) | Last |

---

*This schema is the contract between Claude A (Vibe Orchestrator) and the MDAP execution layer. All micro-tasks must conform to these schemas.*
