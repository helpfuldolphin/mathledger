# MDAP Micro-Agent Templates

**Version**: 1.0.0
**Last Updated**: 2025-11-25
**Applies To**: MathLedger codebase

## Overview

This document defines the MDAP (Micro-task Driven Agent Protocol) templates for automated code fixes in the MathLedger repository. Each template specifies:

1. **Prompt Template** - The exact instructions given to micro-agents
2. **Output Schema** - JSON schema for structured, deterministic output
3. **Red-Flag Criteria** - Conditions that invalidate an agent's output
4. **Validators** - Commands to verify correctness post-application
5. **MDAP Config** - Sampling and voting parameters for consensus

---

## Design Principles

### Layer 1: Vibe + Constraints

All micro-agent prompts adhere to these invariants:

- **Assume everything is broken** until proven otherwise by validators
- **Minimal edits only**: no multi-function rewrites in a single micro-step
- **Deterministic output**: JSON or structured diff, no timestamps/UUIDs
- **Single responsibility**: one fix type per micro-agent invocation
- **Fail-safe defaults**: when uncertain, output `"action": "skip"`

### Layer 2: Task Decomposition

Each micro-task type is atomic:
- One file
- One logical change
- One validation cycle

Complex refactors are decomposed into chains of micro-tasks with explicit dependencies.

### Layer 3: MDAP Integration

For consensus voting (first-to-ahead-by-k):
- Run `sample_count` independent samples at temperature 0.0
- Compare outputs on `comparison_keys` (exact structural equality)
- Accept if `consensus_threshold` samples agree
- Retry up to `max_retries` on red-flag failures

---

## Micro-Task Types

### 1. hash_normalization_fix

**Category**: `cryptographic_integrity`

**Purpose**: Route statement hashing through the canonical `hash_statement()` function which applies domain separation (`DOMAIN_STMT = 0x02`) and normalization via `canonical_bytes()`.

#### Prompt Template

```
System Context:
You are a micro-agent applying a single, targeted fix to MathLedger code. All
statement hashing MUST use the canonical `hash_statement()` function from
`backend.crypto.hashing` which applies domain separation (DOMAIN_STMT = 0x02)
and normalization.

Task:
Locate the hash call at {{file}}:{{line_number}} and replace the direct SHA-256
call with the canonical helper.

Constraints:
- Change at most ONE line of code
- Do NOT modify imports in this step (separate micro-task)
- Do NOT change function signatures
- Preserve indentation exactly
- The replacement must use `hash_statement(normalized_text)` from
  `backend.crypto.hashing`
```

#### Output Schema

```json
{
  "patch_type": "single_line_edit",
  "file": "backend/crypto/hashing.py",
  "line_number": 123,
  "old_line_fragment": "sha256_hex(",
  "new_line": "    result = hash_statement(normalized_text)",
  "explanation": "Routed statement hashing through canonical helper with domain separation."
}
```

| Field | Type | Constraints |
|-------|------|-------------|
| `patch_type` | string | Must be `"single_line_edit"` |
| `file` | string | Pattern: `^(backend\|tests)/.*\.py$` |
| `line_number` | integer | >= 1 |
| `old_line_fragment` | string | <= 200 chars |
| `new_line` | string | <= 300 chars, no newlines |
| `explanation` | string | <= 500 chars |

#### Red-Flag Criteria

| Rule | Description |
|------|-------------|
| `invalid_json` | Output is not valid JSON |
| `missing_required_field` | Any required field is missing |
| `output_too_long` | explanation > 500 chars or new_line > 300 chars |
| `file_outside_scope` | file does not match allowed pattern |
| `multi_line_edit` | new_line contains newline characters |
| `non_deterministic_content` | Contains timestamps, UUIDs, random values |
| `import_modification` | new_line modifies an import statement |
| `missing_domain_tag` | Replacement does not use domain-separated hashing |

#### Validators

```bash
# Must pass:
pytest tests/test_canon.py -v
pytest tests/test_hashing.py -v -k hash_statement
python -c "from backend.crypto.hashing import hash_statement; print('OK')"
```

#### MDAP Config

- **Samples**: 5
- **Consensus**: 3 agreeing
- **Comparison Keys**: `["file", "line_number", "new_line"]`
- **Temperature**: 0.0
- **Max Retries**: 2

---

### 2. import_modernization_fix

**Category**: `code_hygiene`

**Purpose**: Replace deprecated module imports with new canonical paths following the module reorganization.

#### Deprecation Map

| Old Path | New Path |
|----------|----------|
| `backend.logic.canon` | `normalization.canon` |
| `backend.logic.taut` | `normalization.taut` |
| `backend.orchestrator.app` | `interface.api.app` |

#### Prompt Template

```
System Context:
You are a micro-agent updating deprecated imports. The following modules have
been moved:
- `backend.logic.canon` -> `normalization.canon`
- `backend.logic.taut` -> `normalization.taut`
- `backend.orchestrator.app` -> `interface.api.app`

Task:
Update the import statement at {{file}}:{{line_number}}.

Constraints:
- Change ONLY the import path, not the imported names
- Preserve aliasing (e.g., `as normalize`)
- Preserve relative vs absolute import style if semantically equivalent
- Do NOT add new imports
- Do NOT remove existing imports
```

#### Output Schema

```json
{
  "patch_type": "single_line_edit",
  "file": "tests/test_canon.py",
  "line_number": 5,
  "old_line_fragment": "from backend.logic.canon import",
  "new_line": "from normalization.canon import normalize, canonical_bytes",
  "explanation": "Updated import to new canonical path."
}
```

#### Red-Flag Criteria

| Rule | Description |
|------|-------------|
| `import_path_unknown` | New import path is not in the known modernization map |
| `changed_imported_names` | The imported symbols changed, not just the path |
| `removed_import` | The import was removed instead of updated |

#### Validators

```bash
python -c "import {{new_module_path}}"
python -m py_compile {{file}}
```

#### MDAP Config

- **Samples**: 3
- **Consensus**: 2 agreeing
- **Comparison Keys**: `["file", "line_number", "new_line"]`

---

### 3. schema_column_fallback_extraction

**Category**: `code_deduplication`

**Purpose**: Extract repeated schema column fallback chains into reusable helper functions.

#### Problem Pattern

The codebase has repeated patterns like:

```python
normalized = (
    data.get('normalized_text')
    or data.get('content_norm')
    or data.get('statement_norm')
    or data.get('text')
)
```

These should be extracted to helpers in `backend.axiom_engine.derive_utils`.

#### Helper Functions

| Pattern | Helper |
|---------|--------|
| Normalized text fallback | `get_normalized_text(data)` |
| Success flag coercion | `normalize_success_flag(value)` |
| Display text coalescing | `coalesce_display_text(*candidates)` |

#### Output Schema

```json
{
  "patch_type": "multi_line_collapse",
  "file": "backend/axiom_engine/derive_core.py",
  "line_start": 64,
  "line_end": 70,
  "old_lines": [
    "normalized = (",
    "    data.get('normalized_text')",
    "    or data.get('content_norm')",
    "    or data.get('statement_norm')",
    "    or data.get('text')",
    ")"
  ],
  "new_line": "normalized = get_normalized_text(data)",
  "helper_used": "get_normalized_text",
  "explanation": "Extracted repeated fallback chain to reusable helper."
}
```

#### Red-Flag Criteria

| Rule | Description |
|------|-------------|
| `line_range_invalid` | line_end < line_start or range > 10 lines |
| `unknown_helper` | helper_used not in allowed list |
| `logic_change` | The semantics of the fallback chain were altered |

---

### 4. success_flag_normalization

**Category**: `schema_tolerance`

**Purpose**: Replace inline success flag coercion with canonical `_normalize_success_flag()` helper.

#### Canonical Function

```python
def _normalize_success_flag(value: Any) -> Optional[bool]:
    """Best-effort coercion of heterogeneous success indicators into booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "1", "success", "ok", "passed", "pass"}:
            return True
        if lowered in {"false", "f", "0", "fail", "failed", "error"}:
            return False
    return None
```

#### Output Schema

```json
{
  "patch_type": "single_line_edit",
  "file": "interface/api/app.py",
  "line_number": 507,
  "old_line_fragment": "success=payload.get('success') == True",
  "new_line": "success=_normalize_success_flag(payload.get('success')),",
  "requires_import": false,
  "explanation": "Replaced inline bool coercion with canonical helper."
}
```

The `requires_import` field signals whether a follow-up `add_import_statement` micro-task is needed.

---

### 5. domain_tag_addition

**Category**: `cryptographic_integrity`

**Purpose**: Add missing domain separation tags to hash operations to prevent second preimage attacks (CVE-2012-2459 pattern).

#### Domain Tags

| Tag | Value | Usage |
|-----|-------|-------|
| `DOMAIN_LEAF` | `0x00` | Merkle tree leaf nodes |
| `DOMAIN_NODE` | `0x01` | Merkle tree internal nodes |
| `DOMAIN_STMT` | `0x02` | Statement content hashing |
| `DOMAIN_BLCK` | `0x03` | Block header hashing |

#### Prompt Template

```
Context analysis:
- Is this hashing a statement? Use DOMAIN_STMT
- Is this hashing a block? Use DOMAIN_BLCK
- Is this a Merkle leaf? Use DOMAIN_LEAF
- Is this a Merkle internal node? Use DOMAIN_NODE

Constraints:
- Add ONLY the domain= parameter
- Do NOT change the data being hashed
- Import the domain constant if not present (flag in output)
- Preserve all other parameters
```

#### Output Schema

```json
{
  "patch_type": "single_line_edit",
  "file": "backend/ledger/blockchain.py",
  "line_number": 45,
  "old_line_fragment": "sha256_hex(block_data)",
  "new_line": "sha256_hex(block_data, domain=DOMAIN_BLCK)",
  "domain_tag": "DOMAIN_BLCK",
  "requires_import": true,
  "explanation": "Added block domain tag for second preimage resistance."
}
```

#### Red-Flag Criteria

| Rule | Description |
|------|-------------|
| `wrong_domain_tag` | Domain tag does not match the semantic context |
| `data_changed` | The data being hashed was modified |
| `unknown_domain` | domain_tag not in the allowed enum |

---

### 6. curriculum_gate_param_fix

**Category**: `curriculum_control`

**Purpose**: Correct curriculum gate threshold parameters that fail validation.

#### Gate Constraints

| Parameter | Type | Valid Range |
|-----------|------|-------------|
| `coverage.ci_lower_min` | float | (0, 1] |
| `coverage.sample_min` | int | > 0 |
| `abstention.max_rate_pct` | float | [0, 100] |
| `abstention.max_mass` | int | > 0 |
| `velocity.min_pph` | float | > 0 |
| `velocity.stability_cv_max` | float | [0, 1] |
| `velocity.window_minutes` | int | > 0 |
| `caps.min_attempt_mass` | int | > 0 |
| `caps.min_runtime_minutes` | float | > 0 |
| `caps.backlog_max` | float | [0, 1] |

#### Output Schema

```json
{
  "patch_type": "value_edit",
  "file": "config/curriculum.yaml",
  "line_number": 42,
  "parameter_path": "coverage.ci_lower_min",
  "old_value": 1.5,
  "new_value": 0.92,
  "explanation": "Corrected ci_lower_min to valid range (0, 1]."
}
```

#### Validators

```bash
python -c "from curriculum.gates import SliceGates; ..."
pytest tests/frontier/test_curriculum_gates.py -v
```

---

### 7. rfl_runner_contract_update

**Category**: `interface_contract`

**Purpose**: Update RFL runner method signatures to match new interface contracts.

#### Expected Contracts

```python
# RFLRunner
def run_single(self, run_id: int) -> ExperimentResult
def run_with_attestation(self, ctx: AttestedRunContext) -> RflResult

# RFLExperiment
def derive(self, config: Dict[str, Any]) -> ExperimentResult
```

#### Output Schema

```json
{
  "patch_type": "signature_edit",
  "file": "rfl/runner.py",
  "line_number": 156,
  "method_name": "run_with_attestation",
  "old_signature": "def run_with_attestation(self, ctx):",
  "new_signature": "def run_with_attestation(self, ctx: AttestedRunContext) -> RflResult:",
  "explanation": "Added type annotations matching contract."
}
```

#### Red-Flag Criteria

| Rule | Description |
|------|-------------|
| `body_modified` | Method body was changed, not just signature |
| `decorator_removed` | A decorator was removed |
| `contract_mismatch` | New signature does not match expected contract |

---

### 8. api_schema_alignment

**Category**: `api_contract`

**Purpose**: Align API response construction with Pydantic schema definitions.

#### Schema Rules

- All models inherit from `ApiModel` with `extra="forbid"`
- `HexDigest` type: `constr(pattern=r'^[0-9a-f]{64}$')`
- No extra fields allowed on responses

#### Output Schema

```json
{
  "patch_type": "response_alignment",
  "file": "interface/api/app.py",
  "line_number": 634,
  "schema_name": "StatementDetailResponse",
  "field_changes": [
    {"field": "canonical_hash", "action": "rename", "from": "canonical_hash", "to": "hash"},
    {"field": "pretty_text", "action": "remove", "from": "pretty_text", "to": null}
  ],
  "new_code": "return StatementDetailResponse(\n    hash=stmt['hash'],\n    ...\n)",
  "explanation": "Aligned response with StatementDetailResponse schema."
}
```

---

### 9. hex64_validation_addition

**Category**: `input_validation`

**Purpose**: Add 64-character hex validation to hash parameters before use.

#### Validation Patterns

**API Context**:
```python
if not HEX64.match(hash or ''):
    raise HTTPException(status_code=400, detail='Invalid hash format')
```

**Internal Context**:
```python
if not re.match(r'^[a-f0-9]{64}$', hash or ''):
    raise ValueError('Invalid hash format')
```

#### Output Schema

```json
{
  "patch_type": "validation_insertion",
  "file": "interface/api/app.py",
  "line_number": 582,
  "validation_type": "api",
  "new_lines": [
    "if not HEX64.match(hash or ''):",
    "    raise HTTPException(status_code=400, detail='Invalid hash format')"
  ],
  "explanation": "Added hash format validation before DB query."
}
```

---

### 10. test_import_fix

**Category**: `test_hygiene`

**Purpose**: Fix broken test imports after module reorganization.

This is essentially `import_modernization_fix` scoped specifically to test files, with additional validation that test collection still works.

#### Validators

```bash
python -m py_compile {{file}}
pytest {{file}} -v --collect-only  # Verify tests are discoverable
```

---

### 11. add_import_statement

**Category**: `code_hygiene`

**Purpose**: Add a missing import statement following PEP 8 ordering.

#### Import Categories

1. **Standard Library**: `os`, `sys`, `re`, `json`, `hashlib`, `typing`
2. **Third-Party**: `psycopg`, `redis`, `fastapi`, `pydantic`, `numpy`
3. **Local**: `backend`, `interface`, `normalization`, `curriculum`, `rfl`

#### Output Schema

```json
{
  "patch_type": "import_addition",
  "file": "backend/axiom_engine/derive_core.py",
  "insert_after_line": 12,
  "import_statement": "from backend.crypto.hashing import hash_statement",
  "category": "local",
  "explanation": "Added import for hash_statement used at line 76."
}
```

---

## Global Red Flags

These apply to ALL micro-task types:

| Rule | Severity | Description |
|------|----------|-------------|
| `output_contains_secrets` | CRITICAL | Output contains API keys, passwords, or credentials |
| `output_contains_pii` | CRITICAL | Output contains personally identifiable information |
| `contains_eval_or_exec` | CRITICAL | Output contains `eval()` or `exec()` calls |
| `non_utf8_content` | HIGH | Output contains non-UTF-8 characters |
| `output_too_large` | MEDIUM | Total output exceeds 10KB |

---

## MDAP Global Configuration

```json
{
  "default_sample_count": 5,
  "default_consensus_threshold": 3,
  "max_parallel_samples": 10,
  "timeout_per_sample_ms": 30000,
  "retry_on_red_flag": true,
  "log_all_samples": true,
  "comparison_equality_mode": "structural",
  "determinism_seed": 0
}
```

### Voting Strategy: First-to-Ahead-by-K

The MDAP uses **first-to-ahead-by-k** voting:

1. Generate `sample_count` independent outputs at temperature 0.0
2. Extract `comparison_keys` from each output
3. Group outputs by structural equality of comparison keys
4. If any group has >= `consensus_threshold` members, accept that output
5. Otherwise, retry up to `max_retries` times
6. If still no consensus, escalate to human review

### Comparison Equality

Two outputs are equal if their `comparison_keys` match exactly using structural comparison:
- Strings: exact match
- Numbers: exact match
- Arrays: element-wise comparison
- Objects: key-value comparison

---

## Usage Example

### Invoking a Micro-Agent

```python
from ops.microagents import MicroAgentRunner

runner = MicroAgentRunner(template="hash_normalization_fix")

result = runner.execute({
    "file": "backend/ledger/blocking.py",
    "line_number": 45,
    "code_context": "hash_val = sha256_hex(statement_text)",
    "error_message": None
})

if result.consensus_reached:
    apply_patch(result.patch)
else:
    escalate_to_human(result.samples)
```

### Chaining Micro-Tasks

For compound fixes requiring multiple steps:

```python
# Step 1: Fix the hash call
hash_fix = runner.execute(template="hash_normalization_fix", ...)

# Step 2: Add the import (if needed)
if hash_fix.patch.get("requires_import"):
    import_fix = runner.execute(template="add_import_statement", {
        "file": hash_fix.patch["file"],
        "symbol": "hash_statement",
        "canonical_import": "from backend.crypto.hashing import hash_statement",
        "usage_line": hash_fix.patch["line_number"]
    })
```

---

## Appendix: File Locations

| Artifact | Path |
|----------|------|
| Template Catalog | `ops/microagents/templates.json` |
| This Documentation | `docs/MDAP_TEMPLATES.md` |
| Validation Scripts | `ops/microagents/validators/` |
| Sample Outputs | `ops/microagents/samples/` |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-25 | Initial template definitions |
