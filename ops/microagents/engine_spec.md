# MDAP Execution Engine Specification

> **STATUS: PHASE II TOOLING ONLY**
>
> This specification describes future infrastructure. It is **not implemented**
> and was **not used** in any Phase I experiments or Evidence Pack v1.

**Version:** 1.0.0
**Status:** Draft / Phase II Design Document
**Last Updated:** 2025-11-30

---

## Usage in Evidence Pack v1

**This engine has not yet been implemented or wired into FO/RFL runs.**

Evidence Pack v1 claims do not depend on MDAP micro-agent execution. The Phase I
experiments (First Organism closed-loop test, 1000-cycle Dyno runs) were conducted
using the existing `fo_harness.py` and `rfl/runner.py` infrastructure—not MDAP.

This document exists solely as a design specification for potential Phase II
tooling. No code in this repository currently executes the pipeline described
below.

### Clarification Regarding RFL Logs

New RFL logs (e.g., `fo_rfl.jsonl`) may exist on disk from Phase I experiments.
However:

- **MDAP does not consume these logs.** No code path reads RFL output and feeds
  it into any micro-agent template or consensus voting system.
- **Phase I uses no micro-agents**, regardless of log contents or cycle counts.
- The existence of additional RFL cycles does not activate any MDAP engine or
  imply any MDAP readiness.

RFL logs are produced by `rfl/runner.py` for human review and future analysis.
They are not inputs to this (unimplemented) MDAP system.

---

## 1. Overview

The MDAP (Micro-Deterministic Agent Protocol) Execution Engine orchestrates the execution of micro-agent templates defined in `templates.json`. It implements multi-sample consensus voting to ensure deterministic, verifiable code transformations.

### 1.1 Design Principles

1. **Determinism**: Given identical inputs, the engine produces identical outputs
2. **Auditability**: Every execution step is logged with cryptographic provenance
3. **Fail-Safe**: Red flags halt execution; no partial application of changes
4. **Consensus**: Multiple samples must agree before any change is applied

---

## 2. Control Flow

### 2.1 High-Level Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MDAP EXECUTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  LOAD    │──▶│  PREPARE  │──▶│ SAMPLE   │──▶│ CONSENSUS│──▶│ VALIDATE │ │
│  │ Template │   │  Context  │   │ Generate │   │  Vote    │   │  & Apply │ │
│  └──────────┘   └───────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │              │               │              │              │        │
│       ▼              ▼               ▼              ▼              ▼        │
│   [Parse &      [Read file,     [N parallel   [Structural    [Run shell    │
│    Validate]     substitute]     LLM calls]    equality]      validators]  │
│                                                                             │
│                              ┌──────────┐                                   │
│                              │  AUDIT   │◀─────────────────────────────────│
│                              │  JOURNAL │                                   │
│                              └──────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed Phase Descriptions

#### Phase 1: LOAD Template

**Input:** Task type identifier (e.g., `hash_normalization_fix`)

**Steps:**
1. Load `templates.json` from disk
2. Validate against JSON Schema (`$schema` field)
3. Extract the micro-task definition by ID
4. Verify all required fields present:
   - `prompt_template`
   - `output_schema`
   - `red_flag_rules`
   - `validators`
   - `mdap_config`

**Output:** Validated `MicroTaskTemplate` object

**Failure Modes:**
- Template not found → `TemplateNotFoundError`
- Schema validation fails → `SchemaValidationError`
- Missing required field → `IncompleteTemplateError`

---

#### Phase 2: PREPARE Context

**Input:** `MicroTaskTemplate` + task parameters (file, line_number, etc.)

**Steps:**
1. Read the target file from disk
2. Extract code context (surrounding lines)
3. Substitute template variables (`{{file}}`, `{{line_number}}`, etc.)
4. Build the full prompt:
   ```
   [system_context]

   [task_instruction with substitutions]

   Output JSON matching this schema:
   [output_schema]

   Red flags that will cause rejection:
   [red_flag_rules]
   ```

**Output:** `PreparedPrompt` object with:
- Full prompt text
- Original file content
- Substitution map

**Failure Modes:**
- File not found → `FileNotFoundError`
- Line number out of range → `LineRangeError`
- Unresolved template variable → `SubstitutionError`

---

#### Phase 3: SAMPLE Generate

**Input:** `PreparedPrompt` + `mdap_config`

**Steps:**
1. Determine sample count from config (default: 5)
2. Set temperature (typically 0.0 for determinism)
3. Launch N parallel LLM calls with identical prompts
4. Parse each response as JSON
5. Apply red flag checks to each sample
6. Collect valid samples, reject flagged ones

**Concurrency Model:**
```python
async def generate_samples(prompt, n, temperature):
    tasks = [llm_call(prompt, temp=temperature) for _ in range(n)]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Output:** `SampleSet` containing:
- List of parsed JSON responses
- List of red flag violations (if any)
- Sample generation timestamps

**Failure Modes:**
- All samples fail red flag checks → `AllSamplesRejectedError`
- LLM timeout → `SampleTimeoutError`
- JSON parse failure → sample marked as invalid

---

#### Phase 4: CONSENSUS Vote

**Input:** `SampleSet` + `mdap_config`

**Steps:**
1. Extract comparison keys from each sample
2. Group samples by structural equality
3. Apply voting strategy
4. Check if winner meets consensus threshold

**Voting Strategies:**

##### Strategy: `first_to_ahead_by_k`
```python
def first_to_ahead_by_k(groups, k):
    """
    Returns the first group to reach k votes ahead of the next-best.
    Process samples in order; first to achieve lead wins.
    """
    sorted_groups = sorted(groups, key=lambda g: len(g), reverse=True)
    if len(sorted_groups) == 1:
        return sorted_groups[0] if len(sorted_groups[0]) >= k else None

    leader = sorted_groups[0]
    runner_up = sorted_groups[1]

    if len(leader) - len(runner_up) >= 1 and len(leader) >= consensus_threshold:
        return leader
    return None
```

##### Strategy: `simple_majority`
```python
def simple_majority(groups, threshold):
    """
    Returns the group with >= threshold votes.
    """
    for group in groups:
        if len(group) >= threshold:
            return group
    return None
```

##### Strategy: `unanimous`
```python
def unanimous(groups, sample_count):
    """
    All samples must agree exactly.
    """
    if len(groups) == 1 and len(groups[0]) == sample_count:
        return groups[0]
    return None
```

**Output:** `ConsensusResult` containing:
- Winning sample (or None)
- Vote distribution
- Consensus achieved (bool)

**Failure Modes:**
- No consensus reached → `NoConsensusError`
- Tie between groups → `ConsensusTieError`

---

#### Phase 5: VALIDATE & Apply

**Input:** `ConsensusResult` + validators from template

**Steps:**
1. **Pre-application validation:**
   - Re-verify output schema compliance
   - Check file still exists and hasn't changed

2. **Apply the patch:**
   - For `single_line_edit`: Replace line at specified position
   - For `multi_line_collapse`: Replace line range
   - For `import_addition`: Insert after specified line
   - For `validation_insertion`: Insert before specified line

3. **Run validators:**
   ```python
   for validator in validators:
       result = subprocess.run(validator['command'], shell=True)
       if result.returncode != 0:
           if validator['on_failure'] == 'reject':
               rollback()
               raise ValidationFailedError(validator)
           elif validator['on_failure'] == 'warn':
               log_warning(validator)
   ```

4. **Post-validation:**
   - Verify file is syntactically valid
   - Run any specified tests

**Output:** `ApplicationResult` containing:
- Success/failure status
- Applied diff
- Validator results

**Failure Modes:**
- File changed during execution → `ConcurrentModificationError`
- Validator failed with `on_failure=reject` → `ValidationFailedError`
- Patch application failed → `PatchFailedError`

---

## 3. Structural Equality

### 3.1 Definition

Two samples are **structurally equal** if, when compared on the `comparison_keys` specified in `mdap_config`, they produce identical normalized values.

### 3.2 Normalization Rules

Before comparison, values are normalized:

1. **Strings:**
   - Strip leading/trailing whitespace
   - Normalize internal whitespace (collapse multiple spaces)
   - Normalize line endings to `\n`

2. **Numbers:**
   - Compare as floating point with epsilon tolerance (1e-9)

3. **Arrays:**
   - Order-sensitive by default
   - For unordered comparison, sort before comparing

4. **Objects:**
   - Compare only keys listed in `comparison_keys`
   - Ignore all other keys

### 3.3 Comparison Algorithm

```python
def structural_equal(sample_a, sample_b, comparison_keys):
    """
    Compare two samples for structural equality.

    Args:
        sample_a: First sample (parsed JSON dict)
        sample_b: Second sample (parsed JSON dict)
        comparison_keys: List of keys to compare

    Returns:
        bool: True if structurally equal
    """
    for key in comparison_keys:
        val_a = normalize_value(extract_nested(sample_a, key))
        val_b = normalize_value(extract_nested(sample_b, key))

        if val_a != val_b:
            return False
    return True

def extract_nested(obj, key_path):
    """
    Extract value from nested dict using dot notation.
    e.g., 'field_changes.0.action' -> obj['field_changes'][0]['action']
    """
    parts = key_path.split('.')
    current = obj
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = current.get(part)
        if current is None:
            return None
    return current

def normalize_value(value):
    """
    Normalize a value for comparison.
    """
    if isinstance(value, str):
        return ' '.join(value.split()).strip()
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    return value
```

### 3.4 Grouping Algorithm

```python
def group_by_equality(samples, comparison_keys):
    """
    Group samples by structural equality.

    Returns list of groups, where each group contains
    structurally equal samples.
    """
    groups = []

    for sample in samples:
        matched = False
        for group in groups:
            if structural_equal(sample, group[0], comparison_keys):
                group.append(sample)
                matched = True
                break

        if not matched:
            groups.append([sample])

    return groups
```

---

## 4. Validator Execution

### 4.1 Validator Definition

Each validator in the template has the structure:
```json
{
  "command": "shell command with {{substitutions}}",
  "on_failure": "reject" | "warn"
}
```

### 4.2 Substitution Variables

Available variables in validator commands:
- `{{file}}` - Path to the modified file
- `{{new_module_path}}` - For import validators
- `{{schema_name}}` - For API schema validators
- `{{import_statement}}` - For import addition validators

### 4.3 Execution Environment

Validators run in a subprocess with:
- Working directory: Repository root
- Timeout: 60 seconds (configurable)
- Environment: Inherit from parent process
- Shell: `cmd.exe` on Windows, `/bin/sh` on Unix

### 4.4 Result Interpretation

| Exit Code | Interpretation |
|-----------|----------------|
| 0         | Success        |
| Non-zero  | Failure        |

### 4.5 Failure Handling

```python
def run_validators(validators, context):
    """
    Execute validators in sequence.

    Args:
        validators: List of validator definitions
        context: Substitution context

    Returns:
        ValidatorResult with status and details

    Raises:
        ValidationFailedError: If reject-mode validator fails
    """
    results = []

    for validator in validators:
        command = substitute(validator['command'], context)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                timeout=60,
                text=True
            )

            success = result.returncode == 0
            results.append(ValidatorResult(
                command=command,
                success=success,
                stdout=result.stdout,
                stderr=result.stderr
            ))

            if not success:
                if validator['on_failure'] == 'reject':
                    raise ValidationFailedError(
                        command=command,
                        stderr=result.stderr
                    )
                else:  # warn
                    log_warning(f"Validator warning: {command}")

        except subprocess.TimeoutExpired:
            raise ValidatorTimeoutError(command)

    return results
```

---

## 5. Logging and Audit Journal

### 5.1 Journal Format

The audit journal uses JSONL (JSON Lines) format, one entry per line:

```
logs/mdap_audit_{date}_{run_id}.jsonl
```

### 5.2 Entry Types

#### 5.2.1 Run Start Entry
```json
{
  "type": "run_start",
  "timestamp": "2025-11-29T10:30:00.000Z",
  "run_id": "uuid-v4",
  "task_type": "hash_normalization_fix",
  "input_hash": "sha256-of-input-params",
  "template_version": "1.0.0",
  "mdap_config": {
    "sample_count": 5,
    "consensus_threshold": 3,
    "voting_strategy": "first_to_ahead_by_k",
    "temperature": 0.0
  }
}
```

#### 5.2.2 Context Prepared Entry
```json
{
  "type": "context_prepared",
  "timestamp": "2025-11-29T10:30:00.100Z",
  "run_id": "uuid-v4",
  "file": "backend/ledger/blockchain.py",
  "line_number": 42,
  "file_hash": "sha256-of-file-content",
  "prompt_hash": "sha256-of-prepared-prompt"
}
```

#### 5.2.3 Sample Generated Entry
```json
{
  "type": "sample_generated",
  "timestamp": "2025-11-29T10:30:01.500Z",
  "run_id": "uuid-v4",
  "sample_index": 0,
  "response_hash": "sha256-of-response",
  "response_length": 412,
  "parse_success": true,
  "red_flags": [],
  "comparison_key_values": {
    "file": "backend/ledger/blockchain.py",
    "line_number": 42,
    "new_line": "hash_value = hash_statement(normalized)"
  }
}
```

#### 5.2.4 Sample Rejected Entry
```json
{
  "type": "sample_rejected",
  "timestamp": "2025-11-29T10:30:01.600Z",
  "run_id": "uuid-v4",
  "sample_index": 2,
  "response_hash": "sha256-of-response",
  "red_flags": ["multi_line_edit", "output_too_long"],
  "rejection_reason": "Red flag rules violated"
}
```

#### 5.2.5 Consensus Entry
```json
{
  "type": "consensus",
  "timestamp": "2025-11-29T10:30:02.000Z",
  "run_id": "uuid-v4",
  "achieved": true,
  "strategy": "first_to_ahead_by_k",
  "vote_distribution": [
    {"group_hash": "abc123", "count": 4},
    {"group_hash": "def456", "count": 1}
  ],
  "winning_group_hash": "abc123",
  "winning_sample_index": 0
}
```

#### 5.2.6 Validation Entry
```json
{
  "type": "validation",
  "timestamp": "2025-11-29T10:30:03.000Z",
  "run_id": "uuid-v4",
  "validator_index": 0,
  "command": "pytest tests/test_canon.py -v",
  "exit_code": 0,
  "duration_ms": 850,
  "on_failure": "reject",
  "passed": true
}
```

#### 5.2.7 Patch Applied Entry
```json
{
  "type": "patch_applied",
  "timestamp": "2025-11-29T10:30:04.000Z",
  "run_id": "uuid-v4",
  "file": "backend/ledger/blockchain.py",
  "patch_type": "single_line_edit",
  "line_number": 42,
  "old_content_hash": "sha256-of-old-line",
  "new_content_hash": "sha256-of-new-line",
  "file_hash_before": "sha256-before",
  "file_hash_after": "sha256-after"
}
```

#### 5.2.8 Run Complete Entry
```json
{
  "type": "run_complete",
  "timestamp": "2025-11-29T10:30:05.000Z",
  "run_id": "uuid-v4",
  "success": true,
  "total_duration_ms": 5000,
  "samples_generated": 5,
  "samples_valid": 4,
  "consensus_achieved": true,
  "validators_passed": 3,
  "validators_warned": 0,
  "patch_applied": true
}
```

#### 5.2.9 Error Entry
```json
{
  "type": "error",
  "timestamp": "2025-11-29T10:30:02.500Z",
  "run_id": "uuid-v4",
  "error_type": "NoConsensusError",
  "error_message": "No consensus reached after 5 samples",
  "phase": "consensus",
  "recoverable": false,
  "stack_trace": "..."
}
```

### 5.3 Journal Integrity

Each journal file includes:
- Opening entry with file hash of previous journal (chain linking)
- Closing entry with SHA-256 hash of all entries
- Entries are append-only; no modification after write

```python
def finalize_journal(journal_path):
    """
    Compute and append the journal integrity hash.
    """
    hasher = hashlib.sha256()

    with open(journal_path, 'rb') as f:
        for line in f:
            hasher.update(line)

    integrity_entry = {
        "type": "journal_integrity",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "entry_count": count_entries(journal_path),
        "content_hash": hasher.hexdigest()
    }

    with open(journal_path, 'a') as f:
        f.write(json.dumps(integrity_entry) + '\n')
```

---

## 6. Error Handling and Recovery

### 6.1 Error Categories

| Category | Examples | Recovery |
|----------|----------|----------|
| **Retriable** | LLM timeout, network error | Retry with backoff |
| **Reconfigurable** | No consensus | Increase sample count |
| **Fatal** | Template not found | Abort with error |
| **Security** | Red flag: secrets detected | Abort, alert |

### 6.2 Retry Policy

```python
@dataclass
class RetryPolicy:
    max_retries: int = 2
    backoff_base_ms: int = 1000
    backoff_max_ms: int = 30000
    retriable_errors: List[str] = field(default_factory=lambda: [
        'SampleTimeoutError',
        'NetworkError',
        'RateLimitError'
    ])

def should_retry(error, policy, attempt):
    if attempt >= policy.max_retries:
        return False
    return type(error).__name__ in policy.retriable_errors
```

### 6.3 Rollback Procedure

If validation fails after patch application:

1. Read backup from `.mdap_backup/{file}.{run_id}.bak`
2. Restore original file content
3. Log rollback in audit journal
4. Raise `ValidationFailedError` with details

```python
def rollback(file_path, run_id):
    backup_path = f".mdap_backup/{file_path}.{run_id}.bak"

    if not os.path.exists(backup_path):
        raise RollbackError(f"Backup not found: {backup_path}")

    shutil.copy(backup_path, file_path)

    log_entry({
        "type": "rollback",
        "timestamp": now(),
        "run_id": run_id,
        "file": file_path,
        "reason": "validation_failed"
    })
```

---

## 7. Concurrency Model

### 7.1 Sample Generation

Samples are generated in parallel using `asyncio`:

```python
async def generate_samples_parallel(prompt, config):
    semaphore = asyncio.Semaphore(config.max_parallel_samples)

    async def bounded_call(index):
        async with semaphore:
            return await llm_call(prompt, config.temperature)

    tasks = [bounded_call(i) for i in range(config.sample_count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if not isinstance(r, Exception)]
```

### 7.2 File Locking

To prevent concurrent modifications:

```python
def acquire_file_lock(file_path, timeout=30):
    """
    Acquire exclusive lock on file.
    Uses fcntl on Unix, msvcrt on Windows.
    """
    lock_path = f"{file_path}.mdap.lock"

    # Platform-specific implementation
    ...
```

### 7.3 Run Isolation

Each run operates in isolation:
- Unique `run_id` (UUID v4)
- Separate backup directory
- Independent journal entries
- No shared mutable state

---

## 8. Security Considerations

### 8.1 Red Flag Detection

Global red flags are checked on ALL samples:

```python
GLOBAL_RED_FLAGS = [
    RedFlag(
        rule="output_contains_secrets",
        pattern=r"(api[_-]?key|password|secret|token)\s*[=:]\s*['\"]?[a-zA-Z0-9]{8,}",
        severity="critical"
    ),
    RedFlag(
        rule="contains_eval_or_exec",
        pattern=r"\b(eval|exec)\s*\(",
        severity="critical"
    ),
    # ... more patterns
]

def check_red_flags(sample, task_red_flags):
    violations = []

    # Check global flags
    for flag in GLOBAL_RED_FLAGS:
        if flag.matches(sample):
            violations.append(flag)

    # Check task-specific flags
    for flag in task_red_flags:
        if flag.matches(sample):
            violations.append(flag)

    return violations
```

### 8.2 Sandbox Execution

Validators run in a restricted environment:
- No network access (where possible)
- Limited file system access
- Timeout enforcement
- Resource limits (memory, CPU)

### 8.3 Audit Trail

All operations are logged for post-hoc security review:
- Input hashes (verify no tampering)
- Output hashes (verify reproducibility)
- Validator commands and results
- Red flag checks and results

---

## 9. Configuration

### 9.1 Global Configuration

From `mdap_global_config` in templates.json:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_sample_count` | int | 5 | Default number of samples |
| `default_consensus_threshold` | int | 3 | Default votes needed |
| `max_parallel_samples` | int | 10 | Max concurrent LLM calls |
| `timeout_per_sample_ms` | int | 30000 | Per-sample timeout |
| `retry_on_red_flag` | bool | true | Retry if all samples flagged |
| `log_all_samples` | bool | true | Log all samples, not just winner |
| `comparison_equality_mode` | str | "structural" | Equality algorithm |
| `determinism_seed` | int | 0 | RNG seed for determinism |

### 9.2 Per-Task Configuration

From `mdap_config` in each task template:

| Parameter | Type | Description |
|-----------|------|-------------|
| `sample_count` | int | Samples for this task type |
| `consensus_threshold` | int | Votes needed for this task |
| `voting_strategy` | str | Strategy name |
| `comparison_keys` | list[str] | Keys to compare |
| `temperature` | float | LLM temperature |
| `max_retries` | int | Max retry attempts |

---

## 10. Metrics and Monitoring

### 10.1 Metrics Collected

| Metric | Type | Description |
|--------|------|-------------|
| `mdap_runs_total` | counter | Total runs attempted |
| `mdap_runs_success` | counter | Successful runs |
| `mdap_consensus_achieved` | counter | Runs with consensus |
| `mdap_samples_generated` | counter | Total samples generated |
| `mdap_samples_rejected` | counter | Samples rejected by red flags |
| `mdap_validators_passed` | counter | Validators that passed |
| `mdap_validators_failed` | counter | Validators that failed |
| `mdap_run_duration_ms` | histogram | Run duration distribution |
| `mdap_sample_latency_ms` | histogram | Per-sample latency |

### 10.2 Health Checks

```python
def health_check():
    """
    Return health status of MDAP engine.
    """
    return {
        "status": "healthy",
        "templates_loaded": len(templates),
        "last_run_at": last_run_timestamp,
        "pending_runs": queue_length,
        "llm_available": check_llm_connection()
    }
```

---

## 11. Appendix: Example Execution Trace

### Task: Fix hash normalization at blockchain.py:42

```
[10:30:00.000] RUN_START run_id=abc123 task=hash_normalization_fix
[10:30:00.050] LOAD templates.json version=1.0.0
[10:30:00.100] PREPARE file=backend/ledger/blockchain.py line=42
[10:30:00.150] CONTEXT code="h = sha256_hex(text)"
[10:30:00.200] SAMPLE_START count=5 temp=0.0
[10:30:01.100] SAMPLE[0] OK hash=aaa111 keys={file,line,new_line}
[10:30:01.150] SAMPLE[1] OK hash=aaa111 keys={file,line,new_line}
[10:30:01.200] SAMPLE[2] REJECTED red_flag=multi_line_edit
[10:30:01.250] SAMPLE[3] OK hash=aaa111 keys={file,line,new_line}
[10:30:01.300] SAMPLE[4] OK hash=bbb222 keys={file,line,new_line}
[10:30:01.350] CONSENSUS strategy=first_to_ahead_by_k
[10:30:01.360] VOTE_DIST [{hash=aaa111,count=3},{hash=bbb222,count=1}]
[10:30:01.370] CONSENSUS_ACHIEVED winner=aaa111 threshold=3
[10:30:01.400] BACKUP created=.mdap_backup/blockchain.py.abc123.bak
[10:30:01.450] PATCH_APPLY line=42 old="sha256_hex(text)" new="hash_statement(text)"
[10:30:01.500] VALIDATOR[0] "pytest tests/test_canon.py -v" exit=0 pass=true
[10:30:02.500] VALIDATOR[1] "pytest tests/test_hashing.py -v -k hash_statement" exit=0 pass=true
[10:30:03.500] VALIDATOR[2] "python -c 'from backend.crypto.hashing...'" exit=0 pass=true
[10:30:03.550] BACKUP_CLEANUP removed=.mdap_backup/blockchain.py.abc123.bak
[10:30:03.600] RUN_COMPLETE success=true duration_ms=3600
```

---

## 12. Phase II Uplift Orchestration (Design Only)

> **PHASE II DESIGN ONLY — NOT IMPLEMENTED**
>
> This section describes how MDAP *would* orchestrate uplift experiments once
> uplift-capable slices exist. **No current RFL logs are suitable inputs.**
> These agents are not implemented, and Evidence Pack v1 makes no claims about
> MDAP-orchestrated uplift.

### 12.1 Prerequisites for Uplift Experiments

Before any MDAP uplift orchestration can occur:

1. **Non-degenerate uplift slices must exist** — slices where baseline abstention
   is measurably worse than RFL, with statistical significance.
2. **Preregistered experiment configs** — `uplift_experiment_config.json` files
   that define baseline vs RFL comparison parameters before execution.
3. **Fresh logs from uplift runs** — logs produced specifically for uplift
   measurement, not repurposed Phase I plumbing logs.

**Explicit guardrail:** The following Phase I logs are NOT valid uplift inputs:
- `fo_rfl.jsonl`
- `fo_rfl_50.jsonl`
- `fo_baseline.jsonl`
- Any log produced before uplift slices are validated

### 12.2 Uplift Micro-Agent Roles

Three specialized agents would orchestrate uplift experiments:

#### 12.2.1 UpliftRunnerAgent

**Purpose:** Execute preregistered baseline vs RFL experiments on uplift slices.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `uplift_experiment_config.json` | File | Preregistered experiment parameters |
| `slice_name` | String | Target uplift slice identifier |
| `cycle_count` | Integer | Number of cycles to run (e.g., 1000) |
| `baseline_seed` | Integer | RNG seed for baseline run |
| `rfl_seed` | Integer | RNG seed for RFL run (must match baseline) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| `baseline_log_path` | Path | JSONL log from baseline run |
| `rfl_log_path` | Path | JSONL log from RFL run |
| `experiment_manifest.json` | File | Sealed manifest with hashes and timestamps |
| `status` | Enum | `success` / `failed` / `invalid_config` |

**Contracts:**
- MUST refuse to run if `uplift_experiment_config.json` is missing
- MUST refuse to run on any log path matching `fo_rfl*.jsonl` or `fo_baseline*.jsonl`
- MUST produce identical cycle counts for baseline and RFL
- MUST seal manifest before any analysis begins

#### 12.2.2 EvidenceVerifierAgent

**Purpose:** Validate manifests, abstention curves, and non-degeneracy conditions.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `experiment_manifest.json` | File | From UpliftRunnerAgent |
| `baseline_log_path` | Path | Baseline JSONL log |
| `rfl_log_path` | Path | RFL JSONL log |
| `nondegeneracy_threshold` | Float | Minimum abstention delta (e.g., 0.05) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| `verification_report.json` | File | Detailed verification results |
| `experiment_valid` | Boolean | True if all checks pass |
| `nondegeneracy_met` | Boolean | True if abstention delta exceeds threshold |
| `flags` | List[String] | Any warnings or anomalies detected |

**Contracts:**
- MUST verify manifest hashes match actual log file hashes
- MUST verify cycle counts match between baseline and RFL
- MUST compute abstention curves independently (not trust pre-computed values)
- MUST NOT reinterpret Phase I plumbing logs as uplift evidence
- MUST flag if baseline abstention < 1% (degenerate baseline)

**Verification Checks:**
```
1. manifest_hash_valid: SHA-256 of logs matches manifest
2. cycle_count_match: baseline.cycles == rfl.cycles
3. seed_recorded: both seeds present in manifest
4. baseline_abstention_nonzero: abstention_rate(baseline) > 0.01
5. rfl_abstention_lower: abstention_rate(rfl) < abstention_rate(baseline)
6. delta_significant: (baseline - rfl) > nondegeneracy_threshold
7. no_phase_i_contamination: log paths not in PROHIBITED_LOGS
```

#### 12.2.3 GovernanceReporterAgent

**Purpose:** Summarize uplift metrics into governance-ready format.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `verification_report.json` | File | From EvidenceVerifierAgent |
| `experiment_manifest.json` | File | Original manifest |
| `slice_metadata` | Dict | Slice parameters (depth, atoms, etc.) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| `governance_summary.json` | File | Scalar/vector metrics for governance |
| `uplift_scalar` | Float | Single number summarizing uplift (e.g., Δabstention) |
| `confidence_interval` | Tuple[Float, Float] | 95% CI for uplift scalar |
| `recommendation` | Enum | `proceed` / `more_data_needed` / `no_uplift_detected` |

**Contracts:**
- MUST only produce summary if `experiment_valid == true`
- MUST propagate any flags from EvidenceVerifierAgent
- MUST NOT claim uplift if confidence interval crosses zero
- MUST include explicit statement: "Based on Phase II data only"

**Output Schema:**
```json
{
  "experiment_id": "uuid",
  "slice_name": "uplift_slice_001",
  "phase": "II",
  "uplift_scalar": 0.127,
  "confidence_interval": [0.089, 0.165],
  "baseline_abstention_rate": 0.231,
  "rfl_abstention_rate": 0.104,
  "cycles_analyzed": 1000,
  "recommendation": "proceed",
  "flags": [],
  "disclaimer": "Based on Phase II uplift data only. Not derived from Phase I logs."
}
```

### 12.3 Uplift Experiment Control Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE II UPLIFT EXPERIMENT FLOW                          │
│                         (Design Only — Not Implemented)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │ PREREGISTRATION  │  Human creates uplift_experiment_config.json          │
│  │ (Manual)         │  with slice, seeds, cycle count, thresholds           │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ UpliftRunner     │  Executes baseline & RFL runs                         │
│  │ Agent            │  Produces: baseline.jsonl, rfl.jsonl, manifest.json   │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ EvidenceVerifier │  Validates hashes, computes abstention curves         │
│  │ Agent            │  Outputs: verification_report.json, experiment_valid  │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ GovernanceReport │  Summarizes to scalar + CI                            │
│  │ Agent            │  Outputs: governance_summary.json, recommendation     │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ HUMAN REVIEW     │  Governance decision based on summary                 │
│  │ (Manual)         │                                                       │
│  └──────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.4 Prohibited Log Paths (Hardcoded Guardrail)

The following paths MUST be rejected by all uplift agents:

```python
PROHIBITED_PHASE_I_LOGS = [
    "fo_rfl.jsonl",
    "fo_rfl_50.jsonl",
    "fo_baseline.jsonl",
    "fo_baseline_50.jsonl",
    "**/evidence_pack_v1/**/*.jsonl",
    "**/phase_i/**/*.jsonl",
]

def is_prohibited_log(path: str) -> bool:
    """
    Returns True if path matches any prohibited Phase I log pattern.
    Uplift agents MUST call this and refuse to proceed if True.
    """
    for pattern in PROHIBITED_PHASE_I_LOGS:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False
```

### 12.5 Summary

| Agent | Status | Inputs | Outputs | Key Contract |
|-------|--------|--------|---------|--------------|
| UpliftRunnerAgent | NOT IMPLEMENTED | config, slice, seeds | logs, manifest | Refuses Phase I logs |
| EvidenceVerifierAgent | NOT IMPLEMENTED | manifest, logs | report, valid flag | Independent abstention calc |
| GovernanceReporterAgent | NOT IMPLEMENTED | report, manifest | summary, scalar | Only reports if valid |

**The existence of additional RFL cycles (from Phase I) does not activate any
MDAP engine or imply any MDAP readiness. These agents require fresh Phase II
uplift data that does not yet exist.**

---

## 13. References

- `ops/microagents/templates.json` - Task template definitions
- `ops/microagents/runner_skeleton.py` - Implementation skeleton
- `docs/whitepaper.md` - MathLedger architecture
- JSONL specification: https://jsonlines.org/
