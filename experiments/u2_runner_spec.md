# U2 Runner Formal Specification

**Document Version:** 1.1.0
**Phase:** II — NOT RUN IN PHASE I
**Author:** Claude C — U2 Runner Architect (Runner Law)
**Status:** NORMATIVE

> **Revision 1.1.0:** Added Replay-Aware State Extensions (Section 2.4),
> Replay Error Taxonomy (Section 5.5), and Extended Determinism Proof (Section 6.4).

---

## 1. Overview

The U2 Runner (`experiments/run_uplift_u2.py`) executes Phase II uplift experiments comparing baseline (random) and RFL (policy-driven) candidate ordering strategies. This document provides the formal specification for the runner's behavior, contracts, and error handling.

### 1.1 Scope

This specification covers:
- Runner lifecycle and state machine
- **Replay verification lifecycle** (v1.1.0)
- Input/output contracts
- Determinism guarantees
- Error taxonomy
- Compliance requirements

### 1.2 Normative References

| Document | Purpose |
|----------|---------|
| `config/curriculum_uplift_phase2.yaml` | Slice definitions |
| `experiments/prereg/PREREG_UPLIFT_U2.yaml` | Preregistration template |
| `CLAUDE.md` | Project conventions |

---

## 2. Finite State Machine

### 2.1 State Diagram

```
                                    ┌─────────────────────────────────────────┐
                                    │                                         │
                                    ▼                                         │
┌──────────┐    parse_args    ┌──────────┐    validate    ┌──────────────┐   │
│  START   │ ───────────────► │  INIT    │ ─────────────► │  VALIDATED   │   │
└──────────┘                  └──────────┘                └──────────────┘   │
                                    │                           │             │
                                    │ validation_error          │             │
                                    ▼                           │             │
                              ┌──────────┐                      │             │
                              │  ERROR   │ ◄────────────────────┼─────────────┤
                              └──────────┘                      │             │
                                    │                           │             │
                                    │ exit(1)                   │ load_config │
                                    ▼                           ▼             │
                              ┌──────────┐                ┌──────────────┐    │
                              │   EXIT   │                │  CONFIGURED  │    │
                              └──────────┘                └──────────────┘    │
                                                                │             │
                                          ┌─────────────────────┼─────────────┤
                                          │                     │             │
                              dry_run     │                     │ run_cycles  │
                                          ▼                     ▼             │
                                    ┌──────────┐          ┌──────────────┐    │
                                    │ DRY_RUN  │          │   RUNNING    │ ◄──┘
                                    └──────────┘          └──────────────┘
                                          │                     │
                                          │                     │ cycle_complete
                                          │                     ▼
                                          │               ┌──────────────┐
                                          │               │  CYCLE_DONE  │ ─┐
                                          │               └──────────────┘  │
                                          │                     │           │
                                          │         more_cycles │           │ cycle_error
                                          │                     │           │
                                          │                     ▼           ▼
                                          │               ┌──────────────┐  │
                                          │               │   RUNNING    │ ◄┘
                                          │               └──────────────┘
                                          │                     │
                                          │                     │ all_cycles_done
                                          │                     ▼
                                          │               ┌──────────────┐
                                          └─────────────► │  FINALIZE    │
                                                          └──────────────┘
                                                                │
                                                                │ write_manifest
                                                                ▼
                                                          ┌──────────────┐
                                                          │  COMPLETED   │
                                                          └──────────────┘
                                                                │
                                                                │ exit(0)
                                                                ▼
                                                          ┌──────────────┐
                                                          │    EXIT      │
                                                          └──────────────┘
```

### 2.2 State Definitions

| State | Description | Valid Transitions |
|-------|-------------|-------------------|
| `START` | Initial state before argument parsing | `INIT` |
| `INIT` | Arguments parsed, awaiting validation | `VALIDATED`, `ERROR` |
| `VALIDATED` | Inputs validated (dry-run path) | `CONFIGURED`, `DRY_RUN`, `ERROR` |
| `CONFIGURED` | Slice config loaded, runner initialized | `RUNNING`, `ERROR` |
| `RUNNING` | Executing experiment cycles | `CYCLE_DONE`, `ERROR` |
| `CYCLE_DONE` | Single cycle completed | `RUNNING`, `FINALIZE` |
| `DRY_RUN` | Validation-only mode | `FINALIZE` |
| `FINALIZE` | Writing manifest and artifacts | `COMPLETED`, `ERROR` |
| `COMPLETED` | Experiment finished successfully | `EXIT` |
| `ERROR` | Error state (logged, non-zero exit) | `EXIT` |
| `EXIT` | Terminal state | (none) |

### 2.3 State Invariants

```
INV-1: In RUNNING state, cycle_index ∈ [0, total_cycles)
INV-2: In RUNNING state with mode="rfl", policy ≠ null
INV-3: In COMPLETED state, len(results) = total_cycles
INV-4: In ERROR state, error_logger.has_errors() = true
INV-5: In any state, base_seed is immutable after INIT
```

### 2.4 Replay-Aware State Extensions (v1.1.0)

The FSM is extended to support **determinism verification via replay**. A replay run re-executes an experiment using parameters extracted from a completed manifest, then compares attestation roots.

#### 2.4.1 Extended State Diagram

```
                              PRIMARY RUN PATH
                              ================

┌──────────┐                                              ┌──────────────┐
│  START   │ ─────────────────────────────────────────────►│  COMPLETED   │
└──────────┘          (existing FSM, see 2.1)             └──────────────┘
                                                                 │
                                                                 │ replay_requested
                                                                 ▼
                                                          ┌────────────────┐
                                                          │ REPLAY_PENDING │
                                                          └────────────────┘
                                                                 │
                                        ┌────────────────────────┼────────────────────────┐
                                        │                        │                        │
                                        │ manifest_valid         │ manifest_invalid       │
                                        ▼                        ▼                        │
                                  ┌────────────────┐      ┌────────────────┐              │
                                  │ REPLAY_RUNNING │      │  REPLAY_FAILED │ ◄────────────┘
                                  └────────────────┘      └────────────────┘
                                        │                        │
                           ┌────────────┼────────────┐           │
                           │            │            │           │
                           │ cycle_ok   │ cycle_fail │           │
                           ▼            ▼            │           │
                     ┌───────────┐ ┌───────────┐     │           │
                     │  (next)   │ │REPLAY_FAIL│ ◄───┘           │
                     └───────────┘ └───────────┘                 │
                           │                                     │
                           │ all_cycles_verified                 │
                           ▼                                     │
                     ┌─────────────────┐                         │
                     │ REPLAY_VERIFIED │                         │
                     └─────────────────┘                         │
                           │                                     │
                           │ emit_receipt                        │ emit_failure_receipt
                           ▼                                     ▼
                     ┌──────────┐                          ┌──────────┐
                     │   EXIT   │                          │   EXIT   │
                     └──────────┘                          └──────────┘
                      exit(0)                               exit(1)
```

#### 2.4.2 Replay State Definitions

| State | Description | Valid Transitions |
|-------|-------------|-------------------|
| `REPLAY_PENDING` | Manifest loaded, awaiting validation | `REPLAY_RUNNING`, `REPLAY_FAILED` |
| `REPLAY_RUNNING` | Re-executing cycles with original parameters | `REPLAY_RUNNING`, `REPLAY_VERIFIED`, `REPLAY_FAILED` |
| `REPLAY_VERIFIED` | All cycles match original attestation roots | `EXIT` |
| `REPLAY_FAILED` | Mismatch detected or validation error | `EXIT` |

#### 2.4.3 Replay State Invariants

```
INV-R1: In REPLAY_PENDING, manifest ≠ null ∧ manifest.status = "COMPLETED"
INV-R2: In REPLAY_RUNNING, replay_cycle_index ∈ [0, original_cycles)
INV-R3: In REPLAY_RUNNING, base_seed = manifest.experiment.base_seed
INV-R4: In REPLAY_RUNNING, slice_config_hash = manifest.slice_config.config_hash_sha256
INV-R5: In REPLAY_VERIFIED, ∀ i: replay_h_t[i] = original_h_t[i]
INV-R6: In REPLAY_FAILED, ∃ i: replay_h_t[i] ≠ original_h_t[i] ∨ validation_error
```

#### 2.4.4 Replay Transition Rules

| From | Event | Guard | To | Action |
|------|-------|-------|-----|--------|
| `COMPLETED` | `replay_requested` | manifest exists | `REPLAY_PENDING` | Load manifest |
| `REPLAY_PENDING` | `validate_manifest` | valid | `REPLAY_RUNNING` | Initialize replay runner |
| `REPLAY_PENDING` | `validate_manifest` | invalid | `REPLAY_FAILED` | Log RUN-41..RUN-44 |
| `REPLAY_RUNNING` | `cycle_complete` | h_t matches | `REPLAY_RUNNING` | Increment cycle |
| `REPLAY_RUNNING` | `cycle_complete` | h_t mismatch | `REPLAY_FAILED` | Log RUN-44 |
| `REPLAY_RUNNING` | `all_verified` | last cycle ok | `REPLAY_VERIFIED` | Emit receipt |
| `REPLAY_VERIFIED` | `exit` | - | `EXIT` | Return 0 |
| `REPLAY_FAILED` | `exit` | - | `EXIT` | Return 1 |

#### 2.4.5 Replay CLI Extension

| Argument | Type | Required | Default | Constraint |
|----------|------|----------|---------|------------|
| `--replay` | string | No | - | Path to manifest.json |
| `--replay-log` | string | No | - | Path for replay results JSONL |

**Usage:**
```bash
# Replay a completed experiment
python run_uplift_u2.py --replay results/uplift_u2_manifest_slice_uplift_goal.json

# Replay with explicit output
python run_uplift_u2.py --replay manifest.json --replay-log results/replay_verification.jsonl
```

---

## 3. Input Contract

### 3.1 Command-Line Arguments

| Argument | Type | Required | Default | Constraint |
|----------|------|----------|---------|------------|
| `--slice` | string | Yes | - | ∈ VALID_SLICES |
| `--mode` | string | Yes* | - | ∈ {"baseline", "rfl"} |
| `--pair` | flag | Yes* | false | Mutually exclusive with --mode |
| `--cycles` | int | No | 10 | > 0 |
| `--seed` | int | No | 0x4D444150 | ∈ [0, 2^32) |
| `--out` | string | No | - | Valid path |
| `--dry-run` | flag | No | false | - |

*Exactly one of `--mode` or `--pair` must be specified.

### 3.2 Slice Configuration Contract

The slice configuration MUST contain:

```yaml
# REQUIRED fields
name: string                    # Slice identifier
params:                         # OR "parameters"
  total_max: int               # Max candidates per cycle
success_metric:
  kind: enum                   # ∈ {"goal_hit", "density", "chain_length", "multi_goal"}
  # Metric-specific parameters (see 3.2.1)

# OPTIONAL fields
formula_pool_entries: list     # Candidate formulas
budget: object                 # Budget constraints
gates: object                  # Gate parameters
uplift: object                 # Uplift metadata
```

#### 3.2.1 Metric-Specific Parameters

**goal_hit:**
```yaml
target_hashes: list[string]    # Required target formula hashes
min_goal_hits: int             # Default: 1
min_total_verified: int        # Default: 3
```

**density:**
```yaml
min_verified: int              # Default: 5
max_candidates: int            # Default: 40
```

**chain_length:**
```yaml
min_chain_length: int          # Default: 3
chain_target_hash: string      # Required target hash
```

**multi_goal:**
```yaml
required_goal_hashes: list[string]  # All required goals
min_each_goal: int                  # Default: 1
```

### 3.3 Preregistration Binding

The runner MUST:
1. Load `experiments/prereg/PREREG_UPLIFT_U2.yaml`
2. Compute SHA-256 hash of prereg file
3. Include hash in manifest under `preregistration.prereg_hash_sha256`

### 3.4 Seed Schedule Contract

```
DEFINITION: seed_schedule(base_seed, cycle_index) → cycle_seed

FORMULA: cycle_seed = base_seed + cycle_index

CONSTRAINT:
  ∀ i ∈ [0, total_cycles): seed_schedule(base_seed, i) is unique

MDAP_EPOCH_SEED = 0x4D444150 = 1296318800
```

---

## 4. Output Contract

### 4.1 Results JSONL Format

Each line in `{slice}_{mode}.jsonl` MUST be valid JSON with:

```json
{
  "cycle": int,                    // [0, total_cycles)
  "cycle_seed": int,               // base_seed + cycle
  "mode": "baseline" | "rfl",
  "slice_name": string,
  "phase_label": "PHASE II — NOT RUN IN PHASE I",
  "success": bool,
  "metric_value": float,           // ∈ [0.0, 1.0]
  "metric_type": string,           // Metric kind
  "metric_result": {               // Detailed metrics
    "metric_kind": string,
    "metric_type": string,
    "metric_value": float,
    "success": bool,
    // ... metric-specific fields
  },
  "derivation": {
    "candidates_tried": int,
    "verified_count": int,
    "verified_hashes": list[string],
    "candidate_order": list[string]
  },
  "roots": {
    "h_t": string,                 // SHA-256, 64 hex chars
    "r_t": string,                 // SHA-256, 64 hex chars
    "u_t": string                  // SHA-256, 64 hex chars
  },
  "rfl": {                         // Only if mode="rfl"
    "executed": true,
    "policy_weight_count": int,
    "update_count": int,
    "total_attempts": int,
    "total_successes": int
  }
}
```

### 4.2 Manifest JSON Format

```json
{
  "manifest_version": "2.0" | "2.1",  // 2.1 for paired
  "phase": "II",
  "phase_label": "PHASE II — NOT RUN IN PHASE I",
  "experiment_type": "paired",         // Only if paired
  "experiment": {
    "family": "uplift_u2",
    "slice_name": string,
    "cycles": int,
    "base_seed": int,
    "base_seed_hex": string,
    "seed_schedule": "MDAP_SEED + cycle_index",
    "seed_schedule_note": string,
    "seed_schedule_sample": list[int],
    "mode": string,                    // Single mode only
    "modes": ["baseline", "rfl"]       // Paired only
  },
  "slice_config": {
    "config_path": string,
    "config_hash_sha256": string       // 64 hex chars
  },
  "preregistration": {
    "prereg_path": string,
    "prereg_hash_sha256": string,
    "prereg_hash_computed_at": string  // ISO 8601
  },
  "governance": {
    "phase": "II",
    "blocked_by": [],
    "cannot_reference": list[string],
    "requires": list[string]
  },
  "execution": {
    "executor": "experiments/run_uplift_u2.py",
    "executor_role": string,
    "started_at": string,              // ISO 8601
    "completed_at": string,            // ISO 8601
    "environment": {
      "platform": string,
      "python_version": string,
      "mathledger_commit": string
    }
  },
  "artifacts": {
    "manifest_path": string,
    "results_path": string,            // Single mode
    "results_hash_sha256": string,
    "baseline": {...},                 // Paired mode
    "rfl": {...}                       // Paired mode
  },
  "determinism_verification": {
    "determinism_check": "PENDING",
    "seed_schedule_formula": "base_seed + cycle_index",
    "h_t_first": string,
    "h_t_last": string
  },
  "summary": {
    "total_cycles": int,
    "success_count": int,
    "success_rate": float,
    "success_rate_pct": string,
    "uplift": {...}                    // Paired mode only
  }
}
```

### 4.3 Error Log Format

File: `results/uplift_u2_errors_{timestamp}.jsonl`

```json
{
  "timestamp": string,           // ISO 8601
  "error_type": string,          // Error code (RUN-XX)
  "message": string,
  "slice_name": string,          // Optional
  "cycle": int,                  // Optional
  "tb": string                   // Traceback, optional
}
```

---

## 5. Error State Taxonomy

### 5.1 Initialization Errors (RUN-01 to RUN-10)

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| RUN-01 | `INVALID_SLICE` | Slice name not in VALID_SLICES | Exit(1) |
| RUN-02 | `INVALID_MODE` | Mode not in VALID_MODES | Exit(1) |
| RUN-03 | `MISSING_REQUIRED_ARG` | Required argument not provided | Exit(1) |
| RUN-04 | `MUTUALLY_EXCLUSIVE` | Both --mode and --pair specified | Exit(1) |
| RUN-05 | `INVALID_CYCLES` | cycles <= 0 | Exit(1) |
| RUN-06 | `INVALID_SEED` | seed out of range | Exit(1) |
| RUN-07 | `OUTPUT_PATH_ERROR` | Cannot create output directory | Exit(1) |
| RUN-08 | `YAML_IMPORT_ERROR` | PyYAML not installed | Exit(1) |
| RUN-09 | `RESERVED` | Reserved for future use | - |
| RUN-10 | `RESERVED` | Reserved for future use | - |

### 5.2 Configuration Errors (RUN-11 to RUN-20)

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| RUN-11 | `CONFIG_NOT_FOUND` | curriculum_uplift_phase2.yaml missing | Exit(1) |
| RUN-12 | `CONFIG_PARSE_ERROR` | YAML parse failure | Exit(1) |
| RUN-13 | `SLICE_NOT_FOUND` | Slice not in config | Exit(1) |
| RUN-14 | `MISSING_PARAMS` | Slice missing 'params' section | Exit(1) |
| RUN-15 | `MISSING_SUCCESS_METRIC` | Slice missing 'success_metric' | Exit(1) |
| RUN-16 | `INVALID_METRIC_KIND` | Unknown metric kind | Exit(1) |
| RUN-17 | `PREREG_NOT_FOUND` | Preregistration file missing | Warn, continue |
| RUN-18 | `PREREG_PARSE_ERROR` | Preregistration YAML error | Warn, continue |
| RUN-19 | `FORMULA_POOL_EMPTY` | No formula_pool_entries | Warn, continue |
| RUN-20 | `RESERVED` | Reserved for future use | - |

### 5.3 Runtime Errors (RUN-21 to RUN-30)

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| RUN-21 | `CYCLE_ERROR` | Unhandled exception in cycle | Log, continue |
| RUN-22 | `VERIFICATION_TIMEOUT` | Hermetic verify timeout | Mark failed |
| RUN-23 | `METRIC_COMPUTATION_ERROR` | Error computing success metric | Log, mark failed |
| RUN-24 | `POLICY_UPDATE_ERROR` | RFL policy update failed | Log, continue |
| RUN-25 | `HASH_COMPUTATION_ERROR` | SHA-256 computation failed | Fatal |
| RUN-26 | `FILE_WRITE_ERROR` | Cannot write to output file | Fatal |
| RUN-27 | `MEMORY_ERROR` | Out of memory | Fatal |
| RUN-28 | `INTERRUPT` | User interrupt (Ctrl+C) | Save state, exit |
| RUN-29 | `RESERVED` | Reserved for future use | - |
| RUN-30 | `RESERVED` | Reserved for future use | - |

### 5.4 Finalization Errors (RUN-31 to RUN-40)

| Code | Name | Description | Recovery |
|------|------|-------------|----------|
| RUN-31 | `MANIFEST_WRITE_ERROR` | Cannot write manifest | Exit(1) |
| RUN-32 | `RESULTS_HASH_ERROR` | Cannot compute results hash | Warn, continue |
| RUN-33 | `GIT_COMMIT_ERROR` | Cannot get git commit | Use "unknown" |
| RUN-34 | `PAIRED_MISMATCH` | Baseline/RFL cycle count mismatch | Fatal |
| RUN-35 | `EMPTY_RESULTS` | No results to write | Warn |
| RUN-36 | `VALIDATION_FAILED` | Dry-run validation failed | Exit(1) |
| RUN-37 | `RESERVED` | Reserved for future use | - |
| RUN-38 | `RESERVED` | Reserved for future use | - |
| RUN-39 | `RESERVED` | Reserved for future use | - |
| RUN-40 | `UNKNOWN_ERROR` | Catch-all for unclassified errors | Exit(1) |

### 5.5 Replay Errors (RUN-41 to RUN-50) — v1.1.0

Replay errors occur during determinism verification via replay execution.

#### RUN-41: `REPLAY_MANIFEST_MISMATCH`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-41 |
| **Name** | `REPLAY_MANIFEST_MISMATCH` |
| **Precondition** | `--replay` flag provided with manifest path |
| **Symptom** | Manifest schema version incompatible or required fields missing |
| **Observable** | Error log: `"Manifest version X.Y not supported for replay"` |
| **Mandated Action** | Exit(1); emit `REPLAY_FAILED` receipt with mismatch details |
| **Resolution** | Re-run original experiment with current runner version |

#### RUN-42: `REPLAY_SEED_SCHEDULE_MISMATCH`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-42 |
| **Name** | `REPLAY_SEED_SCHEDULE_MISMATCH` |
| **Precondition** | Manifest loaded successfully |
| **Symptom** | `manifest.experiment.seed_schedule` ≠ "MDAP_SEED + cycle_index" |
| **Observable** | Error log: `"Seed schedule mismatch: expected X, found Y"` |
| **Mandated Action** | Exit(1); replay cannot proceed with incompatible schedule |
| **Resolution** | Verify manifest was created by conformant runner |

#### RUN-43: `REPLAY_LOG_MISSING`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-43 |
| **Name** | `REPLAY_LOG_MISSING` |
| **Precondition** | Manifest contains `artifacts.results_path` or `artifacts.{mode}.results_path` |
| **Symptom** | Referenced JSONL log file does not exist at expected path |
| **Observable** | Error log: `"Replay log not found: {path}"` |
| **Mandated Action** | Exit(1); cannot verify without original results |
| **Resolution** | Locate original results file or re-run experiment |

#### RUN-44: `REPLAY_HT_MISMATCH`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-44 |
| **Name** | `REPLAY_HT_MISMATCH` |
| **Precondition** | Replay cycle completed, h_t computed |
| **Symptom** | `replay_h_t[i] ≠ original_h_t[i]` for cycle i |
| **Observable** | Error log: `"H_t mismatch at cycle {i}: expected {orig}, got {replay}"` |
| **Mandated Action** | Transition to `REPLAY_FAILED`; emit detailed mismatch report |
| **Resolution** | Investigate non-determinism source (see Section 6.3) |

#### RUN-45: `REPLAY_RT_MISMATCH`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-45 |
| **Name** | `REPLAY_RT_MISMATCH` |
| **Precondition** | Replay cycle completed, r_t computed |
| **Symptom** | `replay_r_t[i] ≠ original_r_t[i]` for cycle i |
| **Observable** | Error log: `"R_t mismatch at cycle {i}: ordering divergence"` |
| **Mandated Action** | Transition to `REPLAY_FAILED`; log candidate order diff |
| **Resolution** | Check RFL policy initialization or shuffle seed |

#### RUN-46: `REPLAY_UT_MISMATCH`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-46 |
| **Name** | `REPLAY_UT_MISMATCH` |
| **Precondition** | Replay cycle completed with mode="rfl", u_t computed |
| **Symptom** | `replay_u_t[i] ≠ original_u_t[i]` for cycle i |
| **Observable** | Error log: `"U_t mismatch at cycle {i}: policy state divergence"` |
| **Mandated Action** | Transition to `REPLAY_FAILED`; log policy weight diff |
| **Resolution** | Check UCB update formula or float precision |

#### RUN-47: `REPLAY_CYCLE_COUNT_MISMATCH`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-47 |
| **Name** | `REPLAY_CYCLE_COUNT_MISMATCH` |
| **Precondition** | Manifest and results log loaded |
| **Symptom** | Line count in results JSONL ≠ `manifest.experiment.cycles` |
| **Observable** | Error log: `"Cycle count mismatch: manifest={m}, log={l}"` |
| **Mandated Action** | Exit(1); cannot replay incomplete experiment |
| **Resolution** | Ensure original run completed fully |

#### RUN-48: `REPLAY_CONFIG_HASH_MISMATCH`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-48 |
| **Name** | `REPLAY_CONFIG_HASH_MISMATCH` |
| **Precondition** | Replay started, config loaded |
| **Symptom** | Current config SHA-256 ≠ `manifest.slice_config.config_hash_sha256` |
| **Observable** | Error log: `"Config modified since original run"` |
| **Mandated Action** | Warn and continue (config changes may be benign) OR fail if strict mode |
| **Resolution** | Restore original config or acknowledge expected difference |

#### RUN-49: `REPLAY_VERIFICATION_ERROR`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-49 |
| **Name** | `REPLAY_VERIFICATION_ERROR` |
| **Precondition** | Replay cycle in progress |
| **Symptom** | Hermetic verification throws unexpected exception |
| **Observable** | Error log: `"Verification error during replay: {exception}"` |
| **Mandated Action** | Transition to `REPLAY_FAILED`; log full traceback |
| **Resolution** | Debug verification oracle implementation |

#### RUN-50: `REPLAY_UNKNOWN_ERROR`

| Attribute | Value |
|-----------|-------|
| **Code** | RUN-50 |
| **Name** | `REPLAY_UNKNOWN_ERROR` |
| **Precondition** | Any replay state |
| **Symptom** | Unclassified exception during replay |
| **Observable** | Error log: `"Unknown replay error: {exception}"` |
| **Mandated Action** | Transition to `REPLAY_FAILED`; log full context |
| **Resolution** | Classify error and assign specific code if recurring |

---

## 6. Runner Determinism Proof Sketch

### 6.1 Theorem: Paired Mode Determinism

**Statement:** Given fixed inputs (slice_name, cycles, base_seed), paired mode produces identical results across runs.

**Proof Sketch:**

1. **Seed Schedule Determinism**
   ```
   ∀ i ∈ [0, cycles): seed_schedule(base_seed, i) = base_seed + i
   ```
   The seed schedule is a pure function of base_seed and cycle index. No external state.

2. **Baseline Mode Determinism**
   ```
   baseline_order(candidates, cycle_seed) = shuffle(candidates, Random(cycle_seed))
   ```
   Python's `random.shuffle` with a seeded `Random` instance is deterministic.

3. **RFL Mode Determinism**
   ```
   rfl_order(candidates, policy) = sort(candidates, key=policy.score)
   ```
   - Policy weights initialized from `Random(base_seed)`
   - Updates depend only on verification results
   - Verification via `hermetic_verify(hash, pool, Random(cycle_seed))` is deterministic

4. **Hermetic Oracle Determinism**
   ```
   hermetic_verify(hash, pool, rng) → bool
   ```
   - Role lookup is deterministic (first match in pool)
   - Probability selection is deterministic given seeded rng
   - `rng.random() < prob` is deterministic

5. **Attestation Root Determinism**
   ```
   h_t = SHA256(cycle | seed | sorted(verified_hashes))
   r_t = SHA256(cycle | seed | candidate_order)
   u_t = SHA256(cycle | seed | policy_weights)
   ```
   All inputs are deterministic, SHA256 is a pure function.

**Conclusion:** All components are pure functions of deterministic inputs. QED.

### 6.2 Ordering Stability Conditions

For the runner to maintain ordering stability:

1. **Candidate Pool Stability**
   ```
   CONDITION: formula_pool_entries order must be consistent across loads
   ENFORCEMENT: YAML list order is preserved
   ```

2. **Hash Computation Stability**
   ```
   CONDITION: SHA256(formula) must be consistent
   ENFORCEMENT: UTF-8 encoding, no normalization
   ```

3. **Sort Stability**
   ```
   CONDITION: Ties in RFL scoring must be resolved consistently
   ENFORCEMENT: Python's sort is stable; original order preserved for ties
   ```

4. **Float Precision Stability**
   ```
   CONDITION: Policy weight computations must be reproducible
   ENFORCEMENT: Use IEEE 754 double precision, avoid platform-specific ops
   ```

### 6.3 Non-Determinism Sources (Avoided)

| Source | Mitigation |
|--------|------------|
| System time | Not used in computation |
| Process ID | Not used |
| Network I/O | Not used |
| File system ordering | Lists explicitly sorted where needed |
| Dict ordering | Python 3.7+ guarantees insertion order |
| Floating point | Pure arithmetic, no transcendentals |

### 6.4 Extended Determinism Proof: Replay Verification (v1.1.0)

#### 6.4.1 Theorem: Replay Determinism Equivalence

**Statement:** If a primary run P and its replay run R both complete successfully (status `COMPLETED` / `REPLAY_VERIFIED`), then:

```
∀ i ∈ [0, cycles):
  P.h_t[i] = R.h_t[i] ∧
  P.r_t[i] = R.r_t[i] ∧
  P.u_t[i] = R.u_t[i]
```

And consequently:

```
manifest.determinism_verification.h_t_first = replay_manifest.h_t_first ∧
manifest.determinism_verification.h_t_last  = replay_manifest.h_t_last
```

**Proof Sketch:**

1. **Parameter Binding Invariant**
   ```
   GIVEN: replay.base_seed = manifest.experiment.base_seed
          replay.cycles = manifest.experiment.cycles
          replay.slice_name = manifest.experiment.slice_name
          replay.mode = manifest.experiment.mode (or modes for paired)
   ```
   The replay runner extracts all computation-affecting parameters from the manifest. No additional entropy sources are permitted.

2. **Configuration Binding**
   ```
   PRECONDITION: hash(current_config) = manifest.slice_config.config_hash_sha256
   ```
   If this precondition fails, RUN-48 is raised. When it holds, the slice configuration—including formula pools, parameters, and metric definitions—is identical.

3. **Seed Schedule Preservation**
   ```
   ∀ i: replay.seed_schedule(base_seed, i) = base_seed + i = primary.seed_schedule(base_seed, i)
   ```
   The seed schedule formula is verified to match (RUN-42 if mismatch). Both runs use identical cycle seeds.

4. **Hermetic Oracle Determinism (per 6.1.4)**
   ```
   hermetic_verify(h, pool, Random(seed)) is a pure function
   ```
   Given identical pool and seed, verification results are identical.

5. **Policy State Determinism (RFL mode)**
   ```
   policy_0 = initialize_policy(Random(base_seed))
   policy_{i+1} = update(policy_i, verification_results_i)
   ```
   - Initial weights determined by base_seed
   - Updates are pure functions of prior state and verification results
   - With identical verification results (from step 4), policy evolution is identical

6. **Attestation Root Computation**
   ```
   h_t = SHA256(cycle || seed || sorted(verified_hashes))
   r_t = SHA256(cycle || seed || candidate_order)
   u_t = SHA256(cycle || seed || policy_weights)
   ```
   All inputs are deterministic (from steps 1-5). SHA256 is a cryptographic hash with no internal state.

7. **Induction Over Cycles**
   ```
   BASE CASE: i = 0
     - seed_0 = base_seed + 0 (deterministic)
     - candidate_order_0 = f(policy_0, Random(seed_0)) (deterministic per 5, 4)
     - verified_0 = {h : hermetic_verify(h, pool, Random(seed_0))} (deterministic per 4)
     - h_t_0, r_t_0, u_t_0 computed from above (deterministic per 6)

   INDUCTIVE STEP: Assume cycle i matches. For cycle i+1:
     - seed_{i+1} = base_seed + (i+1) (deterministic)
     - policy_{i+1} = update(policy_i, verified_i) (deterministic, by IH verified_i matches)
     - candidate_order_{i+1} = f(policy_{i+1}, Random(seed_{i+1})) (deterministic)
     - verified_{i+1} = {...} (deterministic per 4)
     - h_t_{i+1}, r_t_{i+1}, u_t_{i+1} match (per 6)
   ```

**Conclusion:** By mathematical induction, all attestation roots match. QED.

#### 6.4.2 Corollary: Results Hash Equivalence

**Statement:** If replay verification succeeds, then:

```
hash(replay_results.jsonl) = manifest.artifacts.results_hash_sha256
```

**Proof:** The JSONL file contains cycle records with attestation roots. By Theorem 6.4.1, all roots match. The cycle metadata (index, seed, mode, slice_name) are bound by manifest extraction. Therefore, line-by-line content is identical, and the file hash matches.

#### 6.4.3 Replay Verification Protocol

```
PROTOCOL ReplayVerification(manifest_path):
  1. LOAD manifest from manifest_path
  2. VALIDATE manifest.status = "COMPLETED"
  3. VALIDATE manifest.manifest_version ∈ {"2.0", "2.1"}
  4. VALIDATE manifest.experiment.seed_schedule = "MDAP_SEED + cycle_index"
  5. LOAD original_results from manifest.artifacts.results_path
  6. VALIDATE line_count(original_results) = manifest.experiment.cycles
  7. OPTIONALLY VALIDATE hash(config) = manifest.slice_config.config_hash_sha256

  8. FOR mode IN manifest.experiment.modes (or [manifest.experiment.mode]):
     9. INITIALIZE replay_runner(manifest.experiment, mode)
     10. FOR i IN [0, manifest.experiment.cycles):
         11. RUN replay_cycle(i)
         12. COMPUTE replay_h_t, replay_r_t, replay_u_t
         13. COMPARE with original_results[i].roots
         14. IF mismatch: RAISE RUN-44/45/46, TRANSITION to REPLAY_FAILED
     15. END FOR
  16. END FOR

  17. EMIT replay_receipt with REPLAY_VERIFIED
  18. RETURN success
```

#### 6.4.4 Replay Failure Diagnosis

When replay verification fails (RUN-44, RUN-45, or RUN-46), the following diagnostic information MUST be emitted:

| Field | Description |
|-------|-------------|
| `mismatch_cycle` | Cycle index where first mismatch occurred |
| `expected_root` | Original attestation root value |
| `actual_root` | Replay-computed attestation root value |
| `root_type` | Which root mismatched: "h_t", "r_t", or "u_t" |
| `verified_diff` | Set difference: original_verified △ replay_verified |
| `order_diff` | Edit distance between candidate orderings (if r_t mismatch) |
| `policy_diff` | Weight vector diff (if u_t mismatch, RFL mode only) |

This diagnostic output aids in identifying the specific non-determinism source.

---

## 7. Compliance Checklist

### 7.1 Pre-Modification Checklist

Before modifying `run_uplift_u2.py`:

- [ ] Read this specification in full
- [ ] Identify which states/transitions are affected
- [ ] Verify determinism is preserved (see Section 6)
- [ ] Check error codes in taxonomy (Section 5)
- [ ] Review input/output contracts (Sections 3-4)

### 7.2 Implementation Checklist

During modification:

- [ ] All new errors use taxonomy codes (RUN-XX)
- [ ] New CLI args documented in Section 3.1
- [ ] New output fields documented in Section 4
- [ ] Determinism proof updated if logic changes
- [ ] No platform-specific code introduced
- [ ] No external I/O (network, database) added
- [ ] All logging uses `[U2]` prefix
- [ ] Phase II label included in all artifacts

### 7.3 Post-Modification Checklist

After modification:

- [ ] All existing tests pass
- [ ] Paired mode produces identical results across runs
- [ ] Dry-run mode validates correctly
- [ ] Error logging produces valid JSONL
- [ ] Manifest schema unchanged or version bumped
- [ ] This specification updated if contracts changed

### 7.4 Forbidden Modifications

The following modifications are PROHIBITED without explicit approval:

1. **Removing Phase II labels** - All outputs must be marked
2. **Adding network calls** - Runner must be hermetic
3. **Using system clock for computation** - Only for timestamps
4. **Modifying seed schedule formula** - Breaks determinism proofs
5. **Adding RLHF/preference signals** - RFL uses verifiable feedback only
6. **Referencing Phase I artifacts** - Strict separation required

---

## 8. Appendix

### A. VALID_SLICES Enumeration

```python
VALID_SLICES = frozenset([
    "slice_uplift_goal",
    "slice_uplift_sparse",
    "slice_uplift_tree",
    "slice_uplift_dependency",
])
```

### B. VALID_MODES Enumeration

```python
VALID_MODES = frozenset(["baseline", "rfl"])
```

### C. MDAP_EPOCH_SEED

```python
MDAP_EPOCH_SEED = 0x4D444150  # "MDAP" in ASCII hex = 1296318800
```

### D. Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | Claude C | Initial specification |
| 1.1.0 | 2025-12-06 | Claude C | Added Replay-Aware State Extensions (2.4), Replay Error Taxonomy (5.5), Extended Determinism Proof (6.4) |

---

*End of Specification*
