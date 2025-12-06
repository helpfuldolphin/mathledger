# Lean Control Sandbox Plan

> **STATUS: PHASE II — NOT IMPLEMENTED / NOT ACTIVE IN FO PIPELINE**
>
> This document describes **planned future work** for Lean sandboxing.
> None of this functionality is implemented or used in Evidence Pack v1.
> The Phase I First Organism runs do **not** use sandboxed Lean execution.
> Current Lean verification uses the standard `worker.py` → `lean_mode.py`
> pipeline with basic timeout handling only.

---

## Overview

The Lean Control Sandbox provides an isolated execution environment for Lean 4 proof
verification jobs. It enforces strict security boundaries, prevents resource exhaustion,
and ensures reproducible, hermetic builds by controlling file system access, cache
artifacts, and execution timeouts.

This module integrates with the existing `lean_mode.py` verification strategy and
`worker.py` job execution pipeline to provide an additional layer of isolation when
running Lean in production environments.

---

## 1. Sandbox Invocation Model

### 1.1 Execution Flow

```
Worker              LeanSandbox                  Lean/Lake
  │                     │                            │
  │─prepare_environment()─▶│                         │
  │◀──sandbox_context─────│                          │
  │                       │                          │
  │─execute_job_safe(stmt)─▶│                        │
  │                       │──prepare_isolated_dir()──▶│
  │                       │──set_env_restrictions()──▶│
  │                       │──spawn_lake_build()──────▶│
  │                       │◀────result───────────────│
  │                       │──capture_outputs()───────▶│
  │◀───LeanJobResult──────│                          │
  │                       │                          │
  │─cleanup()─────────────▶│                         │
  │                       │──remove_temp_files()─────▶│
  │                       │──verify_no_leakage()─────▶│
  │◀───cleanup_report─────│                          │
```

### 1.2 Sandbox Entry Points

| Method                     | Purpose                                      |
|----------------------------|----------------------------------------------|
| `prepare_environment()`    | Initialize sandbox directories and env vars |
| `execute_job_safe()`       | Run a single Lean job within the sandbox    |
| `cleanup()`                | Remove temporary files and build artifacts  |
| `verify_sandbox_integrity()`| Confirm no files escaped sandbox boundaries |

---

## 2. Allowed Commands

### 2.1 Whitelisted Executables

Only the following commands may be invoked within the sandbox:

| Command      | Purpose                          | Restrictions              |
|--------------|----------------------------------|---------------------------|
| `lake`       | Lean build orchestrator          | Only `build` subcommand   |
| `lean`       | Lean compiler (invoked by lake)  | Via lake only             |
| `leanc`      | Lean C compiler                  | Via lake only             |

### 2.2 Blocked Commands

The sandbox MUST prevent execution of:

- Shell interpreters (`cmd.exe`, `powershell.exe`, `bash`, `sh`)
- Network utilities (`curl`, `wget`, `Invoke-WebRequest`)
- File transfer tools (`scp`, `rsync`, `ftp`)
- Process manipulation (`kill`, `taskkill`, `pkill`)
- Arbitrary binaries outside the Lean toolchain

### 2.3 Command Invocation Rules

1. **Subcommand Restriction**: Only `lake build <module>` is permitted
2. **No Shell Expansion**: Commands must be invoked as argument lists, never via shell
3. **No Environment Inheritance**: Only explicitly allowed env vars are passed
4. **Working Directory Lock**: CWD must be the isolated job directory

---

## 3. Timeout Rules

### 3.1 Timeout Hierarchy

| Timeout Type        | Default   | Configurable Via              | Scope                 |
|---------------------|-----------|-------------------------------|-----------------------|
| Job Timeout         | 90s       | `LEAN_BUILD_TIMEOUT`          | Single lake build     |
| Cleanup Timeout     | 10s       | `LEAN_CLEANUP_TIMEOUT`        | Post-job cleanup      |
| Session Timeout     | 600s      | `LEAN_SESSION_TIMEOUT`        | Entire sandbox session|
| Kill Grace Period   | 5s        | `LEAN_KILL_GRACE`             | SIGTERM → SIGKILL gap |

### 3.2 Timeout Behavior

1. **Soft Timeout**: Send SIGTERM (or equivalent on Windows) when job timeout expires
2. **Hard Timeout**: Force kill after grace period if process hasn't terminated
3. **Cleanup on Timeout**: Timeout exits trigger full cleanup sequence
4. **Timeout Logging**: All timeout events are logged with job ID and duration

### 3.3 Timeout Result Encoding

Timeout results are encoded in the `CompletedProcess` returncode:

| Returncode | Meaning                      |
|------------|------------------------------|
| 124        | Job timeout (soft)           |
| 137        | Job killed (SIGKILL/hard)    |
| 125        | Sandbox setup timeout        |

---

## 4. Proof File Isolation Rules

### 4.1 Directory Structure

```
{SANDBOX_ROOT}/
├── jobs/
│   └── job_{uuid}/           # Per-job isolation directory
│       ├── job_{uuid}.lean   # Generated proof file
│       └── .lake/            # Ephemeral lake cache (job-scoped)
├── shared/
│   └── ML/                   # Shared MathLedger Lean library (read-only)
│       └── Taut.lean         # Tautology checker
└── logs/
    └── job_{uuid}.log        # Execution log (optional)
```

### 4.2 File System Permissions

| Path Pattern              | Read | Write | Execute | Notes                    |
|---------------------------|------|-------|---------|--------------------------|
| `{SANDBOX_ROOT}/jobs/`    | ✓    | ✓     | ✗       | Job working directories  |
| `{SANDBOX_ROOT}/shared/`  | ✓    | ✗     | ✗       | Shared library (immutable)|
| `{LEAN_HOME}/`            | ✓    | ✗     | ✓       | Lean toolchain binaries  |
| `{MATHLIB_CACHE}/`        | ✓    | ✗     | ✗       | Pre-built Mathlib oleans |
| Everything else           | ✗    | ✗     | ✗       | Blocked by sandbox       |

### 4.3 File Naming Conventions

1. **Job Files**: `job_{uuid}.lean` where UUID is deterministic (see `repro.determinism`)
2. **Build Artifacts**: Stored under `.lake/build/` within job directory
3. **Log Files**: `job_{uuid}.log` with ISO-8601 timestamps

### 4.4 Cross-Job Isolation

- Each job runs in a fresh directory with no access to other jobs
- Shared library is mounted read-only
- No symbolic links allowed within job directories
- File handles are not inherited between jobs

---

## 5. Cache Artifact Prevention

### 5.1 Problem Statement

Lean/Lake generates cache artifacts that can:
- Leak information between jobs
- Consume unbounded disk space
- Cause non-deterministic builds
- Create security vulnerabilities via cached code execution

### 5.2 Cache Locations to Control

| Cache Type           | Default Location              | Sandbox Handling            |
|----------------------|-------------------------------|-----------------------------|
| Lake build cache     | `{project}/.lake/build/`      | Ephemeral per-job directory |
| Lake packages        | `{project}/.lake/packages/`   | Read-only shared mount      |
| Lean home cache      | `~/.elan/` or `$LEAN_HOME`    | Read-only, pre-provisioned  |
| Mathlib cache        | `~/.cache/mathlib/`           | Read-only shared mount      |
| Temp files           | `$TEMP` / `$TMP`              | Redirected to job sandbox   |

### 5.3 Environment Variable Overrides

To prevent cache leakage, the sandbox sets:

```bash
# Redirect caches to sandbox-controlled locations
LAKE_HOME={SANDBOX_ROOT}/jobs/job_{uuid}/.lake
XDG_CACHE_HOME={SANDBOX_ROOT}/jobs/job_{uuid}/.cache
TEMP={SANDBOX_ROOT}/jobs/job_{uuid}/.tmp
TMP={SANDBOX_ROOT}/jobs/job_{uuid}/.tmp

# Disable network operations
LAKE_NO_FETCH=1
LAKE_OFFLINE=1

# Disable auto-updates
ELAN_AUTO_UPDATE=false
```

### 5.4 Post-Job Cleanup

After each job completes:

1. **Immediate**: Remove `job_{uuid}.lean` source file
2. **Immediate**: Remove `.lake/build/` directory for the job
3. **Deferred**: Remove `.olean` and `.c` artifacts from build cache
4. **Periodic**: Prune job directories older than 1 hour

### 5.5 Integrity Verification

After cleanup, verify:

1. No files exist outside the sandbox root
2. No new files in shared directories
3. No modifications to read-only mounts
4. Total disk usage within configured limits

---

## 6. Security Considerations

### 6.1 Threat Model

| Threat                          | Mitigation                              |
|---------------------------------|-----------------------------------------|
| Malicious Lean code execution   | Timeout + process isolation             |
| File system escape              | Strict path validation + chroot         |
| Resource exhaustion             | Memory limits + disk quotas             |
| Information leakage via cache   | Ephemeral directories + cleanup         |
| Network exfiltration            | Blocked network + offline mode          |
| Side-channel via timing         | (Not addressed at this layer)           |

### 6.2 Windows-Specific Considerations

On Windows (primary development platform for MathLedger):

- Use `subprocess.CREATE_NO_WINDOW` to prevent console allocation
- Use `subprocess.CREATE_NEW_PROCESS_GROUP` for clean termination
- Handle long paths via `\\?\` prefix if needed
- Use `shutil.rmtree` with `onerror` handler for locked files

### 6.3 Future Enhancements

1. **Container Isolation**: Docker/Podman-based sandboxing for stronger isolation
2. **cgroups/Resource Limits**: Memory and CPU limits on Linux
3. **Capability Dropping**: Reduce process privileges on Unix systems
4. **Seccomp Filtering**: Restrict system calls on Linux

---

## 7. Integration with Existing Modules

### 7.1 lean_mode.py Integration

The sandbox wraps the existing build runners:

```python
# Current flow (lean_mode.py)
runner = get_build_runner(LeanMode.FULL)
result = runner(module_name)

# With sandbox (lean_control_sandbox.py)
with LeanSandbox() as sandbox:
    sandbox.prepare_environment()
    result = sandbox.execute_job_safe(statement, build_runner=runner)
    sandbox.cleanup()
```

### 7.2 worker.py Integration

The worker uses the sandbox for all non-axiom jobs:

```python
# In worker main loop
if not is_axiom:
    with LeanSandbox(config=sandbox_config) as sandbox:
        lean_job = sandbox.execute_job_safe(stmt.ascii_pretty)
```

### 7.3 Configuration Precedence

1. Environment variables (highest priority)
2. Sandbox constructor parameters
3. `LeanSandboxConfig.from_env()` defaults
4. Hardcoded fallbacks (lowest priority)

---

## 8. Monitoring and Observability

### 8.1 Metrics

| Metric                          | Type      | Description                    |
|---------------------------------|-----------|--------------------------------|
| `sandbox_jobs_total`            | Counter   | Total jobs executed            |
| `sandbox_jobs_timeout`          | Counter   | Jobs that timed out            |
| `sandbox_cleanup_failures`      | Counter   | Failed cleanup operations      |
| `sandbox_disk_usage_bytes`      | Gauge     | Current sandbox disk usage     |
| `sandbox_job_duration_seconds`  | Histogram | Job execution time             |

### 8.2 Logging

All sandbox operations log to the `LeanSandbox` logger with structured fields:

```
[SANDBOX] job_id=abc123 event=prepare status=ok
[SANDBOX] job_id=abc123 event=execute status=ok duration_ms=1234
[SANDBOX] job_id=abc123 event=cleanup status=ok files_removed=3
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

- `test_sandbox_prepare()`: Directory creation and permissions
- `test_sandbox_execute()`: Job execution with mock runner
- `test_sandbox_cleanup()`: File removal and verification
- `test_sandbox_timeout()`: Timeout handling

### 9.2 Integration Tests

- `test_sandbox_with_lean()`: Real Lean execution in sandbox
- `test_sandbox_isolation()`: Cross-job isolation verification
- `test_sandbox_recovery()`: Crash recovery and cleanup

### 9.3 Security Tests

- `test_sandbox_path_escape()`: Attempt to write outside sandbox
- `test_sandbox_command_injection()`: Attempt to run disallowed commands
- `test_sandbox_resource_limits()`: Verify timeout enforcement

---

## 10. Implementation Phases

### Phase 1: Basic Isolation (Current Skeleton)

- [ ] `LeanSandbox` class structure
- [ ] `prepare_environment()` - directory setup
- [ ] `execute_job_safe()` - wrapped execution
- [ ] `cleanup()` - file removal
- [ ] `verify_sandbox_integrity()` - basic checks

### Phase 2: Cache Control

- [ ] Environment variable overrides
- [ ] Cache redirection
- [ ] Automatic cleanup of build artifacts

### Phase 3: Timeout Enforcement

- [ ] Soft timeout with SIGTERM
- [ ] Hard timeout with SIGKILL
- [ ] Grace period handling

### Phase 4: Security Hardening

- [ ] Path validation
- [ ] Command whitelist enforcement
- [ ] Resource limit integration (OS-specific)

---

## References

- MathLedger Whitepaper §3.1 (Verification Ladder)
- MathLedger Whitepaper §4.2 (Dual Root Attestation)
- `backend/lean_mode.py` - Three-mode verification strategy
- `backend/worker.py` - Current Lean job execution
- `backend/lean_interface.py` - Statement sanitization

---

## Phase II Disclaimer

**This entire document describes future work not included in Evidence Pack v1.**

The Lean sandbox is planned infrastructure for production hardening. It is:
- **Not implemented**: All methods raise `NotImplementedError`
- **Not integrated**: `worker.py` does not call `LeanSandbox`
- **Not tested**: No sandbox-related tests exist in the test suite
- **Not required** for Phase I claims: FO attestation and RFL experiments
  do not depend on sandbox isolation

### What Phase I Actually Uses

Current Lean verification in Phase I relies on:
1. `lean_mode.py` three-mode strategy (MOCK/DRY_RUN/FULL)
2. Basic `subprocess.run()` with timeout parameter
3. File cleanup via `remove_build_artifacts()` in `worker.py`
4. No process isolation beyond OS defaults

### RFL Experiments and Lean

**RFL runs do not involve Lean at all.**

- All RFL experiments (baseline, policy-active, including extended 1000-cycle runs)
  operate with `ML_LEAN_MODE=mock`
- Lean verification is disabled; the worker produces deterministic abstention signatures
- The Lean sandbox has zero interaction with RFL logs, attestations, or metrics
- RFL policy decisions are based on abstention counts, not Lean proof outcomes
- This sandbox specification is therefore **completely irrelevant** to any RFL evidence

### Future Work (For Discussion/Limitations Section)

> "Lean sandbox work is planned but not part of the Phase I prototype.
> Production deployment would benefit from stronger isolation guarantees
> including per-job directories, cache artifact prevention, and container-based
> sandboxing for defense against malicious Lean code."

---

## Phase II Uplift: Lean Participation Design

> **STATUS: PHASE II DESIGN ONLY — NOT IMPLEMENTED**
>
> This section describes how future uplift experiments *might* safely reintroduce
> Lean verification. None of this is implemented. Phase I remains Lean-disabled.

### 11.1 Two-Tier Verifier Ladder

Future uplift experiments could use a tiered verification approach:

| Tier | Verifier Type | Characteristics | Phase I Status |
|------|---------------|-----------------|----------------|
| **Tier 1** | Fast deterministic checks | Truth tables, SAT/SMT, simple decision procedures | **Active** (truth_table_is_tautology) |
| **Tier 2** | Lean kernel verification | Full proof checking with Lean 4 | **Disabled** (ML_LEAN_MODE=mock) |

#### Tier 1: Fast Deterministic Checks (Current)

- Truth table evaluation via `normalization.taut.truth_table_is_tautology()`
- Deterministic: same input always produces same output
- Bounded time: O(2^n) for n variables, capped at small n
- No external dependencies beyond Python runtime
- **This is what Phase I actually uses for any "verification"**

#### Tier 2: Lean Kernel Calls (Phase II Only)

- Full Lean 4 type-checking via `lake build`
- Requires Lean toolchain installation
- Non-deterministic factors: filesystem state, cache hits, timing
- Unbounded time without explicit timeouts
- **NOT USED IN PHASE I — all runs use ML_LEAN_MODE=mock**

### 11.2 Environment Variables for Tier 2 Gating

Future Phase II experiments would use explicit environment variables:

```bash
# Phase I (current) — Lean completely disabled
ML_LEAN_MODE=mock
RFL_VERIFIER_TIER=1
# Result: Only Tier 1 checks, deterministic abstention signatures

# Phase II (future) — Lean enabled with safeguards
ML_LEAN_MODE=full
RFL_VERIFIER_TIER=2
RFL_LEAN_TIMEOUT=30
RFL_LEAN_DETERMINISTIC=1
# Result: Tier 2 checks attempted, with structured failure modes
```

| Variable | Values | Description |
|----------|--------|-------------|
| `ML_LEAN_MODE` | `mock`, `dry_run`, `full` | Existing mode selector (Phase I: always `mock`) |
| `RFL_VERIFIER_TIER` | `1`, `2` | Maximum tier to attempt (Phase I: always `1`) |
| `RFL_LEAN_TIMEOUT` | seconds | Per-statement Lean timeout (Phase II only) |
| `RFL_LEAN_DETERMINISTIC` | `0`, `1` | Force deterministic Lean options (Phase II only) |
| `RFL_LEAN_KERNEL_VERSION` | version string | Pin exact Lean version (Phase II only) |

### 11.3 Logging and Attestation Requirements

When Lean is eventually enabled, every verification must log:

```json
{
  "statement_hash": "abc123...",
  "verifier_tier_attempted": 2,
  "verifier_tier_succeeded": 2,
  "lean_involved": true,
  "lean_mode": "full",
  "lean_version": "4.3.0",
  "lean_timeout_ms": 30000,
  "lean_actual_ms": 1234,
  "lean_returncode": 0,
  "lean_stdout_hash": "def456...",
  "lean_stderr_hash": "789abc...",
  "deterministic_settings": true,
  "outcome": "verified"
}
```

**Critical attestation fields for Phase II:**

- `lean_involved: true/false` — Always distinguishable from Tier 1-only runs
- `lean_version` — Exact kernel version for reproducibility claims
- `lean_stdout_hash`, `lean_stderr_hash` — Content-addressable output capture
- `deterministic_settings` — Whether determinism flags were active

**Phase I attestations have:**
- `lean_involved: false` (always)
- `verifier_tier_succeeded: 1` (always)
- `is_mock_abstention: true` (always, when "Lean" path taken)

### 11.4 Safe Lean Participation Profile

For Phase II uplift experiments, Lean must run with a "safe profile":

#### 11.4.1 Deterministic Kernel Configuration

```bash
# Pin exact Lean version (no auto-updates)
ELAN_AUTO_UPDATE=false
LEAN_VERSION=v4.3.0  # or whatever version is validated

# Disable non-deterministic features
LEAN_DETERMINISTIC=1  # hypothetical flag for future Lean
LAKE_NO_CACHE=1       # disable build caching
LAKE_OFFLINE=1        # no network fetches

# Force single-threaded execution
LEAN_NUM_THREADS=1    # eliminate parallelism non-determinism
```

#### 11.4.2 Bounded Resource Usage Per Cycle

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| Time per statement | 30s | subprocess timeout |
| Memory per statement | 2GB | ulimit / cgroups (Linux) or Job Objects (Windows) |
| Disk per statement | 100MB | Sandbox directory quota |
| Total Lean time per cycle | 5min | Cycle-level budget tracking |
| Statements with Lean per cycle | 10 | Counter with early-exit |

#### 11.4.3 Structured Failure Modes

**Timeouts become structured abstentions, never silent successes:**

| Failure Mode | Outcome | Attestation |
|--------------|---------|-------------|
| Lean timeout (soft) | `abstain_timeout` | `lean_returncode: 124, outcome: abstain` |
| Lean timeout (hard kill) | `abstain_killed` | `lean_returncode: 137, outcome: abstain` |
| Lean crash | `abstain_crash` | `lean_returncode: <nonzero>, outcome: abstain` |
| Lean proof failure | `refuted` | `lean_returncode: 0, stdout: "FAILURE", outcome: refuted` |
| Lean proof success | `verified` | `lean_returncode: 0, stdout: "SUCCESS", outcome: verified` |

**Key guarantee:** A timeout or crash NEVER produces `outcome: verified`.
The only path to `verified` is `returncode=0` AND explicit success marker in stdout.

### 11.5 Phase I vs Phase II Summary

| Aspect | Phase I (Current) | Phase II (Future) |
|--------|-------------------|-------------------|
| ML_LEAN_MODE | `mock` (always) | `full` (optional) |
| Verifier tiers used | Tier 1 only | Tier 1 + Tier 2 |
| Lean kernel invoked | Never | Conditionally |
| Attestation includes Lean hashes | No (mock signatures) | Yes (real output) |
| Determinism | Fully deterministic | Best-effort deterministic |
| This sandbox code | Not used | Would be used |
| RFL policy inputs | Abstention counts only | Abstention + verification outcomes |

**Phase I ground truth remains unchanged:**
- No Phase I RFL run uses real Lean proofs
- All Phase I RFL logs are Lean-disabled + abstention only
- This sandbox code is unused in Phase I
- All Phase I claims are independent of Lean verification outcomes

---

## 12. Phase II — Verifier Budget Envelope

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This section defines verifier budget parameters, timeout behavior, and abstention
> semantics for the four asymmetric uplift slices defined in `curriculum_uplift_phase2.yaml`.
> None of this is implemented or active. Phase I runs remain Lean-disabled throughout.

### 12.1 Design Rationale

Phase II introduces asymmetric environments where candidate ordering affects outcomes.
Unlike Phase I's hermetic truth-table-only verification, Phase II *may* optionally
reintroduce Lean verification under strict budget constraints. The Verifier Budget
Envelope ensures:

1. **Bounded resource consumption** — No single cycle exhausts compute
2. **Deterministic timeout semantics** — Timeouts map to structured abstentions
3. **Slice-specific tuning** — Each uplift slice has tailored budget parameters
4. **Fail-safe defaults** — Budget exhaustion → abstention, never silent failure

### 12.2 Budget Parameters by Slice

Each Phase II uplift slice defines a **verifier budget envelope** controlling
Lean participation (if enabled) and truth-table verification costs.

| Slice | `lean_timeout_s` | `taut_timeout_s` | `max_lean_calls_per_cycle` | `cycle_budget_s` | `lean_enabled` |
|-------|------------------|------------------|----------------------------|------------------|----------------|
| `slice_uplift_goal` | 0.0 | 0.10 | 0 | 5.0 | false |
| `slice_uplift_sparse` | 0.0 | 0.12 | 0 | 6.0 | false |
| `slice_uplift_tree` | 0.0 | 0.10 | 0 | 4.0 | false |
| `slice_uplift_dependency` | 0.0 | 0.12 | 0 | 6.0 | false |

**Column Definitions:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lean_timeout_s` | Per-statement Lean timeout (0 = disabled) | 0.0 |
| `taut_timeout_s` | Per-statement truth-table timeout | 0.10 |
| `max_lean_calls_per_cycle` | Maximum Lean kernel invocations per cycle (0 = disabled) | 0 |
| `cycle_budget_s` | Total wall-clock time budget per cycle | 5.0 |
| `lean_enabled` | Whether Lean verification is permitted | false |

**Note:** All Phase II uplift slices currently disable Lean (`lean_enabled: false`).
Future Phase IIb experiments may introduce Lean-enabled variants with non-zero
`lean_timeout_s` and `max_lean_calls_per_cycle`.

### 12.2.1 Chain-Depth Analysis and Max-Depth Considerations

The Phase II goal includes "chain-depth analysis" as a key objective, distinct from the direct verifier budget. This analysis aims to understand the complexity and length of logical dependencies within derived statements. While not a direct *verifier budget parameter* (like `lean_timeout_s` or `RFL_LEAN_MEMORY_MB`), the output of chain-depth analysis can inform future verifier budget allocations or adaptive strategies.

**Current Stance:**
- **No direct "max-depth" verifier parameter:** The current verifier budget envelope does not include a configurable `max-depth` parameter for the Lean kernel's proof search or recursion limits. These limits are implicitly handled by the `lean_timeout_s` and `RFL_LEAN_MEMORY_MB` budget parameters.
- **Chain-depth as a diagnostic:** Chain-depth is primarily a metric for diagnostic and analytical purposes within Phase II, intended to characterize the structural properties of derived statements. It helps in understanding the uplift environments and the type of formulas generated.
- **Future implications:** The insights gained from chain-depth analysis (e.g., identification of overly complex or deep derivation chains) *may* lead to the introduction of explicit `max-depth` verifier constraints in future phases (e.g., Phase IIb or Phase III) to enforce computational bounds based on structural complexity. However, this is not part of the current Phase II verifier budget envelope.

### 12.3 Budget Enforcement Rules

#### 12.3.1 Cycle-Level Budget

Each derivation cycle operates under a wall-clock budget (`cycle_budget_s`).

```
cycle_start_time = now()
while candidates_remaining AND (now() - cycle_start_time) < cycle_budget_s:
    candidate = next_candidate()
    result = verify_within_budget(candidate)
    log_result(result)

if (now() - cycle_start_time) >= cycle_budget_s:
    log_event("CYCLE_BUDGET_EXHAUSTED", remaining_candidates=len(candidates))
```

**Budget exhaustion behavior:**
- Remaining candidates are **not verified** in this cycle
- They receive outcome `budget_skip` (distinct from `abstain` or `timeout`)
- Cycle attestation includes `budget_exhausted: true` flag
- Policy update uses only completed verifications (no partial credit)

#### 12.3.2 Per-Statement Budget

Each statement verification has an individual timeout:

| Verifier Tier | Timeout Source | Behavior on Expiry |
|---------------|----------------|-------------------|
| Tier 1 (truth-table) | `taut_timeout_s` | `outcome: abstain_timeout` |
| Tier 2 (Lean) | `lean_timeout_s` | `outcome: abstain_timeout` |

**Per-statement timeout semantics:**
- Timer starts when verification begins
- On expiry: immediate termination, structured abstention logged
- No partial proofs accepted
- Timeout duration logged for telemetry

#### 12.3.3 Lean Call Quota

When `lean_enabled: true` (future Phase IIb):

```
lean_calls_this_cycle = 0

def verify_with_lean(statement):
    global lean_calls_this_cycle
    if lean_calls_this_cycle >= max_lean_calls_per_cycle:
        return Outcome(status="quota_exceeded", tier_attempted=2, tier_succeeded=1)

    lean_calls_this_cycle += 1
    result = invoke_lean_kernel(statement, timeout=lean_timeout_s)
    return result
```

**Quota exhaustion behavior:**
- Statement falls back to Tier 1 verification only
- Outcome includes `lean_quota_exceeded: true`
- Policy sees this as Tier 1 result (not abstention)

### 12.4 Timeout vs. Abstention Semantics

Phase II requires precise distinction between different non-success outcomes.
Each outcome type has distinct semantics for policy learning.

#### 12.4.1 Outcome Taxonomy

| Outcome | Code | Policy Signal | Attestation Field |
|---------|------|---------------|-------------------|
| **verified** | `V` | Positive reward | `outcome: verified` |
| **refuted** | `R` | Negative reward | `outcome: refuted` |
| **abstain_timeout** | `AT` | Neutral (no signal) | `outcome: abstain, reason: timeout` |
| **abstain_crash** | `AC` | Neutral (no signal) | `outcome: abstain, reason: crash` |
| **abstain_complexity** | `AX` | Neutral (no signal) | `outcome: abstain, reason: complexity` |
| **quota_exceeded** | `QE` | Tier 1 result used | `outcome: <tier1_result>, quota_exceeded: true` |
| **budget_skip** | `BS` | Not observed | `outcome: budget_skip` |

#### 12.4.2 Policy Update Rules

The RFL policy update function treats outcomes differently:

```python
def policy_update(outcome: Outcome) -> PolicyDelta:
    if outcome.status == "verified":
        return positive_reward(outcome.statement_features)
    elif outcome.status == "refuted":
        return negative_reward(outcome.statement_features)
    elif outcome.status in ("abstain_timeout", "abstain_crash", "abstain_complexity"):
        return no_update()  # Abstentions are uninformative
    elif outcome.status == "quota_exceeded":
        # Use the Tier 1 result that was returned
        return policy_update(outcome.tier1_fallback)
    elif outcome.status == "budget_skip":
        return no_update()  # Not observed, cannot learn
    else:
        raise UnknownOutcomeError(outcome)
```

**Critical invariant:** Timeouts and crashes NEVER produce `verified` or `refuted`.
They are information-theoretically uninformative and must be excluded from policy
gradient estimation.

#### 12.4.3 Abstention Rate Calculation

For Phase II experiments, abstention rate is calculated as:

```
abstention_rate = (AT + AC + AX) / (V + R + AT + AC + AX)
```

**Excluded from denominator:**
- `quota_exceeded` — Not an abstention; Tier 1 result was obtained
- `budget_skip` — Not attempted; cycle budget exhausted

This ensures abstention rate reflects verifier capability, not resource limits.

### 12.5 Lean Safety Configuration (Phase IIb Only)

When Lean is eventually enabled for Phase IIb experiments, the following
safety configuration applies:

#### 12.5.1 Environment Variables

```bash
# ─────────────────────────────────────────────────────────────────────────────
# PHASE IIb LEAN CONFIGURATION — NOT ACTIVE IN PHASE II UPLIFT SLICES
# ─────────────────────────────────────────────────────────────────────────────

# Master enable (must be explicitly set)
RFL_LEAN_ENABLED=false                    # Phase II: always false
                                          # Phase IIb: may be true

# Per-statement limits
RFL_LEAN_TIMEOUT_S=30                     # Hard timeout per Lean invocation
RFL_LEAN_MEMORY_MB=2048                   # Memory limit (via ulimit/cgroups)
RFL_LEAN_DISK_MB=100                      # Disk quota for build artifacts

# Per-cycle limits
RFL_MAX_LEAN_CALLS_PER_CYCLE=10           # Quota for Lean kernel calls
RFL_CYCLE_BUDGET_S=120                    # Total cycle wall-clock budget

# Determinism flags
RFL_LEAN_DETERMINISTIC=1                  # Force single-threaded, no cache
LEAN_NUM_THREADS=1                        # Eliminate parallelism variance
LAKE_NO_CACHE=1                           # Disable build cache
LAKE_OFFLINE=1                            # No network fetches

# Version pinning
LEAN_VERSION=v4.3.0                       # Exact version for reproducibility
ELAN_AUTO_UPDATE=false                    # Prevent version drift
```

#### 12.5.2 Structured Timeout Handling

When Lean times out, the sandbox produces a structured result:

```json
{
  "statement_hash": "abc123...",
  "verifier_tier_attempted": 2,
  "verifier_tier_succeeded": 0,
  "lean_involved": true,
  "lean_timeout_configured_s": 30,
  "lean_actual_elapsed_s": 30.0,
  "lean_returncode": 124,
  "lean_termination": "timeout_soft",
  "outcome": "abstain",
  "abstention_reason": "timeout",
  "tier1_fallback_attempted": false,
  "attestation_valid": true
}
```

**Timeout return codes:**
| Code | Meaning | Sandbox Behavior |
|------|---------|------------------|
| 124 | Soft timeout (SIGTERM) | Process terminated gracefully |
| 137 | Hard timeout (SIGKILL) | Process force-killed after grace period |
| 125 | Setup timeout | Sandbox preparation exceeded limit |

#### 12.5.3 Crash vs. Timeout Disambiguation

The sandbox distinguishes crashes from timeouts:

| Signal | Cause | Attestation |
|--------|-------|-------------|
| Timeout (124/137) | Wall-clock limit exceeded | `abstention_reason: timeout` |
| Crash (non-zero, not 124/137) | Lean internal error | `abstention_reason: crash` |
| OOM (137 + specific signal) | Memory limit exceeded | `abstention_reason: oom` |

**All three map to abstention.** The distinction is for diagnostics only.

### 12.6 Slice-Specific Verifier Configuration

Each Phase II uplift slice may override default budget parameters.
Configuration is loaded from `curriculum_uplift_phase2.yaml`:

```yaml
# Example: slice_uplift_goal verifier budget
slice_uplift_goal:
  params:
    timeout_s: 0.1              # truth-table timeout
    lean_timeout_s: 0.0         # Lean disabled

  # Phase II: Verifier Budget Envelope
  verifier_budget:
    tier1_timeout_s: 0.10       # Per-statement taut check
    tier2_timeout_s: 0.0        # Lean disabled (0 = skip)
    max_tier2_calls: 0          # No Lean calls permitted
    cycle_budget_s: 5.0         # Total cycle wall-clock
    budget_exhaustion: "skip"   # skip | fail | extend
    timeout_policy: "abstain"   # abstain | retry_tier1 | fail
```

**Configuration precedence:**
1. Environment variables (highest)
2. Slice-specific `verifier_budget` block
3. Global defaults from `lean_control_sandbox_plan.md`
4. Hardcoded fallbacks (lowest)

### 12.7 Telemetry and Observability

Phase II verifier budget tracking produces the following metrics:

#### 12.7.1 Per-Cycle Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `verifier_tier1_calls` | Counter | Truth-table verifications attempted |
| `verifier_tier1_verified` | Counter | Tier 1 successes |
| `verifier_tier1_refuted` | Counter | Tier 1 refutations |
| `verifier_tier1_timeout` | Counter | Tier 1 timeouts |
| `verifier_tier2_calls` | Counter | Lean verifications attempted (Phase IIb) |
| `verifier_tier2_quota_exhausted` | Counter | Cycles hitting Lean quota |
| `verifier_cycle_budget_exhausted` | Counter | Cycles hitting wall-clock budget |
| `verifier_statements_skipped` | Counter | Statements skipped due to budget |

#### 12.7.2 Per-Statement Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `verifier_tier1_duration_ms` | Histogram | Truth-table verification latency |
| `verifier_tier2_duration_ms` | Histogram | Lean verification latency (Phase IIb) |
| `verifier_abstention_reason` | Label | timeout / crash / complexity / oom |

#### 12.7.3 Attestation Fields

Each cycle attestation includes:

```json
{
  "cycle": 42,
  "verifier_budget": {
    "tier1_timeout_s": 0.10,
    "tier2_timeout_s": 0.0,
    "max_tier2_calls": 0,
    "cycle_budget_s": 5.0
  },
  "verifier_usage": {
    "tier1_calls": 32,
    "tier1_verified": 8,
    "tier1_refuted": 3,
    "tier1_abstained": 21,
    "tier2_calls": 0,
    "tier2_quota_remaining": 0,
    "cycle_elapsed_s": 3.2,
    "cycle_budget_remaining_s": 1.8,
    "statements_skipped": 0
  },
  "budget_flags": {
    "tier2_disabled": true,
    "quota_exhausted": false,
    "cycle_budget_exhausted": false
  }
}
```

### 12.8 Governance and Compliance

#### 12.8.1 PREREG Alignment

This section aligns with `PREREG_UPLIFT_U2.yaml`:
- All budget parameters are deterministic
- Timeout behavior is fully specified
- Abstention semantics match preregistration expectations
- No hidden state or non-reproducible variance

#### 12.8.2 Phase Separation

| Artifact | Phase I | Phase II | Phase IIb |
|----------|---------|----------|-----------|
| Truth-table verifier | Active | Active | Active |
| Lean verifier | Disabled | Disabled | Optional |
| Budget enforcement | N/A | Active | Active |
| Timeout→abstention | N/A | Specified | Specified |
| This section | N/A | Applies | Applies |

#### 12.8.3 Audit Trail

All budget-related decisions are logged:

```
[BUDGET] cycle=42 event=init tier1_timeout=0.10s tier2_timeout=0.00s budget=5.0s
[BUDGET] cycle=42 stmt=abc123 event=tier1_start
[BUDGET] cycle=42 stmt=abc123 event=tier1_complete duration_ms=45 outcome=verified
[BUDGET] cycle=42 stmt=def456 event=tier1_start
[BUDGET] cycle=42 stmt=def456 event=tier1_timeout duration_ms=100 outcome=abstain
[BUDGET] cycle=42 event=complete tier1_calls=32 elapsed_s=3.2 skipped=0
```

### 12.9 Future Work: Phase IIb Lean Reintroduction

When Lean is reintroduced in Phase IIb:

1. **Gradual enablement:** Start with `max_tier2_calls=1` per cycle
2. **Conservative timeouts:** Begin with 60s timeout, reduce based on data
3. **Fallback required:** All Tier 2 failures must fall back to Tier 1
4. **Separate attestation:** Lean-enabled runs use distinct attestation schema
5. **Preregistration required:** New `PREREG_UPLIFT_U2b.yaml` must be filed

**Phase IIb is not part of this specification. This section is forward-looking only.**

---

## 13. Phase II Safety Summary

> **STATUS: PHASE II — NOT RUN IN PHASE I**

### 13.1 Key Safety Properties

| Property | Guarantee |
|----------|-----------|
| **Timeout → Abstention** | Timeouts never produce `verified` |
| **Crash → Abstention** | Crashes never produce `verified` |
| **Budget → Skip** | Budget exhaustion skips, never fabricates |
| **Quota → Fallback** | Lean quota exceeded → Tier 1 result used |
| **Determinism** | Same seed + config → same outcomes |

### 13.2 Failure Mode Matrix

| Failure | Outcome | Policy Signal | Attestation |
|---------|---------|---------------|-------------|
| Tier 1 timeout | abstain | neutral | `reason: timeout` |
| Tier 2 timeout | abstain | neutral | `reason: timeout` |
| Tier 2 crash | abstain | neutral | `reason: crash` |
| Tier 2 OOM | abstain | neutral | `reason: oom` |
| Tier 2 quota | tier1_result | tier1_signal | `quota_exceeded: true` |
| Cycle budget | skip | none | `budget_exhausted: true` |

### 13.3 Non-Goals

This specification does NOT address:
- Side-channel timing attacks (out of scope)
- Malicious Lean code execution (sandbox isolation in §4)
- Network-based threats (offline mode enforced)
- Multi-tenant isolation (single-user system)

---

## 14. Phase II Slice-Specific Verifier Budgets

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> This section provides per-slice verifier budget configurations aligned with
> `curriculum_uplift_phase2.yaml` and `PREREG_UPLIFT_U2.yaml`.

### 14.1 Slice Budget Matrix

The following matrix defines verifier budgets for each Phase II uplift slice:

| Slice | `taut_timeout_s` | `lean_timeout_s` | `max_lean_calls` | `cycle_budget_s` | `max_candidates` | `lean_enabled` |
|-------|------------------|------------------|------------------|------------------|------------------|----------------|
| `slice_uplift_goal` | 0.10 | 0.0 | 0 | 5.0 | 40 | false |
| `slice_uplift_sparse` | 0.12 | 0.0 | 0 | 6.0 | 40 | false |
| `slice_uplift_tree` | 0.10 | 0.0 | 0 | 4.0 | 30 | false |
| `slice_uplift_dependency` | 0.12 | 0.0 | 0 | 6.0 | 40 | false |

**Rationale:**
- All slices disable Lean (`lean_enabled: false`) for Phase II hermetic runs
- `slice_uplift_sparse` and `slice_uplift_dependency` have slightly higher `taut_timeout_s` (0.12s)
  due to 5-atom formula complexity
- `slice_uplift_tree` has tighter `cycle_budget_s` (4.0s) and `max_candidates` (30)
  because chain-depth derivation is more focused

### 14.2 Per-Slice Verifier Configuration YAML

```yaml
# Phase II Verifier Budget Envelope Configuration
# File: config/verifier_budget_phase2.yaml
# STATUS: PHASE II — NOT RUN IN PHASE I

version: 1
phase: "II"

# Global defaults (overridden by slice-specific)
defaults:
  lean_enabled: false
  taut_timeout_s: 0.10
  lean_timeout_s: 0.0
  max_lean_calls_per_cycle: 0
  cycle_budget_s: 5.0
  budget_exhaustion_policy: "skip"   # skip | fail | extend
  timeout_outcome: "abstain"          # abstain | retry | fail

slices:
  slice_uplift_goal:
    taut_timeout_s: 0.10
    cycle_budget_s: 5.0
    max_candidates_per_cycle: 40
    success_metric: "goal_hit"
    # Verifier behavior on timeout
    timeout_behavior:
      tier1_timeout: "abstain_timeout"     # Return abstention, no policy signal
      budget_exhaustion: "skip_remaining"  # Skip unverified candidates
      log_level: "INFO"

  slice_uplift_sparse:
    taut_timeout_s: 0.12                   # Higher due to 5-atom complexity
    cycle_budget_s: 6.0                    # More time for sparse exploration
    max_candidates_per_cycle: 40
    success_metric: "density"
    timeout_behavior:
      tier1_timeout: "abstain_timeout"
      budget_exhaustion: "skip_remaining"
      log_level: "INFO"

  slice_uplift_tree:
    taut_timeout_s: 0.10
    cycle_budget_s: 4.0                    # Tighter budget for focused derivation
    max_candidates_per_cycle: 30           # Fewer candidates, deeper chains
    success_metric: "chain_length"
    timeout_behavior:
      tier1_timeout: "abstain_timeout"
      budget_exhaustion: "skip_remaining"
      log_level: "INFO"

  slice_uplift_dependency:
    taut_timeout_s: 0.12
    cycle_budget_s: 6.0
    max_candidates_per_cycle: 40
    success_metric: "multi_goal"
    timeout_behavior:
      tier1_timeout: "abstain_timeout"
      budget_exhaustion: "skip_remaining"
      log_level: "INFO"
```

### 14.3 Timeout Behavior Specification

Phase II requires precise, deterministic handling of timeout events.

#### 14.3.1 Timeout Event Types

| Event | Trigger | Outcome | Policy Update |
|-------|---------|---------|---------------|
| **Tier 1 Timeout** | `elapsed > taut_timeout_s` | `abstain_timeout` | No update (neutral) |
| **Cycle Budget Exhausted** | `cycle_elapsed >= cycle_budget_s` | `budget_skip` | No update (not observed) |
| **Candidate Limit Reached** | `verified + skipped >= max_candidates` | `cycle_complete` | Normal termination |

#### 14.3.2 Timeout → Abstention Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TIMEOUT BEHAVIOR (Tier 1 Only)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Verification Start                                                        │
│         │                                                                   │
│         ▼                                                                   │
│   ┌───────────────┐                                                         │
│   │ Start Timer   │                                                         │
│   │ t = 0         │                                                         │
│   └───────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────────────┐                                                 │
│   │ Evaluate truth table  │                                                 │
│   │ for candidate formula │                                                 │
│   └───────────┬───────────┘                                                 │
│               │                                                             │
│       ┌───────┴───────┐                                                     │
│       │               │                                                     │
│       ▼               ▼                                                     │
│   t < timeout     t >= timeout                                              │
│       │               │                                                     │
│       ▼               ▼                                                     │
│   ┌─────────┐   ┌──────────────┐                                            │
│   │ Result  │   │ ABSTAIN      │                                            │
│   │ V/R     │   │ reason:      │                                            │
│   └────┬────┘   │ timeout      │                                            │
│        │        └──────┬───────┘                                            │
│        │               │                                                    │
│        ▼               ▼                                                    │
│   ┌──────────┐   ┌──────────────┐                                           │
│   │ Policy   │   │ No Policy    │                                           │
│   │ Update   │   │ Update       │                                           │
│   │ (+/-)    │   │ (neutral)    │                                           │
│   └──────────┘   └──────────────┘                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 14.3.3 Abstention Guarantee

**Critical Safety Property:**

A timeout event **MUST** produce `outcome: abstain` with `reason: timeout`.
It **MUST NEVER** produce:
- `outcome: verified` (would be a false positive)
- `outcome: refuted` (would be a false negative)

This guarantee is enforced by:
1. Timer check occurs **before** result interpretation
2. On timeout expiry, result buffer is discarded
3. Abstention outcome is set atomically
4. Attestation logs capture timeout duration for audit

#### 14.3.4 Budget Exhaustion Behavior

When `cycle_elapsed >= cycle_budget_s`:

```python
# Pseudocode for budget exhaustion handling
def handle_budget_exhaustion(remaining_candidates: List[Candidate]) -> None:
    for candidate in remaining_candidates:
        log_outcome(candidate, outcome="budget_skip", reason="cycle_budget_exhausted")
        # NOT added to policy update batch
        # NOT counted toward abstention rate

    attestation.add_flag("budget_exhausted", True)
    attestation.add_metric("statements_skipped", len(remaining_candidates))
```

**Budget exhaustion semantics:**
- Remaining candidates receive outcome `budget_skip`
- Distinct from `abstain` (was not attempted)
- Excluded from abstention rate denominator
- Excluded from policy gradient estimation
- Logged for diagnostics but does not affect success metrics

### 14.4 PREREG Alignment Verification

This specification aligns with `PREREG_UPLIFT_U2.yaml`:

| PREREG Requirement | Section Reference | Status |
|--------------------|-------------------|--------|
| Deterministic seeding | §12.5.1 | ✓ Specified |
| Budget parameters per slice | §14.1 | ✓ Specified |
| Timeout → abstention | §14.3 | ✓ Specified |
| Budget exhaustion handling | §14.3.4 | ✓ Specified |
| Policy update exclusions | §12.4.2 | ✓ Specified |
| Telemetry requirements | §12.7 | ✓ Specified |

**Governance cross-check:**
- All budget parameters are deterministic
- No hidden state affecting outcomes
- Same seed + config → same verification sequence
- Attestation captures full budget usage

---

## 15. Lean Timeout vs. Abstention Decision Matrix

> **STATUS: PHASE II — NOT RUN IN PHASE I**
>
> Reference matrix for implementers on how to handle various failure modes.

### 15.1 Decision Matrix

| Condition | Detection Method | Outcome | Attestation | Policy Signal |
|-----------|------------------|---------|-------------|---------------|
| **Tier 1 success** | `result == tautology` | `verified` | `outcome: verified` | +reward |
| **Tier 1 refutation** | `result == contradiction` | `refuted` | `outcome: refuted` | -reward |
| **Tier 1 timeout** | `elapsed >= taut_timeout_s` | `abstain` | `reason: timeout` | neutral |
| **Tier 1 exception** | `try/except` catch | `abstain` | `reason: exception` | neutral |
| **Tier 2 success** | `lean_rc == 0 AND stdout contains SUCCESS` | `verified` | `lean_verified: true` | +reward |
| **Tier 2 proof fail** | `lean_rc == 0 AND stdout contains FAILURE` | `refuted` | `lean_refuted: true` | -reward |
| **Tier 2 timeout (soft)** | `lean_rc == 124` | `abstain` | `reason: lean_timeout` | neutral |
| **Tier 2 timeout (hard)** | `lean_rc == 137` | `abstain` | `reason: lean_killed` | neutral |
| **Tier 2 crash** | `lean_rc != 0 AND lean_rc ∉ {124, 137}` | `abstain` | `reason: lean_crash` | neutral |
| **Tier 2 OOM** | `lean_rc == 137 + OOM signal` | `abstain` | `reason: lean_oom` | neutral |
| **Tier 2 quota** | `lean_calls >= max_lean_calls` | tier1_result | `quota_exceeded: true` | tier1_signal |
| **Cycle budget** | `cycle_elapsed >= cycle_budget_s` | `budget_skip` | `budget_exhausted: true` | none |

### 15.2 Implementation Invariants

1. **No silent failures**: Every verification attempt produces an explicit outcome
2. **No false positives**: Timeouts/crashes never yield `verified`
3. **No false negatives**: Timeouts/crashes never yield `refuted`
4. **Deterministic mapping**: Same failure condition → same outcome
5. **Complete logging**: All outcomes are attestation-captured

### 15.3 Phase II Safety Checklist

Before running any Phase II experiment:

- [ ] Verify `lean_enabled: false` in slice config
- [ ] Confirm `lean_timeout_s: 0.0` for all slices
- [ ] Validate `taut_timeout_s` matches slice specification
- [ ] Check `cycle_budget_s` is set per slice requirements
- [ ] Ensure timeout handler returns `abstain`, not `verified`
- [ ] Confirm budget exhaustion produces `budget_skip`
- [ ] Verify attestation schema includes budget fields
- [ ] Test with deterministic seed produces identical outcomes

---

## 16. Phase IIb Future Lean Integration Notes

> **STATUS: FORWARD-LOOKING — NOT PART OF PHASE II**
>
> These notes are for future reference when Lean may be reintroduced.

### 16.1 Prerequisites for Phase IIb

Before enabling Lean in any uplift experiment:

1. **New preregistration**: File `PREREG_UPLIFT_U2b.yaml` with Lean-specific parameters
2. **Sandbox implementation**: Complete §10 implementation phases
3. **Version pinning**: Lock Lean to specific version (e.g., v4.3.0)
4. **Determinism validation**: Demonstrate reproducible Lean outcomes
5. **Fallback testing**: Verify Tier 2 → Tier 1 fallback works correctly

### 16.2 Lean Budget Parameters (Phase IIb Only)

When Lean is enabled, additional budget parameters apply:

```yaml
# Phase IIb ONLY — NOT ACTIVE IN PHASE II
lean_budget:
  enabled: true                        # Master switch
  timeout_s: 30                        # Per-statement timeout
  memory_mb: 2048                      # Memory limit
  disk_mb: 100                         # Disk quota
  max_calls_per_cycle: 10              # Lean call quota
  fallback_to_tier1: true              # On Lean failure, use truth table
  deterministic_flags:
    LEAN_NUM_THREADS: 1
    LAKE_NO_CACHE: 1
    LAKE_OFFLINE: 1
```

### 16.3 Phase IIb Attestation Schema Extension

Phase IIb attestations must include Lean-specific fields:

```json
{
  "lean_budget": {
    "enabled": true,
    "timeout_s": 30,
    "memory_mb": 2048,
    "max_calls": 10
  },
  "lean_usage": {
    "calls_attempted": 5,
    "calls_succeeded": 3,
    "calls_timeout": 1,
    "calls_crash": 0,
    "calls_oom": 1,
    "quota_remaining": 5
  },
  "lean_fallbacks": {
    "count": 2,
    "tier1_results": ["verified", "refuted"]
  }
}
```

---

*End of Phase II Verifier Budget Envelope specification.*
