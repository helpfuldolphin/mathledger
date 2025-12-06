# Drift Response Playbook

## Overview

This playbook provides step-by-step procedures for responding to determinism drift detected by the Drift Sentinel and Determinism Guard systems.

## Detection Systems

### 1. Drift Sentinel
**Purpose**: Detect nondeterministic function calls in Python code  
**Trigger**: Pre-commit hook, CI pipeline  
**Output**: `artifacts/repro/drift_report.json`, `artifacts/repro/drift_patch.diff`

### 2. Determinism Guard (Replay Guard)
**Purpose**: Verify byte-identical outputs across multiple runs  
**Trigger**: CI pipeline, manual verification  
**Output**: Console output with pass/fail status, drift report on failure

## Response Procedures

### Procedure 1: Drift Sentinel Violation

**Symptoms:**
- Pre-commit hook fails with "Drift Sentinel: X violations detected"
- CI job "Drift Sentinel - Detect nondeterministic operations" fails
- `artifacts/repro/drift_report.json` is generated

**Steps:**

1. **Review Drift Report**
   ```bash
   cat artifacts/repro/drift_report.json
   ```
   
   The report contains:
   - `violations`: List of files, line numbers, and patterns detected
   - `recommendation`: Suggested fixes for each violation

2. **Classify Violation Type**

   **Type A: Core Derivation Logic** (CRITICAL)
   - Files: `backend/axiom_engine/derive.py`, `backend/axiom_engine/policy.py`, `backend/ledger/blocking.py`, `backend/axiom_engine/model.py`, `backend/worker.py`
   - Action: MUST fix immediately - these affect proof validity
   
   **Type B: Monitoring/Logging** (WHITELIST CANDIDATE)
   - Files: `backend/tools/progress.py`, `backend/axiom_engine/derive_worker.py`, `backend/axiom_engine/rules.py`, `backend/frontier/curriculum.py`
   - Action: Verify that timestamps are for logging only, then add to whitelist

3. **Fix Type A Violations**

   Replace nondeterministic calls with deterministic helpers:
   
   ```python
   # BEFORE (nondeterministic)
   import datetime
   timestamp = datetime.datetime.utcnow()
   
   # AFTER (deterministic)
   from backend.repro.determinism import deterministic_timestamp, _GLOBAL_SEED
   timestamp = deterministic_timestamp(_GLOBAL_SEED)
   ```
   
   Common replacements:
   - `datetime.utcnow()` → `deterministic_timestamp(_GLOBAL_SEED)`
   - `datetime.now()` → `deterministic_timestamp(_GLOBAL_SEED)`
   - `time.time()` → `deterministic_unix_timestamp(_GLOBAL_SEED)`
   - `uuid.uuid4()` → `deterministic_uuid(content_string)`
   - `np.random.random()` → `SeededRNG(_GLOBAL_SEED).random()`
   - `random.random()` → `SeededRNG(_GLOBAL_SEED).random()`

4. **Fix Type B Violations (Whitelist)**

   If the violation is in monitoring/logging code that does NOT affect proof validity:
   
   ```bash
   # Edit whitelist
   vim artifacts/repro/drift_whitelist.json
   
   # Add file to whitelist array
   {
     "whitelist": [
       "backend/repro/determinism.py",
       "backend/tools/progress.py",
       "backend/your_file.py"  # ← Add here
     ],
     "rationale": {
       "backend/your_file.py": "Brief explanation of why this is safe"
     }
   }
   ```

5. **Verify Fix**

   ```bash
   # Run drift sentinel locally
   python tools/repro/drift_sentinel.py --all --whitelist artifacts/repro/drift_whitelist.json
   
   # Should output: [PASS] Drift Sentinel: No violations detected
   ```

6. **Verify Determinism**

   ```bash
   # Run determinism guard
   python tools/repro/seed_replay_guard.py --seed 0 --runs 3 --path artifacts/repro/
   
   # Should output: [PASS] Determinism Guard: 3/3 byte-identical runs
   ```

### Procedure 2: Determinism Guard Failure

**Symptoms:**
- CI job "Determinism Guard - Verify byte-identical runs" fails
- Console shows "DRIFT DETECTED" with stderr/stdout differences
- `artifacts/repro/drift_report.json` contains hash mismatches

**Steps:**

1. **Review Drift Report**
   ```bash
   cat artifacts/repro/drift_report.json
   ```
   
   Check which outputs differ:
   - `stdout`: Different proof outputs or metrics
   - `stderr`: Different warnings or error messages
   - `return_code`: Different exit codes

2. **Reproduce Locally**

   ```bash
   # Run multiple derivations with same seed
   for i in 1 2 3; do
     echo "=== Run $i ==="
     PYTHONHASHSEED=0 python3 -B -m backend.axiom_engine.derive \
       --system pl --smoke-pl --seed 0 > run${i}_stdout.txt 2> run${i}_stderr.txt
   done
   
   # Compare outputs
   diff run1_stdout.txt run2_stdout.txt
   diff run1_stderr.txt run2_stderr.txt
   ```

3. **Identify Root Cause**

   Common causes:
   - **Timestamps in output**: Check for datetime/time calls in print statements
   - **Random ordering**: Check for dict/set iteration without sorting
   - **UUID generation**: Check for uuid.uuid4() calls
   - **External state**: Check for file system, network, or database dependencies
   - **Python bytecode caching**: Ensure `-B` flag is used

4. **Apply Fix**

   Based on root cause:
   - Add missing `deterministic_*` helper calls
   - Sort dict/set iterations before output
   - Replace UUID generation with content-based hashing
   - Mock external dependencies in tests

5. **Verify Fix**

   ```bash
   # Run determinism guard again
   python tools/repro/seed_replay_guard.py --seed 0 --runs 3 --path artifacts/repro/
   
   # Should output: [PASS] Determinism Guard: 3/3 byte-identical runs
   ```

### Procedure 3: Emergency Rollback

**When to use:**
- Critical production issue caused by determinism changes
- Unable to fix drift within acceptable timeframe
- Need to restore previous deterministic state

**Steps:**

1. **Identify Last Known Good Commit**
   ```bash
   # Check fleet state archive
   cat artifacts/allblue/fleet_state.json
   
   # Note the state_hash (git commit)
   ```

2. **Create Rollback Branch**
   ```bash
   git checkout -b rollback/determinism-$(date +%s)
   git reset --hard <last_known_good_commit>
   ```

3. **Verify Determinism**
   ```bash
   python tools/repro/seed_replay_guard.py --seed 0 --runs 3 --path artifacts/repro/
   ```

4. **Create Emergency PR**
   ```bash
   git push origin rollback/determinism-$(date +%s)
   # Open PR with "[EMERGENCY] Rollback determinism to <commit>" title
   ```

## Whitelist Policy

### Approved Use Cases

Files may be whitelisted if they meet ALL criteria:
1. **Not in critical path**: File is not part of core proof derivation
2. **Logging/monitoring only**: Timestamps are for observability, not computation
3. **No proof impact**: Changes to timestamps do not affect proof validity
4. **Documented rationale**: Clear explanation in `drift_whitelist.json`

### Whitelist Review Process

1. **Propose Addition**
   - Add file to `whitelist` array in `artifacts/repro/drift_whitelist.json`
   - Add rationale to `rationale` object
   - Document why timestamps do not affect proofs

2. **Peer Review**
   - At least one other engineer must review
   - Verify that file is not in critical path
   - Confirm timestamps are logging-only

3. **Test Coverage**
   - Verify determinism guard still passes
   - Add test case demonstrating proof validity is unaffected

## Escalation

### Level 1: Team Lead
- Multiple drift violations in same PR
- Whitelist additions to critical files
- Determinism guard failures that cannot be reproduced locally

### Level 2: Architecture Review
- Systematic drift across multiple modules
- Need to modify determinism helper design
- Whitelist policy violations

### Level 3: Emergency Response
- Production proof validity compromised
- Unable to restore determinism within 4 hours
- Need for emergency rollback

## Monitoring

### Key Metrics

1. **Drift Detection Rate**: Violations per 100 commits
2. **Time to Resolution**: Hours from detection to fix
3. **Whitelist Growth**: Files added to whitelist per month
4. **Determinism Score**: Percentage of byte-identical runs

### Alerts

- **Critical**: Determinism guard fails in CI
- **Warning**: Drift sentinel detects violations in critical files
- **Info**: Whitelist additions

## References

- **Determinism Helpers**: `backend/repro/determinism.py`
- **Drift Sentinel**: `tools/repro/drift_sentinel.py`
- **Determinism Guard**: `tools/repro/seed_replay_guard.py`
- **Whitelist**: `artifacts/repro/drift_whitelist.json`
- **Fleet State**: `artifacts/allblue/fleet_state.json`

## Appendix: Determinism Principles

### Proof-or-Abstain
If determinism cannot be verified, abstain from claiming proof validity.

### RFC 8785 Canonicalization
All JSON artifacts must use canonical form (sorted keys, no whitespace variations).

### ASCII-Only Discipline
All logs, reports, and artifacts must be ASCII-only for reproducibility.

### Mechanical Honesty
Status must reflect actual test results. No greenfaking.

### Domain Separation
Monitoring/logging code is separate from proof derivation logic.
