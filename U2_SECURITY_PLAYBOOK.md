# U2 Security Playbook

**PHASE II — NOT RUN IN PHASE I**

This playbook provides operational procedures for detecting, investigating, and responding to security events during U2 uplift experiments.

---

## Table of Contents

1. [Threat Hunting Workflow](#threat-hunting-workflow)
2. [Hash Tamper Detection Playbook](#hash-tamper-detection-playbook)
3. [Seed Drift Incident Response](#seed-drift-incident-response)
4. [Manifest Substitution Incident Response](#manifest-substitution-incident-response)
5. [Replay Failure Incident Response](#replay-failure-incident-response)
6. [Seed Drift vs Replay Disagreement](#seed-drift-vs-replay-disagreement)
7. [Operation LAST MILE Ready Checklist](#operation-last-mile-ready-checklist)
8. [Phase IIb Security Invariants](#phase-iib-security-invariants)

---

## Threat Hunting Workflow

### Overview

Proactive threat hunting identifies security violations before they compromise experiment validity. Run threat hunts **during** active experiments (real-time) and **after** completion (forensic).

### Hunt Schedule

| Hunt Type | Frequency | Trigger |
|-----------|-----------|---------|
| Real-time monitoring | Continuous | Automated during U2 run |
| Periodic sweep | Every 10 cycles | Cycle counter threshold |
| Post-run forensics | Once | Run completion |
| Ad-hoc investigation | As needed | Anomaly alert |

### Hunt Procedures

#### Hunt 1: Reward Channel Integrity

**Objective**: Confirm all rewards originate from Lean verification.

```bash
# Step 1: Extract reward sources from H_t series
python -c "
import json
from pathlib import Path

log_dir = Path('logs/uplift/<RUN_ID>')
for ht_file in sorted(log_dir.glob('ht_*.json')):
    ht = json.loads(ht_file.read_text())
    source = ht.get('reward_source', 'UNKNOWN')
    if source != 'lean_verification':
        print(f'ALERT: {ht_file.name} has reward_source={source}')
"

# Step 2: Cross-reference with Lean job logs
grep -r "SUCCESS\|FAILURE" logs/uplift/<RUN_ID>/lean_jobs/ | wc -l
# Must equal total rewards in H_t series

# Step 3: Check for orphan rewards (rewards without Lean jobs)
python scripts/audit_reward_sources.py --run-id <RUN_ID> --strict
```

**Expected Result**: Zero non-Lean reward sources. Job count matches reward count.

**Escalation**: If mismatch found → Immediate run abort → Incident Response (External Metric Injection).

#### Hunt 2: Log Integrity Scan

**Objective**: Detect unauthorized modifications to uplift logs.

```bash
# Step 1: Verify append-only property
python -c "
import os
from pathlib import Path

log_dir = Path('logs/uplift/<RUN_ID>')
for log_file in log_dir.glob('*.log'):
    stat = os.stat(log_file)
    # Check if file was modified after creation (mtime >> ctime)
    if stat.st_mtime - stat.st_ctime > 1:
        print(f'ALERT: {log_file.name} modified after creation')
"

# Step 2: Validate per-entry checksums
python scripts/validate_log_checksums.py --log-path logs/uplift/<RUN_ID>/

# Step 3: Check H_t hash chain continuity
python scripts/validate_ht_chain.py --log-path logs/uplift/<RUN_ID>/
```

**Expected Result**: No post-creation modifications. All checksums valid. Chain unbroken.

**Escalation**: If tampering detected → Preserve evidence → Incident Response (Hash Tamper).

#### Hunt 3: Determinism Verification

**Objective**: Confirm PRNG behavior matches expected deterministic output.

```bash
# Step 1: Extract seed from run manifest
python -c "
import yaml
manifest = yaml.safe_load(open('logs/uplift/<RUN_ID>/run_manifest.yaml'))
print(f'Declared seed: {manifest[\"prng_seed\"]}')
print(f'Master seed: {manifest[\"u2_master_seed\"]}')
"

# Step 2: Replay first N cycles and compare outputs
python scripts/replay_determinism_check.py \
    --run-id <RUN_ID> \
    --cycles 10 \
    --compare-mode strict

# Step 3: Check for runtime reseeding events
grep -i "seed\|random" logs/uplift/<RUN_ID>/runner.log
```

**Expected Result**: Replay produces bit-identical outputs. No reseeding events.

**Escalation**: If divergence detected → Incident Response (Seed Drift).

#### Hunt 4: Environment Contamination

**Objective**: Detect state leakage or external influence.

```bash
# Step 1: Scan for prohibited environment variables
python -c "
import os
prohibited = ['REWARD', 'PROXY', 'OVERRIDE', 'INJECT']
for key, val in os.environ.items():
    if any(p in key.upper() for p in prohibited):
        print(f'ALERT: Prohibited env var: {key}={val}')
"

# Step 2: Check process isolation
ps aux | grep -E "python|lean" | grep -v <RUN_ID>
# Should show no other uplift processes

# Step 3: Verify database isolation
python -c "
from backend.db import get_session
session = get_session()
# Check for statements from other runs in current timeframe
result = session.execute('''
    SELECT COUNT(*) FROM statements
    WHERE created_at > NOW() - INTERVAL '1 hour'
    AND run_id != '<RUN_ID>'
''')
if result.scalar() > 0:
    print('ALERT: Concurrent run detected - potential contamination')
"
```

**Expected Result**: No prohibited vars. No concurrent processes. Database isolated.

---

## Hash Tamper Detection Playbook

### Detection Triggers

| Trigger | Source | Severity |
|---------|--------|----------|
| H_t chain break | `validate_ht_chain.py` | CRITICAL |
| Checksum mismatch | `validate_log_checksums.py` | CRITICAL |
| File mtime anomaly | Hunt 2 scan | HIGH |
| Missing log entries | Sequence gap detection | HIGH |

### Investigation Procedure

#### Step 1: Isolate and Preserve (T+0 to T+5 min)

```bash
# Immediately halt the run if still active
kill -SIGSTOP $(pgrep -f "u2_runner.*<RUN_ID>")

# Create forensic snapshot
mkdir -p forensics/<RUN_ID>_$(date +%Y%m%d_%H%M%S)
cp -r logs/uplift/<RUN_ID>/ forensics/<RUN_ID>_$(date +%Y%m%d_%H%M%S)/

# Compute hashes of all files
find forensics/<RUN_ID>_*/ -type f -exec sha256sum {} \; > forensics/snapshot_hashes.txt

# Lock down the directory
chmod -R 444 forensics/<RUN_ID>_*/
```

#### Step 2: Identify Tamper Scope (T+5 to T+15 min)

```bash
# Find exact point of chain break
python -c "
import json
from pathlib import Path

log_dir = Path('forensics/<SNAPSHOT_DIR>')
prev_hash = None
for i, ht_file in enumerate(sorted(log_dir.glob('ht_*.json'))):
    ht = json.loads(ht_file.read_text())
    if prev_hash and ht.get('prev_hash') != prev_hash:
        print(f'CHAIN BREAK at cycle {i}: expected {prev_hash}, got {ht.get(\"prev_hash\")}')
        print(f'Affected file: {ht_file.name}')
        break
    prev_hash = ht.get('current_hash')
"

# Identify affected cycles
python scripts/identify_tampered_cycles.py --snapshot-dir forensics/<SNAPSHOT_DIR>/
```

#### Step 3: Assess Impact (T+15 to T+30 min)

| Impact Level | Criteria | Action |
|--------------|----------|--------|
| TOTAL | Chain break in first 10 cycles | Invalidate entire run |
| PARTIAL | Chain break after cycle N | Results valid only for cycles 0 to N-1 |
| RECOVERABLE | Single entry corruption, chain intact | Restore from replica if available |

```bash
# Generate impact report
python scripts/generate_tamper_impact_report.py \
    --snapshot-dir forensics/<SNAPSHOT_DIR>/ \
    --output forensics/impact_report_<RUN_ID>.md
```

#### Step 4: Remediation

```bash
# For TOTAL invalidation
echo "RUN_STATUS=INVALIDATED_TAMPER" >> logs/uplift/<RUN_ID>/run_manifest.yaml
echo "INVALIDATION_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/uplift/<RUN_ID>/run_manifest.yaml

# Archive for audit
tar -czvf archives/invalidated_<RUN_ID>.tar.gz logs/uplift/<RUN_ID>/ forensics/<SNAPSHOT_DIR>/

# Document in incident log
cat >> incidents/tamper_incidents.log << EOF
---
run_id: <RUN_ID>
detected_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
type: hash_tamper
impact: TOTAL|PARTIAL
cycles_affected: X-Y
resolution: invalidated|partial_recovery
EOF
```

---

## Seed Drift Incident Response

### Detection Triggers

| Trigger | Detection Method | Severity |
|---------|------------------|----------|
| Replay divergence | Determinism check script | CRITICAL |
| Seed value change | Start vs end snapshot mismatch | CRITICAL |
| Reseeding log entry | Log grep for seed/random | HIGH |
| Non-reproducible output | Manual replay attempt | HIGH |

### Response Procedure

#### Step 1: Confirm Drift (T+0 to T+10 min)

```bash
# Compare seed snapshots
python -c "
import yaml

manifest = yaml.safe_load(open('logs/uplift/<RUN_ID>/run_manifest.yaml'))
start_seed = manifest.get('prng_seed_start')
end_seed = manifest.get('prng_seed_end')

if start_seed != end_seed:
    print(f'CONFIRMED DRIFT: {start_seed} -> {end_seed}')
else:
    print('Seeds match - investigate other causes')
"

# Attempt controlled replay
python scripts/replay_determinism_check.py \
    --run-id <RUN_ID> \
    --cycles 5 \
    --verbose
```

#### Step 2: Identify Drift Source

| Source | Evidence | Root Cause |
|--------|----------|------------|
| Explicit reseed | `random.seed()` in logs | Code bug or malicious call |
| Library reseed | Third-party import | Dependency issue |
| Entropy injection | `/dev/urandom` access | System call leak |
| Uninitialized state | No seed in manifest | Init bug |

```bash
# Search for reseeding calls
grep -rn "\.seed\|random\(" backend/rfl/ --include="*.py"

# Check system entropy access (Linux)
strace -f -e openat python -c "import random" 2>&1 | grep -i random

# Review import chain
python -c "
import sys
import backend.rfl.runner
for mod in sorted(sys.modules.keys()):
    if 'random' in mod.lower():
        print(mod)
"
```

#### Step 3: Determine Validity Window

```bash
# Find last known-good cycle (before divergence)
python scripts/find_drift_point.py --run-id <RUN_ID> --output drift_analysis.json

# Output includes:
# - last_valid_cycle: N
# - drift_detected_cycle: N+1
# - confidence: HIGH|MEDIUM|LOW
```

#### Step 4: Resolution

| Drift Point | Valid Data | Action |
|-------------|------------|--------|
| Cycle 0 | None | Full invalidation |
| Cycle N (N > 0) | Cycles 0 to N-1 | Partial validity; document boundary |
| Unknown | Indeterminate | Conservative invalidation |

```bash
# Document incident
cat >> incidents/seed_drift_incidents.log << EOF
---
run_id: <RUN_ID>
detected_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
type: seed_drift
drift_source: <SOURCE>
last_valid_cycle: <N>
resolution: full_invalidation|partial_validity
root_cause: <DESCRIPTION>
preventive_action: <PLANNED_FIX>
EOF
```

---

## Manifest Substitution Incident Response

### Detection Triggers

| Trigger | Detection Method | Severity |
|---------|------------------|----------|
| Hash mismatch | Pre vs post manifest hash | CRITICAL |
| H_t manifest_hash drift | Chain inspection | CRITICAL |
| File metadata change | mtime/ctime anomaly | HIGH |
| Content diff | Byte comparison | HIGH |

### Response Procedure

#### Step 1: Confirm Substitution (T+0 to T+5 min)

```bash
# Compare hashes
DECLARED_HASH=$(grep "manifest_hash" logs/uplift/<RUN_ID>/run_manifest.yaml | cut -d: -f2 | tr -d ' ')
CURRENT_HASH=$(sha256sum PREREG_UPLIFT_U2.yaml | cut -d' ' -f1)

if [ "$DECLARED_HASH" != "$CURRENT_HASH" ]; then
    echo "CONFIRMED SUBSTITUTION"
    echo "Declared: $DECLARED_HASH"
    echo "Current:  $CURRENT_HASH"
fi

# Check H_t series for drift
python -c "
import json
from pathlib import Path

log_dir = Path('logs/uplift/<RUN_ID>')
manifest_hashes = set()
for ht_file in sorted(log_dir.glob('ht_*.json')):
    ht = json.loads(ht_file.read_text())
    mh = ht.get('manifest_hash')
    manifest_hashes.add(mh)

if len(manifest_hashes) > 1:
    print(f'MANIFEST DRIFT DETECTED: {manifest_hashes}')
"
```

#### Step 2: Recover Original Manifest

```bash
# Check git history
git log --oneline -- PREREG_UPLIFT_U2.yaml

# Restore from known-good commit
git show <COMMIT_HASH>:PREREG_UPLIFT_U2.yaml > forensics/original_manifest.yaml

# Verify against declared hash
RECOVERED_HASH=$(sha256sum forensics/original_manifest.yaml | cut -d' ' -f1)
echo "Recovered manifest hash: $RECOVERED_HASH"
```

#### Step 3: Diff Analysis

```bash
# Generate detailed diff
diff -u forensics/original_manifest.yaml PREREG_UPLIFT_U2.yaml > forensics/manifest_diff.txt

# Categorize changes
python -c "
import yaml

original = yaml.safe_load(open('forensics/original_manifest.yaml'))
current = yaml.safe_load(open('PREREG_UPLIFT_U2.yaml'))

critical_fields = ['success_criteria', 'hypothesis', 'seeds', 'cycles', 'policies']
for field in critical_fields:
    if original.get(field) != current.get(field):
        print(f'CRITICAL CHANGE: {field}')
        print(f'  Original: {original.get(field)}')
        print(f'  Current:  {current.get(field)}')
"
```

#### Step 4: Impact Assessment

| Change Type | Impact | Validity |
|-------------|--------|----------|
| Success criteria modified | Goalpost shift | INVALID |
| Seed values changed | Preregistration violated | INVALID |
| Cycle count changed | Scope manipulation | INVALID |
| Cosmetic only (comments, formatting) | None | VALID with note |

#### Step 5: Resolution

```bash
# Restore original manifest
cp forensics/original_manifest.yaml PREREG_UPLIFT_U2.yaml

# If run must be invalidated
echo "RUN_STATUS=INVALIDATED_MANIFEST_SUB" >> logs/uplift/<RUN_ID>/run_manifest.yaml

# Document incident
cat >> incidents/manifest_incidents.log << EOF
---
run_id: <RUN_ID>
detected_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
type: manifest_substitution
original_hash: <HASH>
substituted_hash: <HASH>
changes_detected: <LIST>
impact: critical|cosmetic
resolution: invalidated|accepted_with_note
EOF
```

---

## Replay Failure Incident Response

### Incident Type: REPLAY_FAILURE

A REPLAY_FAILURE occurs when an attempt to reproduce a U2 run from its recorded state produces outputs that differ from the original execution. This is a **CRITICAL** incident that threatens the scientific validity of uplift claims.

### Detection Methods

| Detection Point | Method | Automated |
|-----------------|--------|-----------|
| Post-run validation | `replay_determinism_check.py --full` | Yes |
| Spot check (sampling) | `replay_determinism_check.py --cycles N --sample` | Yes |
| Cross-validator audit | Independent replay on separate machine | Manual |
| H_t divergence | Hash mismatch during replay recording | Yes |

#### Automated Detection Script

```bash
# Full replay validation (run post-completion)
python scripts/replay_determinism_check.py \
    --run-id <RUN_ID> \
    --full \
    --output replay_report_<RUN_ID>.json

# Expected output for valid run:
# {
#   "status": "PASS",
#   "cycles_replayed": 100,
#   "cycles_matched": 100,
#   "divergence_point": null
# }

# REPLAY_FAILURE output:
# {
#   "status": "FAIL",
#   "cycles_replayed": 100,
#   "cycles_matched": 47,
#   "divergence_point": 48,
#   "divergence_type": "output_mismatch|state_mismatch|hash_mismatch"
# }
```

### Triage Procedure

#### Severity Classification

| Divergence Point | Scope | Severity | Response Time |
|------------------|-------|----------|---------------|
| Cycle 0-5 | Total | CRITICAL | Immediate |
| Cycle 6-N/2 | Majority | HIGH | < 1 hour |
| Cycle > N/2 | Minority | MEDIUM | < 4 hours |
| Final cycle only | Minimal | LOW | < 24 hours |

#### Triage Decision Tree

```
REPLAY_FAILURE detected
        │
        ▼
┌───────────────────┐
│ Divergence at     │
│ cycle 0?          │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │ YES       │ NO
    ▼           ▼
┌─────────┐  ┌─────────────────┐
│ Check   │  │ Seeds match?    │
│ init    │  │ (start snapshot │
│ state   │  │  vs replay)     │
└────┬────┘  └────────┬────────┘
     │                │
     ▼           ┌────┴────┐
┌─────────┐      │ YES     │ NO
│ → SEED  │      ▼         ▼
│   DRIFT │  ┌────────┐ ┌────────────┐
└─────────┘  │ SUBSTRATE│ │ → SEED    │
             │ NONDET  │ │    DRIFT   │
             └─────────┘ └────────────┘
```

### Artifact Collection

Upon REPLAY_FAILURE detection, collect the following artifacts **before any remediation**:

#### Required Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Original H_t series | `logs/uplift/<RUN_ID>/ht_*.json` | Ground truth |
| Replay H_t series | `logs/replay/<RUN_ID>/ht_*.json` | Comparison |
| Run manifest | `logs/uplift/<RUN_ID>/run_manifest.yaml` | Seed/config record |
| Divergence report | `replay_report_<RUN_ID>.json` | Analysis |
| System info snapshot | `logs/uplift/<RUN_ID>/system_info.json` | Environment |
| Replay system info | `logs/replay/<RUN_ID>/system_info.json` | Environment delta |

#### Collection Script

```bash
# Create forensic bundle
INCIDENT_ID="REPLAY_$(date +%Y%m%d_%H%M%S)_<RUN_ID>"
mkdir -p forensics/$INCIDENT_ID

# Collect original artifacts
cp -r logs/uplift/<RUN_ID>/ forensics/$INCIDENT_ID/original/

# Collect replay artifacts
cp -r logs/replay/<RUN_ID>/ forensics/$INCIDENT_ID/replay/

# Generate diff report
python scripts/generate_replay_diff.py \
    --original forensics/$INCIDENT_ID/original/ \
    --replay forensics/$INCIDENT_ID/replay/ \
    --output forensics/$INCIDENT_ID/diff_report.md

# Compute checksums
find forensics/$INCIDENT_ID/ -type f -exec sha256sum {} \; \
    > forensics/$INCIDENT_ID/artifact_checksums.txt

# Lock artifacts
chmod -R 444 forensics/$INCIDENT_ID/
```

### Admissibility Impact

| Replay Status | Run Admissibility | Uplift Claim Status |
|---------------|-------------------|---------------------|
| FULL MATCH | Admissible | Valid |
| PARTIAL MATCH (>80% cycles) | Conditionally admissible | Valid with caveat |
| PARTIAL MATCH (50-80% cycles) | Reduced admissibility | Weakened claim |
| PARTIAL MATCH (<50% cycles) | Inadmissible | Invalid |
| NO MATCH (cycle 0 divergence) | Inadmissible | Invalid |

#### Admissibility Documentation

```bash
# Record admissibility determination
cat >> logs/uplift/<RUN_ID>/run_manifest.yaml << EOF

# Replay Validation (added $(date -u +%Y-%m-%dT%H:%M:%SZ))
replay_validation:
  status: FULL_MATCH|PARTIAL_MATCH|NO_MATCH
  cycles_validated: <N>
  cycles_matched: <M>
  match_percentage: <P>
  admissibility: admissible|conditional|reduced|inadmissible
  uplift_claim_status: valid|weakened|invalid
  incident_id: <INCIDENT_ID>
  validator: <OPERATOR_ID>
EOF
```

---

## Seed Drift vs Replay Disagreement

### Distinguishing Root Causes

When replay produces different outputs, two distinct root causes must be differentiated:

| Characteristic | Seed Drift | Substrate Nondeterminism |
|----------------|------------|--------------------------|
| **Definition** | PRNG seed changed between original and replay | Same seed produces different outputs due to platform/runtime variance |
| **Root Cause** | Configuration error, code bug, or tampering | Floating-point variance, thread scheduling, hash ordering |
| **Detectability** | Seed snapshots differ | Seeds match, outputs differ |
| **Reproducibility** | Fixing seed restores determinism | Same seed still produces variance |
| **Scope** | Usually total (affects all random choices) | Often partial (specific operations) |

### Diagnostic Flowchart

```
Replay produces different output
              │
              ▼
    ┌─────────────────────┐
    │ Compare seed values │
    │ Original vs Replay  │
    └──────────┬──────────┘
               │
      ┌────────┴────────┐
      │                 │
   DIFFERENT          SAME
      │                 │
      ▼                 ▼
┌───────────┐    ┌─────────────────┐
│ SEED DRIFT│    │ Seeds match but │
│ (Config)  │    │ outputs differ  │
└─────┬─────┘    └────────┬────────┘
      │                   │
      ▼                   ▼
┌───────────┐    ┌─────────────────┐
│ Restore   │    │ SUBSTRATE       │
│ correct   │    │ NONDETERMINISM  │
│ seed      │    └────────┬────────┘
└───────────┘             │
                          ▼
                 ┌─────────────────┐
                 │ Identify source │
                 │ (see below)     │
                 └─────────────────┘
```

### Seed Drift Handling

**Incident Type**: `SEED_DRIFT`

#### Causes

1. **Explicit reseed**: Code calls `random.seed()` after initialization
2. **Environment variable override**: `PYTHONHASHSEED` or similar changed
3. **Manifest corruption**: Seed value altered in config file
4. **Import side effect**: Library initializes its own RNG state

#### Detection

```bash
# Check 1: Compare declared seeds
python -c "
import yaml
original = yaml.safe_load(open('logs/uplift/<RUN_ID>/run_manifest.yaml'))
replay = yaml.safe_load(open('logs/replay/<RUN_ID>/run_manifest.yaml'))

if original['prng_seed'] != replay['prng_seed']:
    print(f'SEED_DRIFT: {original[\"prng_seed\"]} → {replay[\"prng_seed\"]}')
"

# Check 2: Verify U2_MASTER_SEED derivation
EXPECTED=$(sha256sum PREREG_UPLIFT_U2.yaml | cut -d' ' -f1)
ACTUAL=$(grep u2_master_seed logs/uplift/<RUN_ID>/run_manifest.yaml | cut -d: -f2 | tr -d ' ')

if [ "$EXPECTED" != "$ACTUAL" ]; then
    echo "SEED_DRIFT: Master seed incorrectly derived"
fi
```

#### Resolution

| Cause | Resolution | Preventive Action |
|-------|------------|-------------------|
| Explicit reseed | Fix code, re-run | Add reseed detection assertion |
| Env var override | Restore env, re-run | Lock env vars at init |
| Manifest corruption | Restore manifest, re-run | Use manifest hash verification |
| Import side effect | Isolate import, re-run | Audit dependency chain |

### Substrate Nondeterminism Handling

**Incident Type**: `SUBSTRATE_NONDET`

#### Causes

1. **Floating-point variance**: Different CPU/FPU produces slightly different results
2. **Hash randomization**: Python dict/set ordering varies (`PYTHONHASHSEED`)
3. **Thread scheduling**: Concurrent operations complete in different order
4. **Memory allocation**: Address-dependent behavior
5. **Library nondeterminism**: Third-party code with hidden randomness

#### Detection

```bash
# Check 1: Confirm seeds are identical
python -c "
import yaml
original = yaml.safe_load(open('logs/uplift/<RUN_ID>/run_manifest.yaml'))
replay = yaml.safe_load(open('logs/replay/<RUN_ID>/run_manifest.yaml'))

assert original['prng_seed'] == replay['prng_seed'], 'Not substrate nondet - seeds differ'
print('Seeds match - investigating substrate nondeterminism')
"

# Check 2: Identify divergence type
python scripts/analyze_divergence_type.py \
    --original logs/uplift/<RUN_ID>/ \
    --replay logs/replay/<RUN_ID>/ \
    --output divergence_analysis.json

# Output categories:
# - floating_point: Numeric differences within epsilon
# - ordering: Same elements, different order
# - timing: Time-dependent divergence
# - unknown: Requires manual investigation
```

#### Resolution

| Cause | Resolution | Preventive Action |
|-------|------------|-------------------|
| Floating-point | Use tolerant comparison | Normalize FP to fixed precision |
| Hash randomization | Set `PYTHONHASHSEED=0` | Enforce in runner init |
| Thread scheduling | Force sequential execution | Remove concurrency from critical path |
| Memory allocation | Pin allocator behavior | Use deterministic allocator |
| Library nondet | Patch or replace library | Audit and approve dependencies |

### Combined Decision Matrix

| Seeds Match? | Outputs Match? | Diagnosis | Severity |
|--------------|----------------|-----------|----------|
| Yes | Yes | Valid replay | N/A |
| Yes | No | SUBSTRATE_NONDET | HIGH |
| No | No | SEED_DRIFT | CRITICAL |
| No | Yes | Coincidental (investigate) | MEDIUM |

### Incident Documentation Template

```yaml
# incidents/replay_disagreement_<INCIDENT_ID>.yaml
incident_id: <INCIDENT_ID>
run_id: <RUN_ID>
detected_at: <ISO_TIMESTAMP>
type: SEED_DRIFT | SUBSTRATE_NONDET

diagnosis:
  seeds_match: true | false
  original_seed: <VALUE>
  replay_seed: <VALUE>
  divergence_cycle: <N>
  divergence_type: <TYPE>

root_cause:
  category: explicit_reseed | env_override | manifest_corruption |
            floating_point | hash_randomization | thread_scheduling |
            library_nondet | unknown
  description: <DETAILED_DESCRIPTION>
  evidence: <FILE_REFERENCES>

impact:
  cycles_affected: <RANGE>
  admissibility: admissible | conditional | inadmissible
  uplift_claim: valid | weakened | invalid

resolution:
  action_taken: <DESCRIPTION>
  run_restarted: true | false
  new_run_id: <NEW_RUN_ID> | null

preventive_measures:
  - <MEASURE_1>
  - <MEASURE_2>
```

---

## Operation LAST MILE Ready Checklist

### Pre-Flight Verification

Before any LAST MILE operation, **all** items must be verified. This checklist ensures the security envelope is fully engaged.

### Checklist Items

#### Section A: Replay Enforcement

| # | Check | Required State | Verification Command | Pass Criteria |
|---|-------|----------------|---------------------|---------------|
| A1 | Replay capability enabled | `REPLAY_ENABLED=true` | `echo $REPLAY_ENABLED` | Returns `true` |
| A2 | Replay script accessible | Script exists and executable | `test -x scripts/replay_determinism_check.py && echo OK` | Returns `OK` |
| A3 | Replay storage configured | Directory writable | `test -w logs/replay/ && echo OK` | Returns `OK` |
| A4 | Replay validation scheduled | Post-run hook registered | `grep -q replay hooks/post_run.sh` | Match found |
| A5 | Replay comparison strict mode | Bit-exact comparison | `grep COMPARE_MODE config/replay.yaml` | Returns `strict` |

#### Section B: Deterministic PRNG Guard

| # | Check | Required State | Verification Command | Pass Criteria |
|---|-------|----------------|---------------------|---------------|
| B1 | PYTHONHASHSEED fixed | `PYTHONHASHSEED=0` | `echo $PYTHONHASHSEED` | Returns `0` |
| B2 | Master seed derived | SHA256(manifest) | `python scripts/verify_master_seed.py` | Returns `VALID` |
| B3 | FrozenRandom wrapper active | No raw random calls | `grep -r "import random$" backend/rfl/` | No matches |
| B4 | Reseed detection enabled | Runtime assertion | `grep -q "assert.*seed" backend/rfl/determinism.py` | Match found |
| B5 | Seed snapshot logging | Start/end recorded | `grep -q prng_seed_start run_manifest.yaml` | Match found |

#### Section C: Telemetry Quarantine

| # | Check | Required State | Verification Command | Pass Criteria |
|---|-------|----------------|---------------------|---------------|
| C1 | External telemetry disabled | No outbound metrics | `netstat -an \| grep -E ":443\|:8125"` | No matches |
| C2 | Reward channel isolated | Lean-only rewards | `python scripts/audit_reward_sources.py --dry-run` | Returns `CLEAN` |
| C3 | Proxy metrics blocked | No `*_PROXY_*` vars | `env \| grep -i proxy` | No matches (or only HTTP_PROXY for deps) |
| C4 | Telemetry log quarantine | Separate log path | `test -d logs/quarantine/ && echo OK` | Returns `OK` |
| C5 | Metrics export disabled | No Prometheus/StatsD | `grep -r "prometheus\|statsd" backend/rfl/` | No matches |

#### Section D: Hermetic Execution

| # | Check | Required State | Verification Command | Pass Criteria |
|---|-------|----------------|---------------------|---------------|
| D1 | Network isolation | Localhost only | Firewall audit | Confirmed |
| D2 | Process isolation | Single runner | `pgrep -c u2_runner` | Returns `1` |
| D3 | Log append-only | Write-once mode | `lsattr logs/uplift/<RUN_ID>/` (Linux) | Shows `a` flag |
| D4 | Env var scan passed | No prohibited vars | `python scripts/scan_prohibited_env.py` | Returns `CLEAN` |
| D5 | Database isolation | Partition active | `python scripts/verify_db_partition.py` | Returns `ISOLATED` |

### Checklist Execution

#### Automated Verification

```bash
# Run full LAST MILE readiness check
python scripts/lastmile_readiness_check.py --verbose

# Expected output for ready state:
# ╔════════════════════════════════════════════╗
# ║     OPERATION LAST MILE READY CHECK        ║
# ╠════════════════════════════════════════════╣
# ║ Section A: Replay Enforcement      5/5 ✓   ║
# ║ Section B: PRNG Guard              5/5 ✓   ║
# ║ Section C: Telemetry Quarantine    5/5 ✓   ║
# ║ Section D: Hermetic Execution      5/5 ✓   ║
# ╠════════════════════════════════════════════╣
# ║ TOTAL: 20/20 PASSED                        ║
# ║ STATUS: READY FOR LAST MILE                ║
# ╚════════════════════════════════════════════╝
```

#### Manual Verification Record

```yaml
# logs/uplift/<RUN_ID>/lastmile_checklist.yaml
checklist_version: "1.0"
verified_at: <ISO_TIMESTAMP>
verified_by: <OPERATOR_ID>

section_a_replay:
  a1_replay_enabled: PASS | FAIL
  a2_script_accessible: PASS | FAIL
  a3_storage_configured: PASS | FAIL
  a4_validation_scheduled: PASS | FAIL
  a5_strict_mode: PASS | FAIL
  section_status: PASS | FAIL

section_b_prng:
  b1_hashseed_fixed: PASS | FAIL
  b2_master_seed_valid: PASS | FAIL
  b3_frozen_random: PASS | FAIL
  b4_reseed_detection: PASS | FAIL
  b5_seed_logging: PASS | FAIL
  section_status: PASS | FAIL

section_c_telemetry:
  c1_external_disabled: PASS | FAIL
  c2_reward_isolated: PASS | FAIL
  c3_proxy_blocked: PASS | FAIL
  c4_log_quarantine: PASS | FAIL
  c5_metrics_disabled: PASS | FAIL
  section_status: PASS | FAIL

section_d_hermetic:
  d1_network_isolation: PASS | FAIL
  d2_process_isolation: PASS | FAIL
  d3_append_only: PASS | FAIL
  d4_env_scan: PASS | FAIL
  d5_db_isolation: PASS | FAIL
  section_status: PASS | FAIL

overall_status: READY | NOT_READY
blocking_items: []  # List any FAIL items
```

### Go/No-Go Decision

| Sections Passing | Decision | Action |
|------------------|----------|--------|
| 4/4 (all items) | **GO** | Proceed with LAST MILE |
| 3/4 sections | **CONDITIONAL** | Fix failing section, re-verify |
| <3/4 sections | **NO-GO** | Do not proceed; full remediation required |

### Failure Remediation

| Section | Common Failures | Remediation |
|---------|-----------------|-------------|
| A (Replay) | Script missing | Deploy `replay_determinism_check.py` |
| B (PRNG) | PYTHONHASHSEED unset | Add to runner init: `export PYTHONHASHSEED=0` |
| C (Telemetry) | Proxy vars present | Unset or document exception |
| D (Hermetic) | Multiple runners | Kill orphan processes |

---

## Phase IIb Security Invariants

### Forward-Looking Requirements

Phase IIb introduces additional complexity (multi-environment uplift, cross-slice policies, extended H_t series). The following invariants **must hold** for Phase IIb validity.

### Invariant Set

#### INV-2B-01: Cross-Environment Isolation

**Statement**: No state may flow between asymmetric uplift environments except through the defined feedback interface.

```
∀ env_i, env_j ∈ Environments:
    env_i ≠ env_j →
    shared_state(env_i, env_j) = ∅ ∨ shared_state(env_i, env_j) ⊆ FeedbackInterface
```

**Verification**:
- Process-level isolation audit
- Memory space separation check
- Database partition verification

**Violation Response**: Immediate halt; full invalidation of both environments.

#### INV-2B-02: Policy Update Atomicity

**Statement**: Policy updates must be atomic with respect to H_t recording. No partial updates allowed.

```
∀ policy_update:
    H_t[n].policy_version = V →
    (H_t[n+1].policy_version = V ∨ H_t[n+1].policy_version = V+1)
    ∧ ¬∃ intermediate_version
```

**Verification**:
- H_t policy_version monotonicity check
- Transaction log inspection

**Violation Response**: Rollback to last consistent state; re-execute from checkpoint.

#### INV-2B-03: Slice Boundary Integrity

**Statement**: Slice transitions must be explicitly logged and must not occur mid-cycle.

```
∀ slice_transition(S_old → S_new):
    ∃ H_t[n]: H_t[n].event = "slice_transition"
    ∧ H_t[n].slice_from = S_old
    ∧ H_t[n].slice_to = S_new
    ∧ cycle_complete(n-1)
```

**Verification**:
- Slice boundary event presence in H_t
- Cycle completion verification before transition

**Violation Response**: Log warning; manual review required before proceeding.

#### INV-2B-04: Aggregate Feedback Consistency

**Statement**: When aggregating feedback across multiple environments, aggregation function must be deterministic and commutative.

```
∀ feedback_set F:
    aggregate(F) = aggregate(permute(F))
    ∧ replay(aggregate(F)) = aggregate(F)
```

**Verification**:
- Aggregation replay test
- Permutation invariance test

**Violation Response**: Aggregation invalidated; fall back to per-environment analysis.

#### INV-2B-05: Extended H_t Chain Integrity

**Statement**: H_t chains spanning multiple sessions must maintain cryptographic continuity across session boundaries.

```
∀ session_boundary(S_n, S_n+1):
    H_t[last(S_n)].current_hash = H_t[first(S_n+1)].prev_hash
    ∧ H_t[first(S_n+1)].session_id = S_n+1
    ∧ H_t[first(S_n+1)].continued_from = S_n
```

**Verification**:
- Cross-session hash chain validation
- Session continuation field audit

**Violation Response**: Session gap detected; results valid only within contiguous sessions.

### Phase IIb Security Checklist Additions

| # | Check Item | Required State | Verification |
|---|------------|----------------|--------------|
| 17 | Environment isolation | No cross-env state leakage | `test_env_isolation_2b.py` |
| 18 | Policy update atomicity | No partial updates in H_t | `validate_policy_versions.py` |
| 19 | Slice boundary logging | All transitions recorded | `audit_slice_transitions.py` |
| 20 | Aggregation determinism | Commutative aggregation | `test_aggregation_determinism.py` |
| 21 | Cross-session continuity | Hash chain spans sessions | `validate_cross_session_chain.py` |

### Threat Model Extensions for Phase IIb

| Threat | New in IIb | Mitigation |
|--------|------------|------------|
| Cross-environment contamination | Yes | Process isolation + partition keys |
| Policy version skipping | Yes | Monotonicity enforcement |
| Slice boundary manipulation | Yes | Mandatory transition events |
| Aggregation ordering attack | Yes | Commutative aggregation functions |
| Session discontinuity hiding | Yes | Mandatory continuation fields |

### Migration Path: Phase II → Phase IIb

1. **Validate Phase II completion**: All 16 checklist items PASSED
2. **Extend H_t schema**: Add `session_id`, `continued_from`, `slice_id` fields
3. **Deploy environment isolation**: Container/process separation
4. **Implement aggregation layer**: Deterministic, commutative feedback merge
5. **Update validation scripts**: Add items 17-21
6. **Run Phase IIb dry run**: Non-production validation cycle
7. **Formal Phase IIb commencement**: With updated preregistration

---

*Document Status: PHASE II — NOT RUN IN PHASE I*
*Playbook Version: 1.0*
*Last Updated: 2025-12-06*
