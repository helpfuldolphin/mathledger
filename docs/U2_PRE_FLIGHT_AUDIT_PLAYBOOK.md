# U2 Pre-Flight Audit Playbook

> **PHASE II — NOT RUN IN PHASE I**
>
> This playbook operationalizes the pre-audit gates (PRE-1 through PRE-6) and
> defines what must be true before a U2 experiment run is eligible for use as
> evidence. It does NOT cover the full audit pipeline—only pre-flight checks.

## Document References

| Document | Purpose |
|----------|---------|
| [PROOF_DAG_INVARIANTS.md](./PROOF_DAG_INVARIANTS.md) | Phase II invariant definitions (INV-P2-*) |
| [PROOF_DAG_INVARIANTS.md#u2-full-audit-lifecycle-specification](./PROOF_DAG_INVARIANTS.md#u2-full-audit-lifecycle-specification) | Full audit lifecycle, failure matrix |
| [PREREG_UPLIFT_U2.yaml](../config/PREREG_UPLIFT_U2.yaml) | Preregistration file |

---

## Table of Contents

1. [Pre-Flight Checklist](#pre-flight-checklist)
2. [Audit-Eligibility Criteria](#audit-eligibility-criteria)
3. [Operator Playbook](#operator-playbook)

---

## Pre-Flight Checklist

This checklist must be completed before an experiment is considered for full audit.
Each item maps to one or more pre-audit gates (PRE-1 through PRE-6).

### Registration & Identity (PRE-1)

| # | Check | Gate | Pass Criteria | Fail Action |
|---|-------|------|---------------|-------------|
| 1 | Experiment ID exists in `u2_experiments` | PRE-1 | Row exists with non-null `experiment_id` | STOP: Create experiment record |
| 2 | Theory ID is valid FK | PRE-1 | `theory_id` references existing `theories.id` | STOP: Correct theory reference |
| 3 | Slice ID is non-empty | PRE-1 | `slice_id` is non-null, non-empty string | STOP: Assign slice ID |
| 4 | Status is recognized | PRE-1 | `status` ∈ {pending, running, completed, validated, rolled_back, failed} | STOP: Correct status value |
| 5 | Start time is set (if running/completed) | PRE-1 | `start_time` non-null when `status` ≠ 'pending' | STOP: Set start time |

### Preregistration Integrity (PRE-2)

| # | Check | Gate | Pass Criteria | Fail Action |
|---|-------|------|---------------|-------------|
| 6 | Experiment in PREREG_UPLIFT_U2.yaml | PRE-2 | Entry with matching `experiment_id` exists | FATAL: Not preregistered |
| 7 | Preregistration hash matches | PRE-2 | `sha256(canonicalize(prereg_entry)) == exp.preregistration_hash` | FATAL: Hash tampered |
| 8 | Prereg file itself is unmodified | PRE-2 | Git shows no uncommitted changes to PREREG_UPLIFT_U2.yaml | WARN: Commit prereg first |

### Baseline Snapshot (PRE-3)

| # | Check | Gate | Pass Criteria | Fail Action |
|---|-------|------|---------------|-------------|
| 9 | Snapshot record exists | PRE-3 | Row in `u2_dag_snapshots` for this `experiment_id` | STOP: Create snapshot |
| 10 | Merkle root is non-null | PRE-3 | `root_merkle_hash` is 64-char hex string | STOP: Recompute snapshot |
| 11 | Statement count recorded | PRE-3 | `statement_count` > 0 | STOP: Verify baseline exists |
| 12 | Edge count recorded | PRE-3 | `edge_count` >= 0 (can be 0 for axiom-only baseline) | STOP: Verify baseline |
| 13 | Snapshot timestamp is before experiment start | PRE-3 | `snapshot_timestamp` < `exp.start_time` | STOP: Re-snapshot before start |

### Log Directory Integrity (PRE-4)

| # | Check | Gate | Pass Criteria | Fail Action |
|---|-------|------|---------------|-------------|
| 14 | Log directory exists | PRE-4 | `logs/u2/<exp_id>/` directory present | STOP: Create log directory |
| 15 | manifest.json exists | PRE-4 | File present and non-empty | STOP: Generate manifest |
| 16 | manifest.json is valid JSON | PRE-4 | `json.load()` succeeds | STOP: Fix JSON syntax |
| 17 | At least one cycle log exists | PRE-4 | `cycle_*.jsonl` glob matches >= 1 file | STOP: No cycles recorded |
| 18 | Cycle logs are parseable JSONL | PRE-4 | Each line is valid JSON | STOP: Fix corrupted log |
| 19 | verifications.jsonl exists | PRE-4 | File present | WARN: Verifications missing |
| 20 | verifications.jsonl is parseable | PRE-4 | Each line is valid JSON | STOP: Fix corrupted log |

### Database Connectivity (PRE-5)

| # | Check | Gate | Pass Criteria | Fail Action |
|---|-------|------|---------------|-------------|
| 21 | Database connection succeeds | PRE-5 | Can execute `SELECT 1` | STOP: Fix connection |
| 22 | All U2 tables exist | PRE-5 | Tables: `u2_experiments`, `u2_statements`, `u2_proof_parents`, `u2_goal_attributions`, `u2_dag_snapshots` | STOP: Run migrations |
| 23 | Baseline tables exist | PRE-5 | Tables: `statements`, `proof_parents`, `theories` | STOP: Run migrations |

### Experiment State Eligibility (PRE-6)

| # | Check | Gate | Pass Criteria | Fail Action |
|---|-------|------|---------------|-------------|
| 24 | Status is auditable | PRE-6 | `status` ∈ {running, completed, validated, rolled_back, failed} | STOP: Experiment not started |
| 25 | If running, has completed cycles | PRE-6 | `COUNT(DISTINCT cycle_number) > 0` in `u2_statements` | STOP: No cycles to audit |

---

## Audit-Eligibility Criteria

### Definition: Audit Eligible

An experiment is **Audit Eligible** if and only if:

1. All 25 pre-flight checks pass (or have WARN status with documented waiver)
2. No FATAL conditions exist
3. No STOP conditions remain unresolved

### Definition: Not Audit Eligible

An experiment is **Not Audit Eligible** if any of the following are true:

1. Any FATAL condition exists (see below)
2. Any STOP condition remains unresolved after operator intervention
3. Experiment status is `pending`

---

### Admissibility Classification

Failures are classified into three categories:

#### FATAL: Forever Non-Admissible

These failures render an experiment **permanently inadmissible** as evidence.
No remediation is possible—the experiment must be discarded and re-run.

| Failure | Gate | Reason | Why Unfixable |
|---------|------|--------|---------------|
| **Not in preregistration** | PRE-2 | Experiment ID not found in PREREG_UPLIFT_U2.yaml | Cannot retroactively preregister; violates scientific protocol |
| **Preregistration hash mismatch** | PRE-2 | Stored hash ≠ computed hash | Evidence of tampering or post-hoc modification |
| **Baseline snapshot taken after experiment start** | PRE-3 | `snapshot_timestamp` > `start_time` | Baseline contaminated; cannot prove isolation |
| **Cross-contamination detected** (if discovered pre-flight) | PRE-6 | Shared derived statements with concurrent experiment | Both experiments invalidated |
| **Log files deleted or unrecoverable** | PRE-4 | Cycle logs missing and no backup exists | Cannot reconstruct derivation history |

#### STOP: Fixable Before Audit

These failures block the audit but can be remediated. Once fixed, the experiment
may proceed to full audit.

| Failure | Gate | Remediation | Time Limit |
|---------|------|-------------|------------|
| Experiment record missing | PRE-1 | Create `u2_experiments` row | Before audit |
| Invalid theory/slice reference | PRE-1 | Correct FK references | Before audit |
| Snapshot missing | PRE-3 | Create snapshot (if exp not started) | Before exp start |
| Snapshot incomplete | PRE-3 | Recompute merkle/counts | Before audit |
| Log directory missing | PRE-4 | Create directory, regenerate if possible | Before audit |
| manifest.json missing/invalid | PRE-4 | Generate from experiment config | Before audit |
| Cycle log corruption | PRE-4 | Repair from backup or re-derive | Before audit |
| Database tables missing | PRE-5 | Run migrations | Before audit |
| Experiment still pending | PRE-6 | Start experiment | N/A |

#### WARN: Advisory, Audit May Proceed

These issues are logged but do not block the audit. They may affect the strength
of evidence claims.

| Warning | Gate | Impact | Documentation Required |
|---------|------|--------|------------------------|
| Prereg file has uncommitted changes | PRE-2 | Minor: Git status unclear | Commit before publishing results |
| verifications.jsonl missing | PRE-4 | Verification receipts unavailable | Note in audit report |
| Partial audit (running experiment) | PRE-6 | Only completed cycles audited | Document cycle count in report |

---

### Eligibility Decision Tree

```
START
  │
  ├─ Is experiment_id in PREREG_UPLIFT_U2.yaml?
  │     NO ──────────────────────────────────► FATAL: Not preregistered
  │     YES
  │       │
  │       ├─ Does preregistration hash match?
  │       │     NO ──────────────────────────► FATAL: Hash tampered
  │       │     YES
  │       │       │
  │       │       ├─ Does baseline snapshot exist?
  │       │       │     NO ──────────────────► STOP: Create snapshot
  │       │       │     YES
  │       │       │       │
  │       │       │       ├─ Is snapshot_timestamp < start_time?
  │       │       │       │     NO ──────────► FATAL: Baseline contaminated
  │       │       │       │     YES
  │       │       │       │       │
  │       │       │       │       ├─ Do log files exist and parse?
  │       │       │       │       │     NO (unrecoverable) ─► FATAL: Logs lost
  │       │       │       │       │     NO (recoverable) ──► STOP: Fix logs
  │       │       │       │       │     YES
  │       │       │       │       │       │
  │       │       │       │       │       ├─ Is status auditable?
  │       │       │       │       │       │     NO ──────► STOP: Start experiment
  │       │       │       │       │       │     YES
  │       │       │       │       │       │       │
  │       │       │       │       │       │       └─► AUDIT ELIGIBLE
  │       │       │       │       │       │
```

---

### Eligibility Status Codes

| Code | Meaning | Can Proceed to Full Audit? |
|------|---------|----------------------------|
| `ELIGIBLE` | All checks pass | Yes |
| `ELIGIBLE_PARTIAL` | Running experiment, some cycles complete | Yes (partial audit) |
| `ELIGIBLE_WARNED` | Passes with WARN conditions | Yes (document warnings) |
| `BLOCKED_FIXABLE` | STOP condition exists | No (fix first) |
| `INADMISSIBLE` | FATAL condition exists | Never |

---

## Operator Playbook

This section provides step-by-step remediation instructions for each failure type.

---

### PRE-1 Failures: Registration Issues

#### If experiment record is missing:

```bash
# 1. Verify experiment should exist
grep "experiment_id: <exp_id>" config/PREREG_UPLIFT_U2.yaml

# 2. Create experiment record
psql $DATABASE_URL -c "
INSERT INTO u2_experiments (
    experiment_id, theory_id, slice_id, status, preregistration_hash, created_at
) VALUES (
    '<exp_id>',
    1,  -- theory_id from prereg
    '<slice_id>',
    'pending',
    '<computed_hash>',
    NOW()
);
"

# 3. Re-run pre-flight check
python tools/u2_preflight.py --exp-id <exp_id>
```

#### If theory_id or slice_id is invalid:

```bash
# 1. Check valid theories
psql $DATABASE_URL -c "SELECT id, name FROM theories;"

# 2. Update experiment
psql $DATABASE_URL -c "
UPDATE u2_experiments
SET theory_id = <correct_id>, slice_id = '<correct_slice>'
WHERE experiment_id = '<exp_id>';
"
```

---

### PRE-2 Failures: Preregistration Issues

#### If experiment not in prereg (FATAL):

```
┌─────────────────────────────────────────────────────────────────┐
│  FATAL: EXPERIMENT NOT PREREGISTERED                            │
│                                                                 │
│  This experiment cannot be used as evidence.                    │
│                                                                 │
│  Required Actions:                                              │
│  1. Document why preregistration was missed                     │
│  2. Add experiment to PREREG_UPLIFT_U2.yaml                     │
│  3. Commit and push prereg changes                              │
│  4. Create NEW experiment with new ID                           │
│  5. Re-run experiment from scratch                              │
│                                                                 │
│  The original experiment data may be retained for debugging     │
│  but MUST NOT be cited as evidence.                             │
└─────────────────────────────────────────────────────────────────┘
```

#### If preregistration hash mismatch (FATAL):

```
┌─────────────────────────────────────────────────────────────────┐
│  FATAL: PREREGISTRATION HASH MISMATCH                           │
│                                                                 │
│  Evidence of tampering or post-hoc modification detected.       │
│                                                                 │
│  Investigation Steps:                                           │
│  1. Check git history: git log -p config/PREREG_UPLIFT_U2.yaml  │
│  2. Compare stored vs computed hash                             │
│  3. Identify when/why prereg was modified                       │
│                                                                 │
│  If modification was accidental:                                │
│  - Document incident                                            │
│  - Create new experiment with correct prereg                    │
│  - Re-run from scratch                                          │
│                                                                 │
│  If modification was intentional:                               │
│  - This is a protocol violation                                 │
│  - Escalate to project lead                                     │
└─────────────────────────────────────────────────────────────────┘
```

#### If prereg file has uncommitted changes (WARN):

```bash
# 1. Check changes
git diff config/PREREG_UPLIFT_U2.yaml

# 2. If changes are valid, commit them
git add config/PREREG_UPLIFT_U2.yaml
git commit -m "Update preregistration for <exp_id>"

# 3. If changes are unintended, restore
git checkout config/PREREG_UPLIFT_U2.yaml

# 4. Re-run pre-flight
python tools/u2_preflight.py --exp-id <exp_id>
```

---

### PRE-3 Failures: Snapshot Issues

#### If snapshot record missing:

```bash
# 1. Verify experiment has NOT started
psql $DATABASE_URL -c "
SELECT status, start_time FROM u2_experiments
WHERE experiment_id = '<exp_id>';
"

# 2. If status = 'pending', create snapshot
python tools/u2_create_snapshot.py --exp-id <exp_id>

# 3. If status != 'pending', this may be FATAL
#    Check if snapshot can be reconstructed from git/backup
```

#### If merkle root is null or invalid:

```bash
# 1. Recompute snapshot
python tools/u2_recompute_snapshot.py --exp-id <exp_id>

# 2. Verify new hash
psql $DATABASE_URL -c "
SELECT root_merkle_hash, statement_count, edge_count
FROM u2_dag_snapshots
WHERE experiment_id = '<exp_id>';
"
```

#### If snapshot timestamp after experiment start (FATAL):

```
┌─────────────────────────────────────────────────────────────────┐
│  FATAL: BASELINE SNAPSHOT CONTAMINATED                          │
│                                                                 │
│  The baseline snapshot was taken AFTER the experiment started.  │
│  This means the baseline may include experiment-derived data.   │
│                                                                 │
│  Contamination Details:                                         │
│  - Snapshot timestamp: <snapshot_ts>                            │
│  - Experiment start:   <start_ts>                               │
│  - Delta:              <delta> seconds                          │
│                                                                 │
│  This experiment is permanently inadmissible.                   │
│                                                                 │
│  Required Actions:                                              │
│  1. Mark experiment as 'failed' in u2_experiments               │
│  2. Document contamination in incident log                      │
│  3. Create new experiment with proper snapshot timing           │
│  4. Ensure snapshot is taken BEFORE experiment.start()          │
└─────────────────────────────────────────────────────────────────┘
```

---

### PRE-4 Failures: Log Directory Issues

#### If log directory missing:

```bash
# 1. Create directory
mkdir -p logs/u2/<exp_id>

# 2. Check if logs exist elsewhere (backup, alternative path)
find . -name "*<exp_id>*" -type f

# 3. If logs are lost and experiment has run, this may be FATAL
```

#### If manifest.json missing:

```bash
# 1. Generate from experiment config
python tools/u2_generate_manifest.py --exp-id <exp_id> \
    --output logs/u2/<exp_id>/manifest.json

# 2. Verify manifest
cat logs/u2/<exp_id>/manifest.json | jq .
```

#### If manifest.json is invalid JSON:

```bash
# 1. Validate and identify error
python -c "import json; json.load(open('logs/u2/<exp_id>/manifest.json'))"

# 2. Fix syntax error manually or regenerate
python tools/u2_generate_manifest.py --exp-id <exp_id> --force

# 3. Backup corrupted file first
cp logs/u2/<exp_id>/manifest.json logs/u2/<exp_id>/manifest.json.corrupted
```

#### If cycle logs missing or corrupted:

```bash
# 1. Check which cycles are missing
ls logs/u2/<exp_id>/cycle_*.jsonl | sort -V

# 2. Check for backup
ls backups/logs/u2/<exp_id>/

# 3. If no backup and experiment is running, cycles may still be generated
#    Wait for experiment to complete

# 4. If experiment is complete and logs are unrecoverable: FATAL
```

#### If verifications.jsonl missing (WARN):

```bash
# 1. Check if verifications can be reconstructed from DB
python tools/u2_export_verifications.py --exp-id <exp_id> \
    --output logs/u2/<exp_id>/verifications.jsonl

# 2. If not reconstructable, document in audit report
echo "WARN: Verification receipts unavailable for <exp_id>" >> audit_notes.txt

# 3. Audit may proceed with this warning
```

---

### PRE-5 Failures: Database Issues

#### If database connection fails:

```bash
# 1. Check connection string
echo $DATABASE_URL

# 2. Test connection
psql $DATABASE_URL -c "SELECT 1;"

# 3. Common fixes:
#    - Start PostgreSQL: docker compose up -d postgres
#    - Check credentials
#    - Check network/firewall
```

#### If U2 tables missing:

```bash
# 1. Check which tables exist
psql $DATABASE_URL -c "
SELECT table_name FROM information_schema.tables
WHERE table_name LIKE 'u2_%';
"

# 2. Run U2 migrations
python run_migration.py migrations/u2_*.sql

# 3. Verify tables created
psql $DATABASE_URL -c "\dt u2_*"
```

#### If baseline tables missing:

```bash
# 1. Run base migrations
python run_all_migrations.py

# 2. Verify core tables
psql $DATABASE_URL -c "\dt statements proof_parents theories"
```

---

### PRE-6 Failures: State Eligibility Issues

#### If experiment status is 'pending':

```bash
# 1. Experiment has not started - this is expected
#    Start the experiment first

python tools/u2_start_experiment.py --exp-id <exp_id>

# 2. Wait for at least one cycle to complete

# 3. Re-run pre-flight
python tools/u2_preflight.py --exp-id <exp_id>
```

#### If running but no completed cycles:

```bash
# 1. Check cycle status
psql $DATABASE_URL -c "
SELECT DISTINCT cycle_number
FROM u2_statements
WHERE experiment_id = '<exp_id>'
ORDER BY cycle_number;
"

# 2. If cycles in progress, wait for completion
#    Monitor with:
tail -f logs/u2/<exp_id>/cycle_*.jsonl

# 3. If stuck, check worker status
ps aux | grep u2_worker
```

---

### Quick Reference: Failure Response Matrix

| Failure | Type | Immediate Action | Follow-up |
|---------|------|------------------|-----------|
| Exp not in prereg | FATAL | Stop, document | New experiment required |
| Hash mismatch | FATAL | Stop, investigate | Escalate if tampering |
| Snapshot after start | FATAL | Stop, mark failed | New experiment required |
| Logs unrecoverable | FATAL | Stop, document | New experiment required |
| Exp record missing | STOP | Create record | Re-run pre-flight |
| Invalid FK refs | STOP | Update refs | Re-run pre-flight |
| Snapshot missing | STOP | Create snapshot | Must be before start |
| Manifest missing | STOP | Generate manifest | Re-run pre-flight |
| Log corruption | STOP | Repair/restore | Re-run pre-flight |
| Tables missing | STOP | Run migrations | Re-run pre-flight |
| Status pending | STOP | Start experiment | Wait for cycles |
| Prereg uncommitted | WARN | Commit changes | Document |
| Verifications missing | WARN | Reconstruct if possible | Note in report |
| Partial audit | WARN | Document scope | Proceed with caution |

---

### Pre-Flight Command Reference

```bash
# Run full pre-flight check
python tools/u2_preflight.py --exp-id <exp_id>

# Run specific gate
python tools/u2_preflight.py --exp-id <exp_id> --gate PRE-3

# Generate pre-flight report
python tools/u2_preflight.py --exp-id <exp_id> --output preflight_<exp_id>.json

# Check eligibility status only
python tools/u2_preflight.py --exp-id <exp_id> --status-only

# List all experiments with eligibility status
python tools/u2_preflight.py --list-all
```

---

### Pre-Flight Report Format

```json
{
  "experiment_id": "u2_uplift_2025_06_15",
  "preflight_timestamp": "2025-06-15T01:55:00Z",
  "eligibility_status": "ELIGIBLE",
  "gates": {
    "PRE-1": {"status": "PASS", "checks": [1,2,3,4,5], "passed": 5, "failed": 0},
    "PRE-2": {"status": "PASS", "checks": [6,7,8], "passed": 3, "failed": 0},
    "PRE-3": {"status": "PASS", "checks": [9,10,11,12,13], "passed": 5, "failed": 0},
    "PRE-4": {"status": "WARN", "checks": [14,15,16,17,18,19,20], "passed": 6, "warned": 1},
    "PRE-5": {"status": "PASS", "checks": [21,22,23], "passed": 3, "failed": 0},
    "PRE-6": {"status": "PASS", "checks": [24,25], "passed": 2, "failed": 0}
  },
  "warnings": [
    {"check": 19, "message": "verifications.jsonl missing", "impact": "Verification receipts unavailable"}
  ],
  "fatal_conditions": [],
  "stop_conditions": [],
  "recommendation": "PROCEED_TO_AUDIT",
  "notes": "Experiment is audit-eligible. One WARN condition documented."
}
```

---

## Appendix: Pre-Flight Checklist Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     U2 PRE-FLIGHT AUDIT CHECKLIST                           │
│                     ═══════════════════════════════                         │
│                                                                             │
│  Experiment ID: ________________________  Date: _______________             │
│                                                                             │
│  PRE-1: REGISTRATION                                           GATE: ___   │
│  [ ] 1. Experiment ID exists in u2_experiments                              │
│  [ ] 2. Theory ID is valid FK                                               │
│  [ ] 3. Slice ID is non-empty                                               │
│  [ ] 4. Status is recognized value                                          │
│  [ ] 5. Start time set (if running/completed)                               │
│                                                                             │
│  PRE-2: PREREGISTRATION                                        GATE: ___   │
│  [ ] 6. Experiment in PREREG_UPLIFT_U2.yaml                                 │
│  [ ] 7. Preregistration hash matches                                        │
│  [ ] 8. Prereg file has no uncommitted changes                              │
│                                                                             │
│  PRE-3: BASELINE SNAPSHOT                                      GATE: ___   │
│  [ ] 9. Snapshot record exists                                              │
│  [ ] 10. Merkle root is non-null                                            │
│  [ ] 11. Statement count recorded                                           │
│  [ ] 12. Edge count recorded                                                │
│  [ ] 13. Snapshot timestamp < experiment start                              │
│                                                                             │
│  PRE-4: LOG DIRECTORY                                          GATE: ___   │
│  [ ] 14. Log directory exists                                               │
│  [ ] 15. manifest.json exists                                               │
│  [ ] 16. manifest.json is valid JSON                                        │
│  [ ] 17. At least one cycle log exists                                      │
│  [ ] 18. Cycle logs are parseable JSONL                                     │
│  [ ] 19. verifications.jsonl exists                                         │
│  [ ] 20. verifications.jsonl is parseable                                   │
│                                                                             │
│  PRE-5: DATABASE                                               GATE: ___   │
│  [ ] 21. Database connection succeeds                                       │
│  [ ] 22. All U2 tables exist                                                │
│  [ ] 23. Baseline tables exist                                              │
│                                                                             │
│  PRE-6: STATE ELIGIBILITY                                      GATE: ___   │
│  [ ] 24. Status is auditable                                                │
│  [ ] 25. If running, has completed cycles                                   │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  ELIGIBILITY STATUS:  [ ] ELIGIBLE   [ ] BLOCKED   [ ] INADMISSIBLE        │
│                                                                             │
│  FATAL conditions: ________________________________________________         │
│  STOP conditions:  ________________________________________________         │
│  WARN conditions:  ________________________________________________         │
│                                                                             │
│  Operator: ________________________  Signature: _______________             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [PROOF_DAG_INVARIANTS.md](./PROOF_DAG_INVARIANTS.md) — Full invariant specification
- [DAG_SPEC.md](./DAG_SPEC.md) — DAG schema requirements
- [DETERMINISM_CONTRACT.md](./DETERMINISM_CONTRACT.md) — Determinism requirements
- [RFL_PHASE_I_TRUTH_SOURCE.md](./RFL_PHASE_I_TRUTH_SOURCE.md) — Phase I reference
