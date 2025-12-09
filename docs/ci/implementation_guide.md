# CI Chain Implementation Guide

**Author**: Manus-B (Ledger Replay Architect & PQ Migration Officer)  
**Phase**: IV - Consensus Integration & Enforcement  
**Date**: 2025-12-09  
**Status**: Implementation Ready

---

## Purpose

Implement governance chain CI without requiring GitHub Actions workflow permissions.

**Approach**: Standalone shell script (`run_governance_chain.sh`) that can be invoked from any CI environment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CI Environment (Any)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ GitHub       │  │ GitLab       │  │ Local        │      │
│  │ Actions      │  │ CI           │  │ Machine      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Invoke run_governance_chain.sh
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Governance Chain Runner                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Job 1        │  │ Job 2        │  │ Job 3        │      │
│  │ Schema       │  │ Epoch        │  │ Replay       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Job 4        │  │ Job 5        │  │ Job 6        │      │
│  │ Drift        │  │ Attestation  │  │ PQ           │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐                                           │
│  │ Job 7        │                                           │
│  │ Governance   │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Standalone Script

**File**: `scripts/ci/run_governance_chain.sh`

**Features**:
- No GitHub Actions workflow permissions required
- Can be run locally or in any CI environment
- Configurable via environment variables
- Colored output for readability
- Exit codes for CI integration

---

## Usage

### Local Usage

```bash
# Set environment variables
export DATABASE_URL="postgresql://postgres:test@localhost:5432/mathledger_test"
export START_BLOCK=0
export END_BLOCK=1000
export REPLAY_MODE="sliding_window"
export GOVERNANCE_POLICY="strict"
export FAIL_ON_BLOCK="true"
export OUTPUT_DIR="./reports"

# Run governance chain
./scripts/ci/run_governance_chain.sh
```

---

### GitHub Actions Integration

**File**: `.github/workflows/governance-chain.yml`

```yaml
name: Governance Chain

on:
  push:
    branches:
      - main
      - develop
  pull_request:
    branches:
      - main
      - develop

jobs:
  governance-chain:
    name: Governance Chain
    runs-on: ubuntu-latest
    timeout-minutes: 35
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: mathledger_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install psycopg2-binary
      
      - name: Load test data
        run: |
          python3 scripts/load_test_data.py \
            --database-url postgresql://postgres:test@localhost:5432/mathledger_test \
            --blocks 1000 \
            --hash-version sha256-v1
      
      - name: Run governance chain
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/mathledger_test
          START_BLOCK: 0
          END_BLOCK: 1000
          REPLAY_MODE: sliding_window
          GOVERNANCE_POLICY: strict
          FAIL_ON_BLOCK: true
          OUTPUT_DIR: ./reports
        run: |
          ./scripts/ci/run_governance_chain.sh
      
      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: governance-reports
          path: reports/
          retention-days: 30
```

---

### GitLab CI Integration

**File**: `.gitlab-ci.yml`

```yaml
governance-chain:
  stage: test
  image: python:3.11
  services:
    - postgres:15
  variables:
    POSTGRES_DB: mathledger_test
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: test
    DATABASE_URL: postgresql://postgres:test@postgres:5432/mathledger_test
    START_BLOCK: 0
    END_BLOCK: 1000
    REPLAY_MODE: sliding_window
    GOVERNANCE_POLICY: strict
    FAIL_ON_BLOCK: "true"
    OUTPUT_DIR: ./reports
  before_script:
    - pip install -r requirements.txt
    - pip install psycopg2-binary
    - python3 scripts/load_test_data.py --database-url $DATABASE_URL --blocks 1000 --hash-version sha256-v1
  script:
    - ./scripts/ci/run_governance_chain.sh
  artifacts:
    paths:
      - reports/
    expire_in: 30 days
    when: always
```

---

### CircleCI Integration

**File**: `.circleci/config.yml`

```yaml
version: 2.1

jobs:
  governance-chain:
    docker:
      - image: cimg/python:3.11
      - image: cimg/postgres:15.0
        environment:
          POSTGRES_DB: mathledger_test
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: test
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: |
            pip install -r requirements.txt
            pip install psycopg2-binary
      - run:
          name: Load test data
          command: |
            python3 scripts/load_test_data.py \
              --database-url postgresql://postgres:test@localhost:5432/mathledger_test \
              --blocks 1000 \
              --hash-version sha256-v1
      - run:
          name: Run governance chain
          environment:
            DATABASE_URL: postgresql://postgres:test@localhost:5432/mathledger_test
            START_BLOCK: 0
            END_BLOCK: 1000
            REPLAY_MODE: sliding_window
            GOVERNANCE_POLICY: strict
            FAIL_ON_BLOCK: true
            OUTPUT_DIR: ./reports
          command: |
            ./scripts/ci/run_governance_chain.sh
      - store_artifacts:
          path: reports/
          destination: governance-reports

workflows:
  version: 2
  test:
    jobs:
      - governance-chain
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:test@localhost:5432/mathledger_test` |
| `START_BLOCK` | Start block for verification | `0` |
| `END_BLOCK` | End block for verification | `1000` |
| `REPLAY_MODE` | Replay verification mode | `sliding_window` |
| `GOVERNANCE_POLICY` | Governance policy | `strict` |
| `FAIL_ON_BLOCK` | Fail on BLOCK signal | `true` |
| `OUTPUT_DIR` | Output directory for reports | `./reports` |

---

### Replay Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `full_chain` | Verify all blocks | Nightly builds, main branch |
| `sliding_window` | Verify last N blocks | PRs, develop branch |
| `epoch_validation` | Verify epoch roots only | Feature branches |

---

### Governance Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `strict` | Zero tolerance for CRITICAL/HIGH drift | Production, main branch |
| `moderate` | Allow some HIGH drift | Develop branch |
| `permissive` | Allow exploration | Feature branches |

---

## Jobs

### Job 1: Schema Migration Check

**Script**: `scripts/ci/validate_migrations.py`

**Checks**:
- SQL syntax validation
- Backward compatibility
- Hash-law invariants
- Monotonicity enforcement

**Output**: `reports/migration_validation.json`

---

### Job 2: Epoch Backfill Dry Run

**Script**: `scripts/backfill_epochs.py`

**Checks**:
- Epoch size consistency
- Hash algorithm selection
- Merkle root computation

**Output**: `reports/epoch_backfill.json`

---

### Job 3: Replay Verification

**Script**: `scripts/replay_verify.py`

**Checks**:
- Recompute attestation roots
- Verify hash matches
- Check replay success rate (must be 100%)

**Output**: `reports/replay_verification_{mode}.json`

---

### Job 4: Drift Radar Scan

**Script**: `scripts/ci/drift_radar_scan.py`

**Checks**:
- Detect drift signals (schema, hash-delta, metadata, statement)
- Classify drift severity
- Evaluate governance signal

**Output**: `reports/drift_signals.json`, `reports/evidence_pack.json`

---

### Job 5: Attestation Integrity Sweep

**Script**: `scripts/ci/attestation_integrity_sweep.py`

**Checks**:
- Dual-root computation
- Merkle tree structure
- Domain separation

**Output**: `reports/attestation_integrity.json`

---

### Job 6: PQ Activation Readiness Audit

**Script**: `scripts/ci/check_pq_migration_code.py`

**Checks**:
- Dual-commitment support
- SHA-3 support
- Cross-algorithm validation
- Epoch transition support

**Output**: `reports/pq_readiness.json`

---

### Job 7: Governance Gate

**Script**: `scripts/ci/aggregate_governance_results.py`, `scripts/ci/check_governance_gate.py`

**Checks**:
- Aggregate all reports
- Check pass/fail
- Generate summary

**Output**: `reports/governance_summary.json`

---

## Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | All checks passed |
| 1 | One or more checks failed |

---

## Output Reports

All reports are JSON files in `reports/` directory:

| Report | Description |
|--------|-------------|
| `migration_validation.json` | Schema migration validation |
| `epoch_backfill.json` | Epoch backfill dry run results |
| `replay_verification_{mode}.json` | Replay verification results |
| `drift_signals.json` | Drift signals detected |
| `evidence_pack.json` | Governance evidence pack |
| `attestation_integrity.json` | Attestation integrity results |
| `pq_readiness.json` | PQ activation readiness |
| `governance_summary.json` | Governance summary |

---

## Implementation Checklist

- [x] Create `run_governance_chain.sh` script
- [ ] Implement `scripts/ci/validate_migrations.py`
- [ ] Implement `scripts/ci/check_replay_success_rate.py`
- [ ] Implement `scripts/ci/drift_radar_scan.py`
- [ ] Implement `scripts/ci/check_governance_signal.py`
- [ ] Implement `scripts/ci/attestation_integrity_sweep.py`
- [ ] Implement `scripts/ci/check_integrity_violations.py`
- [ ] Implement `scripts/ci/check_pq_migration_code.py`
- [ ] Implement `scripts/ci/aggregate_governance_results.py`
- [ ] Implement `scripts/ci/check_governance_gate.py`
- [ ] Test locally with test database
- [ ] Integrate into GitHub Actions (optional)
- [ ] Document for team

---

## Conclusion

The CI Chain implementation provides a **standalone, portable governance enforcement pipeline** that can be run in any CI environment without requiring GitHub Actions workflow permissions.

**Key Benefits**:
1. **Portable**: Works in any CI environment (GitHub Actions, GitLab CI, CircleCI, local)
2. **Configurable**: Environment variables for all settings
3. **Fail-fast**: Exits on first failure for fast feedback
4. **Comprehensive**: 7 jobs covering all governance aspects

**Status**: Implementation ready, scripts pending.

---

**"Keep it blue, keep it clean, keep it sealed."**  
— Manus-B, Ledger Replay Architect & PQ Migration Officer
