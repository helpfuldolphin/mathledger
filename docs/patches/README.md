# Determinism Fixes - One-Click Patch

This directory contains patches to fix nondeterministic operations in MathLedger codebase.

## Install Mandatory Gate (GitHub UI)

To install the determinism gate workflow via GitHub UI:

1. Navigate to: https://github.com/helpfuldolphin/mathledger/new/integrate/ledger-v0.1
2. Set filename: `.github/workflows/determinism-guard.yml`
3. Copy content from: `/tmp/determinism-guard_workflow.yml` (or see below)
4. Commit directly to `integrate/ledger-v0.1`
5. Verify gate appears in: Settings > Branches > integrate/ledger-v0.1 > Required checks

## Quick Apply (Automated)

```bash
# Autofix assistant - applies patch and verifies
uv run python tools/repro/autofix_drift.py --apply --verify

# Manual application
git apply docs/patches/determinism-fixes.diff

# Verify fixes
python tools/repro/drift_sentinel.py --all --whitelist artifacts/repro/drift_whitelist.json
python tools/repro/seed_replay_guard.py --seed 0 --runs 3 --path artifacts/repro/
```

## What This Patch Fixes

The patch replaces all nondeterministic operations in critical proof derivation files:

### Replacements Made

1. **datetime.utcnow()** → **deterministic_timestamp(_GLOBAL_SEED)**
   - Files: backend/axiom_engine/derive.py
   - Impact: Timestamps in proof metadata become deterministic

2. **time.time()** → **deterministic_unix_timestamp(_GLOBAL_SEED)**
   - Files: backend/axiom_engine/derive.py, backend/ledger/blocking.py
   - Impact: Unix timestamps become deterministic

3. **np.random.random()** → **SeededRNG(_GLOBAL_SEED).random()**
   - Files: backend/axiom_engine/policy.py
   - Impact: Random number generation becomes deterministic

4. **uuid.uuid4()** → **deterministic_uuid(content)**
   - Files: backend/axiom_engine/model.py
   - Impact: UUIDs become content-based and deterministic

### Critical Files Patched

- `backend/axiom_engine/derive.py` - Core derivation engine
- `backend/axiom_engine/policy.py` - Policy-guided derivation
- `backend/ledger/blocking.py` - Block sealing
- `backend/axiom_engine/model.py` - Database models

## Verification

After applying the patch:

1. **Drift Sentinel** - Checks for nondeterministic calls
   ```bash
   python tools/repro/drift_sentinel.py --all --whitelist artifacts/repro/drift_whitelist.json
   # Expected: [PASS] Drift Sentinel: 0 violations
   ```

2. **Determinism Guard** - Verifies byte-identical runs
   ```bash
   python tools/repro/seed_replay_guard.py --seed 0 --runs 3 --path artifacts/repro/
   # Expected: [PASS] Determinism Guard: 3/3 byte-identical runs
   ```

## Rollback

If the patch causes issues:

```bash
git apply --reverse docs/patches/determinism-fixes.diff
```

## Manual Application

If `git apply` fails, you can manually apply the changes by following the patterns:

### Pattern 1: datetime.utcnow()

```python
# BEFORE
import datetime
timestamp = datetime.datetime.utcnow()

# AFTER
from backend.repro.determinism import deterministic_timestamp
timestamp = deterministic_timestamp(_GLOBAL_SEED)
```

### Pattern 2: time.time()

```python
# BEFORE
import time
unix_time = time.time()

# AFTER
from backend.repro.determinism import deterministic_unix_timestamp
unix_time = deterministic_unix_timestamp(_GLOBAL_SEED)
```

### Pattern 3: np.random

```python
# BEFORE
import numpy as np
value = np.random.random()

# AFTER
from backend.repro.determinism import SeededRNG
value = SeededRNG(_GLOBAL_SEED).random()
```

### Pattern 4: uuid.uuid4()

```python
# BEFORE
import uuid
id = str(uuid.uuid4())

# AFTER
from backend.repro.determinism import deterministic_uuid
id = deterministic_uuid(content_string)
```

## Support

For issues or questions:
- Review: `docs/repro/DRIFT_RESPONSE_PLAYBOOK.md`
- Verify: `tools/repro/test_determinism.py`
- Report: Create GitHub issue with drift_report.json attached
