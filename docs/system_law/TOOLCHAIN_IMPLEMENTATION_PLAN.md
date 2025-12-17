# Toolchain Snapshot Implementation Plan

**Status**: SAVE TO REPO: YES
**Rationale**: Documents exact changes for reproducibility infrastructure. Required for audit trail.

---

## 1. Scope

Make CAL-EXP-1 (and successors) cryptographically reproducible by:
1. Capturing all toolchain inputs in a single fingerprint
2. Recording fingerprint in experiment manifests
3. Providing verification scripts for CI parity

## 2. Files to Add

### 2.1 New Files

| File | Purpose | Lines |
|------|---------|-------|
| `substrate/repro/toolchain.py` | Toolchain fingerprint computation | ~220 |
| `scripts/verify_toolchain_parity.py` | CI parity verification | ~100 |
| `docs/system_law/TOOLCHAIN_SNAPSHOT_SPEC.md` | Specification document | ~180 |
| `.github/workflows/toolchain-parity.yml` | CI workflow (optional) | ~40 |
| `toolchain_baseline.json` | Locked baseline fingerprint | ~25 |

### 2.2 Files to Update

| File | Change | Rationale |
|------|--------|-----------|
| `results/cal_exp_1/cal_exp_1_harness.py` | Add toolchain block to manifest | Captures full fingerprint |
| `experiments/manifest.py` | Import and use toolchain module | Reusable across experiments |
| `substrate/repro/__init__.py` | Export toolchain module | API exposure |

## 3. Implementation Details

### 3.1 `substrate/repro/toolchain.py` (Created)

**Exports**:
- `ToolchainSnapshot` - Dataclass with all version info
- `capture_toolchain_snapshot(repo_root)` - Main capture function
- `compute_toolchain_fingerprint(...)` - Hash computation
- `verify_toolchain_match(snap1, snap2)` - Comparison function
- `save_toolchain_snapshot(snap, path)` - JSON serialization
- `load_toolchain_snapshot(path)` - JSON deserialization

**CLI Interface**:
```bash
# Capture current snapshot
python -m substrate.repro.toolchain --json

# Save to file
python -m substrate.repro.toolchain -o toolchain_snapshot.json

# Verify against baseline
python -m substrate.repro.toolchain -v toolchain_baseline.json
```

### 3.2 `scripts/verify_toolchain_parity.py` (To Create)

**Purpose**: Verify local toolchain matches CI baseline.

**Algorithm**:
1. Load `toolchain_baseline.json` from repo root
2. Capture current toolchain snapshot
3. Compare fingerprints
4. Exit 0 on match, exit 1 on mismatch with diff

**Usage**:
```bash
python scripts/verify_toolchain_parity.py
# Exit 0: PASS - Toolchain matches baseline
# Exit 1: FAIL - Toolchain mismatch (details printed)
```

### 3.3 `toolchain_baseline.json` (To Create)

**Purpose**: Locked baseline for CI parity checks.

**Structure**:
```json
{
  "schema_version": "1.0",
  "fingerprint": "<sha256>",
  "python": {
    "version": "3.11.9",
    "uv_version": "0.8.16",
    "uv_lock_hash": "<sha256>"
  },
  "lean": {
    "version": "leanprover/lean4:v4.23.0-rc2",
    "toolchain_hash": "<sha256>",
    "lake_manifest_hash": "<sha256>",
    "lakefile_hash": "<sha256>"
  },
  "platform": {
    "os": "...",
    "arch": "...",
    "hostname": "..."
  }
}
```

**Update Protocol**:
1. Only update via explicit PR with justification
2. PR must include dependency changelog
3. All experiments must pass with new baseline before merge

### 3.4 CAL-EXP-1 Harness Update

**Current** (`results/cal_exp_1/cal_exp_1_harness.py:448-461`):
```python
uv_lock_path = Path(__file__).parent.parent.parent / "uv.lock"
if uv_lock_path.exists():
    uv_hash = hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
```

**Updated**:
```python
from substrate.repro.toolchain import capture_toolchain_snapshot

toolchain = capture_toolchain_snapshot()
pre_run_checks.append(PreRunCheck(
    check="Toolchain fingerprint computed",
    result="PASS",
    evidence=toolchain.fingerprint
))
```

**Manifest Change** (line 560-563):
```python
# Before
"toolchain_hash": uv_hash if 'uv_hash' in dir() else "UNKNOWN",

# After
"toolchain": toolchain.to_dict(),
```

### 3.5 `substrate/repro/__init__.py` Update

**Add**:
```python
from .toolchain import (
    ToolchainSnapshot,
    capture_toolchain_snapshot,
    verify_toolchain_match,
)
```

## 4. Manifest Schema Changes

### 4.1 Current Schema (CAL-EXP-1)

```json
{
  "schema_version": "1.0.0",
  "experiment_id": "CAL-EXP-1",
  "toolchain_hash": "<uv.lock hash only>",
  ...
}
```

### 4.2 New Schema (v1.1.0)

```json
{
  "schema_version": "1.1.0",
  "experiment_id": "CAL-EXP-1",
  "toolchain": {
    "schema_version": "1.0",
    "fingerprint": "<combined hash>",
    "python": { ... },
    "lean": { ... },
    "platform": { ... }
  },
  ...
}
```

**Backward Compatibility**: `toolchain_hash` field removed. Consumers must migrate to `toolchain.fingerprint`.

## 5. CI Workflow (Optional)

**File**: `.github/workflows/toolchain-parity.yml`

```yaml
name: Toolchain Parity Check

on:
  push:
    paths:
      - 'uv.lock'
      - 'backend/lean_proj/lean-toolchain'
      - 'backend/lean_proj/lake-manifest.json'
      - 'backend/lean_proj/lakefile.lean'
  pull_request:
    paths:
      - 'uv.lock'
      - 'backend/lean_proj/lean-toolchain'
      - 'backend/lean_proj/lake-manifest.json'
      - 'backend/lean_proj/lakefile.lean'

jobs:
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
      - run: uv sync
      - run: python scripts/verify_toolchain_parity.py
```

## 6. Verification Protocol

### 6.1 Local Verification

```bash
# 1. Capture current snapshot
python -m substrate.repro.toolchain -o current_snapshot.json

# 2. Compare with baseline
python -m substrate.repro.toolchain -v toolchain_baseline.json

# 3. Run reproducibility test
python scripts/verify_reproducibility.py
```

### 6.2 CI Verification

1. CI captures toolchain snapshot
2. Compares against `toolchain_baseline.json`
3. Fails PR if mismatch detected
4. Stores snapshot as artifact for debugging

## 7. Migration Checklist

- [x] Create `substrate/repro/toolchain.py`
- [x] Create `docs/system_law/TOOLCHAIN_SNAPSHOT_SPEC.md`
- [x] Create `docs/system_law/TOOLCHAIN_IMPLEMENTATION_PLAN.md`
- [ ] Create `scripts/verify_toolchain_parity.py`
- [ ] Create `scripts/verify_reproducibility.py`
- [ ] Create `toolchain_baseline.json`
- [ ] Update `substrate/repro/__init__.py`
- [ ] Update `results/cal_exp_1/cal_exp_1_harness.py`
- [ ] Update `experiments/manifest.py`
- [ ] (Optional) Add CI workflow

## 8. Testing

### 8.1 Unit Tests

**File**: `tests/substrate/test_toolchain.py`

```python
def test_fingerprint_deterministic():
    """Same files produce same fingerprint."""
    snap1 = capture_toolchain_snapshot()
    snap2 = capture_toolchain_snapshot()
    assert snap1.fingerprint == snap2.fingerprint

def test_fingerprint_changes_on_file_change():
    """Modified file changes fingerprint."""
    # Requires mocking file contents

def test_verify_match():
    """Matching snapshots verify correctly."""
    snap = capture_toolchain_snapshot()
    match, diffs = verify_toolchain_match(snap, snap)
    assert match
    assert len(diffs) == 0
```

### 8.2 Integration Test

**File**: `scripts/verify_reproducibility.py` (Reproducibility Test)

See next section.

## 9. Constraints

- No vendored binaries
- No compiled artifacts committed
- No experiment logic changes
- No new enforcement semantics
- Hashes over descriptions
- Metadata over binaries
- No speculative future-proofing

---

## Appendix: File Sizes

| File | Est. Lines | Purpose |
|------|------------|---------|
| `toolchain.py` | 220 | Core module |
| `verify_toolchain_parity.py` | 80 | CI parity |
| `verify_reproducibility.py` | 120 | Repro test |
| `TOOLCHAIN_SNAPSHOT_SPEC.md` | 180 | Specification |
| `TOOLCHAIN_IMPLEMENTATION_PLAN.md` | 250 | This document |
| `toolchain_baseline.json` | 25 | Baseline |
| **Total** | ~875 | Minimal scope |
