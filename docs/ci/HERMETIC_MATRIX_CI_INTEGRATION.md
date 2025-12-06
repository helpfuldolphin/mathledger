# Hermetic Matrix CI Integration

Automated hermetic matrix health tracking with time-series monitoring and drift detection.

## Overview

The Hermetic Matrix Health workflow automatically tracks H_matrix seal stability and lane health over time after every CI run. It provides:

- **Time-Series Tracking**: Appends H_matrix + lane bitmap to JSONL history
- **Drift Detection**: Alerts when H_matrix changes across runs (ABSTAIN reporting)
- **GitHub Actions Integration**: ASCII reports in workflow summaries
- **Artifact Archival**: 90-day retention of history and reports
- **Automatic Commits**: History committed back to repository (main/integrate only)

## Architecture

### Workflow Trigger

```yaml
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
  workflow_dispatch:  # Manual trigger for testing
```

The workflow runs automatically after CI completion or can be manually triggered via GitHub Actions UI.

### Workflow Steps

1. **Checkout Repository**: Fetches code with full history
2. **Setup Python**: Installs Python 3.11
3. **Validate Current State**: Runs `matrix_trend.py --validate`
4. **Append History**: Runs `matrix_trend.py --append`
5. **Validate History**: Runs `matrix_trend.py --validate-history --required-runs 3`
6. **Check for Drift**: Runs `matrix_trend.py --alert-drift-abstain --threshold 3`
7. **Generate Report**: Runs `matrix_trend.py --save-report --limit 10`
8. **Upload Artifacts**: Uploads history, report, and drift report (90-day retention)
9. **Commit History**: Commits updated history to repository (main/integrate only)

### Fork-Safe Design

- No secrets required
- No external API calls
- Reads from committed artifacts/
- NO_NETWORK=true compatible
- Runs on forks without modification

## Manual Application

Due to GitHub OAuth workflow scope limitations, the workflow file must be applied manually.

### Method 1: GitHub Web UI (Recommended)

1. Navigate to: https://github.com/helpfuldolphin/mathledger/tree/integrate/ledger-v0.1/.github/workflows
2. Click "Add file" → "Create new file"
3. Name: `hermetic-matrix-health.yml`
4. Copy contents from: `docs/ci/workflows/hermetic-matrix-health.yml`
5. Commit message: `ci: add hermetic matrix health tracking workflow [IVL]`
6. Commit directly to `integrate/ledger-v0.1` branch

### Method 2: Local Git (Requires Workflow Scope)

```bash
git checkout integrate/ledger-v0.1
git pull

# Copy workflow file
cp docs/ci/workflows/hermetic-matrix-health.yml .github/workflows/hermetic-matrix-health.yml

git add .github/workflows/hermetic-matrix-health.yml
git commit -m "ci: add hermetic matrix health tracking workflow [IVL]"
git push
```

### Verification

```bash
# Check workflow file exists
ls -la .github/workflows/hermetic-matrix-health.yml

# View workflow in GitHub CLI
gh workflow view hermetic-matrix-health

# Manually trigger workflow
gh workflow run hermetic-matrix-health --ref integrate/ledger-v0.1
```

## Expected Behavior

### First Run

After workflow application, the next CI run will automatically trigger the matrix health workflow.

**Expected Output:**
```
✅ Validate Current Hermetic State
   [PASS] NO_NETWORK HERMETIC v2 TRUE
   [PASS] Hermetic Matrix 08752f3466fcc8c7...

✅ Append Matrix History
   [INFO] Appended history entry: 08752f3466fcc8c7... bitmap=111111

✅ Validate History
   [PASS] Hermetic History: 4/4 sealed

✅ Check for Drift
   [PASS] No drift or consecutive failures in last 3 runs

✅ Generate Matrix Report
   Report saved to matrix_report.txt

✅ Upload Matrix Artifacts
   Artifacts uploaded: hermetic-matrix-health-{run_number}.zip

✅ Commit Updated History
   Committed updated matrix_history.jsonl
```

### GitHub Actions Summary

The workflow uploads an ASCII report to the GitHub Actions summary:

```
## 🔒 Hermetic Matrix Health Report

================================================================================
HERMETIC MATRIX TREND REPORT
================================================================================
Total Entries: 4
Showing Recent: 4

Timestamp            H_matrix         Bitmap   P/F/A    AllBlue 
--------------------------------------------------------------------------------
2025-11-01T01:37:01  08752f3466fcc8c7 111111   6/0/0    YES     
2025-11-01T02:41:21  08752f3466fcc8c7 111111   6/0/0    YES     
2025-11-01T02:41:27  08752f3466fcc8c7 111111   6/0/0    YES     
2025-11-01T03:15:42  08752f3466fcc8c7 111111   6/0/0    YES     
================================================================================

SUMMARY STATISTICS
--------------------------------------------------------------------------------
All Blue Runs: 4/4 (100.0%)
[PASS] No H_matrix drift in recent runs
================================================================================
```

### Drift Detection Scenario

When H_matrix changes across runs (drift detected):

**Console Output:**
```
⚠️ Check for Drift
   [ABSTAIN] H_matrix drift detected in last 3 runs
   Reason: H_matrix changed across 3 runs: 2 unique seals
   Drift report saved: drift_report.txt
```

**GitHub Actions Summary (With Drift):**
```
## 🔒 Hermetic Matrix Health Report
[normal report]

## ⚠️ Drift Report

================================================================================
HERMETIC MATRIX DRIFT REPORT
================================================================================
Drift detected in last 3 runs:

Reason: H_matrix changed across 3 runs: 2 unique seals

Run   Timestamp            H_matrix         Bitmap   P/F/A   
--------------------------------------------------------------------------------
1     2025-11-01T01:37:01  08752f3466fcc8c7 111111   6/0/0   
2     2025-11-01T02:41:21  abcdef1234567890 111111   6/0/0   
3     2025-11-01T02:41:27  08752f3466fcc8c7 111111   6/0/0   
================================================================================
```

## Artifacts

### matrix_history.jsonl

JSONL file tracking H_matrix and lane health over time.

**Format:**
```json
{"timestamp": "2025-11-01T01:37:01.569213Z", "h_matrix": "08752f3466fcc8c7d69d890e80b1b7f3e8fc1003b3ec845259c640c948cada89", "lane_bitmap": "111111", "pass_count": 6, "fail_count": 0, "abstain_count": 0, "all_blue": true}
```

**Fields:**
- `timestamp`: ISO 8601 timestamp with timezone
- `h_matrix`: SHA256 seal of hermetic matrix manifest
- `lane_bitmap`: 6-character bitmap (1=PASS, 0=FAIL, A=ABSTAIN)
- `pass_count`: Number of passing lanes
- `fail_count`: Number of failing lanes
- `abstain_count`: Number of abstained lanes
- `all_blue`: Boolean indicating all lanes hermetic

**Lane Order:**
1. da_ui
2. da_reasoning
3. da_composite
4. browsermcp
5. uplift-omega
6. test

### matrix_report.txt

ASCII table showing recent matrix trend.

**Format:**
```
================================================================================
HERMETIC MATRIX TREND REPORT
================================================================================
Total Entries: 3
Showing Recent: 3

Timestamp            H_matrix         Bitmap   P/F/A    AllBlue 
--------------------------------------------------------------------------------
2025-11-01T01:37:01  08752f3466fcc8c7 111111   6/0/0    YES     
2025-11-01T02:41:21  08752f3466fcc8c7 111111   6/0/0    YES     
2025-11-01T02:41:27  08752f3466fcc8c7 111111   6/0/0    YES     
================================================================================

SUMMARY STATISTICS
--------------------------------------------------------------------------------
All Blue Runs: 3/3 (100.0%)
[PASS] No H_matrix drift in recent runs
================================================================================
```

### drift_report.txt

Generated when drift is detected.

**Format:**
```
================================================================================
HERMETIC MATRIX DRIFT REPORT
================================================================================
Drift detected in last 3 runs:

Reason: H_matrix changed across 3 runs: 2 unique seals

Run   Timestamp            H_matrix         Bitmap   P/F/A   
--------------------------------------------------------------------------------
1     2025-11-01T01:37:01  08752f3466fcc8c7 111111   6/0/0   
2     2025-11-01T02:41:21  abcdef1234567890 111111   6/0/0   
3     2025-11-01T02:41:27  08752f3466fcc8c7 111111   6/0/0   
================================================================================
```

### global_matrix.json

JSON file with cross-shard verification summary.

**Format:**
```json
{
  "timestamp": "2025-11-02T18:03:52.067866+00:00",
  "num_shards": 16,
  "global_h_matrix_root": "f23460c46b9a3ede8f1b07aab568911b24e0181ce0ed8cfdc0873688e427737a",
  "verification_status": "verified",
  "missing_shards": [],
  "total_entries": 48,
  "total_all_blue": 48,
  "all_blue_percentage": 100.0,
  "shards": [
    {
      "shard_id": 0,
      "entries": 3,
      "all_blue_count": 3,
      "latest_h_matrix": "test_hash_00_02_aaaa...",
      "status": "present"
    }
  ]
}
```

**Fields:**
- `timestamp`: ISO 8601 timestamp with timezone
- `num_shards`: Number of shards aggregated (default 16)
- `global_h_matrix_root`: SHA256(concat of all shard roots)
- `verification_status`: "verified" or "abstain"
- `missing_shards`: List of missing shard IDs
- `total_entries`: Total history entries across all shards
- `total_all_blue`: Total All Blue runs across all shards
- `all_blue_percentage`: Percentage of All Blue runs
- `shards`: Per-shard statistics

### shard_XX.jsonl

JSONL files tracking per-shard history (shard_00.jsonl through shard_15.jsonl).

**Format:**
```json
{"timestamp": "2025-11-02T18:03:52.067866+00:00", "h_matrix": "test_hash_00_02_aaaa...", "lane_bitmap": "111111", "pass_count": 6, "fail_count": 0, "abstain_count": 0, "all_blue": true}
```

**Location:** `artifacts/hermetic/shard_XX.jsonl`

## Cross-Shard Verification

### Overview

Cross-shard verification aggregates matrix history from 16 independent shards and computes a global H_matrix root for distributed verification.

**Architecture:**
- 16 shards: `shard_00.jsonl` through `shard_15.jsonl`
- Shard root: Latest H_matrix from each shard
- Global root: SHA256(concat of all 16 shard roots)
- ABSTAIN on missing shards (Proof-or-Abstain discipline)

### Verification Process

1. **Load Shard Histories**: Read all 16 shard JSONL files
2. **Compute Shard Roots**: Extract latest H_matrix from each shard
3. **Concatenate Roots**: Concatenate all 16 shard roots in order
4. **Compute Global Root**: SHA256(concatenated_roots)
5. **Generate Summary**: Create global_matrix.json with statistics

### Pass-Lines

**Success (All Shards Present):**
```
[PASS] Global H-Matrix Verified f23460c46b9a3ede8f1b07aab568911b24e0181ce0ed8cfdc0873688e427737a
```

**ABSTAIN (Missing Shards):**
```
[ABSTAIN] Missing shard data: shard_05, shard_12
```

### Local Testing

**Verify Global Matrix:**
```bash
python tools/hermetic/matrix_trend.py --verify-global --num-shards 16
# Expected (success): [PASS] Global H-Matrix Verified <sha256>
# Expected (missing): [ABSTAIN] Missing shard data: shard_XX, shard_YY
```

**Generate Global Summary:**
```bash
python tools/hermetic/matrix_trend.py --save-global --num-shards 16
# Expected: [INFO] Global matrix saved: artifacts/hermetic/global_matrix.json
```

**View Global Summary:**
```bash
cat artifacts/hermetic/global_matrix.json | python -m json.tool
```

### Workflow Integration

The hermetic-matrix-health.yml workflow includes cross-shard verification:

```yaml
- name: Verify Global Matrix
  run: |
    python tools/hermetic/matrix_trend.py --verify-global --num-shards 16

- name: Generate Global Matrix Summary
  run: |
    python tools/hermetic/matrix_trend.py --save-global --num-shards 16
```

**GitHub Actions Summary:**
```
## 🌐 Global Matrix Summary

{
  "timestamp": "2025-11-02T18:03:52.067866+00:00",
  "num_shards": 16,
  "global_h_matrix_root": "f23460c46b9a3ede8f1b07aab568911b24e0181ce0ed8cfdc0873688e427737a",
  "verification_status": "verified",
  "total_entries": 48,
  "total_all_blue": 48,
  "all_blue_percentage": 100.0
}
```

## Resharding and Global Drift Detection (v3.3)

### Overview

Hermetic Matrix v3.3 adds time-window resharding and global root drift detection for evolutionary verification across epochs.

**Key Features:**
- Time-window partitioning of history into configurable shard count (16, 64, 256, etc.)
- Global root epoch tracking across resharding operations
- Drift detection with delta reporting (Δ=<hash>)
- Automated drift reports with epoch-level analysis

### Resharding Tool

**Purpose:** Partition matrix history into time-window shards for distributed verification.

**Usage:**
```bash
python tools/hermetic/reshard_history.py --num-shards 64
```

**Process:**
1. Loads matrix_history.jsonl
2. Partitions entries into equal-sized time windows
3. Writes shard_XX.jsonl files (shard_00 through shard_63)
4. Generates shard_manifest.json with metadata

**Pass-Lines:**
```
[PASS] Resharding Complete (64 shards)
[ABSTAIN] Insufficient history for resharding
```

### Global Root Epoch Tracking

**Purpose:** Track global H_matrix root changes across epochs for drift detection.

**Append Global Root:**
```bash
python tools/hermetic/matrix_trend.py --append-global-root --num-shards 64
# Expected: [INFO] Appended global root: epoch=1 root=<sha256>...
```

**Global Root History Format:**
```json
{"timestamp": "2025-11-02T18:50:11.123456+00:00", "global_root": "198d2fca704b04c0...", "num_shards": 64, "epoch": 1}
```

**Location:** `artifacts/hermetic/global_root_history.jsonl`

### Global Drift Detection

**Purpose:** Alert when global root changes unexpectedly across epochs with dual severity scoring, drift velocity analysis, and shard history caching.

**Usage:**
```bash
python tools/hermetic/matrix_trend.py --alert-global-drift --num-shards 64 --threshold 3
# Expected (no drift): [PASS] No global drift detected in last 3 epochs
# Expected (drift): [ABSTAIN] Global Drift Detected (historical=major, snapshot=minor, velocity=62.0/epoch) Δ=<hash>
#                   [PASS] Drift Severity historical=major snapshot=minor
#                   [PASS] Shard History Cache enabled=true hits=384 misses=64
#                   Changed shards: 02, 03, 04, 05, 06, 07, 08, 09, 10, 11... (62 total)

# Disable caching (for testing or debugging):
python tools/hermetic/matrix_trend.py --alert-global-drift --num-shards 64 --threshold 3 --no-cache
# Expected: [PASS] Shard History Cache enabled=false hits=0 misses=448
```

**Severity Levels (Dual Calculation):**

*Snapshot Severity* (current state comparison):
- Compares first vs last epoch shard roots
- **minor**: <10% of shards changed
- **moderate**: 10-30% of shards changed
- **major**: ≥30% of shards changed

*Historical Severity* (peak drift across window):
- Analyzes all pairwise epoch transitions
- Reports maximum shards changed in any single epoch
- **minor**: <10% of shards changed
- **moderate**: 10-30% of shards changed
- **major**: ≥30% of shards changed

**Why Dual Severity?**
- Snapshot severity can be misleading when shard count changes
- Historical severity captures actual drift events across the window
- Example: Resharding from 2→64 shards shows snapshot=minor (0/64) but historical=major (62/64)

**Drift Velocity:**
- Measures average shards changed per epoch across the threshold window
- Computed as: `sum(changed_shards_per_epoch) / (threshold - 1)`
- Example: 0.5/epoch = 1 shard changed every 2 epochs
- Example: 2.5/epoch = 5 shards changed every 2 epochs
- Example: 62.0/epoch = 62 shards changed every epoch (major drift)

**Changed Shard IDs:**
- Lists specific shard IDs that have drifted
- Format: zero-padded 2-digit IDs (00, 01, 02, ...)
- Truncated to first 10 shards in console output
- Full list available in drift report

**Shard History Caching (v3.5):**
- Enabled by default for performance optimization
- Reduces file I/O from O(threshold × num_shards) to O(num_shards)
- Cache size: max 512 shard entries
- Cache statistics: hits/misses reported in output
- Disable with `--no-cache` flag for testing or debugging
- Typical cache hit rate: 85-90% for threshold=3

**LRU Cache Eviction (v3.6):**
- OrderedDict-based LRU cache with configurable size
- Automatic eviction of least recently used entries when cache is full
- Configurable cache size via `--cache-size` flag (default: 512)
- Tracks evictions in cache statistics
- Supports arbitrary shard counts (>512) with automatic eviction
- Pass-line format: `[PASS] Shard Cache LRU size=<S> hits=<H> misses=<M> evictions=<E>`

**Parallel Historical Severity (v3.6):**
- Optional parallel shard loading for historical severity calculation
- Uses ThreadPoolExecutor with max_workers=min(16, num_shards)
- Enabled via `--parallel` flag for 16+ shards
- Determinism guarantee: parallel and sequential produce identical results
- Typical speedup: 2-4x for 64 shards with threshold=3
- Determinism testing via `--test-determinism` flag
- Pass-line format: `[PASS] Historical Severity parallel=<true|false> identical=<true>`

**Drift Report Format (v3.5):**
```
================================================================================
GLOBAL MATRIX DRIFT REPORT
================================================================================
Drift detected across 3 epochs:
Severity (Snapshot): MINOR (0/64 shards changed)
Severity (Historical): MAJOR (max 62/64 shards/epoch)
Velocity: 62.0 shards/epoch

Changed Shard IDs:
  02, 03, 04, 05, 06, 07, 08, 09, 10, 11
  12, 13, 14, 15, 16, 17, 18, 19, 20, 21
  22, 23, 24, 25, 26, 27, 28, 29, 30, 31
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41
  42, 43, 44, 45, 46, 47, 48, 49, 50, 51
  52, 53, 54, 55, 56, 57, 58, 59, 60, 61
  62, 63

Expected root: 198d2fca704b04c0...
Actual root:   45daf2deb48f3830...

Epoch    Timestamp            Global Root      Shards  
--------------------------------------------------------------------------------
9        2025-11-02T20:07:32  198d2fca704b04c0 2       
10       2025-11-02T20:08:02  eca77db7425825a6 2       
11       2025-11-02T22:01:46  45daf2deb48f3830 64      
================================================================================
```

**Location:** `artifacts/hermetic/global_drift_report.txt`

### Resharding Workflow

**Recommended Workflow:**
1. Run CI and accumulate matrix history
2. Periodically reshard history (e.g., weekly):
   ```bash
   python tools/hermetic/reshard_history.py --num-shards 64
   ```
3. Append global root after resharding:
   ```bash
   python tools/hermetic/matrix_trend.py --append-global-root --num-shards 64
   ```
4. Check for drift before next reshard:
   ```bash
   python tools/hermetic/matrix_trend.py --alert-global-drift --num-shards 64
   ```

**Scaling Strategy:**
- Start with 16 shards for small history (<100 entries)
- Scale to 64 shards for medium history (100-1000 entries)
- Scale to 256 shards for large history (>1000 entries)

### Parallel Shard Loading

**Purpose:** Accelerate shard loading for large shard counts (≥16 shards) using ThreadPoolExecutor.

**Usage:**
```bash
python tools/hermetic/matrix_trend.py --append-global-root --num-shards 64 --parallel
# Uses 16 parallel workers for shard loading
```

**Performance:**
- Automatically enabled for ≥16 shards when `--parallel` flag is used
- Uses ThreadPoolExecutor with max_workers=16
- Speedup depends on I/O characteristics and shard file sizes
- Best for large shard files (>1000 entries each) or slow I/O

**Note:** For small shard files (<100 entries), sequential loading may be faster due to thread creation overhead.

## Local Testing

### Prerequisites

```bash
# Ensure Python 3.11+ installed
python --version

# Ensure matrix_trend.py exists
ls -la tools/hermetic/matrix_trend.py

# Ensure reshard_history.py exists
ls -la tools/hermetic/reshard_history.py
```

### Test Commands

**Validate Current State:**
```bash
python tools/hermetic/matrix_trend.py --validate
# Expected: [PASS] NO_NETWORK HERMETIC v2 TRUE
# Expected: [PASS] Hermetic Matrix <sha256>
```

**Append History:**
```bash
python tools/hermetic/matrix_trend.py --append
# Expected: [INFO] Appended history entry: <sha256>... bitmap=111111
```

**Validate History:**
```bash
python tools/hermetic/matrix_trend.py --validate-history --required-runs 3
# Expected: [PASS] Hermetic History: X/Y sealed
```

**Check for Drift:**
```bash
python tools/hermetic/matrix_trend.py --alert-drift-abstain --threshold 3
# Expected (no drift): [PASS] No drift or consecutive failures in last 3 runs
# Expected (drift): [ABSTAIN] H_matrix drift detected in last 3 runs
```

**Generate Report:**
```bash
python tools/hermetic/matrix_trend.py --save-report --limit 10
# Expected: [INFO] Report saved: artifacts/no_network/matrix_report.txt
```

### Drift Drill

Test drift detection by injecting an H_matrix change:

```bash
# Backup history
cp artifacts/no_network/matrix_history.jsonl artifacts/no_network/matrix_history.jsonl.bak

# Inject drift (change H_matrix in line 2)
sed -i '2s/08752f3466fcc8c7d69d890e80b1b7f3e8fc1003b3ec845259c640c948cada89/abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890/' artifacts/no_network/matrix_history.jsonl

# Run drift detection
python tools/hermetic/matrix_trend.py --alert-drift-abstain --threshold 3
# Expected: [ABSTAIN] H_matrix drift detected in last 3 runs
# Expected: Drift report saved: artifacts/no_network/drift_report.txt

# View drift report
cat artifacts/no_network/drift_report.txt

# Restore backup
mv artifacts/no_network/matrix_history.jsonl.bak artifacts/no_network/matrix_history.jsonl
```

## Pass-Lines

The workflow emits the following pass-lines:

1. **Hermetic Validation:**
   ```
   [PASS] NO_NETWORK HERMETIC v2 TRUE
   [PASS] Hermetic Matrix <64-hex-sha256>
   ```

2. **History Validation:**
   ```
   [PASS] Hermetic History: X/Y sealed
   ```
   - Requires X >= 3 sealed runs

3. **Drift Detection:**
   ```
   [PASS] No drift or consecutive failures in last N runs
   ```
   - Or:
   ```
   [ABSTAIN] H_matrix drift detected in last N runs
   ```

4. **Cross-Shard Verification:**
   ```
   [PASS] Global H-Matrix Verified <64-hex-sha256>
   ```
   - Or:
   ```
   [ABSTAIN] Missing shard data: shard_XX, shard_YY
   ```

5. **Resharding:**
   ```
   [PASS] Resharding Complete (64 shards)
   ```
   - Or:
   ```
   [ABSTAIN] Insufficient history for resharding
   ```

6. **Global Drift Detection (v3.5):**
   ```
   [PASS] No global drift detected in last N epochs
   ```
   - Or:
   ```
   [ABSTAIN] Global Drift Detected (historical=minor|moderate|major, snapshot=minor|moderate|major, velocity=X.X/epoch) Δ=<hash>
   [PASS] Drift Severity historical=minor|moderate|major snapshot=minor|moderate|major
   [PASS] Shard History Cache enabled=true|false hits=N misses=M
   Changed shards: 02, 03, 04, 05, 06, 07, 08, 09, 10, 11... (N total)
   ```

## Troubleshooting

### Workflow Not Triggering

**Issue:** Workflow doesn't run after CI completion

**Resolution:**
1. Check workflow file exists: `.github/workflows/hermetic-matrix-health.yml`
2. Verify workflow syntax: `gh workflow view hermetic-matrix-health`
3. Check workflow is enabled: GitHub Actions → Workflows → Hermetic Matrix Health
4. Manually trigger: `gh workflow run hermetic-matrix-health --ref integrate/ledger-v0.1`

### History Not Committing

**Issue:** History file not committed back to repository

**Resolution:**
1. Check branch: Commits only happen on `main` or `integrate/ledger-v0.1`
2. Check permissions: Workflow needs `contents: write` permission
3. Check for changes: Workflow only commits if history actually changed
4. Check logs: Look for "No changes to commit" message

### Drift False Positives

**Issue:** Drift detected when no actual drift occurred

**Resolution:**
1. Check history file: Verify H_matrix values are actually different
2. Check threshold: Adjust `--threshold` parameter (default 3)
3. Check manifest: Ensure lane_matrix_manifest.json is deterministic
4. Check RFC 8785: Verify canonical JSON serialization

### Artifacts Not Uploading

**Issue:** Artifacts not appearing in GitHub Actions

**Resolution:**
1. Check artifact name: Should be `hermetic-matrix-health-{run_number}`
2. Check file paths: Verify files exist before upload
3. Check retention: Artifacts retained for 90 days
4. Check permissions: Workflow needs `actions: read` permission

## Guardrails

### ASCII-Only Discipline

All outputs maintain ASCII-only discipline:
- No Unicode characters
- No emoji (except in GitHub Actions summary headers)
- No smart quotes or special punctuation
- Standard ASCII formatting only

### JSONL Append-Only

History file is append-only:
- Never delete entries
- Never modify existing entries
- Only append new entries
- Maintains immutable audit trail

### NO_NETWORK-Safe

Workflow is NO_NETWORK compatible:
- No external API calls
- No network dependencies
- Reads from committed artifacts/
- Writes to local artifacts/
- Fork-safe design

### Proof-or-Abstain Integrity

Follows Proof-or-Abstain discipline:
- PASS only when all lanes hermetic
- FAIL when validation fails
- ABSTAIN when drift detected
- No speculative claims

## Strategic Impact

**Differentiator:** [IVL] Integration & Validation Layer

**Acquisition Narrative:** Automated hermetic health monitoring with time-series drift detection demonstrates production-ready reliability and operational maturity.

**Measurable Outcomes:**
- Time-series matrix tracking (JSONL history)
- Drift alerts (ABSTAIN reporting)
- GitHub Actions integration (ASCII summaries)
- 90-day artifact retention
- Automatic history commits

**Doctrine Alignment:**
- Proof-or-Abstain (ABSTAIN on drift)
- RFC 8785 (canonical JSON)
- Determinism (H_matrix stability)
- ASCII-only (no Unicode)
- Fork-safe (no secrets)

## References

- **Tool:** `tools/hermetic/matrix_trend.py` (1255 lines, v3.6 with LRU eviction + parallel historical severity)
- **Resharding Tool:** `tools/hermetic/reshard_history.py` (243 lines)
- **Workflow:** `docs/ci/workflows/hermetic-matrix-health.yml` (165 lines with v3.3 steps)
- **History:** `artifacts/no_network/matrix_history.jsonl`
- **Report:** `artifacts/no_network/matrix_report.txt`
- **Drift Report:** `artifacts/no_network/drift_report.txt`
- **Global Matrix:** `artifacts/hermetic/global_matrix.json`
- **Shards:** `artifacts/hermetic/shard_XX.jsonl` (configurable count)
- **Shard Manifest:** `artifacts/hermetic/shard_manifest.json`
- **Global Root History:** `artifacts/hermetic/global_root_history.jsonl`
- **Global Drift Report:** `artifacts/hermetic/global_drift_report.txt`

---

For questions or issues, see the troubleshooting section or create a GitHub issue.
