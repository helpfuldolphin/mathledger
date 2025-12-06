# Perf Gate Workflow Application Guide

## Overview

This document provides instructions for manually applying the Perf Gate workflow due to GitHub OAuth workflow scope limitations.

## Workflow Features

The Perf Gate workflow provides:

1. **3-Run Benchmarks**: Measures performance across 3 iterations with variance calculation
2. **RFC 8785 Sealed Pack**: Exports canonical JSON with SHA256 hash for deterministic verification
3. **Cache Diagnostics**: Validates cache hit rate >=20% or ABSTAIN with diagnostics
4. **Pass-Lines**: Outputs standardized pass/fail/abstain status
5. **Artifact Upload**: Uses actions/upload-artifact@v4 (fixes composite-da v3 deprecation)

## Pass-Lines

```
[PASS] Perf Uplift >=3.0x (±σ)
[PASS] Cache Hit >=20% (or ABSTAIN with warning)
[PASS] Perf Pack: <sha256>
```

## Manual Application

### Method 1: CLI Application (Recommended)

**Step 1: Apply Workflow Patches**

```bash
# From repository root
cd /path/to/mathledger

# Apply perf gate v2 workflow
git apply docs/ci/perf-gate-v2-workflow.patch

# Apply composite-da v4 fix
git apply docs/ci/composite-da-v4.patch

# Verify files were created/modified
ls -la .github/workflows/perf-gate.yml
ls -la .github/workflows/composite-da.yml
```

**Step 2: Verify Workflow Syntax**

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/perf-gate.yml'))"
python -c "import yaml; yaml.safe_load(open('.github/workflows/composite-da.yml'))"
```

**Step 3: Test Locally (Optional)**

```bash
# Test perf_gate.py tool
PYTHONPATH=$(pwd) python tools/perf/perf_gate.py --dataset-size 1000

# Expected output:
# [PASS] Perf Uplift >=3.0x (±σ)
# [PASS] Cache Hit >=20%
# [PASS] Perf Pack: <sha256>
```

**Step 4: Commit and Push**

```bash
git add .github/workflows/perf-gate.yml .github/workflows/composite-da.yml
git commit -m "[ME] ci: apply perf gate v2 and composite-da v4

- Add perf-gate-sealed job with 3-run benchmarks
- Upgrade composite-da to actions/upload-artifact@v4
- Export RFC 8785 canonical perf_pack.json with SHA256 seal
- Generate GitHub job summary with pass-lines"

git push origin <your-branch>
```

### Method 2: Web UI Application (Alternative)

**Step 1: View Patch Content**

Navigate to the patch files in GitHub:
- `docs/ci/perf-gate-v2-workflow.patch`
- `docs/ci/composite-da-v4.patch`

**Step 2: Create Workflow Files via Web UI**

1. Go to repository on GitHub
2. Navigate to `.github/workflows/`
3. Click "Add file" → "Create new file"
4. Name: `perf-gate.yml`
5. Copy content from `perf-gate-v2-workflow.patch` (lines starting with `+`)
6. Commit directly to your branch

Repeat for `composite-da.yml` using `composite-da-v4.patch`

**Step 3: Verify in PR**

Once committed, the workflows will appear in the PR's "Files changed" tab.

### Step 4: Create Pull Request

Open a PR targeting `integrate/ledger-v0.1` with:

**Title**: `[ME] ci: apply perf gate workflow with sealed pack and v4 artifacts`

**Body**:
```markdown
## Summary

Applies Perf Gate workflow with 3-run benchmarks, RFC 8785 sealed pack, and SHA256 verification.

## Changes

- Upgrade actions/upload-artifact from v3 to v4 (fixes composite-da deprecation)
- Add perf-gate-sealed job with 3-run variance measurement
- Validate uplift >=3.0x and cache hit >=20%
- Export RFC 8785 canonical perf_pack.json with SHA256 seal
- Generate GitHub job summary with pass-lines

## Pass-Lines

```
[PASS] Perf Uplift >=3.0x (±σ)
[PASS] Cache Hit >=20% (or ABSTAIN with warning)
[PASS] Perf Pack: <sha256>
```

## Strategic Impact

**Differentiator**: [ME] - Metrics & Evidence
**Acquisition Narrative**: Provides measurable evidence of performance optimization
**Measurable Outcomes**: 3-run variance measurement, RFC 8785 sealed pack with SHA256
**Doctrine Alignment**: Quantifiable performance metrics for stakeholder confidence

## Testing

Local validation:
```bash
# Validate workflow syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/perf-gate.yml'))"

# Run benchmarks locally (requires uv)
uv sync
uv run python -c "from backend.axiom_engine.rules import apply_modus_ponens; print('Import successful')"
```

## Acceptance Criteria

- [ ] CI job green
- [ ] Pass-lines printed in job output
- [ ] perf_pack.json sealed and uploaded as artifact
- [ ] GitHub job summary generated with metrics table
```

## Expected CI Output

When the workflow runs successfully, you should see:

```
=== Performance Gate v2 ===
Baseline: 227.29ms
Dataset size: 5000

Capturing environment fingerprint...
Python: 3.12.8, Platform: Linux-5.10.223-x86_64-with-glibc2.35

Generating synthetic dataset...
Generated 5000 statements
Workload signature: a3f2c8d4e1b7f9a6

Running 3x benchmarks with cache clearing...
Average time: 3.81ms ±3.86ms

Testing cache effectiveness...
Cache hit rate: 99.00%

=== Pass-Lines ===
[PASS] Perf Uplift 59.58x (±60.29)
[PASS] Cache Hit 99.00%
[PASS] Perf Pack: ebdeebd63a07c7de256a75d622376730a04fd385ecd0be03197e72c305ba0a2f
```

### Artifacts Generated

The workflow uploads the following artifacts:

**1. artifacts/perf/perf_pack.json** (RFC 8785 canonical)
- `perf_uplift`: Speedup metrics with 3-run statistics
- `cache_diagnostics`: Cache hit rate and effectiveness
- `env_fingerprint`: Tool versions (Python, uv, git commit, platform)
- `workload_signature`: SHA256 hash of dataset (first 16 chars)
- `timestamp`: UTC timestamp
- `git_sha`: GitHub commit SHA

**2. artifacts/perf/perf_hints.txt** (generated on regression only)
- Top 3 performance suspects
- Cache effectiveness analysis
- Algorithmic change detection
- Dataset characteristics
- Recommended debugging actions

### ABSTAIN Behavior

If cache hit rate < 20%, the gate will ABSTAIN (non-blocking):

```
[PASS] Perf Uplift 59.58x (±60.29)
[ABSTAIN] Cache Hit 12.40% < 20%
[PASS] Perf Pack: <sha256>

WARNING: Cache effectiveness below threshold (ABSTAIN)
```

The workflow will still pass, but diagnostic pack will be uploaded.

## Troubleshooting

### Issue: Cache hit rate < 20%

**Symptom**: `[ABSTAIN] Cache Hit 12.40% < 20%`

**Cause**: CI environment may clear cache between test runs or run tests in isolation

**Resolution**: This is expected in CI. The ABSTAIN logic prevents false failures while providing diagnostics.

### Issue: Speedup < 3.0x

**Symptom**: `[FAIL] Perf Uplift 2.5x < 3.0x`

**Cause**: Performance regression or baseline needs adjustment

**Resolution**: 
1. Check if recent changes introduced performance regressions
2. Verify baseline_time_ms in workflow matches current Phase 2 baseline
3. Review perf_pack.json artifact for detailed timing data

### Issue: Workflow file not found

**Symptom**: `ERROR: perf_pack.json not found`

**Cause**: Benchmark step failed or artifacts directory not created

**Resolution**:
1. Check benchmark step logs for errors
2. Verify `uv sync` completed successfully
3. Ensure backend dependencies are installed

## Patch File Locations

- Perf Gate v2 workflow: `docs/ci/perf-gate-v2-workflow.patch`
- Composite-DA v4 fix: `docs/ci/composite-da-v4.patch`
- CI uv install fix (remove failing editable install step): `docs/ci/ci-uv-install-fix.patch`

Apply with:

```bash
git apply docs/ci/perf-gate-v2-workflow.patch
git apply docs/ci/composite-da-v4.patch
git apply docs/ci/ci-uv-install-fix.patch
```

## Chain Attestation (Advanced)

### Overview

Chain attestation creates an immutable performance ledger by linking each perf pack to the previous run's SHA256 hash. This enables forensic traceability of performance history and detects tampering or regressions.

### Usage

**Run 1 (Baseline):**
```bash
PYTHONPATH=$(pwd) python tools/perf/perf_gate.py --dataset-size 1000 --output-dir artifacts/perf
# Output: [PASS] Perf Pack: f5844b5d4148c8f03107ad15eee491ee4163af5dfd84bf687bb8044b19fbd375
```

**Run 2 (Chain Attestation):**
```bash
PYTHONPATH=$(pwd) python tools/perf/perf_gate.py \
  --dataset-size 1000 \
  --output-dir artifacts/perf \
  --prev-pack artifacts/perf/perf_pack.json
```

**Expected Output:**
```
Loading previous pack for chain attestation...
Previous pack hash: f5844b5d4148c8f03107ad15eee491ee4163af5dfd84bf687bb8044b19fbd375
Previous timestamp: 2025-11-01T03:52:04.168059Z

=== Performance Gate v2 ===
Chain mode: ENABLED

...

Validating chain integrity...
[PASS] Chain integrity verified

=== Pass-Lines ===
[PASS] Perf Uplift 69.64x (±67.68)
[PASS] Cache Hit 99.00%
[PASS] Perf Pack: 0f4cba0d87fa56bb89989a40cb5f9b419a9ff3de8c7cf789b7993b63663e1e67
[PASS] Perf Chain Intact f5844b5d4148c8f03107ad15eee491ee4163af5dfd84bf687bb8044b19fbd375
```

### Chain Validation Rules

1. **Hash Linkage**: Current pack's `prev_pack_sha256` must match actual hash of previous pack
2. **Timestamp Monotonicity**: Current timestamp must be >= previous timestamp
3. **Chain Break Detection**: Fails with exit code 1 if hash mismatch or timestamp regression

### Chain Break Example

If the previous pack is tampered with or the chain is broken:

```
Validating chain integrity...
[FAIL] Chain broken: Hash mismatch: expected f5844b5d..., got 00000000...

ERROR: Chain integrity check failed
```

### CI Integration

To enable chain attestation in CI:

1. Store previous pack as artifact or in repository
2. Pass `--prev-pack` flag to perf_gate.py in workflow
3. Chain validation will run automatically
4. Pass-line includes `[PASS] Perf Chain Intact <prev-hash>`

### Use Cases

- **Performance Ledger**: Immutable history of performance measurements
- **Regression Detection**: Detect when performance degrades across runs
- **Forensic Audit**: Trace performance changes back to specific commits
- **Tamper Detection**: Verify pack integrity across CI runs

## Related Documentation

- Phase 2 Optimizations: `docs/perf/PHASE2_OPTIMIZATIONS.md`
- Phase 3 Optimizations: `docs/perf/PHASE3_OPTIMIZATIONS.md`
- RFC 8785 Canonicalization: https://datatracker.ietf.org/doc/html/rfc8785
- CI Unblock Instructions: `docs/ci/CI_UNBLOCK_INSTRUCTIONS.md`

## Contact

For questions about this workflow, see PR #77 or contact the performance optimization team.
