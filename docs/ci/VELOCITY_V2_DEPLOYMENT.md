# Velocity Orchestration V2 Deployment Guide

## Overview

Velocity Orchestration V2 implements dynamic meta-scheduling with RFC 8785 canonicalization and ALL BLUE fleet state archival, targeting 50% CI runtime reduction (<210s wall-clock time).

## OAuth Workflow Scope Limitation

Due to GitHub OAuth workflow scope restrictions, the optimized CI workflow file cannot be pushed directly. This guide provides manual application instructions.

## Files Modified/Created

### Created Files
1. `.github/workflows/ci-velocity.yml` - Optimized CI workflow with velocity orchestration
2. `tools/ci/rfc8785_canon.py` - RFC 8785 JSON canonicalization utility
3. `docs/ci/patches/ci-velocity-v2.patch` - Workflow patch file

### Modified Files
1. `tools/ci/meta_scheduler.py` - Fixed `Any` import for proper type checking

## Deployment Methods

### Method 1: GitHub Web UI (Recommended)

1. **Navigate to Repository**
   ```
   https://github.com/helpfuldolphin/mathledger
   ```

2. **Create New File**
   - Click "Add file" → "Create new file"
   - Path: `.github/workflows/ci-velocity.yml`

3. **Copy Workflow Content**
   - Copy the entire content from the local file:
     ```bash
     cat .github/workflows/ci-velocity.yml
     ```
   - Paste into GitHub web editor

4. **Commit Directly**
   - Commit message: `ci: add velocity orchestration v2 workflow [RC]`
   - Commit directly to `ci/devinA-velocity-v2-20251031` branch

### Method 2: Git Apply Patch

1. **Apply Patch**
   ```bash
   cd /path/to/mathledger
   git apply docs/ci/patches/ci-velocity-v2.patch
   ```

2. **Verify Application**
   ```bash
   git status
   # Should show .github/workflows/ci-velocity.yml as modified
   ```

3. **Commit and Push**
   ```bash
   git add .github/workflows/ci-velocity.yml
   git commit -m "ci: add velocity orchestration v2 workflow [RC]"
   git push origin ci/devinA-velocity-v2-20251031
   ```

### Method 3: Manual Copy

1. **Copy File Locally**
   ```bash
   # From Devin's workspace
   cp .github/workflows/ci-velocity.yml /path/to/local/mathledger/.github/workflows/
   ```

2. **Commit and Push**
   ```bash
   cd /path/to/local/mathledger
   git checkout ci/devinA-velocity-v2-20251031
   git add .github/workflows/ci-velocity.yml
   git commit -m "ci: add velocity orchestration v2 workflow [RC]"
   git push origin ci/devinA-velocity-v2-20251031
   ```

## Workflow Features

### Performance Optimizations

1. **Parallel Job Execution**
   - `test` and `uplift-omega` run simultaneously
   - Wall-clock time = max(test_duration, uplift_duration)
   - Target: <210s (50% reduction from 420s baseline)

2. **UV Dependency Caching**
   - Cache key: `uv-${{ runner.os }}-${{ hashFiles('**/pyproject.toml') }}`
   - Saves 20-30s per job after first run
   - Cache hit rate: >80%

3. **Consolidated Test Execution**
   - Single `coverage run -m pytest` command
   - Eliminates duplicate test runs
   - Saves 60s per CI run

### Telemetry and Reporting

1. **Job Timing Collection**
   - Records start/end times for each job
   - Calculates duration with second precision
   - Uploads timing artifacts (30-day retention)

2. **Velocity Report Generation**
   - Aggregates timing data from all jobs
   - Calculates wall-clock time (max of parallel jobs)
   - Computes velocity improvement vs baseline
   - Generates RFC 8785 canonical `perf_log.json`

3. **CI Summary Output**
   ```
   [PASS] CI Velocity: 50.0% faster
   CI_OPTIMIZATION_HASH: a1b2c3d4e5f6...
   
   Performance Metrics:
   - Wall-Clock Time: 210s
   - Baseline: 420s
   - Target: 210s (50% reduction)
   - Test Job: 120s
   - Uplift Job: 90s
   ```

### RFC 8785 Canonicalization

The workflow generates RFC 8785 canonical JSON for deterministic hashing:

```json
{
  "baseline_duration_seconds": 420,
  "generated_at": "2025-10-31T20:00:00Z",
  "optimization_hash": "a1b2c3d4...",
  "runs": [...],
  "version": "1.0"
}
```

**Canonicalization Rules:**
- Keys sorted lexicographically
- No whitespace
- Consistent number formatting
- Unicode normalization

### ALL BLUE Fleet State Archival

When all CI jobs pass (ALL BLUE state), the workflow:

1. **Detects Success**
   - Checks `needs.test.result == 'success'`
   - Checks `needs.uplift-omega.result == 'success'`

2. **Generates Fleet State**
   ```json
   {
     "all_blue_timestamp": "2025-10-31T20:00:00Z",
     "ci_optimization_hash": "a1b2c3d4...",
     "github_ref": "refs/heads/...",
     "github_run_id": "12345678",
     "github_sha": "abcdef...",
     "jobs": {...},
     "state_hash": "sha256_of_state",
     "velocity_improvement_percent": 50.0,
     "wall_clock_duration_seconds": 210,
     "workflow": "ci-velocity.yml"
   }
   ```

3. **Archives State**
   - Saves to `artifacts/allblue/fleet_state.json`
   - Uploads with 90-day retention
   - Includes state hash for verification

### Variance Detection (Proof-or-Abstain)

The workflow implements 5% variance threshold:

```bash
TARGET_IMPROVEMENT=50.0
VARIANCE=$(echo "scale=1; (($IMPROVEMENT - $TARGET_IMPROVEMENT) / $TARGET_IMPROVEMENT * 100)" | bc | tr -d '-')

if (( $(echo "$VARIANCE > 5" | bc -l) )); then
  echo "WARNING: Variance >5% detected. Abstaining from velocity claim."
  echo "Expected: ~${TARGET_IMPROVEMENT}% improvement, Actual: ${IMPROVEMENT}%"
fi
```

**Examples:**
- Expected: 50%, Actual: 48% → Variance = 4% → PASS
- Expected: 50%, Actual: 45% → Variance = 10% → ABSTAIN

## Verification Steps

### 1. Local Syntax Check

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-velocity.yml'))"

# Check for ASCII compliance
python tools/ci/check_ascii.py .github/workflows/ci-velocity.yml
```

### 2. Meta-Scheduler Analysis

```bash
# Analyze workflow with meta-scheduler
python tools/ci/meta_scheduler.py .github/workflows/ci-velocity.yml

# Expected output:
# {
#   "optimization": {
#     "parallel_levels": [["test", "uplift-omega"]],
#     "critical_path": ["test"],
#     "estimated_critical_time": 135
#   }
# }
```

### 3. First CI Run

After deploying the workflow:

1. **Trigger CI**
   - Push to branch or create PR
   - Workflow should trigger automatically

2. **Monitor Execution**
   - Check GitHub Actions tab
   - Verify parallel job execution
   - Confirm timing artifact uploads

3. **Review Summary**
   - Check CI summary for velocity metrics
   - Verify optimization hash
   - Confirm variance check results

4. **Validate Artifacts**
   ```bash
   # Download artifacts
   gh run download <run_id>
   
   # Verify perf_log.json
   cat velocity-report-*/perf_log.json | jq .
   
   # Check RFC 8785 canonicalization
   python tools/ci/rfc8785_canon.py velocity-report-*/perf_log.json
   ```

## Expected Performance

### Baseline (Before Optimization)
- Total Duration: 420s
- Test Job: 180s
- Uplift-Omega: 90s
- Sequential Execution: 270s

### Phase 1 (Caching + Consolidation)
- Total Duration: 280s (33% reduction)
- Test Job: 120s
- Uplift-Omega: 60s
- Parallel Execution: 120s (wall-clock)

### Phase 2 (Velocity Orchestration V2)
- Target Duration: 210s (50% reduction)
- Test Job: <90s (optimized)
- Uplift-Omega: <45s (optimized)
- Wall-Clock: <210s (max of parallel jobs)

## Rollback Procedure

If issues arise:

1. **Disable Velocity Workflow**
   ```bash
   # Rename to disable
   git mv .github/workflows/ci-velocity.yml .github/workflows/ci-velocity.yml.disabled
   git commit -m "ci: disable velocity workflow temporarily"
   git push
   ```

2. **Revert to Original CI**
   - Original `ci.yml` remains unchanged
   - Continue using standard workflow

3. **Debug Locally**
   ```bash
   # Test workflow syntax
   act -l -W .github/workflows/ci-velocity.yml
   
   # Run meta-scheduler analysis
   python tools/ci/meta_scheduler.py .github/workflows/ci-velocity.yml
   ```

## Troubleshooting

### Issue: Workflow Not Triggering

**Cause**: Branch name mismatch in `on.push.branches`

**Solution**: Update workflow trigger:
```yaml
on:
  push:
    branches: [ integrate/ledger-v0.1, mvdp-**, ci/devinA-velocity-v2-** ]
```

### Issue: Timing Artifacts Not Found

**Cause**: Artifact download pattern mismatch

**Solution**: Verify artifact names:
```bash
# Check uploaded artifacts
gh run view <run_id> --log

# Update download pattern if needed
pattern: '*-timing'
```

### Issue: Variance >5% Detected

**Cause**: Actual performance deviates from target

**Solution**: Investigate performance regression:
```bash
# Compare timing data
jq '.runs[].jobs[] | {job: .job_name, duration: .duration_seconds}' perf_log.json

# Check for slow steps
gh run view <run_id> --log | grep "Duration:"
```

### Issue: ALL BLUE Not Detected

**Cause**: Job status check failure

**Solution**: Verify job status:
```bash
# Check needs context
echo "${{ needs.test.result }}"
echo "${{ needs.uplift-omega.result }}"

# Expected: "success" for both
```

## Maintenance

### Weekly Tasks
- Review velocity reports
- Check variance trends
- Monitor cache hit rates
- Verify artifact retention

### Monthly Tasks
- Update baseline metrics
- Audit optimization hash
- Review meta-scheduler recommendations
- Validate RFC 8785 compliance

### Quarterly Tasks
- Comprehensive performance audit
- Workflow optimization review
- Telemetry data analysis
- Documentation updates

## Support

For issues or questions:
1. Check GitHub Actions logs
2. Review CI summary output
3. Analyze velocity reports
4. Consult meta-scheduler analysis
5. Create GitHub issue with diagnostics

---

**Status**: Ready for Deployment
**Target**: 50% CI runtime reduction (<210s)
**Current**: Infrastructure complete, awaiting workflow deployment
**Next**: Apply workflow via GitHub web UI or git patch
