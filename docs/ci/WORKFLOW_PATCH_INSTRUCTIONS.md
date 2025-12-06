# CI Workflow Optimization Patch Instructions

## Overview

Due to GitHub OAuth workflow scope limitations, the CI workflow optimizations must be applied manually via patch file. This document provides step-by-step instructions for applying the patch.

## Patch Contents

The patch file `docs/ci/patches/ci_workflows.patch` contains optimizations for:
- `.github/workflows/ci.yml`
- `.github/workflows/dual-attestation.yml`
- `.github/workflows/performance-check.yml`
- `.github/workflows/performance-sanity.yml`

## Optimizations Included

1. **UV Dependency Caching** (20-30s savings per job)
2. **Consolidated Test Execution** (60s savings)
3. **ASCII-Only Compliance** (removes emoji and Unicode)
4. **Deterministic Outputs** (consistent Merkle roots)

## Application Methods

### Method 1: Git Apply (Recommended)

```bash
# From repository root
git apply docs/ci/patches/ci_workflows.patch

# Verify changes
git diff

# Commit if satisfied
git add .github/workflows/
git commit -m "ci: apply workflow optimization patch [RC]"
```

### Method 2: Manual Application via GitHub Web UI

For each workflow file:

1. Navigate to the file in GitHub web interface
2. Click "Edit this file" (pencil icon)
3. Apply changes from patch file manually
4. Commit directly to branch

### Method 3: Patch Command

```bash
# From repository root
patch -p1 < docs/ci/patches/ci_workflows.patch

# Verify
git status

# Commit
git add .github/workflows/
git commit -m "ci: apply workflow optimization patch [RC]"
```

## Verification Steps

After applying the patch:

### 1. Verify ASCII Compliance

```bash
python3 << 'PYEOF'
import os
for root, dirs, files in os.walk('.github/workflows'):
    for file in files:
        if file.endswith('.yml'):
            filepath = os.path.join(root, file)
            with open(filepath, 'rb') as f:
                content = f.read()
                try:
                    content.decode('ascii')
                    print(f"PASS: {filepath}")
                except UnicodeDecodeError:
                    print(f"FAIL: {filepath}")
PYEOF
```

Expected output: All files should show "PASS"

### 2. Verify Syntax

```bash
# Install yamllint if not present
pip install yamllint

# Validate all workflow files
yamllint .github/workflows/*.yml
```

### 3. Test Locally (Optional)

```bash
# Install act for local GitHub Actions testing
# https://github.com/nektos/act

# Test ci.yml workflow
act -W .github/workflows/ci.yml --dry-run

# Test dual-attestation.yml workflow
act -W .github/workflows/dual-attestation.yml --dry-run
```

## Expected Performance Improvements

After applying the patch and running CI:

| Workflow | Before | After | Improvement |
|----------|--------|-------|-------------|
| ci.yml (test) | ~180s | ~120s | 33% |
| ci.yml (uplift-omega) | ~90s | ~60s | 33% |
| dual-attestation.yml | ~150s | ~100s | 33% |
| **Total per PR** | **420s** | **280s** | **33%** |

## Troubleshooting

### Patch Fails to Apply

If `git apply` fails:

```bash
# Check what conflicts exist
git apply --check docs/ci/patches/ci_workflows.patch

# Apply with 3-way merge
git apply --3way docs/ci/patches/ci_workflows.patch

# Or apply with reject files
git apply --reject docs/ci/patches/ci_workflows.patch
```

### Workflow Syntax Errors

If GitHub Actions reports syntax errors:

1. Verify YAML indentation (use spaces, not tabs)
2. Check for missing closing braces in `${{ }}` expressions
3. Validate with yamllint

### Cache Not Working

If UV cache doesn't improve performance:

1. Verify cache key matches: `uv-${{ runner.os }}-${{ hashFiles('**/pyproject.toml') }}`
2. Check cache hit/miss in workflow logs
3. Ensure `~/.cache/uv` path is correct for runner OS

## Rollback Procedure

If issues occur after applying patch:

```bash
# Revert workflow changes
git checkout HEAD~1 -- .github/workflows/

# Or restore from specific commit
git checkout <commit-hash> -- .github/workflows/

# Commit rollback
git commit -m "ci: rollback workflow optimization patch"
```

## Support

For questions or issues:
1. Review `docs/ci/CI_OPTIMIZATION_REPORT.md` for detailed analysis
2. Check `docs/ci/CI_INTEGRITY_VERIFICATION.md` for verification procedures
3. Contact repository maintainers

## Patch Metadata

- **Created**: 2025-10-19
- **Author**: Devin A - Pipeline Integrator
- **Version**: 1.0
- **Target Branch**: integrate/ledger-v0.1
- **Verification Status**: [PASS] CI INTEGRITY: VERIFIED
