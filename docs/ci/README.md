# CI Integration for Docs Delta & Failure Lens

This directory contains CI workflow patches and documentation for integrating the Docs Delta Watcher and Failure Lens into the GitHub Actions pipeline.

## Workflow Patch Application

Due to OAuth workflow scope limitations, the CI workflow changes cannot be pushed directly. Maintainers must apply the patch manually via the GitHub web UI.

### Steps to Apply

1. **Navigate to Workflow File**
   - Go to `.github/workflows/docs-manifest.yml` in the GitHub web UI
   - Click "Edit this file" (pencil icon)

2. **Apply Patch**
   - Open `docs/ci/docs_delta_ci.patch`
   - Copy the new `docs-delta` job section (lines 83-124)
   - Paste at the end of the workflow file after the `generate-manifest` job

3. **Verify Changes**
   - Ensure proper YAML indentation (2 spaces)
   - Check that the `needs: generate-manifest` dependency is correct
   - Verify ASCII-only content (no Unicode checkmarks)

4. **Commit and Test**
   - Commit with message: `ci: add docs delta watcher job`
   - Push to trigger CI run
   - Verify `[PASS] Docs Delta: <sha256:...>` appears in CI summary

## Workflow Features

### Docs Delta Job

The new `docs-delta` job runs after manifest generation and performs:

1. **Delta Tracking**: Computes checksums over docs assets and compares to baseline
2. **Failure Detection**: Generates failure lens enumerating missing artifacts and broken cross-links
3. **ASCII Enforcement**: Validates all docs files are ASCII-only
4. **RFC 8785 Canonicalization**: Ensures deterministic JSON serialization

### Pass-Lines

Successful runs output:
```
[PASS] Docs Delta: <sha256:d609bbb38cf53d86a9d000d0277cbc65f146a02a8f7354c8d143c9739a5ecc21>
```

Failed runs output:
```
ABSTAIN: Docs delta watcher detected failures
Total failures: 1
```

### Artifacts

The job uploads two artifacts:
- `docs_delta.json`: RFC 8785 canonicalized delta report
- `failure_lens.json`: Enumeration of failures (only on ABSTAIN)

## Local Testing

Test the docs delta watcher locally before applying the patch:

```bash
# Run delta watcher
python tools/docs/docs_delta.py --docs-dir docs/methods --rfcsign --out artifacts/docs/docs_delta.json

# Expected output
[PASS] Docs Delta: <sha256:...>

# Test failure lens with synthetic break
mv artifacts/wpv5/fol_stats.json /tmp/test_artifact
python tools/docs/docs_delta.py --docs-dir docs/methods --rfcsign --out artifacts/docs/docs_delta.json

# Expected output
FAILURE LENS: 1 issues detected
  Missing artifacts: 1
    - artifacts/wpv5/fol_stats.json (section 4.2)
ABSTAIN: Failures detected - see failure_lens.json

# Restore artifact
mv /tmp/test_artifact artifacts/wpv5/fol_stats.json
```

## ASCII Sweeper

The ASCII sweeper tool automatically converts non-ASCII characters to ASCII equivalents:

```bash
# Scan for non-ASCII
python tools/docs/ascii_sweeper.py --scan docs/

# Fix non-ASCII automatically
python tools/docs/ascii_sweeper.py --fix docs/

# Check and fail on non-ASCII (for CI)
python tools/docs/ascii_sweeper.py --check docs/ --fail-on-non-ascii
```

### ASCII Replacements

| Unicode | ASCII | Description |
|---------|-------|-------------|
| ✓ (U+2713) | [OK] | Checkmark |
| ✗ (U+2717) | [FAIL] | Cross mark |
| — (U+2014) | -- | Em dash |
| – (U+2013) | - | En dash |
| ' ' (U+2018/2019) | ' | Smart quotes |
| " " (U+201C/201D) | " | Smart quotes |
| … (U+2026) | ... | Ellipsis |

## Acceptance Criteria

- [x] Docs delta watcher with --rfcsign and --out flags
- [x] ASCII sweeper for automatic conversion
- [x] Failure lens produced on synthetic break
- [x] Artifact cross-check verification working
- [x] Pass-lines output correctly
- [x] CI workflow patch provided for maintainer application

## Baseline Persistence

The docs delta watcher supports baseline persistence to track documentation changes across CI runs. Two implementation options are available:

### Option A: Git-Based Baseline (Recommended)

**How it works**: Commits baseline file to main branch after successful CI run.

**Pros**:
- Simple implementation
- No external dependencies
- Baseline versioned with code
- Works across all branches

**Cons**:
- Bloats git history with generated files
- Potential merge conflicts on parallel PRs
- Requires write access to main branch

**Implementation**: See `docs/ci/baseline_persistence.patch`

**Workflow excerpt**:
```yaml
- name: Commit baseline on main-merge
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  run: |
    git add artifacts/docs/docs_delta_baseline.json
    git commit -m "ci: update docs delta baseline [skip ci]"
    git push origin main
```

### Option B: Artifact-Based Baseline

**How it works**: Downloads baseline from previous CI run using GitHub Actions artifacts.

**Pros**:
- No git bloat
- No merge conflicts
- Clean separation of concerns
- Independent per-branch baselines

**Cons**:
- Artifacts expire after 90 days
- More complex workflow logic
- Requires fallback for first run
- Cross-branch tracking difficult

**Implementation**: See `docs/ci/artifact_baseline.patch`

**Workflow excerpt**:
```yaml
- name: Download previous baseline
  uses: actions/download-artifact@v4
  with:
    name: docs-delta-baseline
    path: artifacts/docs/
  continue-on-error: true

- name: Upload baseline for next run
  uses: actions/upload-artifact@v4
  with:
    name: docs-delta-baseline
    path: artifacts/docs/docs_delta_baseline.json
    retention-days: 90
```

### Comparison Table

| Feature | Option A (Git) | Option B (Artifact) |
|---------|----------------|---------------------|
| Git bloat | Yes | No |
| Merge conflicts | Possible | No |
| Expiration | Never | 90 days |
| Cross-branch | Shared | Independent |
| Complexity | Low | Medium |
| First-run handling | Automatic | Requires fallback |
| Recommended for | Small teams, simple workflows | Large teams, complex branching |

### Edge Case Handling

The docs delta watcher gracefully handles baseline errors:

**Corrupt baseline (invalid JSON)**:
```
ABSTAIN: Baseline is corrupt (invalid JSON): Expecting property name
Remediation: Delete baseline file and regenerate with --write-baseline
Continuing without baseline (all files will show as 'added')
```

**Wrong format version**:
```
ABSTAIN: Baseline format version mismatch (expected 1.0, got 2.0)
Remediation: Regenerate baseline with current version using --write-baseline
Continuing without baseline (all files will show as 'added')
```

**Missing checksums key**:
```
ABSTAIN: Baseline missing 'checksums' key
Remediation: Regenerate baseline with --write-baseline
Continuing without baseline (all files will show as 'added')
```

**Permission errors**:
```
ABSTAIN: Permission denied reading baseline file
Remediation: Check file permissions (chmod 644) or run with appropriate user
Continuing without baseline (all files will show as 'added')
```

All edge cases follow Proof-or-Abstain discipline: ABSTAIN with remediation text, then continue without baseline rather than failing hard.

## Fleet Directive Compliance

- **Proof-or-Abstain**: ABSTAIN on failures with explicit enumeration
- **RFC 8785**: Canonical JSON for deterministic hashing
- **Determinism > speed**: Byte-identical delta reports across runs
- **ASCII-only**: All docs and scripts enforce ASCII-only discipline
- **Sealed pass-lines**: SHA-256 hash in every pass-line

## Troubleshooting

### Non-ASCII Characters in Workflow

If the workflow file contains non-ASCII characters (e.g., ✓ or ✗), run:

```bash
python tools/docs/ascii_sweeper.py --fix .github/workflows/docs-manifest.yml
```

### Missing Artifacts

If the failure lens detects missing artifacts:

1. Check `artifacts/docs/failure_lens.json` for details
2. Verify artifact paths in `docs/methods/cross_ledger_index.json`
3. Ensure all referenced artifacts exist in `artifacts/`

### Delta Hash Instability

If delta hashes change across runs:

1. Verify RFC 8785 canonicalization is working
2. Check for timestamp or random data in delta report
3. Ensure all inputs are deterministic

---

For questions or issues, see the main documentation or create a GitHub issue.
