# CI Unblock Instructions

## Immediate Action Required

The test job in CI is failing due to a setuptools editable install error. This requires a one-line edit to `.github/workflows/ci.yml`.

## Root Cause

Lines 36-37 in `.github/workflows/ci.yml` attempt an editable install that fails with:
```
error: Multiple top-level packages discovered in a flat-layout: ['ui', 'tmp', 'logs', 'apps', 'infra', 'config', 'backend', 'metrics', 'services', 'templates', 'artifacts', 'migrations', 'ci_verification', 'allblue_archive'].
```

This is inherent to the repository's flat package layout. The editable install is unnecessary because line 28 already runs `uv sync`, which correctly handles all dependencies.

## Fix (Web UI - Recommended)

1. Navigate to: https://github.com/helpfuldolphin/mathledger/blob/integrate/ledger-v0.1/.github/workflows/ci.yml
2. Click "Edit this file" (pencil icon)
3. Find lines 36-37 (immediately after the "Unit tests (network-free)" step):
   ```yaml
   - run: uv pip install --system -e .
   - run: uv sync
   ```
4. Delete both lines
5. Commit directly to `integrate/ledger-v0.1` with message:
   ```
   [ME] ci: remove failing uv editable install step
   
   - Remove 'uv pip install --system -e .' which fails with setuptools flat-layout error
   - Remove redundant 'uv sync' (already at line 28)
   - Fork-safe fix for test job CI failure
   ```

## Fix (CLI - Requires Workflow Scope)

```bash
git checkout integrate/ledger-v0.1
git pull

# Edit .github/workflows/ci.yml manually:
# Delete lines 36-37 (the two 'run:' lines after "Unit tests (network-free)")

git add .github/workflows/ci.yml
git commit -m "[ME] ci: remove failing uv editable install step"
git push origin integrate/ledger-v0.1
```

## Verification

After applying the fix:

1. Re-run CI on any PR targeting `integrate/ledger-v0.1`
2. The test job should now:
   - Pass unit tests (network-free)
   - Run migrations (first pass)
   - Test migration idempotency (second pass)
   - Run full pytest suite
   - Pass coverage enforcement (70% floor)
   - Pass metrics V1 linter
   - Pass performance regression gate

## Expected Outcome

All CI jobs should go green:
- ✅ test
- ✅ composite-da (already fixed with v4 artifacts)
- ✅ dual-attestation
- ✅ browsermcp
- ✅ reasoning
- ✅ uplift-omega
- ✅ velocity-report

## Related Documentation

- Full application guide: `docs/ci/PERF_GATE_APPLICATION.md`
- Perf Gate v2 tool: `tools/perf/perf_gate.py`
- PR with all changes: https://github.com/helpfuldolphin/mathledger/pull/77

## Contact

For questions about this fix, see PR #77 or the Perf Gate v2 implementation.
