# Frozen Version Immutability

## Overview

The frozen version system ensures that published version directories are **immutable-by-construction**. Once a version is frozen, its `site/v{X}/` directory will never be regenerated from templates, protecting against template drift and ensuring byte-identical deployments.

## Key Guarantee

> **Building from a later tag must NOT change any existing `site/v0*` directory bytes** (except `/versions/` and `/versions/status.json`).

## How It Works

### Freeze Manifests

Each frozen version has a manifest stored in `releases/frozen/{version}.json`:

```json
{
  "version": "v0.2.6",
  "frozen_at": "2026-01-04T05:28:42Z",
  "frozen_by_commit": "abc123...",
  "content_hash": "1fb3365a3d76521c...",
  "file_count": 46,
  "files": {
    "index.html": "a1b2c3...",
    "manifest.json": "d4e5f6...",
    ...
  }
}
```

### Build Behavior

1. **Frozen version exists & directory exists**: Skip regeneration, verify hashes match
2. **Frozen version exists & directory missing**: Rebuild and verify against freeze manifest
3. **Not frozen**: Build normally

### Immutability Violation Detection

If a frozen version's files don't match the freeze manifest:
- `--check-immutability` fails with exit code 1
- Building that version raises `BuildError`
- The violation must be fixed before deployment

## CLI Commands

### Freeze a Version
```bash
uv run python scripts/build_static_site.py --freeze v0.2.6
```

### Freeze All Built Versions
```bash
uv run python scripts/build_static_site.py --freeze-all
```

### Check Immutability
```bash
uv run python scripts/build_static_site.py --check-immutability
```

### Force Rebuild Frozen Version
```bash
uv run python scripts/build_static_site.py --version v0.2.6 --no-skip-frozen
```

## Smoke Checklist

### After Any Build

- [ ] Run `--check-immutability` - should pass with "All N frozen versions are immutable"
- [ ] No "IMMUTABILITY VIOLATED" messages
- [ ] Frozen versions show "verified X files (skipping rebuild)"

### After Template Changes

- [ ] `--check-immutability` still passes (frozen versions unchanged)
- [ ] New versions use updated templates
- [ ] Old frozen versions preserve original templates

### Before Deployment

```bash
# 1. Verify all frozen versions are intact
uv run python scripts/build_static_site.py --check-immutability

# 2. Build (frozen versions skipped automatically)
uv run python scripts/build_static_site.py --all

# 3. Run regression tests
uv run pytest tests/governance/test_frozen_version_immutability.py -v

# 4. Deploy
wrangler pages deploy ./site --project-name mathledger-ai
```

### CI/CD Integration

Add to CI pipeline:
```yaml
- name: Check frozen version immutability
  run: uv run python scripts/build_static_site.py --check-immutability

- name: Run immutability regression tests
  run: uv run pytest tests/governance/test_frozen_version_immutability.py -v
```

## Regression Tests

The test file `tests/governance/test_frozen_version_immutability.py` verifies:

| Test | Description |
|------|-------------|
| `test_freeze_creates_manifest` | Freezing creates valid manifest |
| `test_frozen_manifest_structure` | Manifests have correct structure |
| `test_build_twice_produces_identical_output` | Same input -> same output |
| `test_immutability_check_passes` | `--check-immutability` succeeds |
| `test_frozen_version_skips_rebuild` | Frozen versions skip regeneration |
| `test_tampering_detected` | File modifications are detected |
| `test_frozen_versions_unchanged_after_all_build` | `--all` preserves frozen |
| `test_freeze_all_freezes_built_versions` | `--freeze-all` works |

## Troubleshooting

### "IMMUTABILITY VIOLATED" Error

A frozen version's files don't match the freeze manifest. Options:

1. **Restore from git**: `git checkout site/{version}/`
2. **Update freeze manifest**: `--freeze {version}` (only if intentional)
3. **Delete and rebuild**: Remove `site/{version}/` and run `--all`

### "FREEZE MISMATCH" Error

Rebuilt output doesn't match frozen state. This means templates or source files changed. Options:

1. **Restore old templates**: Check out templates from freeze commit
2. **Update freeze**: If template change is intentional, re-freeze

### Missing Frozen Directory

If a frozen version's directory is missing, the build will:
1. Regenerate from templates
2. Verify output matches freeze manifest
3. Fail if they don't match

## Best Practices

1. **Freeze after release**: `--freeze v0.2.6` immediately after v0.2.6 release
2. **Always run `--check-immutability`** before deployment
3. **Never modify frozen version files** manually
4. **Use `--no-skip-frozen`** only for debugging
5. **Commit freeze manifests** to version control

## File Locations

- Freeze manifests: `releases/frozen/{version}.json`
- Regression tests: `tests/governance/test_frozen_version_immutability.py`
- Build script: `scripts/build_static_site.py`
