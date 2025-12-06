# Pre-commit Configuration Improvements

## Summary
Tightened pre-commit configuration to ensure clean commits by fixing trailing whitespace, end-of-file, YAML/JSON/TOML validation, and ASCII-only checks.

## Changes Made
- ✅ Fixed YAML syntax errors in `.pre-commit-config.yaml`
- ✅ Removed BOM (Byte Order Mark) from `backend/orchestrator/pyproject.toml`
- ✅ Cleaned non-ASCII and control characters from documentation and script files
- ✅ All pre-commit hooks now pass cleanly
- ✅ Added comprehensive rebase documentation in `docs/devxp/rebase_safely.md`

## Pre-commit Hooks Status
All hooks now pass:
- ✅ trim trailing whitespace
- ✅ fix end of files
- ✅ check yaml
- ✅ check json
- ✅ check toml
- ✅ check for merge conflicts
- ✅ check for added large files
- ✅ check for case conflicts
- ✅ ASCII-only check for docs/scripts/qa/tools/qa

## Quick Rebase Guide (1-minute)

```bash
# 1. Fetch latest changes
git fetch origin

# 2. Switch to your feature branch
git checkout your-feature-branch

# 3. Rebase onto main
git rebase origin/main

# 4. If conflicts occur, resolve them and continue
git add .
git rebase --continue

# 5. Force push safely
git push --force-with-lease origin your-feature-branch
```

## Files Modified
- `.pre-commit-config.yaml` - Fixed YAML syntax and moved ASCII check to separate script
- `scripts/check_ascii.py` - New ASCII validation script
- `backend/orchestrator/pyproject.toml` - Removed BOM
- `docs/progress.md` - Cleaned non-ASCII characters
- `scripts/metrics-watch.ps1` - Cleaned non-ASCII characters
- `docs/devxp/rebase_safely.md` - New comprehensive rebase guide

## Testing
- ✅ `pre-commit run --all-files` passes cleanly
- ✅ All file format validations pass
- ✅ No trailing whitespace or end-of-file issues
- ✅ All YAML, JSON, and TOML files are valid

## Documentation
Added detailed rebase guide at `docs/devxp/rebase_safely.md` with:
- Quick 1-minute reference
- Step-by-step instructions
- Conflict resolution guide
- Safety tips and best practices
- Emergency recovery procedures
