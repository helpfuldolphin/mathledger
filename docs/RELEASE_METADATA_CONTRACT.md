# Release Metadata Contract

**File**: `releases/releases.json`
**Status**: Authoritative
**Date**: 2026-01-02

---

## Purpose

`releases/releases.json` is the **single source of truth** for deployment metadata.

This file records:
- Current deployed version
- Git tags and commit hashes for all releases
- Date each version was locked
- Tier counts (A/B/C) for governance invariants
- Status of each release (current, superseded, internal)

---

## Contract Rules

### 1. Authoritative for Deployment

Build scripts, CI pipelines, and deployment tooling **MUST** read version information from `releases/releases.json`.

```python
# CORRECT
import json
with open("releases/releases.json") as f:
    releases = json.load(f)
    current = releases["current_version"]

# WRONG - do not infer from other sources
version = subprocess.check_output(["git", "describe", "--tags"])
```

### 2. No Inference from Other Sources

Deployment tooling **MUST NOT** infer tags or commits from:
- Git describe output
- Package version strings
- Other documentation files
- Environment variables

The `releases/releases.json` file is canonical. If it disagrees with git, the file is wrong and must be fixed.

### 3. Immutable After Lock

Once a version's status is `current` or `superseded`, its metadata **MUST NOT** change except to update `status` when superseded.

Fields that are immutable after lock:
- `version`
- `git_tag`
- `commit_hash`
- `date_locked`
- `tier_counts`

### 4. Validation Required

Before deployment, tooling **MUST** validate:

```bash
# Tag exists
git rev-parse "v0.2.0-demo-lock" > /dev/null

# Commit matches
[ "$(git rev-parse v0.2.0-demo-lock^{commit})" = "27a94c8a58139cb10349f6418336c618f528cbab" ]
```

Use `tests/governance/test_release_metadata.py` for automated validation.

---

## Schema

```json
{
  "current_version": "string (semver)",
  "releases": [
    {
      "version": "string (semver)",
      "git_tag": "string (must exist in repo)",
      "commit_hash": "string (40 hex chars)",
      "date_locked": "string (YYYY-MM-DD)",
      "status": "current | superseded | internal",
      "tier_counts": {
        "tier_a": "integer",
        "tier_b": "integer",
        "tier_c": "integer"
      },
      "notes": "string (optional)",
      "verification_commands": ["string array (optional)"],
      "new_invariants": ["string array (optional)"]
    }
  ]
}
```

---

## Status Definitions

| Status | Meaning |
|--------|---------|
| `current` | Active release. Deployments should use this version. |
| `superseded` | Previous release. May be deployed for rollback only. |
| `internal` | Development milestone. Not for external deployment. |

---

## Adding a New Release

1. Create and push the git tag
2. Add entry to `releases/releases.json`
3. Update `current_version`
4. Set previous current release to `superseded`
5. Run `uv run pytest tests/governance/test_release_metadata.py -v`
6. Commit changes

---

## Cross-Reference

This file is referenced by:
- `docs/V0_LOCK.md` - Release notes
- `tests/governance/test_release_metadata.py` - Validation tests
- Build/deployment scripts (must use this as source of truth)

---

**Author**: Claude A (v0.2.0 Release Closure)
**Contract Effective**: 2026-01-02
