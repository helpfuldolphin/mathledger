# Release Metadata Diagnostic

**Purpose**: This document explains how to diagnose and fix release metadata discrepancies.

---

## Canonical Source of Truth

The **only** authoritative source for release metadata is:

```
releases/releases.json
```

All other sources (git describe, environment variables, tool output) are **derived** and may be stale or wrong.

---

## v0.2.0 Canonical Values

| Field | Canonical Value |
|-------|-----------------|
| current_version | `v0.2.0` |
| tag | `v0.2.0-demo-lock` |
| commit | `27a94c8a58139cb10349f6418336c618f528cbab` |
| date_locked | `2026-01-02` |

---

## v0 Canonical Values

| Field | Canonical Value |
|-------|-----------------|
| tag | `v0-demo-lock` |
| commit | `ab8f51ab389aed7b3412cb987fc70d0d4f2bbe0b` |
| date_locked | `2026-01-02` |

---

## If a Tool Reports Different Values

If any tool (Claude B, build script, CI, etc.) reports different tag/commit values:

1. **The tool is wrong.** The tool is reading from the wrong source.
2. **Do NOT update releases.json** to match the tool's output.
3. **Fix the tool** to read from `releases/releases.json`.

### Example: Claude B Reported Wrong Values

Claude B previously reported:
- tag: `v0.9.4-pilot-audit-hardened` (WRONG)
- commit: `07ea0edf02ff4173e81cef8ecfedf50195bb8673` (WRONG)

These values are **incorrect**. Claude B was reading from a different source (possibly git describe or a different branch).

The fix was to:
1. Restore correct values in `releases/releases.json`
2. Add guard tests that fail if wrong values appear
3. Document this diagnostic procedure

---

## Guard Tests

The following tests will FAIL if releases.json is corrupted:

```bash
uv run pytest tests/governance/test_release_metadata_guard.py -v
```

Tests:
- `test_current_version_is_v020` - Fails if current_version != "v0.2.0"
- `test_v020_tag_is_demo_lock` - Fails if tag != "v0.2.0-demo-lock"
- `test_v020_commit_is_canonical` - Fails if commit != "27a94c8a..."
- `test_no_pilot_audit_hardened_tag` - Fails if wrong tag appears anywhere
- `test_no_wrong_commit` - Fails if wrong commit appears anywhere

---

## How to Verify Metadata is Correct

```bash
# 1. Run guard tests
uv run pytest tests/governance/test_release_metadata_guard.py -v

# 2. Verify JSON directly
python -c "
import json
d = json.load(open('releases/releases.json'))
v = d['versions']['v0.2.0']
print(f\"current_version: {d['current_version']}\")
print(f\"v0.2.0 tag: {v['tag']}\")
print(f\"v0.2.0 commit: {v['commit']}\")
"

# 3. Verify git tag matches (if tag exists locally)
git rev-parse v0.2.0-demo-lock^{commit}
# Should output: 27a94c8a58139cb10349f6418336c618f528cbab
```

---

## Root Cause Prevention

To prevent future discrepancies:

1. **Build scripts MUST read from releases.json**, not infer from git
2. **CI validation MUST include guard tests** before deployment
3. **Human review required** before modifying releases.json
4. **Immutability rule**: Once a version is locked, only `status` may change

---

## Contact

If you encounter a discrepancy and are unsure how to proceed:
1. Do NOT modify releases.json
2. Run the guard tests
3. Document the discrepancy in an issue
4. Wait for human review

---

**Author**: Claude A (Reconciliation Fix)
**Date**: 2026-01-02
