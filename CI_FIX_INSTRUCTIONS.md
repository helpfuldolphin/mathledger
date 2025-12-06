# CI Artifact Upload Fix - Manual Application Required

## Issue
GitHub Actions workflow `performance-sanity.yml` is failing due to deprecated `actions/upload-artifact@v3`.
OAuth workflow scope prevents automated push of workflow file changes.

## Error Message
```
This request has been automatically failed because it uses a deprecated 
version of `actions/upload-artifact: v3`.
```

## Fix Required
Update line 46 of `.github/workflows/performance-sanity.yml`:
- **FROM:** `uses: actions/upload-artifact@v3`
- **TO:** `uses: actions/upload-artifact@v4`

## Manual Application Steps

### Option 1: GitHub Web UI (Recommended)
1. Navigate to: https://github.com/helpfuldolphin/mathledger/blob/integrate/ledger-v0.1/.github/workflows/performance-sanity.yml
2. Click "Edit this file" (pencil icon)
3. Find line 46: `uses: actions/upload-artifact@v3`
4. Change to: `uses: actions/upload-artifact@v4`
5. Commit directly to `integrate/ledger-v0.1` with message:
   ```
   ci: fix artifact upload deprecation (v3 -> v4)
   
   Resolves 12+ consecutive CI failures due to GitHub Actions
   platform deprecation of artifact v3.
   ```

### Option 2: Git Patch Application
```bash
# Apply the patch file
cd /path/to/mathledger
git apply CI_FIX_PATCH.diff

# Commit and push
git add .github/workflows/performance-sanity.yml
git commit -m "ci: fix artifact upload deprecation (v3 -> v4)"
git push origin integrate/ledger-v0.1
```

## Patch File
```diff
diff --git a/.github/workflows/performance-sanity.yml b/.github/workflows/performance-sanity.yml
index d005f3d..f616628 100644
--- a/.github/workflows/performance-sanity.yml
+++ b/.github/workflows/performance-sanity.yml
@@ -43,7 +43,7 @@ jobs:
 
     - name: Archive Performance Artifacts
       if: always()
-      uses: actions/upload-artifact@v3
+      uses: actions/upload-artifact@v4
       with:
         name: performance-passport-${{ github.run_number }}
         path: |
```

## Verification
After applying the fix:
1. Monitor next scheduled run (daily at 2 AM UTC)
2. Or trigger manual workflow run via GitHub Actions UI
3. Verify "Archive Performance Artifacts" step succeeds
4. Check that performance passport artifact is uploaded

## Impact
- **Blocker Severity:** CRITICAL
- **Affected Workflow:** performance-sanity (scheduled daily)
- **Failure Duration:** 12+ days (Oct 8-19, 2025)
- **Expected Resolution:** Immediate (single-line change)

## Related Files
- `.github/workflows/performance-sanity.yml` (line 46)
- `SPRINT_STATUS.md` (comprehensive status report)
- `/tmp/ci-artifact-v4-fix.patch` (local patch file)

## Contact
- **Mission Conductor:** Devin J
- **Session:** https://app.devin.ai/sessions/a4d865ce3da54e7ba6119a84a8cbd8e3
- **User:** helpful.dolphin@pm.me (@helpfuldolphin)
