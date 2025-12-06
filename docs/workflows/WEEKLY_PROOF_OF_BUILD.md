# Weekly Proof-of-Build Workflow

This document describes the weekly automated proof-of-build workflow that executes the flight deck operator flows and creates a weekly issue with sealed artifacts.

## Overview

The weekly proof-of-build workflow:
1. Runs `ml flightdeck` to execute all operator flows
2. Generates RFC 8785 sealed artifacts
3. Creates or updates a "Weekly Proof-of-Build" GitHub issue
4. Uploads artifacts for verification

## Workflow File

Due to OAuth workflow scope limitations, the workflow file must be manually added to `.github/workflows/weekly-proof-of-build.yml`.

**Location**: `.github/workflows/weekly-proof-of-build.yml`

**Schedule**: Every Monday at 00:00 UTC (configurable via cron expression)

**Manual Trigger**: Workflow can also be triggered manually via GitHub Actions UI

## Installation

Copy the workflow file from this repository to `.github/workflows/weekly-proof-of-build.yml`:

```bash
# From repository root
cp docs/workflows/weekly-proof-of-build.yml .github/workflows/
git add .github/workflows/weekly-proof-of-build.yml
git commit -m "ci: add weekly proof-of-build workflow"
git push
```

## Workflow Contents

The workflow performs these steps:

1. **Checkout repository** - Gets latest code
2. **Set up Python** - Installs Python 3.11
3. **Install dependencies** - Installs required packages
4. **Run flight deck** - Executes `ml flightdeck` command
5. **Upload artifacts** - Uploads sealed artifacts
6. **Create/update issue** - Creates or updates weekly issue

## Output Format

The workflow creates an issue with this format:

```markdown
## Weekly Proof-of-Build Report

**Timestamp**: 2025-10-31T23:26:49.011608Z
**Signature**: `f0b29101c8e7ed7508b82928ef0ad1b1364193971b75657fabb4be211dfeb793`
**Success**: 3/3
**Run**: [View workflow run](https://github.com/...)

### Artifacts

- [Flight Deck Report](...)
- [Performance Log](...)
- [Audit Trail](...)
- [Fleet State](...)

### Pass-Line

```
[PASS] Flight Deck: f0b29101c8e7ed7508b82928ef0ad1b1364193971b75657fabb4be211dfeb793
```

### Operations Executed

1. **velocity seal** - CI velocity measurement
2. **audit chain** - Audit trail sync+verify
3. **allblue freeze** - Fleet state archival
```

## Artifacts Generated

The workflow generates these RFC 8785 sealed artifacts:

1. **artifacts/ops/flightdeck.json** - Consolidated flight deck report
2. **artifacts/perf/perf_log.json** - CI velocity measurements
3. **artifacts/audit/audit_trail.jsonl** - Audit trail with hash chain
4. **artifacts/allblue/fleet_state.json** - Frozen fleet state

All artifacts are uploaded to GitHub Actions and linked in the issue.

## Manual Execution

To run the flight deck manually:

```bash
# Run flight deck
ml flightdeck

# Verify sealed report
python tools/devin_e_toolbox/flightdeck.py --verify
```

## Verification

To verify the sealed artifacts:

```bash
# Verify flight deck report
python tools/devin_e_toolbox/flightdeck.py --verify

# Verify individual components
python tools/devin_e_toolbox/ci_velocity.py --verify
python tools/devin_e_toolbox/audit_sync.py --verify
python tools/devin_e_toolbox/allblue_archive.py --verify
```

## Troubleshooting

### Workflow not running

- Check that the workflow file exists in `.github/workflows/`
- Verify the cron schedule is correct
- Check GitHub Actions permissions (issues: write required)

### Artifacts not uploading

- Ensure artifacts directory exists
- Check that all operator flows completed successfully
- Verify artifact paths in workflow match actual paths

### Issue not created

- Check that `issues: write` permission is granted
- Verify GitHub Actions token has correct permissions
- Check workflow logs for errors

## Configuration

### Schedule

To change the schedule, modify the cron expression:

```yaml
on:
  schedule:
    - cron: '0 0 * * 1'  # Every Monday at 00:00 UTC
```

Cron format: `minute hour day-of-month month day-of-week`

Examples:
- `0 0 * * 1` - Every Monday at midnight
- `0 0 * * 0` - Every Sunday at midnight
- `0 0 1 * *` - First day of every month
- `0 0 * * *` - Every day at midnight

### Labels

The issue is tagged with these labels:
- `weekly-proof-of-build` - Identifies the weekly report
- `automation` - Indicates automated creation

To change labels, modify the workflow script section.

## Security Considerations

- Workflow uses `contents: read` and `issues: write` permissions only
- No secrets required for basic operation
- All artifacts are public (uploaded to GitHub Actions)
- Signatures are SHA256 hashes (no private keys required)

## Integration with CI

The weekly proof-of-build workflow is independent of regular CI checks. It provides:

- **Historical record** - Weekly snapshots of fleet state
- **Audit trail** - Continuous audit chain verification
- **Performance tracking** - CI velocity trends over time
- **State archival** - All-blue state preservation

---

**Version**: v2.2  
**Maintained by**: Devin E Toolbox  
**Documentation**: See `tools/devin_e_toolbox/README.md`
