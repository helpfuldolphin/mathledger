# AllBlue Epoch Seal - Manual CI Integration

## Overview

This document describes manual steps for integrating AllBlue Epoch Seal generation, registry append, and PR comment posting into CI workflows.

Due to OAuth workflow scope limitations, these steps must be applied manually by repository maintainers.

## Prerequisites

- Python 3.11+ (for `datetime.now(datetime.UTC)` support)
- `gh` CLI tool authenticated with repository access
- Write access to `.github/workflows/` directory

## Integration Steps

### Step 1: Generate Epoch Seal

Run the epoch seal generator after all verification lanes complete:

```bash
python3 scripts/generate_allblue_epoch_seal.py \
  --config config/allblue_lanes.json \
  --output-dir artifacts/allblue \
  --rfcsign
```

**Expected Outputs:**
- `artifacts/allblue/fleet_state.json` (RFC 8785 canonicalized)
- `artifacts/allblue/fleet_state_readable.json` (human-readable)
- `artifacts/allblue/epoch_registry.jsonl` (append-only log)
- `artifacts/allblue/witness_verify.json` (witness verification results)

**Exit Codes:**
- `0`: PASS (all required lanes verified, witnesses verified)
- `1`: ABSTAIN (missing required lanes or witnesses)
- `2`: FAIL (verification failures)

### Step 2: Verify Registry Append

Check that the epoch was appended to the registry:

```bash
tail -1 artifacts/allblue/epoch_registry.jsonl | python3 -c "import sys, json; print(json.dumps(json.loads(sys.stdin.read()), indent=2))"
```

**Expected Fields:**
- `timestamp`: ISO 8601 UTC timestamp
- `epoch_hash`: Base epoch seal (64-hex SHA256)
- `witnessed_epoch_hash`: Crown seal with witnesses (64-hex SHA256)
- `status`: PASS, ABSTAIN, or FAIL
- `witnesses`: Object with three witness signatures

### Step 3: Post PR Comment (Dry-Run First)

**IMPORTANT:** Always test with `--dry-run` before posting:

```bash
python3 scripts/post_epoch_pr_comment.py \
  --fleet-state artifacts/allblue/fleet_state.json \
  --dry-run
```

Review the output to ensure:
- Both epoch seals are present
- Witness status table is correct
- Pass-lines are formatted properly
- Lane summary is accurate

### Step 4: Post PR Comment (Live)

After dry-run verification, post the comment:

```bash
python3 scripts/post_epoch_pr_comment.py \
  --fleet-state artifacts/allblue/fleet_state.json \
  --pr-number ${{ github.event.pull_request.number }}
```

Or with explicit repository:

```bash
python3 scripts/post_epoch_pr_comment.py \
  --fleet-state artifacts/allblue/fleet_state.json \
  --pr-number 58 \
  --repo helpfuldolphin/mathledger
```

### Step 5: Commit Registry Updates (Optional)

If the registry should be committed to the repository:

```bash
git add artifacts/allblue/epoch_registry.jsonl
git commit -m "allblue: append epoch registry [skip ci]"
git push origin ${{ github.head_ref }}
```

**Note:** Use `[skip ci]` to avoid triggering recursive CI runs.

## GitHub Actions Integration Example

```yaml
name: AllBlue Epoch Seal

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  allblue-seal:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Generate Epoch Seal
        run: |
          python3 scripts/generate_allblue_epoch_seal.py \
            --config config/allblue_lanes.json \
            --output-dir artifacts/allblue \
            --rfcsign
      
      - name: Verify Registry Append
        run: |
          echo "Latest registry entry:"
          tail -1 artifacts/allblue/epoch_registry.jsonl | python3 -m json.tool
      
      - name: Post PR Comment (Dry-Run)
        run: |
          python3 scripts/post_epoch_pr_comment.py \
            --fleet-state artifacts/allblue/fleet_state.json \
            --dry-run
      
      - name: Post PR Comment
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          python3 scripts/post_epoch_pr_comment.py \
            --fleet-state artifacts/allblue/fleet_state.json \
            --pr-number ${{ github.event.pull_request.number }}
      
      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: allblue-epoch-seal
          path: |
            artifacts/allblue/fleet_state.json
            artifacts/allblue/epoch_registry.jsonl
            artifacts/allblue/witness_verify.json
```

## Pass-Lines Reference

### PASS (All Witnesses Verified)

```
[PASS] ALL BLUE: sha256=<epoch_hash>
[PASS] Epoch Seal <epoch_hash>
[PASS] Witnessed Epoch <witnessed_epoch_hash>
[PASS] Witnesses Verified
```

### ABSTAIN (Missing Witnesses)

```
[ABSTAIN] AllBlue Gate: 1 required lane(s) missing artifacts
[INFO] Epoch Seal <epoch_hash>
[INFO] Witnessed Epoch <witnessed_epoch_hash>
[ABSTAIN] missing witness (lane=verification_gate, reason=triple_hash_not_verified)
[ABSTAIN] missing witness (lane=hermetic_matrix, reason=dual_attestation_lane_missing)
[ABSTAIN] missing witness (lane=perf_gate, reason=uplift_evaluation_not_verified)
```

### FAIL (Witness Verification Failed)

```
[FAIL] AllBlue Gate: Some required lanes failed verification
[INFO] Epoch Seal <epoch_hash>
[INFO] Witnessed Epoch <witnessed_epoch_hash>
[FAIL] witness verification failed (lane=verification_gate, reason=signature_mismatch)
```

## Troubleshooting

### Issue: `datetime.utcnow()` deprecation warnings

**Solution:** Ensure Python 3.11+ is used. The script has been updated to use `datetime.now(datetime.UTC)`.

### Issue: PR comment fails with "Could not determine PR number"

**Solution:** Provide `--pr-number` explicitly or ensure running in GitHub Actions context with `github.event.pull_request.number`.

### Issue: Witness verification shows ABSTAIN for all witnesses

**Solution:** This is expected on feature branches where CI workflows may not run completely. Witnesses will PASS when:
- `triple_hash` lane has PASS status (verification gate)
- `dual_attestation` lane has PASS status (hermetic matrix)
- `uplift_evaluation` lane has PASS status (perf gate)

### Issue: Registry append creates duplicate entries

**Solution:** This is expected behavior. The registry is append-only. Each run creates a new entry with timestamp.

### Issue: `gh` CLI not authenticated

**Solution:** Run `gh auth login` or set `GH_TOKEN` environment variable in CI.

## RFC 8785 Compliance

All JSON artifacts are canonicalized according to RFC 8785:
- Lexicographic key ordering (`sort_keys=True`)
- No whitespace between tokens (`separators=(',', ':')`)
- ASCII-only output (`ensure_ascii=True`)
- Deterministic number representation

This ensures byte-identical output across platforms and replay attempts.

## Proof-or-Abstain Discipline

The AllBlue Epoch Seal follows strict Proof-or-Abstain discipline:
- Never generates speculative signatures for unverified lanes
- ABSTAIN with explicit reason when artifacts missing
- FAIL with explicit reason when verification fails
- No success without cryptographic proof

## Witness Signature Verification

Witness signatures are verified by recomputing expected signatures:

```python
# Verification Gate
expected = sha256(f"verification_gate:{triple_hash_lane_hash}:{H_t_hash}")

# Hermetic Matrix
expected = sha256(f"hermetic_matrix:{dual_attestation_lane_hash}")

# Perf Gate
expected = sha256(f"perf_gate:{uplift_evaluation_lane_hash}")
```

If actual signature matches expected, witness is PASS. Otherwise, ABSTAIN or FAIL.

## Branch Protection Configuration

### Making Epoch Seal a Required Check

To enforce epoch seal verification on all PRs:

1. Navigate to GitHub repository settings
2. Go to **Branches** → **Branch protection rules**
3. Select or create rule for `integrate/ledger-v0.1` or `main`
4. Enable **Require status checks to pass before merging**
5. Add **AllBlue Epoch Seal** to required checks list
6. Enable **Require branches to be up to date before merging**
7. Save changes

### Override Procedure for Emergency Merges

When epoch seal check fails but merge is critical:

1. **Document the reason** in PR description or comment
2. **Get approval** from two maintainers
3. **Temporarily disable** branch protection:
   - Settings → Branches → Edit rule
   - Uncheck "Require status checks to pass"
   - Save changes
4. **Merge the PR** immediately
5. **Re-enable** branch protection:
   - Settings → Branches → Edit rule
   - Re-check "Require status checks to pass"
   - Save changes
6. **Create follow-up issue** to address epoch seal failure

**Emergency Override Criteria:**
- Production outage requiring immediate fix
- Security vulnerability requiring urgent patch
- Critical data loss prevention
- Epoch seal infrastructure failure (not PR content issue)

**DO NOT override for:**
- Convenience or time pressure
- Missing witness signatures due to incomplete testing
- Failed verification due to actual code issues
- Desire to skip verification process

## CI Infrastructure Fixes

### Issue: Deprecated upload-artifact@v3

Two workflows require upgrade to `upload-artifact@v4`:

**Affected Workflows:**
- `.github/workflows/evidence-gate.yml` (line 44)
- `.github/workflows/verification-gate.yml` (line 37)

**Patch File:** `/tmp/ci-infra-fixes.patch`

**Manual Application:**

```bash
# Option 1: Apply patch
cd /path/to/mathledger
git apply /tmp/ci-infra-fixes.patch
git commit -m "ci: upgrade upload-artifact@v3 to @v4"
git push origin integrate/ledger-v0.1

# Option 2: Manual edit
# Edit .github/workflows/evidence-gate.yml line 44
# Change: uses: actions/upload-artifact@v3
# To:     uses: actions/upload-artifact@v4

# Edit .github/workflows/verification-gate.yml line 37
# Change: uses: actions/upload-artifact@v3
# To:     uses: actions/upload-artifact@v4
```

**Verification:**

```bash
# Check for remaining v3 references
grep -r "upload-artifact@v3" .github/workflows/

# Expected output: (empty)
```

**Impact:**
- Fixes deprecation warnings in CI logs
- Ensures compatibility with GitHub Actions updates
- No functional changes to artifact upload behavior

## Maintainer Checklist

- [ ] Python 3.11+ installed
- [ ] `gh` CLI authenticated
- [ ] Dry-run PR comment tested
- [ ] Pass-lines output verified
- [ ] Registry append confirmed
- [ ] Witness verification status checked
- [ ] CI workflow integrated (if applicable)
- [ ] Branch protection updated (if epoch seal is required check)
- [ ] CI infrastructure fixes applied (upload-artifact@v4)

## Contact

For questions or issues, refer to:
- `docs/BRANCH_PROTECTION.md` - Branch protection configuration
- `docs/ALLBLUE_ACCEPTANCE_TESTS.md` - Acceptance test specifications
- `scripts/generate_allblue_epoch_seal.py` - Epoch seal generator source
- `scripts/post_epoch_pr_comment.py` - PR comment hook source
