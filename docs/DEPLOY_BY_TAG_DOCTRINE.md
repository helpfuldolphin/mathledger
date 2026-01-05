# Deploy-by-Tag Doctrine

**Rule**: Never deploy from a branch. Only deploy from an annotated git tag that exists on origin.

## Deployability Criteria

A version `vX.Y.Z-*` is deployable **if and only if** ALL of the following are true:

### 1. Tag Exists on Origin
```bash
git ls-remote --tags origin vX.Y.Z-*
# Must return a result (tag exists on GitHub)
```

### 2. Tag Points to Commit C
```bash
git rev-parse vX.Y.Z-*
# Returns commit hash C
```

### 3. releases.json Matches
`releases/releases.json` must declare:
```json
{
  "current_version": "vX.Y.Z-*",
  "versions": {
    "vX.Y.Z-*": {
      "commit": "C",
      "tag": "vX.Y.Z-*"
    }
  }
}
```

### 4. Clean Deployment
Both Cloudflare Pages and Fly.io deployments are performed from that exact tag/commit:
- `git status` is clean (no uncommitted changes)
- `git describe --tags --exact-match HEAD` equals `vX.Y.Z-*`

### 5. Post-Deploy Health Check
`/demo/health` reports:
```json
{
  "tag": "vX.Y.Z-*",
  "commit": "C",
  "release_pin": {
    "is_stale": false
  }
}
```

## Decision Matrix

| Condition | Status |
|-----------|--------|
| Tag exists on origin | Required |
| Tag points to commit C | Required |
| releases.json current_version == tag | Required |
| releases.json commit == C | Required |
| Git status clean | Required |
| HEAD exactly matches tag | Required |
| /demo/health tag matches | Required |
| /demo/health commit matches | Required |
| release_pin.is_stale == false | Required |

**If ANY condition fails: NO-GO.**

## Pre-Deploy Gate

Before deploying, run:
```bash
uv run python tools/predeploy_gate.py vX.Y.Z-*
```

Or manually verify:
```bash
# 1. Clean status
git status --porcelain
# Must be empty

# 2. Exact tag match
git describe --tags --exact-match HEAD
# Must equal target tag

# 3. Tag exists on origin
git ls-remote --tags origin vX.Y.Z-*
# Must return result

# 4. releases.json matches
python -c "
import json
with open('releases/releases.json') as f:
    data = json.load(f)
print('current:', data['current_version'])
print('commit:', data['versions'][data['current_version']]['commit'])
print('tag:', data['versions'][data['current_version']]['tag'])
"
# Must match target tag and commit
```

## Post-Deploy Verification

After deployment:
```bash
# Check health endpoint
curl -s https://mathledger.ai/demo/health | python -m json.tool

# Expected output includes:
# "tag": "vX.Y.Z-*"
# "commit": "C"
# "release_pin": {"is_stale": false}
```

## Why This Matters

Tags alone can be abused if you:
- Deploy from a dirty tree
- Deploy from a branch head that isn't the tag
- Change releases.json without retagging

The real control is: **tag + release metadata + health pin must all match**.

This doctrine eliminates deployment chaos by making version identity a three-way checksum:
1. Git tag (immutable reference)
2. releases.json (declared intent)
3. /demo/health (runtime attestation)

If any disagree, the deployment is suspect.

---

**Enforcement**: This doctrine is checked by `tools/predeploy_gate.py` and should be run before every deployment.
