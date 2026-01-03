# Hosted Demo GO Checklist

**Purpose**: Verification that hosted demo infrastructure matches releases.json.

**Current Version**: Read from `releases/releases.json` (`current_version` field)
**Last Updated**: 2026-01-03

---

## CRITICAL: Operator Deployment Flow

After updating `releases/releases.json` with a new version:

```
1. Update releases/releases.json "current_version"
2. Deploy static site:     wrangler pages deploy site/
3. Rebuild & deploy demo:  docker build -t mathledger-demo . && fly deploy
4. Verify match:           uv run python tools/check_hosted_demo_matches_release.py
5. Run hostile audit:      powershell tools/hostile_audit.ps1 v0.2.2
                           (replace v0.2.2 with current version)
6. ONLY PROCEED IF BOTH PASS
```

**After Pages deploy, you MUST fly deploy; then rerun hostile_audit.ps1 until PASS.**

---

## Quick GO/NO-GO (run all checks)

```powershell
# 1. One-liner anti-drift check (MUST PASS)
uv run python tools/check_hosted_demo_matches_release.py
# Exit code 0 = GO, Exit code 1 = NO-GO (deploy required), Exit code 3 = container rebuild

# 2. Hostile audit (MUST PASS)
powershell tools/hostile_audit.ps1 v0.2.2
# ALL checks must pass

# 3. Manual /healthz header check
curl.exe -sI https://mathledger.ai/demo/healthz
# Expected headers:
#   X-MathLedger-Version: v0.2.2
#   X-MathLedger-Tag: v0.2.2-link-integrity
#   X-MathLedger-Commit: 27a94c8a58139cb10349f6418336c618f528cbab
#   X-MathLedger-Stale-Deploy: false
```

---

## Expected Values (from releases.json)

The demo must report values matching `releases/releases.json`:

| Field | Source | How to Check |
|-------|--------|--------------|
| version | `current_version` (without leading v) | `/demo/health` JSON `version` field |
| tag | `versions[current_version].tag` | `/demo/health` JSON `tag` field |
| commit | `versions[current_version].commit` | `/demo/health` JSON `commit` field |

**Dynamic**: Values are read from releases.json at demo startup. Update releases.json and redeploy to change.

---

## Failure Modes and Fixes

### Version mismatch (most common)
- **Symptom**: `hostile_audit.ps1` fails "Version match"
- **Cause**: Demo container has old releases.json baked in
- **Fix**:
  ```bash
  docker build --no-cache -t mathledger-demo .
  fly deploy -a mathledger-demo-v0-2-0-helpfuldolphin
  ```

### Release pin stale
- **Symptom**: `/demo/health` shows `release_pin.is_stale: true`
- **Cause**: Container's releases.json differs from code expectations
- **Fix**: Same as above - rebuild with `--no-cache`

### X-Proxied-By missing
- **Symptom**: `/demo/healthz` missing `X-Proxied-By` header
- **Cause**: Cloudflare Worker not deployed or route broken
- **Fix**: Verify Worker at `mathledger.ai/demo/*` route

### Verifier returns 404
- **Symptom**: `/v0.2.2/evidence-pack/verify/` returns 404
- **Cause**: Pages missing the verifier directory
- **Fix**: Rebuild static site and redeploy Pages

---

## Architecture Reference

```
                                    ┌─────────────────────────┐
                                    │   Cloudflare Pages      │
                                    │   (immutable archive)   │
    /v0.2.2/*  ──────────────────► │   mathledger.ai/v0.2.2/ │
                                    └─────────────────────────┘

    Browser ─────┐
                 │
                 ▼
    ┌────────────────────────┐      ┌─────────────────────────┐
    │   Cloudflare Worker    │      │      Fly.io App         │
    │   (rewrites /demo/*)   │ ───► │   ROOT MOUNT (/)        │
    │   X-Proxied-By header  │      │   reads releases.json   │
    └────────────────────────┘      └─────────────────────────┘
         /demo/*  ──────────────────► /*  (prefix stripped)
```

**Key insight**: The Fly.io app reads `releases/releases.json` at startup. Rebuilding the Docker image with updated releases.json automatically updates the reported version.

---

## Contract (Dynamic from releases.json)

| Component | Source |
|-----------|--------|
| Fly App | `mathledger-demo-v0-2-0-helpfuldolphin` |
| Version | `releases.json current_version` |
| Tag | `releases.json versions[current].tag` |
| Commit | `releases.json versions[current].commit` |
| Worker Header | `X-Proxied-By: mathledger-demo-proxy` |
| Health Headers | `X-MathLedger-Version`, `X-MathLedger-Tag`, `X-MathLedger-Commit` |

---

## Pre-Deployment Checklist

Before deploying a new version:

- [ ] `releases/releases.json` updated with new version entry
- [ ] `current_version` field points to new version
- [ ] New version has correct `tag`, `commit`, `date_locked`
- [ ] Static site built: `uv run python scripts/build_static_site.py`
- [ ] Pages deployed: `wrangler pages deploy site/`
- [ ] Docker rebuilt: `docker build --no-cache -t mathledger-demo .`
- [ ] Fly deployed: `fly deploy -a mathledger-demo-v0-2-0-helpfuldolphin`
- [ ] Anti-drift check passes: `uv run python tools/check_hosted_demo_matches_release.py`
- [ ] Hostile audit passes: `powershell tools/hostile_audit.ps1 <version>`

---

**Author**: Claude A
**Ritual**: Run `uv run python tools/check_hosted_demo_matches_release.py` AND `hostile_audit.ps1` before any demo or external review
