# Hosted Demo GO Checklist

**Purpose**: 60-second daily verification that the hosted demo infrastructure is correct.

**Version**: 0.2.1
**Last Updated**: 2026-01-03

---

## Quick GO/NO-GO (run all 5)

```powershell
# 1. Pages (archive) must remain untouched
curl.exe -sI https://mathledger.ai/v0.2.1/ | findstr /I "cache-control"
# Expected: cache-control: public, max-age=31536000, immutable

# 2. Worker routing must be live
curl.exe -sI https://mathledger.ai/demo/healthz | findstr /I "X-Proxied-By"
# Expected: X-Proxied-By: mathledger-demo-proxy

# 3. Demo version pinning must be correct
curl.exe -s https://mathledger.ai/demo/health | python -c "import sys,json; d=json.load(sys.stdin); print(f'version={d.get(\"version\")} tag={d.get(\"tag\")}')"
# Expected: version=0.2.1 tag=v0.2.1-cohesion

# 4. Release pin validation (anti-drift)
curl.exe -s https://mathledger.ai/demo/health | python -c "import sys,json; d=json.load(sys.stdin); print('STALE' if d.get('release_pin',{}).get('is_stale') else 'OK')"
# Expected: OK

# 5. Evidence-pack verifier must be present (static auditor tool)
curl.exe -s -o NUL -w "%{http_code}" https://mathledger.ai/v0.2.1/evidence-pack/verify/
# Expected: 200
```

---

## Expected Results Table

| Check | Command | Expected | GO if |
|-------|---------|----------|-------|
| Pages immutable | `findstr cache-control` | `immutable` | Header contains "immutable" |
| Worker routing | `findstr X-Proxied-By` | `mathledger-demo-proxy` | Header present |
| Version pinned | `/demo/health` | `version=0.2.1` | Matches exactly |
| Tag pinned | `/demo/health` | `tag=v0.2.1-cohesion` | Matches exactly |
| Release pin | `/demo/health` | `is_stale=false` | Not stale |
| Verifier page | `%{http_code}` | `200` | Status 200 |

---

## One-Liner (PowerShell)

```powershell
# All-in-one GO check using anti-drift tool
uv run python tools/check_hosted_demo_matches_release.py
# Exit code 0 = GO, Exit code 1 = NO-GO (deploy required), Exit code 3 = container rebuild
```

## Manual PowerShell Check

```powershell
# Manual version check
$health = curl.exe -s https://mathledger.ai/demo/health | ConvertFrom-Json
if ($health.version -eq "0.2.1" -and $health.release_pin.is_stale -eq $false) {
    Write-Host "GO - Version and release pin valid" -ForegroundColor Green
} else {
    Write-Host "NO-GO - Mismatch detected" -ForegroundColor Red
    Write-Host "  Version: $($health.version)"
    Write-Host "  Stale: $($health.release_pin.is_stale)"
}
```

---

## Failure Modes

### Pages returns wrong cache-control
- **Cause**: Cloudflare Pages misconfigured or redeployed without immutable headers
- **Fix**: Check `_headers` file in Pages deployment

### X-Proxied-By missing
- **Cause**: Cloudflare Worker not deployed or route not matching `/demo/*`
- **Fix**: Verify Worker is published and route is `mathledger.ai/demo/*`

### Version mismatch
- **Cause**: Fly app redeployed with wrong version, or wrong app targeted
- **Fix**: Verify `fly status -a mathledger-demo-v0-2-0-helpfuldolphin` shows correct deployment

### Verifier returns 404
- **Cause**: Pages missing `/v0.2.1/evidence-pack/verify/` directory
- **Fix**: Redeploy Pages with `site/v0.2.1/evidence-pack/verify/index.html`

### Release pin stale
- **Cause**: Container's releases.json doesn't match running code
- **Fix**: Rebuild Docker with --no-cache and redeploy

---

## Architecture Reference

```
                                    ┌─────────────────────────┐
                                    │   Cloudflare Pages      │
                                    │   (immutable archive)   │
    /v0.2.1/*  ──────────────────► │   mathledger.ai/v0.2.1/ │
                                    └─────────────────────────┘

    Browser ─────┐
                 │
                 ▼
    ┌────────────────────────┐      ┌─────────────────────────┐
    │   Cloudflare Worker    │      │      Fly.io App         │
    │   (rewrites /demo/*)   │ ───► │   ROOT MOUNT (/)        │
    │   X-Proxied-By header  │      │   v0.2.1-cohesion       │
    └────────────────────────┘      └─────────────────────────┘
         /demo/*  ──────────────────► /*  (prefix stripped)
```

---

## Contract

| Component | Authoritative Value |
|-----------|---------------------|
| Fly App | `mathledger-demo-v0-2-0-helpfuldolphin` |
| Version | `0.2.1` |
| Tag | `v0.2.1-cohesion` |
| Commit | `27a94c8a58139cb10349f6418336c618f528cbab` |
| Worker Header | `X-Proxied-By: mathledger-demo-proxy` |
| Pages Cache | `immutable` |
| Release Pin | `is_stale: false` |

---

**Author**: Claude A
**Ritual**: Run `uv run python tools/check_hosted_demo_matches_release.py` before any demo or external review
