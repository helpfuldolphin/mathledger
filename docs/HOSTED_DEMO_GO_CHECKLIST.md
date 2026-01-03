# Hosted Demo GO Checklist

**Purpose**: 60-second daily verification that the hosted demo infrastructure is correct.

**Version**: 0.2.0
**Last Updated**: 2026-01-03

---

## Quick GO/NO-GO (run all 4)

```powershell
# 1. Pages (archive) must remain untouched
curl.exe -sI https://mathledger.ai/v0.2.0/ | findstr /I "cache-control"
# Expected: cache-control: public, max-age=31536000, immutable

# 2. Worker routing must be live
curl.exe -sI https://mathledger.ai/demo/healthz | findstr /I "X-Proxied-By"
# Expected: X-Proxied-By: mathledger-demo-proxy

# 3. Demo version pinning must be correct
curl.exe -sI https://mathledger.ai/demo/healthz | findstr /I "x-mathledger-version x-mathledger-commit"
# Expected: x-mathledger-version: v0.2.0
# Expected: x-mathledger-commit: 27a94c8a58139cb10349f6418336c618f528cbab

# 4. Evidence-pack verifier must be present (static auditor tool)
curl.exe -s -o NUL -w "%{http_code}" https://mathledger.ai/v0.2.0/evidence-pack/verify/
# Expected: 200
```

---

## Expected Results Table

| Check | Command | Expected | GO if |
|-------|---------|----------|-------|
| Pages immutable | `findstr cache-control` | `immutable` | Header contains "immutable" |
| Worker routing | `findstr X-Proxied-By` | `mathledger-demo-proxy` | Header present |
| Version pinned | `findstr x-mathledger-version` | `v0.2.0` | Matches exactly |
| Commit pinned | `findstr x-mathledger-commit` | `27a94c8a...` | First 8 chars match |
| Verifier page | `%{http_code}` | `200` | Status 200 |

---

## One-Liner (PowerShell)

```powershell
# All-in-one GO check (returns 0 if all pass)
$pages = (curl.exe -sI https://mathledger.ai/v0.2.0/ | Select-String "immutable")
$worker = (curl.exe -sI https://mathledger.ai/demo/healthz | Select-String "X-Proxied-By")
$version = (curl.exe -sI https://mathledger.ai/demo/healthz | Select-String "x-mathledger-version: v0.2.0")
$verifier = (curl.exe -s -o NUL -w "%{http_code}" https://mathledger.ai/v0.2.0/evidence-pack/verify/)

if ($pages -and $worker -and $version -and ($verifier -eq "200")) {
    Write-Host "GO - All checks passed" -ForegroundColor Green
} else {
    Write-Host "NO-GO - Check failed" -ForegroundColor Red
    if (-not $pages) { Write-Host "  FAIL: Pages not immutable" }
    if (-not $worker) { Write-Host "  FAIL: Worker not routing" }
    if (-not $version) { Write-Host "  FAIL: Version mismatch" }
    if ($verifier -ne "200") { Write-Host "  FAIL: Verifier page missing ($verifier)" }
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
- **Cause**: Pages missing `/v0.2.0/evidence-pack/verify/` directory
- **Fix**: Redeploy Pages with `site/v0.2.0/evidence-pack/verify/index.html`

---

## Architecture Reference

```
                                    ┌─────────────────────────┐
                                    │   Cloudflare Pages      │
                                    │   (immutable archive)   │
    /v0.2.0/*  ──────────────────► │   mathledger.ai/v0.2.0/ │
                                    └─────────────────────────┘

    Browser ─────┐
                 │
                 ▼
    ┌────────────────────────┐      ┌─────────────────────────┐
    │   Cloudflare Worker    │      │      Fly.io App         │
    │   (rewrites /demo/*)   │ ───► │   ROOT MOUNT (/)        │
    │   X-Proxied-By header  │      │   v0.2.0-demo-lock      │
    └────────────────────────┘      └─────────────────────────┘
         /demo/*  ──────────────────► /*  (prefix stripped)
```

---

## Contract

| Component | Authoritative Value |
|-----------|---------------------|
| Fly App | `mathledger-demo-v0-2-0-helpfuldolphin` |
| Version | `0.2.0` |
| Tag | `v0.2.0-demo-lock` |
| Commit | `27a94c8a58139cb10349f6418336c618f528cbab` |
| Worker Header | `X-Proxied-By: mathledger-demo-proxy` |
| Pages Cache | `immutable` |

---

**Author**: Claude A
**Ritual**: Run daily before any demo or external review
