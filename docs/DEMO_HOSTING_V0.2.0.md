# Demo Hosting Runbook v0.2.0

**Purpose**: Deploy MathLedger UVIL v0 governance demo as a version-pinned web service.

**Version**: 0.2.0
**Tag**: v0.2.0-demo-lock
**Commit**: 27a94c8a58139cb10349f6418336c618f528cbab

---

## Architecture: ROOT MOUNT (Option B)

**IMPORTANT**: The demo app serves at ROOT (`/`), NOT at `/demo`.

Cloudflare Worker MUST rewrite `/demo/*` → `/*` before forwarding to origin.

```
┌─────────────┐     /demo/healthz     ┌──────────────────┐     /healthz     ┌─────────────────┐
│   Browser   │ ──────────────────► │ Cloudflare Worker │ ──────────────► │   Fly.io App    │
└─────────────┘                       └──────────────────┘                  └─────────────────┘
                                       strips /demo prefix                  serves at /
```

---

## DEMO_ORIGIN (Canonical)

```
DEMO_ORIGIN=https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev
```

**Fly App Name**: `mathledger-demo-v0-2-0-helpfuldolphin`

| Endpoint at Origin | Response |
|--------------------|----------|
| `/` | 200 HTML (UI) |
| `/healthz` | 200 `ok` |
| `/health` | 200 JSON |
| `/demo/healthz` | **404** (app does NOT serve /demo/*) |

---

## X-MathLedger-Base-Path Header

The header is **pinned to "/"** because the app serves at root.

| Header | Pinned Value |
|--------|--------------|
| `X-MathLedger-Version` | `v0.2.0` |
| `X-MathLedger-Commit` | `27a94c8a58139cb10349f6418336c618f528cbab` |
| `X-MathLedger-Base-Path` | `/` (always, never `/demo`) |
| `Cache-Control` | `no-store, no-cache, must-revalidate` |

---

## Cloudflare Worker Requirements

The Cloudflare Worker at `mathledger.ai` MUST:

1. Match requests to `/demo/*`
2. Strip the `/demo` prefix before forwarding to origin
3. Forward to `https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev`

**Example Worker Logic**:
```javascript
// Cloudflare Worker
export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname.startsWith('/demo')) {
      // Strip /demo prefix
      const originPath = url.pathname.replace(/^\/demo/, '') || '/';
      const originUrl = `https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev${originPath}${url.search}`;
      return fetch(originUrl, request);
    }

    // ... other routes
  }
}
```

---

## PowerShell Verification Commands

### Verify Origin Directly (curl.exe)

```powershell
# Health check at origin root
curl.exe -s "https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev/healthz"
# Expected: ok

# Health JSON at origin root
curl.exe -s "https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev/health"
# Expected: {"status":"ok","version":"0.2.0","tag":"v0.2.0-demo-lock",...}

# Verify headers
curl.exe -I "https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev/health"
# Expected headers:
#   X-MathLedger-Version: v0.2.0
#   X-MathLedger-Commit: 27a94c8a58139cb10349f6418336c618f528cbab
#   X-MathLedger-Base-Path: /
#   Cache-Control: no-store, no-cache, must-revalidate

# Confirm /demo/* returns 404 at origin (this is CORRECT behavior)
curl.exe -s -o nul -w "%{http_code}" "https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev/demo/healthz"
# Expected: 404
```

### Verify via Cloudflare (after Worker deployed)

```powershell
# After Cloudflare Worker is configured to rewrite /demo/* -> /*
curl.exe -s "https://mathledger.ai/demo/healthz"
# Expected: ok

curl.exe -s "https://mathledger.ai/demo/health"
# Expected: {"status":"ok","version":"0.2.0",...}

curl.exe -I "https://mathledger.ai/demo/health"
# Headers should still show X-MathLedger-Base-Path: / (from origin)
```

### PowerShell Native Commands

```powershell
# Verify origin health
$r = Invoke-WebRequest -Uri "https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev/health" -UseBasicParsing
$r.Headers["X-MathLedger-Version"]     # v0.2.0
$r.Headers["X-MathLedger-Commit"]      # 27a94c8a58139cb10349f6418336c618f528cbab
$r.Headers["X-MathLedger-Base-Path"]   # /  (ALWAYS "/", never "/demo")
$r.Headers["Cache-Control"]            # no-store, no-cache, must-revalidate

# Verify healthz
(Invoke-WebRequest -Uri "https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev/healthz" -UseBasicParsing).Content
# ok
```

---

## Fly.io Deployment

### fly.toml Configuration

```toml
app = "mathledger-demo-v0-2-0-helpfuldolphin"
primary_region = "iad"

[env]
  BASE_PATH = ""          # MUST be empty (root mount)
  PYTHONUNBUFFERED = "1"
```

### Deploy Commands

```powershell
# Check current app status
fly status -a mathledger-demo-v0-2-0-helpfuldolphin

# List all apps
fly apps list

# Deploy
fly deploy

# View logs
fly logs -a mathledger-demo-v0-2-0-helpfuldolphin
```

---

## Local Development

```bash
# Run at root (default)
uv run python demo/app.py
# Access: http://localhost:8000/

# Docker
docker build -t mathledger-demo .
docker run -p 8000:8000 mathledger-demo
# Access: http://localhost:8000/
```

**Note**: For local development with `/demo` prefix, use a reverse proxy (nginx) that strips the prefix, OR set `BASE_PATH=/demo` for testing. Production Fly deployment always uses root mount.

---

## Health Endpoints

| Origin Path | Response |
|-------------|----------|
| `/healthz` | `ok` (plain text, 200) |
| `/health` | `{"status":"ok","version":"0.2.0","tag":"v0.2.0-demo-lock","commit":"27a94c8a...","base_path":"/"}` |

---

## Why Root Mount?

1. **Simpler deployment**: App doesn't need to know about `/demo` prefix
2. **Cleaner separation**: Cloudflare handles routing, Fly handles serving
3. **Flexible**: Can mount at any path by changing Worker, not app
4. **Already deployed**: Current Fly app works this way

---

## Smoke Checklist

| Check | Command | Expected |
|-------|---------|----------|
| Fly app exists | `fly apps list` | Shows `mathledger-demo-v0-2-0-helpfuldolphin` |
| Fly status | `fly status -a mathledger-demo-v0-2-0-helpfuldolphin` | Running |
| Origin healthz | `curl.exe https://...fly.dev/healthz` | `ok` |
| Origin health | `curl.exe https://...fly.dev/health` | JSON with version |
| Version header | `curl.exe -I .../health` | `X-MathLedger-Version: v0.2.0` |
| Commit header | `curl.exe -I .../health` | `X-MathLedger-Commit: 27a94c8a...` |
| Base-Path header | `curl.exe -I .../health` | `X-MathLedger-Base-Path: /` |
| /demo/* at origin | `curl.exe .../demo/healthz` | **404** (correct!) |
| CF Worker /demo/healthz | `curl.exe https://mathledger.ai/demo/healthz` | `ok` (after Worker) |

---

## Contract

- **Fly app**: `mathledger-demo-v0-2-0-helpfuldolphin`
- **Origin URL**: `https://mathledger-demo-v0-2-0-helpfuldolphin.fly.dev`
- **BASE_PATH**: `""` (empty, root mount)
- **X-MathLedger-Base-Path**: Always `/`
- **Cloudflare**: MUST rewrite `/demo/*` → `/*`

Any deviation indicates misconfiguration.

---

**Author**: Claude A
**Date**: 2026-01-03
**Architecture**: ROOT MOUNT (Option B)
