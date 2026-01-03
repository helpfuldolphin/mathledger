# Demo Hosting Runbook v0.2.0

**Purpose**: Deploy MathLedger UVIL v0 governance demo as a version-pinned web service.

**Version**: 0.2.0
**Tag**: v0.2.0-demo-lock
**Commit**: 27a94c8a58139cb10349f6418336c618f528cbab

---

## DEMO_ORIGIN

**Hosted Origin URL**: `https://mathledger-demo.fly.dev`

For reverse proxy mounting at `/demo/`:
```
DEMO_ORIGIN=https://mathledger-demo.fly.dev
```

When reverse-proxied at `/demo/`:
- UI loads at: `/demo/`
- Health check: `/demo/healthz` → 200 `ok`
- Health JSON: `/demo/health` → 200 JSON with version info

---

## Overview

The demo is a FastAPI application that serves:
- Interactive governance demo (HTML/JS frontend)
- UVIL API endpoints (`/uvil/*`)
- Documentation viewer (`/docs/view/*`)
- Health endpoints (`/health`, `/healthz`)

---

## Quick Start

### Local Development

```bash
# Run directly
uv run python demo/app.py

# Open http://localhost:8000
```

### Docker (Recommended)

```bash
# Build and run
docker build -t mathledger-demo .
docker run -p 8000:8000 mathledger-demo

# With docker-compose
docker-compose up --build
```

### Mounted at Sub-Path

For reverse proxy mounting at `/demo`:

```bash
docker run -p 8000:8000 -e BASE_PATH=/demo mathledger-demo
```

---

## Fly.io Deployment

### First-Time Setup

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch (creates app, doesn't deploy)
fly launch --copy-config --no-deploy

# Deploy
fly deploy
```

### Subsequent Deploys

```bash
fly deploy
```

### Useful Commands

```bash
# Check status
fly status

# View logs
fly logs

# SSH into container
fly ssh console

# Scale
fly scale count 1

# Destroy
fly apps destroy mathledger-demo
```

---

## PowerShell Deployment & Verification

### Deploy to Fly.io (PowerShell)

```powershell
# First-time setup
fly auth login
fly launch --copy-config --no-deploy
fly deploy

# Subsequent deploys
fly deploy

# Check status
fly status
```

### Verify Headers (PowerShell)

```powershell
# Direct deployment (no BASE_PATH)
$response = Invoke-WebRequest -Uri "https://mathledger-demo.fly.dev/health" -UseBasicParsing
$response.Headers["X-MathLedger-Version"]     # Expected: v0.2.0
$response.Headers["X-MathLedger-Commit"]      # Expected: 27a94c8a58139cb10349f6418336c618f528cbab
$response.Headers["X-MathLedger-Base-Path"]   # Expected: /
$response.Headers["Cache-Control"]            # Expected: no-store, no-cache, must-revalidate

# Verify all headers at once
$headers = @("X-MathLedger-Version", "X-MathLedger-Commit", "X-MathLedger-Base-Path", "Cache-Control")
$headers | ForEach-Object { "$_`: $($response.Headers[$_])" }

# Verify healthz endpoint
(Invoke-WebRequest -Uri "https://mathledger-demo.fly.dev/healthz" -UseBasicParsing).Content
# Expected: ok
```

### Verify When Proxied at /demo/ (PowerShell)

```powershell
# After configuring reverse proxy to mount at /demo/
$baseUrl = "https://your-domain.com/demo"

# Test UI loads
(Invoke-WebRequest -Uri "$baseUrl/" -UseBasicParsing).StatusCode  # Expected: 200

# Test healthz
(Invoke-WebRequest -Uri "$baseUrl/healthz" -UseBasicParsing).Content  # Expected: ok

# Test health with version info
$health = Invoke-RestMethod -Uri "$baseUrl/health"
$health.version     # Expected: 0.2.0
$health.tag         # Expected: v0.2.0-demo-lock
$health.base_path   # Expected: /demo
```

---

## Version Headers

All responses include version and cache headers:

| Header | Value |
|--------|-------|
| `X-MathLedger-Version` | `v0.2.0` |
| `X-MathLedger-Commit` | `27a94c8a58139cb10349f6418336c618f528cbab` |
| `X-MathLedger-Base-Path` | `/` or `/demo` |
| `Cache-Control` | `no-store, no-cache, must-revalidate` |
| `Pragma` | `no-cache` |

---

## Health Endpoints

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/healthz` | K8s liveness | `ok` (plain text, 200) |
| `/health` | Detailed health | `{"status":"ok","version":"0.2.0",...}` |

---

## BASE_PATH Configuration

The `BASE_PATH` environment variable allows mounting the demo behind a reverse proxy at a sub-path.

| Deployment | BASE_PATH | Access URL |
|------------|-----------|------------|
| Direct | `` (empty) | `http://localhost:8000/` |
| Mounted | `/demo` | `https://mathledger.ai/demo/` |

### NGINX Example

```nginx
location /demo/ {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

---

## Security Notes

1. **No secrets**: Demo has no database, no auth, no secrets
2. **Stateless**: All state is in-memory per session
3. **No persistence**: Restarting clears all data
4. **No PII**: No user data collected

---

## Monitoring

### Fly.io Metrics

```bash
fly dashboard
```

### Health Check

```bash
curl -s https://mathledger-demo.fly.dev/health | jq .
```

Expected:
```json
{
  "status": "ok",
  "version": "0.2.0",
  "tag": "v0.2.0-demo-lock",
  "commit": "27a94c8a58139cb10349f6418336c618f528cbab",
  "base_path": "/"
}
```

---

## Troubleshooting

### Container won't start

1. Check logs: `fly logs` or `docker logs <container>`
2. Verify Python dependencies installed
3. Check `PYTHONPATH=/app` is set

### API calls fail with 404

1. Check `BASE_PATH` is set correctly
2. Verify reverse proxy configuration
3. Check browser console for actual request URLs

### Version mismatch

1. Compare header `X-MathLedger-Version` with expected
2. Rebuild container: `docker build --no-cache`
3. Redeploy: `fly deploy`

---

## Contract

This hosting configuration is version-locked to v0.2.0-demo-lock:
- Docker labels include version, tag, commit
- Health endpoint reports version
- All responses include version headers
- UI shows version banner

Any version drift indicates deployment misconfiguration.

---

**Author**: Claude A
**Date**: 2026-01-03
