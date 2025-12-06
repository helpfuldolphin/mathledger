# Domain & Edge Setup: mathledger.ai

**Owner**: Claude E (UI/UX + Wrapper)
**Last updated**: 2025-09-27

This document describes how to wire **mathledger.ai** (GoDaddy registrar) through **Cloudflare** to serve:
- The Next.js UI (Cloudflare Pages)
- The FastAPI wrapper (Cloudflare Tunnel from HP)
- The Bridge API (Cloudflare Tunnel from HP)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ mathledger.ai (GoDaddy registrar)                          │
│   ↓ DNS delegated to Cloudflare nameservers                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Cloudflare (DNS + Pages + Tunnels + WAF)                   │
│                                                             │
│  https://mathledger.ai           → Cloudflare Pages (UI)   │
│  https://staging.mathledger.ai   → Cloudflare Pages (staging)
│  https://api.mathledger.ai       → Tunnel → HP:5210 (wrapper)
│  https://bridge.mathledger.ai    → Tunnel → HP:5055 (bridge)
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ HP Machine (Windows)                                        │
│  - FastAPI wrapper on 127.0.0.1:5210                        │
│  - Bridge API on 127.0.0.1:5055                             │
│  - cloudflared tunnel service (named: mathledger-hp)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Add mathledger.ai to Cloudflare

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com) → **Add a Site** → `mathledger.ai`
2. Choose **Free** plan (or Pro if needed)
3. Cloudflare scans existing DNS; you'll replace it in the next step

---

## Step 2: Update GoDaddy Nameservers

1. Log in to [GoDaddy](https://godaddy.com) → My Domains → `mathledger.ai` → **DNS Settings**
2. Under **Nameservers**, click **Change** → **Use custom nameservers**
3. Paste the two Cloudflare nameservers shown in step 1 (e.g., `ben.ns.cloudflare.com`, `lucy.ns.cloudflare.com`)
4. **Save**. Propagation takes a few minutes to an hour.

---

## Step 3: Configure Cloudflare Pages for UI

### Deploy Next.js to Cloudflare Pages

1. In Cloudflare → **Pages** → **Create Project** → **Connect to Git**
2. Select your GitHub repo (`helpfuldolphin/mathledger`)
3. **Build settings**:
   - **Framework**: Next.js
   - **Build command**: `cd apps/ui && npm install && npm run build`
   - **Output directory**: `apps/ui/.next`
     _(Cloudflare has a Next.js preset—use that if prompted)_
4. Click **Save and Deploy**

### Custom Domains

1. Pages → Your Project → **Custom Domains** → **Add domain**
2. Add `mathledger.ai` (apex) → Cloudflare auto-creates DNS `CNAME` or `A` + `AAAA` records
3. (Optional) Add `staging.mathledger.ai` → map to a staging branch

Pages is now live at `https://mathledger.ai`!

---

## Step 4: Create Cloudflare Tunnel for API & Bridge (on HP)

### Install `cloudflared` (if not already)

Download from [Cloudflare Tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/) or:

```powershell
# Windows (PowerShell, admin mode)
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "C:\Windows\System32\cloudflared.exe"
```

### Login & Create Named Tunnel

```powershell
# Authenticate with Cloudflare
cloudflared tunnel login

# Create a named tunnel
cloudflared tunnel create mathledger-hp
# → Saves credentials to C:\Users\<you>\.cloudflared\<tunnel-id>.json
```

### Create Tunnel Config

**File**: `C:\Users\<you>\.cloudflared\config.yml`

```yaml
tunnel: mathledger-hp
credentials-file: C:\Users\<you>\.cloudflared\<tunnel-id>.json

ingress:
  # API wrapper (FastAPI on HP, port 5210)
  - hostname: api.mathledger.ai
    service: http://127.0.0.1:5210

  # Bridge API (Manus conduit on HP, port 5055)
  - hostname: bridge.mathledger.ai
    service: http://127.0.0.1:5055

  # Fallback (404)
  - service: http_status:404
```

Replace `<tunnel-id>` with the UUID from the tunnel creation step.

### Route DNS to Tunnel

```powershell
cloudflared tunnel route dns mathledger-hp api.mathledger.ai
cloudflared tunnel route dns mathledger-hp bridge.mathledger.ai
```

This creates **proxied CNAMEs** in Cloudflare DNS automatically.

### Run Tunnel as a Windows Service

```powershell
# Install as service
cloudflared service install

# Start service
net start cloudflared

# Or run foreground (for testing):
# cloudflared tunnel run mathledger-hp
```

---

## Step 5: SSL/TLS & Security in Cloudflare

1. **SSL/TLS** → **Overview** → Set to **Full (strict)**
2. **SSL/TLS** → **Edge Certificates** → Enable:
   - Always Use HTTPS: **On**
   - HTTP Strict Transport Security (HSTS): **On** (after confirming all subdomains are HTTPS)
3. **Security** → **WAF** → Add rate limiting rules:
   - `api.mathledger.ai`: 60 requests / 60 seconds per IP
   - `bridge.mathledger.ai`: 30 requests / 60 seconds per IP (tighter, since it's internal)
4. **DNS** → Add redirect rule:
   - `www.mathledger.ai` → `https://mathledger.ai` (301)

---

## Step 6: Verify Everything Works

### UI

```bash
curl -I https://mathledger.ai
# → 200 OK, HTML from Cloudflare Pages
```

### API Wrapper

```bash
curl -sS https://api.mathledger.ai/theory/graph
# → JSON with nodes/edges
```

### Bridge API

```bash
curl -sS -H "X-Token: <your-bridge-token>" https://bridge.mathledger.ai/health
# → {"status": "ok"}
```

---

## Environment Variables

### Wrapper (`services/wrapper/.env`)

```bash
BRIDGE_BASE_URL=https://bridge.mathledger.ai
BRIDGE_TOKEN=your-bridge-token-here
PROOF_BASE_URL=http://127.0.0.1:8080
```

### UI (Next.js)

Add to `apps/ui/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=https://api.mathledger.ai
```

---

## Ownership & Handshakes

- **Claude E**: Owns this doc, tunnel config template, UI env setup
- **Manus A**: Switches Bridge connector to use `https://bridge.mathledger.ai` (stable URL, no more random trycloudflare)
- **Claude D**: Integrates this PR into `main` after green gates

---

## Optional: Staging & Zero Trust

### Staging

Map `staging.mathledger.ai` to Pages' staging branch (e.g., `qa/*` branches) for feature demos.

### Zero Trust (Private Bridge)

If the Bridge should be private:

1. Cloudflare → **Zero Trust** → **Access** → **Add an Application**
2. Set `bridge.mathledger.ai` behind SSO (e.g., Google Workspace, GitHub)
3. Only your team can reach the Bridge API

---

## Troubleshooting

**DNS not resolving?**

- Check nameservers at GoDaddy: `nslookup mathledger.ai` should return Cloudflare IPs
- Cloudflare → **DNS** → Ensure records are **Proxied** (orange cloud)

**Tunnel not connecting?**

- Check service status: `Get-Service cloudflared`
- Check logs: `C:\ProgramData\cloudflared\logs\`
- Verify tunnel config: `cloudflared tunnel info mathledger-hp`

**API 502 Bad Gateway?**

- Ensure wrapper is running on `127.0.0.1:5210`
- Check Cloudflare → **SSL/TLS** → Set to **Full (strict)** if origin has valid cert, or **Full** if self-signed

---

## Next Steps

1. ✅ UI at `mathledger.ai` (Cloudflare Pages)
2. ✅ Wrapper at `api.mathledger.ai` (Tunnel)
3. ✅ Bridge at `bridge.mathledger.ai` (Tunnel)
4. 🔜 Wire UI to consume `api.mathledger.ai/theory/graph` (next PR)
5. 🔜 Wire wrapper to Bridge API with X-Token auth
6. 🔜 Add POA/Proof service integration (`/verify`)

---

**Questions?** Open an issue or ask Manus/Claude D!
