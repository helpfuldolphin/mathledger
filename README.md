# MathLedger UI & Wrapper (Claude E)

**Google Maps for Math & Truth**

This branch (`qa/claudeE-ui-2025-09-27`) contains the **UI/UX layer** and **API wrapper** for MathLedger.

---

## Structure

```
ml-claudeE/
├── apps/
│   └── ui/                  # Next.js app (Universe view, Factory panel, Theorem cards)
│       ├── src/
│       │   └── app/
│       │       └── page.tsx # Main Universe view
│       └── package.json
│
├── services/
│   └── wrapper/             # FastAPI wrapper (UI-safe endpoints)
│       ├── main.py          # /theory/graph, /theorem/:id, /verify, /factory/agents
│       ├── requirements.txt
│       └── .env.example
│
└── docs/
    └── edge_setup.md        # Cloudflare + GoDaddy + Tunnel setup for mathledger.ai
```

---

## Quick Start

### UI (Next.js)

```powershell
cd apps/ui
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the **Universe View**.

### Wrapper (FastAPI)

```powershell
cd services/wrapper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 5210 --reload
```

Visit [http://127.0.0.1:5210/docs](http://127.0.0.1:5210/docs) for the API docs (Swagger).

---

## Key Features

### UI Components

- **Universe View** (zoomable graph placeholder)
- **Factory Floor Panel** (live agent lanes: Cursor A, Replit A, Grok A, Gemini A)
- **Theorem Cards** with ProofBadge (PROVED/PENDING/ABSTAIN)
- **"Verify with POA" Button** (routes to `/verify` endpoint)
- **Footer status badges** (UI ✓, API —, Bridge —)

### API Endpoints (Wrapper)

| Endpoint | Description |
|----------|-------------|
| `GET /` | Service status |
| `GET /theory/graph` | Zoomable theory graph (nodes + edges) |
| `GET /theorem/{id}` | Theorem detail (statement, deps, source) |
| `POST /verify/{id}` | Endpoint to request verification from the POA service (stub for now) |
| `GET /factory/agents` | Agent status from ledger (stub for now) |

---

## Grounding Policy

**Claude E can speculate** (e.g., suggest UI copy, propose explorations), but:

1. All **speculative content** must be labeled as `SPECULATIVE`
2. For **truth claims**, the wrapper requests verification from the POA service:
   - Status: `PROVED` (green) or `ABSTAIN` (grey)
   - Do not upgrade `ABSTAIN` to `PROVED`
3. Every theorem displays **provenance** (source file, line, verification status)
4. **"Verify with POA"** button routes to `/verify/{id}` → sends request to the verification service

---

## Domain Setup

See **[docs/edge_setup.md](docs/edge_setup.md)** for full instructions on wiring:

- **mathledger.ai** (Cloudflare Pages for UI)
- **api.mathledger.ai** (Cloudflare Tunnel → HP:5210 wrapper)
- **bridge.mathledger.ai** (Cloudflare Tunnel → HP:5055 Bridge API)

---

## Handshakes

- **Manus A** (Conductor): Switch Bridge connector to `https://bridge.mathledger.ai` (stable URL)
- **Claude D** (Integrator): Merge this branch when green (UI builds, wrapper 200/OK)
- **POA/Verification**: Future integration for `/verify` endpoint

---

## Next Steps

1. ✅ Scaffold UI + wrapper with mock data
2. ✅ Document domain/edge setup (Cloudflare + Tunnel)
3. 🔜 Wire UI to consume `api.mathledger.ai/theory/graph`
4. 🔜 Wire wrapper to Bridge API (read-only, X-Token auth)
5. 🔜 Integrate POA/Verification service for `/verify`
6. 🔜 Add live Factory panel (read `agent_ledger.jsonl`)

---

## Acceptance Criteria (M1 - 48h)

- [ ] `npm run dev` launches Universe view with mock graph
- [ ] `uvicorn main:app` returns 200 on `/theory/graph`
- [ ] UI displays ProofBadge + "Verify with POA" button
- [ ] Speculative content labeled; Verification route wired (stub OK)
- [ ] Footer shows "Connected to mathledger.ai"

---

**Owner**: Claude E (UI/UX + Wrapper Architect)
**Branch**: `qa/claudeE-ui-2025-09-27`
**Date**: 2025-09-27
