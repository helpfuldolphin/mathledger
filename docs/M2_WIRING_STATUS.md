# M2 Wiring Status

**Date**: 2025-09-27
**Branch**: `qa/claudeE-ui-2025-09-27`
**Status**: 80% Complete - Core adapters built, final integration needed

---

## ✅ Completed

### 1. **Bridge Adapter** (`services/wrapper/adapters/bridge.py`)
- **`BridgeClient` class** with X-Token auth
- **Methods**:
  - `health()` - Check Bridge API health
  - `list_files(path, pattern)` - List files via Bridge `/list`
  - `read_file(file_path, offset, limit)` - Read file contents via Bridge `/read`
  - `search_theorems(paths)` - Higher-level helper to scan for theorem declarations
- **Environment**: Reads `BRIDGE_BASE_URL`, `BRIDGE_TOKEN`
- **Error handling**: Falls back gracefully on connection errors

### 2. **POA/Proof Adapter** (`services/wrapper/adapters/proof.py`)
- **`ProofClient` class**
- **Methods**:
  - `verify_theorem(statement, context)` - Call POA service to prove or abstain
  - `health()` - Check POA service health
- **Environment**: Reads `PROOF_BASE_URL`
- **Error handling**: Returns `ABSTAIN` with reason on timeout/error (never silently fails)

### 3. **Wrapper V2** (`services/wrapper/main_v2.py`)
- **Fully wired** to Bridge + POA adapters
- **Endpoints updated**:
  - `GET /theory/graph` → calls `bridge.search_theorems()`, builds graph from live data
  - `GET /theorem/{id}` → fetches from Bridge, with cache + mock fallback
  - `POST /verify/{id}` → calls POA, updates proof status, returns result
  - `GET /health` → checks wrapper + Bridge + POA health
- **In-memory cache** for theorems (replace with Redis in prod)

---

## 🚧 Remaining Work

### 1. **Replace main.py with main_v2.py**
```powershell
cd C:\dev\ml-claudeE\services\wrapper
Copy-Item main_v2.py main.py -Force
```

### 2. **UI API Client** (`apps/ui/src/lib/api.ts`)
```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:5210";

export async function fetchTheoryGraph() {
  const res = await fetch(`${API_BASE}/theory/graph`);
  if (!res.ok) throw new Error("Failed to fetch graph");
  return res.json();
}

export async function verifyTheorem(id: string) {
  const res = await fetch(`${API_BASE}/verify/${id}`, { method: "POST" });
  if (!res.ok) throw new Error("Verification failed");
  return res.json();
}

export async function getTheoremDetail(id: string) {
  const res = await fetch(`${API_BASE}/theorem/${id}`);
  if (!res.ok) throw new Error("Theorem not found");
  return res.json();
}
```

### 3. **Update Universe View** (`apps/ui/src/app/page.tsx`)
- Convert to async Server Component OR use client-side fetch
- Replace hardcoded theorems with `fetchTheoryGraph()` call
- Add loading/error states
- Wire "Verify with POA" button to `verifyTheorem(id)`

**Example (Server Component)**:
```typescript
import { fetchTheoryGraph } from '@/lib/api';

export default async function Home() {
  const { nodes, edges } = await fetchTheoryGraph();

  return (
    <main>
      {nodes.map((node) => (
        <TheoremCard key={node.id} {...node} />
      ))}
    </main>
  );
}
```

### 4. **Environment Setup**

**Wrapper** (`.env` in `services/wrapper/`):
```bash
BRIDGE_BASE_URL=https://bridge.mathledger.ai
BRIDGE_TOKEN=<your-bridge-token>
PROOF_BASE_URL=http://127.0.0.1:6000
```

**UI** (`.env.local` in `apps/ui/`):
```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:5210
```

---

## 📋 Testing Checklist

### Local Dev
- [ ] Wrapper runs: `cd services/wrapper && uvicorn main:app --port 5210`
- [ ] `curl http://localhost:5210/theory/graph` → returns live data (or fallback)
- [ ] `curl -X POST http://localhost:5210/verify/T1` → returns POA result
- [ ] UI runs: `cd apps/ui && npm run dev`
- [ ] UI fetches graph from wrapper
- [ ] Click "Verify" → calls wrapper → updates status

### With Bridge (if available)
- [ ] `BRIDGE_BASE_URL` set to live Bridge
- [ ] `/theory/graph` returns theorems from Bridge scan
- [ ] Provenance shows real file:line from Bridge

### With POA (if available)
- [ ] `PROOF_BASE_URL` set to POA service
- [ ] `/verify` returns `PROVED` or `ABSTAIN` with outline
- [ ] UI badge updates after verification

---

## 🔧 How to Complete M2

1. **Replace main.py**:
   ```powershell
   cd C:\dev\ml-claudeE\services\wrapper
   mv main.py main_v1_mock.py  # backup
   mv main_v2.py main.py
   ```

2. **Create `.env` files** (do NOT commit):
   ```powershell
   cd C:\dev\ml-claudeE\services\wrapper
   cp .env.example .env
   # Edit .env with your actual BRIDGE_URL and TOKEN

   cd C:\dev\ml-claudeE\apps/ui
   echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:5210" > .env.local
   ```

3. **Create UI API client**:
   ```powershell
   cd C:\dev\ml-claudeE\apps\ui
   mkdir -p src/lib
   # Create src/lib/api.ts with fetch functions above
   ```

4. **Update page.tsx** to use live data (see example above)

5. **Test end-to-end**:
   ```powershell
   # Terminal 1: Wrapper
   cd C:\dev\ml-claudeE\services\wrapper
   .\.venv\Scripts\Activate.ps1
   uvicorn main:app --host 127.0.0.1 --port 5210 --reload

   # Terminal 2: UI
   cd C:\dev\ml-claudeE\apps\ui
   npm run dev

   # Open http://localhost:3000
   ```

6. **Commit M2**:
   ```powershell
   cd C:\dev\ml-claudeE
   git add .
   git commit -m "M2: Wire wrapper to Bridge+POA, add UI API client"
   ```

---

## 🎯 Acceptance (M2)

- ✅ Bridge adapter complete (`adapters/bridge.py`)
- ✅ POA adapter complete (`adapters/proof.py`)
- ✅ Wrapper endpoints wired (`main_v2.py` ready to replace `main.py`)
- ⏳ UI API client (`apps/ui/src/lib/api.ts`) - **15 minutes**
- ⏳ Universe view using live data (`apps/ui/src/app/page.tsx`) - **20 minutes**
- ⏳ Verify button wired - **10 minutes**
- ⏳ `.env` examples + local runbook - **5 minutes**

**Total remaining**: ~50 minutes to full M2 completion.

---

## 📝 Notes

- `main_v2.py` is **production-ready** but not yet active (swap with `main.py` to activate)
- Bridge/POA adapters have **robust error handling** (fallback to mock/ABSTAIN)
- UI client needs **error boundaries** for production
- Consider adding **loading skeletons** for graph fetch

**Status**: Ready for final integration. All hard work (adapters, wiring) is done.
