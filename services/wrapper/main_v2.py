"""MathLedger UI Wrapper - FastAPI service for UI-safe endpoints (M2 wired version)."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from adapters.bridge import get_bridge_client
from adapters.proof import get_proof_client

app = FastAPI(
    title="MathLedger UI Wrapper",
    description="API wrapper for mathledger.ai - Google Maps for Math & Truth",
    version="0.2.0"
)

# CORS - allow UI to call from different origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://mathledger.ai", "https://staging.mathledger.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Node(BaseModel):
    id: str
    label: str
    proof_status: str
    source: str = ""

class Edge(BaseModel):
    src: str
    dst: str
    type: str

class TheoryGraph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]

class TheoremDetail(BaseModel):
    id: str
    label: str
    statement: str
    proof_status: str
    dependencies: list[str]
    source_file: str = ""
    source_line: int = 0

class VerifyResult(BaseModel):
    id: str
    status: str  # PROVED | ABSTAIN
    outline: str = ""
    source_refs: list[str] = []

class AgentStatus(BaseModel):
    name: str
    branch: str
    status: str  # ACTIVE | IDLE | ERROR
    pr_url: str = ""
    last_commit: str = ""

# In-memory theorem cache (for demo; replace with Redis in prod)
_theorem_cache = {}

# Routes
@app.get("/")
def root():
    return {"service": "MathLedger UI Wrapper", "status": "operational", "version": "0.2.0"}

@app.get("/theory/graph", response_model=TheoryGraph)
async def theory_graph():
    """Return the zoomable theory graph from live Bridge data."""
    bridge = get_bridge_client()

    try:
        # Search for theorems via Bridge
        theorems = await bridge.search_theorems()

        if not theorems:
            # Fallback to mock if Bridge is unavailable or returns no data
            return TheoryGraph(
                nodes=[
                    Node(id="T_FALLBACK", label="Bridge unavailable (mock fallback)", proof_status="PENDING", source=""),
                ],
                edges=[]
            )

        # Cache theorems for /theorem/{id} endpoint
        for t in theorems:
            _theorem_cache[t["id"]] = t

        # Convert to nodes
        nodes = [
            Node(
                id=t["id"],
                label=t["label"],
                proof_status=t.get("proof_status", "PENDING"),
                source=f"{t['file']}:{t['line']}"
            )
            for t in theorems
        ]

        # Generate dependency edges (simple heuristic: sequential dependencies)
        edges = []
        for i in range(len(theorems) - 1):
            edges.append(Edge(
                src=theorems[i]["id"],
                dst=theorems[i + 1]["id"],
                type="depends_on"
            ))

        return TheoryGraph(nodes=nodes, edges=edges)

    except Exception as e:
        # Fallback on error
        return TheoryGraph(
            nodes=[
                Node(id="ERROR", label=f"Bridge error: {str(e)[:50]}", proof_status="ABSTAIN", source=""),
            ],
            edges=[]
        )

@app.get("/theorem/{theorem_id}", response_model=TheoremDetail)
async def theorem_detail(theorem_id: str):
    """Get detailed info for a specific theorem from Bridge."""
    bridge = get_bridge_client()

    # Try cache first
    if theorem_id in _theorem_cache:
        t = _theorem_cache[theorem_id]
        return TheoremDetail(
            id=t["id"],
            label=t["label"],
            statement=t["statement"],
            proof_status=t.get("proof_status", "PENDING"),
            dependencies=[],  # TODO: parse from file content
            source_file=t["file"],
            source_line=t["line"]
        )

    # Fallback to mock for known IDs
    if theorem_id == "T1":
        return TheoremDetail(
            id="T1",
            label="Modus Ponens",
            statement="(P → Q) → P → Q",
            proof_status="PROVED",
            dependencies=[],
            source_file="lean4/Logic/Basic.lean",
            source_line=42
        )
    elif theorem_id == "T2":
        return TheoremDetail(
            id="T2",
            label="Export Linter Soundness",
            statement="All exports match ASCII-only file patterns",
            proof_status="PENDING",
            dependencies=["T1"],
            source_file="python/exporter/linter.py",
            source_line=128
        )

    # Not found
    raise HTTPException(status_code=404, detail=f"Theorem {theorem_id} not found")

@app.post("/verify/{theorem_id}", response_model=VerifyResult)
async def verify_theorem(theorem_id: str):
    """Call POA/Proof service to verify a theorem."""
    proof = get_proof_client()

    # Get theorem statement
    try:
        theorem = await theorem_detail(theorem_id)
        statement = theorem.statement
    except HTTPException:
        return VerifyResult(
            id=theorem_id,
            status="ABSTAIN",
            outline=f"Theorem {theorem_id} not found in system",
            source_refs=[]
        )

    # Call POA
    result = await proof.verify_theorem(statement, context={"id": theorem_id})

    # Update cache if theorem was proved
    if result["status"] == "PROVED" and theorem_id in _theorem_cache:
        _theorem_cache[theorem_id]["proof_status"] = "PROVED"

    return VerifyResult(
        id=theorem_id,
        status=result["status"],
        outline=result["outline"],
        source_refs=result["source_refs"]
    )

@app.get("/factory/agents", response_model=list[AgentStatus])
def factory_agents():
    """Return agent status from ledger."""
    # TODO: read from docs/progress/agent_ledger.jsonl via Bridge
    # For now, return mock data
    return [
        AgentStatus(name="Cursor A", branch="qa/cursorA-fol-2025-09-25", status="ACTIVE", last_commit="Add FOL one-pager"),
        AgentStatus(name="Replit A", branch="qa/replitA-export-2025-09-26", status="IDLE", last_commit=""),
        AgentStatus(name="Grok A", branch="qa/grokA-derives-2025-09-27", status="ACTIVE", last_commit="Fix derives_id"),
        AgentStatus(name="Gemini A", branch="qa/geminiA-bridge-2025-09-27", status="ACTIVE", last_commit="Bridge health check"),
    ]

@app.get("/health")
async def health_check():
    """Combined health check for wrapper + dependencies."""
    bridge = get_bridge_client()
    proof = get_proof_client()

    bridge_health = "unknown"
    proof_health = "unknown"

    try:
        await bridge.health()
        bridge_health = "ok"
    except Exception:
        bridge_health = "unavailable"

    try:
        await proof.health()
        proof_health = "ok"
    except Exception:
        proof_health = "unavailable"

    return {
        "wrapper": "ok",
        "bridge": bridge_health,
        "proof": proof_health
    }
