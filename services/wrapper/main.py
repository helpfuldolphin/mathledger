"""MathLedger UI Wrapper - FastAPI service for UI-safe endpoints."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="MathLedger UI Wrapper",
    description="API wrapper for mathledger.ai - Google Maps for Math & Truth",
    version="0.1.0"
)

# CORS - allow UI to call from different origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://mathledger.ai", "https://staging.mathledger.ai"],
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

# Routes
@app.get("/")
def root():
    return {"service": "MathLedger UI Wrapper", "status": "operational"}

@app.get("/theory/graph", response_model=TheoryGraph)
def theory_graph():
    """Return the zoomable theory graph (mock data for now)."""
    return TheoryGraph(
        nodes=[
            Node(id="T1", label="Modus Ponens", proof_status="PROVED", source="lean4/Logic/Basic.lean:42"),
            Node(id="T2", label="Export Linter Soundness", proof_status="PENDING", source="python/exporter/linter.py:128"),
            Node(id="T3", label="Derives Identity Coherence", proof_status="ABSTAIN", source="python/derives_id.py:56"),
        ],
        edges=[
            Edge(src="T1", dst="T2", type="depends_on"),
            Edge(src="T2", dst="T3", type="leads_to"),
        ]
    )

@app.get("/theorem/{theorem_id}", response_model=TheoremDetail)
def theorem_detail(theorem_id: str):
    """Get detailed info for a specific theorem."""
    # Mock data
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
            dependencies=["T1", "T3"],
            source_file="python/exporter/linter.py",
            source_line=128
        )
    return TheoremDetail(
        id=theorem_id,
        label="Unknown Theorem",
        statement="Not found",
        proof_status="ABSTAIN",
        dependencies=[]
    )

@app.post("/verify/{theorem_id}", response_model=VerifyResult)
def verify_theorem(theorem_id: str):
    """Call POA/Proof service to verify a theorem (stub for now)."""
    # TODO: wire to actual Proof service
    return VerifyResult(
        id=theorem_id,
        status="ABSTAIN",
        outline="POA service not yet connected",
        source_refs=[]
    )

@app.get("/factory/agents", response_model=list[AgentStatus])
def factory_agents():
    """Return agent status from ledger (stub for now)."""
    # TODO: read from docs/progress/agent_ledger.jsonl
    return [
        AgentStatus(name="Cursor A", branch="qa/cursorA-fol-2025-09-25", status="ACTIVE", last_commit="Add FOL one-pager"),
        AgentStatus(name="Replit A", branch="qa/replitA-export-2025-09-26", status="IDLE", last_commit=""),
        AgentStatus(name="Grok A", branch="qa/grokA-derives-2025-09-27", status="ACTIVE", last_commit="Fix derives_id"),
        AgentStatus(name="Gemini A", branch="qa/geminiA-bridge-2025-09-27", status="ACTIVE", last_commit="Bridge health check"),
    ]
