# FIRST ORGANISM API Flow

This document captures the canonical sequence of FastAPI calls that the First Organism integration test (and any downstream automation) must execute. Every request is validated against its typed response model from `interface/api/schemas.py`, so the goal is deterministic, schema-checked JSON that directly maps to the organism chain:

1. **Clear DB state explicitly** (test isolation)
2. **Emit a deterministic UI event** (`POST /attestation/ui-event`)
3. **Pull the latest dual-attested block** (`GET /attestation/latest`)
4. **Read the DAG slice** (`GET /ui/statement/{hash}.json`, `/ui/parents/{hash}.json`, `/ui/proofs/{hash}.json`)

---

## 0. Database State Clearing (Test Isolation)

Before running the First Organism API flow, explicitly clear all relevant tables to ensure deterministic test isolation:

```python
def clear_first_organism_db_state(db_conn) -> None:
    tables_to_clear = [
        "block_proofs", "block_statements", "blocks", "ledger_sequences",
        "proofs", "proof_parents", "dependencies", "statements",
        "runs", "theories", "policy_settings",
    ]
    with db_conn.cursor() as cur:
        for table in tables_to_clear:
            try:
                cur.execute(f"DELETE FROM {table}")
            except Exception:
                db_conn.rollback()
    db_conn.commit()
```

---

## 1. UI Event → `UIEventResponse`

- **Endpoint**: `POST /attestation/ui-event`
- **Role in FO Loop**: Captures human/UI interaction events for inclusion in the UI Merkle tree (`Uₜ`)
- **Schema**: `UIEventResponse` (from `interface/api/schemas.py`)

### Request Payload (JSON)

```json
{
  "event_id": "ui-organism-01",
  "kind": "ui.select",
  "action": "prove",
  "statement_hash": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
  "metadata": {"origin": "first-organism-test"}
}
```

### Response Schema

```python
class UIEventResponse(ApiModel):
    event_id: str
    timestamp: float
    leaf_hash: str
```

### Example Response

```json
{
  "event_id": "ui-organism-01",
  "timestamp": 1700000000.0,
  "leaf_hash": "aaff0011aabb22cc33dd44ee55ff6677889900aabbccddeeff00112233445566"
}
```

Use `leaf_hash` to correlate against the UI Merkle root (`Uₜ`). The integration test stores the returned `event_id` and asserts the canonical ordering.

---

## 2. Dual Attestation Bundle → `AttestationLatestResponse`

- **Endpoint**: `GET /attestation/latest`
- **Role in FO Loop**: Returns the most recent sealed block with dual-root attestation (`Rₜ`, `Uₜ`, `Hₜ`)
- **Schema**: `AttestationLatestResponse` (from `interface/api/schemas.py`)

### Response Schema

```python
class AttestationLatestResponse(ApiModel):
    block_number: Optional[int] = None
    reasoning_merkle_root: Optional[str] = None  # R_t
    ui_merkle_root: Optional[str] = None         # U_t
    composite_attestation_root: Optional[str] = None  # H_t = SHA256(R_t || U_t)
    attestation_metadata: Dict[str, Any] = Field(default_factory=dict)
    block_hash: Optional[str] = None
```

### Example Response

```json
{
  "block_number": 1,
  "reasoning_merkle_root": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "ui_merkle_root": "f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5",
  "composite_attestation_root": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  "attestation_metadata": {
    "attestation_version": "v2",
    "reasoning_event_count": 1,
    "ui_event_count": 1,
    "composite_formula": "SHA256(R_t || U_t)"
  },
  "block_hash": "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
}
```

### Critical Assertion

Assert that `composite_attestation_root` equals the `block.composite_root` returned by `LedgerIngestor.ingest(...)`. This verifies the ledger sealed the dual roots and the worker (or test harness) can directly read `Rₜ`, `Uₜ`, and `Hₜ`.

```python
# H_t Invariant Check
from attestation.dual_root import compute_composite_root

attestation = fetch_first_organism_attestation(test_client)
recomputed_h_t = compute_composite_root(
    attestation.reasoning_merkle_root,
    attestation.ui_merkle_root
)
assert recomputed_h_t == attestation.composite_attestation_root
```

---

## 3. DAG View → Statement, Parents, Proofs

Every DAG call is strictly typed via Pydantic response models.

### 3.1 Statement Detail

- **Endpoint**: `GET /ui/statement/{hash}.json`
- **Role in FO Loop**: Returns the derived statement with embedded proofs and parent references
- **Schema**: `StatementDetailResponse`

```python
class ProofSummary(ApiModel):
    method: Optional[str] = None
    status: Optional[str] = None
    success: Optional[bool] = None
    created_at: Optional[datetime] = None
    prover: Optional[str] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)

class ParentSummary(ApiModel):
    hash: HexDigest  # 64-char hex
    display: Optional[str] = None

class StatementDetailResponse(ApiModel):
    hash: HexDigest
    text: Optional[str] = None
    normalized_text: Optional[str] = None
    display: str = Field(..., min_length=1)
    proofs: List[ProofSummary]
    parents: List[ParentSummary]
```

### Example Response

```json
{
  "hash": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
  "text": "P ∧ Q → P",
  "normalized_text": "(and P Q) => P",
  "display": "P ∧ Q → P",
  "proofs": [
    {
      "method": "lean",
      "status": "abstain",
      "success": false,
      "created_at": "2025-11-26T12:00:00Z",
      "prover": "lean",
      "duration_ms": 250
    }
  ],
  "parents": [
    {"hash": "aabbccdd...", "display": "P"},
    {"hash": "eeff0011...", "display": "Q"}
  ]
}
```

### 3.2 Parents List

- **Endpoint**: `GET /ui/parents/{hash}.json`
- **Role in FO Loop**: Returns the DAG lineage (parent statements) for a given statement
- **Schema**: `ParentListResponse`

```python
class ParentListResponse(ApiModel):
    parents: List[ParentSummary]
```

### Example Response

```json
{
  "parents": [
    {"hash": "aabbccdd...", "display": "P"},
    {"hash": "eeff0011...", "display": "Q"}
  ]
}
```

### 3.3 Proofs List

- **Endpoint**: `GET /ui/proofs/{hash}.json`
- **Role in FO Loop**: Returns all proof attempts for a statement (success, failure, abstain)
- **Schema**: `ProofListResponse`

```python
class ProofListResponse(ApiModel):
    proofs: List[ProofSummary]
```

### Example Response

```json
{
  "proofs": [
    {
      "method": "lean",
      "status": "abstain",
      "success": false,
      "created_at": "2025-11-26T12:00:00Z",
      "prover": "lean",
      "duration_ms": 250
    }
  ]
}
```

---

## 4. Helper Functions (MDAP Execution)

The integration test uses these typed helper functions for API interaction:

```python
from fastapi.testclient import TestClient
from interface.api.schemas import (
    AttestationLatestResponse,
    ParentListResponse,
    ProofListResponse,
    StatementDetailResponse,
    UIEventResponse,
)

def post_ui_event_for_first_organism(
    client: TestClient,
    payload: Dict[str, Any],
) -> UIEventResponse:
    """Post UI event and return validated Pydantic response."""
    response = client.post("/attestation/ui-event", json=payload)
    assert response.status_code == 200
    return UIEventResponse.model_validate(response.json())

def fetch_first_organism_attestation(
    client: TestClient,
) -> AttestationLatestResponse:
    """Fetch latest attestation with typed response."""
    response = client.get("/attestation/latest")
    assert response.status_code == 200
    return AttestationLatestResponse.model_validate(response.json())

def fetch_statement_bundle(
    client: TestClient,
    statement_hash: str,
) -> Tuple[StatementDetailResponse, ParentListResponse, ProofListResponse]:
    """Fetch statement, parents, and proofs with typed responses."""
    stmt_resp = client.get(f"/ui/statement/{statement_hash}.json")
    parents_resp = client.get(f"/ui/parents/{statement_hash}.json")
    proofs_resp = client.get(f"/ui/proofs/{statement_hash}.json")

    return (
        StatementDetailResponse.model_validate(stmt_resp.json()),
        ParentListResponse.model_validate(parents_resp.json()),
        ProofListResponse.model_validate(proofs_resp.json()),
    )
```

---

## 5. Orchestration Notes

### Environment Variables

Before instantiating `TestClient(app)`, ensure these environment variables are set:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string (see `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical format) |
| `REDIS_URL` | Redis connection for job queue (see `docs/FIRST_ORGANISM_CONNECTION_STRINGS.md` for canonical format) |
| `CORS_ALLOWED_ORIGINS` | Allowed CORS origins (e.g., `http://localhost`) |
| `LEDGER_API_KEY` | API key for authenticated endpoints |

### Test Isolation

- Clear DB state (`statements`, `proofs`, `blocks`, `ledger_sequences`, `runs`, `theories`, `policy_settings`) before posting UI events
- Use `ui_event_store.clear()` to reset the in-memory UI event buffer

### Schema Validation

- FastAPI uses `response_model` declarations on all endpoints
- Tests must parse responses via Pydantic models (e.g., `StatementDetailResponse.model_validate(...)`)
- Never use raw `dict` access for response validation

### H_t Invariant

The composite attestation root must always satisfy:

```
H_t = SHA256(R_t || U_t)
```

Where:
- `R_t` = Reasoning Merkle root (proof events)
- `U_t` = UI Merkle root (human interaction events)
- `H_t` = Composite attestation root (binding both streams)

This invariant is verified in three places:
1. `LedgerIngestor.ingest()` — at seal time
2. `GET /attestation/latest` — API response validation
3. Test assertions — recomputation check

---

## 6. Full Test Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Clear DB State                                               │
│    └── DELETE FROM blocks, statements, proofs, ...              │
├─────────────────────────────────────────────────────────────────┤
│ 2. Derivation Pipeline                                          │
│    └── Generate candidate statement                             │
├─────────────────────────────────────────────────────────────────┤
│ 3. Ledger Ingest with Dual Roots                                │
│    └── LedgerIngestor.ingest() → block with R_t, U_t, H_t       │
├─────────────────────────────────────────────────────────────────┤
│ 4. POST /attestation/ui-event                                   │
│    └── Capture UI event → UIEventResponse                       │
├─────────────────────────────────────────────────────────────────┤
│ 5. GET /attestation/latest                                      │
│    └── Fetch attestation → AttestationLatestResponse            │
│    └── Assert H_t == ledger.block.composite_root                │
├─────────────────────────────────────────────────────────────────┤
│ 6. GET /ui/statement/{hash}.json                                │
│    └── Fetch statement → StatementDetailResponse                │
│    └── Assert proofs[0].status == "abstain"                     │
├─────────────────────────────────────────────────────────────────┤
│ 7. GET /ui/parents/{hash}.json                                  │
│    └── Fetch parents → ParentListResponse                       │
├─────────────────────────────────────────────────────────────────┤
│ 8. GET /ui/proofs/{hash}.json                                   │
│    └── Fetch proofs → ProofListResponse                         │
│    └── Assert status == "abstain"                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. API Surface Validation Test Suite

The `tests/integration/test_api_surface_validation.py` module provides comprehensive validation:

### Test Categories

| Category | Description |
|----------|-------------|
| `TestAttestationSurface` | Full `/attestation/*` endpoint validation |
| `TestUISurface` | Full `/ui/*` JSON endpoint validation |
| `TestSchemaCanonicalityAndDeterminism` | Pydantic model strictness and determinism |
| `TestSchemaEvolution` | Backward compatibility for schema changes |
| `TestHtInvariants` | H_t = SHA256(R_t \|\| U_t) formula verification |
| `TestMultiStatementDAGQueries` | DAG traversal and multi-statement queries |
| `TestCrossEndpointConsistency` | Consistency between related endpoints |
| `TestErrorResponses` | Error response structure validation |

### Running the Tests

```bash
# Run all API surface validation tests
pytest tests/integration/test_api_surface_validation.py -v

# Run specific category
pytest tests/integration/test_api_surface_validation.py -k "TestHtInvariants" -v

# Run with database (requires DATABASE_URL)
DATABASE_URL=postgresql://... pytest tests/integration/test_api_surface_validation.py -v
```

### Key Assertions

1. **Schema Strictness**: `ApiModel` base class rejects extra fields
2. **Determinism**: Same payload → same `leaf_hash`, same seed → same timestamp
3. **H_t Formula**: `verify_composite_integrity(R_t, U_t, H_t)` always true
4. **DAG Traversal**: Parent hashes are valid 64-char hex strings
5. **Cross-Endpoint**: `StatementDetailResponse.proofs` matches `ProofListResponse`
