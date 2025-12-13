# MathLedger API

FastAPI-based API for MathLedger mathematical verification system.

## Quick Start

1. **Start the server:**
   ```bash
   python start_api_server.py
   ```
   Or manually:
   ```bash
   uv run uvicorn backend.orchestrator.app:app --port 8010
   ```

2. **Test the endpoints:**
   ```bash
   python test_api_endpoints.py
   ```

## API Endpoints

### Authentication
All API endpoints (except UI) require the `X-API-Key` header:
```
X-API-Key: devkey
```

### Endpoints

#### `/metrics` (GET)
Get system metrics including verification attempt success/failure counts, record count, and max depth.

**Response:**
```json
{
  "proofs": {
    "success": 150,
    "failure": 25
  },
  "block_count": 10,
  "max_depth": 5,
  "queue_length": 3
}
```

#### `/records/latest` (GET)
Get the latest record information.

**Response:**
```json
{
  "block_number": 10,
  "merkle_root": "abc123...",
  "created_at": "2024-01-01T12:00:00",
  "header": {
    "run_id": 1,
    "counts": {"statements": 100}
  }
}
```

#### `/statements?hash=<hash>` (GET)
Get a statement by hash with its proofs and parent statements.

**Parameters:**
- `hash`: Statement hash (hex string)

**Response:**
```json
{
  "id": 123,
  "hash": "abc123...",
  "text": "(and p q)",
  "proofs": [
    {
      "id": 456,
      "statement_id": 123,
      "system_id": 1,
      "prover": "lean4",
      "status": "success",
      "duration_ms": 150,
      "created_at": "2024-01-01T12:00:00"
    }
  ],
  "parents": ["def456...", "ghi789..."]
}
```

## UI Endpoints

### `/ui` (GET)
Dashboard with system metrics, depth histogram, and recent statements.

### `/ui/s/<hash>` (GET)
Statement detail page with proofs and parent links.

## Environment Variables

- `LEDGER_API_KEY`: API key for authentication (default: "devkey")
- `CORS_ORIGINS`: Comma-separated list of allowed CORS origins (default: "*")
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

## Testing

Run the test script to verify all endpoints:
```bash
python test_api_endpoints.py
```

## API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8010/docs
- ReDoc: http://localhost:8010/redoc
