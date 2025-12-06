# MathLedger API Reference

## Overview

The MathLedger API provides programmatic access to mathematical proof system data, including statements, proofs, blocks, and system metrics. The API is built with FastAPI and provides both REST endpoints and a web-based UI.

## Base URL

```
http://localhost:8010
```

## Authentication

All API endpoints (except UI and health check) require authentication via the `X-API-Key` header:

```http
X-API-Key: devkey
```

## Endpoints

### System Metrics

#### `GET /metrics`

Get comprehensive system metrics including proof statistics, block information, and system health.

**Headers:**
- `X-API-Key`: API key for authentication

**Response:**
```json
{
  "proofs": {
    "success": 150,
    "failure": 25
  },
  "proofs_by_prover": {
    "lean4": 170,
    "z3": 5
  },
  "proofs_by_method": {
    "tactics": 120,
    "smt": 30,
    "manual": 25
  },
  "block_count": 10,
  "max_depth": 5,
  "queue_length": 3,
  "statements_by_status": {
    "proven": 120,
    "disproven": 5,
    "open": 30,
    "unknown": 20
  },
  "derivation_rules": {
    "modus_ponens": 45,
    "conjunction_intro": 30,
    "disjunction_intro": 25
  },
  "recent_activity": {
    "proofs_last_hour": 12,
    "proofs_last_day": 89
  }
}
```

### Block Information

#### `GET /blocks/latest`

Get the latest block information.

**Headers:**
- `X-API-Key`: API key for authentication

**Query Parameters:**
- `system` (optional): System slug to filter by (e.g., 'pl')

**Response:**
```json
{
  "block_number": 10,
  "merkle_root": "abc123def456...",
  "created_at": "2024-01-01T12:00:00",
  "header": {
    "run_name": "nightly_run_20240101",
    "statements": ["hash1", "hash2", "hash3"],
    "metadata": {
      "statements_count": 3,
      "run_id": 1
    }
  }
}
```

### Statement Information

#### `GET /statements`

Get a statement by hash or text content.

**Headers:**
- `X-API-Key`: API key for authentication

**Query Parameters:**
- `hash` (optional): Statement hash (hex string)
- `text` (optional): Statement text content (normalized)

**Note:** Provide either `hash` or `text`, not both.

**Response:**
```json
{
  "id": 123,
  "hash": "abc123def456...",
  "text": "(and p q)",
  "status": "proven",
  "derivation_rule": "modus_ponens",
  "derivation_depth": 2,
  "proofs": [
    {
      "id": 456,
      "statement_id": 123,
      "system_id": 1,
      "prover": "lean4",
      "method": "tactics",
      "status": "success",
      "duration_ms": 150,
      "created_at": "2024-01-01T12:00:00"
    }
  ],
  "parents": ["def456ghi789...", "ghi789jkl012..."]
}
```

### UI Endpoints

#### `GET /ui`

MathLedger Dashboard - Web-based interface showing system metrics, proof statistics, and recent statements.

**No authentication required.**

#### `GET /ui/s/{statement_hash}`

Statement Detail Page - Web-based interface for viewing a specific statement with its proofs and parent statements.

**Parameters:**
- `statement_hash`: Hex string hash of the statement

**No authentication required.**

### Health Check

#### `GET /health`

Simple health check endpoint.

**Response:**
```json
{
  "ok": true,
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Either 'hash' or 'text' parameter must be provided"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid API key"
}
```

### 404 Not Found
```json
{
  "detail": "Statement with hash 'abc123...' not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Database error: connection timeout expired"
}
```

## Data Models

### Statement

A mathematical statement with the following properties:

- `id`: Unique integer identifier
- `hash`: SHA-256 hash of the normalized statement (hex string)
- `text`: The statement content in normalized form
- `status`: Current status (`proven`, `disproven`, `open`, `unknown`)
- `derivation_rule`: Rule used to derive this statement (if applicable)
- `derivation_depth`: Depth in the derivation tree
- `created_at`: Timestamp when the statement was created

### Proof

A proof attempt for a statement:

- `id`: Unique integer identifier
- `statement_id`: ID of the statement being proven
- `prover`: The prover used (`lean4`, `z3`, etc.)
- `method`: Proof method used (`tactics`, `smt`, `manual`, etc.)
- `status`: Result of the proof attempt (`success`, `failed`)
- `duration_ms`: Time taken for the proof attempt
- `created_at`: Timestamp when the proof was attempted

### Block

A block in the MathLedger blockchain:

- `block_number`: Sequential block number
- `merkle_root`: Merkle root hash of statements in the block
- `header`: Block metadata including run information and statement counts
- `created_at`: Timestamp when the block was created

## Text Normalization

When querying statements by text, the API normalizes Unicode characters:

- `→` becomes `->`
- `∧` becomes `and`
- `∨` becomes `or`
- `¬` becomes `not`

This allows queries to work with both ASCII and Unicode representations of logical operators.

## Rate Limiting

Currently no rate limiting is implemented. Consider implementing rate limiting for production use.

## CORS

CORS is enabled and configurable via the `CORS_ORIGINS` environment variable. Default allows all origins (`*`).

## Examples

### Get System Metrics

```bash
curl -H "X-API-Key: devkey" http://localhost:8010/metrics
```

### Get Statement by Hash

```bash
curl -H "X-API-Key: devkey" "http://localhost:8010/statements?hash=abc123def456..."
```

### Get Statement by Text

```bash
curl -H "X-API-Key: devkey" "http://localhost:8010/statements?text=(and p q)"
```

### Get Latest Block

```bash
curl -H "X-API-Key: devkey" http://localhost:8010/blocks/latest
```

## Interactive Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8010/docs
- ReDoc: http://localhost:8010/redoc
