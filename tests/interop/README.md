# Cross-Language Interoperability Test Suite

**Mission**: Guarantee seamless integration between Python (FastAPI backend), JavaScript (UI client), and PowerShell (automation scripts).

## Overview

This test suite validates that the MathLedger API maintains consistent contracts across all three language ecosystems:

- **Python (FastAPI)** → Defines the API contracts
- **JavaScript (Client SDK)** → Consumes the API from UI
- **PowerShell (Scripts)** → Automates workflows via API

### What We Test

1. **API Contracts**: Verify all endpoints return expected fields and structure
2. **Type Safety**: Ensure types don't coerce unexpectedly (int vs float, bool vs int, null vs undefined)
3. **JSON Fidelity**: Validate JSON round-trip preserves types and values
4. **Field Naming**: Check snake_case consistency and case sensitivity handling
5. **Serialization**: Test booleans, nulls, numbers, strings, timestamps, arrays, objects
6. **Error Responses**: Verify error formats are parseable across languages

## Test Files

### Python Tests

- **`test_api_contracts.py`** - FastAPI endpoint contract validation
  - Tests `/metrics`, `/heartbeat.json`, `/blocks/latest`, `/statements`, `/health`
  - Validates required fields, types, and structures
  - Uses mocked FastAPI TestClient

- **`test_type_coercion.py`** - JSON serialization and type coercion tests
  - Boolean: `True` → `true` → `$true` / `true`
  - Null: `None` → `null` → `$null` / `null`
  - Numbers: Integer vs float preservation
  - Strings: UTF-8 encoding, escaping
  - Timestamps: ISO 8601 format
  - Arrays and nested objects

### JavaScript Tests

- **`mathledger_client.test.js`** - Client SDK contract validation
  - Tests MathLedger Client SDK against mocked API
  - Validates JSON parsing, type handling, field access
  - Checks latency tracking and error handling
  - Tests nested object navigation

### PowerShell Tests

- **`Test-APIContracts.ps1`** - PowerShell script contract validation
  - Tests all endpoints used by `healthcheck.ps1` and `sanity.ps1`
  - Validates PowerShell-specific type coercion (JSON → PS types)
  - Tests boolean conversion: `true` → `$true`
  - Tests null handling: `null` → `$null`
  - Tests timestamp parsing: ISO 8601 → `[DateTime]`

## Running the Tests

### Prerequisites

```bash
# Python tests require:
pip install pytest fastapi

# JavaScript tests require:
node --version  # Node.js 14+

# PowerShell tests require:
# PowerShell 5.1+ or PowerShell Core 7+
# Running FastAPI server at http://localhost:8000
```

### Run All Tests

```bash
# From repository root
python tests/interop/run_all_tests.py
```

Or run individually:

### Python Tests Only

```bash
# API contracts
pytest tests/interop/test_api_contracts.py -v

# Type coercion
pytest tests/interop/test_type_coercion.py -v

# All Python interop tests
pytest tests/interop/ -v
```

### JavaScript Tests Only

```bash
node tests/interop/mathledger_client.test.js
```

### PowerShell Tests Only

```powershell
# Requires running API server
powershell -File tests/interop/Test-APIContracts.ps1
```

## Expected Output

When all tests pass, you should see:

```
[PASS] Interop Verified langs=3 drift≤ε
```

This confirms:
- ✅ Python FastAPI endpoints match contracts
- ✅ JavaScript client correctly parses responses
- ✅ PowerShell scripts correctly handle types
- ✅ JSON round-trip preserves types and values
- ✅ No type coercion drift detected

## Common Issues

### 1. Type Coercion Drift

**Symptom**: JavaScript receives `150.0` instead of `150`

**Cause**: Python serializes integer as float

**Fix**: Ensure FastAPI models use `int` type annotations, not `float`

### 2. Field Name Mismatch

**Symptom**: PowerShell script can't access `blockCount`

**Cause**: JavaScript uses camelCase, Python uses snake_case

**Fix**: Use snake_case consistently (Python convention for JSON APIs)

### 3. Null vs Undefined

**Symptom**: JavaScript receives `undefined` instead of `null`

**Cause**: Python omits null fields entirely

**Fix**: Include null fields explicitly: `{"merkle": None}` → `{"merkle": null}`

### 4. Boolean as Integer

**Symptom**: PowerShell receives `1` instead of `$true`

**Cause**: Database stores boolean as integer

**Fix**: Ensure FastAPI returns `bool` type, not `int`

### 5. Timestamp Format

**Symptom**: PowerShell can't parse timestamp string

**Cause**: Non-ISO 8601 format

**Fix**: Use `.isoformat()` for datetime serialization

## Integration with CI

Add to `.github/workflows/test.yml`:

```yaml
- name: Run Interop Tests
  run: |
    pytest tests/interop/ -v
    node tests/interop/mathledger_client.test.js
```

For PowerShell tests, ensure API server is running:

```yaml
- name: Start API Server
  run: uvicorn backend.orchestrator.app:app --host 0.0.0.0 --port 8000 &

- name: Run PowerShell Interop Tests
  shell: pwsh
  run: |
    Start-Sleep -Seconds 5  # Wait for API to start
    pwsh -File tests/interop/Test-APIContracts.ps1
```

## Contract Documentation

### Metrics Endpoint (`/metrics`)

**Used by**: `sanity.ps1`, `mathledger-client.js`

**Required fields**:
```json
{
  "proofs": {"success": int, "failure": int},
  "block_count": int,
  "max_depth": int,
  "statement_counts": int,
  "success_rate": float,
  "queue_length": int
}
```

### Heartbeat Endpoint (`/heartbeat.json`)

**Used by**: `healthcheck.ps1`

**Required fields**:
```json
{
  "ok": bool,
  "ts": str (ISO 8601),
  "proofs": {"success": int},
  "proofs_per_sec": float,
  "blocks": {
    "height": int,
    "latest": {"merkle": str | null}
  },
  "policy": {"hash": str | null},
  "redis": {"ml_jobs_len": int}
}
```

### Blocks Endpoint (`/blocks/latest`)

**Used by**: `mathledger-client.js`

**Required fields**:
```json
{
  "block_number": int,
  "merkle_root": str,
  "created_at": str (ISO 8601),
  "header": object
}
```

**Returns**: `404` with `{"detail": "no blocks"}` if empty

### Statements Endpoint (`/statements?hash=<hex64>`)

**Used by**: `mathledger-client.js`

**Authentication**: Requires `X-API-Key` header

**Required fields**:
```json
{
  "statement": str,
  "hash": str (64 hex chars),
  "proofs": array[object],
  "parents": array[object]
}
```

### Health Endpoint (`/health`)

**Used by**: All health checks

**Required fields**:
```json
{
  "status": str ("healthy"),
  "timestamp": str (ISO 8601)
}
```

## Maintenance

When adding new API endpoints:

1. ✅ Add contract tests to `test_api_contracts.py`
2. ✅ Add client method to `mathledger-client.js`
3. ✅ Add JavaScript test to `mathledger_client.test.js`
4. ✅ If used by PowerShell, add test to `Test-APIContracts.ps1`
5. ✅ Document contract in this README

When modifying existing endpoints:

1. ✅ Update contract tests first (TDD)
2. ✅ Ensure all three language tests pass
3. ✅ Update contract documentation
4. ✅ Check for type coercion issues

## Reference

- **healthcheck.ps1**: Uses `/heartbeat.json` endpoint (lines 142-146)
- **sanity.ps1**: Uses `/metrics` endpoint (lines 144-150)
- **mathledger-client.js**: JavaScript SDK (lines 108-160)
- **app.py**: FastAPI endpoint definitions (lines 397-603)

## Success Criteria

✅ All tests pass: `[PASS] Interop Verified langs=3 drift≤ε`

✅ Type safety: No unexpected coercions (int→float, bool→int)

✅ Field consistency: snake_case throughout

✅ Null handling: Explicit `null` (not missing fields)

✅ Timestamp format: ISO 8601 strings

✅ Boolean format: JSON `true`/`false` (not 1/0)

✅ Error responses: Parseable `{"detail": "..."}` format
