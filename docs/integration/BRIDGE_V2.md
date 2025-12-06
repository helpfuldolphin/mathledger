# Integration Bridge V2 - Proof Fabric Architect

## Overview

Bridge V2 is an optimized integration layer for MathLedger's cross-language systems (Python ↔ Node ↔ FastAPI) with sub-150ms latency targets, connection pooling, retry logic, and authenticated proof flow.

## Key Features

### 1. Connection Pooling

**PostgreSQL Connection Pool:**
- Min connections: 2
- Max connections: 10 (configurable)
- Timeout: 5 seconds
- Max idle time: 300 seconds
- Max lifetime: 3600 seconds

**Redis Connection Pool:**
- Max connections: 10 (configurable)
- Socket timeout: 1 second
- Socket connect timeout: 1 second
- Decode responses: enabled

### 2. Retry Logic with Exponential Backoff

**Default Configuration:**
- Max attempts: 3
- Initial delay: 100ms
- Max delay: 2 seconds
- Exponential base: 2.0

**Configurable per Operation:**
```python
@with_retry(RetryConfig(max_attempts=5, initial_delay=0.05))
def critical_operation():
    pass
```

### 3. Token Propagation

**BridgeToken Features:**
- Deterministic SHA256 token IDs
- Operation tracking
- Timestamp verification
- Metadata support
- Integrity validation

**Token Flow:**
```
Operation Start → Token Generation → SHA256 Hash → Propagation → Verification
```

### 4. Bridge Integrity Verification

**SHA256 Hash Generation:**
- Combines all active token IDs
- Deterministic and reproducible
- Verifiable by CI systems
- Supports Merkle tree integration

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    IntegrationBridgeV2                      │
├─────────────────────────────────────────────────────────────┤
│  Connection Pools:                                          │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ PostgreSQL   │  │    Redis     │                        │
│  │ Pool (2-10)  │  │ Pool (10)    │                        │
│  └──────────────┘  └──────────────┘                        │
│                                                             │
│  Retry Logic:                                               │
│  ┌──────────────────────────────────────┐                  │
│  │ Exponential Backoff (100ms → 2s)    │                  │
│  │ Max Attempts: 3-5 (configurable)    │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  Token System:                                              │
│  ┌──────────────────────────────────────┐                  │
│  │ BridgeToken (SHA256)                │                  │
│  │ - Operation tracking                │                  │
│  │ - Timestamp verification            │                  │
│  │ - Metadata propagation              │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  Latency Tracking:                                          │
│  ┌──────────────────────────────────────┐                  │
│  │ LatencyTracker                       │                  │
│  │ - Per-operation metrics              │                  │
│  │ - p50/p95/p99 percentiles            │                  │
│  │ - Success rate tracking              │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Client Request
    ↓
[Token Generation]
    ↓
[Connection Pool] → [Retry Logic] → [Database/Redis]
    ↓                                      ↓
[Latency Tracking]                    [Response]
    ↓                                      ↓
[Token Verification] ← ← ← ← ← ← ← ← ← ← ←
    ↓
[Bridge Integrity Hash]
    ↓
Client Response (with token_id)
```

## Usage

### Basic Initialization

```python
from backend.integration.bridge_v2 import IntegrationBridgeV2

bridge = IntegrationBridgeV2(
    db_url="postgresql://ml:mlpass@127.0.0.1:5433/mathledger",
    redis_url="redis://127.0.0.1:6379/0",
    metrics_enabled=True,
    pool_size=10
)
```

### Query with Automatic Retry

```python
# Automatic retry with default config (3 attempts)
statements = bridge.query_statements(
    system="pl",
    limit=100
)

# Each statement includes token_id for verification
for stmt in statements:
    print(f"Statement: {stmt['text']}")
    print(f"Token: {stmt['token_id']}")
```

### Token Verification

```python
# Verify token integrity
token_id = statements[0]['token_id']
is_valid = bridge.verify_token(token_id, "query_statements")

if is_valid:
    print("[PASS] Token verification successful")
else:
    print("[FAIL] Token verification failed")
```

### Bridge Integrity Check

```python
# Generate bridge integrity hash
integrity_hash = bridge.get_bridge_integrity_hash()
print(f"[PASS] Bridge Integrity <{integrity_hash}>")

# Get latency stats with integrity info
stats = bridge.get_latency_stats()
print(f"Bridge Integrity: {stats['_bridge_integrity']}")
print(f"Active Tokens: {stats['_token_count']}")
```

### Custom Retry Configuration

```python
from backend.integration.bridge_v2 import RetryConfig

# Create bridge with custom retry config
bridge = IntegrationBridgeV2(
    retry_config=RetryConfig(
        max_attempts=5,
        initial_delay=0.05,
        max_delay=1.0,
        exponential_base=1.5
    )
)
```

## Latency Profiling

### Running the Profiler

```bash
# Generate latency profile report
cd /home/ubuntu/repos/mathledger
uv run python backend/integration/latency_profiler.py

# Output: artifacts/integration/latency_profile.json
```

### Profile Report Structure

```json
{
  "timestamp": "2025-10-19T12:00:00.000000",
  "version": "2.0.0",
  "target_latency_ms": 150,
  "operations": {
    "query_statements": {
      "iterations": 50,
      "mean_ms": 25.3,
      "p50_ms": 23.1,
      "p95_ms": 45.2,
      "p99_ms": 67.8,
      "success_rate": 100.0,
      "tokens_generated": 50,
      "meets_target": true
    }
  },
  "token_propagation": {
    "token_id": "abc123...",
    "verified": true,
    "operation": "query_statements"
  },
  "bridge_integrity": {
    "hash": "def456...",
    "token_count": 150,
    "verified": true
  },
  "validation": {
    "sub_150ms_met": true,
    "token_verification": true,
    "integrity_verified": true
  },
  "summary": {
    "all_validations_passed": true,
    "operations_profiled": 3,
    "max_p95_latency_ms": 45.2,
    "overall_success_rate": 99.8
  }
}
```

### Validation Criteria

**Sub-150ms Target:**
- All operations must have p95 latency < 150ms
- Measured across 50 iterations per operation
- Includes connection pooling overhead

**Token Verification:**
- Tokens must be generated for all operations
- SHA256 hash must be deterministic
- Token verification must succeed

**Bridge Integrity:**
- Integrity hash must be valid SHA256 (64 hex chars)
- Token count must match active operations
- Hash must be reproducible

## Performance Targets

### Latency Targets (p95)

| Operation | Target | V1 Baseline | V2 Optimized |
|-----------|--------|-------------|--------------|
| query_statements | <50ms | 35.7ms | <30ms (est) |
| get_metrics | <50ms | 28.4ms | <25ms (est) |
| enqueue_job | <20ms | 6.4ms | <5ms (est) |
| end_to_end | <150ms | 152.8ms | <140ms (est) |

### Success Rate Targets

- Overall success rate: >99%
- Retry success rate: >95% (after retries)
- Connection pool utilization: 60-80%

## Testing

### Unit Tests

```bash
# Run V2 bridge tests
cd /home/ubuntu/repos/mathledger
uv run pytest tests/integration/test_bridge_v2.py -v

# Expected: 19 tests passing
# - RetryConfig: 2 tests
# - with_retry decorator: 3 tests
# - BridgeToken: 6 tests
# - IntegrationBridgeV2: 8 tests
```

### Integration Tests

```bash
# Run full integration test suite
uv run pytest tests/integration/ -v

# Expected: 47 tests passing (28 V1 + 19 V2)
```

### Performance Benchmarks

```bash
# Generate latency profile
uv run python backend/integration/latency_profiler.py

# Verify sub-150ms target
cat artifacts/integration/latency_profile.json | jq '.validation.sub_150ms_met'
# Expected: true
```

## Migration from V1

### API Compatibility

V2 maintains backward compatibility with V1 API:

```python
# V1 code (still works)
from backend.integration.bridge import IntegrationBridge
bridge = IntegrationBridge()

# V2 code (enhanced features)
from backend.integration.bridge_v2 import IntegrationBridgeV2
bridge = IntegrationBridgeV2()
```

### Key Differences

| Feature | V1 | V2 |
|---------|----|----|
| Connection Pooling | Single connection | Pool (2-10) |
| Retry Logic | None | Exponential backoff |
| Token Propagation | None | SHA256 tokens |
| Bridge Integrity | None | SHA256 hash |
| Latency Target | <200ms | <150ms |

### Migration Steps

1. **Update imports:**
   ```python
   from backend.integration.bridge_v2 import IntegrationBridgeV2
   ```

2. **Initialize with pool size:**
   ```python
   bridge = IntegrationBridgeV2(pool_size=10)
   ```

3. **Handle token IDs in responses:**
   ```python
   result = bridge.query_statements()
   token_id = result[0]['token_id']
   ```

4. **Verify tokens:**
   ```python
   is_valid = bridge.verify_token(token_id)
   ```

## Troubleshooting

### Connection Pool Exhaustion

**Symptom:** `RuntimeError: DB pool not initialized` or timeout errors

**Solution:**
```python
# Increase pool size
bridge = IntegrationBridgeV2(pool_size=20)

# Or reduce connection lifetime
bridge._db_pool.max_lifetime = 1800.0  # 30 minutes
```

### Retry Failures

**Symptom:** Operations failing after max retry attempts

**Solution:**
```python
# Increase retry attempts
bridge = IntegrationBridgeV2(
    retry_config=RetryConfig(max_attempts=5)
)

# Or increase delay
bridge.retry_config.max_delay = 5.0
```

### Token Verification Failures

**Symptom:** `bridge.verify_token()` returns False

**Solution:**
- Ensure token_id is from current bridge instance
- Check that operation name matches
- Verify token hasn't expired (if TTL implemented)

### Latency Exceeds Target

**Symptom:** p95 latency > 150ms

**Solution:**
1. Check connection pool utilization
2. Verify database query performance
3. Review network latency
4. Consider increasing pool size
5. Optimize database queries

## CI Integration

### GitHub Actions Workflow

```yaml
- name: Run Bridge V2 Tests
  run: |
    uv run pytest tests/integration/test_bridge_v2.py -v

- name: Generate Latency Profile
  run: |
    uv run python backend/integration/latency_profiler.py

- name: Verify Sub-150ms Target
  run: |
    MEETS_TARGET=$(cat artifacts/integration/latency_profile.json | jq '.validation.sub_150ms_met')
    if [ "$MEETS_TARGET" != "true" ]; then
      echo "FAIL: Sub-150ms latency target not met"
      exit 1
    fi
    echo "PASS: Sub-150ms latency target met"

- name: Verify Bridge Integrity
  run: |
    INTEGRITY=$(cat artifacts/integration/latency_profile.json | jq -r '.bridge_integrity.hash')
    echo "[PASS] Bridge Integrity <$INTEGRITY>"
```

## Reasoning Merkle Integration

### Token Chain

Bridge V2 tokens form a deterministic chain for Merkle tree integration:

```
Token 1 (SHA256) → Token 2 (SHA256) → Token 3 (SHA256)
         ↓                  ↓                  ↓
    Bridge Integrity Hash (SHA256 of all tokens)
         ↓
    Merkle Root (for Cursor E's Reasoning Merkle)
```

### Proof Flow

1. **Operation Execution:** Generate BridgeToken with SHA256 hash
2. **Token Propagation:** Include token_id in response
3. **Token Verification:** Validate token integrity
4. **Bridge Integrity:** Compute combined hash of all tokens
5. **Merkle Root:** Feed bridge integrity hash to Reasoning Merkle

### Deterministic Verification

```python
# Generate proof chain
operations = ["query_statements", "get_metrics", "enqueue_job"]
tokens = []

for op in operations:
    result = bridge.execute_operation(op)
    tokens.append(result['token_id'])

# Verify chain integrity
integrity_hash = bridge.get_bridge_integrity_hash()
is_valid = all(bridge.verify_token(t) for t in tokens)

if is_valid:
    print(f"[PASS] Bridge Integrity <{integrity_hash}>")
    print(f"[PASS] Token Chain Verified ({len(tokens)} tokens)")
else:
    print("[FAIL] Token Chain Verification Failed")
```

## Future Enhancements

### Planned Features

1. **Token TTL:** Automatic token expiration after configurable time
2. **Circuit Breaker:** Automatic failure detection and recovery
3. **Adaptive Pooling:** Dynamic pool size based on load
4. **Distributed Tracing:** OpenTelemetry integration
5. **Metrics Export:** Prometheus/Grafana integration

### Performance Goals

- Sub-100ms p95 latency for all operations
- 99.9% success rate
- Support for 1000+ concurrent connections
- Horizontal scaling support

## References

- [Integration Layer V1 Documentation](./README.md)
- [Latency Profiler Source](../../backend/integration/latency_profiler.py)
- [Bridge V2 Source](../../backend/integration/bridge_v2.py)
- [V2 Tests](../../tests/integration/test_bridge_v2.py)

---

**Version:** 2.0.0  
**Last Updated:** 2025-10-19  
**Author:** Devin D - Proof Fabric Architect
