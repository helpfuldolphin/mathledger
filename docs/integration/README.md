# MathLedger Integration Layer

Cross-language integration bridge for Python, Node, and FastAPI systems with <200ms latency guarantee.

## Overview

The MathLedger integration layer provides elegant bridges between heterogeneous components:

- **Python Backend** (axiom_engine, orchestrator, worker, fol_euf)
- **FastAPI** (REST API layer)
- **Node.js/Svelte** (UI frontend)
- **PostgreSQL** (persistent storage)
- **Redis** (job queue)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     UI Layer (Node/Svelte)                  │
│                  MathLedgerClient SDK                       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/JSON
                         │ <120ms p99
┌────────────────────────▼────────────────────────────────────┐
│              FastAPI Orchestrator (Python)                  │
│         IntegrationMonitoringMiddleware                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ <60ms p99
┌────────────────────────▼────────────────────────────────────┐
│            IntegrationBridge (Python)                       │
│              LatencyTracker                                 │
└─────┬──────────────┬──────────────┬────────────────────────┘
      │              │              │
      │ <60ms        │ <10ms        │ <200ms
      ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│PostgreSQL│   │  Redis   │   │Backend Modules│
│          │   │          │   │ axiom_engine │
│          │   │          │   │ logic/canon  │
│          │   │          │   │ fol_euf      │
└──────────┘   └──────────┘   └──────────────┘
```

## Components

### 1. IntegrationBridge (Python)

Unified interface for backend components with latency tracking.

**Location**: `backend/integration/bridge.py`

**Features**:
- Connection pooling for PostgreSQL and Redis
- Automatic latency tracking for all operations
- Schema-tolerant database queries
- Graceful error handling

**Usage**:
```python
from backend.integration.bridge import IntegrationBridge

bridge = IntegrationBridge(
    db_url="postgresql://...",
    redis_url="redis://...",
    metrics_enabled=True
)

# Execute derivation
results = bridge.execute_derivation(system="pl", steps=10)

# Query statements
statements = bridge.query_statements(system="pl", limit=100)

# Get metrics
metrics = bridge.get_metrics_summary()

# Get latency stats
stats = bridge.get_latency_stats()

bridge.close()
```

### 2. LatencyTracker (Python)

Performance monitoring and metrics collection.

**Location**: `backend/integration/metrics.py`

**Features**:
- Context manager for operation tracking
- Percentile calculations (p50, p95, p99)
- Success rate tracking
- Comprehensive statistics

**Usage**:
```python
from backend.integration.metrics import LatencyTracker

tracker = LatencyTracker()

with tracker.track("my_operation", {"key": "value"}):
    # Your code here
    pass

stats = tracker.get_stats("my_operation")
print(f"Mean: {stats['mean_ms']:.2f}ms")
print(f"P95: {stats['p95_ms']:.2f}ms")
print(f"Success Rate: {stats['success_rate']:.1f}%")
```

### 3. IntegrationMonitoringMiddleware (Python)

FastAPI middleware for request/response tracking.

**Location**: `backend/integration/middleware.py`

**Features**:
- Automatic request latency tracking
- Performance headers (X-Response-Time)
- Endpoint-specific metrics
- Success/failure rate monitoring

**Usage**:
```python
from fastapi import FastAPI
from backend.integration.middleware import setup_integration_middleware
from backend.integration.metrics import LatencyTracker

app = FastAPI()
tracker = LatencyTracker()

monitoring = setup_integration_middleware(app, tracker)

# Access stats later
stats = monitoring.get_stats()
```

### 4. MathLedgerClient (Node.js)

JavaScript SDK for UI-to-FastAPI communication.

**Location**: `ui/src/lib/mathledger-client.js`

**Features**:
- Promise-based API
- Automatic latency tracking
- Timeout handling
- Request retry logic

**Usage**:
```javascript
import MathLedgerClient from './lib/mathledger-client.js';

const client = new MathLedgerClient('http://localhost:8000', {
  apiKey: 'devkey',
  timeout: 5000,
  trackLatency: true
});

// Get metrics
const metrics = await client.getMetrics();

// Query statements
const statements = await client.getStatements({ limit: 10 });

// Get latency stats
const stats = client.getLatencyStats();
console.log(`P95 latency: ${stats.p95_ms}ms`);

// Check if target met
if (client.isLatencyTargetMet()) {
  console.log('Latency target <200ms met!');
}
```

## Performance Targets

All integration points must meet these latency targets:

| Component | Target (p95) | Target (p99) |
|-----------|--------------|--------------|
| Redis operations | <10ms | <15ms |
| Database queries | <40ms | <60ms |
| API endpoints | <100ms | <120ms |
| End-to-end requests | <150ms | <180ms |
| Derivation operations | <180ms | <200ms |

## Benchmarking

Run integration benchmarks:

```bash
cd /home/ubuntu/repos/mathledger
uv run python backend/integration/generate_report.py
```

This generates `artifacts/integration/report.json` with:
- Component-level latency metrics
- Benchmark results
- Validation status
- Recommendations

## Testing

Run integration tests:

```bash
# Test integration bridge
uv run pytest tests/integration/test_integration_bridge.py -v

# Test metrics tracking
uv run pytest tests/integration/test_integration_metrics.py -v

# Run all integration tests
uv run pytest tests/integration/ -v
```

## Monitoring

### Python Backend

```python
from backend.integration.metrics import IntegrationMetrics

metrics = IntegrationMetrics()

# Track operations
tracker = metrics.get_tracker("my_component")
with tracker.track("operation"):
    # Your code
    pass

# Generate report
report = metrics.generate_report()
metrics.save_report("artifacts/integration/report.json")
```

### FastAPI Middleware

Middleware automatically adds performance headers:

```
X-Response-Time: 15.23ms
X-Integration-Version: 1.0.0
```

### Node.js Client

```javascript
const stats = client.getLatencyStats();
console.log(`Operations: ${stats.count}`);
console.log(`Mean: ${stats.mean_ms}ms`);
console.log(`P95: ${stats.p95_ms}ms`);
console.log(`Success Rate: ${stats.success_rate}%`);
```

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with bridge.track_operation("my_op"):
    result = do_something()

# Bad
start = time.time()
result = do_something()
duration = time.time() - start
```

### 2. Close Connections

```python
bridge = IntegrationBridge()
try:
    results = bridge.query_statements()
finally:
    bridge.close()
```

### 3. Monitor Latency

```python
stats = bridge.get_latency_stats()
for op, metrics in stats.items():
    if metrics['p95_ms'] > 200:
        print(f"WARNING: {op} exceeds latency target")
```

### 4. Handle Errors Gracefully

```python
try:
    results = bridge.execute_derivation()
except Exception as e:
    logger.error(f"Derivation failed: {e}")
    # Fallback logic
```

## Troubleshooting

### High Latency

1. Check database connection pool settings
2. Review query complexity
3. Monitor Redis connection count
4. Check network latency

### Connection Errors

1. Verify DATABASE_URL and REDIS_URL
2. Check firewall rules
3. Verify service availability
4. Review connection timeout settings

### Memory Issues

1. Clear latency measurements periodically
2. Use connection pooling
3. Close connections after use
4. Monitor measurement buffer size

## Integration Checklist

- [ ] IntegrationBridge configured with correct URLs
- [ ] LatencyTracker enabled for all operations
- [ ] FastAPI middleware installed
- [ ] Node.js client configured with API key
- [ ] Latency targets validated (<200ms)
- [ ] Integration tests passing
- [ ] Monitoring dashboard configured
- [ ] Error handling implemented
- [ ] Connection pooling optimized
- [ ] Performance report generated

## Future Enhancements

1. **WebSocket Support**: Real-time updates for UI
2. **GraphQL Integration**: Alternative to REST API
3. **Distributed Tracing**: OpenTelemetry integration
4. **Circuit Breakers**: Fault tolerance patterns
5. **Rate Limiting**: Request throttling
6. **Caching Layer**: Redis-based response caching
7. **Load Balancing**: Multi-instance support
8. **Health Checks**: Automated service monitoring

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [psycopg3 Documentation](https://www.psycopg.org/psycopg3/)
- [Redis Python Client](https://redis-py.readthedocs.io/)
- [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)

## Support

For issues or questions:
1. Check integration test results
2. Review latency report
3. Examine error logs
4. Verify configuration
5. Contact integration team
